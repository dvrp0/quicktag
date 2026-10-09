mod cache;
mod runner;
pub use runner::RunnerShellAssembly;

/// Schema names participate in technique classification during model decoding.
pub(crate) fn invalidate_cached_models() {
    cache::invalidate();
}

use anyhow::{Context, bail};
use binrw::Endian;
use itertools::Itertools;
use quicktag_core::classes::get_class_by_id;
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::TagCache;
use std::cmp::Reverse;
use std::sync::Arc;
use tiger_pkg::{TagHash, TagHash64, Version, package::UEntryHeader, package_manager};
use wgpu::util::DeviceExt;

use crate::material::{
    MaterialPreviewKind, MaterialTagPreview, TechniqueMaterialConstants, TechniqueTextureBinding,
    interpret_tfx_stack_with_object_channels, is_technique_entry, material_constants_for_technique,
    primary_sampler_for_technique, render_state_for_technique, sampler_for_technique_slot,
    texture_bindings_for_technique, tfx_has_marathon_decal_abi,
};
use crate::texture::Texture;

const MAX_PREVIEW_VERTICES: usize = 200_000;
const MAX_PREVIEW_INDICES: usize = 900_000;
const TECHNIQUE_SCAN_CHILD_LIMIT: usize = 128;
const CLASS_GEOMETRY_RESOURCE: u32 = 0x8080881C;
const CLASS_GEOMETRY_BUFFER_SET: u32 = 0x808087CB;
const CLASS_GEOMETRY_PART: u32 = 0x808087D1;
const CLASS_VERTEX_INPUT_LAYOUT_MAPPING: u32 = 0x80808664;
const CLASS_VERTEX_INPUT_ELEMENT_SETS: u32 = 0x80808668;
const CLASS_VERTEX_LAYOUT_ARRAY: u32 = 0x80808667;
const CLASS_VERTEX_INPUT_ELEMENT_ARRAY: u32 = 0x8080866D;
const CLASS_ENTITY_RESOURCE: u32 = 0x80809B06;
const CLASS_PATTERN: u32 = 0x8080BAAD;
const CLASS_PATTERN_COMPONENT: u32 = 0x8080BADB;
const CLASS_DECORATOR: u32 = 0x80806C98;
const SEMANTIC_POSITION: u8 = 0x00;
const SEMANTIC_NORMAL: u8 = 0x03;
const SEMANTIC_TEXCOORD: u8 = 0x05;
const SEMANTIC_TANGENT: u8 = 0x06;

#[derive(Debug, Clone)]
pub struct GeometryTagPreview {
    pub kind: GeometryPreviewKind,
}

#[derive(Debug, Clone)]
pub enum GeometryPreviewKind {
    VertexBuffer(VertexBufferPreview),
    IndexBuffer(IndexBufferPreview),
    Model(ModelPreview),
}

#[derive(Debug, Clone)]
pub struct VertexBufferPreview {
    pub header: VertexBufferHeader,
    pub data_tag: TagHash,
    pub data_len: usize,
    pub element_count: u32,
    pub candidates: Vec<VertexPositionCandidate>,
    pub uv_candidates: Vec<VertexUvCandidate>,
    pub wireframe: Option<WireframePreview>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IndexBufferPreview {
    pub header: IndexBufferHeader,
    pub data_tag: TagHash,
    pub data_len: usize,
    pub index_count: usize,
    pub min_index: Option<u32>,
    pub max_index: Option<u32>,
    pub first_indices: Vec<u32>,
    pub indices: Vec<u32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ModelPreview {
    pub label: &'static str,
    pub class_name: Option<String>,
    pub mesh_source: Option<MeshSourcePreview>,
    pub vertex_buffers: Vec<(TagHash, UEntryHeader)>,
    pub index_buffers: Vec<(TagHash, UEntryHeader)>,
    pub techniques: Vec<(TagHash, UEntryHeader)>,
    pub textures: Vec<(TagHash, UEntryHeader)>,
    pub shaders: Vec<(TagHash, UEntryHeader)>,
    pub geometry_parts: Vec<TagHash>,
    pub wireframe: Option<WireframePreview>,
}

impl ModelPreview {
    pub fn preview_uv_transform(&self) -> Option<UvTransformPreview> {
        (self.geometry_parts.len() <= 1)
            .then(|| self.mesh_source.as_ref().and_then(|mesh| mesh.uv_transform))
            .flatten()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTagRole {
    Mesh,
    MeshData,
    Geometry,
    Dynamic,
    Container,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelTagInfo {
    pub role: ModelTagRole,
    pub label: &'static str,
}

impl ModelTagInfo {
    /// Whether this tag represents a complete, user-facing model rather than one
    /// of the implementation tags consumed while assembling that model.
    pub fn is_catalog_entry(self) -> bool {
        (matches!(self.role, ModelTagRole::Dynamic) && self.label != "Dynamic mesh")
            || matches!(self.role, ModelTagRole::Container)
    }
}

#[derive(Debug, Clone)]
pub struct VertexBufferHeader {
    pub data_size: u32,
    pub stride: u16,
    pub vtype: u16,
    pub deadbeef: u32,
}

#[derive(Debug, Clone)]
pub struct IndexBufferHeader {
    pub unk0: i8,
    pub is_32bit: bool,
    pub unk1: u16,
    pub zero: u32,
    pub data_size: u64,
    pub deadbeef: u32,
    pub zero1: u32,
}

#[derive(Debug, Clone)]
pub struct VertexPositionCandidate {
    pub format: PositionFormat,
    pub label: &'static str,
    pub offset: usize,
    pub valid_vertices: usize,
    pub sampled_vertices: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct VertexUvCandidate {
    pub format: UvFormat,
    pub label: String,
    pub offset: usize,
    pub valid_vertices: usize,
    pub sampled_vertices: usize,
    pub min: [f32; 2],
    pub max: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionFormat {
    F32x3,
    I16x4,
    I16x3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvFormat {
    F32x2,
    F16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputLayoutFormat {
    R32G32Float,
    R32G32B32Float,
    R32G32B32A32Float,
    R16G16Snorm,
    R16G16B16A16Snorm,
}

#[derive(Debug, Clone, Copy)]
struct InputLayoutTexcoord {
    buffer_index: usize,
    offset: usize,
    format: InputLayoutFormat,
}

#[derive(Debug, Clone, Copy)]
struct InputLayoutVector {
    buffer_index: usize,
    offset: usize,
    format: InputLayoutFormat,
}

#[derive(Clone, Copy)]
struct TagArray {
    class: u32,
    count: usize,
    data_offset: usize,
    end_offset: usize,
}

#[derive(Debug, Clone, Copy)]
struct VertexInputElement {
    semantic: u8,
    semantic_index: u8,
    format: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredVertexElementDescriptor {
    pub semantic: u8,
    pub semantic_index: u8,
    pub format: u8,
    pub offset: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredVertexStreamLayoutDescriptor {
    pub stream_index: u8,
    pub element_set_index: u32,
    pub instanced: bool,
    pub elements: Vec<AuthoredVertexElementDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredInputLayoutDescriptor {
    pub layout_id: u8,
    pub streams: Vec<AuthoredVertexStreamLayoutDescriptor>,
}

#[derive(Debug, Clone)]
pub struct AuthoredVertexStreamRef {
    pub stream_index: u8,
    pub header_tag: TagHash,
    pub data_tag: TagHash,
    pub stride: u16,
    pub vertex_type: u16,
    pub data_size: u32,
    pub element_count: u32,
}

#[derive(Debug, Clone)]
pub struct AuthoredIndexBufferRef {
    pub header_tag: TagHash,
    pub data_tag: TagHash,
    pub is_32bit: bool,
    pub data_size: u64,
    pub index_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryPositionTransform {
    pub scale: [f32; 3],
    pub offset: [f32; 3],
    pub procedural_scale: f32,
}

#[derive(Debug, Clone)]
pub struct AuthoredStageInputLayout {
    pub raw_stage: u8,
    pub layout_id: u8,
    pub descriptor: Option<AuthoredInputLayoutDescriptor>,
}

#[derive(Debug, Clone)]
pub struct AuthoredGeometryInput {
    pub geometry: TagHash,
    pub vertex_streams: Vec<AuthoredVertexStreamRef>,
    pub color_buffer: Option<AuthoredVertexStreamRef>,
    pub skinning_buffer: Option<AuthoredVertexStreamRef>,
    pub index_buffer: Option<AuthoredIndexBufferRef>,
    pub stage_layouts: Vec<AuthoredStageInputLayout>,
    pub position_transform: Option<GeometryPositionTransform>,
    pub uv_transform: Option<UvTransformPreview>,
    pub attachment_pose: Option<WeaponModAttachmentPose>,
}

impl std::fmt::Display for InputLayoutFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            InputLayoutFormat::R32G32Float => "R32G32_FLOAT TEXCOORD",
            InputLayoutFormat::R32G32B32Float => "R32G32B32_FLOAT TEXCOORD.xy",
            InputLayoutFormat::R32G32B32A32Float => "R32G32B32A32_FLOAT TEXCOORD.xy",
            InputLayoutFormat::R16G16Snorm => "R16G16_SNORM TEXCOORD",
            InputLayoutFormat::R16G16B16A16Snorm => "R16G16B16A16_SNORM",
        };
        f.write_str(label)
    }
}

impl UvFormat {
    fn label(self) -> &'static str {
        match self {
            UvFormat::F32x2 => "f32x2 uv",
            UvFormat::F16x2 => "f16x2 uv",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WireframePreview {
    pub source: String,
    pub position_format: &'static str,
    pub uv_format: Option<String>,
    pub vertices: Vec<[f32; 3]>,
    /// Rigid skeleton node selected by POSITION.w. Present only for layouts
    /// whose fourth position component is an authored bone index.
    pub rigid_indices: Option<Vec<u16>>,
    pub normals: Option<Vec<[f32; 3]>>,
    /// Raw shader-input POSITION consumed by common-surface procedural passes.
    /// For `R16G16B16A16_SNORM` geometry this is the hardware-decoded
    /// `[-1, 1]` value, before geometry scale/offset and attachment transforms.
    pub procedural_positions: Option<Vec<[f32; 3]>>,
    /// Model-local NORMAL consumed by GearDye TEXCOORD6.
    pub procedural_normals: Option<Vec<[f32; 3]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub normal_format: Option<String>,
    pub tangent_format: Option<String>,
    pub indices: Vec<u32>,
    pub material_ranges: Vec<WireframeMaterialRange>,
    /// Package-native vertex/input ABI retained for Strict Tiger rendering.
    /// Unsupported authored paths may fall back to reconstructed arrays.
    pub authored_inputs: Vec<AuthoredGeometryInput>,
    /// Exact highest-detail parts authored for Marathon ShadowGenerate.
    ///
    /// These are deliberately kept separate from the visible preview ranges:
    /// visible geometry deduplicates repeated stage records, while the strict
    /// shadow renderer must retain the package-authored stage-4 draw contract.
    pub authored_shadow_ranges: Vec<WireframeAuthoredStageRange>,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub vertex_count_total: usize,
    pub index_count_total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireframeAuthoredDrawMetadata {
    pub part_index: usize,
    pub source_index_start: u32,
    pub source_index_count: u32,
    pub primitive_type: u8,
    pub variant_shader_index: u16,
    pub flags: u32,
    pub lod_run: u8,
}

#[derive(Debug, Clone)]
pub struct WireframeMaterialRange {
    pub index_start: usize,
    pub index_count: usize,
    /// Raw game/build-specific LOD membership byte. Never normalized here.
    pub raw_lod_category: Option<u8>,
    pub render_stage: Option<u8>,
    pub technique: Option<TagHash>,
    pub gear_dye_change_color_index: Option<u8>,
    /// Index into `WireframePreview::authored_inputs` for this draw.
    pub authored_source: Option<usize>,
    /// Original package IA range before preview triangle-list reconstruction.
    /// Geometry resources populate this so strict stage paths can bind the
    /// authored index buffer/topology instead of the flattened preview copy.
    pub authored_draw: Option<WireframeAuthoredDrawMetadata>,
    /// Rigid-model `position_offset.w` / skinning `offset_scale.w` consumed
    /// by common-surface procedural branches through `scope_skinning[5].w`.
    pub procedural_scale: f32,
    pub texture: Option<TagHash>,
    pub textures: WireframeMaterialTextures,
}

#[derive(Debug, Clone)]
pub struct WireframeAuthoredStageRange {
    pub render_stage: u8,
    pub input_layout_id: u8,
    pub part_index: usize,
    pub source_index_start: u32,
    pub source_index_count: u32,
    pub primitive_type: u8,
    pub raw_lod_category: u8,
    pub variant_shader_index: u16,
    pub flags: u32,
    pub lod_run: u8,
    pub technique: Option<TagHash>,
    pub gear_dye_change_color_index: Option<u8>,
    /// Index into `WireframePreview::authored_inputs` for this authored draw.
    pub authored_source: Option<usize>,
    pub procedural_scale: f32,
    /// Triangle-list indices into the parent WireframePreview vertex array.
    pub indices: Vec<u32>,
    pub texture: Option<TagHash>,
    pub textures: WireframeMaterialTextures,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearDyeMaterial {
    pub color: [f32; 4],
    pub roughness_remap: [f32; 4],
    pub metal_remap: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterSurfaceMaterial {
    /// 1 = common detail-gate, 2 = palette-mask, 3 = physical-only common surface.
    pub mode: u8,
    pub surface: TagHash,
    pub selector: TagHash,
    pub detail_color: TagHash,
    pub detail_normal: TagHash,
    /// Optional object-space procedural field used by palette-mask runner
    /// shaders (Arata t2). Kept separate from the UV selector and normal map.
    pub procedural: Option<TagHash>,
    pub detail_transform: [f32; 4],
    pub detail_base: [f32; 4],
    pub detail_scale: [f32; 4],
    pub detail_gate: f32,
    pub extra: [[f32; 4]; 2],
    pub palette: [[f32; 4]; 2],
    pub procedural_constants: [[f32; 4]; 11],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunnerLayeredSurfaceMaterial {
    /// 1 = full9 dual gated layers, 2 = t1.r switched dual layer,
    /// 3 = full10 four-layer runner surface, 4 = procedural full9 dual layer,
    /// 5 = t1.r-gated single detail layer, 6 = local full9 four-detail stack,
    /// 8 = procedural full8 with a t1.r-gated t3 detail over base t4,
    /// 9 = local full10 three-detail stack, 10 = full11 four-detail stack,
    /// 11 = full10 A/G/B/R-gated three-detail stack,
    /// 12 = local full11 A/G/B/R-gated three-detail stack,
    /// 13 = A/B/R selected three-detail stack, 14 = G/B/R selected stack,
    /// 15 = procedural A/B/G/R-gated dual-detail stack,
    /// 16 = A/B-gated dual-detail stack with material response,
    /// 17 = A/G/B/R-selected four-detail stack,
    /// 18 = G/B/R-selected dual-detail stack with material response,
    /// 19 = procedural pattern/mask surface with an authored detail normal,
    /// 20 = G/B/R-selected stack with two authored R variants,
    /// 21 = A/G/B/R-selected stack with two authored R variants,
    /// 22 = A/G-gated stack with a B-selected authored detail and response,
    /// 23 = local full9 stack with response, 24 = expanded local full9 with response,
    /// 25 = local three-detail stack with response,
    /// 26 = A/G-gated B/R-selected dual-detail stack,
    /// 27 = G/B/R-selected stack with two R variants plus response/AO,
    /// 28 = procedural A/G/B/R stack with response/AO,
    /// 29 = A/G-gated dual B/R switch stack,
    /// 30 = A/G/B/R-gated procedural stack with response/AO,
    /// 31 = character full11 A/B/R-selected three-detail stack,
    /// 32 = character full13 A/G/B/R-selected four-detail stack,
    /// 33 = character full11 A/B/R stack sharing the B/R detail texture,
    /// 34 = procedural character full11 A/G/B/R stack,
    /// 35 = compact character A/R-selected dual-detail stack,
    /// 36 = local panel R-selected single-detail stack with response,
    /// 37 = package-394 full13 A/G/B/R four-detail stack,
    /// 41 = runner skin/subsurface surface with authored pore mask and AO,
    /// 42/43 = generated procedural full12/compact sibling stacks.
    pub mode: u8,
    /// Packed surface/selector response sampled at the mesh UV.
    pub surface: TagHash,
    /// Optional two-channel field sampled at the mesh UV. Red controls the
    /// layered normal response and green contributes authored roughness.
    pub material_response: Option<TagHash>,
    /// Authored detail normals selected by channels from the packed surface.
    pub detail_normal_a: TagHash,
    pub detail_normal_b: TagHash,
    pub detail_normal_c: Option<TagHash>,
    pub detail_normal_d: Option<TagHash>,
    /// Optional object-space procedural field used by generated runner
    /// material branches. It shares no draw ABI with weapon gear patterns.
    pub procedural: Option<TagHash>,
    /// Optional authored sRGB colour mask. This is independent from the
    /// packed selector, material-response, and object-space procedural maps.
    pub color_overlay: Option<TagHash>,
    /// Shader-family constants for the optional colour layer. Kept separate
    /// from normal-stack constants because wide runner shaders use all 24
    /// normal rows already.
    pub color_overlay_constants: [[f32; 4]; 7],
    /// Optional runner condition stack: UV scratch mask, RG distortion field,
    /// and UV breakup field. This ABI occupies t9..t11 in 80A9E4DB.
    pub procedural_wear: Option<[TagHash; 3]>,
    /// Shader-family constants in the semantic layout decoded below.
    pub constants: [[f32; 4]; 24],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunnerOcclusionMaterial {
    pub texture: TagHash,
    pub channel: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaMaskMaterial {
    /// PS t1 scalar coverage texture. Tiger samples this independently from
    /// the t0 colour texture; using t0 alpha turns runner cutouts into blocks.
    pub texture: TagHash,
    /// Coverage = sample.r * remap[1] + remap[0]. Some compiled runner
    /// surfaces amplify their scalar mask before the authored cutoff.
    pub remap: [f32; 2],
    pub threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedAtlasDetailMaterial {
    /// Linear PS t1 field multiplied into the direct sRGB PS t0 atlas.
    pub detail: TagHash,
    pub projection: [f32; 4],
    pub exponent: f32,
    pub base: [f32; 3],
    pub scale: [f32; 3],
    /// Constant RT2.g written by this audited material ABI.
    pub ambient_occlusion: f32,
}

#[derive(Debug, Clone)]
pub struct WireframeMaterialTextures {
    pub color: Option<TagHash>,
    pub normal: Option<TagHash>,
    pub emissive: Option<TagHash>,
    pub control: Option<TagHash>,
    /// Compiled character common-surface ABI. PS t0 remains local albedo; this
    /// block preserves t1/t2/t3/t7 and the constants that compose detail.
    pub character_surface: Option<CharacterSurfaceMaterial>,
    /// Additional packed/detail layers consumed by audited runner shaders.
    pub runner_layered_surface: Option<RunnerLayeredSurfaceMaterial>,
    /// Independent runner AO mask. The audited alpha/physical shader averages
    /// this scalar with its geometry/procedural occlusion before RT2.g.
    pub runner_occlusion: Option<RunnerOcclusionMaterial>,
    pub alpha_mask: Option<AlphaMaskMaterial>,
    pub shared_atlas_detail: Option<SharedAtlasDetailMaterial>,
    pub roughness_channel: u8,
    pub sampler: Option<TagHash>,
    pub aux: Vec<TagHash>,
    pub layers: Vec<WireframeMaterialLayer>,
    pub color_tint: [u8; 4],
    pub mask_palette: Option<[[f32; 4]; 2]>,
    pub gear_dye: Option<GearDyeMaterial>,
    pub gear_dye_default: Option<[f32; 4]>,
    pub gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    /// Inherited GearDye `Worn Dye` object-channel contribution. Compiled
    /// shaders add this to the selected Dye channel before sampling t0.
    pub gear_worn_dye_palette: Option<[[f32; 4]; 6]>,
    /// Mod-local GearDye `Dye Detail` contribution. This is independent from
    /// the physical t5/t6/t7 rarity wear maps.
    pub gear_dye_detail_palette: Option<[[f32; 4]; 6]>,
    pub mod_wear: Option<WeaponModConditionMaterial>,
    pub surface_condition: Option<WeaponSurfaceConditionMaterial>,
    /// Object-space contour/detail layer decoded from the common gear surface
    /// shader. The control map selects which material IDs receive the layer;
    /// the bound field texture perturbs the authored tri-planar line function.
    pub gear_pattern: Option<GearPatternMaterial>,
    /// The technique directly owns a shared colour/decal atlas. Shared atlases
    /// often resemble engine debug sheets globally, but must not be removed
    /// when shader bindings prove this material consumes one as visible data.
    pub authored_shared_atlas: bool,
    /// Authored investment-decal shader inputs. Runner decals encode a source
    /// selector in UV.x and keep opacity in a separate BC4 atlas; treating the
    /// colour atlas as an ordinary material produces the large white quads.
    pub investment_decal: Option<InvestmentDecalMaterial>,
    pub emissive_strength: u8,
    /// Textureless Tiger material base colour decoded from the pixel shader's
    /// authored constant-buffer defaults. `frame_color == -1` is an engine
    /// sentinel selecting this value; it is not transparency.
    pub solid_color: Option<[f32; 4]>,
    /// Authored `(roughness, metalness)` for a decoded textureless material.
    pub solid_surface: Option<[f32; 2]>,
    /// Authored discrete iridescence/material-response selector. `None` means
    /// shader ABI exposes no such channel; renderer must display ID 0.
    pub iridescence_id: Option<f32>,
    /// Authored colour filters used by stage-8 transmission/distortion
    /// shaders. Colours come from compiled pixel-shader material blocks;
    /// stage 8 itself stores displacement, not a universal blue surface.
    pub transmission: Option<TransmissionMaterial>,
    /// Shader-proven stage-8 forward coating. Presence comes from compiled
    /// PS/TFX ABI and resource shape, never weapon or skin identity.
    pub forward_coating: Option<ForwardCoatingMaterial>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionMaterial {
    pub colors: [[f32; 4]; 2],
    /// `(roughness, metalness, _, _)` paired with each decoded colour.
    pub surfaces: [[f32; 4]; 2],
    pub color_count: u8,
    /// `(opacity, scene gain)` of a refractive-glass program. Its colours are
    /// the near and far absorption tints applied to the scene behind it.
    pub absorption: Option<[f32; 2]>,
    /// Rows of a halftone-glow program: scroll rates, noise remap, dot grid
    /// and dot/alpha remap. Its colours are the two glow tints (gain in the
    /// first alpha) and its surfaces the two noise transforms at time zero.
    pub halftone: Option<[[f32; 4]; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardCoatingMaterial {
    /// Linear PS t1 procedural response sampled with object-space triplanar UVs.
    pub detail: TagHash,
    /// PS t2 authored environment cubemap. Quicktag uses scene IBL for preview
    /// lighting but retains this dependency as part of material evidence.
    pub environment: TagHash,
    /// Authored PS sampler bound to the local environment cubemap at s2.
    pub environment_sampler: TagHash,
    pub colors: [[f32; 4]; 2],
    pub incidence_remap: [f32; 2],
    pub coverage: f32,
    pub projection: [f32; 4],
    pub projection_exponent: f32,
    pub detail_remap: [f32; 2],
    pub response_remap: [f32; 2],
    /// Minimum/maximum authored mip floor selected by the detail response.
    pub environment_lod: [f32; 2],
    pub environment_remap: [f32; 2],
    pub environment_strength: f32,
    /// `(base scale, base bias, _, _)` applied before the local cubemap term.
    pub environment_params: [f32; 4],
    pub specular_colors: [[f32; 4]; 2],
    pub specular_exponents: [f32; 2],
    pub specular_strengths: [f32; 2],
    /// Normal-direction scales used by the two authored grazing lobes.
    pub lobe_direction_scales: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvestmentDecalMode {
    SelectorMask,
    DetailSelectorMask,
    /// Stage-2 decals whose pixel shader consumes the already-rendered
    /// screen-space surface normal as its external t2 input.
    SceneNormalColorMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvestmentDecalMaskMode {
    Threshold,
    UvSplit,
    Binary,
}

/// Decoded constants and resources from Marathon's investment-decal pixel
/// shader ABI. This is shared by weapon and runner geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvestmentDecalMaterial {
    pub mode: InvestmentDecalMode,
    pub color: TagHash,
    pub mask: TagHash,
    pub detail: Option<TagHash>,
    pub selector_colors: [[f32; 4]; 5],
    pub selector_color_count: u8,
    pub atlas_selector_max: u8,
    pub mask_mode: InvestmentDecalMaskMode,
    pub mask_threshold: f32,
    pub detail_transform: [f32; 4],
    pub detail_base: [f32; 4],
    pub detail_scale: [f32; 4],
    pub grayscale_remap: [f32; 4],
    pub positive_mask_remap: [f32; 4],
    pub negative_mask_remap: [f32; 4],
    pub output_gate: f32,
}

/// The three condition tiers authored for Marathon weapon mods. The game uses
/// rarity as physical age: Enhanced is oldest, Deluxe is the middle tier, and
/// Superior is newest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeaponModRarity {
    Enhanced,
    Deluxe,
    Superior,
}

impl WeaponModRarity {
    pub fn tier(self) -> u8 {
        match self {
            Self::Enhanced => 1,
            Self::Deluxe => 2,
            Self::Superior => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeaponModPreviewAttachment {
    pub model_tag: TagHash,
    pub rarity: Option<WeaponModRarity>,
    /// Engine `unique_id` object channel. Pattern spawn assigns a random
    /// inclusive 0..1 scalar; TFX turns it into wear projection offsets.
    pub unique_id: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeaponModConditionMaterial {
    /// Additional common-surface texture authored at PS t5.
    pub scratches: TagHash,
    /// Additional common-surface texture authored at PS t6.
    pub grime: TagHash,
    /// Broad chipped/worn mask authored at PS t7.
    pub damage: TagHash,
    /// Triplanar scale/offset read from PS cbuffer 14 for t5.
    pub scratches_projection: [f32; 4],
    /// Triplanar scale/offset evaluated from PS TFX output 39 for t6.
    pub grime_projection: [f32; 4],
    /// Triplanar scale/offset evaluated from PS TFX output 40 for t7.
    pub damage_projection: [f32; 4],
    /// Projection change authored by TFX when `unique_id` moves 0 -> 1.
    pub grime_projection_unique_delta: [f32; 4],
    pub damage_projection_unique_delta: [f32; 4],
    /// PS cbuffer 15/16: `base + t5.r * scale`, before shader's common
    /// 4.5947933 normalization.
    pub scratches_remap_base: [f32; 4],
    pub scratches_remap_scale: [f32; 4],
    /// PS cbuffer 50: blend between base/scratch branch and t6/t7 wear branch.
    pub condition_blend: f32,
    /// PS outputs 24, 42, and 49 for age-channel values 1..3. Rows map to
    /// Enhanced, Deluxe, Superior. These are shader controls, not guessed
    /// brightness multipliers.
    pub condition_controls: [[f32; 3]; 3],
    /// Per-spawn Pattern object-channel value.
    pub unique_id: f32,
    /// `None` while inspecting the standalone mesh; populated for a selected
    /// mod instance in the weapon simulator.
    pub rarity: Option<WeaponModRarity>,
}

/// Common weapon-body condition pass. Unlike detachable-mod wear, this ABI
/// owns one packed breakup field at PS t6 and changes albedo, roughness, and
/// detail-normal response together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeaponSurfaceConditionMaterial {
    /// PS t2 authored material response. Common weapon shaders remap red to
    /// G-buffer metalness; it is not the control atlas roughness channel.
    pub response: TagHash,
    /// PS t4 object-space material-detail field selected by t3 RGB IDs.
    pub detail: TagHash,
    pub breakup: TagHash,
    pub detail_projection: [f32; 4],
    pub detail_exponent: f32,
    /// Material-class fallback roughness mixed with PS t3.a by the projected
    /// t4 field. These are physical-surface parameters, never albedo gains.
    pub detail_roughness: f32,
    pub detail_remap: [f32; 2],
    pub projection: [f32; 4],
    pub phase: f32,
    pub triangle: [f32; 4],
    pub orientation: [f32; 2],
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub normal_flatten: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WireframeMaterialLayer {
    pub color: Option<TagHash>,
    pub normal: Option<TagHash>,
    pub emissive: Option<TagHash>,
}

impl Default for WireframeMaterialTextures {
    fn default() -> Self {
        Self {
            color: None,
            normal: None,
            emissive: None,
            control: None,
            character_surface: None,
            runner_layered_surface: None,
            runner_occlusion: None,
            alpha_mask: None,
            shared_atlas_detail: None,
            roughness_channel: 0,
            sampler: None,
            aux: vec![],
            layers: vec![],
            color_tint: [255, 255, 255, 255],
            mask_palette: None,
            gear_dye: None,
            gear_dye_default: None,
            gear_dye_palette: None,
            gear_worn_dye_palette: None,
            gear_dye_detail_palette: None,
            mod_wear: None,
            surface_condition: None,
            gear_pattern: None,
            authored_shared_atlas: false,
            investment_decal: None,
            emissive_strength: 0,
            solid_color: None,
            solid_surface: None,
            iridescence_id: None,
            transmission: None,
            forward_coating: None,
        }
    }
}

/// Decoded constants/resources for Tiger's procedural gear-pattern branch.
///
/// This is deliberately data-driven. Weapon/skin tags do not identify the
/// effect: the compiled pixel-shader ABI (control t3, field t6, material IDs
/// 2/4, and its constant block) does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearPatternMaterial {
    pub field: TagHash,
    pub projection: [f32; 4],
    pub normal_power: f32,
    pub field_midpoint: f32,
    pub warp: [f32; 2],
    pub stripe: [f32; 4],
    pub contour: [f32; 4],
    pub contour_remap: [f32; 4],
    pub colors: [[f32; 4]; 2],
}

#[derive(Debug, Clone)]
pub struct MeshSourcePreview {
    pub kind: &'static str,
    pub buffer_index: usize,
    pub technique: Option<TagHash>,
    pub index_start: u32,
    pub index_count: u32,
    pub primitive_type: u8,
    pub raw_lod_category: u8,
    /// Viewer-friendly LOD level derived from `raw_lod_category`.
    pub lod_category: u8,
    pub input_layout_index: Option<u8>,
    pub index_buffer: TagHash,
    pub vertex0_buffer: TagHash,
    pub vertex1_buffer: TagHash,
    pub color_buffer: TagHash,
    pub uv_transform: Option<UvTransformPreview>,
    pub shader_constants: Vec<ShaderConstantPreview>,
}

#[derive(Debug, Clone, Copy)]
pub struct UvTransformPreview {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct ShaderConstantPreview {
    pub name: &'static str,
    pub value: [f32; 4],
    pub source: &'static str,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPreviewVertex {
    pub position: [f32; 3],
}

pub struct GpuWireframePreview {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub line_count: u32,
}

impl GpuWireframePreview {
    pub fn create(device: &wgpu::Device, wireframe: &WireframePreview) -> Option<Self> {
        if wireframe.vertices.is_empty() {
            return None;
        }

        let vertices = wireframe
            .vertices
            .iter()
            .map(|position| GpuPreviewVertex {
                position: *position,
            })
            .collect_vec();
        let line_indices = triangle_indices_to_line_indices(&wireframe.indices);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quicktag_geometry_preview_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = (!line_indices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quicktag_geometry_preview_indices"),
                contents: bytemuck::cast_slice(&line_indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        });

        Some(Self {
            vertex_buffer,
            index_buffer,
            vertex_count: vertices.len() as u32,
            index_count: line_indices.len() as u32,
            line_count: line_indices.len() as u32 / 2,
        })
    }

    pub fn has_vertex_buffer(&self) -> bool {
        let _ = &self.vertex_buffer;
        true
    }

    pub fn has_index_buffer(&self) -> bool {
        self.index_buffer.is_some()
    }
}

impl GeometryTagPreview {
    pub fn load(
        cache: Arc<TagCache>,
        tag: TagHash,
        entry: &UEntryHeader,
        tag_type: TagType,
        tag_data: &[u8],
    ) -> Option<Self> {
        if matches!(tag_type, TagType::VertexBuffer { is_header: true }) {
            return Some(Self {
                kind: GeometryPreviewKind::VertexBuffer(
                    load_vertex_buffer_preview_for_tag(tag, entry, tag_data).ok()?,
                ),
            });
        }

        if matches!(tag_type, TagType::IndexBuffer { is_header: true }) {
            return Some(Self {
                kind: GeometryPreviewKind::IndexBuffer(
                    load_index_buffer_preview_for_tag(tag, entry, tag_data).ok()?,
                ),
            });
        }

        model_label_for_reference(entry.reference).map(|label| Self {
            kind: GeometryPreviewKind::Model(load_model_preview(cache, tag, entry, label)),
        })
    }

    pub fn wireframe(&self) -> Option<&WireframePreview> {
        match &self.kind {
            GeometryPreviewKind::VertexBuffer(buffer) => buffer.wireframe.as_ref(),
            GeometryPreviewKind::IndexBuffer(_) => None,
            GeometryPreviewKind::Model(model) => model.wireframe.as_ref(),
        }
    }

    pub fn load_model_with_weapon_mod_attachments(
        cache: Arc<TagCache>,
        tag: TagHash,
        entry: &UEntryHeader,
        weapon_pattern: TagHash,
        weapon_owner: TagHash,
        attachments: &[WeaponModPreviewAttachment],
    ) -> Option<Self> {
        let label = model_label_for_reference(entry.reference)?;
        let mut model_tags = selected_model_geometry_tags(&cache, tag, entry.reference);
        let authored_default_geometry =
            weapon_default_mod_patterns(&cache, weapon_pattern, weapon_owner)
                .into_iter()
                .flat_map(|pattern| pattern_geometry_tags(&cache, pattern))
                .collect::<rustc_hash::FxHashSet<_>>();
        // Concrete weapon Patterns already reference empty-slot meshes. Strip
        // those branches before rebuilding active slot state below; otherwise
        // selecting a runtime mod leaves its default underneath it.
        model_tags.retain(|geometry| !authored_default_geometry.contains(geometry));
        let explicit_geometry_by_attachment = attachments
            .iter()
            .flat_map(|attachment| {
                let Some(pose) =
                    weapon_mod_attachment_pose(&cache, weapon_owner, attachment.model_tag)
                else {
                    return vec![];
                };
                package_manager()
                    .get_entry(attachment.model_tag)
                    .map(|entry| match entry.reference {
                        CLASS_GEOMETRY_RESOURCE => {
                            vec![(
                                attachment.model_tag,
                                attachment.model_tag,
                                pose,
                                attachment.rarity,
                                attachment.unique_id,
                            )]
                        }
                        CLASS_PATTERN | CLASS_PATTERN_COMPONENT => {
                            pattern_nearest_geometry_tags(&cache, attachment.model_tag)
                                .into_iter()
                                .map(|geometry| {
                                    (
                                        attachment.model_tag,
                                        geometry,
                                        pose,
                                        attachment.rarity,
                                        attachment.unique_id,
                                    )
                                })
                                .collect()
                        }
                        _ => vec![],
                    })
                    .unwrap_or_default()
            })
            .unique_by(|(attachment, geometry, _pose, _rarity, _unique_id)| {
                (*attachment, *geometry)
            })
            .collect_vec();
        let explicit_geometry = explicit_geometry_by_attachment
            .iter()
            .map(|(_attachment, geometry, _pose, _rarity, _unique_id)| *geometry)
            .unique()
            .collect_vec();
        // Only the current selector may add mod geometry. Weapon Pattern graphs
        // also reference unequipped/default visual branches; treating those as
        // model parts was the abandoned simulator's source of gray ghost mods.
        model_tags.extend(explicit_geometry);
        model_tags = model_tags.into_iter().unique().collect();
        let attachment_poses = explicit_geometry_by_attachment
            .iter()
            .map(
                |(attachment, geometry, pose, rarity, unique_id)| ResolvedWeaponModAttachment {
                    pattern: *attachment,
                    geometry: *geometry,
                    pose: *pose,
                    rarity: *rarity,
                    unique_id: *unique_id,
                },
            )
            .collect_vec();
        Some(Self {
            kind: GeometryPreviewKind::Model(load_model_preview_from_tags(
                cache,
                tag,
                entry,
                label,
                model_tags,
                &attachment_poses,
            )),
        })
    }
}

const CLASS_WEAPON_MOD_ATTACHMENTS: u32 = 0x80809F82;
const CLASS_PATTERN_CHANNEL_BINDINGS: u32 = 0x8080BACC;
const CLASS_WEAPON_MOD_VISUAL_BINDING: u32 = 0x808032C9;
const CLASS_SKELETON_NODE_HIERARCHY: u32 = 0x8080AF42;
const CLASS_SKELETON_TRANSFORMS: u32 = 0x8080BF47;
const CLASS_VECTOR4: u32 = 0x80800090;
const CLASS_PATTERN_VECTOR_BINDINGS: u32 = 0x8080AF85;
const CLASS_PATTERN_OBJECT_CHANNELS: u32 = 0x8080AF86;
const PATTERN_LOCAL_SCOPE_HASH: u32 = 0x811C9DC5;

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct PatternObjectChannelEvidence {
    pub owner: TagHash,
    pub depth: usize,
    pub vectors: Vec<[f32; 4]>,
    pub channels: Vec<PatternObjectChannelDeclaration>,
    pub bindings: Vec<PatternVectorBindingEvidence>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct PatternObjectChannelDeclaration {
    pub hash: u32,
    pub bytecode: Vec<u8>,
    pub constants: Vec<[f32; 4]>,
    pub interpolation: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct PatternVectorBindingEvidence {
    pub scope: u32,
    pub parameter: u32,
    pub vector_index: u32,
    pub value: Option<[f32; 4]>,
}

// The pattern compiler hashes the six material parameters independently from
// their serialized vector positions. Those positions are intentionally
// shuffled between patterns, so the binding table is the source of truth.
const GEAR_DYE_COLOR_PARAMETERS: [u32; 6] = [
    0x1B3D64F3, 0x1B3D64F6, 0x1B3D64F0, 0x1B3D64F1, 0x1B3D64F7, 0x1B3D64F4,
];
const GEAR_WORN_DYE_COLOR_PARAMETERS: [u32; 6] = [
    0xC8939EBF, 0xC8939EBA, 0xC8939EBC, 0xC8939EBD, 0xC8939EBB, 0xC8939EB8,
];
const GEAR_DYE_DETAIL_COLOR_PARAMETERS: [u32; 6] = [
    0x3CC0E32F, 0x3CC0E32A, 0x3CC0E32C, 0x3CC0E32D, 0x3CC0E32B, 0x3CC0E328,
];
const GEAR_DYE_ROUGHNESS_PARAMETERS: [u32; 6] = [
    0xBF1554A8, 0xBF1554AA, 0xBF1554AB, 0xBF1554AD, 0xBF1554AC, 0xBF1554AF,
];
const GEAR_DYE_METAL_PARAMETERS: [u32; 6] = [
    0xD5754C52, 0xD5754C50, 0xD5754C51, 0xD5754C57, 0xD5754C56, 0xD5754C55,
];

#[derive(Debug, Clone, Copy)]
struct ResolvedWeaponModAttachment {
    pattern: TagHash,
    geometry: TagHash,
    pose: WeaponModAttachmentPose,
    rarity: Option<WeaponModRarity>,
    unique_id: f32,
}

/// Authored, weapon-specific transform for one visual mod family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeaponModAttachmentPose {
    pub family_id: u32,
    pub variant_id: u32,
    pub bone_index: u32,
    pub rotation: [f32; 4],
    pub translation: [f32; 3],
    pub scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ObjectSpaceTransform {
    rotation: [f32; 4],
    translation: [f32; 3],
    scale: f32,
}

#[derive(Debug, Clone)]
struct SkeletonPreview {
    node_hashes: Vec<u32>,
    parents: Vec<i32>,
    transforms: Vec<ObjectSpaceTransform>,
}

/// Resolves a mod pattern against the selected weapon's authored socket table.
/// The family identifier is deliberately discovered from tag structure rather
/// than inferred from display names or descriptions.
pub fn weapon_mod_attachment_pose(
    cache: &TagCache,
    weapon: TagHash,
    modification: TagHash,
) -> Option<WeaponModAttachmentPose> {
    // A mesh resource can be shared by many weapons. Only walk descendants of
    // the selected owning container; climbing through parents would merge the
    // socket tables of unrelated guns and produce plausible-but-wrong poses.
    let weapon_poses = descendant_pattern_nodes(cache, weapon, 12)
        .into_iter()
        .flat_map(|node| {
            weapon_attachment_poses(node)
                .into_iter()
                .map(move |pose| (node, pose))
        })
        .collect_vec();
    if weapon_poses.is_empty() {
        return None;
    }
    let families = weapon_poses
        .iter()
        .map(|(_node, pose)| pose.family_id)
        .collect::<rustc_hash::FxHashSet<_>>();

    let matched_family = mod_attachment_family(cache, modification, &families)?;
    let mut pose = weapon_poses
        .into_iter()
        .find(|(_node, pose)| pose.family_id == matched_family)
        .map(|(_node, pose)| pose)?;
    let bone = weapon_skeleton_bone_transform(cache, weapon, pose.bone_index as usize);
    if pose.bone_index != 0 && bone.is_none() {
        return None;
    }
    if let Some(bone) = bone {
        let local_translation = pose.translation.map(|value| value * bone.scale);
        let rotated_translation = rotate_quaternion(local_translation, bone.rotation);
        pose.translation =
            std::array::from_fn(|axis| bone.translation[axis] + rotated_translation[axis]);
        pose.rotation = multiply_quaternions(bone.rotation, pose.rotation);
        pose.scale *= bone.scale;
    }
    // A visual mod is itself a Pattern entity. Its visual binding names the
    // local attachment frame that must meet the weapon socket. Tiger therefore
    // places it as `weapon_socket * inverse(mod_attachment_frame)`, not by
    // treating the mod mesh origin as the socket origin.
    if let Some(anchor) = weapon_mod_local_attachment_anchor(cache, modification, matched_family) {
        let socket = ObjectSpaceTransform {
            rotation: pose.rotation,
            translation: pose.translation,
            scale: pose.scale,
        };
        let placed =
            compose_object_space_transforms(socket, inverse_object_space_transform(anchor)?);
        pose.rotation = placed.rotation;
        pose.translation = placed.translation;
        pose.scale = placed.scale;
    }
    Some(pose)
}

fn weapon_mod_local_attachment_anchor(
    cache: &TagCache,
    modification: TagHash,
    socket_family: u32,
) -> Option<ObjectSpaceTransform> {
    let endian = package_manager().version.endian();
    let anchor_families = descendant_pattern_nodes(cache, modification, 8)
        .into_iter()
        .filter_map(|node| package_manager().read_tag(node).ok())
        .flat_map(|data| {
            (0..data.len().saturating_sub(11))
                .step_by(4)
                .filter_map(move |offset| {
                    (read_u32_at(&data, offset, endian) == Some(CLASS_WEAPON_MOD_VISUAL_BINDING)
                        && read_u32_at(&data, offset + 4, endian) == Some(socket_family))
                    .then(|| read_u32_at(&data, offset + 8, endian))
                    .flatten()
                    .filter(|family| *family != 0)
                })
                .collect_vec()
        })
        .unique()
        .collect_vec();
    let [anchor_family] = anchor_families.as_slice() else {
        return None;
    };

    let mut anchors = descendant_pattern_nodes(cache, modification, 12)
        .into_iter()
        .flat_map(weapon_attachment_poses)
        .filter(|pose| pose.family_id == *anchor_family)
        .map(|pose| ObjectSpaceTransform {
            rotation: pose.rotation,
            translation: pose.translation,
            scale: pose.scale,
        });
    let first = anchors.next()?;
    anchors
        .all(|anchor| object_space_transforms_match(first, anchor))
        .then_some(first)
}

fn inverse_object_space_transform(transform: ObjectSpaceTransform) -> Option<ObjectSpaceTransform> {
    (transform.scale.is_finite() && transform.scale.abs() > f32::EPSILON).then(|| {
        let rotation = [
            -transform.rotation[0],
            -transform.rotation[1],
            -transform.rotation[2],
            transform.rotation[3],
        ];
        let inverse_scale = transform.scale.recip();
        let translation = rotate_quaternion(
            transform.translation.map(|value| -value * inverse_scale),
            rotation,
        );
        ObjectSpaceTransform {
            rotation,
            translation,
            scale: inverse_scale,
        }
    })
}

fn compose_object_space_transforms(
    parent: ObjectSpaceTransform,
    child: ObjectSpaceTransform,
) -> ObjectSpaceTransform {
    let local_translation = child.translation.map(|value| value * parent.scale);
    let rotated_translation = rotate_quaternion(local_translation, parent.rotation);
    ObjectSpaceTransform {
        rotation: multiply_quaternions(parent.rotation, child.rotation),
        translation: std::array::from_fn(|axis| {
            parent.translation[axis] + rotated_translation[axis]
        }),
        scale: parent.scale * child.scale,
    }
}

/// Returns attachment Patterns spawned by the weapon Pattern when no runtime
/// mod overrides their socket family. Defaults are authored as nested Pattern
/// branches beneath the concrete weapon, while socket transforms live in a
/// separate Pattern component. Walking only Pattern/PatternComponent edges is
/// important: shared runtime resources also reference every mod in the game.
pub fn weapon_default_mod_patterns(
    cache: &TagCache,
    weapon_pattern: TagHash,
    socket_owner: TagHash,
) -> Vec<TagHash> {
    let socket_families = descendant_pattern_nodes(cache, socket_owner, 12)
        .into_iter()
        .flat_map(weapon_attachment_poses)
        .map(|pose| pose.family_id)
        .collect::<rustc_hash::FxHashSet<_>>();
    if socket_families.is_empty() {
        return vec![];
    }

    let mut defaults = rustc_hash::FxHashMap::<u32, (usize, TagHash)>::default();
    for (pattern, depth) in descendant_pattern_nodes_with_depth(cache, weapon_pattern, 12) {
        if depth == 0
            || package_manager()
                .get_entry(pattern)
                .is_none_or(|entry| entry.reference != CLASS_PATTERN)
            || pattern_nearest_geometry_tags(cache, pattern).is_empty()
        {
            continue;
        }
        let Some(family) = mod_attachment_family(cache, pattern, &socket_families) else {
            continue;
        };
        match defaults.entry(family) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((depth, pattern));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if (depth, pattern) < *entry.get() {
                    entry.insert((depth, pattern));
                }
            }
        }
    }

    defaults
        .into_iter()
        .sorted_by_key(|(family, (depth, pattern))| (*depth, *family, *pattern))
        .map(|(_family, (_depth, pattern))| pattern)
        .collect()
}

/// Filters authored defaults using same family override rule as engine: an
/// equipped mod replaces only empty-slot visual sharing its socket family.
pub fn weapon_unoccupied_default_mod_patterns(
    cache: &TagCache,
    weapon_pattern: TagHash,
    socket_owner: TagHash,
    equipped_mods: &[TagHash],
) -> Vec<TagHash> {
    let occupied_families = equipped_mods
        .iter()
        .filter_map(|modification| {
            weapon_mod_attachment_pose(cache, socket_owner, *modification)
                .map(|pose| pose.family_id)
        })
        .collect::<rustc_hash::FxHashSet<_>>();
    weapon_default_mod_patterns(cache, weapon_pattern, socket_owner)
        .into_iter()
        .filter(|default| {
            weapon_mod_attachment_pose(cache, socket_owner, *default)
                .is_some_and(|pose| !occupied_families.contains(&pose.family_id))
        })
        .collect()
}

/// Reusable index of authored weapon socket tables. Building this once avoids
/// rereading every Pattern component for each weapon in the Models catalog.
pub struct WeaponModSocketIndex {
    candidates: Vec<(TagHash, Vec<WeaponModAttachmentPose>)>,
}

/// Pattern layout with tag references removed, paired with its quantized mesh
/// bounds transform. Exact payload identity is the strongest match; cosmetic
/// assets compiled separately from a weapon can still retain its coordinate
/// frame while meshes, materials, and Pattern topology receive new tag IDs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelPatternStructureSignature {
    payload: Vec<u8>,
    geometry_transforms: Vec<ModelGeometryTransformSignature>,
}

impl ModelPatternStructureSignature {
    /// Distance between the closest authored geometry quantization frames.
    /// Weapon skins frequently recompile the mesh and Pattern wrapper, but
    /// retain the owning weapon's coordinate frame to keep sockets, effects,
    /// and first-person animation aligned. Tag IDs and mesh topology are not
    /// stable across those recompiles; this frame is.
    pub fn closest_geometry_distance(&self, other: &Self) -> Option<f32> {
        self.geometry_transforms
            .iter()
            .cartesian_product(&other.geometry_transforms)
            .map(|(left, right)| {
                let normalization = left
                    .scale
                    .into_iter()
                    .chain(right.scale)
                    .map(i32::unsigned_abs)
                    .max()
                    .unwrap_or(1)
                    .max(1) as f32;
                left.scale
                    .into_iter()
                    .chain(left.offset)
                    .zip(right.scale.into_iter().chain(right.offset))
                    .map(|(left, right)| left.abs_diff(right) as f32 / normalization)
                    .fold(0.0_f32, f32::max)
            })
            .min_by(f32::total_cmp)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ModelGeometryTransformSignature {
    scale: [i32; 3],
    offset: [i32; 3],
}

pub fn model_pattern_structure_signature(
    cache: &TagCache,
    model: TagHash,
) -> Option<ModelPatternStructureSignature> {
    let entry = package_manager().get_entry(model)?;
    if entry.reference != CLASS_PATTERN {
        return None;
    }
    let payload = normalized_pattern_payload(cache, model)?;
    const PRECISION: f32 = 100_000.0;
    let quantize = |value: f32| (value * PRECISION).round() as i32;
    let geometry_transforms = selected_model_geometry_tags(cache, model, entry.reference)
        .into_iter()
        .filter_map(|geometry| package_manager().read_tag(geometry).ok())
        .filter_map(|data| {
            let transform =
                read_geometry_position_transform(&data, package_manager().version.endian())?;
            Some(ModelGeometryTransformSignature {
                scale: transform.scale.map(quantize),
                offset: transform.offset.map(quantize),
            })
        })
        .collect();
    Some(ModelPatternStructureSignature {
        payload,
        geometry_transforms,
    })
}

impl WeaponModSocketIndex {
    pub fn new() -> Self {
        let candidates = package_manager()
            .get_all_by_reference(CLASS_PATTERN)
            .into_iter()
            .chain(package_manager().get_all_by_reference(CLASS_PATTERN_COMPONENT))
            .filter_map(|(candidate, _entry)| {
                let poses = weapon_attachment_poses(candidate);
                (!poses.is_empty()).then_some((candidate, poses))
            })
            .collect();
        Self { candidates }
    }

    /// Finds the Pattern component whose attachment family/variant pairs were
    /// compiled into the selected weapon render Pattern. This joins two
    /// authored investment branches without names, descriptions, or tag IDs.
    pub fn owner_for(
        &self,
        cache: &TagCache,
        model: TagHash,
        modifications: &[TagHash],
    ) -> Option<TagHash> {
        weapon_mod_socket_owner_from_index(cache, model, modifications, &self.candidates)
    }

    /// Returns the authored socket-family/variant identity carried by a weapon
    /// or cosmetic Pattern, without needing an already-classified mod list.
    /// This is the engine join used to classify updated hash-only weapon skins:
    /// cosmetic group ordinals can move, while socket identities must remain
    /// stable for attachments to render at the correct weapon-specific pose.
    pub fn signature_for_model(&self, cache: &TagCache, model: TagHash) -> Option<Vec<(u32, u32)>> {
        if let Some(signature) = weapon_mod_socket_signature_from_ancestry(cache, model) {
            return Some(signature);
        }
        let owner = weapon_mod_socket_owner_for_model(cache, model, &self.candidates)?;
        let poses = self
            .candidates
            .iter()
            .find_map(|(candidate, poses)| (*candidate == owner).then_some(poses))?;
        let mut signature = poses
            .iter()
            .map(|pose| (pose.family_id, pose.variant_id))
            .collect_vec();
        signature.sort_unstable();
        signature.dedup();
        (!signature.is_empty()).then_some(signature)
    }
}

fn weapon_mod_socket_signature_from_ancestry(
    cache: &TagCache,
    model: TagHash,
) -> Option<Vec<(u32, u32)>> {
    let tables = descendant_pattern_nodes_with_depth(cache, model, 12)
        .into_iter()
        .filter_map(|(node, depth)| {
            let poses = weapon_attachment_poses(node);
            (!poses.is_empty()).then_some((depth, poses))
        })
        .collect_vec();
    // Runtime feasibility is the union of authored socket tables. Depth-1
    // carries visual-mod families; deeper tables carry defaults/variants.
    // Compatibility is subsequently intersected with investment archetypes,
    // so shared visual families do not make unrelated guns compatible.
    let mut signature = tables
        .into_iter()
        .flat_map(|(_, poses)| {
            poses
                .into_iter()
                .map(|pose| (pose.family_id, pose.variant_id))
        })
        .collect_vec();
    signature.sort_unstable();
    signature.dedup();
    (!signature.is_empty()).then_some(signature)
}

impl Default for WeaponModSocketIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn weapon_mod_socket_owner_for_model(
    cache: &TagCache,
    model: TagHash,
    socket_candidates: &[(TagHash, Vec<WeaponModAttachmentPose>)],
) -> Option<TagHash> {
    let authored = model_authored_socket_pairs(cache, model);
    if authored.is_empty() {
        return None;
    }

    // Several compiled components may duplicate one table. Group by the
    // semantic pair set, then require one best signature instead of choosing a
    // tag by iteration order when two weapons merely share one broad family.
    let mut signatures = rustc_hash::FxHashMap::<Vec<(u32, u32)>, (usize, TagHash)>::default();
    for (candidate, poses) in socket_candidates {
        let mut signature = poses
            .iter()
            .map(|pose| (pose.family_id, pose.variant_id))
            .collect_vec();
        signature.sort_unstable();
        signature.dedup();
        let overlap = signature
            .iter()
            .filter(|pair| authored.contains(pair))
            .count();
        if overlap == 0 {
            continue;
        }
        signatures
            .entry(signature)
            .and_modify(|entry| {
                if overlap > entry.0 || (overlap == entry.0 && *candidate < entry.1) {
                    *entry = (overlap, *candidate);
                }
            })
            .or_insert((overlap, *candidate));
    }
    let best = signatures.values().map(|(score, _)| *score).max()?;
    let mut winners = signatures
        .values()
        .filter(|(score, _)| *score == best)
        .map(|(_, candidate)| *candidate);
    let winner = winners.next()?;
    winners.next().is_none().then_some(winner)
}

fn weapon_mod_socket_owner_from_index(
    cache: &TagCache,
    model: TagHash,
    modifications: &[TagHash],
    socket_candidates: &[(TagHash, Vec<WeaponModAttachmentPose>)],
) -> Option<TagHash> {
    let rooted = rooted_pattern_equivalent(cache, model);
    let ancestry = std::iter::once((0usize, model))
        .chain(rooted.map(|candidate| (0usize, candidate)))
        .chain(ancestor_pattern_roots(cache, model))
        .chain(
            rooted
                .into_iter()
                .flat_map(|candidate| ancestor_pattern_roots(cache, candidate)),
        )
        .unique_by(|(_depth, candidate)| *candidate)
        .flat_map(|(root_depth, candidate)| {
            descendant_pattern_nodes_with_depth(cache, candidate, 12)
                .into_iter()
                .filter(|(node, _node_depth)| !weapon_attachment_poses(*node).is_empty())
                .map(move |(node, node_depth)| (root_depth + node_depth, node))
        })
        .unique_by(|(_depth, node)| *node)
        .filter_map(|(depth, socket)| {
            let matches = modifications
                .iter()
                .filter(|modification| {
                    weapon_mod_attachment_pose(cache, socket, **modification).is_some()
                })
                .count();
            (matches > 0 && matches == modifications.len()).then_some((matches, depth, socket))
        })
        .max_by_key(|(matches, depth, socket)| {
            (
                *matches,
                std::cmp::Reverse(*depth),
                std::cmp::Reverse(*socket),
            )
        })
        .map(|(_matches, _depth, socket)| socket);
    if ancestry.is_some() {
        return ancestry;
    }

    let model_authored_pairs = model_authored_socket_pairs(cache, model);
    if model_authored_pairs.is_empty() {
        return None;
    }
    socket_candidates
        .iter()
        .filter_map(|(candidate, poses)| {
            let authored_pair_overlap = poses
                .iter()
                .filter(|pose| model_authored_pairs.contains(&(pose.family_id, pose.variant_id)))
                .count();
            if authored_pair_overlap == 0 {
                return None;
            }
            let matches = modifications
                .iter()
                .filter(|modification| {
                    weapon_mod_attachment_pose(cache, *candidate, **modification).is_some()
                })
                .count();
            (matches > 0 && matches == modifications.len()).then_some((
                authored_pair_overlap,
                matches,
                *candidate,
            ))
        })
        .max_by_key(|(authored_pair_overlap, matches, candidate)| {
            (
                *matches,
                *authored_pair_overlap,
                std::cmp::Reverse(*candidate),
            )
        })
        .map(|(_authored_pair_overlap, _matches, candidate)| candidate)
}

fn model_authored_socket_pairs(
    cache: &TagCache,
    model: TagHash,
) -> rustc_hash::FxHashSet<(u32, u32)> {
    let endian = package_manager().version.endian();
    descendant_pattern_nodes(cache, model, 12)
        .into_iter()
        .filter_map(|node| package_manager().read_tag(node).ok())
        .flat_map(|data| {
            (0..data.len().saturating_sub(7))
                .step_by(4)
                .filter_map(move |offset| {
                    Some((
                        read_u32_at(&data, offset, endian)?,
                        read_u32_at(&data, offset + 4, endian)?,
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn mod_attachment_family(
    cache: &TagCache,
    modification: TagHash,
    weapon_families: &rustc_hash::FxHashSet<u32>,
) -> Option<u32> {
    let authored = weapon_mod_authored_families(cache, modification);
    let authored_matches = authored
        .intersection(weapon_families)
        .copied()
        .collect::<rustc_hash::FxHashSet<_>>();
    if !authored.is_empty() {
        // An explicit visual binding is authoritative. Falling through to a
        // raw word overlap when it does not match this weapon can select an
        // unrelated behavior channel and place the mod on the wrong socket.
        return unique_family(authored_matches);
    }

    let endian = package_manager().version.endian();
    let mut structural_matches = rustc_hash::FxHashSet::default();
    let mut fallback_matches = rustc_hash::FxHashSet::default();
    for node in descendant_pattern_nodes(cache, modification, 8) {
        let Ok(data) = package_manager().read_tag(node) else {
            continue;
        };
        let arrays = scan_arrays(&data, endian);
        // The mod visual binding resource authors the socket family as the
        // first value after its class marker. Later values are material/decal
        // channels and can also occur in a weapon's socket table, so treating
        // every raw overlap equally makes multi-part mods ambiguous.
        for offset in (0..data.len().saturating_sub(7)).step_by(4) {
            if read_u32_at(&data, offset, endian) == Some(CLASS_WEAPON_MOD_VISUAL_BINDING)
                && let Some(family) = read_u32_at(&data, offset + 4, endian)
                && weapon_families.contains(&family)
            {
                structural_matches.insert(family);
            }
        }
        for array in arrays
            .iter()
            .filter(|array| array.class == CLASS_PATTERN_CHANNEL_BINDINGS)
        {
            for offset in (array.data_offset..array.end_offset.saturating_sub(3)).step_by(4) {
                if let Some(family) = read_u32_at(&data, offset, endian)
                    && weapon_families.contains(&family)
                {
                    structural_matches.insert(family);
                }
            }
        }
        let socket_ranges = arrays
            .iter()
            .filter(|array| array.class == CLASS_WEAPON_MOD_ATTACHMENTS)
            .map(|array| array.data_offset..array.end_offset)
            .collect_vec();
        for offset in (0..data.len().saturating_sub(3)).step_by(4) {
            if socket_ranges.iter().any(|range| range.contains(&offset)) {
                continue;
            }
            if let Some(family) = read_u32_at(&data, offset, endian)
                && weapon_families.contains(&family)
            {
                fallback_matches.insert(family);
            }
        }
    }
    unique_family(structural_matches).or_else(|| unique_family(fallback_matches))
}

/// Socket-family identifiers explicitly authored by a mod's visual binding.
/// These are the engine join key between investment mods and weapon Pattern
/// attachment tables; names and compatibility prose are not involved.
pub fn weapon_mod_authored_families(
    cache: &TagCache,
    modification: TagHash,
) -> rustc_hash::FxHashSet<u32> {
    let endian = package_manager().version.endian();
    let mut families = rustc_hash::FxHashSet::default();
    for node in descendant_pattern_nodes(cache, modification, 8) {
        let Ok(data) = package_manager().read_tag(node) else {
            continue;
        };
        for offset in (0..data.len().saturating_sub(7)).step_by(4) {
            if read_u32_at(&data, offset, endian) == Some(CLASS_WEAPON_MOD_VISUAL_BINDING)
                && let Some(family) = read_u32_at(&data, offset + 4, endian)
            {
                families.insert(family);
            }
        }
    }
    families
}

fn unique_family(families: rustc_hash::FxHashSet<u32>) -> Option<u32> {
    (families.len() == 1)
        .then(|| families.into_iter().next())
        .flatten()
}

fn weapon_skeleton_bone_transform(
    cache: &TagCache,
    weapon: TagHash,
    bone_index: usize,
) -> Option<ObjectSpaceTransform> {
    let siblings = cache
        .hashes
        .get(&weapon)?
        .references
        .iter()
        .copied()
        .unique()
        .flat_map(|parent| pattern_graph_children(cache, parent))
        .filter(|candidate| *candidate != weapon)
        .unique()
        .sorted()
        .collect_vec();
    let mut transforms = siblings
        .into_iter()
        .filter_map(|candidate| skeleton_bone_transform(candidate, bone_index));
    let first = transforms.next()?;
    transforms
        .all(|transform| object_space_transforms_match(first, transform))
        .then_some(first)
}

fn object_space_transforms_match(left: ObjectSpaceTransform, right: ObjectSpaceTransform) -> bool {
    left.rotation
        .into_iter()
        .chain(left.translation)
        .chain([left.scale])
        .zip(
            right
                .rotation
                .into_iter()
                .chain(right.translation)
                .chain([right.scale]),
        )
        .all(|(left, right)| (left - right).abs() < 0.000001)
}

fn skeleton_bone_transform(tag: TagHash, bone_index: usize) -> Option<ObjectSpaceTransform> {
    skeleton_object_space_transforms(tag)
        .get(bone_index)
        .copied()
}

fn skeleton_object_space_transforms(tag: TagHash) -> Vec<ObjectSpaceTransform> {
    let endian = package_manager().version.endian();
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };
    let arrays = scan_arrays(&data, endian);
    let Some(hierarchy_index) = arrays
        .iter()
        .position(|array| array.class == CLASS_SKELETON_NODE_HIERARCHY)
    else {
        return vec![];
    };
    let hierarchy_count = arrays[hierarchy_index].count;
    let Some(transforms) = arrays
        .iter()
        .skip(hierarchy_index + 1)
        .find(|array| array.class == CLASS_SKELETON_TRANSFORMS && array.count == hierarchy_count)
    else {
        return vec![];
    };
    array_records(&data, *transforms, 0x20)
        .into_iter()
        .filter_map(|record| {
            let rotation = read_vec4_f32(record, 0, endian)?;
            let translation = read_vec4_f32(record, 0x10, endian)?;
            Some(ObjectSpaceTransform {
                rotation,
                translation: [translation[0], translation[1], translation[2]],
                scale: translation[3],
            })
        })
        .collect()
}

fn skeleton_preview(tag: TagHash) -> Option<SkeletonPreview> {
    let endian = package_manager().version.endian();
    let data = package_manager().read_tag(tag).ok()?;
    let arrays = scan_arrays(&data, endian);
    let hierarchy_index = arrays
        .iter()
        .position(|array| array.class == CLASS_SKELETON_NODE_HIERARCHY)?;
    let hierarchy = arrays[hierarchy_index];
    let transforms = arrays
        .iter()
        .skip(hierarchy_index + 1)
        .find(|array| array.class == CLASS_SKELETON_TRANSFORMS && array.count == hierarchy.count)?;
    let hierarchy_records = array_records(&data, hierarchy, 0x10);
    let node_hashes = hierarchy_records
        .iter()
        .filter_map(|record| read_u32_at(record, 0, endian))
        .collect_vec();
    let parents = hierarchy_records
        .iter()
        .filter_map(|record| read_u32_at(record, 4, endian).map(|value| value as i32))
        .collect_vec();
    let transforms = array_records(&data, *transforms, 0x20)
        .into_iter()
        .filter_map(|record| {
            let rotation = read_vec4_f32(record, 0, endian)?;
            let translation = read_vec4_f32(record, 0x10, endian)?;
            Some(ObjectSpaceTransform {
                rotation,
                translation: [translation[0], translation[1], translation[2]],
                scale: translation[3],
            })
        })
        .collect_vec();
    (node_hashes.len() == hierarchy.count
        && parents.len() == hierarchy.count
        && transforms.len() == hierarchy.count)
        .then_some(SkeletonPreview {
            node_hashes,
            parents,
            transforms,
        })
}

fn related_skeletons(
    cache: &TagCache,
    selected_pattern: TagHash,
    model_tags: impl IntoIterator<Item = TagHash>,
) -> Vec<SkeletonPreview> {
    let mut frontier = std::collections::VecDeque::from_iter(
        std::iter::once(selected_pattern)
            .chain(model_tags)
            .map(|tag| (tag, 0_usize)),
    );
    let mut seen = rustc_hash::FxHashSet::default();
    let mut candidates = rustc_hash::FxHashSet::default();
    while let Some((tag, depth)) = frontier.pop_front() {
        if !seen.insert(tag) {
            continue;
        }
        candidates.insert(tag);
        if depth >= 6 {
            continue;
        }
        let neighbors = pattern_graph_children(cache, tag).into_iter().chain(
            cache
                .hashes
                .get(&tag)
                .into_iter()
                .flat_map(|scan| scan.references.iter().copied()),
        );
        for neighbor in neighbors.unique() {
            if package_manager().get_entry(neighbor).is_some_and(|entry| {
                matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
            }) {
                frontier.push_back((neighbor, depth + 1));
            }
        }
    }
    candidates
        .into_iter()
        .filter(|candidate| {
            package_manager()
                .get_entry(*candidate)
                .is_some_and(|entry| entry.reference == CLASS_PATTERN_COMPONENT)
        })
        .filter_map(skeleton_preview)
        .collect()
}

fn apply_inventory_rigid_chain_visibility(
    cache: &TagCache,
    selected_pattern: TagHash,
    parts: &mut [(TagHash, MeshSourcePreview, WireframePreview)],
) {
    let skeletons = related_skeletons(cache, selected_pattern, parts.iter().map(|part| part.0));
    if skeletons.is_empty() {
        return;
    }
    for (_tag, _source, wireframe) in parts {
        let Some(rigid_indices) = wireframe.rigid_indices.as_ref() else {
            continue;
        };
        if rigid_indices.len() != wireframe.vertices.len() {
            continue;
        }
        let mut counts = rustc_hash::FxHashMap::<usize, usize>::default();
        for index in rigid_indices {
            *counts.entry(*index as usize).or_default() += 1;
        }
        let best = skeletons
            .iter()
            .filter_map(|skeleton| {
                let chain = longest_repeated_rigid_chain(skeleton, &counts);
                (!chain.is_empty()).then_some((skeleton, chain))
            })
            .max_by_key(|(_skeleton, chain)| chain.len());
        let Some((skeleton, chain)) = best else {
            continue;
        };
        if chain.len() < 7 {
            continue;
        }
        retain_rigid_chain_window(wireframe, skeleton, &chain);
    }
}

fn longest_repeated_rigid_chain(
    skeleton: &SkeletonPreview,
    counts: &rustc_hash::FxHashMap<usize, usize>,
) -> Vec<usize> {
    let eligible = counts
        .iter()
        .filter_map(|(&index, &count)| {
            (count >= 64 && index < skeleton.parents.len() && index < skeleton.transforms.len())
                .then_some((index, count))
        })
        .collect::<rustc_hash::FxHashMap<_, _>>();
    let mut best = vec![];
    for &tail in eligible.keys() {
        let mut chain = vec![tail];
        let mut current = tail;
        while let Some(parent) = skeleton.parents.get(current).copied() {
            let Ok(parent) = usize::try_from(parent) else {
                break;
            };
            let (Some(&child_count), Some(&parent_count)) =
                (eligible.get(&current), eligible.get(&parent))
            else {
                break;
            };
            let ratio = child_count as f32 / parent_count as f32;
            if !(0.75..=1.25).contains(&ratio) {
                break;
            }
            chain.push(parent);
            current = parent;
        }
        chain.reverse();
        if chain.len() > best.len() {
            best = chain;
        }
    }
    best
}

fn retain_rigid_chain_window(
    wireframe: &mut WireframePreview,
    skeleton: &SkeletonPreview,
    chain: &[usize],
) {
    let Some(rigid_indices) = wireframe.rigid_indices.as_ref() else {
        return;
    };
    let centers = chain
        .iter()
        .map(|&bone| {
            skeleton
                .transforms
                .get(bone)
                .map(|transform| transform.translation)
        })
        .collect::<Option<Vec<_>>>();
    let Some(centers) = centers else { return };
    // The bind pose contains the complete 11-round state chain. Static item
    // presentation exposes only the rounds between two package-authored
    // endpoints: b_bolt at the receiver and the duplicated magazine anchor.
    // The five rounds in that interval already have the correct parallel pose;
    // neither their transforms nor their spacing may be changed.
    let chain_set = chain.iter().copied().collect::<rustc_hash::FxHashSet<_>>();
    let Some(visible) = rigid_chain_window_bones(skeleton, chain, &centers) else {
        return;
    };
    for triangle in wireframe.indices.chunks_exact_mut(3) {
        let hides_chain_vertex = triangle.iter().any(|index| {
            let bone = rigid_indices[*index as usize] as usize;
            chain_set.contains(&bone) && !visible.contains(&bone)
        });
        if hides_chain_vertex {
            triangle[1] = triangle[0];
            triangle[2] = triangle[0];
        }
    }
}

fn rigid_chain_window_bones(
    skeleton: &SkeletonPreview,
    chain: &[usize],
    centers: &[[f32; 3]],
) -> Option<rustc_hash::FxHashSet<usize>> {
    if centers.len() != chain.len() || centers.len() < 2 {
        return None;
    }
    let chain_set = chain.iter().copied().collect::<rustc_hash::FxHashSet<_>>();
    let rest = vec3_normalize(vec3_sub(*centers.last()?, centers[0]));
    let chain_start = centers[0];
    let chain_length = vec3_length(vec3_sub(*centers.last()?, chain_start));
    let bolt_hash = quicktag_core::util::fnv1(b"b_bolt");
    let upper = skeleton
        .node_hashes
        .iter()
        .position(|hash| *hash == bolt_hash)
        .and_then(|index| skeleton.transforms.get(index))
        .map(|transform| vec3_dot(vec3_sub(transform.translation, chain_start), rest))?;
    let mut lower = None::<f32>;
    for (left_index, left) in skeleton.transforms.iter().enumerate() {
        if chain_set.contains(&left_index) {
            continue;
        }
        let duplicate = skeleton
            .transforms
            .iter()
            .enumerate()
            .skip(left_index + 1)
            .any(|(right_index, right)| {
                !chain_set.contains(&right_index)
                    && vec3_length(vec3_sub(left.translation, right.translation)) < 0.00001
            });
        if !duplicate {
            continue;
        }
        let along = vec3_dot(vec3_sub(left.translation, chain_start), rest);
        if (upper..=chain_length).contains(&along) && lower.is_none_or(|current| along < current) {
            lower = Some(along);
        }
    }
    let lower = lower?;
    let spacing = centers
        .windows(2)
        .map(|pair| vec3_length(vec3_sub(pair[1], pair[0])))
        .sum::<f32>()
        / (centers.len() - 1) as f32;
    let visible = chain
        .iter()
        .zip(centers)
        .filter_map(|(&bone, center)| {
            let along = vec3_dot(vec3_sub(*center, chain_start), rest);
            ((upper - spacing * 0.25..=lower + spacing * 0.25).contains(&along)).then_some(bone)
        })
        .collect::<rustc_hash::FxHashSet<_>>();
    (visible.len() >= 2 && visible.len() < chain.len()).then_some(visible)
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn vec3_scale(v: [f32; 3], scale: f32) -> [f32; 3] {
    v.map(|value| value * scale)
}
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn vec3_length(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}
fn vec3_try_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let length = vec3_length(v);
    (length > 0.000001).then(|| vec3_scale(v, length.recip()))
}
fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    vec3_try_normalize(v).unwrap_or([0.0, 0.0, 1.0])
}

fn descendant_pattern_nodes(cache: &TagCache, root: TagHash, max_depth: usize) -> Vec<TagHash> {
    let mut frontier = vec![(root, 0usize)];
    let mut seen = rustc_hash::FxHashSet::default();
    let mut result = vec![];
    seen.insert(root);
    while let Some((node, depth)) = frontier.pop() {
        let Some(entry) = package_manager().get_entry(node) else {
            continue;
        };
        if matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT) {
            result.push(node);
        }
        if depth >= max_depth {
            continue;
        }
        for child in pattern_graph_children(cache, node) {
            if seen.insert(child)
                && package_manager().get_entry(child).is_some_and(|entry| {
                    matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                })
            {
                frontier.push((child, depth + 1));
            }
        }
    }
    result
}

fn descendant_pattern_nodes_with_depth(
    cache: &TagCache,
    root: TagHash,
    max_depth: usize,
) -> Vec<(TagHash, usize)> {
    let mut frontier = std::collections::VecDeque::from([(root, 0usize)]);
    let mut seen = rustc_hash::FxHashSet::default();
    let mut result = vec![];
    seen.insert(root);
    while let Some((node, depth)) = frontier.pop_front() {
        let Some(entry) = package_manager().get_entry(node) else {
            continue;
        };
        if matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT) {
            result.push((node, depth));
        }
        if depth >= max_depth {
            continue;
        }
        for child in pattern_graph_children(cache, node) {
            if seen.insert(child)
                && package_manager().get_entry(child).is_some_and(|entry| {
                    matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                })
            {
                frontier.push_back((child, depth + 1));
            }
        }
    }
    result
}

/// Captures authored Pattern object-channel declarations and their serialized
/// vector bindings without assigning runtime semantics to either structure.
/// A declaration's expression and a binding-table value are distinct evidence;
/// callers must not treat one as the other's default implicitly.
pub(crate) fn pattern_object_channel_evidence(
    cache: &TagCache,
    root: TagHash,
) -> Vec<PatternObjectChannelEvidence> {
    let endian = package_manager().version.endian();
    descendant_pattern_nodes_with_depth(cache, root, 8)
        .into_iter()
        .filter_map(|(owner, depth)| {
            let data = package_manager().read_tag(owner).ok()?;
            let arrays = scan_arrays(&data, endian);
            let channels = arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_PATTERN_OBJECT_CHANNELS)
                .flat_map(|array| {
                    array_records(&data, array, 0x70)
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, record)| {
                            let record_offset = array.data_offset + index * 0x70;
                            Some(PatternObjectChannelDeclaration {
                                hash: read_u32_at(record, 0, endian)?,
                                bytecode: read_array(&data, record_offset + 0x08, 1, endian)?
                                    .to_vec(),
                                constants: read_array(&data, record_offset + 0x18, 0x10, endian)?
                                    .chunks_exact(0x10)
                                    .filter_map(|value| read_vec4_f32(value, 0, endian))
                                    .collect(),
                                interpolation: read_u64_at(record, 0x60, endian)?,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            if channels.is_empty() {
                return None;
            }
            let vectors = arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_VECTOR4)
                .flat_map(|array| array_records(&data, array, 0x10))
                .filter_map(|value| read_vec4_f32(value, 0, endian))
                .collect::<Vec<_>>();
            let bindings = arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
                .flat_map(|array| array_records(&data, array, 0x0c))
                .filter_map(|record| {
                    let scope = read_u32_at(record, 0, endian)?;
                    let parameter = read_u32_at(record, 4, endian)?;
                    let vector_index = read_u32_at(record, 8, endian)?;
                    Some(PatternVectorBindingEvidence {
                        scope,
                        parameter,
                        vector_index,
                        value: vectors.get(vector_index as usize).copied(),
                    })
                })
                .collect();
            Some(PatternObjectChannelEvidence {
                owner,
                depth,
                vectors,
                channels,
                bindings,
            })
        })
        .collect()
}

fn weapon_attachment_poses(tag: TagHash) -> Vec<WeaponModAttachmentPose> {
    let endian = package_manager().version.endian();
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };
    scan_arrays(&data, endian)
        .into_iter()
        .filter(|array| array.class == CLASS_WEAPON_MOD_ATTACHMENTS)
        .flat_map(|array| array_records(&data, array, 0x30))
        .filter_map(|record| {
            let rotation = read_vec4_f32(record, 0, endian)?;
            let translation = read_vec4_f32(record, 0x10, endian)?;
            let bone_index = read_u32_at(record, 0x24, endian)?;
            let family_id = read_u32_at(record, 0x28, endian)?;
            let variant_id = read_u32_at(record, 0x2c, endian)?;
            (family_id != 0).then_some(WeaponModAttachmentPose {
                family_id,
                variant_id,
                bone_index,
                rotation,
                translation: [translation[0], translation[1], translation[2]],
                scale: translation[3],
            })
        })
        .collect()
}

fn pattern_graph_children(cache: &TagCache, tag: TagHash) -> Vec<TagHash> {
    cache
        .hashes
        .get(&tag)
        .into_iter()
        .flat_map(|scan| {
            scan.file_hashes
                .iter()
                .map(|reference| reference.hash)
                .chain(
                    scan.file_hashes64
                        .iter()
                        .filter_map(|reference| tag64_to_hash32(reference.hash)),
                )
        })
        .unique()
        .collect()
}

impl VertexBufferPreview {
    pub fn summary(&self) -> String {
        format!(
            "{} bytes, stride {}, {} vertices",
            self.header.data_size, self.header.stride, self.element_count
        )
    }
}

impl IndexBufferPreview {
    pub fn summary(&self) -> String {
        let width = if self.header.is_32bit { 32 } else { 16 };
        format!(
            "{} bytes, u{}, {} indices",
            self.header.data_size, width, self.index_count
        )
    }
}

pub fn model_info_for_reference(reference: u32) -> Option<ModelTagInfo> {
    match reference {
        0x80806D44 | 0x80808635 => Some(ModelTagInfo {
            role: ModelTagRole::Mesh,
            label: "Mesh",
        }),
        0x80806D30 | 0x80808620 => Some(ModelTagInfo {
            role: ModelTagRole::MeshData,
            label: "Mesh data",
        }),
        0x80808567 => Some(ModelTagInfo {
            role: ModelTagRole::Mesh,
            label: "Terrain mesh",
        }),
        0x8080881C => Some(ModelTagInfo {
            role: ModelTagRole::Geometry,
            label: "Geometry",
        }),
        0x80806F07 => Some(ModelTagInfo {
            role: ModelTagRole::Dynamic,
            label: "Dynamic model",
        }),
        CLASS_DECORATOR => Some(ModelTagInfo {
            role: ModelTagRole::Dynamic,
            label: "Decorator / trees",
        }),
        0x80806EC5 => Some(ModelTagInfo {
            role: ModelTagRole::Dynamic,
            label: "Dynamic mesh",
        }),
        0x8080BADB | 0x8080BAAD => Some(ModelTagInfo {
            role: ModelTagRole::Container,
            label: "Model container",
        }),
        _ => None,
    }
}

pub fn model_label_for_reference(reference: u32) -> Option<&'static str> {
    model_info_for_reference(reference).map(|info| info.label)
}

pub fn is_model_catalog_reference(reference: u32) -> bool {
    // Components remain loadable by the assembler, but never get catalog rows.
    reference != CLASS_PATTERN_COMPONENT
        && model_info_for_reference(reference).is_some_and(ModelTagInfo::is_catalog_entry)
}

/// Fast catalog-only test for whether a model can lead to renderable geometry.
/// This deliberately stays on the prebuilt TagCache graph: it does not read tag
/// payloads, decode materials, build wireframes, or populate the model cache.
/// Finding either a vertex or index-buffer header is sufficient to prove the
/// entry is not literally empty.
pub fn model_has_render_geometry(
    cache: &TagCache,
    root: TagHash,
    memo: &mut rustc_hash::FxHashMap<TagHash, bool>,
) -> bool {
    if let Some(&result) = memo.get(&root) {
        return result;
    }

    let mut seen = rustc_hash::FxHashSet::default();
    let mut parent = rustc_hash::FxHashMap::default();
    let mut frontier = vec![root];
    while let Some(tag) = frontier.pop() {
        if !seen.insert(tag) {
            continue;
        }
        if let Some(&known) = memo.get(&tag) {
            if !known {
                continue;
            }
            let mut current = tag;
            memo.insert(current, true);
            while let Some(&owner) = parent.get(&current) {
                memo.insert(owner, true);
                current = owner;
            }
            return true;
        }
        let Some(scan) = cache.hashes.get(&tag) else {
            continue;
        };
        for child in scan
            .file_hashes
            .iter()
            .map(|reference| reference.hash)
            .chain(
                scan.file_hashes64
                    .iter()
                    .filter_map(|reference| tag64_to_hash32(reference.hash)),
            )
        {
            let Some(entry) = package_manager().get_entry(child) else {
                continue;
            };
            let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
            if matches!(
                tag_type,
                TagType::VertexBuffer { is_header: true }
                    | TagType::IndexBuffer { is_header: true }
            ) {
                let mut current = tag;
                memo.insert(current, true);
                while let Some(&owner) = parent.get(&current) {
                    memo.insert(owner, true);
                    current = owner;
                }
                return true;
            }
            if tag_type.is_tag() && !seen.contains(&child) {
                parent.entry(child).or_insert(tag);
                frontier.push(child);
            }
        }
    }

    for tag in seen {
        memo.insert(tag, false);
    }
    false
}

fn load_vertex_buffer_preview_for_tag(
    tag: TagHash,
    entry: &UEntryHeader,
    tag_data: &[u8],
) -> anyhow::Result<VertexBufferPreview> {
    let endian = package_manager().version.endian();
    let header = VertexBufferHeader::parse(tag_data, endian)?;
    let data_tag = TagHash(entry.reference);
    let data = package_manager()
        .read_tag(data_tag)
        .with_context(|| format!("Failed to read vertex buffer data tag {data_tag}"))?;

    let element_count = if header.stride == 0 {
        0
    } else {
        header.data_size / header.stride as u32
    };

    let mut warnings = vec![];
    if header.stride == 0 {
        warnings.push("Stride is zero".to_string());
    }
    if header.data_size as usize != data.len() {
        warnings.push(format!(
            "Header data_size {} differs from referenced data length {}",
            header.data_size,
            data.len()
        ));
    }
    if header.deadbeef != 0xDEADBEEF {
        warnings.push(format!(
            "Unexpected header marker 0x{:08X}",
            header.deadbeef
        ));
    }

    let candidates = find_position_candidates(&data, header.stride as usize, endian);
    let uv_candidates = find_uv_candidates(&data, header.stride as usize, endian);
    let wireframe = build_vertex_wireframe(
        tag,
        &data,
        header.stride as usize,
        endian,
        &candidates,
        element_count as usize,
    );

    Ok(VertexBufferPreview {
        candidates,
        uv_candidates,
        wireframe,
        header,
        data_tag,
        data_len: data.len(),
        element_count,
        warnings,
    })
}

fn load_index_buffer_preview_for_tag(
    _tag: TagHash,
    entry: &UEntryHeader,
    tag_data: &[u8],
) -> anyhow::Result<IndexBufferPreview> {
    let endian = package_manager().version.endian();
    let header = IndexBufferHeader::parse(tag_data, endian)?;
    let data_tag = TagHash(entry.reference);
    let data = package_manager()
        .read_tag(data_tag)
        .with_context(|| format!("Failed to read index buffer data tag {data_tag}"))?;

    let index_size = if header.is_32bit { 4 } else { 2 };
    let index_count = header.data_size as usize / index_size;
    let mut indices = Vec::with_capacity(index_count.min(64));
    let mut preview_indices = Vec::with_capacity(index_count.min(MAX_PREVIEW_INDICES));
    let mut min_index = None::<u32>;
    let mut max_index = None::<u32>;

    for chunk in data.chunks_exact(index_size) {
        let value = if header.is_32bit {
            read_u32(chunk, endian)
        } else {
            read_u16(chunk, endian) as u32
        };
        min_index = Some(min_index.map(|v| v.min(value)).unwrap_or(value));
        max_index = Some(max_index.map(|v| v.max(value)).unwrap_or(value));
        if indices.len() < 64 {
            indices.push(value);
        }
        if preview_indices.len() < MAX_PREVIEW_INDICES {
            preview_indices.push(value);
        }
    }

    let mut warnings = vec![];
    if header.data_size as usize != data.len() {
        warnings.push(format!(
            "Header data_size {} differs from referenced data length {}",
            header.data_size,
            data.len()
        ));
    }
    if header.deadbeef != 0xDEADBEEF {
        warnings.push(format!(
            "Unexpected header marker 0x{:08X}",
            header.deadbeef
        ));
    }
    if data.len() % index_size != 0 {
        warnings.push(format!(
            "Referenced data length is not divisible by {}",
            index_size
        ));
    }
    if index_count > MAX_PREVIEW_INDICES {
        warnings.push(format!(
            "Preview indices truncated to {} of {}",
            MAX_PREVIEW_INDICES, index_count
        ));
    }

    Ok(IndexBufferPreview {
        header,
        data_tag,
        data_len: data.len(),
        index_count,
        min_index,
        max_index,
        first_indices: indices,
        indices: preview_indices,
        warnings,
    })
}

fn load_model_preview(
    cache: Arc<TagCache>,
    tag: TagHash,
    entry: &UEntryHeader,
    label: &'static str,
) -> ModelPreview {
    if entry.reference == CLASS_DECORATOR
        && let Some(preview) = load_decorator_preview(cache.clone(), tag, entry)
    {
        return preview;
    }
    let model_tags = selected_model_geometry_tags(&cache, tag, entry.reference);
    load_model_preview_from_tags(cache, tag, entry, label, model_tags, &[])
}

fn load_decorator_preview(
    cache: Arc<TagCache>,
    tag: TagHash,
    entry: &UEntryHeader,
) -> Option<ModelPreview> {
    let endian = package_manager().version.endian();
    let decorator = package_manager().read_tag(tag).ok()?;
    let entity = read_tag_array(&decorator, 0x08, endian)
        .into_iter()
        .filter_map(|wrapper| package_manager().read_tag(wrapper).ok())
        .filter_map(|wrapper| read_u32_at(&wrapper, 0x08, endian).map(TagHash))
        .find(|entity| package_manager().get_entry(*entity).is_some())?;
    let entity_entry = package_manager().get_entry(entity)?;
    let mut model = load_model_preview(cache, entity, &entity_entry, "Decorator model");
    let base = model.wireframe.take()?;

    let placement_resource = TagHash(read_u32_at(&decorator, 0x48, endian)?);
    let placement_resource = package_manager().read_tag(placement_resource).ok()?;
    let constants_tag = TagHash(read_u32_at(&placement_resource, 0x14, endian)?);
    let instance_data_tag = TagHash(read_u32_at(&placement_resource, 0x1C, endian)?);
    let constants = package_manager().read_tag(constants_tag).ok()?;
    let instance_data = package_manager().read_tag(instance_data_tag).ok()?;
    let scale = read_vec4_f32(&constants, 0x00, endian)?;
    let offset = read_vec4_f32(&constants, 0x10, endian)?;
    let elements = read_array(&instance_data, 0x08, 0x10, endian)?;
    let placements = elements
        .chunks_exact(0x10)
        .take(512)
        .map(|element| {
            let position = [
                read_u16(&element[0..2], endian) as f32 / 65535.0 * scale[0] + offset[0],
                read_u16(&element[2..4], endian) as f32 / 65535.0 * scale[1] + offset[1],
                read_u16(&element[4..6], endian) as f32 / 65535.0 * scale[2] + offset[2],
            ];
            let mut rotation = [
                element[8] as f32 / 127.5 - 1.0,
                element[9] as f32 / 127.5 - 1.0,
                element[10] as f32 / 127.5 - 1.0,
                element[11] as f32 / 127.5 - 1.0,
            ];
            let length = rotation
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if length > 0.0001 {
                rotation.iter_mut().for_each(|value| *value /= length);
            } else {
                rotation = [0.0, 0.0, 0.0, 1.0];
            }
            DecoratorPlacement { position, rotation }
        })
        .collect_vec();
    model.wireframe = instantiate_decorator_wireframe(&base, &placements);
    model.label = "Decorator / trees";
    model.class_name = get_class_by_id(entry.reference).map(|class| class.name.to_string());
    model.geometry_parts.insert(0, tag);
    Some(model)
}

#[derive(Clone, Copy)]
struct DecoratorPlacement {
    position: [f32; 3],
    rotation: [f32; 4],
}

fn rotate_quaternion(vector: [f32; 3], quaternion: [f32; 4]) -> [f32; 3] {
    let [qx, qy, qz, qw] = quaternion;
    let uv = [
        qy * vector[2] - qz * vector[1],
        qz * vector[0] - qx * vector[2],
        qx * vector[1] - qy * vector[0],
    ];
    let uuv = [
        qy * uv[2] - qz * uv[1],
        qz * uv[0] - qx * uv[2],
        qx * uv[1] - qy * uv[0],
    ];
    [
        vector[0] + 2.0 * (qw * uv[0] + uuv[0]),
        vector[1] + 2.0 * (qw * uv[1] + uuv[1]),
        vector[2] + 2.0 * (qw * uv[2] + uuv[2]),
    ]
}

fn multiply_quaternions(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    let [lx, ly, lz, lw] = left;
    let [rx, ry, rz, rw] = right;
    let mut product = [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ];
    let length = product
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length > f32::EPSILON {
        product.iter_mut().for_each(|value| *value /= length);
    }
    product
}

fn instantiate_decorator_wireframe(
    base: &WireframePreview,
    placements: &[DecoratorPlacement],
) -> Option<WireframePreview> {
    let max_instances = (MAX_PREVIEW_VERTICES / base.vertices.len().max(1))
        .min(MAX_PREVIEW_INDICES / base.indices.len().max(1))
        .min(placements.len());
    let mut output = base.clone();
    output.source = format!("{} decorator instances", max_instances);
    output.vertices.clear();
    output.indices.clear();
    output.material_ranges.clear();
    output.normals.as_mut().map(Vec::clear);
    output.procedural_positions.as_mut().map(Vec::clear);
    output.procedural_normals.as_mut().map(Vec::clear);
    output.tangents.as_mut().map(Vec::clear);
    output.uvs.as_mut().map(Vec::clear);
    for placement in placements.iter().take(max_instances) {
        let vertex_base = output.vertices.len() as u32;
        let index_base = output.indices.len();
        output.vertices.extend(base.vertices.iter().map(|position| {
            let rotated = rotate_quaternion(*position, placement.rotation);
            [
                rotated[0] + placement.position[0],
                rotated[1] + placement.position[1],
                rotated[2] + placement.position[2],
            ]
        }));
        if let (Some(target), Some(source)) =
            (&mut output.procedural_positions, &base.procedural_positions)
        {
            target.extend(source.iter().copied());
        }
        if let (Some(target), Some(source)) = (&mut output.normals, &base.normals) {
            target.extend(
                source
                    .iter()
                    .map(|normal| rotate_quaternion(*normal, placement.rotation)),
            );
        }
        if let (Some(target), Some(source)) =
            (&mut output.procedural_normals, &base.procedural_normals)
        {
            target.extend(source.iter().copied());
        }
        if let (Some(target), Some(source)) = (&mut output.tangents, &base.tangents) {
            target.extend(source.iter().map(|tangent| {
                let rotated =
                    rotate_quaternion([tangent[0], tangent[1], tangent[2]], placement.rotation);
                [rotated[0], rotated[1], rotated[2], tangent[3]]
            }));
        }
        if let (Some(target), Some(source)) = (&mut output.uvs, &base.uvs) {
            target.extend(source.iter().copied());
        }
        output
            .indices
            .extend(base.indices.iter().map(|index| index + vertex_base));
        output
            .material_ranges
            .extend(base.material_ranges.iter().cloned().map(|mut range| {
                range.index_start += index_base;
                range
            }));
    }
    output.vertex_count_total = output.vertices.len();
    output.index_count_total = output.indices.len();
    let (min, max) = bounds(&output.vertices)?;
    output.min = min;
    output.max = max;
    Some(output)
}

fn read_vec4_f32(data: &[u8], offset: usize, endian: Endian) -> Option<[f32; 4]> {
    Some(std::array::from_fn(|index| {
        f32::from_bits(read_u32_at(data, offset + index * 4, endian).unwrap_or_default())
    }))
}

fn selected_model_geometry_tags(cache: &TagCache, tag: TagHash, reference: u32) -> Vec<TagHash> {
    match reference {
        CLASS_PATTERN => {
            let nearest = pattern_nearest_geometry_tags(cache, tag);
            if nearest.is_empty() {
                pattern_geometry_tags(cache, tag)
            } else {
                nearest
            }
        }
        CLASS_GEOMETRY_RESOURCE | CLASS_PATTERN_COMPONENT => {
            related_pattern_geometry_tags(cache, tag)
        }
        _ => vec![tag],
    }
}

fn load_model_preview_from_tags(
    cache: Arc<TagCache>,
    tag: TagHash,
    entry: &UEntryHeader,
    label: &'static str,
    model_tags: Vec<TagHash>,
    attachments: &[ResolvedWeaponModAttachment],
) -> ModelPreview {
    cache::model(
        &cache,
        tag,
        entry.reference,
        label,
        &model_tags,
        attachments,
        || {
            decode_model_preview_from_tags(
                cache.clone(),
                tag,
                entry,
                label,
                &model_tags,
                attachments,
            )
        },
    )
}

fn decode_model_preview_from_tags(
    cache: Arc<TagCache>,
    tag: TagHash,
    entry: &UEntryHeader,
    label: &'static str,
    model_tags: &[TagHash],
    attachments: &[ResolvedWeaponModAttachment],
) -> ModelPreview {
    let class_name = get_class_by_id(entry.reference).map(|c| c.name.to_string());
    let model_entries = model_tags
        .iter()
        .filter_map(|model_tag| {
            package_manager()
                .get_entry(*model_tag)
                .map(|entry| (*model_tag, entry))
        })
        .collect_vec();
    let vertex_buffers = model_tags
        .iter()
        .flat_map(|model_tag| find_related_tags(&cache, *model_tag, TagSearchKind::VertexBuffer, 8))
        .unique_by(|(tag, _entry)| *tag)
        .collect_vec();
    let index_buffers = model_tags
        .iter()
        .flat_map(|model_tag| find_related_tags(&cache, *model_tag, TagSearchKind::IndexBuffer, 8))
        .unique_by(|(tag, _entry)| *tag)
        .collect_vec();
    let techniques = model_entries
        .iter()
        .flat_map(|(model_tag, model_entry)| {
            find_model_technique_entries(&cache, *model_tag, model_entry)
        })
        .unique_by(|(tag, _entry)| *tag)
        .collect_vec();
    let textures = model_tags
        .iter()
        .flat_map(|model_tag| find_model_textures(&cache, *model_tag, &techniques))
        .unique_by(|(tag, _entry)| *tag)
        .take(256)
        .collect_vec();
    let shaders = model_tags
        .iter()
        .flat_map(|model_tag| find_related_tags(&cache, *model_tag, TagSearchKind::Shader, 8))
        .unique_by(|(tag, _entry)| *tag)
        .collect_vec();
    let mut parsed = model_entries
        .iter()
        .filter_map(|(model_tag, model_entry)| {
            parse_model_wireframe(*model_tag, model_entry)
                .map(|(source, wireframe)| (*model_tag, source, wireframe))
        })
        .collect_vec();
    apply_inventory_rigid_chain_visibility(&cache, tag, &mut parsed);
    apply_weapon_mod_attachment_poses(&mut parsed, attachments);
    let mesh_source = parsed
        .iter()
        .find(|(model_tag, _source, _wireframe)| *model_tag == tag)
        .or_else(|| parsed.first())
        .map(|(_model_tag, source, _wireframe)| source.clone());
    let attached_geometry = attachments
        .iter()
        .map(|attachment| {
            (
                attachment.geometry,
                (attachment.pattern, attachment.rarity, attachment.unique_id),
            )
        })
        .collect::<rustc_hash::FxHashMap<_, _>>();
    let gear_dye_palette = (!attached_geometry.is_empty())
        .then(|| weapon_skin_gear_dye_palette(&cache, tag))
        .flatten();
    let gear_worn_dye_palette = (!attached_geometry.is_empty())
        .then(|| pattern_gear_dye_color_contribution(&cache, tag, GEAR_WORN_DYE_COLOR_PARAMETERS))
        .flatten();
    for (model_tag, _source, wireframe) in &mut parsed {
        assign_wireframe_material_textures(wireframe, &cache, &textures);
        if let Some((_pattern, rarity, unique_id)) = attached_geometry.get(model_tag).copied() {
            for range in &mut wireframe.material_ranges {
                if let Some(wear) = &mut range.textures.mod_wear {
                    wear.rarity = rarity;
                    wear.unique_id = unique_id.clamp(0.0, 1.0);
                }
            }
        }
        if attached_geometry.contains_key(model_tag)
            && let Some(palette) = gear_dye_palette
        {
            let detail_palette =
                attached_geometry
                    .get(model_tag)
                    .and_then(|(pattern, _rarity, _unique_id)| {
                        pattern_gear_dye_color_contribution(
                            &cache,
                            *pattern,
                            GEAR_DYE_DETAIL_COLOR_PARAMETERS,
                        )
                    });
            // The compiled GearDye outputs are the exact sum of three
            // independently-authored channels. Base Dye is inherited from the
            // selected skin, Worn Dye may also be inherited, and Dye Detail is
            // local to the attached mod Pattern.
            let attachment_palette = palette;
            for range in &mut wireframe.material_ranges {
                let Some(default) = range.technique.and_then(technique_default_gear_dye_color)
                else {
                    continue;
                };
                if let Some(dye) = range
                    .gear_dye_change_color_index
                    .and_then(|index| attachment_palette.get(index as usize).copied())
                {
                    range.textures.gear_dye = Some(dye);
                    range.textures.gear_dye_default = Some(default);
                    range.textures.gear_dye_palette = Some(attachment_palette);
                    range.textures.gear_worn_dye_palette = gear_worn_dye_palette;
                    range.textures.gear_dye_detail_palette = detail_palette;
                }
            }
        }
    }
    let wireframe = merge_model_wireframes(parsed)
        .or_else(|| build_model_wireframe(&vertex_buffers, &index_buffers));

    ModelPreview {
        label,
        class_name,
        mesh_source,
        vertex_buffers,
        index_buffers,
        techniques,
        textures,
        shaders,
        geometry_parts: model_tags.to_vec(),
        wireframe,
    }
}

fn technique_default_gear_dye_color(technique: TagHash) -> Option<[f32; 4]> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview
        .stages
        .into_iter()
        .find(|stage| stage.stage == "PS")?;
    // GearDye shaders map material IDs 1..6 to six consecutive cbuffer
    // outputs. ID 0 reads the immediately preceding inline output. Output
    // bases vary between shader families (8 in older gear shaders, 11 in the
    // D54 optic), so derive the ABI layout from the authored channel hashes.
    let material_id_parameters = [
        GEAR_DYE_COLOR_PARAMETERS[0],
        GEAR_DYE_COLOR_PARAMETERS[2],
        GEAR_DYE_COLOR_PARAMETERS[3],
        GEAR_DYE_COLOR_PARAMETERS[1],
        GEAR_DYE_COLOR_PARAMETERS[4],
        GEAR_DYE_COLOR_PARAMETERS[5],
    ];
    let first_output = pixel
        .bytecode
        .expressions
        .iter()
        .filter(|expression| {
            expression
                .expression
                .contains(&format!("0x{:08X}", material_id_parameters[0]))
        })
        .filter_map(|expression| {
            expression
                .target
                .strip_prefix("output[")?
                .strip_suffix(']')?
                .parse::<usize>()
                .ok()
        })
        .find(|first_output| {
            material_id_parameters
                .into_iter()
                .enumerate()
                .all(|(material_id, parameter)| {
                    let target = format!("output[{}]", first_output + material_id);
                    pixel.bytecode.expressions.iter().any(|expression| {
                        expression.target == target
                            && expression
                                .expression
                                .contains(&format!("0x{parameter:08X}"))
                    })
                })
        })?;
    let default_output = first_output.checked_sub(1)?;
    let color = pixel.inline_constants.get(default_output).copied()?;
    valid_dye_color(color).then_some(color)
}

fn bindings_use_character_gear_surface(
    bindings: &[TechniqueTextureBinding],
    normal_slot: Option<u32>,
) -> bool {
    bindings.iter().any(|binding| binding.slot >= 10)
        && [0, 1, 2, 3]
            .into_iter()
            .all(|slot| bindings.iter().any(|binding| binding.slot == slot))
        && bindings
            .iter()
            .any(|binding| binding.slot == 0 && texture_is_srgb(binding.tag))
        // Caller already resolved the compiled shader's exact normal ABI.
        // Re-running generic inference here rejects legitimate shared runner
        // normals and silently drops the whole character material.
        && normal_slot.is_some()
}

fn bindings_use_compact_character_surface(
    bindings: &[TechniqueTextureBinding],
    normal_slot: Option<u32>,
) -> bool {
    let max_slot = bindings
        .iter()
        .map(|binding| binding.slot)
        .max()
        .unwrap_or(0);
    (5..10).contains(&max_slot)
        && bindings
            .iter()
            .any(|binding| binding.slot == 0 && texture_is_srgb(binding.tag))
        && [2, 3].into_iter().all(|slot| {
            bindings.iter().any(|binding| {
                binding.slot == slot
                    && !texture_is_srgb(binding.tag)
                    && material_control_texture_candidate(binding.tag)
            })
        })
        && normal_slot.is_some()
}

fn runner_alpha_mask_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<AlphaMaskMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;

    // Match compiled shader ABIs, never skin hashes.
    let (remap, threshold) = match pixel.shader {
        // t0 supplies RGB; t1.r supplies coverage; c1.x is the cutoff.
        Some(TagHash(0x80A9BE22)) => ([0.0, 1.0], pixel.inline_constants.get(1)?[0]),
        // Two-mask runner surface: coverage = c89.y * t1.r + c89.x, then
        // discard below c90.x. t2 is a separate material mask.
        Some(TagHash(0x80A9A9D3)) => {
            let c89 = *pixel.inline_constants.get(89)?;
            ([c89[0], c89[1]], pixel.inline_constants.get(90)?[0])
        }
        _ => return None,
    };
    let texture = bindings
        .iter()
        .find(|binding| binding.slot == 1 && texture_preview_format(binding.tag).contains("Bc4"))?
        .tag;
    (remap.into_iter().all(f32::is_finite)
        && threshold.is_finite()
        && (0.0..=1.0).contains(&threshold))
    .then_some(AlphaMaskMaterial {
        texture,
        remap,
        threshold,
    })
}

fn runner_solid_surface_material(technique: TagHash) -> Option<[f32; 2]> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let shader = preview
        .stages
        .iter()
        .find(|stage| stage.stage == "PS")?
        .shader?;
    match shader {
        // 80A9B032 writes RT1.a = 0.67 and RT2.r = 0.0. Its PS t1 is a
        // shared 1D lighting lookup, not an ORM texture.
        TagHash(0x80A9B032) => Some([0.67, 0.0]),
        // Skin/subsurface PS writes literal RT1.a = 0.67 and RT2.r = 0.
        TagHash(0x80A9B860) => Some([0.67, 0.0]),
        _ => None,
    }
}

fn runner_layered_surface_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<RunnerLayeredSurfaceMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;

    let tag_at = |slot| {
        bindings
            .iter()
            .find(|binding| binding.slot == slot)
            .map(|binding| binding.tag)
    };
    let (
        mode,
        surface,
        detail_normal_a,
        detail_normal_b,
        detail_normal_c,
        detail_normal_d,
        material_response,
        constants,
    ) = match pixel.shader? {
        // Runner skin/subsurface ABI. t0 is authored pore/feature field, t1
        // is a shared cellular response LUT (never direct albedo), t2 is
        // scalar AO, and t3 is tangent normal. c121/c122/c123 provide primary,
        // variation, and recessed-feature colours; c130 is subsurface tint.
        TagHash(0x80A9B860) if pixel.inline_constants.len() >= 233 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[121];
            rows[1] = pixel.inline_constants[122];
            rows[2] = pixel.inline_constants[123];
            rows[3] = pixel.inline_constants[130];
            rows[4] = pixel.inline_constants[120];
            rows[5] = pixel.inline_constants[127];
            rows[6] = pixel.inline_constants[159];
            rows[7] = pixel.inline_constants[231];
            rows[8] = pixel.inline_constants[232];
            rows[9] = pixel.inline_constants[124];
            (
                41,
                tag_at(0)?,
                tag_at(3)?,
                tag_at(3)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Full9: t4/t5 with c29..c38, selected by t1.b/t1.r, base t6.
        TagHash(0x80A9B07B) if pixel.inline_constants.len() >= 39 => {
            (1, tag_at(2)?, tag_at(4)?, tag_at(5)?, None, None, None, {
                let mut rows = [[0.0; 4]; 24];
                rows[0] = pixel.inline_constants[13];
                rows[1..=10].copy_from_slice(&pixel.inline_constants[29..=38]);
                rows
            })
        }
        // Switched dual layer: t1.r < c83 chooses t2; otherwise t3. c77..c82
        // are the two affine UV/remap triples and t4 is the base normal.
        TagHash(0x80A9B855) if pixel.inline_constants.len() >= 87 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[83];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[77..=79]);
            rows[4..=6].copy_from_slice(&pixel.inline_constants[80..=82]);
            rows[7] = pixel.inline_constants[86];
            (
                2,
                tag_at(1)?,
                tag_at(2)?,
                tag_at(3)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Full10: t4 is always applied. t5/t6/t7 are gated by t2 G/B/R;
        // t6 owns two authored transforms selected by the literal c10.x.
        // The base normal is t8 and remains resolved by the normal-slot ABI.
        TagHash(0x80A9C244) if pixel.inline_constants.len() >= 42 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[10];
            rows[1..=21].copy_from_slice(&pixel.inline_constants[21..=41]);
            (
                3,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(7)?),
                None,
                rows,
            )
        }
        // Procedural full9: t4/t5 authored normals are gated by t2 G/B.
        // t6 is the base normal; t9 is an independent object-space field.
        TagHash(0x80A9AFB4) if pixel.inline_constants.len() >= 48 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=5].copy_from_slice(&pixel.inline_constants[38..=42]);
            rows[6..=10].copy_from_slice(&pixel.inline_constants[43..=47]);
            (
                4,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                None,
                None,
                None,
                rows,
            )
        }
        // Sibling switched surfaces: t1.r < c15 gates t2; c12/c13/c14 are
        // its affine UV and normal remap. t3 is the base normal.
        TagHash(0x80A9B857) | TagHash(0x80A9B85A) if pixel.inline_constants.len() >= 16 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[15];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[12..=14]);
            (
                5,
                tag_at(1)?,
                tag_at(2)?,
                tag_at(2)?,
                None,
                None,
                None,
                rows,
            )
        }
        // Local full9 stack: t3 is G-gated, t4 is B-selected, and t5 carries
        // the two R-selected transforms. t6 is the base normal.
        TagHash(0x80A9B93E) | TagHash(0x80A9BBDD) | TagHash(0x80A9DC9A)
            if pixel.inline_constants.len() >= 41 =>
        {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=17].copy_from_slice(&pixel.inline_constants[24..=40]);
            (
                6,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(5)?),
                None,
                rows,
            )
        }
        // Same local full9 normal ABI with the normal block at c24..c40.
        TagHash(0x80A9DEEC) if pixel.inline_constants.len() >= 41 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=17].copy_from_slice(&pixel.inline_constants[24..=40]);
            (
                23,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(5)?),
                Some(tag_at(2)?),
                rows,
            )
        }
        // E4DB embeds the same local full9+response ABI before its larger
        // procedural material branch. Normal rows are shifted to c16..c32.
        TagHash(0x80A9E4DB) if pixel.inline_constants.len() >= 135 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=17].copy_from_slice(&pixel.inline_constants[16..=32]);
            // UV transforms shared by t9/t11, six authored t11 exponents,
            // then the final two condition remaps packed into one row.
            rows[18..=20].copy_from_slice(&pixel.inline_constants[107..=109]);
            rows[21] = [
                pixel.inline_constants[126][0],
                pixel.inline_constants[127][0],
                pixel.inline_constants[128][0],
                pixel.inline_constants[129][0],
            ];
            rows[22] = [
                pixel.inline_constants[130][0],
                pixel.inline_constants[131][0],
                pixel.inline_constants[84][0],
                pixel.inline_constants[114][0],
            ];
            rows[23] = [
                pixel.inline_constants[133][0],
                pixel.inline_constants[133][1],
                pixel.inline_constants[134][0],
                pixel.inline_constants[134][1],
            ];
            (
                23,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(5)?),
                Some(tag_at(2)?),
                rows,
            )
        }
        // Same generated full9 ABI with six fewer preceding material rows.
        TagHash(0x80A9CBAE) if pixel.inline_constants.len() >= 35 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=17].copy_from_slice(&pixel.inline_constants[18..=34]);
            (
                6,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(5)?),
                None,
                rows,
            )
        }
        // Expanded local full9 stack: t1 A/G independently gate t3/t4,
        // t1 B selects t3, and t1 R selects between two t5 transforms.
        TagHash(0x80A9D6FD) if pixel.inline_constants.len() >= 42 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=18].copy_from_slice(&pixel.inline_constants[24..=41]);
            (
                7,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(5)?),
                None,
                rows,
            )
        }
        // A-expanded sibling of the local full9 ABI: c24..c41 contains the
        // A/G gates, B range, and two authored t5 transforms.
        TagHash(0x80A9E0FD) if pixel.inline_constants.len() >= 42 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=18].copy_from_slice(&pixel.inline_constants[24..=41]);
            (
                24,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(5)?),
                Some(tag_at(2)?),
                rows,
            )
        }
        // Procedural full8: t4 is the base tangent normal. t3 is transformed
        // by c17/c18, remapped by c19, then applied only while t1.r lies
        // between c20/c21. t5/t6 are later object-space procedural fields,
        // not substitutes for the base normal.
        TagHash(0x80A9AFB6) if pixel.inline_constants.len() >= 22 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=5].copy_from_slice(&pixel.inline_constants[17..=21]);
            (
                8,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(3)?,
                None,
                None,
                None,
                rows,
            )
        }
        // Same generated procedural ABI with thirteen preceding material
        // rows: t4 base, transformed t3, t1.r band c33/c34.
        TagHash(0x80A9BBBE) | TagHash(0x80A9BBBF) if pixel.inline_constants.len() >= 35 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=5].copy_from_slice(&pixel.inline_constants[30..=34]);
            (
                8,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(3)?,
                None,
                None,
                None,
                rows,
            )
        }
        // Full10 three-detail stack: t7 is base normal. t4 is selected by
        // round(t2.g-c24), t5 by the c28/c29 t2.b band, and t6 by the
        // c33/c34 t2.r band.
        TagHash(0x80A9C3F6) if pixel.inline_constants.len() >= 35 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=14].copy_from_slice(&pixel.inline_constants[21..=34]);
            (
                9,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                None,
                rows,
            )
        }
        // Generated sibling of the Full10 three-detail ABI with three more
        // material rows before the normal block: t7 base; t4/t5/t6 selected
        // by t2 G/B/R using c24..c37.
        TagHash(0x80A9B610) if pixel.inline_constants.len() >= 38 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=14].copy_from_slice(&pixel.inline_constants[24..=37]);
            (
                9,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                None,
                rows,
            )
        }
        // Full9 sibling of the same generated three-detail stack. Its
        // selector/detail/base resources are shifted down one register:
        // t1 selects t3/t4/t5 and t6 is the base tangent normal. LLVM shows
        // the identical c24..c37 G/B/R gate and affine-transform block.
        TagHash(0x80A9E76C) if pixel.inline_constants.len() >= 38 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=14].copy_from_slice(&pixel.inline_constants[24..=37]);
            (
                25,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Full11 surface: t4 is the packed selector. Its G channel selects
        // t5/t6, B selects either authored transform of t7, and R gates t8.
        // t9 is the base tangent normal. The two generated siblings differ
        // only by the number of material rows preceding this shared ABI.
        TagHash(0x80A9A9AD) if pixel.inline_constants.len() >= 105 => {
            let mut rows = [[0.0; 4]; 24];
            rows.copy_from_slice(&pixel.inline_constants[81..=104]);
            (
                10,
                tag_at(4)?,
                tag_at(5)?,
                tag_at(6)?,
                Some(tag_at(7)?),
                Some(tag_at(8)?),
                None,
                rows,
            )
        }
        TagHash(0x80A9A9B1) if pixel.inline_constants.len() >= 72 => {
            let mut rows = [[0.0; 4]; 24];
            rows.copy_from_slice(&pixel.inline_constants[48..=71]);
            (
                10,
                tag_at(4)?,
                tag_at(5)?,
                tag_at(6)?,
                Some(tag_at(7)?),
                Some(tag_at(8)?),
                None,
                rows,
            )
        }
        // Earlier full11 runner-skin permutation. DXIL uses the identical
        // 24-row normal ABI at c24..c47, shifted resources: t1 selector;
        // t3/t4/t5/t6 details (t5 has two transforms); t7 base tangent
        // normal. t2 independently supplies response and green-channel AO.
        TagHash(0x80A9A8CF) if pixel.inline_constants.len() >= 48 => {
            let mut rows = [[0.0; 4]; 24];
            rows.copy_from_slice(&pixel.inline_constants[24..=47]);
            (
                10,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(6)?),
                Some(tag_at(2)?),
                rows,
            )
        }
        // Generated full12 runner surface. t4 is the packed A/G/B/R
        // selector; t6/t7/t8 are its authored normal layers and t9 is the
        // base tangent normal. The normal program occupies c53..c71.
        TagHash(0x80A9AD5D) if pixel.inline_constants.len() >= 72 => {
            let mut rows = [[0.0; 4]; 24];
            rows[..=18].copy_from_slice(&pixel.inline_constants[53..=71]);
            (
                42,
                tag_at(4)?,
                tag_at(6)?,
                tag_at(7)?,
                Some(tag_at(8)?),
                None,
                Some(tag_at(5)?),
                rows,
            )
        }
        // Compact sibling of AD5D. t3 selects authored t5/t6 over the t7
        // base tangent normal. c46..c61 is the complete normal/response ABI.
        TagHash(0x80A9AD68) if pixel.inline_constants.len() >= 62 => {
            let mut rows = [[0.0; 4]; 24];
            rows[..=15].copy_from_slice(&pixel.inline_constants[46..=61]);
            (
                43,
                tag_at(3)?,
                tag_at(5)?,
                tag_at(6)?,
                None,
                None,
                Some(tag_at(4)?),
                rows,
            )
        }
        // B71C compact A/G/B/R surface. t1 is selector; all four selector
        // tests gate transformed t3 over base t4. Local t2 is response/AO.
        TagHash(0x80A9B71C) if pixel.inline_constants.len() >= 37 => {
            let mut rows = [[0.0; 4]; 24];
            rows[..=8].copy_from_slice(&pixel.inline_constants[24..=32]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[35..=36]);
            (
                44,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(3)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Full10 A/G/B/R stack: t1 is the selector, t3 is A/G gated,
        // t4 is B selected, t5 is R selected, and t6 is the base normal.
        TagHash(0x80A9B065) if pixel.inline_constants.len() >= 43 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=19].copy_from_slice(&pixel.inline_constants[24..=42]);
            (
                11,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                None,
                None,
                rows,
            )
        }
        // Response/AO-bearing sibling of B065. LLVM has the identical
        // c24..c42 A/G/B/R normal program; local t2 additionally supplies
        // material response and green-channel AO.
        TagHash(0x80A9B06A) if pixel.inline_constants.len() >= 43 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=19].copy_from_slice(&pixel.inline_constants[24..=42]);
            (
                11,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        TagHash(0x80A9DE77) if pixel.inline_constants.len() >= 35 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=19].copy_from_slice(&pixel.inline_constants[16..=34]);
            (
                11,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                None,
                None,
                rows,
            )
        }
        // Local full11: t2 selector, t4 A/G controlled, t5 B selected,
        // t6 R gated, t7 base. c38..c56 are the complete normal ABI.
        TagHash(0x80A9DB9A) if pixel.inline_constants.len() >= 57 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=19].copy_from_slice(&pixel.inline_constants[38..=56]);
            (
                12,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                None,
                rows,
            )
        }
        TagHash(0x80A9AE19) if pixel.inline_constants.len() >= 39 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=19].copy_from_slice(&pixel.inline_constants[20..=38]);
            (
                12,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                None,
                rows,
            )
        }
        TagHash(0x80A9C27C) if pixel.inline_constants.len() >= 38 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1] = pixel.inline_constants[27];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[24..=26]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[28..=30]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[31..=32]);
            rows[10..=12].copy_from_slice(&pixel.inline_constants[33..=35]);
            rows[13..=14].copy_from_slice(&pixel.inline_constants[36..=37]);
            (
                13,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                None,
                None,
                rows,
            )
        }
        TagHash(0x80A9DAC9) if pixel.inline_constants.len() >= 40 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1] = pixel.inline_constants[28];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[25..=27]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[29..=31]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[33..=34]);
            rows[10..=12].copy_from_slice(&pixel.inline_constants[35..=37]);
            rows[13..=14].copy_from_slice(&pixel.inline_constants[38..=39]);
            (
                14,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                None,
                rows,
            )
        }
        TagHash(0x80A9AFBF) if pixel.inline_constants.len() >= 49 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1] = pixel.inline_constants[40];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[37..=39]);
            rows[5] = pixel.inline_constants[41];
            rows[6..=7].copy_from_slice(&pixel.inline_constants[42..=43]);
            rows[8..=10].copy_from_slice(&pixel.inline_constants[44..=46]);
            rows[11..=12].copy_from_slice(&pixel.inline_constants[47..=48]);
            (
                15,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                None,
                None,
                None,
                rows,
            )
        }
        TagHash(0x80AA0261) if pixel.inline_constants.len() >= 31 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1] = pixel.inline_constants[25];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[22..=24]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[26..=28]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[29..=30]);
            (
                16,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        TagHash(0x80AA0263) if pixel.inline_constants.len() >= 33 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1] = pixel.inline_constants[27];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[24..=26]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[28..=30]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[31..=32]);
            (
                16,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        TagHash(0x80A9B86A) if pixel.inline_constants.len() >= 31 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1] = pixel.inline_constants[10];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[11..=13]);
            rows[5] = pixel.inline_constants[14];
            rows[6..=8].copy_from_slice(&pixel.inline_constants[15..=17]);
            rows[9..=11].copy_from_slice(&pixel.inline_constants[18..=20]);
            rows[12..=13].copy_from_slice(&pixel.inline_constants[21..=22]);
            rows[14..=16].copy_from_slice(&pixel.inline_constants[23..=25]);
            rows[17..=19].copy_from_slice(&pixel.inline_constants[26..=28]);
            rows[20..=21].copy_from_slice(&pixel.inline_constants[29..=30]);
            (
                17,
                tag_at(6)?,
                tag_at(2)?,
                tag_at(3)?,
                Some(tag_at(4)?),
                Some(tag_at(5)?),
                None,
                rows,
            )
        }
        TagHash(0x80A9D952) if pixel.inline_constants.len() >= 65 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[47..=49]);
            rows[4..=5].copy_from_slice(&pixel.inline_constants[50..=51]);
            rows[6..=8].copy_from_slice(&pixel.inline_constants[52..=54]);
            rows[9..=11].copy_from_slice(&pixel.inline_constants[55..=57]);
            rows[12..=13].copy_from_slice(&pixel.inline_constants[58..=59]);
            rows[14..=16].copy_from_slice(&pixel.inline_constants[60..=62]);
            rows[17..=18].copy_from_slice(&pixel.inline_constants[63..=64]);
            (
                18,
                tag_at(3)?,
                tag_at(5)?,
                tag_at(6)?,
                None,
                None,
                Some(tag_at(4)?),
                rows,
            )
        }
        // Procedural runner panel family. DXIL uses t2 as the packed selector,
        // t4 as the scalar pattern atlas, t5 as its RG procedural field, t6 as
        // the local panel mask, t8 as the authored detail normal, and t9 as the
        // base tangent normal. The generated AFB8/AFBA siblings share this ABI.
        TagHash(0x80A9AFB8) | TagHash(0x80A9AFBA) if pixel.inline_constants.len() >= 103 => {
            let mut rows = [[0.0; 4]; 24];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[94..=96]);
            rows[4..=6].copy_from_slice(&pixel.inline_constants[60..=62]);
            rows[7..=9].copy_from_slice(&pixel.inline_constants[78..=80]);
            rows[10..=13].copy_from_slice(&pixel.inline_constants[90..=93]);
            rows[14] = pixel.inline_constants[97];
            rows[15] = pixel.inline_constants[98];
            rows[16] = pixel.inline_constants[101];
            rows[17] = pixel.inline_constants[102];
            (
                19,
                tag_at(2)?,
                tag_at(8)?,
                tag_at(4)?,
                Some(tag_at(6)?),
                None,
                Some(tag_at(5)?),
                rows,
            )
        }
        // Full10 generated siblings: t2.g gates t4, t2.b gates t5, and t2.r
        // selects either authored transform of t6 before the t7 base normal.
        TagHash(0x80A9BD17) if pixel.inline_constants.len() >= 39 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[7];
            rows[1] = pixel.inline_constants[21];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[18..=20]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[22..=24]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[25..=26]);
            rows[10..=12].copy_from_slice(&pixel.inline_constants[27..=29]);
            rows[13..=15].copy_from_slice(&pixel.inline_constants[30..=32]);
            rows[16..=17].copy_from_slice(&pixel.inline_constants[33..=34]);
            (
                20,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(6)?),
                None,
                rows,
            )
        }
        TagHash(0x80A9DA29) if pixel.inline_constants.len() >= 40 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[7];
            rows[1] = pixel.inline_constants[22];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[19..=21]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[23..=25]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[26..=27]);
            rows[10..=12].copy_from_slice(&pixel.inline_constants[28..=30]);
            rows[13..=15].copy_from_slice(&pixel.inline_constants[31..=33]);
            rows[16..=17].copy_from_slice(&pixel.inline_constants[34..=35]);
            (
                20,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(6)?),
                None,
                rows,
            )
        }
        // Expanded sibling also gates t4 from t2.a. Its normal block starts
        // four rows later but otherwise shares the dual-R ABI above.
        TagHash(0x80A9E64F) if pixel.inline_constants.len() >= 43 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[7];
            rows[1] = pixel.inline_constants[21];
            rows[2] = pixel.inline_constants[25];
            rows[3..=5].copy_from_slice(&pixel.inline_constants[22..=24]);
            rows[6..=8].copy_from_slice(&pixel.inline_constants[26..=28]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[29..=30]);
            rows[11..=13].copy_from_slice(&pixel.inline_constants[31..=33]);
            rows[14..=16].copy_from_slice(&pixel.inline_constants[34..=36]);
            rows[17..=18].copy_from_slice(&pixel.inline_constants[37..=38]);
            (
                21,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(6)?),
                None,
                rows,
            )
        }
        // Full9 A/G/B stack. t1.a gates the composite, t1.g gates t3,
        // t1.b selects the authored t4 detail, and t5 is the base tangent
        // normal. DXIL c24..c33 owns the exact gate/transform block.
        TagHash(0x80A9C31E) if pixel.inline_constants.len() >= 34 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[13];
            rows[1] = pixel.inline_constants[24];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[25..=27]);
            rows[5] = pixel.inline_constants[28];
            rows[6..=8].copy_from_slice(&pixel.inline_constants[29..=31]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[32..=33]);
            (
                22,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // DCF8 compact dual-detail ABI. t2 carries A/G/B/R selectors; B
        // selects transformed t4 and R selects transformed t5 over t6 base.
        TagHash(0x80A9DCF8) if pixel.inline_constants.len() >= 35 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[9];
            rows[1..=2].copy_from_slice(&pixel.inline_constants[20..=21]);
            rows[3..=5].copy_from_slice(&pixel.inline_constants[22..=24]);
            rows[6..=7].copy_from_slice(&pixel.inline_constants[25..=26]);
            rows[8..=10].copy_from_slice(&pixel.inline_constants[27..=29]);
            rows[11..=12].copy_from_slice(&pixel.inline_constants[30..=31]);
            rows[13] = pixel.inline_constants[34];
            (
                26,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                None,
                None,
                None,
                rows,
            )
        }
        // C96F wide runner stack: t3 selector; t5 G detail; t6 B detail;
        // t7 has two R-selected transforms; t8 base. t4.r/t4.g provide
        // authored normal response and AO respectively.
        TagHash(0x80A9C96F) if pixel.inline_constants.len() >= 44 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[14];
            rows[1] = pixel.inline_constants[28];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[25..=27]);
            rows[5..=7].copy_from_slice(&pixel.inline_constants[29..=31]);
            rows[8..=9].copy_from_slice(&pixel.inline_constants[32..=33]);
            rows[10..=12].copy_from_slice(&pixel.inline_constants[34..=36]);
            rows[13..=15].copy_from_slice(&pixel.inline_constants[37..=39]);
            rows[16..=17].copy_from_slice(&pixel.inline_constants[40..=41]);
            // Object-space t2 field: B-band thresholds, tri-planar
            // exponent/projection, then authored colour base and scale. This
            // generated technique's object transform is identity/zero.
            rows[18] = pixel.inline_constants[11];
            rows[19] = pixel.inline_constants[12];
            rows[20] = pixel.inline_constants[2];
            rows[21] = pixel.inline_constants[8];
            rows[22] = pixel.inline_constants[9];
            rows[23] = pixel.inline_constants[10];
            (
                27,
                tag_at(3)?,
                tag_at(5)?,
                tag_at(6)?,
                Some(tag_at(7)?),
                Some(tag_at(7)?),
                Some(tag_at(4)?),
                rows,
            )
        }
        // D3FC combines a large procedural material branch with the standard
        // A/G/B/R normal ABI. t3 selects t5/t6/t7 over t8 base; t4.r/t4.g
        // remain the authored normal response and AO channels.
        TagHash(0x80A9D3FC) if pixel.inline_constants.len() >= 62 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[43];
            rows[1] = pixel.inline_constants[44];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[46..=48]);
            rows[5] = pixel.inline_constants[45];
            rows[6..=8].copy_from_slice(&pixel.inline_constants[49..=51]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[52..=53]);
            rows[11..=13].copy_from_slice(&pixel.inline_constants[54..=56]);
            rows[14..=15].copy_from_slice(&pixel.inline_constants[57..=58]);
            // Object-space t0 field. c0/c1 are identity/zero for this
            // generated technique; c2 is the tri-planar exponent, c3 the
            // projection, and c4/c5 the authored colour base/scale.
            rows[16] = pixel.inline_constants[2];
            rows[17] = pixel.inline_constants[3];
            rows[18] = pixel.inline_constants[4];
            rows[19] = pixel.inline_constants[5];
            (
                28,
                tag_at(3)?,
                tag_at(5)?,
                tag_at(6)?,
                Some(tag_at(7)?),
                None,
                Some(tag_at(4)?),
                rows,
            )
        }
        // D2D7: t2 A/G gates the authored stack. B switches t4/t5; R
        // switches t6/t7; t8 is the base tangent normal.
        TagHash(0x80A9D2D7) if pixel.inline_constants.len() >= 61 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[38];
            rows[1] = pixel.inline_constants[39];
            rows[2] = pixel.inline_constants[43];
            rows[3..=5].copy_from_slice(&pixel.inline_constants[40..=42]);
            rows[6..=8].copy_from_slice(&pixel.inline_constants[44..=46]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[47..=48]);
            rows[11..=13].copy_from_slice(&pixel.inline_constants[49..=51]);
            rows[14..=16].copy_from_slice(&pixel.inline_constants[52..=54]);
            rows[17..=18].copy_from_slice(&pixel.inline_constants[55..=56]);
            (
                29,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(7)?),
                None,
                rows,
            )
        }
        // D569: t4 is selector; A gates t6, G gates t7, B/R jointly gate
        // t8 over t9 base. t5.r/t5.g provide material response and AO.
        TagHash(0x80A9D569) if pixel.inline_constants.len() >= 72 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[52];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[53..=55]);
            rows[4] = pixel.inline_constants[56];
            rows[5..=7].copy_from_slice(&pixel.inline_constants[57..=59]);
            rows[8] = pixel.inline_constants[60];
            rows[9..=10].copy_from_slice(&pixel.inline_constants[61..=62]);
            rows[11..=13].copy_from_slice(&pixel.inline_constants[63..=65]);
            rows[14..=15].copy_from_slice(&pixel.inline_constants[66..=67]);
            (
                30,
                tag_at(4)?,
                tag_at(6)?,
                tag_at(7)?,
                Some(tag_at(8)?),
                None,
                Some(tag_at(5)?),
                rows,
            )
        }
        // Full character surface used by Emerald Impact. t2.a gates t4,
        // t2.b selects t5, and t2.r selects t6 over the remapped t7 base.
        // Unlike the compact character path these authored normals are a
        // visible part of the shell material and cannot be discarded merely
        // because the same shader also owns procedural t1/t9/t10 effects.
        TagHash(0x80A9F4E5) if pixel.inline_constants.len() >= 33 => {
            let mut rows = [[0.0; 4]; 24];
            let detail_scale = pixel.inline_constants[0];
            rows[0] = [detail_scale[0], 0.0, detail_scale[2], 0.0];
            rows[14] = [0.0, detail_scale[1], detail_scale[3], 0.0];
            rows[1] = pixel.inline_constants[18];
            rows[2] = pixel.inline_constants[19];
            rows[3..=7].copy_from_slice(&pixel.inline_constants[20..=24]);
            rows[8..=12].copy_from_slice(&pixel.inline_constants[25..=29]);
            rows[13] = pixel.inline_constants[32];
            (
                31,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                None,
                rows,
            )
        }
        // Emerald Impact sibling: t2 selects t4/t5/t6/t7 through A/G/B/R,
        // then t8 supplies the authored base normal. LLVM establishes the
        // normal block at c18..c35; t6/t7 share c24/c25 UV transforms.
        TagHash(0x80A9F500) if pixel.inline_constants.len() >= 36 => {
            let mut rows = [[0.0; 4]; 24];
            let a_uv = pixel.inline_constants[0];
            rows[0] = [a_uv[0], 0.0, a_uv[2], 0.0];
            rows[1] = [0.0, a_uv[1], a_uv[3], 0.0];
            rows[2..=3].copy_from_slice(&pixel.inline_constants[18..=19]);
            rows[4..=7].copy_from_slice(&pixel.inline_constants[20..=23]);
            rows[8..=12].copy_from_slice(&pixel.inline_constants[24..=28]);
            rows[13] = pixel.inline_constants[29];
            rows[14..=15].copy_from_slice(&pixel.inline_constants[30..=31]);
            rows[16] = pixel.inline_constants[34];
            rows[17] = pixel.inline_constants[35];
            (
                32,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(7)?),
                None,
                rows,
            )
        }
        // Compact Emerald sibling: A selects t4. B and R independently
        // select two authored transforms of t5, over the remapped t7 base.
        TagHash(0x80A9F518) if pixel.inline_constants.len() >= 36 => {
            let mut rows = [[0.0; 4]; 24];
            let a_uv = pixel.inline_constants[0];
            rows[0] = [a_uv[0], 0.0, a_uv[2], 0.0];
            rows[1] = [0.0, a_uv[1], a_uv[3], 0.0];
            rows[2..=3].copy_from_slice(&pixel.inline_constants[18..=19]);
            rows[4..=8].copy_from_slice(&pixel.inline_constants[20..=24]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[25..=26]);
            rows[11..=15].copy_from_slice(&pixel.inline_constants[28..=32]);
            rows[16] = pixel.inline_constants[35];
            (
                33,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(5)?),
                None,
                None,
                rows,
            )
        }
        // Procedural Emerald sibling. Its normal block is shifted to c42:
        // t4 A, t5 G, t6 B, a second t4 transform for R, then t7 base.
        TagHash(0x80A9F4B3) if pixel.inline_constants.len() >= 64 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0..=3].copy_from_slice(&pixel.inline_constants[42..=45]);
            rows[4..=7].copy_from_slice(&pixel.inline_constants[46..=49]);
            rows[8..=15].copy_from_slice(&pixel.inline_constants[50..=57]);
            rows[16..=18].copy_from_slice(&pixel.inline_constants[50..=52]);
            rows[19..=20].copy_from_slice(&pixel.inline_constants[58..=59]);
            rows[21..=22].copy_from_slice(&pixel.inline_constants[62..=63]);
            rows[23] = pixel.inline_constants[31];
            (
                34,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                Some(tag_at(4)?),
                None,
                rows,
            )
        }
        // Compact character surface: t2.a gates transformed t4, t2.r selects
        // t5, and t6 is the remapped base normal (80A9F528 LLVM c18..c28).
        TagHash(0x80A9F528) if pixel.inline_constants.len() >= 29 => {
            let mut rows = [[0.0; 4]; 24];
            let a_uv = pixel.inline_constants[0];
            rows[0] = [a_uv[0], 0.0, a_uv[2], 0.0];
            rows[1] = [0.0, a_uv[1], a_uv[3], 0.0];
            rows[2..=3].copy_from_slice(&pixel.inline_constants[18..=19]);
            rows[4..=8].copy_from_slice(&pixel.inline_constants[20..=24]);
            rows[9..=10].copy_from_slice(&pixel.inline_constants[27..=28]);
            (
                35,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                None,
                None,
                None,
                rows,
            )
        }
        // Local panel/face surface: t1.r selects transformed t3 over t4.
        // t2 is its independent material-response field.
        TagHash(0x80A9F589) if pixel.inline_constants.len() >= 23 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0..=4].copy_from_slice(&pixel.inline_constants[14..=18]);
            rows[5..=6].copy_from_slice(&pixel.inline_constants[21..=22]);
            (
                36,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(3)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Full13 package-394 surface. t3 carries A/G/B/R selectors; t5/t6
        // provide A/G, t7/t8 are authored B alternatives, t7 is reused for
        // R, and t9 is base tangent normal. DXIL c43..c66 is exact ABI block.
        TagHash(0x80B142A5) if pixel.inline_constants.len() >= 67 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[32];
            rows[1..=5].copy_from_slice(&pixel.inline_constants[43..=47]);
            rows[6..=10].copy_from_slice(&pixel.inline_constants[48..=52]);
            rows[11..=18].copy_from_slice(&pixel.inline_constants[53..=60]);
            rows[19..=20].copy_from_slice(&pixel.inline_constants[61..=62]);
            rows[21..=22].copy_from_slice(&pixel.inline_constants[65..=66]);
            (
                37,
                tag_at(3)?,
                tag_at(5)?,
                tag_at(6)?,
                Some(tag_at(7)?),
                Some(tag_at(8)?),
                Some(tag_at(4)?),
                rows,
            )
        }
        // Package-394 full10 surface. t1 is an A/G/B/R selector. G gates
        // t3, B and R select t4/t5, and t6 is the authored base normal.
        // The response/AO field remains independently packed in t2.
        TagHash(0x80B143A0) if pixel.inline_constants.len() >= 42 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[13];
            rows[1..=15].copy_from_slice(&pixel.inline_constants[24..=38]);
            rows[16] = pixel.inline_constants[41];
            (
                38,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Package-394 full10 siblings. t2 selects the t4/t5/t6 stack using
        // G/B/R; t7 is the remapped base and t3 is response/AO. The shaders
        // differ only by two preceding constant rows.
        TagHash(0x80B1444E) if pixel.inline_constants.len() >= 39 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[11];
            rows[1..=14].copy_from_slice(&pixel.inline_constants[22..=35]);
            rows[15] = pixel.inline_constants[38];
            (
                39,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                Some(tag_at(3)?),
                rows,
            )
        }
        TagHash(0x80B14BF9) if pixel.inline_constants.len() >= 41 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[13];
            rows[1..=14].copy_from_slice(&pixel.inline_constants[24..=37]);
            rows[15] = pixel.inline_constants[40];
            (
                39,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(5)?,
                Some(tag_at(6)?),
                None,
                Some(tag_at(3)?),
                rows,
            )
        }
        // Package-394 expanded full11 surface. t1 G selects t3/t4, B selects
        // two transforms of t5, R gates t6, and t7 is the remapped base.
        // t2 owns response/AO.
        TagHash(0x80B14701) if pixel.inline_constants.len() >= 48 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[13];
            rows[1..=21].copy_from_slice(&pixel.inline_constants[24..=44]);
            rows[22] = pixel.inline_constants[47];
            (
                40,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                Some(tag_at(5)?),
                Some(tag_at(6)?),
                Some(tag_at(2)?),
                rows,
            )
        }
        // Woven runner fabric. t1 is the packed selector, t3 is a broad
        // authored detail normal, t4 is the tiled weave normal, t5 is the
        // base tangent normal, and t2 carries material response/AO.
        TagHash(0x80A9C430) if pixel.inline_constants.len() >= 45 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[13];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[24..=26]);
            rows[4] = pixel.inline_constants[27];
            rows[5] = pixel.inline_constants[28];
            rows[6..=8].copy_from_slice(&pixel.inline_constants[29..=31]);
            rows[9] = pixel.inline_constants[44];
            (
                45,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(4)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Compact woven runner fabric. t1 selects transformed t3 relief over
        // the authored t4 base normal; t2 carries response/AO.
        TagHash(0x80A9C65C) if pixel.inline_constants.len() >= 25 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[1];
            rows[1..=3].copy_from_slice(&pixel.inline_constants[14..=16]);
            rows[4] = pixel.inline_constants[24];
            (
                46,
                tag_at(1)?,
                tag_at(3)?,
                tag_at(3)?,
                None,
                None,
                Some(tag_at(2)?),
                rows,
            )
        }
        // Dual-gated armor relief. t2.a/t2.g gate transformed t4 over the
        // authored t5 base normal; t3 supplies material response/AO.
        TagHash(0x80A9CBEE) if pixel.inline_constants.len() >= 49 => {
            let mut rows = [[0.0; 4]; 24];
            rows[0] = pixel.inline_constants[37];
            rows[1] = pixel.inline_constants[41];
            rows[2..=4].copy_from_slice(&pixel.inline_constants[38..=40]);
            rows[5] = pixel.inline_constants[48];
            (
                47,
                tag_at(2)?,
                tag_at(4)?,
                tag_at(4)?,
                None,
                None,
                Some(tag_at(3)?),
                rows,
            )
        }
        _ => return None,
    };
    let procedural = match pixel.shader? {
        TagHash(0x80A9B860) => Some(tag_at(1)?),
        TagHash(0x80A9C96F) => Some(tag_at(2)?),
        TagHash(0x80A9D3FC) => Some(tag_at(0)?),
        TagHash(0x80A9AD5D) => Some(tag_at(3)?),
        TagHash(0x80A9AD68) => Some(tag_at(10)?),
        _ => None,
    };
    let color_overlay = match pixel.shader? {
        TagHash(0x80A9D569) => Some(tag_at(1)?),
        TagHash(0x80B142A5) => Some(tag_at(2)?),
        _ => None,
    };
    let mut color_overlay_constants = [[0.0; 4]; 7];
    match pixel.shader? {
        TagHash(0x80A9D569) => {
            // c0 colour, c1/c2 UV rows, c3 selector-alpha reference.
            color_overlay_constants[0..=3].copy_from_slice(&pixel.inline_constants[0..=3]);
        }
        TagHash(0x80B142A5) => {
            // c24/c25 UV rows, c28/c29 colour remap, c30 selector-G reference.
            for (target, source) in [24usize, 25, 28, 29, 30].into_iter().enumerate() {
                color_overlay_constants[target] = pixel.inline_constants[source];
            }
        }
        _ => {}
    }
    let procedural_wear = match pixel.shader? {
        TagHash(0x80A9E4DB) => Some([tag_at(9)?, tag_at(10)?, tag_at(11)?]),
        _ => None,
    };
    constants
        .iter()
        .chain(color_overlay_constants.iter())
        .flatten()
        .all(|value| value.is_finite())
        .then_some(RunnerLayeredSurfaceMaterial {
            mode,
            surface,
            detail_normal_a,
            detail_normal_b,
            detail_normal_c,
            detail_normal_d,
            procedural,
            color_overlay,
            color_overlay_constants,
            procedural_wear,
            material_response,
            constants,
        })
}

fn runner_occlusion_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<RunnerOcclusionMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let shader = preview
        .stages
        .iter()
        .find(|stage| stage.stage == "PS")?
        .shader?;
    let (slot, channel) = match shader {
        TagHash(0x80A9A9D3) | TagHash(0x80A9B860) => (2, 0),
        TagHash(0x80A9A8CF) => (2, 1),
        // AF8D/AF93 sample local t2 at mesh UV. DXIL routes t2.g into RT2.g
        // by averaging it with the generated geometry/procedural occlusion.
        TagHash(0x80A9AF8D) | TagHash(0x80A9AF93) => (2, 1),
        TagHash(0x80A9B06A) => (2, 1),
        TagHash(0x80A9B71C) => (2, 1),
        // Face/skin surface: t3.g is written into RT2.g after combining with
        // the generated response. t2 is the independent packed dye selector.
        TagHash(0x80A9B85C) => (3, 1),
        TagHash(0x80A9C430) => (2, 1),
        TagHash(0x80A9C65C) => (2, 1),
        TagHash(0x80A9CBEE) => (3, 1),
        TagHash(0x80A9AFB6) | TagHash(0x80A9B75C) | TagHash(0x80A9B93E) | TagHash(0x80A9BBBE)
        | TagHash(0x80A9BBBF) | TagHash(0x80A9BBDD) | TagHash(0x80A9CBAE) | TagHash(0x80A9D6FD)
        | TagHash(0x80A9DC9A) => (2, 1),
        TagHash(0x80A9C3F6) => (3, 1),
        TagHash(0x80A9AE19) | TagHash(0x80A9AFBF) | TagHash(0x80A9DAC9) | TagHash(0x80A9DB9A) => {
            (3, 1)
        }
        TagHash(0x80A9A9AD) | TagHash(0x80A9A9B1) | TagHash(0x80A9AFC1) | TagHash(0x80A9B065)
        | TagHash(0x80A9C27C) | TagHash(0x80A9DE77) | TagHash(0x80AA02A7) => (2, 1),
        // These generated runner surfaces use t2.r as normal response and
        // feed t2.g into RT2.g AO. Keep both roles bound independently.
        TagHash(0x80A9C31E) | TagHash(0x80A9DEEC) | TagHash(0x80A9E0FD) | TagHash(0x80A9E4DB)
        | TagHash(0x80A9E76C) => (2, 1),
        TagHash(0x80A9C96F) | TagHash(0x80A9D3FC) => (4, 1),
        TagHash(0x80A9D569) => (5, 1),
        TagHash(0x80B142A5) => (4, 1),
        TagHash(0x80B143A0) | TagHash(0x80B14701) => (2, 1),
        TagHash(0x80B1444E) | TagHash(0x80B14BF9) => (3, 1),
        _ => return None,
    };
    Some(RunnerOcclusionMaterial {
        texture: bindings.iter().find(|binding| binding.slot == slot)?.tag,
        channel,
    })
}

fn character_surface_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
    normal_slot: Option<u32>,
) -> Option<CharacterSurfaceMaterial> {
    if !bindings_use_character_gear_surface(bindings, normal_slot)
        && !bindings_use_compact_character_surface(bindings, normal_slot)
    {
        return None;
    }
    let tag_at = |slot| {
        bindings
            .iter()
            .find(|binding| binding.slot == slot)
            .map(|binding| binding.tag)
    };
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let constants = &preview
        .stages
        .iter()
        .find(|stage| stage.stage == "PS")?
        .inline_constants;
    let detail_transform = *constants.first()?;
    let detail_base = *constants.get(3)?;
    let detail_scale = *constants.get(4)?;
    let detail_gate = constants.get(5)?[0];
    let common_valid = detail_transform
        .iter()
        .chain(&detail_base)
        .chain(&detail_scale)
        .all(|value| value.is_finite())
        && detail_transform[0].abs() > 0.0001
        && detail_transform[1].abs() > 0.0001
        && detail_transform[..2]
            .iter()
            .all(|value| value.abs() <= 1024.0)
        // Common-character c3 is an authored linear RGBA colour. Several
        // unrelated runner shaders share the same wide t0..tn resource shape,
        // but place UV/scalar rows in c3/c4 and leave c3.a at zero. Treating
        // those rows as RGB produced the vivid cyan/green/red shell panels.
        && (detail_base[3] - 1.0).abs() <= 0.0001
        && detail_base[..3]
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
        && detail_scale[..3]
            .iter()
            .all(|value| (-1.0..=1.0).contains(value))
        && (0.0..=1.0).contains(&detail_gate)
        && texture_is_srgb(tag_at(1)?);
    if common_valid {
        return Some(CharacterSurfaceMaterial {
            mode: 1,
            surface: tag_at(2)?,
            selector: tag_at(3)?,
            detail_color: tag_at(1)?,
            detail_normal: tag_at(normal_slot?)?,
            procedural: None,
            detail_transform,
            detail_base,
            detail_scale,
            detail_gate,
            extra: [[0.0; 4]; 2],
            palette: [[1.0; 4]; 2],
            procedural_constants: [[0.0; 4]; 11],
        });
    }

    // Newer runner cosmetics use two transformed samples from t1 as a
    // procedural palette mask. t4.g selects that palette over local t0; t3
    // contributes the authored secondary mask. This is the literal layout in
    // Arata-family DXIL, detected by its four affine UV rows rather than tags.
    let palette = [*constants.first()?, *constants.get(1)?];
    let transforms = [
        *constants.get(2)?,
        *constants.get(3)?,
        *constants.get(4)?,
        *constants.get(5)?,
    ];
    let palette_gate = constants.get(25)?[0];
    let palette_valid = palette
        .iter()
        .flatten()
        .chain(transforms.iter().flatten())
        .all(|value| value.is_finite())
        && palette
            .iter()
            .all(|color| color[..3].iter().all(|value| (0.0..=1.0).contains(value)))
        && transforms
            .iter()
            .all(|row| row[..2].iter().any(|value| value.abs() > 0.0001))
        && (0.0..=1.0).contains(&palette_gate)
        && texture_preview_format(tag_at(2)?).contains("Rg16")
        && texture_is_srgb(tag_at(3)?)
        && !texture_is_srgb(tag_at(4)?)
        && normal_slot.is_some();
    if palette_valid {
        return Some(CharacterSurfaceMaterial {
            mode: 2,
            surface: tag_at(4)?,
            selector: tag_at(3)?,
            detail_color: tag_at(1)?,
            detail_normal: tag_at(normal_slot?)?,
            procedural: Some(tag_at(2)?),
            detail_transform: transforms[0],
            detail_base: transforms[1],
            detail_scale: transforms[2],
            detail_gate: palette_gate,
            extra: [transforms[3], [0.0; 4]],
            palette,
            procedural_constants: [
                *constants.get(6)?,
                *constants.get(10)?,
                *constants.get(11)?,
                *constants.get(12)?,
                *constants.get(13)?,
                *constants.get(14)?,
                *constants.get(15)?,
                *constants.get(20)?,
                *constants.get(22)?,
                *constants.get(23)?,
                *constants.get(24)?,
            ],
        });
    }

    // A wide resource table is not a material ABI. These remaining shaders
    // are separate cloth/skin/eye/layer families; routing all of them through
    // one invented palette branch caused the missing and white runner panels.
    // Leave them unclassified until their compiled shader contract is decoded.
    None
}

pub(crate) fn weapon_skin_gear_dye_palette(
    cache: &TagCache,
    selected_pattern: TagHash,
) -> Option<[GearDyeMaterial; 6]> {
    weapon_skin_gear_dye_palette_with_source(cache, selected_pattern)
        .map(|(palette, _object_channels)| palette)
}

fn weapon_skin_gear_dye_palette_with_source(
    cache: &TagCache,
    selected_pattern: TagHash,
) -> Option<([GearDyeMaterial; 6], bool)> {
    let mut queue = std::collections::VecDeque::from([(selected_pattern, 0usize)]);
    let mut seen = rustc_hash::FxHashSet::default();
    seen.insert(selected_pattern);

    while let Some((tag, depth)) = queue.pop_front() {
        let entry = package_manager().get_entry(tag)?;
        if entry.reference == CLASS_PATTERN_COMPONENT
            && let Ok(data) = package_manager().read_tag(tag)
            && let Some(palette) = decode_weapon_skin_gear_dye_palette_with_source(&data)
        {
            return Some(palette);
        }
        if depth >= 4 {
            continue;
        }
        for child in cache
            .hashes
            .get(&tag)
            .into_iter()
            .flat_map(|scan| scan.file_hashes.iter().map(|reference| reference.hash))
            .unique()
        {
            let Some(child_entry) = package_manager().get_entry(child) else {
                continue;
            };
            if matches!(
                child_entry.reference,
                CLASS_PATTERN | CLASS_PATTERN_COMPONENT
            ) && seen.insert(child)
            {
                queue.push_back((child, depth + 1));
            }
        }
    }
    None
}

/// Resolves one six-channel GearDye contribution from authored Pattern object
/// channels. Worn Dye may be inherited from the selected skin; Dye Detail is
/// normally local to the attached mod. The compiled material selects these by
/// parameter hash, never by serialized vector position.
fn pattern_gear_dye_color_contribution(
    cache: &TagCache,
    root: TagHash,
    parameters: [u32; 6],
) -> Option<[[f32; 4]; 6]> {
    let endian = package_manager().version.endian();
    let mut values = [None; 6];
    for (node, _depth) in descendant_pattern_nodes_with_depth(cache, root, 8) {
        let Ok(data) = package_manager().read_tag(node) else {
            continue;
        };
        for channel_array in scan_arrays(&data, endian)
            .into_iter()
            .filter(|array| array.class == CLASS_PATTERN_OBJECT_CHANNELS)
        {
            for (index, record) in array_records(&data, channel_array, 0x70)
                .into_iter()
                .enumerate()
            {
                let Some(parameter) = read_u32_at(record, 0, endian) else {
                    continue;
                };
                let Some(slot) = parameters
                    .iter()
                    .position(|candidate| *candidate == parameter)
                else {
                    continue;
                };
                if values[slot].is_some() {
                    continue;
                }
                let record_offset = channel_array.data_offset + index * 0x70;
                let Some(constants) = read_array(&data, record_offset + 0x18, 0x10, endian) else {
                    continue;
                };
                // These contribution records are constant TFX expressions.
                // Reject compound expressions instead of treating an arbitrary
                // literal pool entry as the channel output.
                if constants.len() != 0x10
                    || read_array(&data, record_offset + 0x08, 1, endian)
                        .is_none_or(|bytecode| bytecode.is_empty())
                {
                    continue;
                }
                let Some(mut value) = read_vec4_f32(constants, 0, endian) else {
                    continue;
                };
                if value[..3]
                    .iter()
                    .any(|component| !component.is_finite() || !(-4.0..=4.0).contains(component))
                {
                    continue;
                }
                value[3] = 0.0;
                values[slot] = Some(value);
            }
        }
    }
    values
        .into_iter()
        .collect::<Option<Vec<_>>>()?
        .try_into()
        .ok()
}

fn decode_weapon_skin_gear_dye_palette(data: &[u8]) -> Option<[GearDyeMaterial; 6]> {
    decode_weapon_skin_gear_dye_palette_with_source(data).map(|(palette, _object_channels)| palette)
}

fn decode_weapon_skin_gear_dye_palette_with_source(
    data: &[u8],
) -> Option<([GearDyeMaterial; 6], bool)> {
    let endian = package_manager().version.endian();
    let arrays = scan_arrays(data, endian);
    // Updated Goliath components bind parameters to object-channel expression
    // records. Read each record's authored Vector4 constant by binding index;
    // singleton Vector4 arrays alone lose gaps occupied by scalar expressions.
    for binding_array in arrays
        .iter()
        .copied()
        .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
    {
        let parameter_indices = array_records(data, binding_array, 0x0c)
            .into_iter()
            .filter_map(|record| {
                (read_u32_at(record, 0x00, endian)? == PATTERN_LOCAL_SCOPE_HASH).then_some(())?;
                Some((
                    read_u32_at(record, 0x04, endian)?,
                    usize::try_from(read_u32_at(record, 0x08, endian)?).ok()?,
                ))
            })
            .collect::<rustc_hash::FxHashMap<_, _>>();
        for channel_array in arrays
            .iter()
            .copied()
            .filter(|array| array.class == CLASS_PATTERN_OBJECT_CHANNELS)
        {
            let channel_records = array_records(data, channel_array, 0x70);
            let parameter_vector = |parameter: u32| {
                let index = *parameter_indices.get(&parameter)?;
                let record = *channel_records.get(index)?;
                (read_u32_at(record, 0, endian)? == parameter).then_some(())?;
                let record_offset = channel_array.data_offset + index * 0x70;
                let constants = read_array(data, record_offset + 0x18, 0x10, endian)?;
                let mut value = read_vec4_f32(constants.get(..0x10)?, 0, endian)?;
                value[3] = 1.0;
                Some(value)
            };
            if let Some(palette) = decode_gear_dye_palette(parameter_vector) {
                return Some((palette, true));
            }
        }
    }

    let singleton_vectors = arrays
        .iter()
        .copied()
        .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
        .collect_vec();
    // Goliath Pattern components can carry unrelated extension vectors. The
    // authored local-scope binding table, not a build-specific total count,
    // identifies the 18 GearDye color/roughness/metal vectors.
    let vectors = singleton_vectors
        .iter()
        .copied()
        .map(|array| read_serialized_dye_vector(data, array, endian))
        .collect::<Option<Vec<_>>>()?;
    let parameter_indices = arrays
        .iter()
        .copied()
        .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
        .flat_map(|array| array_records(data, array, 0x0c))
        .filter_map(|record| {
            (read_u32_at(record, 0x00, endian)? == PATTERN_LOCAL_SCOPE_HASH).then_some(())?;
            Some((
                read_u32_at(record, 0x04, endian)?,
                usize::try_from(read_u32_at(record, 0x08, endian)?).ok()?,
            ))
        })
        .collect::<rustc_hash::FxHashMap<_, _>>();
    decode_gear_dye_palette(|parameter| {
        parameter_indices
            .get(&parameter)
            .and_then(|index| vectors.get(*index))
            .copied()
    })
    .map(|palette| (palette, false))
}

fn decode_gear_dye_palette(
    parameter_vector: impl Fn(u32) -> Option<[f32; 4]>,
) -> Option<[GearDyeMaterial; 6]> {
    let palette = std::array::from_fn(|slot| GearDyeMaterial {
        color: parameter_vector(GEAR_DYE_COLOR_PARAMETERS[slot]).unwrap_or_default(),
        roughness_remap: parameter_vector(GEAR_DYE_ROUGHNESS_PARAMETERS[slot]).unwrap_or_default(),
        metal_remap: parameter_vector(GEAR_DYE_METAL_PARAMETERS[slot]).unwrap_or_default(),
    });
    GEAR_DYE_COLOR_PARAMETERS
        .into_iter()
        .chain(GEAR_DYE_ROUGHNESS_PARAMETERS)
        .chain(GEAR_DYE_METAL_PARAMETERS)
        .all(|parameter| parameter_vector(parameter).is_some())
        .then_some(())
        .and_then(|()| {
            palette
                .iter()
                .all(|dye| {
                    valid_dye_color(dye.color)
                        && valid_dye_remap(dye.roughness_remap)
                        && valid_dye_remap(dye.metal_remap)
                })
                .then_some(palette)
        })
}

fn read_serialized_dye_vector(data: &[u8], array: TagArray, endian: Endian) -> Option<[f32; 4]> {
    let record = data.get(array.data_offset..array.data_offset + 0x10)?;
    read_dye_vector(record, endian)
}

fn read_dye_vector(record: &[u8], endian: Endian) -> Option<[f32; 4]> {
    Some([
        read_f32(record.get(0x0..0x4)?, endian),
        read_f32(record.get(0x4..0x8)?, endian),
        read_f32(record.get(0x8..0xc)?, endian),
        1.0,
    ])
}

fn valid_dye_color(color: [f32; 4]) -> bool {
    color[..3]
        .iter()
        .all(|channel| channel.is_finite() && (0.0..=4.0).contains(channel))
}

fn valid_dye_remap(remap: [f32; 4]) -> bool {
    remap[..3]
        .iter()
        .all(|component| component.is_finite() && (-16.0..=16.0).contains(component))
}

fn apply_weapon_mod_attachment_poses(
    parts: &mut [(TagHash, MeshSourcePreview, WireframePreview)],
    attachments: &[ResolvedWeaponModAttachment],
) {
    let attachment_poses = attachments
        .iter()
        .map(|attachment| (attachment.geometry, attachment.pose))
        .collect::<rustc_hash::FxHashMap<_, _>>();
    for (tag, _source, wireframe) in parts.iter_mut() {
        let Some(pose) = attachment_poses.get(tag) else {
            continue;
        };
        for authored in &mut wireframe.authored_inputs {
            authored.attachment_pose = Some(*pose);
        }
        // Common-surface VS forwards raw shader-input POSITION/NORMAL directly
        // to PS procedural varyings. Preserve those before socket transforms.
        wireframe
            .procedural_positions
            .get_or_insert_with(|| wireframe.vertices.clone());
        if wireframe.procedural_normals.is_none() {
            wireframe.procedural_normals = wireframe.normals.clone();
        }
        for vertex in &mut wireframe.vertices {
            let local = vertex.map(|value| value * pose.scale);
            let rotated = rotate_quaternion(local, pose.rotation);
            *vertex = [
                rotated[0] + pose.translation[0],
                rotated[1] + pose.translation[1],
                rotated[2] + pose.translation[2],
            ];
        }
        if let Some(normals) = &mut wireframe.normals {
            normals
                .iter_mut()
                .for_each(|normal| *normal = rotate_quaternion(*normal, pose.rotation));
        }
        if let Some(tangents) = &mut wireframe.tangents {
            tangents.iter_mut().for_each(|tangent| {
                let rotated =
                    rotate_quaternion([tangent[0], tangent[1], tangent[2]], pose.rotation);
                *tangent = [rotated[0], rotated[1], rotated[2], tangent[3]];
            });
        }
        if let Some((min, max)) = bounds(&wireframe.vertices) {
            wireframe.min = min;
            wireframe.max = max;
        }
    }
}

fn pattern_nearest_geometry_tags(cache: &TagCache, root: TagHash) -> Vec<TagHash> {
    let mut frontier = vec![(root, 0usize)];
    let mut seen = rustc_hash::FxHashSet::default();
    let mut geometry = vec![];
    let mut nearest_depth = usize::MAX;
    seen.insert(root);

    while let Some((parent, depth)) = frontier.pop() {
        if depth >= nearest_depth {
            continue;
        }
        for child in cache.hashes.get(&parent).into_iter().flat_map(|scan| {
            scan.file_hashes.iter().map(|child| child.hash).chain(
                scan.file_hashes64
                    .iter()
                    .filter_map(|child| tag64_to_hash32(child.hash)),
            )
        }) {
            let Some(reference) = package_manager()
                .get_entry(child)
                .map(|entry| entry.reference)
            else {
                continue;
            };
            if reference == CLASS_GEOMETRY_RESOURCE {
                let child_depth = depth + 1;
                if child_depth < nearest_depth {
                    nearest_depth = child_depth;
                    geometry.clear();
                }
                if child_depth == nearest_depth {
                    geometry.push(child);
                }
            } else if matches!(reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                && seen.insert(child)
                && seen.len() < 256
            {
                frontier.push((child, depth + 1));
            }
        }
    }

    geometry
        .into_iter()
        .unique()
        .sorted_by_key(|tag| (tag.pkg_id(), tag.entry_index()))
        .collect()
}

fn ancestor_pattern_roots(cache: &TagCache, tag: TagHash) -> Vec<(usize, TagHash)> {
    let mut frontier = vec![(tag, 0usize)];
    let mut seen = rustc_hash::FxHashSet::default();
    let mut roots = vec![];
    seen.insert(tag);
    while let Some((child, depth)) = frontier.pop() {
        if depth >= 8 {
            continue;
        }
        for parent in cache
            .hashes
            .get(&child)
            .into_iter()
            .flat_map(|scan| scan.references.iter().copied())
        {
            let Some(reference) = package_manager()
                .get_entry(parent)
                .map(|entry| entry.reference)
            else {
                continue;
            };
            if !matches!(reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT) {
                continue;
            }
            if reference == CLASS_PATTERN {
                roots.push((depth + 1, parent));
            }
            if seen.insert(parent) {
                frontier.push((parent, depth + 1));
            }
        }
    }
    roots.sort_by_key(|(depth, tag)| (*depth, tag.pkg_id(), tag.entry_index()));
    roots.dedup_by_key(|(_depth, tag)| *tag);
    roots
}

fn rooted_pattern_equivalent(cache: &TagCache, selected: TagHash) -> Option<TagHash> {
    let selected_entry = package_manager().get_entry(selected)?;
    if selected_entry.reference != CLASS_PATTERN {
        return None;
    }
    let signature = normalized_pattern_payload(cache, selected)?;
    let selected_transform = pattern_nearest_geometry_tags(cache, selected)
        .first()
        .and_then(|tag| package_manager().read_tag(*tag).ok())
        .and_then(|data| {
            read_geometry_position_transform(&data, package_manager().version.endian())
        });
    package_manager()
        .get_all_by_reference(CLASS_PATTERN)
        .into_iter()
        .filter(|(tag, entry)| *tag != selected && entry.file_size == selected_entry.file_size)
        .filter(|(tag, _entry)| {
            normalized_pattern_payload(cache, *tag).as_ref() == Some(&signature)
        })
        .filter(|(tag, _entry)| !ancestor_pattern_roots(cache, *tag).is_empty())
        .filter(|(tag, _entry)| {
            let transform = pattern_nearest_geometry_tags(cache, *tag)
                .first()
                .and_then(|geometry| package_manager().read_tag(*geometry).ok())
                .and_then(|data| {
                    read_geometry_position_transform(&data, package_manager().version.endian())
                });
            transforms_match(selected_transform, transform)
        })
        .map(|(tag, _entry)| tag)
        .min()
}

fn normalized_pattern_payload(cache: &TagCache, tag: TagHash) -> Option<Vec<u8>> {
    let mut data = package_manager().read_tag(tag).ok()?;
    let scan = cache.hashes.get(&tag)?;
    for reference in &scan.file_hashes {
        let offset = usize::try_from(reference.offset).ok()?;
        if let Some(bytes) = data.get_mut(offset..offset.saturating_add(4)) {
            bytes.fill(0);
        }
    }
    for reference in &scan.file_hashes64 {
        let offset = usize::try_from(reference.offset).ok()?;
        if let Some(bytes) = data.get_mut(offset..offset.saturating_add(8)) {
            bytes.fill(0);
        }
    }
    Some(data)
}

fn transforms_match(
    left: Option<GeometryPositionTransform>,
    right: Option<GeometryPositionTransform>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left
            .scale
            .into_iter()
            .chain(left.offset)
            .zip(right.scale.into_iter().chain(right.offset))
            .all(|(left, right)| (left - right).abs() < 0.00001),
        (None, None) => true,
        _ => false,
    }
}

fn related_pattern_geometry_tags(cache: &TagCache, tag: TagHash) -> Vec<TagHash> {
    let mut frontier = vec![(tag, 0usize)];
    let mut seen = rustc_hash::FxHashSet::default();
    let mut roots = vec![];
    seen.insert(tag);

    while let Some((child, depth)) = frontier.pop() {
        if depth >= 8 {
            continue;
        }
        for parent in cache
            .hashes
            .get(&child)
            .into_iter()
            .flat_map(|scan| scan.references.iter().copied())
        {
            let Some(reference) = package_manager()
                .get_entry(parent)
                .map(|entry| entry.reference)
            else {
                continue;
            };
            if !matches!(reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT) {
                continue;
            }
            let parent_depth = depth + 1;
            if reference == CLASS_PATTERN {
                roots.push((parent_depth, parent));
            }
            if seen.insert(parent) {
                frontier.push((parent, parent_depth));
            }
        }
    }

    roots.sort_by_key(|(depth, root)| (*depth, root.pkg_id(), root.entry_index()));
    roots.dedup_by_key(|(_depth, root)| *root);
    let Some(mut geometry) = roots
        .iter()
        .map(|(_depth, root)| pattern_geometry_tags(cache, *root))
        .find(|geometry| pattern_geometry_is_assembly(geometry))
        .or_else(|| {
            roots
                .last()
                .map(|(_depth, root)| pattern_geometry_tags(cache, *root))
        })
    else {
        let direct = pattern_geometry_tags(cache, tag);
        return if direct.is_empty() { vec![tag] } else { direct };
    };

    geometry.push(tag);
    geometry
        .into_iter()
        .filter(|geometry_tag| {
            package_manager()
                .get_entry(*geometry_tag)
                .is_some_and(|entry| entry.reference == CLASS_GEOMETRY_RESOURCE)
        })
        .unique()
        .sorted_by_key(|geometry_tag| (geometry_tag.pkg_id(), geometry_tag.entry_index()))
        .take(64)
        .collect()
}

fn pattern_geometry_is_assembly(geometry: &[TagHash]) -> bool {
    if geometry.len() < 2 {
        return false;
    }
    let endian = package_manager().version.endian();
    let transforms = geometry
        .iter()
        .filter_map(|tag| package_manager().read_tag(*tag).ok())
        .filter_map(|data| read_geometry_position_transform(&data, endian))
        .collect_vec();
    if transforms.len() < 2 {
        return false;
    }
    let mut diameters = transforms
        .iter()
        .map(|transform| {
            transform
                .scale
                .iter()
                .map(|scale| scale.abs() * 2.0)
                .fold(0.0_f32, f32::max)
        })
        .collect_vec();
    diameters.sort_by(f32::total_cmp);
    let median_part = diameters[diameters.len() / 2].max(0.0001);
    let assembly_extent = (0..3)
        .map(|axis| {
            let min = transforms
                .iter()
                .map(|transform| transform.offset[axis] - transform.scale[axis].abs())
                .fold(f32::INFINITY, f32::min);
            let max = transforms
                .iter()
                .map(|transform| transform.offset[axis] + transform.scale[axis].abs())
                .fold(f32::NEG_INFINITY, f32::max);
            max - min
        })
        .fold(0.0_f32, f32::max);
    assembly_extent >= median_part * 2.0
}

fn pattern_geometry_tags(cache: &TagCache, root: TagHash) -> Vec<TagHash> {
    let mut geometry = vec![];
    let mut frontier = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    seen.insert(root);
    while let Some(parent) = frontier.pop() {
        for child in cache.hashes.get(&parent).into_iter().flat_map(|scan| {
            scan.file_hashes.iter().map(|child| child.hash).chain(
                scan.file_hashes64
                    .iter()
                    .filter_map(|child| tag64_to_hash32(child.hash)),
            )
        }) {
            let Some(reference) = package_manager()
                .get_entry(child)
                .map(|entry| entry.reference)
            else {
                continue;
            };
            if reference == CLASS_GEOMETRY_RESOURCE {
                geometry.push(child);
            } else if matches!(reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                && seen.insert(child)
                && seen.len() < 256
            {
                frontier.push(child);
            }
        }
    }

    geometry
        .into_iter()
        .filter(|geometry_tag| {
            package_manager()
                .get_entry(*geometry_tag)
                .is_some_and(|entry| entry.reference == CLASS_GEOMETRY_RESOURCE)
        })
        .unique()
        .sorted_by_key(|geometry_tag| (geometry_tag.pkg_id(), geometry_tag.entry_index()))
        .collect()
}

fn merge_model_wireframes(
    parts: Vec<(TagHash, MeshSourcePreview, WireframePreview)>,
) -> Option<WireframePreview> {
    if parts.len() == 1 {
        return parts
            .into_iter()
            .next()
            .map(|(_tag, _source, wireframe)| wireframe);
    }
    if parts.is_empty() {
        return None;
    }

    let position_format = parts[0].2.position_format;
    let uv_format = parts
        .iter()
        .map(|(_tag, _source, wireframe)| wireframe.uv_format.as_deref())
        .all_equal_value()
        .ok()
        .flatten()
        .map(|format| format!("{format} (assembled; transforms baked)"));
    let has_complete_uvs = parts
        .iter()
        .all(|(_tag, _source, wireframe)| wireframe.uvs.is_some());
    let has_complete_normals = parts
        .iter()
        .all(|(_tag, _source, wireframe)| wireframe.normals.is_some());
    let has_complete_tangents = parts
        .iter()
        .all(|(_tag, _source, wireframe)| wireframe.tangents.is_some());
    let has_complete_rigid_indices = parts
        .iter()
        .all(|(_tag, _source, wireframe)| wireframe.rigid_indices.is_some());
    let normal_format = has_complete_normals
        .then(|| {
            parts
                .iter()
                .filter_map(|(_tag, _source, wireframe)| wireframe.normal_format.as_deref())
                .all_equal_value()
                .ok()
                .map(|format| format!("{format} (assembled)"))
        })
        .flatten();
    let tangent_format = has_complete_tangents
        .then(|| {
            parts
                .iter()
                .filter_map(|(_tag, _source, wireframe)| wireframe.tangent_format.as_deref())
                .all_equal_value()
                .ok()
                .map(|format| format!("{format} (assembled)"))
        })
        .flatten();
    let vertex_count_total = parts
        .iter()
        .map(|(_tag, _source, wireframe)| wireframe.vertex_count_total)
        .sum();
    let index_count_total = parts
        .iter()
        .map(|(_tag, _source, wireframe)| wireframe.index_count_total)
        .sum();
    let source = format!(
        "assembled {}",
        parts
            .iter()
            .map(|(tag, _source, _wireframe)| tag)
            .format(" + ")
    );
    let mut vertices = Vec::new();
    let mut rigid_indices = has_complete_rigid_indices.then(Vec::new);
    let mut normals = has_complete_normals.then(Vec::new);
    let mut procedural_positions = Some(Vec::new());
    let mut procedural_normals = has_complete_normals.then(Vec::new);
    let mut tangents = has_complete_tangents.then(Vec::new);
    let mut uvs = has_complete_uvs.then(Vec::new);
    let mut indices = Vec::new();
    let mut material_ranges = Vec::new();
    let mut authored_inputs = Vec::new();
    let mut authored_shadow_ranges = Vec::new();
    let mut authored_shadow_index_count = 0usize;

    for (_tag, source, wireframe) in parts {
        let available_vertices = MAX_PREVIEW_VERTICES.saturating_sub(vertices.len());
        if available_vertices == 0 {
            break;
        }
        let copied_vertices = wireframe.vertices.len().min(available_vertices);
        let vertex_base = vertices.len() as u32;
        vertices.extend(wireframe.vertices.iter().copied().take(copied_vertices));
        if let (Some(output), Some(part)) = (&mut rigid_indices, &wireframe.rigid_indices) {
            output.extend(part.iter().copied().take(copied_vertices));
        }
        if let Some(output_positions) = &mut procedural_positions {
            output_positions.extend(
                wireframe
                    .procedural_positions
                    .as_ref()
                    .unwrap_or(&wireframe.vertices)
                    .iter()
                    .copied()
                    .take(copied_vertices),
            );
        }
        if let (Some(output_normals), Some(part_normals)) = (&mut normals, &wireframe.normals) {
            output_normals.extend(part_normals.iter().copied().take(copied_vertices));
        }
        if let Some(output_normals) = &mut procedural_normals {
            let part_normals = wireframe
                .procedural_normals
                .as_ref()
                .or(wireframe.normals.as_ref());
            if let Some(part_normals) = part_normals {
                output_normals.extend(part_normals.iter().copied().take(copied_vertices));
            }
        }
        if let (Some(output_tangents), Some(part_tangents)) = (&mut tangents, &wireframe.tangents) {
            output_tangents.extend(part_tangents.iter().copied().take(copied_vertices));
        }
        if let (Some(output_uvs), Some(part_uvs)) = (&mut uvs, &wireframe.uvs) {
            output_uvs.extend(part_uvs.iter().copied().take(copied_vertices).map(|uv| {
                source
                    .uv_transform
                    .map(|transform| {
                        [
                            uv[0] * transform.scale[0] + transform.offset[0],
                            uv[1] * transform.scale[1] + transform.offset[1],
                        ]
                    })
                    .unwrap_or(uv)
            }));
        }

        let authored_source_base = authored_inputs.len();
        let authored_source_count = wireframe.authored_inputs.len();
        authored_inputs.extend(wireframe.authored_inputs.iter().cloned());

        for mut range in wireframe.authored_shadow_ranges {
            range.authored_source = range
                .authored_source
                .filter(|source| *source < authored_source_count)
                .map(|source| authored_source_base + source);
            let available = MAX_PREVIEW_INDICES.saturating_sub(authored_shadow_index_count);
            if available < 3 {
                break;
            }
            let mut remapped = Vec::new();
            for triangle in range.indices.chunks_exact(3) {
                if remapped.len() + 3 > available
                    || triangle
                        .iter()
                        .any(|index| *index as usize >= copied_vertices)
                {
                    continue;
                }
                remapped.extend(triangle.iter().map(|index| index + vertex_base));
            }
            if !remapped.is_empty() {
                authored_shadow_index_count += remapped.len();
                range.indices = remapped;
                authored_shadow_ranges.push(range);
            }
        }

        let ranges = if wireframe.material_ranges.is_empty() {
            vec![WireframeMaterialRange {
                index_start: 0,
                index_count: wireframe.indices.len(),
                raw_lod_category: Some(source.raw_lod_category),
                render_stage: None,
                technique: None,
                gear_dye_change_color_index: None,
                authored_source: (authored_source_count != 0).then_some(authored_source_base),
                authored_draw: None,
                procedural_scale: 1.0,
                texture: None,
                textures: WireframeMaterialTextures::default(),
            }]
        } else {
            wireframe.material_ranges
        };
        for mut range in ranges {
            range.authored_source = range
                .authored_source
                .filter(|source| *source < authored_source_count)
                .map(|source| authored_source_base + source);
            let source = wireframe
                .indices
                .get(
                    range.index_start.min(wireframe.indices.len())
                        ..range
                            .index_start
                            .saturating_add(range.index_count)
                            .min(wireframe.indices.len()),
                )
                .unwrap_or_default();
            let index_start = indices.len();
            for triangle in source.chunks_exact(3) {
                if indices.len() + 3 > MAX_PREVIEW_INDICES
                    || triangle
                        .iter()
                        .any(|index| *index as usize >= copied_vertices)
                {
                    continue;
                }
                indices.extend(triangle.iter().map(|index| index + vertex_base));
            }
            let index_count = indices.len() - index_start;
            if index_count > 0 {
                material_ranges.push(WireframeMaterialRange {
                    index_start,
                    index_count,
                    raw_lod_category: range.raw_lod_category,
                    render_stage: range.render_stage,
                    technique: range.technique,
                    gear_dye_change_color_index: range.gear_dye_change_color_index,
                    authored_source: range.authored_source,
                    authored_draw: range.authored_draw,
                    procedural_scale: range.procedural_scale,
                    texture: range.texture,
                    textures: range.textures,
                });
            }
        }
        if indices.len() >= MAX_PREVIEW_INDICES {
            break;
        }
    }

    let (min, max) = bounds(&vertices)?;
    Some(WireframePreview {
        source,
        position_format,
        uv_format,
        vertices,
        rigid_indices,
        normals,
        procedural_positions,
        procedural_normals,
        tangents,
        uvs,
        normal_format,
        tangent_format,
        indices,
        material_ranges,
        authored_inputs,
        authored_shadow_ranges,
        min,
        max,
        vertex_count_total,
        index_count_total,
    })
}

fn build_model_wireframe(
    vertex_buffers: &[(TagHash, UEntryHeader)],
    index_buffers: &[(TagHash, UEntryHeader)],
) -> Option<WireframePreview> {
    let (vertex_tag, vertex_entry) = vertex_buffers.first()?;
    let vertex_header_data = package_manager().read_tag(*vertex_tag).ok()?;
    let vertex_preview =
        load_vertex_buffer_preview_for_tag(*vertex_tag, vertex_entry, &vertex_header_data).ok()?;

    let mut wireframe = vertex_preview.wireframe?;
    if let Some((index_tag, index_entry)) = index_buffers.first() {
        let index_header_data = package_manager().read_tag(*index_tag).ok()?;
        if let Ok(index_preview) =
            load_index_buffer_preview_for_tag(*index_tag, index_entry, &index_header_data)
        {
            wireframe.index_count_total = index_preview.index_count;
            wireframe.indices = index_preview.indices;
            wireframe.source = format!("{vertex_tag} + {index_tag}");
        }
    }

    Some(wireframe)
}

fn find_model_textures(
    cache: &TagCache,
    tag: TagHash,
    techniques: &[(TagHash, UEntryHeader)],
) -> Vec<(TagHash, UEntryHeader)> {
    let technique_order = techniques
        .iter()
        .enumerate()
        .map(|(index, (tag, _entry))| (*tag, index))
        .collect::<rustc_hash::FxHashMap<_, _>>();
    let mut textures = techniques
        .iter()
        .map(|(tag, _entry)| *tag)
        .into_iter()
        .enumerate()
        .flat_map(texture_bindings_from_technique)
        .filter_map(|texture_tag| {
            package_manager()
                .get_entry(texture_tag.binding.tag)
                .map(|entry| {
                    (
                        texture_tag.binding.tag,
                        entry,
                        texture_tag.technique_order,
                        texture_binding_rank(texture_tag.binding),
                        (texture_tag.binding.tag.pkg_id() != tag.pkg_id()) as u8,
                        0usize,
                        texture_tag.technique_order,
                        texture_preview_rank(texture_tag.binding.tag),
                    )
                })
        })
        .chain(
            find_related_tags(cache, tag, TagSearchKind::Texture, 8)
                .into_iter()
                .map(|(texture_tag, entry)| {
                    let (parent_count, parent_order) =
                        texture_parent_technique_rank(cache, texture_tag, &technique_order);
                    (
                        texture_tag,
                        entry,
                        usize::MAX,
                        200,
                        (texture_tag.pkg_id() != tag.pkg_id()) as u8,
                        parent_count,
                        parent_order,
                        texture_preview_rank(texture_tag),
                    )
                }),
        )
        .collect_vec();

    textures.sort_by_key(
        |(
            tag,
            entry,
            technique_order,
            rank,
            locality,
            parent_count,
            parent_order,
            preview_rank,
        )| {
            (
                *technique_order,
                *rank,
                *locality,
                *preview_rank,
                Reverse(*parent_count),
                *parent_order,
                Reverse(entry.file_size),
                tag.0,
            )
        },
    );
    textures
        .into_iter()
        .unique_by(
            |(
                tag,
                _entry,
                _technique_order,
                _rank,
                _locality,
                _parent_count,
                _parent_order,
                _preview_rank,
            )| { *tag },
        )
        .map(
            |(
                tag,
                entry,
                _technique_order,
                _rank,
                _locality,
                _parent_count,
                _parent_order,
                _preview_rank,
            )| { (tag, entry) },
        )
        .take(128)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TexturePreviewRank {
    shape: u8,
    format: u8,
    area: Reverse<u64>,
    width: Reverse<u32>,
}

fn texture_preview_rank(tag: TagHash) -> TexturePreviewRank {
    let Ok(desc) = Texture::validated_descriptor_d2(tag) else {
        return TexturePreviewRank {
            shape: 100,
            format: 100,
            area: Reverse(0),
            width: Reverse(0),
        };
    };

    let format = format!("{:?}", desc.format);
    let format_rank = if format.contains("Srgb") {
        0
    } else if format.contains("Bc7") || format.contains("Rgba") {
        20
    } else if format.contains("Bc4") {
        80
    } else {
        50
    };
    let shape_rank = if desc.depth == 1 && desc.array_size == 1 && desc.width > 1 && desc.height > 1
    {
        0
    } else {
        50
    };

    TexturePreviewRank {
        shape: shape_rank,
        format: format_rank,
        area: Reverse(desc.width as u64 * desc.height as u64),
        width: Reverse(desc.width as u32),
    }
}

fn texture_parent_technique_rank(
    cache: &TagCache,
    texture_tag: TagHash,
    technique_order: &rustc_hash::FxHashMap<TagHash, usize>,
) -> (usize, usize) {
    let Some(scan) = cache.hashes.get(&texture_tag) else {
        return (0, usize::MAX);
    };

    let parents = scan
        .references
        .iter()
        .filter_map(|parent| technique_order.get(parent).copied())
        .collect_vec();
    (
        parents.len(),
        parents.into_iter().min().unwrap_or(usize::MAX),
    )
}

fn assign_wireframe_material_textures(
    wireframe: &mut WireframePreview,
    cache: &TagCache,
    textures: &[(TagHash, UEntryHeader)],
) {
    if wireframe.material_ranges.is_empty() && wireframe.authored_shadow_ranges.is_empty() {
        return;
    }

    let mut materials = rustc_hash::FxHashMap::default();
    for range in &mut wireframe.material_ranges {
        let Some(technique) = range.technique else {
            range.texture = textures.first().map(|(tag, _entry)| *tag);
            range.textures.color = range.texture;
            continue;
        };

        range.textures = materials
            .entry(technique)
            .or_insert_with(|| material_textures_for_technique(technique, cache, textures))
            .clone();
        range.texture = range.textures.color;
    }

    for range in &mut wireframe.authored_shadow_ranges {
        let Some(technique) = range.technique else {
            range.texture = textures.first().map(|(tag, _entry)| *tag);
            range.textures.color = range.texture;
            continue;
        };

        range.textures = materials
            .entry(technique)
            .or_insert_with(|| material_textures_for_technique(technique, cache, textures))
            .clone();
        range.texture = range.textures.color;
    }
}

fn material_textures_for_technique(
    technique: TagHash,
    cache: &TagCache,
    textures: &[(TagHash, UEntryHeader)],
) -> WireframeMaterialTextures {
    let direct = package_manager()
        .get_entry(technique)
        .zip(package_manager().read_tag(technique).ok())
        .map(|(entry, data)| {
            (
                texture_bindings_for_technique(&entry, &data),
                material_constants_for_technique(&entry, &data),
                primary_sampler_for_technique(&entry, &data),
            )
        })
        .unwrap_or_default();

    let mut material = WireframeMaterialTextures::default();
    apply_material_constants(&mut material, direct.1);
    material.sampler = direct.2;
    let candidates = direct
        .0
        .into_iter()
        .filter(|binding| binding.stage == "PS")
        .sorted_by_key(|binding| texture_binding_rank(*binding))
        .collect_vec();
    let normal_slot = material_normal_texture_slot_for_technique(technique, &candidates);
    let control_slot = material_control_texture_slot(technique, &candidates, normal_slot);
    let color_slot = material_color_texture_slot_for_technique(technique, &candidates);
    let character_surface = character_surface_material(technique, &candidates, normal_slot);
    let runner_layered_surface = runner_layered_surface_material(technique, &candidates);
    let runner_occlusion = runner_occlusion_material(technique, &candidates);
    let alpha_mask = runner_alpha_mask_material(technique, &candidates);
    let investment_decal = investment_decal_for_technique(technique, &candidates);
    let direct_shared_color_atlas = direct_shared_color_atlas_for_technique(technique, &candidates);
    material.shared_atlas_detail = direct_shared_color_atlas
        .and_then(|_| shared_atlas_detail_material(technique, &candidates));
    material.mod_wear = weapon_mod_wear_material(technique, &candidates);
    material.surface_condition = weapon_surface_condition_material(technique, &candidates);
    // Both families can expose eight PS textures, but slots 5..7 mean physical
    // age/wear when the TFX wear ABI is present. Never reinterpret those wear
    // resources as decorative contour inputs.
    material.gear_pattern = (material.mod_wear.is_none())
        .then(|| gear_pattern_material(technique, &candidates))
        .flatten();

    for binding in &candidates {
        let binding = *binding;
        let mut role = if color_slot == Some(binding.slot) {
            MaterialTextureRole::Color
        } else if color_slot.is_some() && binding.slot == 0 {
            MaterialTextureRole::Aux
        } else {
            material_texture_role(binding, normal_slot, control_slot)
        };
        if investment_decal
            .as_ref()
            .is_some_and(|decal| decal.color() == binding.tag)
        {
            role = MaterialTextureRole::Color;
        } else if direct_shared_color_atlas == Some(binding.tag) {
            role = MaterialTextureRole::Color;
        } else if fallback_aux_texture(binding.tag)
            && !matches!(
                role,
                MaterialTextureRole::Control(_) | MaterialTextureRole::Normal
            )
        {
            role = MaterialTextureRole::Aux;
        }
        assign_material_texture(&mut material, binding.tag, role);
    }

    material.character_surface = character_surface;
    material.runner_layered_surface = runner_layered_surface;
    material.runner_occlusion = runner_occlusion;
    material.alpha_mask = alpha_mask;
    if let Some(alpha_mask) = alpha_mask {
        material.control = Some(alpha_mask.texture);
        material
            .aux
            .retain(|texture| *texture != alpha_mask.texture);
    }

    // Compact character permutations omit the extended t10+ character block,
    // but their compiled shaders write the same Goliath MRT contract as the
    // full family: RT1.a = 0.67 roughness and RT2.r = 0 metalness. Their t2/t3
    // textures are selectors/packed masks, not an ORM map.
    if material.character_surface.is_none()
        && bindings_use_compact_character_surface(&candidates, normal_slot)
    {
        material.solid_surface = Some([0.67, 0.0]);
    }
    if material.solid_surface.is_none() {
        material.solid_surface = runner_solid_surface_material(technique);
    }

    if let Some(investment_decal) = investment_decal {
        material.color = Some(investment_decal.color());
        material.authored_shared_atlas =
            matches!(investment_decal, InvestmentDecalResolution::Atlas(_));
        if let InvestmentDecalResolution::Shader(shader) = investment_decal {
            // Reuse the renderer's linear data-texture bindings. Investment
            // decal shaders do not have a material normal map: PS t3/t4/t5 are
            // detail, colour, and opacity respectively.
            material.normal = shader.detail;
            material.control = Some(shader.mask);
            material.investment_decal = Some(shader);
        }
    }
    if let Some(atlas) = direct_shared_color_atlas {
        material.color = Some(atlas);
        material.authored_shared_atlas = true;
        material.aux.retain(|texture| *texture != atlas);
        if material.shared_atlas_detail.is_some() {
            // Audited MRT ABI: RT1.a = 0.67 and RT2.r = 0 for this direct
            // atlas + linear triplanar-response family.
            material.solid_surface = Some([0.67, 0.0]);
        }
    }
    if let Some(condition) = material.surface_condition {
        material.aux.retain(|texture| *texture != condition.breakup);
    }

    for texture in textures
        .iter()
        .map(|(texture, _entry)| *texture)
        .filter(|texture| texture_has_parent_technique(cache, *texture, technique))
    {
        let role = guessed_related_texture_role(texture, technique, &material);
        assign_material_texture(&mut material, texture, role);
    }

    if material.color.is_none() {
        material.color = textures
            .iter()
            .map(|(texture, _entry)| *texture)
            .find(|texture| {
                texture_has_parent_technique(cache, *texture, technique)
                    && fallback_color_candidate(*texture, technique)
            });
    }

    promote_auxiliary_preview_color(&mut material, technique);
    material.forward_coating = forward_coating_material_for_technique(technique);
    material.transmission = material
        .forward_coating
        .is_none()
        .then(|| transmission_material_for_technique(technique))
        .flatten();

    if material.color.is_none()
        && let Some(flat) = textureless_flat_material_for_technique(technique)
    {
        material.solid_color = Some(flat.color);
        material.solid_surface = Some([flat.roughness, flat.metalness]);
    } else if material.color.is_none()
        && let Some(surface) = procedural_surface_material_for_technique(technique)
    {
        material.solid_color = Some(surface.color);
        material.solid_surface = Some([surface.roughness, surface.metalness]);
    }

    material
}

/// Decode Tiger's authored forward-coating ABI.
///
/// Signature is deliberately narrow: dedicated PS, premultiplied stage state,
/// direct linear-detail/cubemap resources, fixed material block, and TFX
/// coverage driven by object channel 0x37CD36CF. Only material ranges that
/// author this shader contract receive coating.
fn forward_coating_material_for_technique(technique: TagHash) -> Option<ForwardCoatingMaterial> {
    const FORWARD_COATING_PS: TagHash = TagHash(0x80A9FBAC);
    const COVERAGE_OBJECT_CHANNEL: u32 = 0x37CD36CF;

    if render_state_for_technique(technique).blend != Some(8) {
        return None;
    }
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    if pixel.shader != Some(FORWARD_COATING_PS) || pixel.inline_constants.len() < 44 {
        return None;
    }

    let bindings = texture_bindings_for_technique(&entry, &data)
        .into_iter()
        .filter(|binding| binding.stage == "PS")
        .collect_vec();
    let detail = bindings.iter().find(|binding| binding.slot == 1)?.tag;
    let environment = bindings.iter().find(|binding| binding.slot == 2)?.tag;
    let environment_sampler = sampler_for_technique_slot(&entry, &data, "PS", 2)?;
    if Texture::load_desc(detail).ok()?.kind() != crate::texture::TextureType::Texture2D
        || Texture::load_desc(environment).ok()?.kind() != crate::texture::TextureType::TextureCube
    {
        return None;
    }

    let constants = &pixel.inline_constants;
    let finite = |value: &[f32; 4]| value.iter().all(|component| component.is_finite());
    for index in [
        15, 16, 17, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43,
    ] {
        if !finite(constants.get(index)?) {
            return None;
        }
    }
    let projection_exponent = constants[15][0];
    let incidence_remap = [constants[24][0], constants[24][1]];
    let authored_coverage = constants[25][0];
    let lobe_direction_scales = [constants[39][0], constants[43][0]];
    if !(1.0..=128.0).contains(&projection_exponent)
        || constants[16][0].abs() <= 0.0001
        || constants[16][1].abs() <= 0.0001
        || !(0.0..=4.0).contains(&incidence_remap[0])
        || !(0.0..=1.0).contains(&authored_coverage)
        || !(0.0..=4.0).contains(&lobe_direction_scales[0])
        || !(0.0..=4.0).contains(&lobe_direction_scales[1])
    {
        return None;
    }

    let channel = format!("0x{COVERAGE_OBJECT_CHANNEL:08X}");
    let coverage_channel = pixel
        .bytecode
        .ops
        .iter()
        .any(|op| op.name == "push_object_channel" && op.detail == channel);
    let coverage_output = pixel.bytecode.expressions.iter().any(|expression| {
        expression.target == "output[131]"
            && expression.expression.contains("spline4_const")
            && expression
                .expression
                .contains(&format!("object_channel({channel})"))
    });
    if !coverage_channel || !coverage_output {
        return None;
    }
    let object_channels = std::collections::HashMap::from([(COVERAGE_OBJECT_CHANNEL, [1.0; 4])]);
    let (_, expressions) = interpret_tfx_stack_with_object_channels(
        &pixel.bytecode.ops,
        &pixel.constants,
        &object_channels,
    );
    let coverage_input = expressions
        .iter()
        .find(|expression| expression.target == "output[131]")
        .and_then(|expression| expression.value)?[0];
    if !coverage_input.is_finite() {
        return None;
    }

    Some(ForwardCoatingMaterial {
        detail,
        environment,
        environment_sampler,
        colors: [
            [constants[22][0], constants[22][1], constants[22][2], 1.0],
            [constants[23][0], constants[23][1], constants[23][2], 1.0],
        ],
        incidence_remap,
        coverage: (authored_coverage * coverage_input).clamp(0.0, 1.0),
        projection: constants[16],
        projection_exponent,
        detail_remap: [constants[17][0], constants[17][1]],
        response_remap: [constants[31][0], constants[31][1]],
        environment_lod: [constants[29][0], constants[30][0]],
        environment_remap: [constants[32][0], constants[32][1]],
        environment_strength: constants[33][0],
        environment_params: constants[34],
        specular_colors: [constants[36], constants[40]],
        specular_exponents: [constants[37][0], constants[41][0]],
        specular_strengths: [constants[38][0], constants[42][0]],
        lobe_direction_scales,
    })
}

/// Decode common stage-8 transmission material block.
///
/// Marathon surface permutations keep each base colour followed by metalness
/// at +17 vectors and roughness at +21. Transmission permutations reuse this
/// ABI for one or two absorption colours. Locating relative surface fields
/// avoids shader/tag/weapon-specific constant indices.
fn transmission_material_for_technique(technique: TagHash) -> Option<TransmissionMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let stage = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    refractive_glass_material(stage)
        .or_else(|| halftone_glow_material(technique, stage))
        .or_else(|| transmission_material_from_constants(&stage.inline_constants))
}

/// Constant rows a halftone-glow pixel program reads, taken from its source:
/// two scrolling samples of one noise texture are multiplied and remapped,
/// a dot grid `frac(uv * grid) - centre` is subtracted, and what remains
/// lights the surface additively in a colour between the two tints.
struct HalftoneGlowRows {
    source_sha256: [u8; 32],
    tints: [usize; 2],
    noise_transforms: [usize; 2],
    noise_remap: usize,
    noise_gain: usize,
    tint_exponent: usize,
    grid: usize,
    centre: usize,
    dot_remap: usize,
    alpha_remap: usize,
    gains: [usize; 2],
}

const HALFTONE_GLOW_PROGRAMS: &[HalftoneGlowRows] = &[HalftoneGlowRows {
    // PS 80AA01D0 (Outland Dusk Eliminator side panels).
    source_sha256: crate::render::authored_program::decode_sha256(
        "d8e022b69d8f8b1aaed8696dceefebf6ae81286ec5308f3a92e35306982f7c4a",
    ),
    tints: [17, 18],
    noise_transforms: [19, 20],
    noise_remap: 21,
    noise_gain: 22,
    tint_exponent: 23,
    grid: 24,
    centre: 25,
    dot_remap: 27,
    alpha_remap: 29,
    gains: [31, 32],
}];

fn halftone_glow_material(
    technique: TagHash,
    stage: &crate::material::TechniqueStagePreview,
) -> Option<TransmissionMaterial> {
    use sha2::{Digest, Sha256};
    let entry = package_manager().get_entry(stage.shader?)?;
    let payload = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let digest: [u8; 32] = Sha256::digest(payload).into();
    let rows = HALFTONE_GLOW_PROGRAMS.iter().find(|rows| rows.source_sha256 == digest)?;
    // The noise transforms scroll with the clock: evaluate them a second
    // apart and hand the shader the rate.
    let descriptor = crate::render::technique::TechniqueDescriptor::load(technique)?;
    let pixel = descriptor.stages.iter().find(|stage| stage.stage == crate::render::technique::ShaderStage::Pixel)?;
    let at = |seconds: f32| {
        let mut inputs = crate::render::tfx::TfxRuntimeInputs::default();
        inputs.apply_marathon_global_defaults();
        inputs.time_seconds = seconds;
        pixel.runtime_state(&inputs).constant_registers
    };
    let (start, later) = (at(0.0), at(1.0));
    let row = |index: usize| start.get(index).copied();
    let rate = |index: usize| Some([later.get(index)?[2] - row(index)?[2], later.get(index)?[3] - row(index)?[3]]);
    let gain = row(rows.gains[0])?[0] * row(rows.gains[1])?[0];
    let (rate_a, rate_b) = (rate(rows.noise_transforms[0])?, rate(rows.noise_transforms[1])?);
    Some(TransmissionMaterial {
        colors: [
            [row(rows.tints[0])?[0], row(rows.tints[0])?[1], row(rows.tints[0])?[2], gain],
            [row(rows.tints[1])?[0], row(rows.tints[1])?[1], row(rows.tints[1])?[2], 1.0],
        ],
        surfaces: [row(rows.noise_transforms[0])?, row(rows.noise_transforms[1])?],
        color_count: 2,
        absorption: None,
        halftone: Some([
            [rate_a[0], rate_a[1], rate_b[0], rate_b[1]],
            [row(rows.noise_remap)?[0], row(rows.noise_remap)?[1], row(rows.noise_gain)?[0], row(rows.tint_exponent)?[0]],
            [row(rows.grid)?[0], row(rows.grid)?[1], row(rows.centre)?[0], row(rows.centre)?[1]],
            [row(rows.dot_remap)?[0], row(rows.dot_remap)?[1], row(rows.alpha_remap)?[0], row(rows.alpha_remap)?[1]],
        ]),
    })
}

/// Constant rows a refractive-glass pixel program reads, taken from its source:
/// the tint is `lerp(near, far, thickness)`, multiplied onto `gain` times the
/// scene colour behind the surface, and the result is blended at `opacity`.
struct RefractiveGlassRows {
    source_sha256: [u8; 32],
    near: usize,
    far: usize,
    gain: usize,
    opacity: usize,
}

const REFRACTIVE_GLASS_PROGRAMS: &[RefractiveGlassRows] = &[RefractiveGlassRows {
    // PS 80B14F45 (Assassin Vox Nocturna arm shell and finger guards).
    source_sha256: crate::render::authored_program::decode_sha256(
        "60b2ef28d04f4aae432cbe97143a6b75aeb662b8b48a578f272914296ff69cd9",
    ),
    near: 28,
    far: 29,
    gain: 24,
    opacity: 31,
}];

fn refractive_glass_material(
    stage: &crate::material::TechniqueStagePreview,
) -> Option<TransmissionMaterial> {
    use sha2::{Digest, Sha256};
    let entry = package_manager().get_entry(stage.shader?)?;
    let payload = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let digest: [u8; 32] = Sha256::digest(payload).into();
    let rows = REFRACTIVE_GLASS_PROGRAMS.iter().find(|rows| rows.source_sha256 == digest)?;
    let row = |index: usize| stage.inline_constants.get(index).copied();
    let color = |index: usize| row(index).map(|value| [value[0], value[1], value[2], 1.0]);
    Some(TransmissionMaterial {
        colors: [color(rows.near)?, color(rows.far)?],
        surfaces: [[0.0; 4]; 2],
        color_count: 2,
        absorption: Some([row(rows.opacity)?[0], row(rows.gain)?[0]]),
        halftone: None,
    })
}

fn transmission_material_from_constants(constants: &[[f32; 4]]) -> Option<TransmissionMaterial> {
    let mut candidates = Vec::new();
    for (index, mut color) in constants.iter().copied().enumerate() {
        let Some(metalness) = constants.get(index + 17).map(|value| value[0]) else {
            continue;
        };
        let Some(roughness) = constants.get(index + 21).map(|value| value[0]) else {
            continue;
        };
        let valid_color = color[..3]
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && color[..3].iter().filter(|value| **value > 0.001).count() >= 2
            && color[3].is_finite()
            && (0.0..=1.0).contains(&color[3]);
        if valid_color
            && metalness.is_finite()
            && (0.0..=1.0).contains(&metalness)
            && roughness.is_finite()
            && (0.02..=1.0).contains(&roughness)
        {
            color[3] = 1.0;
            candidates.push((index, color, [roughness, metalness, 0.0, 0.0]));
        }
    }
    if candidates.is_empty() {
        return stage8_alpha_color_material(constants);
    }

    // Procedural surface blocks carry the same epsilon sentinel used by the
    // compiled stage-8 shader. Other inline constants can coincidentally look
    // like valid colour/surface tuples (Vox Nocturna has white and green false
    // positives before its authored orange block), so sentinel-backed tuples
    // take precedence. Older permutations without the sentinel retain the
    // relative-offset fallback.
    let marked = candidates
        .iter()
        .filter(|(index, _, _)| {
            constants
                .get(index.saturating_sub(1))
                .is_some_and(|marker| {
                    (0.0001..=0.01).contains(&marker[0])
                        && marker[1..].iter().all(|value| value.abs() < 0.0001)
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    let selected = if marked.is_empty() {
        &candidates
    } else {
        &marked
    };
    let mut colors = Vec::with_capacity(2);
    let mut surfaces = Vec::with_capacity(2);
    for (_, color, surface) in selected {
        if !colors.contains(color) {
            colors.push(*color);
            surfaces.push(*surface);
        }
        if colors.len() == 2 {
            break;
        }
    }
    let color_count = colors.len() as u8;
    colors.resize(2, colors[0]);
    surfaces.resize(2, surfaces[0]);
    Some(TransmissionMaterial {
        colors: [colors[0], colors[1]],
        surfaces: [surfaces[0], surfaces[1]],
        color_count,
        absorption: None,
        halftone: None,
    })
}

/// Decode the compact stage-8 material ABI used by runner-shell effects.
///
/// Unlike the extended surface block above, these permutations mark authored
/// colours with alpha = 1. A scalar immediately after the colour is roughness;
/// older variants place that scalar shortly before the colour. Remaining
/// effect constants have alpha = 0, which keeps this decoder structural rather
/// than tied to a particular tag or colour.
fn stage8_alpha_color_material(constants: &[[f32; 4]]) -> Option<TransmissionMaterial> {
    let mut authored_colors = constants
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, color)| {
            color[3].is_finite()
                && (color[3] - 1.0).abs() <= 0.001
                && color[..3]
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                && color[..3].iter().any(|value| *value > 0.001)
        })
        .take(2)
        .collect::<Vec<_>>();
    if authored_colors.is_empty() {
        // Procedural runner energy uses an emissive RGB vector with alpha 0,
        // followed by its signed edge remap (positive scale, negative bias and
        // epsilon). This is a separate compiled stage-8 ABI from absorption
        // glass; preserve its authored blue/orange/etc. instead of falling
        // back to the first bound single-channel mask.
        authored_colors = constants
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, color)| {
                color[3].abs() <= 0.001
                    && color[..3]
                        .iter()
                        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                    && color[..3].iter().filter(|value| **value > 0.01).count() >= 2
                    && constants.get(index + 1).is_some_and(|remap| {
                        (1.0..=4.0).contains(&remap[0])
                            && remap[1] < -1.0
                            && (-0.1..0.0).contains(&remap[2])
                            && remap[3].abs() <= 0.001
                    })
            })
            .take(1)
            .collect();
        if authored_colors.is_empty() {
            return None;
        }
    }

    let scalar_roughness = |color_index: usize| {
        let scalar = |value: &[f32; 4]| {
            value[0].is_finite()
                && (0.02..=1.0).contains(&value[0])
                && value[1..].iter().all(|component| component.abs() <= 0.0001)
        };
        constants
            .get(color_index + 1)
            .filter(|value| scalar(value))
            .map(|value| value[0])
            .or_else(|| {
                (color_index.saturating_sub(6)..color_index)
                    .rev()
                    .find_map(|index| constants.get(index).filter(|value| scalar(value)))
                    .map(|value| value[0])
            })
            .unwrap_or(0.5)
    };

    let mut colors = Vec::with_capacity(2);
    let mut surfaces = Vec::with_capacity(2);
    for (index, mut color) in authored_colors {
        color[3] = 1.0;
        if !colors.contains(&color) {
            colors.push(color);
            surfaces.push([scalar_roughness(index), 0.0, 0.0, 0.0]);
        }
    }
    let color_count = colors.len() as u8;
    colors.resize(2, colors[0]);
    surfaces.resize(2, surfaces[0]);
    Some(TransmissionMaterial {
        colors: [colors[0], colors[1]],
        surfaces: [surfaces[0], surfaces[1]],
        color_count,
        absorption: None,
        halftone: None,
    })
}

/// Resolve an opaque material whose authored colour is a shared atlas.
///
/// Shared atlases are normally technical resources and must not win generic
/// albedo guessing. An opaque decal technique, however, can bind that same
/// atlas as its direct PS t0 colour source, optionally beside linear response
/// maps. Direct-slot ownership makes that use unambiguous without naming a
/// weapon or technique tag.
fn direct_shared_color_atlas_for_technique(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<TagHash> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    if tfx_has_marathon_decal_abi(&pixel.bytecode) {
        return None;
    }
    let pixel_textures = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .copied()
        .unique_by(|binding| binding.slot)
        .collect_vec();
    let color_sources = pixel_textures
        .iter()
        .filter(|binding| texture_is_srgb(binding.tag))
        .copied()
        .collect_vec();
    let [atlas] = color_sources.as_slice() else {
        return None;
    };
    // Shared-atlas surface shaders may pair t0 colour with linear BC4/BC5
    // response maps. Requiring one total texture discarded the proven t0
    // colour and replaced it with Quicktag's white fallback.
    (atlas.slot == 0 && !matches!(render_state_for_technique(technique).blend, Some(26 | 27)))
        .then_some(atlas.tag)
}

/// Decode the common two-texture shared-atlas surface ABI.
///
/// Its compiled pixel shader writes
/// `t0.rgb * saturate(base + scale * triplanar(t1)) * 4.5947933` to RT0.
/// Recognize the binding/constant shape, never a weapon or tag hash.
fn shared_atlas_detail_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<SharedAtlasDetailMaterial> {
    let detail = bindings
        .iter()
        .find(|binding| {
            binding.stage == "PS" && binding.slot == 1 && !texture_is_srgb(binding.tag)
        })?
        .tag;
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let constants = &preview
        .stages
        .iter()
        .find(|stage| stage.stage == "PS")?
        .inline_constants;
    let exponent = constants.get(3)?[0];
    let projection = *constants.get(4)?;
    let base = constants.get(5)?;
    let scale = constants.get(6)?;
    (exponent.is_finite()
        && exponent > 0.0
        && projection[0].is_finite()
        && projection[1].is_finite()
        && projection[0] > 0.0
        && projection[1] > 0.0)
        .then_some(SharedAtlasDetailMaterial {
            detail,
            projection,
            exponent,
            base: [base[0], base[1], base[2]],
            scale: [scale[0], scale[1], scale[2]],
            ambient_occlusion: 0.5,
        })
}

/// Resolve Tiger's investment-decal pass.
///
/// Weapon and runner geometry carries decal quads with selector UVs already
/// baked into the mesh. Their pixel techniques read Marathon `Decal` extern
/// fields at raw scope 45 (`normals_read` texture +0x8 and resolution/offset
/// vec4 +0x30), then use either a sole colour atlas or a colour + opacity +
/// detail texture ABI. Treating these bindings as an ordinary material drops
/// the authored stencil and renders Quicktag's white fallback. Decode the pass
/// from its blend state, TFX extern, texture formats, and constant layout; no
/// asset/tag rule is used.
#[derive(Debug, Clone, Copy, PartialEq)]
enum InvestmentDecalResolution {
    Atlas(TagHash),
    Shader(InvestmentDecalMaterial),
}

impl InvestmentDecalResolution {
    fn color(self) -> TagHash {
        match self {
            Self::Atlas(color) => color,
            Self::Shader(material) => material.color,
        }
    }
}

fn investment_decal_for_technique(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<InvestmentDecalResolution> {
    if !matches!(render_state_for_technique(technique).blend, Some(26 | 27)) {
        return None;
    }
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    let has_marathon_decal_abi = tfx_has_marathon_decal_abi(&pixel.bytecode);
    let pixel_textures = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .copied()
        .sorted_by_key(|binding| binding.slot)
        .unique_by(|binding| binding.slot)
        .collect_vec();

    match pixel_textures.as_slice() {
        [atlas] if has_marathon_decal_abi => Some(InvestmentDecalResolution::Atlas(atlas.tag)),
        [color, mask]
            if color.slot == 3
                && mask.slot == 4
                && has_marathon_decal_abi
                && texture_is_srgb(color.tag)
                && texture_is_single_channel(mask.tag)
                && pixel.inline_constants.len() == 9 =>
        {
            let constants = &pixel.inline_constants;
            Some(InvestmentDecalResolution::Shader(InvestmentDecalMaterial {
                mode: InvestmentDecalMode::SceneNormalColorMask,
                color: color.tag,
                mask: mask.tag,
                detail: None,
                selector_colors: [[0.0; 4]; 5],
                selector_color_count: 0,
                atlas_selector_max: 0,
                mask_mode: InvestmentDecalMaskMode::Threshold,
                mask_threshold: constants[1][0],
                detail_transform: [1.0, 1.0, 0.0, 0.0],
                detail_base: [1.0; 4],
                detail_scale: [0.0; 4],
                grayscale_remap: [0.0, 1.0, 0.0, 0.0],
                positive_mask_remap: [0.0, 1.0, 0.0, 0.0],
                negative_mask_remap: [0.0, 1.0, 0.0, 0.0],
                output_gate: 1.0,
            }))
        }
        [color, mask]
            if ((color.slot == 2 && mask.slot == 3) || (color.slot == 3 && mask.slot == 4))
                && texture_is_srgb(color.tag)
                && texture_is_single_channel(mask.tag)
                && pixel.inline_constants.len() > 10 =>
        {
            if color.slot == 2 && mask.slot == 3 && pixel.inline_constants.len() == 18 {
                let constants = &pixel.inline_constants;
                return Some(InvestmentDecalResolution::Shader(InvestmentDecalMaterial {
                    mode: InvestmentDecalMode::SelectorMask,
                    color: color.tag,
                    mask: mask.tag,
                    detail: None,
                    selector_colors: [[0.0; 4]; 5],
                    selector_color_count: 0,
                    atlas_selector_max: 0,
                    mask_mode: InvestmentDecalMaskMode::Threshold,
                    mask_threshold: constants[1][0],
                    detail_transform: [1.0, 1.0, 0.0, 0.0],
                    detail_base: [1.0; 4],
                    detail_scale: [0.0; 4],
                    grayscale_remap: [0.0, 1.0, 0.0, 0.0],
                    positive_mask_remap: [0.0, 1.0, 0.0, 0.0],
                    negative_mask_remap: [0.0, 1.0, 0.0, 0.0],
                    output_gate: constants[17][1].clamp(0.0, 1.0),
                }));
            }

            let mut selector_colors = [[0.0; 4]; 5];
            selector_colors[0] = pixel.inline_constants[1];
            selector_colors[1] = pixel.inline_constants[2];
            Some(InvestmentDecalResolution::Shader(InvestmentDecalMaterial {
                mode: InvestmentDecalMode::SelectorMask,
                color: color.tag,
                mask: mask.tag,
                detail: None,
                selector_colors,
                selector_color_count: 2,
                atlas_selector_max: 0,
                mask_mode: InvestmentDecalMaskMode::Threshold,
                mask_threshold: pixel.inline_constants[3][0],
                detail_transform: [1.0, 1.0, 0.0, 0.0],
                detail_base: [1.0; 4],
                detail_scale: [0.0; 4],
                grayscale_remap: [0.0, 1.0, 0.0, 0.0],
                positive_mask_remap: [0.0, 1.0, 0.0, 0.0],
                negative_mask_remap: [0.0, 1.0, 0.0, 0.0],
                output_gate: pixel.inline_constants[10][1].clamp(0.0, 1.0),
            }))
        }
        [detail, color, mask]
            if has_marathon_decal_abi
                && texture_is_single_channel(detail.tag)
                && texture_is_srgb(color.tag)
                && texture_is_single_channel(mask.tag)
                && pixel.inline_constants.len() > 15 =>
        {
            let constants = &pixel.inline_constants;
            let selector_color_count = constants[4..]
                .iter()
                .take(5)
                .take_while(|color| color[3] > 0.5)
                .count();
            if selector_color_count == 0 {
                return None;
            }
            let mut selector_colors = [[0.0; 4]; 5];
            selector_colors[..selector_color_count]
                .copy_from_slice(&constants[4..4 + selector_color_count]);
            let remap = 4 + selector_color_count;
            let uv_split = selector_color_count == 1;
            Some(InvestmentDecalResolution::Shader(InvestmentDecalMaterial {
                mode: InvestmentDecalMode::DetailSelectorMask,
                color: color.tag,
                mask: mask.tag,
                detail: Some(detail.tag),
                selector_colors,
                selector_color_count: selector_color_count as u8,
                atlas_selector_max: u8::from(uv_split),
                mask_mode: if uv_split {
                    InvestmentDecalMaskMode::UvSplit
                } else {
                    InvestmentDecalMaskMode::Binary
                },
                mask_threshold: 0.0,
                detail_transform: constants[1],
                detail_base: constants[2],
                detail_scale: constants[3],
                grayscale_remap: constants[remap],
                positive_mask_remap: constants
                    .get(remap + 1)
                    .copied()
                    .unwrap_or([0.0, 1.0, 0.0, 0.0]),
                negative_mask_remap: constants
                    .get(remap + 2)
                    .copied()
                    .unwrap_or([0.0, 1.0, 0.0, 0.0]),
                output_gate: constants[15][1].clamp(0.0, 1.0),
            }))
        }
        _ => None,
    }
}

/// Decode common gear shader's procedural contour branch.
///
/// DXIL samples PS t6 in three object-space planes, perturbs a periodic line
/// function, then uses the t3 RGB bit selector to blend authored constants over
/// t0 for material IDs 2 and 4. Missing t6 therefore removes only the marble /
/// topographic strokes while leaving every ordinary decal intact.
fn gear_pattern_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<GearPatternMaterial> {
    let pixel_bindings = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .copied()
        .sorted_by_key(|binding| binding.slot)
        .unique_by(|binding| binding.slot)
        .collect_vec();
    // Distinguish this family from seven-slot environment surface shaders and
    // eight-slot mod-wear shaders using authored layout plus constant ABI.
    if pixel_bindings.last()?.slot != 7
        || !pixel_bindings.iter().any(|binding| binding.slot == 3)
        || !pixel_bindings.iter().any(|binding| binding.slot == 7)
    {
        return None;
    }
    let field = pixel_bindings.iter().find(|binding| binding.slot == 6)?.tag;
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let constants = &preview
        .stages
        .iter()
        .find(|stage| stage.stage == "PS")?
        .inline_constants;

    let projection = *constants.get(43)?;
    let normal_power = constants.get(42)?[0];
    let base_warp = constants.get(44)?[0];
    let field_midpoint = constants.get(52)?[0];
    let stripe_source = *constants.get(53)?;
    let triangle = *constants.get(54)?;
    let line_strength = constants.get(59)?[0];
    let contour_phase = constants.get(61)?[0];
    let contour_source = *constants.get(38)?;
    let colors = [*constants.get(33)?, *constants.get(35)?];
    let warp = [
        base_warp * constants.get(47)?[0],
        base_warp * constants.get(49)?[0],
    ];
    let decoded = GearPatternMaterial {
        field,
        projection,
        normal_power,
        field_midpoint,
        warp,
        stripe: [stripe_source[0], stripe_source[2], triangle[0], triangle[1]],
        contour: [triangle[2], line_strength, contour_phase, 0.0],
        contour_remap: contour_source,
        colors,
    };

    let values = decoded
        .projection
        .into_iter()
        .chain([decoded.normal_power, decoded.field_midpoint])
        .chain(decoded.warp)
        .chain(decoded.stripe)
        .chain(decoded.contour)
        .chain(decoded.contour_remap)
        .chain(decoded.colors.into_iter().flatten());
    (values.clone().all(f32::is_finite)
        && (1.0..=32.0).contains(&normal_power)
        && projection[0].abs() > 0.0001
        && projection[1].abs() > 0.0001
        && (0.0..=1.0).contains(&field_midpoint)
        && warp.into_iter().all(|value| value.abs() <= 8.0)
        && decoded.stripe[0].abs() > 0.0001
        && decoded.stripe[3] < 0.0
        && line_strength > 0.0
        && contour_source[1] < 0.0
        && colors
            .iter()
            .flatten()
            .all(|value| (0.0..=4.0).contains(value)))
    .then_some(decoded)
}

fn texture_is_srgb(texture: TagHash) -> bool {
    Texture::validated_descriptor_d2(texture)
        .map(|desc| format!("{:?}", desc.format).contains("Srgb"))
        .unwrap_or(false)
}

fn texture_is_single_channel(texture: TagHash) -> bool {
    Texture::validated_descriptor_d2(texture)
        .map(|desc| {
            let format = format!("{:?}", desc.format);
            format.contains("Bc4") || format.contains("R8Unorm")
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TexturelessFlatMaterial {
    color: [f32; 4],
    roughness: f32,
    metalness: f32,
}

/// Decode Marathon's textureless weapon-panel shader family.
///
/// The TFX program sends `frame_color` to PS output 2 and its component sum to
/// output 3. The compiled pixel shader compares that sum with cbuffer[4].x;
/// negative `frame_color` (the Pattern's `-1` sentinel) selects cbuffer[1] as
/// the authored base colour. G-buffer target 2 stores roughness 0.5 and the
/// authored metal value from cbuffer[36].x. This is the same branch the game
/// executes, rather than a weapon/skin-specific colour substitution.
fn textureless_flat_material_for_technique(technique: TagHash) -> Option<TexturelessFlatMaterial> {
    let Some(entry) = package_manager().get_entry(technique) else {
        return None;
    };
    let Ok(data) = package_manager().read_tag(technique) else {
        return None;
    };
    if texture_bindings_for_technique(&entry, &data)
        .into_iter()
        .any(|binding| binding.stage == "PS")
    {
        return None;
    }
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    let sends_frame_color = pixel.bytecode.expressions.iter().any(|expression| {
        expression.target == "output[2]"
            && expression.expression.contains("object_channel(0xC9A5E5AC)")
    });
    let sends_selector = pixel.bytecode.expressions.iter().any(|expression| {
        expression.target == "output[3]"
            && expression.expression.contains("object_channel(0xC9A5E5AC)")
    });
    if sends_frame_color && sends_selector {
        let color = *pixel.inline_constants.get(1)?;
        let epsilon = pixel.inline_constants.get(4)?.first().copied()?;
        if !color.iter().all(|value| value.is_finite())
            || !color[..3].iter().all(|value| (0.0..=1.0).contains(value))
            || !(0.0..=0.001).contains(&epsilon)
        {
            return None;
        }
        let metalness = pixel
            .inline_constants
            .get(36)
            .map(|value| value[0].clamp(0.0, 1.0))
            .unwrap_or(0.0);
        return Some(TexturelessFlatMaterial {
            color,
            roughness: 0.5,
            metalness,
        });
    }

    // Animated textureless panels can build their base colour through a TFX
    // spline/gradient instead of storing one fixed inline vector. Evaluate
    // every authored object-channel dependency at the preview default (zero),
    // then consume the proven colour output. Syntax Disrupt resolves through
    // this path to the gradient's [0.0, 0.4, 1.0, 1.0] endpoint.
    let color_target = pixel.bytecode.expressions.iter().find(|expression| {
        expression.target == "output[0]"
            && expression.expression.contains("gradient4_const")
            && expression.expression.contains("object_channel(")
    })?;
    let object_channels = pixel
        .bytecode
        .ops
        .iter()
        .filter(|op| op.name == "push_object_channel")
        .filter_map(|op| {
            let hash = op.detail.strip_prefix("0x")?;
            Some((u32::from_str_radix(hash, 16).ok()?, [0.0; 4]))
        })
        .collect::<std::collections::HashMap<_, _>>();
    if object_channels.is_empty() {
        return None;
    }
    let (_bindings, expressions) = interpret_tfx_stack_with_object_channels(
        &pixel.bytecode.ops,
        &pixel.constants,
        &object_channels,
    );
    let color = expressions
        .iter()
        .find(|expression| expression.target == color_target.target)
        .and_then(|expression| expression.value)?;
    if !color.iter().all(|value| value.is_finite())
        || !color.iter().all(|value| (0.0..=1.0).contains(value))
        || color[3] <= 0.001
    {
        return None;
    }
    Some(TexturelessFlatMaterial {
        color,
        roughness: 0.5,
        metalness: 0.0,
    })
}

/// Decode Marathon's textureless procedural-surface pass.
///
/// Updated Goliath geometry may place these parts outside the ordinary G-buffer
/// stage. The compiled shader's material block is stable relative to its base
/// colour: metalness is +17 vectors and roughness is +21 vectors. The preceding
/// epsilon vector identifies the block without relying on a weapon, skin, tag,
/// or absolute constant index.
fn procedural_surface_material_for_technique(
    technique: TagHash,
) -> Option<TexturelessFlatMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    if render_state_for_technique(technique).blend != Some(8) {
        return None;
    }

    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    let (color_index, color) =
        pixel
            .inline_constants
            .windows(2)
            .enumerate()
            .find_map(|(index, pair)| {
                let epsilon = pair[0];
                let color = pair[1];
                let epsilon_marker = (0.0001..=0.01).contains(&epsilon[0])
                    && epsilon[1..].iter().all(|value| value.abs() < 0.0001);
                let valid_color = color[..3]
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                    && color[3].abs() < 0.0001;
                let (min, max) = color[..3]
                    .iter()
                    .copied()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                        (min.min(value), max.max(value))
                    });
                (epsilon_marker && valid_color && max - min > 0.1 && max > 0.08)
                    .then_some((index + 1, [color[0], color[1], color[2], 1.0]))
            })?;
    let metalness = pixel.inline_constants.get(color_index + 17)?[0];
    let roughness = pixel.inline_constants.get(color_index + 21)?[0];
    if !metalness.is_finite()
        || !roughness.is_finite()
        || !(0.0..=1.0).contains(&metalness)
        || !(0.02..=1.0).contains(&roughness)
    {
        return None;
    }

    Some(TexturelessFlatMaterial {
        color,
        roughness,
        metalness,
    })
}

pub(crate) const WEAPON_MOD_AGE_CHANNEL: u32 = 0x138D_E801;
pub(crate) const UNIQUE_ID_CHANNEL: u32 = 0xD358_3E54;
const WEAPON_MOD_SCRATCHES_PROJECTION_AGE_DELTA: usize = 10;
const WEAPON_MOD_SCRATCHES_REMAP_BASE_AGE_DELTA: usize = 9;
const WEAPON_MOD_SCRATCHES_REMAP_SCALE_AGE_DELTA: usize = 8;

fn weapon_surface_condition_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<WeaponSurfaceConditionMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;

    // Compiled common-weapon body ABI. These object-channel expressions are
    // the stable semantic signature; register numbers move between shader
    // permutations, so locate the condition gate first and read relative rows.
    let gate = pixel.bytecode.expressions.iter().find(|expression| {
        expression.expression.contains("object_channel(0xA590EEC6)")
            && expression.expression.contains("object_channel(0x714FE9CA)")
    })?;
    if !pixel.bytecode.expressions.iter().any(|expression| {
        expression.expression.contains("object_channel(0xD4FB5E33)")
            && expression.expression.contains("object_channel(0xEAA8E3CF)")
    }) {
        return None;
    }
    let gate_index = gate
        .target
        .strip_prefix("output[")?
        .strip_suffix(']')?
        .parse::<usize>()
        .ok()?;
    let row = |delta: isize| {
        let index = gate_index.checked_add_signed(delta)?;
        pixel.inline_constants.get(index).copied()
    };
    let projection_target = format!("output[{}]", gate_index.checked_sub(4)?);
    let projection_expression = pixel
        .bytecode
        .expressions
        .iter()
        .find(|expression| expression.target == projection_target)?;
    let projection_constant = projection_expression
        .expression
        .split("constant[")
        .nth(1)?
        .split(']')
        .next()?
        .parse::<usize>()
        .ok()?;
    let projection = *pixel.constants.get(projection_constant)?;
    if projection[0] <= 0.0 || projection[1] <= 0.0 {
        return None;
    }
    let breakup = bindings
        .iter()
        .find(|binding| binding.stage == "PS" && binding.slot == 6)?
        .tag;
    let response = bindings
        .iter()
        .find(|binding| binding.stage == "PS" && binding.slot == 2)?
        .tag;
    let detail = bindings
        .iter()
        .find(|binding| binding.stage == "PS" && binding.slot == 4)?
        .tag;

    let detail_projection = row(-70)?;
    let detail_exponent = row(-71)?[0];
    let detail_roughness = row(14)?[0];
    let detail_remap = [row(15)?[0], row(15)?[1]];
    if !detail_projection.into_iter().all(f32::is_finite)
        || detail_projection[0] <= 0.0
        || detail_projection[1] <= 0.0
        || !detail_exponent.is_finite()
        || detail_exponent <= 0.0
        || !detail_roughness.is_finite()
        || !(0.0..=1.0).contains(&detail_roughness)
        || !detail_remap.into_iter().all(f32::is_finite)
    {
        return None;
    }

    Some(WeaponSurfaceConditionMaterial {
        response,
        detail,
        breakup,
        detail_projection,
        detail_exponent,
        detail_roughness,
        detail_remap,
        projection,
        phase: row(-8)?[0],
        triangle: [row(-7)?[0], row(-6)?[0], row(-5)?[0], row(-5)?[1]],
        orientation: [row(-2)?[0], row(-2)?[1]],
        albedo: [row(3)?[0], row(4)?[0], row(6)?[0]],
        roughness: 0.9,
        normal_flatten: row(5)?[0],
    })
}

/// Marathon's common weapon-part shader exposes extra surface inputs at PS
/// t5..t7. Inventory rarities share one Pattern and texture set. Runtime sends
/// the authored tier (1 Enhanced, 2 Deluxe, 3 Superior) through object channel
/// 0x138DE801; three TFX outputs select progressively cleaner shader branches.
/// Shader permutations shift every cbuffer register, so output indices are
/// discovered from the object-channel expressions instead of being hardcoded.
fn weapon_mod_wear_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<WeaponModConditionMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    let output_index = |target: &str| {
        target
            .strip_prefix("output[")?
            .strip_suffix(']')?
            .parse::<usize>()
            .ok()
    };
    let age_channel = format!("object_channel(0x{WEAPON_MOD_AGE_CHANNEL:08X})");
    let unique_channel = format!("object_channel(0x{UNIQUE_ID_CHANNEL:08X})");
    let age_outputs = pixel
        .bytecode
        .expressions
        .iter()
        .filter(|expression| expression.expression.contains(&age_channel))
        .filter_map(|expression| {
            Some((
                output_index(&expression.target)?,
                expression.target.as_str(),
            ))
        })
        .collect::<Vec<_>>();
    let unique_outputs = pixel
        .bytecode
        .expressions
        .iter()
        .filter(|expression| expression.expression.contains(&unique_channel))
        .filter_map(|expression| {
            expression
                .target
                .starts_with("output[")
                .then_some(expression.target.as_str())
        })
        .collect::<Vec<_>>();
    let [
        (first_age_output, first_age_target),
        (_, middle_age_target),
        (last_age_output, last_age_target),
    ] = age_outputs.as_slice()
    else {
        return None;
    };
    let [grime_projection_target, damage_projection_target] = unique_outputs.as_slice() else {
        return None;
    };

    let output_value = |expressions: &[crate::material::TfxExpressionPreview], target: &str| {
        expressions
            .iter()
            .find(|expression| expression.target == target)
            .and_then(|expression| expression.value)
    };
    let valid_projection = |projection: [f32; 4]| {
        projection.iter().all(|value| value.is_finite())
            && projection[0].abs() > f32::EPSILON
            && projection[1].abs() > f32::EPSILON
    };

    // Unique-id normally shifts wear patches per instance. Zero keeps preview
    // deterministic while retaining authored scales. Age 1 also gives exact
    // Enhanced condition controls for same evaluation pass.
    let enhanced_channels = std::collections::HashMap::from([
        (UNIQUE_ID_CHANNEL, [0.0; 4]),
        (WEAPON_MOD_AGE_CHANNEL, [1.0; 4]),
    ]);
    let (_bindings, enhanced_expressions) = interpret_tfx_stack_with_object_channels(
        &pixel.bytecode.ops,
        &pixel.constants,
        &enhanced_channels,
    );
    // Shader cb0 starts from the technique's inline constant block; TFX pops
    // overwrite selected registers (including 24, 39, 40, 42, and 49).
    // The bytecode literal pool is unrelated and must never be used as cb0.
    let scratches_projection = *pixel
        .inline_constants
        .get(first_age_output.checked_sub(WEAPON_MOD_SCRATCHES_PROJECTION_AGE_DELTA)?)
        .filter(|projection| valid_projection(**projection))?;
    let scratches_remap_base = *pixel
        .inline_constants
        .get(first_age_output.checked_sub(WEAPON_MOD_SCRATCHES_REMAP_BASE_AGE_DELTA)?)?;
    let scratches_remap_scale = *pixel
        .inline_constants
        .get(first_age_output.checked_sub(WEAPON_MOD_SCRATCHES_REMAP_SCALE_AGE_DELTA)?)?;
    let condition_blend = pixel.inline_constants.get(last_age_output + 1)?[0];
    if scratches_remap_base
        .into_iter()
        .chain(scratches_remap_scale)
        .chain([condition_blend])
        .any(|value| !value.is_finite())
    {
        return None;
    }
    let grime_projection = output_value(&enhanced_expressions, grime_projection_target)
        .filter(|projection| valid_projection(*projection))?;
    let damage_projection = output_value(&enhanced_expressions, damage_projection_target)
        .filter(|projection| valid_projection(*projection))?;
    let unique_channels = std::collections::HashMap::from([
        (UNIQUE_ID_CHANNEL, [1.0; 4]),
        (WEAPON_MOD_AGE_CHANNEL, [1.0; 4]),
    ]);
    let (_bindings, unique_expressions) = interpret_tfx_stack_with_object_channels(
        &pixel.bytecode.ops,
        &pixel.constants,
        &unique_channels,
    );
    let unique_grime_projection = output_value(&unique_expressions, grime_projection_target)
        .filter(|projection| valid_projection(*projection))?;
    let unique_damage_projection = output_value(&unique_expressions, damage_projection_target)
        .filter(|projection| valid_projection(*projection))?;
    let grime_projection_unique_delta =
        std::array::from_fn(|axis| unique_grime_projection[axis] - grime_projection[axis]);
    let damage_projection_unique_delta =
        std::array::from_fn(|axis| unique_damage_projection[axis] - damage_projection[axis]);

    let mut condition_controls = [[0.0_f32; 3]; 3];
    for (index, tier) in [1.0_f32, 2.0, 3.0].into_iter().enumerate() {
        let channels = std::collections::HashMap::from([
            (UNIQUE_ID_CHANNEL, [0.0; 4]),
            (WEAPON_MOD_AGE_CHANNEL, [tier; 4]),
        ]);
        let (_bindings, expressions) = interpret_tfx_stack_with_object_channels(
            &pixel.bytecode.ops,
            &pixel.constants,
            &channels,
        );
        for (control, target) in [*first_age_target, *middle_age_target, *last_age_target]
            .into_iter()
            .enumerate()
        {
            let value = output_value(&expressions, target)?[0];
            if !value.is_finite() {
                return None;
            }
            condition_controls[index][control] = value.clamp(0.0, 1.0);
        }
    }

    let texture_at = |slot| {
        bindings
            .iter()
            .find(|binding| binding.stage == "PS" && binding.slot == slot)
            .map(|binding| binding.tag)
            .filter(|texture| local_surface_texture_candidate(*texture, technique))
    };
    Some(WeaponModConditionMaterial {
        scratches: texture_at(5)?,
        grime: texture_at(6)?,
        damage: texture_at(7)?,
        scratches_projection,
        grime_projection,
        damage_projection,
        grime_projection_unique_delta,
        damage_projection_unique_delta,
        scratches_remap_base,
        scratches_remap_scale,
        condition_blend: condition_blend.clamp(0.0, 1.0),
        condition_controls,
        unique_id: 0.0,
        rarity: None,
    })
}

fn promote_auxiliary_preview_color(material: &mut WireframeMaterialTextures, technique: TagHash) {
    let local_color = material
        .aux
        .iter()
        .copied()
        .find(|texture| local_surface_texture_candidate(*texture, technique));
    let replace_common_global = material.color.is_some_and(|texture| {
        texture.pkg_id() == 0x130 && technique.pkg_id() != 0x130 && local_color.is_some()
    });
    if material.color.is_none() || replace_common_global {
        material.color = local_color;
    }

    if material.color.is_none() {
        material.color = material
            .aux
            .iter()
            .copied()
            .find(|texture| preview_mask_candidate(*texture));
    }
    if let Some(color) = material.color {
        material.aux.retain(|texture| *texture != color);
    }
}

fn local_surface_texture_candidate(texture: TagHash, technique: TagHash) -> bool {
    if texture.pkg_id() != technique.pkg_id() || fallback_aux_texture(texture) {
        return false;
    }
    let Ok(desc) = Texture::validated_descriptor_d2(texture) else {
        return false;
    };
    let format = format!("{:?}", desc.format);
    desc.depth == 1
        && desc.array_size == 1
        && desc.width > 1
        && desc.height > 1
        && format.contains("Srgb")
        && !format.contains("Bc4")
}

fn preview_mask_candidate(texture: TagHash) -> bool {
    if fallback_aux_texture(texture) {
        return false;
    }
    let Ok(desc) = Texture::validated_descriptor_d2(texture) else {
        return false;
    };
    let format = format!("{:?}", desc.format);
    desc.depth == 1
        && desc.array_size == 1
        && desc.width > 1
        && desc.height > 1
        && format.contains("Bc4")
}

fn apply_material_constants(
    material: &mut WireframeMaterialTextures,
    constants: TechniqueMaterialConstants,
) {
    material.color_tint = constants
        .color_tint
        .map(quantize_color_tint)
        .unwrap_or([255, 255, 255, 255]);
    material.mask_palette = constants.mask_palette;
    material.emissive_strength = constants
        .emissive_strength
        .map(|value| ((value / 4.0).clamp(0.0, 1.0) * 255.0) as u8)
        .unwrap_or(0);
}

fn quantize_color_tint(value: [f32; 4]) -> [u8; 4] {
    [
        (value[0].clamp(0.0, 1.0) * 255.0) as u8,
        (value[1].clamp(0.0, 1.0) * 255.0) as u8,
        (value[2].clamp(0.0, 1.0) * 255.0) as u8,
        (value[3].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

fn assign_material_texture(
    material: &mut WireframeMaterialTextures,
    texture: TagHash,
    role: MaterialTextureRole,
) {
    match role {
        MaterialTextureRole::Color => {
            material.aux.retain(|candidate| *candidate != texture);
            if !material
                .layers
                .iter()
                .any(|layer| layer.color == Some(texture))
            {
                assign_color_layer(material, texture);
            }
        }
        MaterialTextureRole::Normal => {
            material.aux.retain(|candidate| *candidate != texture);
            if !material
                .layers
                .iter()
                .any(|layer| layer.normal == Some(texture))
            {
                assign_normal_layer(material, texture);
            }
        }
        MaterialTextureRole::Control(channel) => {
            material.aux.retain(|candidate| *candidate != texture);
            material.control = Some(texture);
            material.roughness_channel = channel;
        }
        MaterialTextureRole::Aux => {
            let semantic = Some(texture) == material.color
                || Some(texture) == material.normal
                || Some(texture) == material.emissive
                || Some(texture) == material.control
                || material.layers.iter().any(|layer| {
                    layer.color == Some(texture)
                        || layer.normal == Some(texture)
                        || layer.emissive == Some(texture)
                });
            if !semantic && !material.aux.contains(&texture) {
                material.aux.push(texture);
            }
        }
    }
}

fn assign_color_layer(material: &mut WireframeMaterialTextures, texture: TagHash) {
    if material.color.is_none() {
        material.color = Some(texture);
    }

    material.layers.push(WireframeMaterialLayer {
        color: Some(texture),
        ..Default::default()
    });
}

fn assign_normal_layer(material: &mut WireframeMaterialTextures, texture: TagHash) {
    if material.normal.is_none() {
        material.normal = Some(texture);
    }

    if let Some(layer) = material
        .layers
        .iter_mut()
        .find(|layer| layer.normal.is_none())
    {
        layer.normal = Some(texture);
    } else {
        material.layers.push(WireframeMaterialLayer {
            normal: Some(texture),
            ..Default::default()
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaterialTextureRole {
    Color,
    Normal,
    Control(u8),
    Aux,
}

fn material_texture_role(
    binding: TechniqueTextureBinding,
    normal_slot: Option<u32>,
    control_slot: Option<(u32, u8)>,
) -> MaterialTextureRole {
    if let Some((control_slot, channel)) = control_slot
        && binding.slot == control_slot
    {
        return MaterialTextureRole::Control(channel);
    }
    let format = texture_preview_format(binding.tag);
    if binding.slot == 0 && !format.contains("Bc4") {
        return MaterialTextureRole::Color;
    }
    let rank = texture_preview_rank(binding.tag);
    if rank.format >= 70 {
        return MaterialTextureRole::Aux;
    }

    match binding.slot {
        slot if Some(slot) == normal_slot => MaterialTextureRole::Normal,
        0 => MaterialTextureRole::Color,
        _ => MaterialTextureRole::Aux,
    }
}

fn material_color_texture_slot_for_technique(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<u32> {
    let pixel_shader = package_manager()
        .get_entry(technique)
        .zip(package_manager().read_tag(technique).ok())
        .and_then(|(entry, data)| MaterialTagPreview::load(&entry, &data))
        .and_then(|preview| {
            let MaterialPreviewKind::Technique(preview) = preview.kind;
            preview
                .stages
                .iter()
                .find(|stage| stage.stage == "PS")
                .and_then(|stage| stage.shader)
        });

    let exact = match pixel_shader {
        // B860 generates skin colour from c121/c122/c123. t1 is a shared
        // cellular response LUT, not model albedo. Keep t0 as the material
        // key so local normal/AO resources remain attached; mode 41 replaces
        // its scalar preview colour in WGSL.
        Some(TagHash(0x80A9B860)) => Some(0),
        // Compiled hair/fiber shaders bind shared environment lookup at t0
        // and local strand albedo at t1. Treating t0 as model color discards
        // every authored hair texture on Thief/Vandal-style three-part shells.
        Some(TagHash(0x80A4840A)) | Some(TagHash(0x80B073D2)) | Some(TagHash(0x80B08209)) => {
            Some(1)
        }
        _ => None,
    };
    if exact.is_some() {
        return exact;
    }

    // Scalar t0 cannot supply albedo. Generated procedural runner shaders
    // place authored sRGB color later in their resource table. Prefer it over
    // rendering the scalar selector as gray.
    let slot0_is_scalar = bindings
        .iter()
        .any(|binding| binding.slot == 0 && texture_preview_format(binding.tag).contains("Bc4"));
    if !slot0_is_scalar {
        return None;
    }
    bindings
        .iter()
        .filter(|binding| texture_is_srgb(binding.tag) && !fallback_aux_texture(binding.tag))
        .min_by_key(|binding| (binding.tag.pkg_id() != technique.pkg_id(), binding.slot))
        .map(|binding| binding.slot)
}

fn material_control_texture_slot(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
    normal_slot: Option<u32>,
) -> Option<(u32, u8)> {
    // Face/skin surface 80A9B85C binds its packed region selector at t2.
    // t3 is a shared response field, while t5 is the tangent-space normal.
    // The compact-family heuristic otherwise mistakes t3 for the dye selector.
    match material_pixel_shader(technique) {
        Some(TagHash(0x80A9B85C)) => return Some((2, 4)),
        Some(TagHash(0x80A9C430)) => return Some((1, 4)),
        Some(TagHash(0x80A9C65C)) => return Some((1, 4)),
        Some(TagHash(0x80A9CBEE)) => return Some((2, 4)),
        _ => {}
    }
    let max_slot = bindings.iter().map(|binding| binding.slot).max()?;
    let usable = |slot| {
        Some(slot) != normal_slot
            && bindings.iter().any(|binding| {
                binding.slot == slot
                    && !fallback_aux_texture(binding.tag)
                    && material_control_texture_candidate(binding.tag)
            })
    };
    let usable_multichannel = |slot| {
        Some(slot) != normal_slot
            && bindings.iter().any(|binding| {
                binding.slot == slot
                    && !fallback_aux_texture(binding.tag)
                    && material_control_texture_candidate(binding.tag)
                    && !texture_preview_format(binding.tag).contains("Bc4")
            })
    };

    if max_slot >= 10 && usable_multichannel(3) {
        return Some((3, 3));
    }
    if max_slot <= 4 && normal_slot == Some(2) && usable(1) {
        return Some((1, 1));
    }
    if (5..10).contains(&max_slot) && usable_multichannel(3) {
        return Some((3, 4));
    }
    if (5..10).contains(&max_slot) && usable_multichannel(1) {
        // Compact/full runner gear permutations bind their local packed
        // region/surface selector at t1. t3 is often the shared 80A613F5
        // lighting ramp and must never drive GearDye region IDs.
        return Some((1, 4));
    }
    None
}

fn texture_preview_format(texture: TagHash) -> String {
    Texture::validated_descriptor_d2(texture)
        .map(|desc| format!("{:?}", desc.format))
        .unwrap_or_default()
}

fn material_control_texture_candidate(texture: TagHash) -> bool {
    let Ok(desc) = Texture::validated_descriptor_d2(texture) else {
        return false;
    };
    let format = format!("{:?}", desc.format);
    desc.depth == 1
        && desc.array_size == 1
        && desc.width > 1
        && desc.height > 1
        && !format.contains("Srgb")
}

fn material_normal_texture_slot_for_technique(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<u32> {
    // Generated runner shaders do not consistently place their base tangent
    // normal in the last linear 2D slot. These two audited families append
    // procedural/detail fields after the base normal:
    //
    // 80A9B75C: t3 base normal, t4 procedural 2D field, t5 3D field.
    // 80A9AFB6: t4 base normal, t3 gated detail normal, t5 procedural field.
    //
    // Resolve from compiled PS ABI before applying the generic table rule.
    let pixel_shader = material_pixel_shader(technique);
    match pixel_shader {
        // Two-mask runner surface. PS t1 is scalar coverage, t2 is scalar AO,
        // and DXIL samples t3 at mesh UV before reconstructing tangent-space Z.
        Some(TagHash(0x80A9A9D3)) => return Some(3),
        Some(TagHash(0x80A9B860)) => return Some(3),
        Some(TagHash(0x80A9B07B)) => return Some(6),
        Some(TagHash(0x80A9B855)) => return Some(4),
        Some(TagHash(0x80A9B857)) | Some(TagHash(0x80A9B85A)) => return Some(3),
        Some(TagHash(0x80A9C244)) => return Some(8),
        Some(TagHash(0x80A9AFB4)) => return Some(6),
        Some(TagHash(0x80A9B75C)) => return Some(3),
        Some(TagHash(0x80A9AFB6)) | Some(TagHash(0x80A9BBBE)) | Some(TagHash(0x80A9BBBF)) => {
            return Some(4);
        }
        Some(TagHash(0x80A9B93E))
        | Some(TagHash(0x80A9BBDD))
        | Some(TagHash(0x80A9CBAE))
        | Some(TagHash(0x80A9D6FD))
        | Some(TagHash(0x80A9DC9A))
        | Some(TagHash(0x80A9DEEC))
        | Some(TagHash(0x80A9E0FD))
        | Some(TagHash(0x80A9E4DB))
        | Some(TagHash(0x80A9E76C)) => return Some(6),
        Some(TagHash(0x80A9B610)) | Some(TagHash(0x80A9C3F6)) => return Some(7),
        Some(TagHash(0x80A9A8CF)) => return Some(7),
        Some(TagHash(0x80A9AD5D)) => return Some(9),
        Some(TagHash(0x80A9AD68)) => return Some(7),
        Some(TagHash(0x80A9A9AD)) | Some(TagHash(0x80A9A9B1)) => return Some(9),
        Some(TagHash(0x80A9B065)) | Some(TagHash(0x80A9DE77)) => return Some(6),
        Some(TagHash(0x80A9B06A)) => return Some(6),
        Some(TagHash(0x80A9B71C)) => return Some(4),
        Some(TagHash(0x80A9B85C)) => return Some(5),
        Some(TagHash(0x80A9C430)) => return Some(5),
        Some(TagHash(0x80A9C65C)) => return Some(4),
        Some(TagHash(0x80A9CBEE)) => return Some(5),
        Some(TagHash(0x80A9C27C)) => return Some(6),
        Some(TagHash(0x80A9AFBF)) => return Some(6),
        Some(TagHash(0x80AA0261)) | Some(TagHash(0x80AA0263)) => return Some(5),
        Some(TagHash(0x80AA02A7)) => return Some(4),
        Some(TagHash(0x80A9B86A)) => return Some(7),
        Some(TagHash(0x80A9D952)) => return Some(7),
        Some(TagHash(0x80A9AE19)) | Some(TagHash(0x80A9DAC9)) | Some(TagHash(0x80A9DB9A)) => {
            return Some(7);
        }
        Some(TagHash(0x80A9AFB8)) | Some(TagHash(0x80A9AFBA)) => return Some(9),
        Some(TagHash(0x80A9BD17)) | Some(TagHash(0x80A9DA29)) | Some(TagHash(0x80A9E64F)) => {
            return Some(7);
        }
        Some(TagHash(0x80A9C31E)) => return Some(5),
        Some(TagHash(0x80A9DCF8)) => return Some(6),
        Some(TagHash(0x80A9C96F)) | Some(TagHash(0x80A9D2D7)) | Some(TagHash(0x80A9D3FC)) => {
            return Some(8);
        }
        Some(TagHash(0x80A9D569)) => return Some(9),
        Some(TagHash(0x80A9F4E5)) => return Some(7),
        Some(TagHash(0x80A9F500)) => return Some(8),
        Some(TagHash(0x80A9F518)) | Some(TagHash(0x80A9F4B3)) => return Some(7),
        Some(TagHash(0x80A9F528)) => return Some(6),
        Some(TagHash(0x80A9F589)) => return Some(4),
        // Runner character permutations below t10 need explicit PS ABI slots;
        // generic material inference intentionally stays weapon-safe.
        Some(TagHash(0x80AA043B)) => return Some(5),
        Some(TagHash(0x80AA046E)) => return Some(6),
        Some(TagHash(0x80AA0498)) | Some(TagHash(0x80B14062)) => return Some(8),
        Some(TagHash(0x80B142A5)) => return Some(9),
        Some(TagHash(0x80B143A0)) => return Some(6),
        Some(TagHash(0x80B1444E)) | Some(TagHash(0x80B14701)) | Some(TagHash(0x80B14BF9)) => {
            return Some(7);
        }
        _ => {}
    }

    material_normal_texture_slot(bindings)
}

fn material_pixel_shader(technique: TagHash) -> Option<TagHash> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    preview
        .stages
        .iter()
        .find(|stage| stage.stage == "PS")?
        .shader
}

fn material_normal_texture_slot(bindings: &[TechniqueTextureBinding]) -> Option<u32> {
    let max_slot = bindings.iter().map(|binding| binding.slot).max()?;
    let usable = |slot| {
        bindings.iter().any(|binding| {
            binding.slot == slot
                && !fallback_aux_texture(binding.tag)
                && normal_surface_texture_candidate(binding.tag)
        })
    };

    // Goliath character shaders bind packed dye/material masks near the front of
    // the table and their tangent normal near the end. Compact hair shaders use
    // slot 2, while the common surface family uses slot 1. These layouts match
    // the bindings Alkahest passes unchanged to the original Tiger shaders.
    if max_slot >= 10 {
        return [12, 11, 10].into_iter().find(|slot| usable(*slot));
    }

    if max_slot <= 4 && usable(2) {
        return Some(2);
    }

    usable(1).then_some(1)
}

fn normal_surface_texture_candidate(texture: TagHash) -> bool {
    let Ok(desc) = Texture::validated_descriptor_d2(texture) else {
        return false;
    };
    let format = format!("{:?}", desc.format);
    desc.depth == 1
        && desc.array_size == 1
        && desc.width > 1
        && desc.height > 1
        && !format.contains("Srgb")
        && !format.contains("Bc4")
}

fn guessed_related_texture_role(
    texture: TagHash,
    technique: TagHash,
    material: &WireframeMaterialTextures,
) -> MaterialTextureRole {
    let Ok(desc) = Texture::validated_descriptor_d2(texture) else {
        return MaterialTextureRole::Aux;
    };

    let format = format!("{:?}", desc.format);
    if format.contains("Srgb") && fallback_color_candidate(texture, technique) {
        if material.color.is_none() {
            MaterialTextureRole::Color
        } else {
            MaterialTextureRole::Aux
        }
    } else {
        MaterialTextureRole::Aux
    }
}

fn fallback_color_candidate(texture: TagHash, technique: TagHash) -> bool {
    let Ok(desc) = Texture::validated_descriptor_d2(texture) else {
        return false;
    };
    let format = format!("{:?}", desc.format);
    let _same_package_as_technique = texture.pkg_id() == technique.pkg_id();
    desc.depth == 1
        && desc.array_size == 1
        && desc.width > 1
        && desc.height > 1
        && format.contains("Srgb")
        && !fallback_aux_texture(texture)
}

fn fallback_aux_texture(texture: TagHash) -> bool {
    texture == TagHash::new(288, 1296)
        || matches!(
            texture.0,
            0x80A60000
                | 0x80A60058
                | 0x80A60089
                | 0x80A613F5
                | 0x80A4050F
                | 0x80A46D44
                | 0x80A46D48
                | 0x80A60055
                | 0x80A6007D
                | 0x80A43539
                | 0x80B6CC6E
        )
}

fn texture_has_parent_technique(cache: &TagCache, texture: TagHash, technique: TagHash) -> bool {
    cache
        .hashes
        .get(&texture)
        .is_some_and(|scan| scan.references.contains(&technique))
}

fn find_model_technique_entries(
    cache: &TagCache,
    tag: TagHash,
    entry: &UEntryHeader,
) -> Vec<(TagHash, UEntryHeader)> {
    find_model_technique_tags(cache, tag, entry)
        .into_iter()
        .filter_map(|tag| package_manager().get_entry(tag).map(|entry| (tag, entry)))
        .unique_by(|(tag, _entry)| *tag)
        .take(128)
        .collect()
}

struct TechniqueOrderedTextureBinding {
    technique_order: usize,
    binding: TechniqueTextureBinding,
}

fn texture_bindings_from_technique(
    (technique_order, tag): (usize, TagHash),
) -> Vec<TechniqueOrderedTextureBinding> {
    let Some(entry) = package_manager().get_entry(tag) else {
        return vec![];
    };
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };

    texture_bindings_for_technique(&entry, &data)
        .into_iter()
        .map(|binding| TechniqueOrderedTextureBinding {
            technique_order,
            binding,
        })
        .collect()
}

fn texture_binding_rank(binding: TechniqueTextureBinding) -> u8 {
    let slot_rank = match binding.slot {
        0 => 0,
        2 => 20,
        5 => 50,
        10 => 60,
        _ => 100,
    };
    let stage_rank = match binding.stage {
        "PS" => 0,
        "VS" => 5,
        "GS" => 10,
        "CS" => 15,
        _ => 20,
    };
    slot_rank + stage_rank
}

fn find_model_technique_tags(cache: &TagCache, tag: TagHash, entry: &UEntryHeader) -> Vec<TagHash> {
    let Ok(data) = package_manager().read_tag(tag) else {
        let recursive = find_related_tags(cache, tag, TagSearchKind::Technique, 8)
            .into_iter()
            .map(|(tag, _entry)| tag)
            .collect_vec();
        return recursive
            .into_iter()
            .chain(find_child_technique_tags(
                cache,
                tag,
                package_manager().version.endian(),
            ))
            .unique()
            .collect();
    };
    let endian = package_manager().version.endian();

    let direct = match entry.reference {
        0x80806D44 | 0x80808635 => static_mesh_technique_tags(&data, endian),
        0x80806D30 | 0x80808620 => find_static_meshdata_owner_techniques(cache, tag),
        0x80806F07 => dynamic_model_technique_tags(&data, endian),
        0x80806EC5 => dynamic_mesh_technique_tags(&data, endian),
        CLASS_GEOMETRY_RESOURCE => geometry_resource_technique_tags(&data, endian),
        _ => vec![],
    };
    let parent_entity_resource_materials =
        find_parent_entity_resource_material_techniques(cache, tag, endian);
    let raw_current = technique_tags_in_blob(&data, endian);
    let raw_children = find_child_technique_tags(cache, tag, endian);
    let recursive = find_related_tags(cache, tag, TagSearchKind::Technique, 8)
        .into_iter()
        .map(|(tag, _entry)| tag);

    direct
        .into_iter()
        .chain(parent_entity_resource_materials)
        .chain(raw_current)
        .chain(raw_children)
        .chain(recursive)
        .unique()
        .collect()
}

fn find_parent_entity_resource_material_techniques(
    cache: &TagCache,
    tag: TagHash,
    endian: Endian,
) -> Vec<TagHash> {
    cache
        .hashes
        .get(&tag)
        .into_iter()
        .flat_map(|scan| scan.references.iter().copied())
        .filter_map(|parent| {
            let entry = package_manager().get_entry(parent)?;
            (entry.reference == CLASS_ENTITY_RESOURCE).then_some(parent)
        })
        .filter_map(|parent| package_manager().read_tag(parent).ok())
        .flat_map(|data| entity_resource_material_technique_tags(&data, tag, endian))
        .unique()
        .collect()
}

fn entity_resource_material_technique_tags(
    data: &[u8],
    referenced_model: TagHash,
    endian: Endian,
) -> Vec<TagHash> {
    // Alkahest reads SEntityResource.unk18.offset + 0x224 for model,
    // +0x3c0 for material variant map, +0x400 for material techniques.
    let Some(resource_offset) = read_u32_at(data, 0x18, endian).map(|offset| offset as usize)
    else {
        return vec![];
    };
    if resource_offset + 0x410 > data.len() {
        return vec![];
    }

    let model_matches = read_tag_at(data, resource_offset + 0x224, endian)
        .is_some_and(|model| model == referenced_model);
    let parent_references_model = candidate_tag_hashes_in_blob(data, endian)
        .into_iter()
        .any(|tag| tag == referenced_model);
    if !model_matches && !parent_references_model {
        return vec![];
    }

    read_tag_array(data, resource_offset + 0x400, endian)
        .into_iter()
        .filter(|tag| is_technique_tag(*tag))
        .unique()
        .collect()
}

fn find_static_meshdata_owner_techniques(cache: &TagCache, mesh_data_tag: TagHash) -> Vec<TagHash> {
    cache
        .hashes
        .get(&mesh_data_tag)
        .into_iter()
        .flat_map(|scan| scan.references.iter().copied())
        .filter_map(|parent| {
            let entry = package_manager().get_entry(parent)?;
            matches!(entry.reference, 0x80806D44 | 0x80808635).then_some((parent, entry))
        })
        .flat_map(|(parent, _entry)| {
            let data = package_manager().read_tag(parent).ok()?;
            Some(static_mesh_technique_tags(
                &data,
                package_manager().version.endian(),
            ))
        })
        .flatten()
        .unique()
        .collect()
}

fn find_child_technique_tags(cache: &TagCache, tag: TagHash, endian: Endian) -> Vec<TagHash> {
    let Some(scan) = cache.hashes.get(&tag) else {
        return vec![];
    };

    scan.file_hashes
        .iter()
        .map(|scanned| scanned.hash)
        .chain(
            scan.file_hashes64
                .iter()
                .filter_map(|scanned| tag64_to_hash32(scanned.hash)),
        )
        .filter(|child| *child != tag)
        .unique()
        .take(TECHNIQUE_SCAN_CHILD_LIMIT)
        .flat_map(|child| {
            let Some(entry) = package_manager().get_entry(child) else {
                return vec![];
            };
            if is_technique_tag(child) {
                return vec![child];
            }

            let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
            if !tag_type.is_tag() {
                return vec![];
            }

            package_manager()
                .read_tag(child)
                .map(|data| technique_tags_in_blob(&data, endian))
                .unwrap_or_default()
        })
        .unique()
        .collect()
}

fn triangle_indices_to_line_indices(indices: &[u32]) -> Vec<u32> {
    let mut lines = Vec::with_capacity(indices.len().saturating_mul(2));
    for tri in indices.chunks_exact(3) {
        lines.extend_from_slice(&[tri[0], tri[1], tri[1], tri[2], tri[2], tri[0]]);
    }

    lines
}

fn parse_model_wireframe(
    tag: TagHash,
    entry: &UEntryHeader,
) -> Option<(MeshSourcePreview, WireframePreview)> {
    cache::mesh(tag, entry, || decode_model_wireframe(tag, entry))
}

fn decode_model_wireframe(
    tag: TagHash,
    entry: &UEntryHeader,
) -> Option<(MeshSourcePreview, WireframePreview)> {
    match entry.reference {
        0x80806D44 => parse_static_mesh_wireframe(tag),
        0x80806D30 => {
            let data = package_manager().read_tag(tag).ok()?;
            parse_static_mesh_data_wireframe(&data)
        }
        0x80806F07 => parse_dynamic_model_wireframe(tag),
        0x80806EC5 => {
            let data = package_manager().read_tag(tag).ok()?;
            parse_dynamic_mesh_wireframe(&data)
        }
        CLASS_GEOMETRY_RESOURCE => parse_geometry_resource_wireframe(tag),
        _ => None,
    }
}

fn parse_static_mesh_wireframe(tag: TagHash) -> Option<(MeshSourcePreview, WireframePreview)> {
    let data = package_manager().read_tag(tag).ok()?;
    let endian = package_manager().version.endian();
    let mesh_data_tag = TagHash(read_u32(data.get(0x8..0xc)?, endian));
    let mesh_data = package_manager().read_tag(mesh_data_tag).ok()?;
    parse_static_mesh_data_wireframe(&mesh_data)
}

fn parse_static_mesh_data_wireframe(data: &[u8]) -> Option<(MeshSourcePreview, WireframePreview)> {
    let endian = package_manager().version.endian();
    let parts = read_static_mesh_parts(data, 0x18, endian);
    let groups = read_static_mesh_groups(data, 0x8, endian);
    let buffers = read_static_buffer_tuples(data, 0x28, endian);
    let group = groups
        .iter()
        .filter(|group| {
            group.render_stage == crate::render::stage::MarathonRenderStage::GenerateGbuffer.raw()
        })
        .filter_map(|group| {
            parts
                .get(group.part_index as usize)
                .map(|part| (group, part))
        })
        .min_by_key(|(_group, part)| (lod_selection_rank(part.lod_category), part.index_start))
        .map(|(group, _part)| group)
        .or_else(|| {
            groups
                .iter()
                .filter_map(|group| {
                    parts
                        .get(group.part_index as usize)
                        .map(|part| (group, part))
                })
                .min_by_key(|(_group, part)| {
                    (lod_selection_rank(part.lod_category), part.index_start)
                })
                .map(|(group, _part)| group)
        })?;
    let part = parts.get(group.part_index as usize)?;
    let buffers = buffers.get(part.buffer_index as usize)?;

    let uv_transform = read_static_uv_transform(data, endian);
    let source = MeshSourcePreview {
        kind: "static mesh data",
        buffer_index: part.buffer_index as usize,
        technique: None,
        index_start: part.index_start,
        index_count: part.index_count,
        primitive_type: part.primitive_type,
        raw_lod_category: part.lod_category,
        lod_category: lod_preview_value(part.lod_category),
        input_layout_index: Some(group.input_layout_index),
        index_buffer: buffers.index_buffer,
        vertex0_buffer: buffers.vertex0_buffer,
        vertex1_buffer: buffers.vertex1_buffer,
        color_buffer: buffers.color_buffer,
        uv_transform,
        shader_constants: shader_constants_from_uv_transform(uv_transform),
    };
    let wireframe = build_wireframe_from_refs(
        &[source.vertex0_buffer, source.vertex1_buffer],
        source.index_buffer,
        &[PreviewIndexRange {
            range: source.index_start as usize..(source.index_start + source.index_count) as usize,
            primitive_type: source.primitive_type,
            raw_lod_category: Some(source.raw_lod_category),
            render_stage: None,
            technique: source.technique,
            gear_dye_change_color_index: None,
            authored_draw: None,
        }],
        source.input_layout_index,
    )?;

    Some((source, wireframe))
}

fn parse_dynamic_model_wireframe(tag: TagHash) -> Option<(MeshSourcePreview, WireframePreview)> {
    let data = package_manager().read_tag(tag).ok()?;
    let endian = package_manager().version.endian();
    let mesh_array = read_array(data.get(..)?, 0x10, 0x80, endian)?;
    let (mut source, wireframe) =
        parse_dynamic_mesh_wireframe(mesh_array.chunks_exact(0x80).next()?)?;
    source.uv_transform = read_dynamic_model_uv_transform(&data, endian);
    source.shader_constants = shader_constants_from_uv_transform(source.uv_transform);
    Some((source, wireframe))
}

fn parse_dynamic_mesh_wireframe(data: &[u8]) -> Option<(MeshSourcePreview, WireframePreview)> {
    let endian = package_manager().version.endian();
    let parts = read_dynamic_mesh_parts(data, 0x20, endian);
    let part = parts
        .iter()
        .min_by_key(|part| (lod_selection_rank(part.lod_category), part.index_start))
        .or_else(|| parts.first())?;
    let source = MeshSourcePreview {
        kind: "dynamic mesh",
        buffer_index: 0,
        technique: Some(part.technique),
        index_start: part.index_start,
        index_count: part.index_count,
        primitive_type: part.primitive_type,
        raw_lod_category: part.lod_category,
        lod_category: lod_preview_value(part.lod_category),
        input_layout_index: first_dynamic_input_layout(data),
        index_buffer: TagHash(read_u32(data.get(0x10..0x14)?, endian)),
        vertex0_buffer: TagHash(read_u32(data.get(0x0..0x4)?, endian)),
        vertex1_buffer: TagHash(read_u32(data.get(0x4..0x8)?, endian)),
        color_buffer: TagHash(read_u32(data.get(0x14..0x18)?, endian)),
        uv_transform: None,
        shader_constants: vec![],
    };
    let wireframe = build_wireframe_from_refs(
        &[source.vertex0_buffer, source.vertex1_buffer],
        source.index_buffer,
        &[PreviewIndexRange {
            range: source.index_start as usize..(source.index_start + source.index_count) as usize,
            primitive_type: source.primitive_type,
            raw_lod_category: Some(source.raw_lod_category),
            render_stage: None,
            technique: source.technique,
            gear_dye_change_color_index: None,
            authored_draw: None,
        }],
        source.input_layout_index,
    )?;

    Some((source, wireframe))
}

fn parse_geometry_resource_wireframe(
    tag: TagHash,
) -> Option<(MeshSourcePreview, WireframePreview)> {
    let data = package_manager().read_tag(tag).ok()?;
    let endian = package_manager().version.endian();
    let uv_transform = read_geometry_uv_transform(&data, endian);
    let position_transform = read_geometry_position_transform(&data, endian);
    let ranges = geometry_primary_index_ranges(&data, endian);
    let range = ranges.first();
    let index_count = ranges
        .iter()
        .map(|range| range.index_count)
        .fold(0u32, u32::saturating_add);

    let mesh_records = scan_arrays(&data, endian)
        .into_iter()
        .filter(|array| array.class == CLASS_GEOMETRY_BUFFER_SET)
        .flat_map(|array| array_records(&data, array, 0x80))
        .collect_vec();

    if let Some(mesh) = mesh_records.first() {
        let vertex0_buffer = read_tag_at(mesh, 0x0, endian)?;
        let vertex1_buffer = read_tag_at(mesh, 0x4, endian).unwrap_or(TagHash(0));
        let index_buffer = read_tag_at(mesh, 0x10, endian)?;
        let input_layout_index = geometry_buffer_set_input_layout_id(mesh);
        let index_ranges = index_ranges_from_geometry_ranges(&ranges);
        let source = MeshSourcePreview {
            kind: "geometry resource",
            buffer_index: 0,
            technique: range.map(|range| range.technique),
            index_start: range.map(|range| range.index_start).unwrap_or(0),
            index_count: if index_count == 0 {
                u32::MAX
            } else {
                index_count
            },
            primitive_type: range.map(|range| range.primitive_type).unwrap_or(0),
            raw_lod_category: range.map(|range| range.lod_category).unwrap_or(0),
            lod_category: range
                .map(|range| lod_preview_value(range.lod_category))
                .unwrap_or(0),
            input_layout_index,
            index_buffer,
            vertex0_buffer,
            vertex1_buffer,
            color_buffer: TagHash(0),
            uv_transform,
            shader_constants: shader_constants_from_uv_transform(uv_transform),
        };
        let mut wireframe = build_wireframe_from_refs(
            &[source.vertex0_buffer, source.vertex1_buffer],
            source.index_buffer,
            index_ranges.as_slice(),
            source.input_layout_index,
        )?;
        wireframe.authored_inputs = vec![authored_geometry_input(
            tag,
            mesh,
            &data,
            endian,
            position_transform,
            uv_transform,
        )];
        for range in &mut wireframe.material_ranges {
            range.authored_source = Some(0);
        }
        wireframe.authored_shadow_ranges = geometry_authored_stage_ranges(
            &data,
            endian,
            source.index_buffer,
            wireframe.vertices.len(),
            crate::render::stage::MarathonRenderStage::ShadowGenerate,
        );
        if let Some(transform) = position_transform {
            apply_geometry_position_transform(&mut wireframe, transform);
        }

        return Some((source, wireframe));
    }

    let index_buffer = read_tag_at(&data, 0x140, endian)?;
    let vertex0_buffer = read_tag_at(&data, 0x130, endian)?;
    let vertex1_buffer = read_tag_at(&data, 0x134, endian).unwrap_or(TagHash(0));
    let input_layout_index = geometry_primary_layout_id(&data, endian);
    let index_ranges = index_ranges_from_geometry_ranges(&ranges);
    let source = MeshSourcePreview {
        kind: "geometry resource",
        buffer_index: 0,
        technique: range.map(|range| range.technique),
        index_start: range.map(|range| range.index_start).unwrap_or(0),
        index_count: if index_count == 0 {
            u32::MAX
        } else {
            index_count
        },
        primitive_type: range.map(|range| range.primitive_type).unwrap_or(0),
        raw_lod_category: range.map(|range| range.lod_category).unwrap_or(0),
        lod_category: range
            .map(|range| lod_preview_value(range.lod_category))
            .unwrap_or(0),
        input_layout_index,
        index_buffer,
        vertex0_buffer,
        vertex1_buffer,
        color_buffer: TagHash(0),
        uv_transform,
        shader_constants: shader_constants_from_uv_transform(uv_transform),
    };
    let mut wireframe = build_wireframe_from_refs(
        &[source.vertex0_buffer, source.vertex1_buffer],
        source.index_buffer,
        index_ranges.as_slice(),
        source.input_layout_index,
    )?;
    let stage_layouts = geometry_render_stage_abi(&data, endian)
        .map(|abi| {
            abi.input_layouts
                .iter()
                .copied()
                .enumerate()
                .map(|(raw_stage, layout_id)| AuthoredStageInputLayout {
                    raw_stage: raw_stage as u8,
                    layout_id,
                    descriptor: authored_input_layout_descriptor(layout_id),
                })
                .collect_vec()
        })
        .unwrap_or_default();
    wireframe.authored_inputs = vec![AuthoredGeometryInput {
        geometry: tag,
        vertex_streams: [
            authored_vertex_stream_ref(0, source.vertex0_buffer),
            authored_vertex_stream_ref(1, source.vertex1_buffer),
        ]
        .into_iter()
        .flatten()
        .collect(),
        color_buffer: None,
        skinning_buffer: None,
        index_buffer: authored_index_buffer_ref(source.index_buffer),
        stage_layouts,
        position_transform,
        uv_transform,
        attachment_pose: None,
    }];
    for range in &mut wireframe.material_ranges {
        range.authored_source = Some(0);
    }
    wireframe.authored_shadow_ranges = geometry_authored_stage_ranges(
        &data,
        endian,
        source.index_buffer,
        wireframe.vertices.len(),
        crate::render::stage::MarathonRenderStage::ShadowGenerate,
    );
    if let Some(transform) = position_transform {
        apply_geometry_position_transform(&mut wireframe, transform);
    }

    Some((source, wireframe))
}

fn build_wireframe_from_refs(
    vertex_tags: &[TagHash],
    index_tag: TagHash,
    index_ranges: &[PreviewIndexRange],
    input_layout_index: Option<u8>,
) -> Option<WireframePreview> {
    let mut vertex_previews = vertex_tags
        .iter()
        .copied()
        .filter_map(|vertex_tag| {
            let vertex_entry = package_manager().get_entry(vertex_tag)?;
            let vertex_header = package_manager().read_tag(vertex_tag).ok()?;
            let vertex_preview =
                load_vertex_buffer_preview_for_tag(vertex_tag, &vertex_entry, &vertex_header)
                    .ok()?;
            Some((vertex_tag, vertex_preview))
        })
        .collect_vec();

    let (vertex_tag, mut wireframe) = vertex_previews
        .iter_mut()
        .find_map(|(vertex_tag, preview)| Some((*vertex_tag, preview.wireframe.take()?)))?;

    wireframe.rigid_indices = input_layout_rigid_indices(
        &vertex_previews,
        input_layout_index,
        wireframe.vertices.len(),
    );

    if input_layout_index == Some(13)
        && let Some((_position_tag, InputLayoutFormat::R32G32B32A32Float, positions)) =
            input_layout_vectors(
                &vertex_previews,
                input_layout_index,
                SEMANTIC_POSITION,
                wireframe.vertices.len(),
            )
    {
        wireframe.vertices = positions
            .into_iter()
            .map(|position| [position[0], position[1], position[2]])
            .collect();
        wireframe.procedural_positions = Some(wireframe.vertices.clone());
        wireframe.position_format = "R32G32B32A32_FLOAT POSITION layout 13";
        if let Some((min, max)) = bounds(&wireframe.vertices) {
            wireframe.min = min;
            wireframe.max = max;
        }
    }

    if let Some((uv_tag, uv_format, uvs)) = input_layout_uvs(
        &vertex_previews,
        input_layout_index,
        wireframe.vertices.len(),
    ) {
        wireframe.uv_format = Some(format!(
            "{uv_format} from {uv_tag} layout {input_layout_index:?}"
        ));
        wireframe.uvs = Some(uvs);
    } else if let Some((uv_tag, uv_candidate, uvs)) =
        best_model_uvs(&vertex_previews, wireframe.vertices.len())
    {
        wireframe.uv_format = Some(format!("{} from {}", uv_candidate.label, uv_tag));
        wireframe.uvs = Some(uvs);
    }

    if let Some((normal_tag, normal_format, raw_normals)) = input_layout_vectors(
        &vertex_previews,
        input_layout_index,
        SEMANTIC_NORMAL,
        wireframe.vertices.len(),
    ) {
        wireframe.normal_format = Some(format!(
            "{normal_format} from {normal_tag} layout {input_layout_index:?}"
        ));
        // The compiled common-surface VS forwards input NORMAL to TEXCOORD7
        // without normalization. That raw SNORM vector is also authored data
        // for procedural object-space masks. Keep it byte-faithful there,
        // while normalizing the separate lighting copy.
        wireframe.procedural_normals = Some(
            raw_normals
                .iter()
                .map(|normal| [normal[0], normal[1], normal[2]])
                .collect(),
        );
        wireframe.normals = Some(
            raw_normals
                .into_iter()
                .filter_map(|normal| normalize_input_layout_vector(normal, false))
                .map(|normal| [normal[0], normal[1], normal[2]])
                .collect::<Vec<[f32; 3]>>(),
        )
        .filter(|normals| normals.len() == wireframe.vertices.len());
    }
    if let Some((tangent_tag, tangent_format, raw_tangents)) = input_layout_vectors(
        &vertex_previews,
        input_layout_index,
        SEMANTIC_TANGENT,
        wireframe.vertices.len(),
    ) {
        wireframe.tangent_format = Some(format!(
            "{tangent_format} from {tangent_tag} layout {input_layout_index:?}"
        ));
        wireframe.tangents = raw_tangents
            .into_iter()
            .map(|tangent| normalize_input_layout_vector(tangent, true))
            .collect::<Option<Vec<_>>>();
    }

    let index_entry = package_manager().get_entry(index_tag)?;
    let index_header = package_manager().read_tag(index_tag).ok()?;
    let index_preview =
        load_index_buffer_preview_for_tag(index_tag, &index_entry, &index_header).ok()?;
    wireframe.index_count_total = index_preview.index_count;
    wireframe.material_ranges.clear();
    if index_ranges.is_empty() {
        wireframe.indices = index_preview
            .indices
            .into_iter()
            .take(MAX_PREVIEW_INDICES)
            .collect();
    } else {
        let mut indices = Vec::new();
        for range in index_ranges {
            let source = index_preview
                .indices
                .get(
                    range.range.start.min(index_preview.indices.len())
                        ..range.range.end.min(index_preview.indices.len()),
                )
                .unwrap_or_default();
            let triangles = preview_triangles_from_indices(source, range.primitive_type)
                .chunks_exact(3)
                .filter(|triangle| {
                    triangle
                        .iter()
                        .all(|index| (*index as usize) < wireframe.vertices.len())
                })
                .flat_map(|triangle| triangle.iter().copied())
                .collect_vec();
            if triangles.is_empty() {
                continue;
            }
            let available = MAX_PREVIEW_INDICES.saturating_sub(indices.len());
            if available == 0 {
                break;
            }
            let index_start = indices.len();
            let index_count = triangles.len().min(available);
            indices.extend(triangles.into_iter().take(index_count));
            wireframe.material_ranges.push(WireframeMaterialRange {
                index_start,
                index_count,
                raw_lod_category: range.raw_lod_category,
                render_stage: range.render_stage,
                technique: range.technique,
                gear_dye_change_color_index: range.gear_dye_change_color_index,
                authored_source: None,
                authored_draw: range.authored_draw,
                procedural_scale: 1.0,
                texture: None,
                textures: WireframeMaterialTextures::default(),
            });
        }
        wireframe.indices = indices;
    }
    wireframe.source = format!("{vertex_tag} + {index_tag}");

    Some(wireframe)
}

fn best_model_uvs(
    vertex_previews: &[(TagHash, VertexBufferPreview)],
    vertex_count: usize,
) -> Option<(TagHash, VertexUvCandidate, Vec<[f32; 2]>)> {
    vertex_previews
        .iter()
        .flat_map(|(tag, preview)| {
            preview
                .uv_candidates
                .iter()
                .map(move |candidate| (*tag, preview, candidate))
        })
        .max_by_key(|(_tag, _preview, candidate)| uv_candidate_score(candidate))
        .and_then(|(tag, preview, candidate)| {
            let uvs = load_uvs_for_vertex_buffer(tag, preview, candidate, vertex_count)?;
            Some((tag, candidate.clone(), uvs))
        })
}

fn input_layout_uvs(
    vertex_previews: &[(TagHash, VertexBufferPreview)],
    input_layout_index: Option<u8>,
    vertex_count: usize,
) -> Option<(TagHash, InputLayoutFormat, Vec<[f32; 2]>)> {
    let layout_index = input_layout_index?;
    let layout = resolved_input_layout_texcoord0(layout_index)
        .or_else(|| alkahest_input_layout_texcoord0(layout_index))?;
    let (tag, preview) = vertex_previews.get(layout.buffer_index)?;
    let entry = package_manager().get_entry(*tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let endian = package_manager().version.endian();
    let uvs = decode_input_layout_uvs(
        &data,
        preview.header.stride as usize,
        endian,
        layout,
        vertex_count,
    );

    (!uvs.is_empty()).then_some((*tag, layout.format, uvs))
}

fn resolved_input_layout_texcoord0(layout_id: u8) -> Option<InputLayoutTexcoord> {
    let layout = resolved_input_layout_vector(layout_id, SEMANTIC_TEXCOORD, 0)?;
    Some(InputLayoutTexcoord {
        buffer_index: layout.buffer_index,
        offset: layout.offset,
        format: layout.format,
    })
}

fn resolved_input_layout_vector(
    layout_id: u8,
    semantic: u8,
    semantic_index: u8,
) -> Option<InputLayoutVector> {
    let stream_sets = vertex_layout_stream_sets_for_any_mapping(layout_id)?;

    for element_tag in tags_by_class(CLASS_VERTEX_INPUT_ELEMENT_SETS) {
        let sets = vertex_input_element_sets(element_tag);
        for (buffer_index, set_index) in stream_sets.iter().copied().enumerate() {
            let Some(elements) = sets.get(set_index) else {
                continue;
            };
            let mut offset = 0usize;
            for element in elements {
                let size = vertex_input_element_size(*element)?;
                if element.semantic == semantic && element.semantic_index == semantic_index {
                    return Some(InputLayoutVector {
                        buffer_index,
                        offset,
                        format: input_layout_format_from_vertex_format(element.format)?,
                    });
                }
                offset = offset.checked_add(size)?;
            }
        }
    }

    None
}

fn alkahest_input_layout_vector(layout_id: u8, semantic: u8) -> Option<InputLayoutVector> {
    match (layout_id, semantic) {
        (7, SEMANTIC_NORMAL) => Some(InputLayoutVector {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R16G16B16A16Snorm,
        }),
        (7, SEMANTIC_TANGENT) => Some(InputLayoutVector {
            buffer_index: 0,
            offset: 16,
            format: InputLayoutFormat::R16G16B16A16Snorm,
        }),
        _ => None,
    }
}

fn input_layout_vectors(
    vertex_previews: &[(TagHash, VertexBufferPreview)],
    input_layout_index: Option<u8>,
    semantic: u8,
    vertex_count: usize,
) -> Option<(TagHash, InputLayoutFormat, Vec<[f32; 4]>)> {
    let layout_index = input_layout_index?;
    let layout = resolved_input_layout_vector(layout_index, semantic, 0)
        .or_else(|| alkahest_input_layout_vector(layout_index, semantic))?;
    let (tag, preview) = vertex_previews.get(layout.buffer_index)?;
    let entry = package_manager().get_entry(*tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let endian = package_manager().version.endian();
    let vectors = decode_input_layout_vectors(
        &data,
        preview.header.stride as usize,
        endian,
        layout,
        vertex_count,
    )?;

    (vectors.len() == vertex_count).then_some((*tag, layout.format, vectors))
}

fn input_layout_rigid_indices(
    vertex_previews: &[(TagHash, VertexBufferPreview)],
    input_layout_index: Option<u8>,
    vertex_count: usize,
) -> Option<Vec<u16>> {
    let layout = resolved_input_layout_vector(input_layout_index?, SEMANTIC_POSITION, 0)?;
    if layout.format != InputLayoutFormat::R16G16B16A16Snorm {
        return None;
    }
    let (tag, preview) = vertex_previews.get(layout.buffer_index)?;
    let entry = package_manager().get_entry(*tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let stride = preview.header.stride as usize;
    let offset = layout.offset.checked_add(6)?;
    (stride >= offset + 2).then_some(())?;
    let endian = package_manager().version.endian();
    let indices = data
        .chunks_exact(stride)
        .take(vertex_count.min(MAX_PREVIEW_VERTICES))
        .map(|vertex| read_i16(&vertex[offset..offset + 2], endian).max(0) as u16)
        .collect_vec();
    (indices.len() == vertex_count).then_some(indices)
}

fn normalize_input_layout_vector(
    mut vector: [f32; 4],
    preserve_handedness: bool,
) -> Option<[f32; 4]> {
    let length = vector[..3]
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if !length.is_finite() || length < 0.000001 {
        return None;
    }
    for component in &mut vector[..3] {
        *component /= length;
    }
    if preserve_handedness {
        vector[3] = if vector[3] < 0.0 { -1.0 } else { 1.0 };
    }
    Some(vector)
}

fn alkahest_input_layout_texcoord0(layout_id: u8) -> Option<InputLayoutTexcoord> {
    match layout_id {
        2 | 5 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R32G32Float,
        }),
        3 | 12 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 12,
            format: InputLayoutFormat::R32G32Float,
        }),
        6 | 75 | 76 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 40,
            format: InputLayoutFormat::R32G32Float,
        }),
        7 | 8 | 13 => Some(InputLayoutTexcoord {
            buffer_index: 1,
            offset: 0,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        9 | 18 | 26 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        11 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R32G32B32Float,
        }),
        14 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 40,
            format: InputLayoutFormat::R32G32B32A32Float,
        }),
        15 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 16,
            format: InputLayoutFormat::R32G32Float,
        }),
        17 | 21 => Some(InputLayoutTexcoord {
            buffer_index: 1,
            offset: 0,
            format: InputLayoutFormat::R32G32B32Float,
        }),
        19 | 20 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 16,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        24 | 25 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 12,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44
        | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61
        | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 => {
            Some(InputLayoutTexcoord {
                buffer_index: 0,
                offset: 0,
                format: InputLayoutFormat::R32G32B32A32Float,
            })
        }
        _ => None,
    }
}

fn decode_input_layout_uvs(
    data: &[u8],
    stride: usize,
    endian: Endian,
    layout: InputLayoutTexcoord,
    vertex_count: usize,
) -> Vec<[f32; 2]> {
    if stride == 0 || layout.offset >= stride {
        return vec![];
    }

    data.chunks_exact(stride)
        .take(vertex_count.min(MAX_PREVIEW_VERTICES))
        .filter_map(|vertex| {
            decode_input_layout_uv(vertex.get(layout.offset..)?, endian, layout.format)
        })
        .collect()
}

fn decode_input_layout_uv(
    bytes: &[u8],
    endian: Endian,
    format: InputLayoutFormat,
) -> Option<[f32; 2]> {
    let uv = match format {
        InputLayoutFormat::R32G32Float
        | InputLayoutFormat::R32G32B32Float
        | InputLayoutFormat::R32G32B32A32Float => [
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
        ],
        InputLayoutFormat::R16G16Snorm => [
            read_snorm16(bytes.get(0..2)?, endian),
            read_snorm16(bytes.get(2..4)?, endian),
        ],
        InputLayoutFormat::R16G16B16A16Snorm => return None,
    };

    uv.iter()
        .all(|v| v.is_finite() && v.abs() <= 1024.0)
        .then_some(uv)
}

fn decode_input_layout_vectors(
    data: &[u8],
    stride: usize,
    endian: Endian,
    layout: InputLayoutVector,
    vertex_count: usize,
) -> Option<Vec<[f32; 4]>> {
    if stride == 0 || layout.offset >= stride {
        return None;
    }

    data.chunks_exact(stride)
        .take(vertex_count.min(MAX_PREVIEW_VERTICES))
        .map(|vertex| {
            decode_input_layout_vector(vertex.get(layout.offset..)?, endian, layout.format)
        })
        .collect()
}

fn decode_input_layout_vector(
    bytes: &[u8],
    endian: Endian,
    format: InputLayoutFormat,
) -> Option<[f32; 4]> {
    let vector = match format {
        InputLayoutFormat::R32G32B32Float => [
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
            read_f32(bytes.get(8..12)?, endian),
            1.0,
        ],
        InputLayoutFormat::R32G32B32A32Float => [
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
            read_f32(bytes.get(8..12)?, endian),
            read_f32(bytes.get(12..16)?, endian),
        ],
        InputLayoutFormat::R16G16B16A16Snorm => [
            read_snorm16(bytes.get(0..2)?, endian),
            read_snorm16(bytes.get(2..4)?, endian),
            read_snorm16(bytes.get(4..6)?, endian),
            read_snorm16(bytes.get(6..8)?, endian),
        ],
        InputLayoutFormat::R32G32Float | InputLayoutFormat::R16G16Snorm => return None,
    };

    vector
        .iter()
        .all(|value| value.is_finite())
        .then_some(vector)
}

fn load_uvs_for_vertex_buffer(
    tag: TagHash,
    preview: &VertexBufferPreview,
    candidate: &VertexUvCandidate,
    vertex_count: usize,
) -> Option<Vec<[f32; 2]>> {
    let entry = package_manager().get_entry(tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let endian = package_manager().version.endian();
    Some(decode_uvs(
        &data,
        preview.header.stride as usize,
        endian,
        candidate.format,
        candidate.offset,
        vertex_count,
    ))
    .filter(|uvs| !uvs.is_empty())
}

fn uv_candidate_score(candidate: &VertexUvCandidate) -> usize {
    let span_u = (candidate.max[0] - candidate.min[0]).abs();
    let span_v = (candidate.max[1] - candidate.min[1]).abs();
    let format_bonus = match candidate.format {
        UvFormat::F16x2 => 10_000,
        UvFormat::F32x2 => 0,
    };
    let compact_bonus = if candidate
        .min
        .into_iter()
        .chain(candidate.max)
        .all(|v| v.abs() <= 16.0)
    {
        20_000
    } else {
        0
    };
    let span_penalty = ((span_u + span_v) * 10.0).max(0.0) as usize;
    (candidate.valid_vertices + format_bonus + compact_bonus)
        .saturating_sub(span_penalty.min(10_000))
}

#[derive(Debug, Clone)]
struct StaticMeshPartPreview {
    index_start: u32,
    index_count: u32,
    buffer_index: u8,
    lod_category: u8,
    primitive_type: u8,
}

#[derive(Debug, Clone)]
struct StaticMeshGroupPreview {
    part_index: u16,
    render_stage: u8,
    input_layout_index: u8,
}

#[derive(Debug, Clone)]
struct DynamicMeshPartPreview {
    technique: TagHash,
    index_start: u32,
    index_count: u32,
    primitive_type: u8,
    lod_category: u8,
}

#[derive(Debug, Clone)]
struct GeometryIndexRangePreview {
    record_offset: usize,
    part_index: usize,
    render_stage: Option<u8>,
    technique: TagHash,
    variant_shader_index: u16,
    index_start: u32,
    index_count: u32,
    primitive_type: u8,
    flags: u32,
    gear_dye_change_color_index: u8,
    lod_category: u8,
    lod_run: u8,
}

#[derive(Debug, Clone)]
struct PreviewIndexRange {
    range: std::ops::Range<usize>,
    primitive_type: u8,
    raw_lod_category: Option<u8>,
    render_stage: Option<u8>,
    technique: Option<TagHash>,
    gear_dye_change_color_index: Option<u8>,
    authored_draw: Option<WireframeAuthoredDrawMetadata>,
}

#[derive(Debug, Clone)]
struct StaticBufferTuple {
    index_buffer: TagHash,
    vertex0_buffer: TagHash,
    vertex1_buffer: TagHash,
    color_buffer: TagHash,
}

fn read_static_mesh_parts(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<StaticMeshPartPreview> {
    read_array(data, vec_offset, 12, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(12))
        .filter_map(|part| {
            Some(StaticMeshPartPreview {
                index_start: read_u32(part.get(0x0..0x4)?, endian),
                index_count: read_u32(part.get(0x4..0x8)?, endian),
                buffer_index: *part.get(0x8)?,
                lod_category: *part.get(0xa)?,
                primitive_type: *part.get(0xb)?,
            })
        })
        .collect()
}

fn read_static_mesh_groups(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<StaticMeshGroupPreview> {
    read_array(data, vec_offset, 6, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(6))
        .filter_map(|group| {
            Some(StaticMeshGroupPreview {
                part_index: read_u16(group.get(0x0..0x2)?, endian),
                render_stage: *group.get(0x2)?,
                input_layout_index: *group.get(0x3)?,
            })
        })
        .collect()
}

fn read_static_buffer_tuples(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<StaticBufferTuple> {
    read_array(data, vec_offset, 16, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(16))
        .filter_map(|tuple| {
            Some(StaticBufferTuple {
                index_buffer: TagHash(read_u32(tuple.get(0x0..0x4)?, endian)),
                vertex0_buffer: TagHash(read_u32(tuple.get(0x4..0x8)?, endian)),
                vertex1_buffer: TagHash(read_u32(tuple.get(0x8..0xc)?, endian)),
                color_buffer: TagHash(read_u32(tuple.get(0xc..0x10)?, endian)),
            })
        })
        .collect()
}

fn static_mesh_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    let opaque = read_tag_array(data, 0x10, endian);
    let special = read_array(data, 0x20, 0x24, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x24))
        .filter_map(|mesh| Some(TagHash(read_u32(mesh.get(0x20..0x24)?, endian))));

    opaque
        .into_iter()
        .chain(special)
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn dynamic_model_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    read_array(data, 0x10, 0x80, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x80))
        .flat_map(|mesh| dynamic_mesh_technique_tags(mesh, endian))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn dynamic_mesh_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    read_array(data, 0x20, 0x24, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x24))
        .filter_map(|part| Some(TagHash(read_u32(part.get(0x0..0x4)?, endian))))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn geometry_resource_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    geometry_primary_index_ranges(data, endian)
        .into_iter()
        .map(|range| range.technique)
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn technique_tags_in_blob(data: &[u8], endian: Endian) -> Vec<TagHash> {
    let alternate = match endian {
        Endian::Little => Endian::Big,
        Endian::Big => Endian::Little,
    };

    candidate_tag_hashes_in_blob(data, endian)
        .into_iter()
        .chain(candidate_tag_hashes_in_blob(data, alternate))
        .chain(candidate_tag64_hashes_in_blob(data, endian))
        .chain(candidate_tag64_hashes_in_blob(data, alternate))
        .filter(|tag| is_technique_tag(*tag))
        .unique()
        .collect()
}

fn candidate_tag_hashes_in_blob(data: &[u8], endian: Endian) -> Vec<TagHash> {
    data.chunks_exact(4)
        .map(|bytes| TagHash(read_u32(bytes, endian)))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn candidate_tag64_hashes_in_blob(data: &[u8], endian: Endian) -> Vec<TagHash> {
    data.chunks_exact(8)
        .filter_map(|bytes| tag64_to_hash32(TagHash64(read_u64(bytes, endian))))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn tag64_to_hash32(tag: TagHash64) -> Option<TagHash> {
    package_manager()
        .lookup
        .tag64_entries
        .get(&tag.0)
        .map(|entry| entry.hash32)
}

fn is_technique_tag(tag: TagHash) -> bool {
    package_manager()
        .get_entry(tag)
        .is_some_and(|entry| is_technique_entry(&entry))
}

fn read_tag_array(data: &[u8], vec_offset: usize, endian: Endian) -> Vec<TagHash> {
    read_array(data, vec_offset, 4, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(4))
        .filter_map(|tag| Some(TagHash(read_u32(tag, endian))))
        .collect()
}

fn read_dynamic_mesh_parts(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<DynamicMeshPartPreview> {
    read_array(data, vec_offset, 0x24, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x24))
        .filter_map(|part| {
            Some(DynamicMeshPartPreview {
                technique: TagHash(read_u32(part.get(0x0..0x4)?, endian)),
                primitive_type: *part.get(0x6)?,
                index_start: read_u32(part.get(0x8..0xc)?, endian),
                index_count: read_u32(part.get(0xc..0x10)?, endian),
                lod_category: *part.get(0x1d)?,
            })
        })
        .collect()
}

fn geometry_buffer_set_input_layout_id(mesh: &[u8]) -> Option<u8> {
    mesh.get(0x64)
        .copied()
        .filter(|layout| *layout != 0)
        .or_else(|| mesh.get(0x62).copied().filter(|layout| *layout != 0))
}

fn geometry_primary_index_ranges(data: &[u8], endian: Endian) -> Vec<GeometryIndexRangePreview> {
    let selected_parts = geometry_preview_part_indices(data, endian);
    let all_candidates = geometry_index_range_candidates(data, endian);
    let candidates = all_candidates
        .iter()
        .cloned()
        .filter(|range| {
            selected_parts
                .as_ref()
                .is_none_or(|parts| parts.contains(&range.part_index))
        })
        .collect_vec();
    if candidates.is_empty() {
        return vec![];
    }
    let mut ranges = candidates
        .iter()
        .filter(|range| is_highest_detail_lod(range.lod_category))
        .cloned()
        .collect_vec();
    if ranges.is_empty() {
        let fallback_lod = candidates
            .iter()
            .min_by_key(|range| lod_selection_rank(range.lod_category))
            .map(|range| range.lod_category)
            .unwrap_or_default();
        ranges.extend(
            candidates
                .into_iter()
                .filter(|range| range.lod_category == fallback_lod),
        );
    }

    // Goliath's updated geometry can author opaque procedural-surface pieces in
    // a later render pass rather than GenerateGbuffer. Keep one such pass when
    // it fills source-index coverage absent from ordinary preview stages. This
    // recovers real authored panels/decals without drawing shadow/depth copies
    // or making sentinel-coloured geometry transparent.
    let mut covered = ranges
        .iter()
        .map(|range| range.index_start..range.index_start.saturating_add(range.index_count))
        .collect_vec();
    for supplemental in all_candidates.into_iter().filter(|range| {
        is_highest_detail_lod(range.lod_category)
            && procedural_surface_material_for_technique(range.technique).is_some()
    }) {
        let interval = supplemental.index_start
            ..supplemental
                .index_start
                .saturating_add(supplemental.index_count);
        let overlaps = covered
            .iter()
            .any(|existing| interval.start < existing.end && existing.start < interval.end);
        if !overlaps {
            covered.push(interval);
            ranges.push(supplemental);
        }
    }
    ranges.sort_by_key(|range| (lod_selection_rank(range.lod_category), range.index_start));
    ranges
}

const GOLIATH_RENDER_STAGE_COUNT: usize = crate::render::stage::MARATHON_RENDER_STAGE_COUNT;
const GOLIATH_RENDER_STAGE_BOUNDARY_COUNT: usize = GOLIATH_RENDER_STAGE_COUNT + 1;
const GOLIATH_RENDER_STAGE_BOUNDARY_OFFSET: usize = 0x30;
const GOLIATH_RENDER_STAGE_LAYOUT_OFFSET: usize = 0x64;

#[derive(Debug, Clone)]
struct GeometryRenderStageAbi {
    boundaries: [usize; GOLIATH_RENDER_STAGE_BOUNDARY_COUNT],
    input_layouts: [u8; GOLIATH_RENDER_STAGE_COUNT],
    part_count: usize,
}

impl GeometryRenderStageAbi {
    fn part_range(&self, stage: usize) -> Option<std::ops::Range<usize>> {
        (stage < GOLIATH_RENDER_STAGE_COUNT)
            .then(|| self.boundaries[stage]..self.boundaries[stage + 1])
    }
}

/// One package draw record of a geometry, in whichever render stage owns it.
pub(crate) struct GeometryStagePart {
    pub stage: Option<u8>,
    pub technique: TagHash,
    pub lod_category: u8,
    pub index_count: u32,
}

/// Every draw record the package authors for this geometry: all render stages
/// and all levels of detail, before any preview selection.
pub(crate) fn geometry_stage_parts(geometry: TagHash) -> Vec<GeometryStagePart> {
    let Ok(data) = package_manager().read_tag(geometry) else {
        return vec![];
    };
    geometry_index_range_candidates_raw(&data, package_manager().version.endian())
        .into_iter()
        .map(|range| GeometryStagePart {
            stage: range.render_stage,
            technique: range.technique,
            lod_category: range.lod_category,
            index_count: range.index_count,
        })
        .collect()
}

/// Authored compute part membership, independent of visible surface selection.
pub(crate) fn geometry_compute_technique(geometry: TagHash, visible_part: usize,
    range: std::ops::Range<u32>) -> Option<TagHash> {
    let data = package_manager().read_tag(geometry).ok()?;
    let endian = package_manager().version.endian();
    let abi = geometry_render_stage_abi(&data, endian)?;
    let parts = scan_arrays(&data, endian).into_iter().find(|a| a.class == CLASS_GEOMETRY_PART)?;
    if visible_part >= parts.count { return None; }
    let visible=parts.data_offset+visible_part*0x28;
    if read_u32_at(&data,visible+8,endian)? != range.start
        || read_u32_at(&data,visible+12,endian)?.checked_add(range.start)? != range.end {
        return None;
    }
    let lod=*data.get(visible+0x1c)?;
    let lod_run=*data.get(visible+0x1f)?;
    let mut found=None;
    for part in abi.part_range(24)? {
        let offset=parts.data_offset+part*0x28;
        let start=read_u32_at(&data,offset+8,endian)?;
        let count=read_u32_at(&data,offset+12,endian)?;
        if start==range.start && start.checked_add(count)?==range.end
            && data.get(offset+0x1c)==Some(&lod) && data.get(offset+0x1f)==Some(&lod_run) {
            let technique=read_tag_at(&data,offset,endian)?;
            if found.is_some_and(|existing| existing!=technique) { return None; }
            found=Some(technique);
        }
    }
    found
}

fn geometry_render_stage_abi(data: &[u8], endian: Endian) -> Option<GeometryRenderStageAbi> {
    let arrays = scan_arrays(data, endian);
    let part_count = arrays
        .iter()
        .find(|array| array.class == CLASS_GEOMETRY_PART)?
        .count;
    let buffer_set = arrays
        .iter()
        .find(|array| array.class == CLASS_GEOMETRY_BUFFER_SET)?;
    let record = data.get(
        buffer_set.data_offset
            ..(buffer_set.data_offset + 0x80)
                .min(buffer_set.end_offset)
                .min(data.len()),
    )?;
    if record.len() < GOLIATH_RENDER_STAGE_LAYOUT_OFFSET + GOLIATH_RENDER_STAGE_COUNT {
        return None;
    }

    let boundaries: [usize; GOLIATH_RENDER_STAGE_BOUNDARY_COUNT] = (0
        ..GOLIATH_RENDER_STAGE_BOUNDARY_COUNT)
        .map(|index| {
            record
                .get(GOLIATH_RENDER_STAGE_BOUNDARY_OFFSET + index * 2..)
                .map(|bytes| read_u16(bytes, endian) as usize)
        })
        .collect::<Option<Vec<_>>>()?
        .try_into()
        .ok()?;
    if boundaries.windows(2).any(|pair| pair[0] > pair[1])
        || boundaries.last().copied()? > part_count
    {
        return None;
    }

    let input_layouts: [u8; GOLIATH_RENDER_STAGE_COUNT] = record
        .get(
            GOLIATH_RENDER_STAGE_LAYOUT_OFFSET
                ..GOLIATH_RENDER_STAGE_LAYOUT_OFFSET + GOLIATH_RENDER_STAGE_COUNT,
        )?
        .try_into()
        .ok()?;

    Some(GeometryRenderStageAbi {
        boundaries,
        input_layouts,
        part_count,
    })
}

fn geometry_authored_stage_ranges(
    data: &[u8],
    endian: Endian,
    index_tag: TagHash,
    vertex_count: usize,
    stage: crate::render::stage::MarathonRenderStage,
) -> Vec<WireframeAuthoredStageRange> {
    let Some(abi) = geometry_render_stage_abi(data, endian) else {
        return vec![];
    };
    let stage_index = stage.raw() as usize;
    let Some(part_range) = abi.part_range(stage_index) else {
        return vec![];
    };

    let mut candidates = geometry_index_range_candidates_raw(data, endian)
        .into_iter()
        .filter(|range| part_range.contains(&range.part_index))
        .collect_vec();
    if candidates.is_empty() {
        return vec![];
    }

    let has_highest_detail = candidates
        .iter()
        .any(|range| is_highest_detail_lod(range.lod_category));
    if has_highest_detail {
        candidates.retain(|range| is_highest_detail_lod(range.lod_category));
    } else if let Some(fallback_lod) = candidates
        .iter()
        .min_by_key(|range| lod_selection_rank(range.lod_category))
        .map(|range| range.lod_category)
    {
        candidates.retain(|range| range.lod_category == fallback_lod);
    }

    let Some(index_entry) = package_manager().get_entry(index_tag) else {
        return vec![];
    };
    let Ok(index_header) = package_manager().read_tag(index_tag) else {
        return vec![];
    };
    let Ok(index_preview) =
        load_index_buffer_preview_for_tag(index_tag, &index_entry, &index_header)
    else {
        return vec![];
    };

    candidates
        .into_iter()
        .filter_map(|range| {
            let source = index_preview.indices.get(
                range.index_start as usize
                    ..range.index_start.saturating_add(range.index_count) as usize,
            )?;
            let indices = preview_triangles_from_indices(source, range.primitive_type)
                .chunks_exact(3)
                .filter(|triangle| {
                    triangle
                        .iter()
                        .all(|index| (*index as usize) < vertex_count)
                })
                .flat_map(|triangle| triangle.iter().copied())
                .collect_vec();
            (!indices.is_empty()).then_some(WireframeAuthoredStageRange {
                render_stage: stage.raw(),
                input_layout_id: abi.input_layouts[stage_index],
                part_index: range.part_index,
                source_index_start: range.index_start,
                source_index_count: range.index_count,
                primitive_type: range.primitive_type,
                raw_lod_category: range.lod_category,
                variant_shader_index: range.variant_shader_index,
                flags: range.flags,
                lod_run: range.lod_run,
                technique: range.technique.is_some().then_some(range.technique),
                gear_dye_change_color_index: Some(range.gear_dye_change_color_index),
                authored_source: Some(0),
                procedural_scale: 1.0,
                indices,
                texture: None,
                textures: WireframeMaterialTextures::default(),
            })
        })
        .collect()
}

fn geometry_preview_part_indices(data: &[u8], endian: Endian) -> Option<Vec<usize>> {
    let abi = geometry_render_stage_abi(data, endian)?;
    preview_part_indices_from_boundaries(&abi.boundaries, abi.part_count)
}

fn preview_part_indices_from_boundaries(
    boundaries: &[usize],
    part_count: usize,
) -> Option<Vec<usize>> {
    if boundaries.len() < GOLIATH_RENDER_STAGE_BOUNDARY_COUNT
        || boundaries.windows(2).any(|pair| pair[0] > pair[1])
        || boundaries.last().copied()? > part_count
    {
        return None;
    }

    Some(
        (0..GOLIATH_RENDER_STAGE_COUNT)
            .flat_map(|stage| boundaries[stage]..boundaries[stage + 1])
            .unique()
            .collect(),
    )
}

fn preview_part_stages_from_boundaries(
    boundaries: &[usize],
    part_count: usize,
) -> Option<Vec<Option<u8>>> {
    if boundaries.len() < GOLIATH_RENDER_STAGE_BOUNDARY_COUNT
        || boundaries.windows(2).any(|pair| pair[0] > pair[1])
        || boundaries.last().copied()? > part_count
    {
        return None;
    }

    let mut stages = vec![None; part_count];
    for stage in 0..GOLIATH_RENDER_STAGE_COUNT {
        for part in boundaries[stage]..boundaries[stage + 1] {
            stages[part] = Some(stage as u8);
        }
    }
    Some(stages)
}

fn geometry_preview_part_stages(data: &[u8], endian: Endian) -> Option<Vec<Option<u8>>> {
    let abi = geometry_render_stage_abi(data, endian)?;
    preview_part_stages_from_boundaries(&abi.boundaries, abi.part_count)
}

fn is_highest_detail_lod(lod: u8) -> bool {
    matches!(lod, 0 | 1 | 2 | 3 | 10)
}

fn index_ranges_from_geometry_ranges(
    ranges: &[GeometryIndexRangePreview],
) -> Vec<PreviewIndexRange> {
    ranges
        .iter()
        .map(|range| PreviewIndexRange {
            range: range.index_start as usize
                ..range.index_start.saturating_add(range.index_count) as usize,
            primitive_type: range.primitive_type,
            raw_lod_category: Some(range.lod_category),
            render_stage: range.render_stage,
            technique: Some(range.technique),
            gear_dye_change_color_index: (range.gear_dye_change_color_index <= 5)
                .then_some(range.gear_dye_change_color_index),
            authored_draw: Some(WireframeAuthoredDrawMetadata {
                part_index: range.part_index,
                source_index_start: range.index_start,
                source_index_count: range.index_count,
                primitive_type: range.primitive_type,
                variant_shader_index: range.variant_shader_index,
                flags: range.flags,
                lod_run: range.lod_run,
            }),
        })
        .collect()
}

fn preview_triangles_from_indices(indices: &[u32], primitive_type: u8) -> Vec<u32> {
    if primitive_type != 5 {
        return indices
            .chunks_exact(3)
            .flat_map(|triangle| triangle.iter().copied())
            .collect();
    }

    let mut out = Vec::with_capacity(indices.len().saturating_sub(2).saturating_mul(3));
    let mut strip = Vec::<u32>::with_capacity(3);
    let mut triangle_index = 0usize;
    for &index in indices {
        if matches!(index, 0xFFFF | 0xFFFF_FFFF) {
            strip.clear();
            triangle_index = 0;
            continue;
        }
        strip.push(index);
        if strip.len() < 3 {
            continue;
        }
        let a = strip[strip.len() - 3];
        let b = strip[strip.len() - 2];
        let c = strip[strip.len() - 1];
        if a == b || b == c || a == c {
            triangle_index += 1;
            continue;
        }
        if triangle_index % 2 == 0 {
            out.extend_from_slice(&[a, b, c]);
        } else {
            out.extend_from_slice(&[b, a, c]);
        }
        triangle_index += 1;
    }
    out
}

fn geometry_index_range_candidates_raw(
    data: &[u8],
    endian: Endian,
) -> Vec<GeometryIndexRangePreview> {
    let mut candidates = Vec::<GeometryIndexRangePreview>::new();
    let part_stages = geometry_preview_part_stages(data, endian).unwrap_or_default();
    let exact_offsets = scan_arrays(data, endian)
        .into_iter()
        .filter(|array| array.class == CLASS_GEOMETRY_PART)
        .flat_map(|array| {
            (0..array.count).map(move |index| (index, array.data_offset + index * 0x28))
        })
        .collect_vec();
    let offsets = if exact_offsets.is_empty() {
        (0..data.len().saturating_sub(0x27))
            .step_by(4)
            .enumerate()
            .map(|(index, offset)| (index, offset))
            .collect_vec()
    } else {
        exact_offsets
    };

    for (part_index, offset) in offsets {
        let Some(material) = read_tag_at(data, offset, endian) else {
            continue;
        };
        if !package_manager()
            .get_entry(material)
            .is_some_and(|entry| is_technique_entry(&entry))
        {
            continue;
        }

        let Some(index_start) = read_u32_at(data, offset + 0x8, endian) else {
            continue;
        };
        let Some(index_count) = read_u32_at(data, offset + 0xc, endian) else {
            continue;
        };
        let Some(primitive_type) = data.get(offset + 0x6).copied() else {
            continue;
        };
        // Goliath's 0x28-byte geometry part keeps the LOD category at +0x1c
        // and the change-color index at +0x1d. Reversing these fields makes
        // every range look like LOD 0 and renders its high and low-detail
        // meshes coplanar, which produces camera-dependent z-fighting.
        let Some(gear_dye_change_color_index) = data.get(offset + 0x1d).copied() else {
            continue;
        };
        let Some(lod_category) = data.get(offset + 0x1c).copied() else {
            continue;
        };
        if index_count < 3
            || !matches!(primitive_type, 3 | 5)
            || !matches!(lod_category, 0 | 1 | 2 | 3 | 4 | 7 | 8 | 9 | 10)
        {
            continue;
        }
        let Some(variant_shader_index) = data
            .get(offset + 0x4..offset + 0x6)
            .map(|bytes| read_u16(bytes, endian))
        else {
            continue;
        };
        let Some(flags) = read_u32_at(data, offset + 0x18, endian) else {
            continue;
        };
        let Some(lod_run) = data.get(offset + 0x1f).copied() else {
            continue;
        };

        candidates.push(GeometryIndexRangePreview {
            record_offset: offset,
            part_index,
            render_stage: part_stages.get(part_index).copied().flatten(),
            technique: material,
            variant_shader_index,
            index_start,
            index_count,
            primitive_type,
            flags,
            gear_dye_change_color_index,
            lod_category,
            lod_run,
        });
    }

    candidates
}

fn geometry_index_range_candidates(data: &[u8], endian: Endian) -> Vec<GeometryIndexRangePreview> {
    geometry_index_range_candidates_raw(data, endian)
        .into_iter()
        .unique_by(|range| {
            (
                range.index_start,
                range.index_count,
                range.primitive_type,
                range.lod_category,
            )
        })
        .collect()
}

fn geometry_primary_layout_id(data: &[u8], endian: Endian) -> Option<u8> {
    let mut best = None::<(u8, usize)>;

    for array in scan_arrays(data, endian) {
        if array.class != CLASS_GEOMETRY_BUFFER_SET {
            continue;
        }

        let record =
            data.get(array.data_offset.min(data.len())..array.end_offset.min(data.len()))?;
        let mut counts = [0usize; 0x40];
        for &byte in record.iter().skip(record.len() / 2) {
            if byte > 0 && (byte as usize) < counts.len() {
                counts[byte as usize] += 1;
            }
        }

        if let Some((layout_id, count)) = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .filter(|(_, count)| **count >= 4)
        {
            let replace = best
                .map(|(_, best_count)| *count > best_count)
                .unwrap_or(true);
            if replace {
                best = Some((layout_id as u8, *count));
            }
        }
    }

    best.map(|(layout_id, _)| layout_id)
}

fn read_geometry_uv_transform(data: &[u8], endian: Endian) -> Option<UvTransformPreview> {
    let scale = [
        read_f32(data.get(0xc0..0xc4)?, endian),
        read_f32(data.get(0xc4..0xc8)?, endian),
    ];
    let offset = [
        read_f32(data.get(0xc8..0xcc)?, endian),
        read_f32(data.get(0xcc..0xd0)?, endian),
    ];
    (scale
        .into_iter()
        .chain(offset)
        .all(|value| value.is_finite())
        && scale.iter().any(|value| value.abs() > 0.000001))
    .then_some(UvTransformPreview { scale, offset })
}

fn read_geometry_position_transform(
    data: &[u8],
    endian: Endian,
) -> Option<GeometryPositionTransform> {
    let scale = [
        read_f32(data.get(0xa0..0xa4)?, endian),
        read_f32(data.get(0xa4..0xa8)?, endian),
        read_f32(data.get(0xa8..0xac)?, endian),
    ];
    let offset = [
        read_f32(data.get(0xb0..0xb4)?, endian),
        read_f32(data.get(0xb4..0xb8)?, endian),
        read_f32(data.get(0xb8..0xbc)?, endian),
    ];
    let procedural_scale = read_f32(data.get(0xbc..0xc0)?, endian);
    (scale
        .into_iter()
        .chain(offset)
        .all(|value| value.is_finite())
        && procedural_scale.is_finite()
        && procedural_scale.abs() > 0.000001
        && scale.iter().all(|value| value.abs() > 0.000001))
    .then_some(GeometryPositionTransform {
        scale,
        offset,
        procedural_scale,
    })
}

fn apply_geometry_position_transform(
    wireframe: &mut WireframePreview,
    transform: GeometryPositionTransform,
) {
    let quantized = wireframe.position_format.starts_with("i16");
    for position in &mut wireframe.vertices {
        for axis in 0..3 {
            let source = if quantized {
                read_snorm_position(position[axis])
            } else {
                position[axis]
            };
            position[axis] = source * transform.scale[axis] + transform.offset[axis];
        }
    }
    // Compiled common-surface VS writes input POSITION straight to its
    // procedural varying, while rendered position follows geometry
    // dequantization. Vertex fetch supplies R16G16B16A16_SNORM, so preserve
    // exactly that normalized pre-transform value for pattern/wear.
    if quantized && let Some(positions) = &mut wireframe.procedural_positions {
        for position in positions {
            for axis in 0..3 {
                position[axis] = read_snorm_position(position[axis]);
            }
        }
    }
    // The same scope_skinning row that dequantizes POSITION carries a
    // separate object-space frequency multiplier in .w. Compiled common-
    // surface shaders apply it before evaluating procedural wear/patterns.
    // Keep that authored value on every draw range instead of silently using
    // the preview default (1.0).
    for range in &mut wireframe.material_ranges {
        range.procedural_scale = transform.procedural_scale;
    }
    for range in &mut wireframe.authored_shadow_ranges {
        range.procedural_scale = transform.procedural_scale;
    }
    if let Some((min, max)) = bounds(&wireframe.vertices) {
        wireframe.min = min;
        wireframe.max = max;
    }
}

fn read_snorm_position(value: f32) -> f32 {
    (value / 32767.0).clamp(-1.0, 1.0)
}

fn scan_arrays(data: &[u8], endian: Endian) -> Vec<TagArray> {
    let marker_offsets = (0..data.len().saturating_sub(4))
        .step_by(4)
        .filter(|offset| read_u32_at(data, *offset, endian) == Some(0x8080BFCD))
        .collect_vec();

    marker_offsets
        .iter()
        .enumerate()
        .filter_map(|(i, marker_offset)| {
            Some(TagArray {
                class: read_u32_at(data, marker_offset + 12, endian)?,
                count: read_u64_at(data, marker_offset + 4, endian)? as usize,
                data_offset: marker_offset + 20,
                end_offset: marker_offsets
                    .get(i + 1)
                    .copied()
                    .unwrap_or(data.len())
                    .min(data.len()),
            })
        })
        .collect()
}

fn array_records<'a>(data: &'a [u8], array: TagArray, stride: usize) -> Vec<&'a [u8]> {
    let start = array.data_offset.min(data.len());
    let end = array.end_offset.min(data.len());
    data[start..end]
        .chunks_exact(stride)
        .take(array.count)
        .collect()
}

fn vertex_layout_stream_bindings_for_any_mapping(
    layout_id: u8,
) -> Option<Vec<(usize, usize, bool)>> {
    for layout_tag in tags_by_class(CLASS_VERTEX_INPUT_LAYOUT_MAPPING) {
        if let Some(bindings) = vertex_layout_stream_bindings(layout_tag, layout_id) {
            return Some(bindings);
        }
    }

    None
}

fn vertex_layout_stream_bindings(
    layout_tag: TagHash,
    layout_id: u8,
) -> Option<Vec<(usize, usize, bool)>> {
    let endian = package_manager().version.endian();
    let data = package_manager().read_tag(layout_tag).ok()?;
    for array in scan_arrays(&data, endian) {
        if array.class != CLASS_VERTEX_LAYOUT_ARRAY {
            continue;
        }

        for record in array_records(&data, array, 0x1c) {
            let record_layout_id = (read_u32_at(record, 0, endian)? & 0xffff) as u8;
            if record_layout_id != layout_id {
                continue;
            }

            let bindings = (0..4usize)
                .filter_map(|stream_index| {
                    let set_index = read_u32_at(record, 0x8 + stream_index * 4, endian)?;
                    (set_index != u32::MAX).then_some((
                        stream_index,
                        set_index as usize,
                        record.get(0x18 + stream_index).copied().unwrap_or(0) != 0,
                    ))
                })
                .collect_vec();
            return (!bindings.is_empty()).then_some(bindings);
        }
    }

    None
}

fn vertex_layout_stream_sets_for_any_mapping(layout_id: u8) -> Option<Vec<usize>> {
    Some(
        vertex_layout_stream_bindings_for_any_mapping(layout_id)?
            .into_iter()
            .map(|(_stream_index, set_index, _instanced)| set_index)
            .collect(),
    )
}

fn vertex_input_element_sets(tag: TagHash) -> Vec<Vec<VertexInputElement>> {
    let endian = package_manager().version.endian();
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };

    scan_arrays(&data, endian)
        .into_iter()
        .filter(|array| array.class == CLASS_VERTEX_INPUT_ELEMENT_ARRAY)
        .map(|array| {
            data[array.data_offset.min(data.len())..array.end_offset.min(data.len())]
                .chunks_exact(3)
                .take(array.count)
                .map(|record| VertexInputElement {
                    semantic: record[0],
                    semantic_index: record[1],
                    format: record[2],
                })
                .collect()
        })
        .collect()
}

fn authored_input_layout_descriptor(layout_id: u8) -> Option<AuthoredInputLayoutDescriptor> {
    let bindings = vertex_layout_stream_bindings_for_any_mapping(layout_id)?;
    for element_tag in tags_by_class(CLASS_VERTEX_INPUT_ELEMENT_SETS) {
        let sets = vertex_input_element_sets(element_tag);
        if !bindings
            .iter()
            .all(|(_stream_index, set_index, _instanced)| *set_index < sets.len())
        {
            continue;
        }

        let streams = bindings
            .iter()
            .map(|(stream_index, set_index, instanced)| {
                let mut offset = 0usize;
                let mut elements = Vec::new();
                for element in sets.get(*set_index)? {
                    elements.push(AuthoredVertexElementDescriptor {
                        semantic: element.semantic,
                        semantic_index: element.semantic_index,
                        format: element.format,
                        offset: u16::try_from(offset).ok()?,
                    });
                    offset = offset.checked_add(vertex_input_element_size(*element)?)?;
                }
                Some(AuthoredVertexStreamLayoutDescriptor {
                    stream_index: u8::try_from(*stream_index).ok()?,
                    element_set_index: u32::try_from(*set_index).ok()?,
                    instanced: *instanced,
                    elements,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        return Some(AuthoredInputLayoutDescriptor { layout_id, streams });
    }

    None
}

fn authored_vertex_stream_ref(
    stream_index: u8,
    header_tag: TagHash,
) -> Option<AuthoredVertexStreamRef> {
    let entry = package_manager().get_entry(header_tag)?;
    let header_data = package_manager().read_tag(header_tag).ok()?;
    let header =
        VertexBufferHeader::parse(&header_data, package_manager().version.endian()).ok()?;
    let data_tag = TagHash(entry.reference);
    let element_count = (header.stride != 0)
        .then(|| header.data_size / u32::from(header.stride))
        .unwrap_or(0);

    Some(AuthoredVertexStreamRef {
        stream_index,
        header_tag,
        data_tag,
        stride: header.stride,
        vertex_type: header.vtype,
        data_size: header.data_size,
        element_count,
    })
}

fn authored_index_buffer_ref(header_tag: TagHash) -> Option<AuthoredIndexBufferRef> {
    let entry = package_manager().get_entry(header_tag)?;
    let header_data = package_manager().read_tag(header_tag).ok()?;
    let header = IndexBufferHeader::parse(&header_data, package_manager().version.endian()).ok()?;
    let data_tag = TagHash(entry.reference);
    let index_width = if header.is_32bit { 4 } else { 2 };
    let index_count = u32::try_from(header.data_size / index_width).ok()?;

    Some(AuthoredIndexBufferRef {
        header_tag,
        data_tag,
        is_32bit: header.is_32bit,
        data_size: header.data_size,
        index_count,
    })
}

fn authored_geometry_input(
    geometry: TagHash,
    mesh: &[u8],
    data: &[u8],
    endian: Endian,
    position_transform: Option<GeometryPositionTransform>,
    uv_transform: Option<UvTransformPreview>,
) -> AuthoredGeometryInput {
    let vertex_streams = [0x0usize, 0x4, 0x8, 0xc]
        .into_iter()
        .enumerate()
        .filter_map(|(stream_index, offset)| {
            authored_vertex_stream_ref(
                stream_index as u8,
                read_tag_at(mesh, offset, endian).unwrap_or(TagHash(0)),
            )
        })
        .collect_vec();
    let color_buffer =
        authored_vertex_stream_ref(4, read_tag_at(mesh, 0x14, endian).unwrap_or(TagHash(0)));
    let skinning_buffer =
        authored_vertex_stream_ref(5, read_tag_at(mesh, 0x18, endian).unwrap_or(TagHash(0)));
    let index_buffer = read_tag_at(mesh, 0x10, endian).and_then(authored_index_buffer_ref);
    let stage_layouts = geometry_render_stage_abi(data, endian)
        .map(|abi| {
            abi.input_layouts
                .iter()
                .copied()
                .enumerate()
                .map(|(raw_stage, layout_id)| AuthoredStageInputLayout {
                    raw_stage: raw_stage as u8,
                    layout_id,
                    descriptor: authored_input_layout_descriptor(layout_id),
                })
                .collect_vec()
        })
        .unwrap_or_default();

    AuthoredGeometryInput {
        geometry,
        vertex_streams,
        color_buffer,
        skinning_buffer,
        index_buffer,
        stage_layouts,
        position_transform,
        uv_transform,
        attachment_pose: None,
    }
}

fn vertex_input_element_size(element: VertexInputElement) -> Option<usize> {
    match element.format {
        0x00 => Some(0),
        0x01 => Some(4),
        0x02 => Some(8),
        0x03 => Some(12),
        0x04 => Some(16),
        0x05 | 0x06 | 0x07 => Some(4),
        0x08 | 0x09 => Some(8),
        0x0A => Some(4),
        0x0B => Some(8),
        0x0C => Some(4),
        0x0D => Some(8),
        0x0E | 0x0F | 0x10 | 0x11 | 0x12 => Some(4),
        0x13 => Some(8),
        0x14 => Some(16),
        0x15 => Some(4),
        0x16 => Some(8),
        0x17 => Some(16),
        0x18 => Some(2),
        0x19 => Some(1),
        0x1F | 0x20 => Some(4),
        0x21 => Some(8),
        _ => None,
    }
}

fn input_layout_format_from_vertex_format(format: u8) -> Option<InputLayoutFormat> {
    match format {
        0x02 => Some(InputLayoutFormat::R32G32Float),
        0x03 => Some(InputLayoutFormat::R32G32B32Float),
        0x04 => Some(InputLayoutFormat::R32G32B32A32Float),
        0x0A => Some(InputLayoutFormat::R16G16Snorm),
        0x0B => Some(InputLayoutFormat::R16G16B16A16Snorm),
        _ => None,
    }
}

fn tags_by_class(class: u32) -> Vec<TagHash> {
    let pm = package_manager();
    let mut tags = Vec::new();
    for (pkg_id, entries) in &pm.lookup.tag32_entries_by_pkg {
        for (index, entry) in entries.iter().enumerate() {
            if entry.reference == class {
                tags.push(TagHash::new(*pkg_id, index as u16));
            }
        }
    }
    tags.sort_by_key(|tag| (tag.pkg_id(), tag.entry_index()));
    tags
}

fn read_tag_at(data: &[u8], offset: usize, endian: Endian) -> Option<TagHash> {
    Some(TagHash(read_u32_at(data, offset, endian)?))
        .filter(|tag| package_manager().get_entry(*tag).is_some())
}

fn read_u32_at(data: &[u8], offset: usize, endian: Endian) -> Option<u32> {
    data.get(offset..offset + 4)
        .map(|bytes| read_u32(bytes, endian))
}

fn read_u64_at(data: &[u8], offset: usize, endian: Endian) -> Option<u64> {
    data.get(offset..offset + 8)
        .map(|bytes| read_u64(bytes, endian))
}

fn read_array(data: &[u8], vec_offset: usize, elem_size: usize, endian: Endian) -> Option<&[u8]> {
    let count = read_u64(data.get(vec_offset..vec_offset + 8)?, endian) as usize;
    if count == 0 || elem_size == 0 {
        return Some(&[]);
    }

    let rel = read_i64(data.get(vec_offset + 8..vec_offset + 16)?, endian);
    let header_offset = (vec_offset as i64 + 8).checked_add(rel)? as usize;
    let header_count = read_u64(data.get(header_offset..header_offset + 8)?, endian) as usize;
    if header_count != count {
        return None;
    }

    let data_start = header_offset + 16;
    let data_end = data_start.checked_add(count.checked_mul(elem_size)?)?;
    data.get(data_start..data_end)
}

fn first_dynamic_input_layout(data: &[u8]) -> Option<u8> {
    data.get(0x62..0x62 + 24)?.iter().copied().find(|v| *v != 0)
}

fn read_static_uv_transform(data: &[u8], endian: Endian) -> Option<UvTransformPreview> {
    let scale = read_f32(data.get(0x54..0x58)?, endian);
    let offset = [
        read_f32(data.get(0x58..0x5c)?, endian),
        read_f32(data.get(0x5c..0x60)?, endian),
    ];
    Some(UvTransformPreview {
        scale: [scale, scale],
        offset,
    })
}

fn read_dynamic_model_uv_transform(data: &[u8], endian: Endian) -> Option<UvTransformPreview> {
    Some(UvTransformPreview {
        scale: [
            read_f32(data.get(0x60..0x64)?, endian),
            read_f32(data.get(0x64..0x68)?, endian),
        ],
        offset: [
            read_f32(data.get(0x68..0x6c)?, endian),
            read_f32(data.get(0x6c..0x70)?, endian),
        ],
    })
}

fn shader_constants_from_uv_transform(
    uv_transform: Option<UvTransformPreview>,
) -> Vec<ShaderConstantPreview> {
    uv_transform
        .map(|uv| {
            vec![ShaderConstantPreview {
                name: "uv_scale_offset",
                value: [uv.scale[0], uv.scale[1], uv.offset[0], uv.offset[1]],
                source: "mesh instance UV transform",
            }]
        })
        .unwrap_or_default()
}

fn lod_selection_rank(lod: u8) -> u8 {
    match lod {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        10 => 4,
        4 => 10,
        7 => 20,
        8 => 21,
        9 => 30,
        _ => 40,
    }
}

fn lod_preview_value(lod: u8) -> u8 {
    match lod {
        0 | 1 | 2 | 3 | 10 => 0,
        4 => 1,
        7 | 8 => 2,
        9 => 3,
        _ => lod,
    }
}

#[derive(Clone, Copy)]
enum TagSearchKind {
    VertexBuffer,
    IndexBuffer,
    Texture,
    Technique,
    Shader,
}

fn find_related_tags(
    cache: &TagCache,
    start: TagHash,
    kind: TagSearchKind,
    max_depth: usize,
) -> Vec<(TagHash, UEntryHeader)> {
    let mut out = vec![];
    let mut seen = rustc_hash::FxHashSet::default();
    find_related_tags_recursive(cache, start, kind, 0, max_depth, &mut seen, &mut out);
    out.into_iter()
        .unique_by(|(tag, _)| *tag)
        .take(128)
        .collect()
}

fn find_related_tags_recursive(
    cache: &TagCache,
    tag: TagHash,
    kind: TagSearchKind,
    depth: usize,
    max_depth: usize,
    seen: &mut rustc_hash::FxHashSet<TagHash>,
    out: &mut Vec<(TagHash, UEntryHeader)>,
) {
    if depth > max_depth || !seen.insert(tag) {
        return;
    }

    let Some(scan) = cache.hashes.get(&tag) else {
        return;
    };

    let children = scan
        .file_hashes
        .iter()
        .map(|scanned| scanned.hash)
        .chain(
            scan.file_hashes64
                .iter()
                .filter_map(|scanned| tag64_to_hash32(scanned.hash)),
        )
        .unique()
        .collect_vec();

    for child in children {
        let Some(entry) = package_manager().get_entry(child) else {
            continue;
        };
        let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
        if tag_matches_kind(&entry, tag_type, kind) {
            out.push((child, entry));
        } else if tag_type.is_tag() {
            find_related_tags_recursive(cache, child, kind, depth + 1, max_depth, seen, out);
        }
    }
}

fn tag_matches_kind(entry: &UEntryHeader, tag_type: TagType, kind: TagSearchKind) -> bool {
    match kind {
        TagSearchKind::VertexBuffer => {
            matches!(tag_type, TagType::VertexBuffer { is_header: true })
        }
        TagSearchKind::IndexBuffer => matches!(tag_type, TagType::IndexBuffer { is_header: true }),
        TagSearchKind::Texture => tag_type.is_texture() && tag_type.is_header(),
        TagSearchKind::Technique => is_technique_entry(entry),
        TagSearchKind::Shader => tag_type.is_shader() && tag_type.is_header(),
    }
}

impl VertexBufferHeader {
    fn parse(data: &[u8], endian: Endian) -> anyhow::Result<Self> {
        if data.len() < 12 {
            bail!("Vertex buffer header needs 12 bytes, got {}", data.len());
        }
        Ok(Self {
            data_size: read_u32(&data[0..4], endian),
            stride: read_u16(&data[4..6], endian),
            vtype: read_u16(&data[6..8], endian),
            deadbeef: read_u32(&data[8..12], endian),
        })
    }
}

impl IndexBufferHeader {
    fn parse(data: &[u8], endian: Endian) -> anyhow::Result<Self> {
        if data.len() < 24 {
            bail!("Index buffer header needs 24 bytes, got {}", data.len());
        }
        Ok(Self {
            unk0: data[0] as i8,
            is_32bit: data[1] != 0,
            unk1: read_u16(&data[2..4], endian),
            zero: read_u32(&data[4..8], endian),
            data_size: read_u64(&data[8..16], endian),
            deadbeef: read_u32(&data[16..20], endian),
            zero1: read_u32(&data[20..24], endian),
        })
    }
}

fn find_position_candidates(
    data: &[u8],
    stride: usize,
    endian: Endian,
) -> Vec<VertexPositionCandidate> {
    if stride == 0 || data.len() < stride {
        return vec![];
    }

    let mut candidates = vec![];
    if stride >= 12 {
        if let Some(candidate) = candidate_f32x3(data, stride, 0, endian) {
            candidates.push(candidate);
        }
    }
    if stride >= 8 {
        if let Some(candidate) = candidate_i16x4(data, stride, 0, endian) {
            candidates.push(candidate);
        }
    }
    if stride >= 6 {
        if let Some(candidate) = candidate_i16x3(data, stride, 0, endian) {
            candidates.push(candidate);
        }
    }

    candidates
}

fn find_uv_candidates(data: &[u8], stride: usize, endian: Endian) -> Vec<VertexUvCandidate> {
    if stride == 0 || data.len() < stride {
        return vec![];
    }

    let mut candidates = vec![];
    for offset in (0..stride).step_by(2) {
        if offset + 4 <= stride
            && let Some(candidate) = candidate_f16x2_uv(data, stride, offset, endian)
        {
            candidates.push(candidate);
        }
    }

    candidates
        .into_iter()
        .sorted_by_key(|candidate| std::cmp::Reverse(uv_candidate_score(candidate)))
        .take(8)
        .collect()
}

fn candidate_f16x2_uv(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexUvCandidate> {
    build_uv_candidate(data, stride, offset, UvFormat::F16x2, |bytes| {
        Some([
            read_f16(bytes.get(0..2)?, endian),
            read_f16(bytes.get(2..4)?, endian),
        ])
    })
}

fn build_uv_candidate(
    data: &[u8],
    stride: usize,
    offset: usize,
    format: UvFormat,
    mut decode: impl FnMut(&[u8]) -> Option<[f32; 2]>,
) -> Option<VertexUvCandidate> {
    let vertex_count = data.len() / stride;
    let sample_count = vertex_count.min(2048);
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    let mut valid = 0usize;

    for i in 0..sample_count {
        let start = i * stride + offset;
        let vertex_end = (i + 1) * stride;
        let uv = decode(data.get(start..vertex_end)?)?;
        if !uv.iter().all(|v| v.is_finite() && v.abs() <= 1024.0) {
            continue;
        }
        for axis in 0..2 {
            min[axis] = min[axis].min(uv[axis]);
            max[axis] = max[axis].max(uv[axis]);
        }
        valid += 1;
    }

    if valid < sample_count.saturating_div(2).max(1) {
        return None;
    }
    let span_u = (max[0] - min[0]).abs();
    let span_v = (max[1] - min[1]).abs();
    if span_u < 0.0001 && span_v < 0.0001 {
        return None;
    }

    Some(VertexUvCandidate {
        format,
        label: format!("{} @ +{}", format.label(), offset),
        offset,
        valid_vertices: valid,
        sampled_vertices: sample_count,
        min,
        max,
    })
}

fn candidate_f32x3(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(
        data,
        stride,
        offset,
        PositionFormat::F32x3,
        "f32x3 @ +0",
        |bytes| {
            Some([
                read_f32(bytes.get(0..4)?, endian),
                read_f32(bytes.get(4..8)?, endian),
                read_f32(bytes.get(8..12)?, endian),
            ])
        },
    )
}

fn candidate_i16x4(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(
        data,
        stride,
        offset,
        PositionFormat::I16x4,
        "i16x4.xyz @ +0",
        |bytes| {
            Some([
                read_i16(bytes.get(0..2)?, endian) as f32,
                read_i16(bytes.get(2..4)?, endian) as f32,
                read_i16(bytes.get(4..6)?, endian) as f32,
            ])
        },
    )
}

fn candidate_i16x3(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(
        data,
        stride,
        offset,
        PositionFormat::I16x3,
        "i16x3 @ +0",
        |bytes| {
            Some([
                read_i16(bytes.get(0..2)?, endian) as f32,
                read_i16(bytes.get(2..4)?, endian) as f32,
                read_i16(bytes.get(4..6)?, endian) as f32,
            ])
        },
    )
}

fn build_candidate(
    data: &[u8],
    stride: usize,
    offset: usize,
    format: PositionFormat,
    label: &'static str,
    mut decode: impl FnMut(&[u8]) -> Option<[f32; 3]>,
) -> Option<VertexPositionCandidate> {
    let vertex_count = data.len() / stride;
    let sample_count = vertex_count.min(2048);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut valid = 0usize;

    for i in 0..sample_count {
        let start = i * stride + offset;
        let position = decode(data.get(start..start + stride)?)?;
        if !position.iter().all(|v| v.is_finite() && v.abs() < 1.0e8) {
            continue;
        }
        if position == [0.0, 0.0, 0.0] {
            continue;
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
        valid += 1;
    }

    if valid == 0 {
        return None;
    }

    Some(VertexPositionCandidate {
        format,
        label,
        offset,
        valid_vertices: valid,
        sampled_vertices: sample_count,
        min,
        max,
    })
}

fn build_vertex_wireframe(
    tag: TagHash,
    data: &[u8],
    stride: usize,
    endian: Endian,
    candidates: &[VertexPositionCandidate],
    vertex_count_total: usize,
) -> Option<WireframePreview> {
    let candidate = candidates
        .iter()
        .filter(|candidate| candidate.valid_vertices > 0)
        .max_by_key(|candidate| candidate.valid_vertices)?;
    let vertices = decode_positions(data, stride, endian, candidate.format, candidate.offset);
    let (min, max) = bounds(&vertices)?;

    Some(WireframePreview {
        source: tag.to_string(),
        position_format: candidate.label,
        uv_format: None,
        index_count_total: 0,
        vertex_count_total,
        procedural_positions: Some(vertices.clone()),
        vertices,
        rigid_indices: None,
        normals: None,
        procedural_normals: None,
        tangents: None,
        uvs: None,
        normal_format: None,
        tangent_format: None,
        indices: vec![],
        material_ranges: vec![],
        authored_inputs: vec![],
        authored_shadow_ranges: vec![],
        min,
        max,
    })
}

fn decode_positions(
    data: &[u8],
    stride: usize,
    endian: Endian,
    format: PositionFormat,
    offset: usize,
) -> Vec<[f32; 3]> {
    if stride == 0 {
        return vec![];
    }

    data.chunks_exact(stride)
        .take(MAX_PREVIEW_VERTICES)
        .filter_map(|vertex| {
            let bytes = vertex.get(offset..)?;
            match format {
                PositionFormat::F32x3 => Some([
                    read_f32(bytes.get(0..4)?, endian),
                    read_f32(bytes.get(4..8)?, endian),
                    read_f32(bytes.get(8..12)?, endian),
                ]),
                PositionFormat::I16x4 | PositionFormat::I16x3 => Some([
                    read_i16(bytes.get(0..2)?, endian) as f32,
                    read_i16(bytes.get(2..4)?, endian) as f32,
                    read_i16(bytes.get(4..6)?, endian) as f32,
                ]),
            }
        })
        .filter(|position| position.iter().all(|v| v.is_finite() && v.abs() < 1.0e8))
        .collect()
}

fn decode_uvs(
    data: &[u8],
    stride: usize,
    endian: Endian,
    format: UvFormat,
    offset: usize,
    vertex_count: usize,
) -> Vec<[f32; 2]> {
    if stride == 0 {
        return vec![];
    }

    data.chunks_exact(stride)
        .take(vertex_count.min(MAX_PREVIEW_VERTICES))
        .filter_map(|vertex| {
            let bytes = vertex.get(offset..)?;
            let uv = match format {
                UvFormat::F32x2 => [
                    read_f32(bytes.get(0..4)?, endian),
                    read_f32(bytes.get(4..8)?, endian),
                ],
                UvFormat::F16x2 => [
                    read_f16(bytes.get(0..2)?, endian),
                    read_f16(bytes.get(2..4)?, endian),
                ],
            };
            uv.iter()
                .all(|v| v.is_finite() && v.abs() <= 1024.0)
                .then_some(uv)
        })
        .collect()
}

fn bounds(vertices: &[[f32; 3]]) -> Option<([f32; 3], [f32; 3])> {
    if vertices.is_empty() {
        return None;
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }

    Some((min, max))
}

fn read_u16(data: &[u8], endian: Endian) -> u16 {
    let bytes = data[0..2].try_into().expect("u16 slice length checked");
    match endian {
        Endian::Big => u16::from_be_bytes(bytes),
        Endian::Little => u16::from_le_bytes(bytes),
    }
}

fn read_i16(data: &[u8], endian: Endian) -> i16 {
    let bytes = data[0..2].try_into().expect("i16 slice length checked");
    match endian {
        Endian::Big => i16::from_be_bytes(bytes),
        Endian::Little => i16::from_le_bytes(bytes),
    }
}

fn read_snorm16(data: &[u8], endian: Endian) -> f32 {
    (read_i16(data, endian) as f32 / 32767.0).clamp(-1.0, 1.0)
}

fn read_f32(data: &[u8], endian: Endian) -> f32 {
    let bytes = data[0..4].try_into().expect("f32 slice length checked");
    match endian {
        Endian::Big => f32::from_be_bytes(bytes),
        Endian::Little => f32::from_le_bytes(bytes),
    }
}

fn read_f16(data: &[u8], endian: Endian) -> f32 {
    let bits = read_u16(data, endian);
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x03ff) as u32;

    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut mantissa = mant;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13),
    };

    f32::from_bits(f32_bits)
}

fn read_u32(data: &[u8], endian: Endian) -> u32 {
    let bytes = data[0..4].try_into().expect("u32 slice length checked");
    match endian {
        Endian::Big => u32::from_be_bytes(bytes),
        Endian::Little => u32::from_le_bytes(bytes),
    }
}

fn read_u64(data: &[u8], endian: Endian) -> u64 {
    let bytes = data[0..8].try_into().expect("u64 slice length checked");
    match endian {
        Endian::Big => u64::from_be_bytes(bytes),
        Endian::Little => u64::from_le_bytes(bytes),
    }
}

fn read_i64(data: &[u8], endian: Endian) -> i64 {
    let bytes = data[0..8].try_into().expect("i64 slice length checked");
    match endian {
        Endian::Big => i64::from_be_bytes(bytes),
        Endian::Little => i64::from_le_bytes(bytes),
    }
}

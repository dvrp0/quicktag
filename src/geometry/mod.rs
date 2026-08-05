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
    interpret_tfx_stack_with_object_channels, interpret_tfx_stack_with_runtime_inputs,
    is_technique_entry, material_constants_for_technique, primary_sampler_for_technique,
    render_state_for_technique, texture_bindings_for_technique,
};
use crate::texture::Texture;

#[cfg(test)]
use crate::material::tfx_texture_bindings_for_technique;

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
        matches!(self.role, ModelTagRole::Dynamic) && self.label != "Dynamic mesh"
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
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub vertex_count_total: usize,
    pub index_count_total: usize,
}

#[derive(Debug, Clone)]
pub struct WireframeMaterialRange {
    pub index_start: usize,
    pub index_count: usize,
    pub render_stage: Option<u8>,
    pub technique: Option<TagHash>,
    pub gear_dye_change_color_index: Option<u8>,
    /// Rigid-model `position_offset.w` / skinning `offset_scale.w` consumed
    /// by common-surface procedural branches through `scope_skinning[5].w`.
    pub procedural_scale: f32,
    pub texture: Option<TagHash>,
    pub textures: WireframeMaterialTextures,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearDyeMaterial {
    pub color: [f32; 4],
    pub roughness_remap: [f32; 4],
    pub metal_remap: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct WireframeMaterialTextures {
    pub color: Option<TagHash>,
    pub normal: Option<TagHash>,
    pub emissive: Option<TagHash>,
    pub control: Option<TagHash>,
    pub roughness_channel: u8,
    pub sampler: Option<TagHash>,
    pub aux: Vec<TagHash>,
    pub layers: Vec<WireframeMaterialLayer>,
    pub color_tint: [u8; 4],
    pub mask_palette: Option<[[f32; 4]; 2]>,
    pub gear_dye: Option<GearDyeMaterial>,
    pub gear_dye_default: Option<[f32; 4]>,
    pub gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    pub mod_wear: Option<WeaponModWearMaterial>,
    /// Time-driven common-surface overlay used by animated inventory skins.
    /// Detection comes from the compiled PS/TFX ABI: a slot-7 field atlas,
    /// frame-driven outputs 66/67/71, and the authored response constants.
    pub animated_dither: Option<AnimatedDitherMaterial>,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvestmentDecalMode {
    SelectorMask,
    DetailSelectorMask,
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
pub struct WeaponModWearMaterial {
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
            roughness_channel: 0,
            sampler: None,
            aux: vec![],
            layers: vec![],
            color_tint: [255, 255, 255, 255],
            mask_palette: None,
            gear_dye: None,
            gear_dye_default: None,
            gear_dye_palette: None,
            mod_wear: None,
            animated_dither: None,
            gear_pattern: None,
            authored_shared_atlas: false,
            investment_decal: None,
            emissive_strength: 0,
            solid_color: None,
            solid_surface: None,
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

/// Decoded parameters for Tiger's animated circular/dither surface branch.
///
/// The game scrolls two scales of the shared slot-7 atlas in object space,
/// converts its blue/alpha pair into a triangular time mask, gates it by the
/// authored object-space face normal, then reshapes the material colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimatedDitherMaterial {
    pub field: TagHash,
    /// Slot-6 technical LUT sampled with mesh UVs before the animated response.
    pub technical_mask: TagHash,
    /// Per control-material affine remaps (PS constants 82..88).
    pub technical_mask_remap: [[f32; 2]; 7],
    /// Common-surface detail branch (PS t5). Engine projects this field in
    /// object space and routes it into roughness, not base colour.
    pub dot_detail: Option<AnimatedDotDetailMaterial>,
    pub phase_speed: f32,
    /// xy = object-space scale, zw = scroll rate.
    pub primary_transform: [f32; 4],
    /// xy = object-space scale, zw = scroll rate.
    pub secondary_transform: [f32; 4],
    /// x = waveform numerator, y = divisor, z/w = affine remap.
    pub waveform: [f32; 4],
    /// xy = normal-facing affine gate, z = response scale (c75*c76*c77 default),
    /// w = authored wave strength (c74 default).
    pub facing_response: [f32; 4],
    /// x = colour exponent, y = colour scale, z = normal response, w = mask exponent.
    pub surface_response: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimatedDotDetailMaterial {
    pub texture: TagHash,
    pub position_scale: [f32; 3],
    pub position_offset: [f32; 3],
    /// PS c4: coarse triplanar projection used by control material IDs 0, 1, 3, and 4.
    pub projection: [f32; 4],
    /// PS c5: fine triplanar projection used only by control material ID 2.
    pub fine_projection: [f32; 4],
    pub normal_power: f32,
    /// PS c89.x: roughness selected by the projected field.
    pub roughness_target: f32,
    /// PS c90.xy: affine remap applied to the projected field.
    pub roughness_remap: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct MeshSourcePreview {
    pub kind: &'static str,
    pub buffer_index: usize,
    pub technique: Option<TagHash>,
    pub index_start: u32,
    pub index_count: u32,
    pub primitive_type: u8,
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

    pub fn load_model_with_weapon_mods(
        cache: Arc<TagCache>,
        tag: TagHash,
        entry: &UEntryHeader,
        weapon_owner: TagHash,
        attachments: &[TagHash],
    ) -> Option<Self> {
        let attachments = attachments
            .iter()
            .copied()
            .map(|model_tag| WeaponModPreviewAttachment {
                model_tag,
                rarity: None,
                unique_id: 0.5,
            })
            .collect_vec();
        Self::load_model_with_weapon_mod_attachments(
            cache,
            tag,
            entry,
            tag,
            weapon_owner,
            &attachments,
        )
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
                |(_attachment, geometry, pose, rarity, unique_id)| ResolvedWeaponModAttachment {
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
const PATTERN_LOCAL_SCOPE_HASH: u32 = 0x811C9DC5;

// The pattern compiler hashes the six material parameters independently from
// their serialized vector positions. Those positions are intentionally
// shuffled between patterns, so the binding table is the source of truth.
const GEAR_DYE_COLOR_PARAMETERS: [u32; 6] = [
    0x1B3D64F3, 0x1B3D64F6, 0x1B3D64F0, 0x1B3D64F1, 0x1B3D64F7, 0x1B3D64F4,
];
const GEAR_DYE_ROUGHNESS_PARAMETERS: [u32; 6] = [
    0xBF1554A8, 0xBF1554AA, 0xBF1554AB, 0xBF1554AD, 0xBF1554AC, 0xBF1554AF,
];
const GEAR_DYE_METAL_PARAMETERS: [u32; 6] = [
    0xD5754C52, 0xD5754C50, 0xD5754C51, 0xD5754C57, 0xD5754C56, 0xD5754C55,
];

#[derive(Debug, Clone, Copy)]
struct ResolvedWeaponModAttachment {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ObjectSpaceTransform {
    rotation: [f32; 4],
    translation: [f32; 3],
    scale: f32,
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
    if pose.bone_index != 0 {
        let bone = weapon_skeleton_bone_transform(cache, weapon, pose.bone_index as usize)?;
        let local_translation = pose.translation.map(|value| value * bone.scale);
        let rotated_translation = rotate_quaternion(local_translation, bone.rotation);
        pose.translation =
            std::array::from_fn(|axis| bone.translation[axis] + rotated_translation[axis]);
        pose.rotation = multiply_quaternions(bone.rotation, pose.rotation);
    }
    Some(pose)
}

/// Geometry branch spawned by one visual-mod Pattern. Kept public so Models
/// integration and live-package verification use same nearest-branch rule as
/// preview assembly.
pub fn weapon_mod_geometry_tags(cache: &TagCache, modification: TagHash) -> Vec<TagHash> {
    if package_manager()
        .get_entry(modification)
        .is_some_and(|entry| entry.reference == CLASS_GEOMETRY_RESOURCE)
    {
        vec![modification]
    } else {
        pattern_nearest_geometry_tags(cache, modification)
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

#[cfg(test)]
pub(crate) fn debug_weapon_mod_visual_binding_words(
    cache: &TagCache,
    modification: TagHash,
) -> Vec<(TagHash, usize, [u32; 6])> {
    let endian = package_manager().version.endian();
    descendant_pattern_nodes(cache, modification, 8)
        .into_iter()
        .filter_map(|node| Some((node, package_manager().read_tag(node).ok()?)))
        .flat_map(|(node, data)| {
            (0..data.len().saturating_sub(23))
                .step_by(4)
                .filter_map(move |offset| {
                    (read_u32_at(&data, offset, endian) == Some(CLASS_WEAPON_MOD_VISUAL_BINDING))
                        .then(|| {
                            let words = std::array::from_fn(|index| {
                                read_u32_at(&data, offset + index * 4, endian).unwrap_or_default()
                            });
                            (node, offset, words)
                        })
                })
                .collect_vec()
        })
        .collect()
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
    let endian = package_manager().version.endian();
    let data = package_manager().read_tag(tag).ok()?;
    let arrays = scan_arrays(&data, endian);
    let hierarchy_index = arrays.iter().position(|array| {
        array.class == CLASS_SKELETON_NODE_HIERARCHY && array.count > bone_index
    })?;
    let hierarchy_count = arrays[hierarchy_index].count;
    let transforms = arrays
        .iter()
        .skip(hierarchy_index + 1)
        .find(|array| array.class == CLASS_SKELETON_TRANSFORMS && array.count == hierarchy_count)?;
    let record = array_records(&data, *transforms, 0x20)
        .get(bone_index)
        .copied()?;
    let rotation = read_vec4_f32(record, 0, endian)?;
    let translation = read_vec4_f32(record, 0x10, endian)?;
    Some(ObjectSpaceTransform {
        rotation,
        translation: [translation[0], translation[1], translation[2]],
        scale: translation[3],
    })
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
    // Marathon gear is exposed through top-level patterns. Pattern components
    // use CLASS_PATTERN_COMPONENT and are only implementation nodes beneath it.
    reference == CLASS_PATTERN
        || model_info_for_reference(reference).is_some_and(ModelTagInfo::is_catalog_entry)
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
                (attachment.rarity, attachment.unique_id),
            )
        })
        .collect::<rustc_hash::FxHashMap<_, _>>();
    let gear_dye_palette = (!attached_geometry.is_empty())
        .then(|| weapon_skin_gear_dye_palette(&cache, tag))
        .flatten();
    for (model_tag, _source, wireframe) in &mut parsed {
        assign_wireframe_material_textures(wireframe, &cache, &textures);
        if let Some((rarity, unique_id)) = attached_geometry.get(model_tag).copied() {
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
            // A selected skin supplies all six GearDye object channels. Mod
            // Patterns consume those exact materials; their other local
            // vectors are fallback/category data and must not be promoted to
            // a color offset.
            let attachment_palette = palette;
            for range in &mut wireframe.material_ranges {
                if let Some(dye) = range
                    .gear_dye_change_color_index
                    .and_then(|index| attachment_palette.get(index as usize).copied())
                {
                    range.textures.gear_dye = Some(dye);
                    range.textures.gear_dye_default =
                        range.technique.and_then(technique_default_gear_dye_color);
                    range.textures.gear_dye_palette = Some(attachment_palette);
                }
            }
        }
        if wireframe
            .material_ranges
            .iter()
            .any(|range| range.textures.animated_dither.is_some())
            && let Some(palette) =
                gear_dye_palette.or_else(|| weapon_skin_gear_dye_palette(&cache, tag))
        {
            for range in &mut wireframe.material_ranges {
                if range.textures.animated_dither.is_some()
                    && let Some(dye) = range
                        .gear_dye_change_color_index
                        .and_then(|index| palette.get(index as usize).copied())
                {
                    range.textures.gear_dye = Some(dye);
                    range.textures.gear_dye_default =
                        range.technique.and_then(technique_default_gear_dye_color);
                    range.textures.gear_dye_palette = Some(palette);
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
        geometry_parts: model_tags,
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
    // The common Tiger GearDye material reserves cbuffer outputs 8..13 for
    // the six change-color regions. Output 7 is initialized from inline
    // constant 7 and is the seventh/default (black-ID) material color.
    let gear_dye_shader = (8..=13).all(|output| {
        pixel.bytecode.expressions.iter().any(|expression| {
            expression.target == format!("output[{output}]")
                && expression.expression.contains("object_channel")
        })
    });
    let color = gear_dye_shader.then(|| pixel.inline_constants.get(7).copied())??;
    valid_dye_color(color).then_some(color)
}

fn weapon_skin_gear_dye_palette(
    cache: &TagCache,
    selected_pattern: TagHash,
) -> Option<[GearDyeMaterial; 6]> {
    let mut queue = std::collections::VecDeque::from([(selected_pattern, 0usize)]);
    let mut seen = rustc_hash::FxHashSet::default();
    seen.insert(selected_pattern);

    while let Some((tag, depth)) = queue.pop_front() {
        let entry = package_manager().get_entry(tag)?;
        if entry.reference == CLASS_PATTERN_COMPONENT
            && let Ok(data) = package_manager().read_tag(tag)
            && let Some(palette) = decode_weapon_skin_gear_dye_palette(&data)
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

fn decode_weapon_skin_gear_dye_palette(data: &[u8]) -> Option<[GearDyeMaterial; 6]> {
    let endian = package_manager().version.endian();
    let arrays = scan_arrays(data, endian);
    let singleton_vectors = arrays
        .iter()
        .copied()
        .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
        .collect_vec();
    if singleton_vectors.len() != 29 {
        return None;
    }
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
    let parameter_vector = |parameter: u32| {
        parameter_indices
            .get(&parameter)
            .and_then(|index| vectors.get(*index))
            .copied()
    };
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
    Some([
        read_f32(&record[0x0..0x4], endian),
        read_f32(&record[0x4..0x8], endian),
        read_f32(&record[0x8..0xc], endian),
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
        // Common-surface VS forwards raw shader-input POSITION/NORMAL directly
        // to PS procedural varyings. Preserve those before socket transforms.
        wireframe
            .procedural_positions
            .get_or_insert_with(|| wireframe.vertices.clone());
        if wireframe.procedural_normals.is_none() {
            wireframe.procedural_normals = wireframe.normals.clone();
        }
        for vertex in &mut wireframe.vertices {
            let rotated = rotate_quaternion(*vertex, pose.rotation);
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
    let mut normals = has_complete_normals.then(Vec::new);
    let mut procedural_positions = Some(Vec::new());
    let mut procedural_normals = has_complete_normals.then(Vec::new);
    let mut tangents = has_complete_tangents.then(Vec::new);
    let mut uvs = has_complete_uvs.then(Vec::new);
    let mut indices = Vec::new();
    let mut material_ranges = Vec::new();

    for (_tag, source, wireframe) in parts {
        let available_vertices = MAX_PREVIEW_VERTICES.saturating_sub(vertices.len());
        if available_vertices == 0 {
            break;
        }
        let copied_vertices = wireframe.vertices.len().min(available_vertices);
        let vertex_base = vertices.len() as u32;
        vertices.extend(wireframe.vertices.iter().copied().take(copied_vertices));
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

        let ranges = if wireframe.material_ranges.is_empty() {
            vec![WireframeMaterialRange {
                index_start: 0,
                index_count: wireframe.indices.len(),
                render_stage: None,
                technique: None,
                gear_dye_change_color_index: None,
                procedural_scale: 1.0,
                texture: None,
                textures: WireframeMaterialTextures::default(),
            }]
        } else {
            wireframe.material_ranges
        };
        for range in ranges {
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
                    render_stage: range.render_stage,
                    technique: range.technique,
                    gear_dye_change_color_index: range.gear_dye_change_color_index,
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
        normals,
        procedural_positions,
        procedural_normals,
        tangents,
        uvs,
        normal_format,
        tangent_format,
        indices,
        material_ranges,
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
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(tag, false) else {
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
    if wireframe.material_ranges.is_empty() {
        return;
    }

    for range in &mut wireframe.material_ranges {
        let Some(technique) = range.technique else {
            range.texture = textures.first().map(|(tag, _entry)| *tag);
            range.textures.color = range.texture;
            continue;
        };

        range.textures = material_textures_for_technique(technique, cache, textures);
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
    let normal_slot = material_normal_texture_slot(&candidates);
    let control_slot = material_control_texture_slot(&candidates, normal_slot);
    let investment_decal = investment_decal_for_technique(technique, &candidates);
    let direct_shared_color_atlas = direct_shared_color_atlas_for_technique(technique, &candidates);
    material.mod_wear = weapon_mod_wear_material(technique, &candidates);
    material.animated_dither = animated_dither_material(technique, &candidates);
    // Both families can expose eight PS textures, but slots 5..7 mean physical
    // age/wear when the TFX wear ABI is present. Never reinterpret those wear
    // resources as decorative contour inputs.
    material.gear_pattern = (material.mod_wear.is_none() && material.animated_dither.is_none())
        .then(|| gear_pattern_material(technique, &candidates))
        .flatten();

    for binding in &candidates {
        let binding = *binding;
        let mut role = material_texture_role(binding, normal_slot, control_slot);
        if investment_decal
            .as_ref()
            .is_some_and(|decal| decal.color() == binding.tag)
        {
            role = MaterialTextureRole::Color;
        } else if direct_shared_color_atlas == Some(binding.tag) {
            role = MaterialTextureRole::Color;
        } else if fallback_aux_texture(binding.tag) {
            role = MaterialTextureRole::Aux;
        }
        assign_material_texture(&mut material, binding.tag, role);
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
    } else if material.color.is_none()
        && let Some(surface) = animated_flat_material_for_technique(technique)
    {
        material.solid_color = Some(surface.color);
        material.solid_surface = Some([surface.roughness, surface.metalness]);
    }

    material
}

/// Resolve an opaque material whose authored colour is a shared atlas.
///
/// Shared atlases are normally technical resources and must not win generic
/// albedo guessing. An opaque decal technique, however, can bind that same
/// atlas as its sole PS t0 colour source. Direct-slot ownership makes that use
/// unambiguous without naming a weapon or technique tag.
fn direct_shared_color_atlas_for_technique(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<TagHash> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    if pixel
        .bytecode
        .expressions
        .iter()
        .any(|expression| expression.expression.contains("DecalSetTransform"))
    {
        return None;
    }
    let pixel_textures = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .copied()
        .unique_by(|binding| binding.slot)
        .collect_vec();
    let [atlas] = pixel_textures.as_slice() else {
        return None;
    };
    (atlas.slot == 0
        && texture_is_srgb(atlas.tag)
        && !matches!(render_state_for_technique(technique).blend, Some(26 | 27)))
    .then_some(atlas.tag)
}

/// Resolve Tiger's investment-decal pass.
///
/// Weapon and runner geometry carries decal quads with selector UVs already
/// baked into the mesh. Their pixel techniques read `DecalSetTransform` and
/// use either a sole colour atlas or a colour + opacity + detail texture ABI.
/// Treating these bindings as an ordinary material drops the authored stencil
/// and renders Quicktag's white fallback. Decode the pass from its blend state,
/// TFX extern, texture formats, and constant layout; no asset/tag rule is used.
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
    let reads_decal_transform = pixel
        .bytecode
        .expressions
        .iter()
        .any(|expression| expression.expression.contains("DecalSetTransform"));
    if !reads_decal_transform {
        return None;
    }

    let pixel_textures = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .copied()
        .sorted_by_key(|binding| binding.slot)
        .unique_by(|binding| binding.slot)
        .collect_vec();

    match pixel_textures.as_slice() {
        [atlas] => Some(InvestmentDecalResolution::Atlas(atlas.tag)),
        [color, mask]
            if texture_is_srgb(color.tag)
                && texture_is_single_channel(mask.tag)
                && pixel.inline_constants.len() > 10 =>
        {
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
            if texture_is_single_channel(detail.tag)
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

/// Decode animated inventory-surface branch from shader resources + TFX.
/// No weapon, skin, package, shader, or texture hash participates in matching.
fn animated_dither_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<AnimatedDitherMaterial> {
    let pixel_bindings = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .copied()
        .sorted_by_key(|binding| binding.slot)
        .unique_by(|binding| binding.slot)
        .collect_vec();
    let field = pixel_bindings.iter().find(|binding| binding.slot == 7)?.tag;
    let technical_mask = pixel_bindings.iter().find(|binding| binding.slot == 6)?.tag;
    if pixel_bindings.first()?.slot != 0
        || pixel_bindings.last()?.slot != 7
        || pixel_bindings.len() != 8
    {
        return None;
    }

    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    let expression = |target: &str| {
        pixel
            .bytecode
            .expressions
            .iter()
            .find(|expression| expression.target == target)
            .map(|expression| expression.expression.as_str())
    };
    if !expression("output[67]")?.contains("Frame+0x0")
        || !expression("output[66]")?.contains("Frame+0x0")
        || !expression("output[71]")?.contains("Frame+0x0")
    {
        return None;
    }

    let tfx = &pixel.constants;
    let inline = &pixel.inline_constants;
    let phase_speed = tfx.get(0)?[0];
    let secondary_scale = *tfx.get(10)?;
    let secondary_scroll = [tfx.get(11)?[0], tfx.get(12)?[0]];
    let primary_scale = *tfx.get(14)?;
    let primary_scroll = [tfx.get(15)?[0], tfx.get(16)?[0]];
    let waveform = [
        inline.get(68)?[0],
        inline.get(69)?[0],
        inline.get(70)?[0],
        inline.get(70)?[1],
    ];
    // With preview object channels at zero, TFX output c74 evaluates to this
    // product. Outputs c75 and c77 evaluate to one; c76 is the inline master
    // response. Keep c74 inside `(1 - wave*c74)`, exactly as the compiled PS.
    let authored_strength = tfx.get(2)?[0] * tfx.get(4)?[0] * tfx.get(5)?[0] * tfx.get(6)?[0];
    let response_scale = inline.get(76)?[0];
    let dot_detail = pixel_bindings
        .iter()
        .find(|binding| binding.slot == 5)
        .map(|binding| AnimatedDotDetailMaterial {
            texture: binding.tag,
            position_scale: inline[1][..3].try_into().expect("three-vector"),
            position_offset: inline[2][..3].try_into().expect("three-vector"),
            projection: inline[4],
            fine_projection: inline[5],
            normal_power: inline[3][0],
            roughness_target: inline[89][0],
            roughness_remap: inline[90][..2].try_into().expect("two-vector"),
        })
        .filter(|detail| {
            (1.0..=128.0).contains(&detail.normal_power)
                && detail.projection[0].abs() > 0.0001
                && detail.projection[1].abs() > 0.0001
                && detail.fine_projection[0].abs() > 0.0001
                && detail.fine_projection[1].abs() > 0.0001
                && (0.0..=1.0).contains(&detail.roughness_target)
                && detail.roughness_remap[1] > 0.0
        });
    let decoded = AnimatedDitherMaterial {
        field,
        technical_mask,
        technical_mask_remap: std::array::from_fn(|index| {
            let constant = inline[82 + index];
            [constant[0], constant[1]]
        }),
        dot_detail,
        phase_speed,
        primary_transform: [
            primary_scale[0],
            primary_scale[1],
            primary_scroll[0],
            primary_scroll[1],
        ],
        secondary_transform: [
            secondary_scale[0],
            secondary_scale[1],
            secondary_scroll[0],
            secondary_scroll[1],
        ],
        waveform,
        facing_response: [
            inline.get(73)?[0],
            inline.get(73)?[1],
            response_scale,
            authored_strength,
        ],
        surface_response: [
            inline.get(78)?[0],
            inline.get(79)?[0],
            inline.get(80)?[0],
            inline.get(81)?[0],
        ],
    };
    let values = [decoded.phase_speed]
        .into_iter()
        .chain(decoded.technical_mask_remap.into_iter().flatten())
        .chain(decoded.dot_detail.into_iter().flat_map(|detail| {
            detail
                .position_scale
                .into_iter()
                .chain(detail.position_offset)
                .chain(detail.projection)
                .chain(detail.fine_projection)
                .chain([detail.normal_power, detail.roughness_target])
                .chain(detail.roughness_remap)
        }))
        .chain(decoded.primary_transform)
        .chain(decoded.secondary_transform)
        .chain(decoded.waveform)
        .chain(decoded.facing_response)
        .chain(decoded.surface_response);
    (values.clone().all(f32::is_finite)
        && (0.01..=20.0).contains(&decoded.phase_speed)
        && decoded.primary_transform[..2]
            .iter()
            .all(|value| (0.1..=100.0).contains(&value.abs()))
        && decoded.secondary_transform[..2]
            .iter()
            .all(|value| (0.1..=100.0).contains(&value.abs()))
        && decoded.primary_transform[2..]
            .iter()
            .chain(&decoded.secondary_transform[2..])
            .all(|value| value.abs() <= 2.0)
        && decoded.waveform[0] > 0.0
        && decoded.waveform[1] > 0.0
        && decoded.facing_response[1] > 0.0
        && decoded.facing_response[2] > 0.0
        && decoded.surface_response[0] > 0.0
        && decoded.surface_response[1] > 0.0)
        .then_some(decoded)
}

fn texture_is_srgb(texture: TagHash) -> bool {
    Texture::load_data_d2(texture, false)
        .map(|(desc, _data, _comment)| format!("{:?}", desc.format).contains("Srgb"))
        .unwrap_or(false)
}

fn texture_is_single_channel(texture: TagHash) -> bool {
    Texture::load_data_d2(texture, false)
        .map(|(desc, _data, _comment)| {
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
    if !sends_frame_color || !sends_selector {
        return None;
    }

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
    Some(TexturelessFlatMaterial {
        color,
        roughness: 0.5,
        metalness,
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

/// Decode Tiger's textureless, TFX-driven G-buffer material family.
///
/// These techniques author their visible base colour in TFX output 1 rather
/// than in a texture. The pixel shader consumes that output as cbuffer[1].rgb;
/// falling back to Quicktag's white texture therefore produces solid white
/// panels over otherwise-correct investment decals. Evaluate the same TFX
/// program at the preview clock origin and use its authored linear colour.
/// Animated materials still receive a deterministic, valid frame instead of
/// an invented white albedo.
fn animated_flat_material_for_technique(technique: TagHash) -> Option<TexturelessFlatMaterial> {
    let entry = package_manager().get_entry(technique)?;
    let data = package_manager().read_tag(technique).ok()?;
    if texture_bindings_for_technique(&entry, &data)
        .into_iter()
        .any(|binding| binding.stage == "PS")
    {
        return None;
    }

    let preview = MaterialTagPreview::load(&entry, &data)?;
    let MaterialPreviewKind::Technique(preview) = preview.kind;
    let pixel = preview.stages.iter().find(|stage| stage.stage == "PS")?;
    if !pixel
        .bytecode
        .expressions
        .iter()
        .any(|expression| expression.target == "output[1]")
    {
        return None;
    }

    let object_channels = pixel
        .bytecode
        .ops
        .iter()
        .filter(|op| op.name == "push_object_channel")
        .filter_map(|op| {
            Some((
                u32::from_str_radix(op.detail.strip_prefix("0x")?, 16).ok()?,
                [0.0; 4],
            ))
        })
        .collect::<std::collections::HashMap<_, _>>();
    let extern_values = std::collections::HashMap::from([("Frame+0x0".to_string(), [0.0; 4])]);
    let (_bindings, expressions) = interpret_tfx_stack_with_runtime_inputs(
        &pixel.bytecode.ops,
        &pixel.constants,
        &object_channels,
        &extern_values,
    );
    let color = expressions
        .iter()
        .find(|expression| expression.target == "output[1]")?
        .value?;
    if !color.iter().all(|value| value.is_finite())
        || !color[..3].iter().all(|value| (0.0..=1.0).contains(value))
    {
        return None;
    }

    Some(TexturelessFlatMaterial {
        color: [color[0], color[1], color[2], color[3].clamp(0.0, 1.0)],
        roughness: 0.5,
        metalness: 0.0,
    })
}

const WEAPON_MOD_AGE_CHANNEL: u32 = 0x138D_E801;
const UNIQUE_ID_CHANNEL: u32 = 0xD358_3E54;
const WEAPON_MOD_SCRATCHES_PROJECTION_AGE_DELTA: usize = 10;
const WEAPON_MOD_SCRATCHES_REMAP_BASE_AGE_DELTA: usize = 9;
const WEAPON_MOD_SCRATCHES_REMAP_SCALE_AGE_DELTA: usize = 8;

/// Marathon's common weapon-part shader exposes extra surface inputs at PS
/// t5..t7. Inventory rarities share one Pattern and texture set. Runtime sends
/// the authored tier (1 Enhanced, 2 Deluxe, 3 Superior) through object channel
/// 0x138DE801; three TFX outputs select progressively cleaner shader branches.
/// Shader permutations shift every cbuffer register, so output indices are
/// discovered from the object-channel expressions instead of being hardcoded.
fn weapon_mod_wear_material(
    technique: TagHash,
    bindings: &[TechniqueTextureBinding],
) -> Option<WeaponModWearMaterial> {
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
    Some(WeaponModWearMaterial {
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
}

fn local_surface_texture_candidate(texture: TagHash, technique: TagHash) -> bool {
    if texture.pkg_id() != technique.pkg_id() || fallback_aux_texture(texture) {
        return false;
    }
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(texture, false) else {
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
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(texture, false) else {
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
    if Some(texture) == material.color
        || Some(texture) == material.normal
        || Some(texture) == material.emissive
        || Some(texture) == material.control
        || material.aux.contains(&texture)
        || material.layers.iter().any(|layer| {
            layer.color == Some(texture)
                || layer.normal == Some(texture)
                || layer.emissive == Some(texture)
        })
    {
        return;
    }

    match role {
        MaterialTextureRole::Color => assign_color_layer(material, texture),
        MaterialTextureRole::Normal => assign_normal_layer(material, texture),
        MaterialTextureRole::Control(channel) => {
            material.control = Some(texture);
            material.roughness_channel = channel;
        }
        _ => material.aux.push(texture),
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

fn material_control_texture_slot(
    bindings: &[TechniqueTextureBinding],
    normal_slot: Option<u32>,
) -> Option<(u32, u8)> {
    let max_slot = bindings.iter().map(|binding| binding.slot).max()?;
    let usable = |slot| {
        bindings.iter().any(|binding| {
            binding.slot == slot
                && !fallback_aux_texture(binding.tag)
                && material_control_texture_candidate(binding.tag)
        })
    };
    let usable_multichannel = |slot| {
        bindings.iter().any(|binding| {
            binding.slot == slot
                && !fallback_aux_texture(binding.tag)
                && material_control_texture_candidate(binding.tag)
                && !texture_preview_format(binding.tag).contains("Bc4")
        })
    };

    if max_slot >= 10 && usable_multichannel(2) {
        return Some((2, 3));
    }
    if max_slot <= 4 && normal_slot == Some(2) && usable(1) {
        return Some((1, 1));
    }
    if (5..10).contains(&max_slot) && usable_multichannel(3) {
        return Some((3, 4));
    }
    None
}

fn texture_preview_format(texture: TagHash) -> String {
    Texture::load_data_d2(texture, false)
        .map(|(desc, _data, _comment)| format!("{:?}", desc.format))
        .unwrap_or_default()
}

fn material_control_texture_candidate(texture: TagHash) -> bool {
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(texture, false) else {
        return false;
    };
    let format = format!("{:?}", desc.format);
    desc.depth == 1
        && desc.array_size == 1
        && desc.width > 1
        && desc.height > 1
        && !format.contains("Srgb")
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
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(texture, false) else {
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
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(texture, false) else {
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
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(texture, false) else {
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
    matches!(
        texture.0,
        0x80A60058
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
        .filter_map(|group| {
            parts
                .get(group.part_index as usize)
                .map(|part| (group, part))
        })
        .min_by_key(|(_group, part)| (lod_selection_rank(part.lod_category), part.index_start))
        .map(|(group, _part)| group)
        .or_else(|| groups.first())?;
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
            render_stage: None,
            technique: source.technique,
            gear_dye_change_color_index: None,
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
            render_stage: None,
            technique: source.technique,
            gear_dye_change_color_index: None,
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
        // for procedural facing masks (animated inventory skins). Keep it
        // byte-faithful there, while normalizing the separate lighting copy.
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
                render_stage: range.render_stage,
                technique: range.technique,
                gear_dye_change_color_index: range.gear_dye_change_color_index,
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
    render_stage: Option<u8>,
    technique: Option<TagHash>,
    gear_dye_change_color_index: Option<u8>,
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

fn geometry_preview_part_indices(data: &[u8], endian: Endian) -> Option<Vec<usize>> {
    // Marathon adds one render stage: 26 boundaries for 25 stage ranges.
    const STAGE_BOUNDARY_COUNT: usize = 26;

    let part_count = scan_arrays(data, endian)
        .into_iter()
        .find(|array| array.class == CLASS_GEOMETRY_PART)?
        .count;
    let buffer_set = scan_arrays(data, endian)
        .into_iter()
        .find(|array| array.class == CLASS_GEOMETRY_BUFFER_SET)?;
    let boundaries = (0..STAGE_BOUNDARY_COUNT)
        .map(|index| {
            data.get(buffer_set.data_offset + 0x30 + index * 2..)
                .map(|bytes| read_u16(bytes, endian) as usize)
        })
        .collect::<Option<Vec<_>>>()?;
    preview_part_indices_from_boundaries(&boundaries, part_count)
}

fn preview_part_indices_from_boundaries(
    boundaries: &[usize],
    part_count: usize,
) -> Option<Vec<usize>> {
    const PREVIEW_STAGES: [usize; 5] = [0, 1, 2, 6, 7];
    if boundaries.len() <= PREVIEW_STAGES.into_iter().max()? + 1
        || boundaries.windows(2).any(|pair| pair[0] > pair[1])
        || boundaries.last().copied()? > part_count
    {
        return None;
    }

    Some(
        PREVIEW_STAGES
            .into_iter()
            .flat_map(|stage| boundaries[stage]..boundaries[stage + 1])
            .unique()
            .collect(),
    )
}

fn preview_part_stages_from_boundaries(
    boundaries: &[usize],
    part_count: usize,
) -> Option<Vec<Option<u8>>> {
    const PREVIEW_STAGES: [usize; 5] = [0, 1, 2, 6, 7];
    if boundaries.len() <= PREVIEW_STAGES.into_iter().max()? + 1
        || boundaries.windows(2).any(|pair| pair[0] > pair[1])
        || boundaries.last().copied()? > part_count
    {
        return None;
    }

    let mut stages = vec![None; part_count];
    for stage in PREVIEW_STAGES {
        for part in boundaries[stage]..boundaries[stage + 1] {
            stages[part] = Some(stage as u8);
        }
    }
    Some(stages)
}

fn geometry_preview_part_stages(data: &[u8], endian: Endian) -> Option<Vec<Option<u8>>> {
    const STAGE_BOUNDARY_COUNT: usize = 26;
    let arrays = scan_arrays(data, endian);
    let part_count = arrays
        .iter()
        .find(|array| array.class == CLASS_GEOMETRY_PART)?
        .count;
    let buffer_set = arrays
        .iter()
        .find(|array| array.class == CLASS_GEOMETRY_BUFFER_SET)?;
    let boundaries = (0..STAGE_BOUNDARY_COUNT)
        .map(|index| {
            data.get(buffer_set.data_offset + 0x30 + index * 2..)
                .map(|bytes| read_u16(bytes, endian) as usize)
        })
        .collect::<Option<Vec<_>>>()?;
    preview_part_stages_from_boundaries(&boundaries, part_count)
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
            render_stage: range.render_stage,
            technique: Some(range.technique),
            gear_dye_change_color_index: (range.gear_dye_change_color_index <= 5)
                .then_some(range.gear_dye_change_color_index),
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

fn geometry_index_range_candidates(data: &[u8], endian: Endian) -> Vec<GeometryIndexRangePreview> {
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct GeometryPositionTransform {
    scale: [f32; 3],
    offset: [f32; 3],
    procedural_scale: f32,
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
    if !wireframe.position_format.starts_with("i16") {
        return;
    }
    for position in &mut wireframe.vertices {
        for axis in 0..3 {
            position[axis] = read_snorm_position(position[axis]) * transform.scale[axis]
                + transform.offset[axis];
        }
    }
    // Compiled common-surface VS writes input POSITION straight to its
    // procedural varying, while rendered position follows geometry
    // dequantization. Vertex fetch supplies R16G16B16A16_SNORM, so preserve
    // exactly that normalized pre-transform value for pattern/wear/animation.
    if let Some(positions) = &mut wireframe.procedural_positions {
        for position in positions {
            for axis in 0..3 {
                position[axis] = read_snorm_position(position[axis]) * transform.scale[axis]
                    + transform.offset[axis];
            }
        }
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

fn vertex_layout_stream_sets_for_any_mapping(layout_id: u8) -> Option<Vec<usize>> {
    for layout_tag in tags_by_class(CLASS_VERTEX_INPUT_LAYOUT_MAPPING) {
        if let Some(stream_sets) = vertex_layout_stream_sets(layout_tag, layout_id) {
            return Some(stream_sets);
        }
    }

    None
}

fn vertex_layout_stream_sets(layout_tag: TagHash, layout_id: u8) -> Option<Vec<usize>> {
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

            let stream_sets = [0x8, 0xc, 0x10, 0x14]
                .into_iter()
                .filter_map(|offset| read_u32_at(record, offset, endian))
                .filter(|set_index| *set_index != u32::MAX)
                .map(|set_index| set_index as usize)
                .collect_vec();
            return (!stream_sets.is_empty()).then_some(stream_sets);
        }
    }

    None
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

fn candidate_f32x2_uv(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexUvCandidate> {
    build_uv_candidate(data, stride, offset, UvFormat::F32x2, |bytes| {
        Some([
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
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
        normals: None,
        procedural_normals: None,
        tangents: None,
        uvs: None,
        normal_format: None,
        tangent_format: None,
        indices: vec![],
        material_ranges: vec![],
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

fn read_unorm16(data: &[u8], endian: Endian) -> f32 {
    read_u16(data, endian) as f32 / 65535.0
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_decorator_instances_with_normalized_quaternions() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let rotated = rotate_quaternion([1.0, 0.0, 0.0], [0.0, 0.0, half, half]);
        assert!(rotated[0].abs() < 0.0001);
        assert!((rotated[1] - 1.0).abs() < 0.0001);
        assert!(rotated[2].abs() < 0.0001);
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_geometry_resource_transforms() {
        init_goliath_test_package_manager();
        for tag in [
            TagHash(0x80B14039),
            TagHash(0x80B140B7),
            TagHash(0x80B140E6),
            TagHash(0x80B1477B),
        ] {
            let data = package_manager().read_tag(tag).expect("geometry resource");
            let endian = package_manager().version.endian();
            eprintln!("{tag} len={}", data.len());
            for offset in (0..data.len().min(0x130)).step_by(0x10) {
                let values = (0..4)
                    .filter_map(|component| {
                        data.get(offset + component * 4..offset + component * 4 + 4)
                    })
                    .map(|bytes| read_f32(bytes, endian))
                    .collect_vec();
                eprintln!("  {offset:03X}: {values:?}");
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_problem_surface_materials() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        for tag in [
            TagHash(0x80B140B7),
            TagHash(0x80B140E6),
            TagHash(0x80A9E3DE),
            TagHash(0x80A9E49B),
            TagHash(0x80AA03CA),
            TagHash(0x80B6CD7A),
        ] {
            let assembly = related_pattern_geometry_tags(&cache, tag);
            eprintln!("\n{tag}: assembly={assembly:?}");
            let entry = package_manager().get_entry(tag).expect("geometry entry");
            let data = package_manager().read_tag(tag).expect("geometry payload");
            let techniques = find_model_technique_entries(&cache, tag, &entry);
            let textures = find_model_textures(&cache, tag, &techniques);
            let (_, mut wireframe) = parse_model_wireframe(tag, &entry).expect("geometry preview");
            assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
            let parsed = geometry_primary_index_ranges(&data, package_manager().version.endian());
            eprintln!(
                "\n{tag}: ranges={} textures={}",
                parsed.len(),
                textures.len()
            );
            for (range, material) in parsed.iter().zip(&wireframe.material_ranges) {
                eprintln!(
                    "  part={} stage={:?} flags={:#010X} tech={} indices={}+{} selected color={:?} tint={:?} mask={:?} normal={:?} control={:?} aux={:?}",
                    range.part_index,
                    range.render_stage,
                    range.flags,
                    range.technique,
                    range.index_start,
                    range.index_count,
                    material.textures.color,
                    material.textures.color_tint,
                    material.textures.mask_palette,
                    material.textures.normal,
                    material.textures.control,
                    material.textures.aux,
                );
                if range.technique == TagHash(0x80B140CC) {
                    assert!(
                        material.textures.mask_palette.is_some(),
                        "hair mask shader must expose authored palette constants"
                    );
                }
                if range.technique == TagHash(0x80A9B3FA) {
                    assert!(
                        crate::material::is_sticker_proxy_technique(range.technique),
                        "solid investment proxy must obey Stickers toggle"
                    );
                }
                let technique_entry = package_manager()
                    .get_entry(range.technique)
                    .expect("technique entry");
                let technique_data = package_manager()
                    .read_tag(range.technique)
                    .expect("technique payload");
                if let Some(preview) =
                    crate::material::MaterialTagPreview::load(&technique_entry, &technique_data)
                {
                    let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
                    if let Some(stage) = preview.stages.iter().find(|stage| stage.stage == "PS") {
                        eprintln!(
                            "    shader={:?} constants={:?} inline={:?} cbuffer={:?}",
                            stage.shader,
                            stage.constants,
                            stage.inline_constants,
                            stage
                                .constant_buffer_preview
                                .as_ref()
                                .map(|buffer| &buffer.first_values),
                        );
                    }
                }
                for binding in texture_bindings_for_technique(&technique_entry, &technique_data)
                    .into_iter()
                    .filter(|binding| binding.stage == "PS")
                {
                    let descriptor = Texture::load_data_d2(binding.tag, false)
                        .map(|(desc, _, _)| {
                            format!("{}x{} {:?}", desc.width, desc.height, desc.format)
                        })
                        .unwrap_or_else(|_| "non-texture".into());
                    eprintln!("    PS t{}={} {descriptor}", binding.slot, binding.tag);
                }
                eprintln!(
                    "    TFX {:?}",
                    tfx_texture_bindings_for_technique(&technique_entry, &technique_data)
                        .into_iter()
                        .filter(|binding| binding.stage == "PS")
                        .map(|binding| format!(
                            "t{}:{}+{:?}",
                            binding.slot,
                            binding.source_scope.as_deref().unwrap_or("?"),
                            binding.source_offset
                        ))
                        .collect_vec()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages and GPU"]
    fn probes_goliath_face_texture_contact_sheet() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let tag = TagHash(0x80B140B7);
        let entry = package_manager().get_entry(tag).expect("geometry entry");
        let model = load_model_preview(cache, tag, &entry, "Geometry");

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU adapter");
        let required_features = adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            ..Default::default()
        }))
        .expect("GPU device");
        let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let renderer = eframe::egui_wgpu::Renderer::new(
            &device,
            target_format,
            eframe::egui_wgpu::RendererOptions::default(),
        );
        let render_state = eframe::egui_wgpu::RenderState {
            adapter,
            available_adapters: vec![],
            device,
            queue,
            target_format,
            renderer: Arc::new(eframe::egui::mutex::RwLock::new(renderer)),
        };

        let candidates = model
            .textures
            .iter()
            .map(|(texture, _)| *texture)
            .filter(|texture| {
                Texture::load_desc(*texture).is_ok_and(|desc| {
                    desc.kind() == crate::texture::TextureType::Texture2D
                        && format!("{:?}", desc.format).contains("Srgb")
                        && desc.width > 16
                        && desc.height > 16
                })
            })
            .unique()
            .collect_vec();
        let tile = 160_u32;
        let columns = 6_u32;
        let rows = (candidates.len() as u32).div_ceil(columns);
        let mut sheet = image::RgbaImage::new(columns * tile, rows * tile);
        let output = std::path::Path::new("target/quicktag-texture-probe");
        std::fs::create_dir_all(output).expect("probe directory");
        for (index, texture) in candidates.iter().copied().enumerate() {
            let loaded = Texture::load(&render_state, texture, false).expect("texture upload");
            let image = loaded.to_image(&render_state, 0).expect("texture capture");
            image
                .save(output.join(format!("{texture}.png")))
                .expect("texture export");
            let thumbnail = image.thumbnail(tile, tile).to_rgba8();
            let x = index as u32 % columns * tile + (tile - thumbnail.width()) / 2;
            let y = index as u32 / columns * tile + (tile - thumbnail.height()) / 2;
            image::imageops::overlay(&mut sheet, &thumbnail, x.into(), y.into());
            eprintln!("tile {index}: {texture}");
        }
        sheet
            .save(output.join(format!("{tag}-sheet.png")))
            .expect("contact sheet");
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_sample_model_uvs_and_textures() {
        init_goliath_test_package_manager();

        let tag = TagHash(0x80B14039);
        let entry = package_manager().get_entry(tag).expect("sample tag entry");
        assert_eq!(entry.reference, CLASS_GEOMETRY_RESOURCE);

        let (source, mut wireframe) = parse_model_wireframe(tag, &entry).expect("sample wireframe");
        eprintln!(
            "mesh kind={} technique={:?} layout={:?} ib={} vb0={} vb1={} index_start={} index_count={} lod={} wireframe verts={} indices={} source={} uv={:?}",
            source.kind,
            source.technique,
            source.input_layout_index,
            source.index_buffer,
            source.vertex0_buffer,
            source.vertex1_buffer,
            source.index_start,
            source.index_count,
            source.lod_category,
            wireframe.vertices.len(),
            wireframe.indices.len(),
            wireframe.source,
            wireframe.uv_format
        );
        assert_eq!(source.lod_category, 0, "sample preview should expose lod0");
        assert_eq!(source.input_layout_index, Some(7));
        let normal_layout =
            resolved_input_layout_vector(7, SEMANTIC_NORMAL, 0).expect("layout 7 normal");
        assert_eq!(normal_layout.buffer_index, 0);
        assert_eq!(normal_layout.offset, 8);
        assert_eq!(normal_layout.format, InputLayoutFormat::R16G16B16A16Snorm);
        let tangent_layout =
            resolved_input_layout_vector(7, SEMANTIC_TANGENT, 0).expect("layout 7 tangent");
        assert_eq!(tangent_layout.buffer_index, 0);
        assert_eq!(tangent_layout.offset, 16);
        assert_eq!(tangent_layout.format, InputLayoutFormat::R16G16B16A16Snorm);
        assert_eq!(source.technique, Some(TagHash(0x80B1247F)));
        let source_data = package_manager().read_tag(tag).expect("sample resource");
        for axis in 0..3 {
            let expected_min = read_f32(
                source_data
                    .get(0xe0 + axis * 4..0xe4 + axis * 4)
                    .expect("resource minimum"),
                package_manager().version.endian(),
            );
            let expected_max = read_f32(
                source_data
                    .get(0xf0 + axis * 4..0xf4 + axis * 4)
                    .expect("resource maximum"),
                package_manager().version.endian(),
            );
            assert!((wireframe.min[axis] - expected_min).abs() < 0.01);
            assert!((wireframe.max[axis] - expected_max).abs() < 0.01);
        }
        assert!(
            wireframe.uvs.as_ref().is_some_and(|uvs| !uvs.is_empty()),
            "sample must decode real UVs"
        );
        assert!(
            wireframe
                .uv_format
                .as_deref()
                .is_some_and(|uv| uv.contains("R16G16_SNORM") && uv.contains("layout Some(7)")),
            "sample must use declared geometry layout UVs, got {:?}",
            wireframe.uv_format
        );
        assert_eq!(
            wireframe.normals.as_ref().map(Vec::len),
            Some(wireframe.vertices.len()),
            "sample must decode authored normals"
        );
        assert_eq!(
            wireframe.tangents.as_ref().map(Vec::len),
            Some(wireframe.vertices.len()),
            "sample must decode authored tangents"
        );
        assert!(
            wireframe
                .normal_format
                .as_deref()
                .is_some_and(|format| format.contains("R16G16B16A16_SNORM")),
            "unexpected normal format: {:?}",
            wireframe.normal_format
        );
        assert!(
            wireframe
                .tangents
                .as_ref()
                .is_some_and(|tangents| tangents.iter().all(|tangent| tangent[3].abs() == 1.0)),
            "sample tangent handedness must be normalized"
        );

        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let assembled = related_pattern_geometry_tags(&cache, tag);
        for expected in [
            TagHash(0x80B14039),
            TagHash(0x80B140B7),
            TagHash(0x80B140E6),
        ] {
            assert!(
                assembled.contains(&expected),
                "root pattern assembly should include {expected}; got {assembled:?}"
            );
        }
        let assembled_preview = load_model_preview(cache.clone(), tag, &entry, "Geometry");
        assert!(
            assembled_preview
                .wireframe
                .as_ref()
                .is_some_and(|wireframe| wireframe.source.contains("assembled")),
            "root pattern geometry should merge into one preview"
        );
        assert!(
            assembled_preview
                .wireframe
                .as_ref()
                .is_some_and(|wireframe| {
                    wireframe.normals.as_ref().map(Vec::len) == Some(wireframe.vertices.len())
                        && wireframe.tangents.as_ref().map(Vec::len)
                            == Some(wireframe.vertices.len())
                }),
            "assembled preview should preserve complete authored tangent frames"
        );
        let assembled_bounds = assembled_preview
            .wireframe
            .as_ref()
            .expect("assembled bounds");
        assert!(
            assembled_bounds.max[2] - assembled_bounds.min[2] > 1.5
                && assembled_bounds.max[2] - assembled_bounds.min[2] < 2.5,
            "assembled character should retain authored world scale: {:?}..{:?}",
            assembled_bounds.min,
            assembled_bounds.max
        );

        let second_tag = TagHash(0x80B1477B);
        let second_entry = package_manager()
            .get_entry(second_tag)
            .expect("second character");
        let second = load_model_preview(cache.clone(), second_tag, &second_entry, "Geometry");
        let second_wireframe = second.wireframe.as_ref().expect("second assembled preview");
        let second_extent = (0..3)
            .map(|axis| second_wireframe.max[axis] - second_wireframe.min[axis])
            .fold(0.0_f32, f32::max);
        assert!(
            second_extent > 1.0 && second_extent < 3.0,
            "80B1477B sibling parts should share authored scale; extent={second_extent} bounds={:?}..{:?}",
            second_wireframe.min,
            second_wireframe.max
        );
        let techniques = find_model_technique_entries(&cache, tag, &entry);
        let technique_tag = TagHash(0x80B1247F);
        let technique_entry = package_manager()
            .get_entry(technique_tag)
            .expect("sample technique entry");
        let technique_data = package_manager()
            .read_tag(technique_tag)
            .expect("sample technique data");
        let technique_preview =
            crate::material::MaterialTagPreview::load(&technique_entry, &technique_data)
                .expect("sample technique preview");
        let crate::material::MaterialPreviewKind::Technique(technique_preview) =
            technique_preview.kind;
        let pixel_stage = technique_preview
            .stages
            .iter()
            .find(|stage| stage.stage == "PS")
            .expect("sample pixel stage");
        assert_eq!(pixel_stage.shader, Some(TagHash(0x80B12478)));
        assert_eq!(pixel_stage.textures.len(), 15);
        assert_eq!(pixel_stage.textures[0].slot, 0);
        assert_eq!(
            pixel_stage.textures[0].texture.resolved,
            Some(TagHash(0x80B14069))
        );
        assert!(pixel_stage.bytecode_len > 0);
        let pixel_bytecode = read_array(
            &technique_data,
            0x278 + 0x20,
            1,
            package_manager().version.endian(),
        )
        .expect("sample pixel bytecode");
        assert!(
            pixel_stage.bytecode.decoded_ops > 0,
            "undecoded Marathon TFX prefix: {:02X?}",
            &pixel_bytecode[..pixel_bytecode.len().min(64)]
        );
        eprintln!(
            "techniques={:?}",
            techniques
                .iter()
                .take(16)
                .map(|(tag, _)| format!("{tag}"))
                .collect_vec()
        );
        assert!(!techniques.is_empty(), "sample must resolve techniques");
        assert_eq!(
            techniques.first().map(|(tag, _)| *tag),
            source.technique,
            "selected geometry technique should rank first"
        );

        let textures = find_model_textures(&cache, tag, &techniques);
        eprintln!(
            "textures={:?}",
            textures
                .iter()
                .take(32)
                .map(|(tag, entry)| {
                    let desc = crate::texture::Texture::load_data_d2(*tag, false)
                        .map(|(desc, _, _)| {
                            format!(
                                "{}x{}x{} {:?}",
                                desc.width, desc.height, desc.depth, desc.format
                            )
                        })
                        .unwrap_or_else(|err| format!("load_err={err}"));
                    format!("{tag}:{}:{desc}", entry.file_size)
                })
                .collect_vec()
        );
        assert!(!textures.is_empty(), "sample must resolve textures");
        assert!(
            textures.iter().any(|(tag, _)| *tag == TagHash(0x80B14064)),
            "sample must retain body primary texture"
        );
        assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
        let assigned_textures = wireframe
            .material_ranges
            .iter()
            .filter_map(|range| range.texture)
            .unique()
            .collect_vec();
        eprintln!(
            "assigned_textures={:?}",
            assigned_textures
                .iter()
                .map(|tag| format!("{tag}"))
                .collect_vec()
        );
        let assigned_normals = wireframe
            .material_ranges
            .iter()
            .filter_map(|range| range.textures.normal)
            .unique()
            .collect_vec();
        let assigned_emissive = wireframe
            .material_ranges
            .iter()
            .filter_map(|range| range.textures.emissive)
            .unique()
            .collect_vec();
        let assigned_layer_colors = wireframe
            .material_ranges
            .iter()
            .flat_map(|range| range.textures.layers.iter())
            .filter_map(|layer| layer.color)
            .unique()
            .collect_vec();
        eprintln!(
            "assigned_normals={:?} assigned_emissive={:?} assigned_layer_colors={:?}",
            assigned_normals
                .iter()
                .map(|tag| format!("{tag}"))
                .collect_vec(),
            assigned_emissive
                .iter()
                .map(|tag| format!("{tag}"))
                .collect_vec(),
            assigned_layer_colors
                .iter()
                .map(|tag| format!("{tag}"))
                .collect_vec()
        );
        for expected in [
            TagHash(0x80B14064),
            TagHash(0x80B14069),
            TagHash(0x80B1405F),
            TagHash(0x80B14047),
        ] {
            assert!(
                assigned_textures.contains(&expected),
                "sample material ranges should use {expected}"
            );
        }
        for expected in [
            TagHash(0x80B14064),
            TagHash(0x80B14069),
            TagHash(0x80B1405F),
            TagHash(0x80B14047),
        ] {
            assert!(
                assigned_layer_colors.contains(&expected),
                "sample material layers should preserve {expected}"
            );
        }
        assert!(
            !assigned_normals.is_empty() || !assigned_emissive.is_empty(),
            "sample material ranges should classify non-color material textures"
        );

        let complete_tag = TagHash(0x80B140B7);
        let complete_entry = package_manager()
            .get_entry(complete_tag)
            .expect("sample complete geometry entry");
        let Some((complete_source, complete_wireframe)) =
            parse_model_wireframe(complete_tag, &complete_entry)
        else {
            panic!("sample complete geometry should parse");
        };
        eprintln!(
            "complete mesh kind={} technique={:?} index_start={} index_count={} lod={} wireframe indices={}",
            complete_source.kind,
            complete_source.technique,
            complete_source.index_start,
            complete_source.index_count,
            complete_source.lod_category,
            complete_wireframe.indices.len()
        );
        assert_eq!(complete_source.lod_category, 0);
        assert!(
            complete_source.index_count > 3849,
            "geometry resources should merge all primary ranges, not just the first range"
        );
        assert!(
            complete_wireframe.indices.len() > 3849,
            "wireframe should include all primary ranges for 80B140B7"
        );
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_problem_model_materials() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());

        for tag in [
            TagHash(0x80B14039),
            TagHash(0x80B140B7),
            TagHash(0x80B140E6),
            TagHash(0x80B6C372),
            TagHash(0x80B6E78F),
            TagHash(0x80B6CDB3),
            TagHash(0x80B6CDB4),
        ] {
            let Some(entry) = package_manager().get_entry(tag) else {
                eprintln!("{tag}: missing");
                continue;
            };
            let Some((source, mut wireframe)) = parse_model_wireframe(tag, &entry) else {
                eprintln!("{tag}: no wireframe");
                continue;
            };
            let techniques = find_model_technique_entries(&cache, tag, &entry);
            let textures = find_model_textures(&cache, tag, &techniques);
            assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
            let assigned_colors = wireframe
                .material_ranges
                .iter()
                .filter_map(|range| range.texture)
                .unique()
                .collect_vec();
            assert!(
                !assigned_colors.contains(&TagHash(0x80A60058)),
                "{tag}: global decal atlas must not become preview albedo"
            );
            assert!(
                !assigned_colors.iter().copied().any(fallback_aux_texture),
                "{tag}: technical fallback textures must stay out of albedo"
            );
            if tag == TagHash(0x80B140B7) {
                let eye_material =
                    material_textures_for_technique(TagHash(0x80B14088), &cache, &textures);
                assert_eq!(
                    material_textures_for_technique(TagHash(0x80B1248B), &cache, &textures).color,
                    Some(TagHash(0x80B14062)),
                    "mask-only detail material should retain its local BC4 primary slot"
                );
                assert_eq!(
                    eye_material.color,
                    Some(TagHash(0x80B140BB)),
                    "eye material should promote its local tint texture"
                );
                assert_eq!(
                    eye_material.color_tint,
                    [196, 66, 13, 255],
                    "eye BC4 mask should use its authored orange inline tint"
                );
                assert_eq!(
                    material_textures_for_technique(TagHash(0x80B124CC), &cache, &textures).color,
                    Some(TagHash(0x80B14062)),
                    "decal material should promote its local BC4 mask"
                );
            }
            if tag == TagHash(0x80B140E6) {
                assert_eq!(
                    material_textures_for_technique(TagHash(0x80B140CC), &cache, &textures).color,
                    Some(TagHash(0x80A613F7)),
                    "hair mask 80B140EF must not replace its slot-0 surface texture"
                );
            }
            eprintln!(
                "{tag}: class={:08X} kind={} source={} tech={:?} ranges={} vertices={} indices={} uv={:?}",
                entry.reference,
                source.kind,
                wireframe.source,
                source.technique,
                wireframe.material_ranges.len(),
                wireframe.vertices.len(),
                wireframe.indices.len(),
                wireframe.uv_format
            );
            eprintln!(
                "{tag}: assigned_colors={:?}",
                assigned_colors
                    .iter()
                    .map(|tag| format!("{tag}"))
                    .collect_vec()
            );
            eprintln!(
                "{tag}: textures={:?}",
                textures
                    .iter()
                    .map(|(texture, _)| format!("{texture}"))
                    .take(16)
                    .collect_vec()
            );
            for (technique, _entry) in techniques.iter().take(4) {
                let material = material_textures_for_technique(*technique, &cache, &textures);
                let expected_normal = match technique.0 {
                    0x80B1247F => Some(TagHash(0x80B1405D)),
                    0x80B12499 => Some(TagHash(0x80B14057)),
                    0x80B124B5 => Some(TagHash(0x80B14060)),
                    0x80B124C1 => Some(TagHash(0x80B14049)),
                    0x80B1407C => Some(TagHash(0x80B140EB)),
                    0x80B140CC => Some(TagHash(0x80B140F1)),
                    0x80A9F7E6 => Some(TagHash(0x80A9F81A)),
                    _ => None,
                };
                if let Some(expected_normal) = expected_normal {
                    assert_eq!(
                        material.normal,
                        Some(expected_normal),
                        "{technique}: normal must follow its shader-family resource slot"
                    );
                }
                let expected_control = match technique.0 {
                    0x80B1247F => Some((TagHash(0x80B14052), 3)),
                    0x80B12499 => Some((TagHash(0x80B14059), 3)),
                    0x80B124B5 => Some((TagHash(0x80B14051), 3)),
                    0x80B1407C => Some((TagHash(0x80B140BC), 3)),
                    0x80B140CC => Some((TagHash(0x80B140EF), 1)),
                    0x80A9F7E6 => Some((TagHash(0x80A9F814), 4)),
                    _ => None,
                };
                if let Some((expected_control, expected_channel)) = expected_control {
                    assert_eq!(
                        (material.control, material.roughness_channel),
                        (Some(expected_control), expected_channel),
                        "{technique}: roughness control must follow proven DXIL channel"
                    );
                }
                let expected_sampler = match technique.0 {
                    0x80B1247F | 0x80B12499 | 0x80B124B5 | 0x80B124C1 | 0x80B1407C => {
                        Some(TagHash(0x80A60082))
                    }
                    0x80B14088 | 0x80B140CC | 0x80A9F7E6 => Some(TagHash(0x80A60020)),
                    _ => None,
                };
                if let Some(expected_sampler) = expected_sampler {
                    assert_eq!(
                        material.sampler,
                        Some(expected_sampler),
                        "{technique}: PS sampler must follow TFX sampler binding"
                    );
                }
                if *technique != TagHash(0x80B14088) {
                    assert_eq!(
                        material.color_tint,
                        [255, 255, 255, 255],
                        "{tag}: unresolved TFX constants must not become speculative albedo tint"
                    );
                }
                assert_eq!(material.emissive_strength, 0);
                let render_state = render_state_for_technique(*technique);
                eprintln!(
                    "{tag}: technique={technique} state={render_state:?} color={:?} normal={:?} emissive={:?} control={:?}.{} sampler={:?} aux={:?} layers={:?}",
                    material.color,
                    material.normal,
                    material.emissive,
                    material.control,
                    material.roughness_channel,
                    material.sampler,
                    material.aux,
                    material.layers
                );
                if let Some(entry) = package_manager().get_entry(*technique)
                    && let Ok(data) = package_manager().read_tag(*technique)
                {
                    let direct_bindings = texture_bindings_for_technique(&entry, &data)
                        .into_iter()
                        .filter(|binding| binding.stage == "PS")
                        .collect_vec();
                    let normal_slot = material_normal_texture_slot(&direct_bindings);
                    let control_slot = material_control_texture_slot(&direct_bindings, normal_slot);
                    eprintln!(
                        "{tag}: direct={:?}",
                        direct_bindings
                            .into_iter()
                            .map(|binding| format!(
                                "{}:{} -> role {:?}",
                                binding.slot,
                                binding.tag,
                                material_texture_role(binding, normal_slot, control_slot)
                            ))
                            .collect_vec()
                    );
                    eprintln!(
                        "{tag}: tfx={:?}",
                        tfx_texture_bindings_for_technique(&entry, &data)
                            .into_iter()
                            .filter(|binding| binding.stage == "PS")
                            .map(|binding| format!(
                                "slot {} {:?}+{:?}",
                                binding.slot, binding.source_scope, binding.source_offset
                            ))
                            .collect_vec()
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires current installed Marathon packages"]
    fn decodes_updated_weapon_flat_materials_without_gear_dye() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let skin = TagHash(0x80A9F63F);
        let techniques = find_model_technique_entries(
            &cache,
            skin,
            &package_manager().get_entry(skin).expect("V85 Vox Nocturna"),
        );
        let textures = find_model_textures(&cache, skin, &techniques);
        let geometry = TagHash(0x80A9F63A);
        let geometry_data = package_manager()
            .read_tag(geometry)
            .expect("V85 Vox Nocturna geometry");
        let selected =
            geometry_primary_index_ranges(&geometry_data, package_manager().version.endian());
        assert!(selected.iter().any(|range| {
            range.technique == TagHash(0x80A9FBB1)
                && range.index_start == 0
                && range.index_count == 2996
        }));
        let preview = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            cache.clone(),
            skin,
            &package_manager().get_entry(skin).expect("V85 Vox Nocturna"),
            skin,
            skin,
            &[],
        )
        .expect("V85 model preview");
        let GeometryPreviewKind::Model(model) = preview.kind else {
            panic!("V85 must load as model")
        };
        let wireframe = model.wireframe.expect("V85 wireframe");
        let supplemental = wireframe
            .material_ranges
            .iter()
            .find(|range| range.technique == Some(TagHash(0x80A9FBB1)))
            .expect("procedural surface must be rendered");
        assert_eq!(
            supplemental.textures.solid_color,
            Some([0.44368514, 0.18800727, 0.035853356, 1.0])
        );
        assert_eq!(supplemental.textures.solid_surface, Some([0.54, 0.08]));
        assert_eq!(supplemental.textures.gear_dye, None);
        let dark = material_textures_for_technique(TagHash(0x80A9F5CD), &cache, &textures);
        assert_eq!(dark.color, None);
        assert_eq!(dark.solid_surface, Some([0.5, 0.0]));
        assert!(dark.solid_color.is_some_and(|color| {
            color[..3]
                .iter()
                .all(|value| (*value - 0.09845916).abs() < 0.000001)
        }));
        assert_eq!(dark.gear_dye, None);

        let light = material_textures_for_technique(TagHash(0x80A9FBE0), &cache, &textures);
        assert_eq!(light.color, None);
        assert_eq!(light.solid_surface, Some([0.5, 0.25]));
        assert!(light.solid_color.is_some_and(|color| {
            color[..3]
                .iter()
                .all(|value| (*value - 0.89789754).abs() < 0.000001)
        }));
        assert_eq!(light.gear_dye, None);

        let base = material_textures_for_technique(TagHash(0x80A9F5D8), &cache, &textures);
        assert!(base.color.is_some());
        assert_eq!(base.solid_color, None);
        assert_eq!(base.solid_surface, None);
        assert_eq!(base.gear_dye, None);
    }

    #[test]
    #[ignore = "requires current installed Marathon packages"]
    fn decodes_textureless_tfx_companion_materials() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();

        let animated = material_textures_for_technique(TagHash(0x80A9B87F), &cache, &[]);
        assert_eq!(animated.color, None);
        assert_eq!(animated.solid_color, Some([1.0, 0.617207, 0.040915, 1.0]));
        assert_eq!(animated.solid_surface, Some([0.5, 0.0]));

        let channel_driven = material_textures_for_technique(TagHash(0x80A9B889), &cache, &[]);
        assert_eq!(channel_driven.color, None);
        assert_eq!(channel_driven.solid_color, Some([0.0, 0.4, 1.0, 1.0]));
        assert_eq!(channel_driven.solid_surface, Some([0.5, 0.0]));

        let shared_atlas = material_textures_for_technique(TagHash(0x80A60033), &cache, &[]);
        assert_eq!(shared_atlas.color, Some(TagHash(0x80A60058)));
        assert_eq!(shared_atlas.solid_color, None);
        assert!(shared_atlas.authored_shared_atlas);

        let blended_atlas = material_textures_for_technique(TagHash(0x80A9B0ED), &cache, &[]);
        assert_eq!(blended_atlas.color, Some(TagHash(0x80A60058)));
        assert!(blended_atlas.authored_shared_atlas);
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_visible_render_states() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let mut counts = std::collections::BTreeMap::new();
        let mut techniques = rustc_hash::FxHashSet::default();

        for (&tag, _scan) in &cache.hashes {
            let Some(entry) = package_manager().get_entry(tag) else {
                continue;
            };
            if entry.reference != CLASS_GEOMETRY_RESOURCE {
                continue;
            }
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            for part in geometry_primary_index_ranges(&data, package_manager().version.endian()) {
                if techniques.insert(part.technique) {
                    let state = render_state_for_technique(part.technique);
                    *counts
                        .entry((
                            state.blend,
                            state.depth_stencil,
                            state.rasterizer,
                            state.depth_bias,
                        ))
                        .or_insert(0usize) += 1;
                }
            }
        }

        eprintln!("visible render states ({})", techniques.len());
        for (state, count) in counts {
            eprintln!("  {state:?}: {count}");
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_render_global_scopes() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        for class in [0x8080B61C, 0x80808070, 0x80808075, 0x808031DC] {
            eprintln!("class {class:08X}");
            for (tag, entry) in package_manager().get_all_by_reference(class) {
                let scan = cache.hashes.get(&tag);
                eprintln!(
                    "  {tag} len={} strings={:?} children={:?}",
                    entry.file_size,
                    scan.map(|scan| &scan.raw_strings),
                    scan.into_iter()
                        .flat_map(|scan| scan.file_hashes.iter())
                        .map(|child| {
                            let child_class = package_manager()
                                .get_entry(child.hash)
                                .map(|entry| entry.reference)
                                .unwrap_or_default();
                            format!("{}@0x{:X}:{child_class:08X}", child.hash, child.offset)
                        })
                        .collect_vec()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_gear_dye_scope_buffers() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let endian = package_manager().version.endian();

        for (scope, entry) in package_manager().get_all_by_reference(0x808031DC) {
            let Some(name) = cache
                .hashes
                .get(&scope)
                .and_then(|scan| scan.raw_strings.first())
                .filter(|name| name.starts_with("gear_dye"))
            else {
                continue;
            };
            let data = package_manager().read_tag(scope).expect("scope data");
            if name == "gear_dye_skin" && scope.pkg_id() == 0x13A {
                for (line, bytes) in data[..data.len().min(0x100)].chunks(16).enumerate() {
                    eprintln!(
                        "{scope} {:04X}: {}",
                        line * 16,
                        bytes.iter().map(|byte| format!("{byte:02X}")).join(" ")
                    );
                }
                eprintln!(
                    "{scope} arrays={:?}",
                    scan_arrays(&data, endian)
                        .into_iter()
                        .map(|array| (
                            array.class,
                            array.count,
                            array.data_offset,
                            array.end_offset
                        ))
                        .collect_vec()
                );
            }
            let constant_buffer = data
                .get(0xAC..0xB0)
                .map(|bytes| TagHash(read_u32(bytes, endian)))
                .filter(|tag| tag.is_some());
            let payload = constant_buffer
                .and_then(|header| package_manager().get_entry(header))
                .map(|entry| TagHash(entry.reference))
                .and_then(|payload| package_manager().read_tag(payload).ok())
                .unwrap_or_default();
            let stages = crate::material::scope_stages(&entry, &data);
            eprintln!(
                "{scope} {name} len={} cbuffer={constant_buffer:?} payload={} values={:?} stages={:?}",
                entry.file_size,
                payload.len(),
                payload
                    .chunks_exact(16)
                    .take(24)
                    .map(|chunk| [
                        read_f32(&chunk[0..4], endian),
                        read_f32(&chunk[4..8], endian),
                        read_f32(&chunk[8..12], endian),
                        read_f32(&chunk[12..16], endian),
                    ])
                    .collect_vec(),
                stages
                    .iter()
                    .map(|stage| (
                        stage.stage,
                        stage.bytecode.decoded_ops,
                        stage.bytecode.unknown_ops,
                        stage
                            .bytecode
                            .externs
                            .iter()
                            .map(|external| (
                                external.scope.clone(),
                                external.byte_offset,
                                external.value_type,
                            ))
                            .collect_vec(),
                        stage
                            .bytecode
                            .expressions
                            .iter()
                            .map(|expression| (
                                expression.target.clone(),
                                expression.expression.clone(),
                            ))
                            .collect_vec(),
                        stage
                            .bytecode
                            .ops
                            .iter()
                            .map(|op| (op.name, op.detail.clone()))
                            .collect_vec(),
                    ))
                    .collect_vec()
            );
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_character_part_metadata() {
        init_goliath_test_package_manager();
        for tag in [
            TagHash(0x80B14039),
            TagHash(0x80B140B7),
            TagHash(0x80B140E6),
            TagHash(0x80B6CD7A),
            TagHash(0x80AA03CA),
        ] {
            let data = package_manager().read_tag(tag).expect("geometry resource");
            let endian = package_manager().version.endian();
            eprintln!("{tag}: arrays");
            for array in scan_arrays(&data, endian) {
                eprintln!(
                    "  class={:08X} {} count={} data=0x{:X}..0x{:X}",
                    array.class,
                    get_class_by_id(array.class)
                        .map(|class| class.name)
                        .unwrap_or_else(|| "unknown".into()),
                    array.count,
                    array.data_offset,
                    array.end_offset
                );
                if array.class == CLASS_GEOMETRY_BUFFER_SET {
                    for (line, bytes) in data[array.data_offset..array.end_offset]
                        .chunks(16)
                        .enumerate()
                    {
                        eprintln!(
                            "    {:04X}: {}",
                            line * 16,
                            bytes.iter().map(|byte| format!("{byte:02X}")).join(" ")
                        );
                    }
                }
            }
            eprintln!("{tag}: parts");
            eprintln!(
                "{tag}: preview part indices={:?}",
                geometry_preview_part_indices(&data, endian)
            );
            for part in geometry_index_range_candidates(&data, endian) {
                eprintln!(
                    "  off=0x{:X} tech={} variant={} idx={}+{} prim={} flags=0x{:08X} dye={} lod={} run={}",
                    part.record_offset,
                    part.technique,
                    part.variant_shader_index,
                    part.index_start,
                    part.index_count,
                    part.primitive_type,
                    part.flags,
                    part.gear_dye_change_color_index,
                    part.lod_category,
                    part.lod_run
                );
            }
            eprintln!(
                "{tag}: selected={:?}",
                geometry_primary_index_ranges(&data, endian)
                    .iter()
                    .map(|part| (part.part_index, part.technique, part.lod_category))
                    .collect_vec()
            );
            for technique in geometry_primary_index_ranges(&data, endian)
                .into_iter()
                .map(|part| part.technique)
                .unique()
            {
                let Some(entry) = package_manager().get_entry(technique) else {
                    continue;
                };
                let Ok(technique_data) = package_manager().read_tag(technique) else {
                    continue;
                };
                let scopes = crate::material::MaterialTagPreview::load(&entry, &technique_data)
                    .map(|preview| match preview.kind {
                        crate::material::MaterialPreviewKind::Technique(technique) => {
                            technique.used_scope_names()
                        }
                    })
                    .unwrap_or_default();
                eprintln!(
                    "  material {technique}: scopes={scopes:?} state={:?} ps={:?}",
                    render_state_for_technique(technique),
                    texture_bindings_for_technique(&entry, &technique_data)
                        .into_iter()
                        .filter(|binding| binding.stage == "PS")
                        .map(|binding| (binding.slot, binding.tag))
                        .collect_vec()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_character_pattern_material_data() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let mut frontier = vec![TagHash(0x80B14039)];
        let mut seen = rustc_hash::FxHashSet::default();

        while let Some(tag) = frontier.pop() {
            if !seen.insert(tag) {
                continue;
            }
            let Some(entry) = package_manager().get_entry(tag) else {
                continue;
            };
            let class = get_class_by_id(entry.reference)
                .map(|class| class.name)
                .unwrap_or_else(|| "unknown".into());
            eprintln!("{tag}: class={:08X} {class}", entry.reference);
            let Some(scan) = cache.hashes.get(&tag) else {
                continue;
            };
            eprintln!("  parents={:?}", scan.references);
            eprintln!(
                "  children={:?}",
                scan.file_hashes
                    .iter()
                    .map(|child| {
                        let class = package_manager()
                            .get_entry(child.hash)
                            .map(|entry| entry.reference)
                            .unwrap_or_default();
                        format!("{}:{class:08X}", child.hash)
                    })
                    .collect_vec()
            );

            if matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT) {
                let data = package_manager().read_tag(tag).expect("pattern payload");
                eprintln!("  payload={} bytes", data.len());
                for (line, bytes) in data[..data.len().min(0x200)].chunks(16).enumerate() {
                    eprintln!(
                        "    {:04X}: {}",
                        line * 16,
                        bytes.iter().map(|byte| format!("{byte:02X}")).join(" ")
                    );
                }
                for parent in &scan.references {
                    if package_manager().get_entry(*parent).is_some_and(|entry| {
                        matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                    }) {
                        frontier.push(*parent);
                    }
                }
                for child in &scan.file_hashes {
                    if package_manager()
                        .get_entry(child.hash)
                        .is_some_and(|entry| {
                            matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                        })
                    {
                        frontier.push(child.hash);
                    }
                }
            } else {
                for parent in &scan.references {
                    if package_manager().get_entry(*parent).is_some_and(|entry| {
                        matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                    }) {
                        frontier.push(*parent);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_pattern_value_records() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        for component in (0x80B1403A..=0x80B14042).map(TagHash) {
            let Some(scan) = cache.hashes.get(&component) else {
                continue;
            };
            eprintln!("component {component}");
            for child in scan.file_hashes.iter().map(|child| child.hash).unique() {
                let Some(entry) = package_manager().get_entry(child) else {
                    continue;
                };
                if !matches!(
                    entry.reference,
                    0x8080BA53 | 0x8080BAF8 | CLASS_GEOMETRY_RESOURCE
                ) {
                    continue;
                }
                let data = package_manager().read_tag(child).unwrap_or_default();
                eprintln!(
                    "  {child} class={:08X} len={} bytes={} f32={:?}",
                    entry.reference,
                    data.len(),
                    data.iter().map(|byte| format!("{byte:02X}")).join(" "),
                    data.chunks_exact(4)
                        .map(|bytes| read_f32(bytes, package_manager().version.endian()))
                        .collect_vec()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_character_shader_reflection_strings() {
        init_goliath_test_package_manager();
        for texture in [
            TagHash(0x80B14062),
            TagHash(0x80B140BB),
            TagHash(0x80B140BF),
            TagHash(0x80B140EF),
            TagHash(0x80B140F1),
            TagHash(0x80A613F7),
        ] {
            let (desc, data, _) = Texture::load_data_d2(texture, false).expect("texture");
            eprintln!(
                "texture={texture} {}x{} {:?} bytes={}",
                desc.width,
                desc.height,
                desc.format,
                data.len()
            );
        }
        for technique in [
            TagHash(0x80B1247F),
            TagHash(0x80B1407C),
            TagHash(0x80B14088),
            TagHash(0x80B14094),
            TagHash(0x80B140CC),
            TagHash(0x80A9F7E6),
            TagHash(0x80B6CD83),
        ] {
            let entry = package_manager().get_entry(technique).expect("technique");
            let data = package_manager()
                .read_tag(technique)
                .expect("technique data");
            let preview = crate::material::MaterialTagPreview::load(&entry, &data)
                .expect("technique preview");
            let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
            let Some(shader) = preview
                .stages
                .iter()
                .find(|stage| stage.stage == "PS")
                .and_then(|stage| stage.shader)
            else {
                continue;
            };
            let shader_entry = package_manager().get_entry(shader).expect("shader header");
            let shader_data = package_manager()
                .read_tag(TagHash(shader_entry.reference))
                .expect("shader bytecode");
            let strings = shader_data
                .split(|byte| !byte.is_ascii_graphic() && *byte != b' ')
                .filter(|bytes| bytes.len() >= 4)
                .filter_map(|bytes| std::str::from_utf8(bytes).ok())
                .unique()
                .collect_vec();
            eprintln!(
                "tech={technique} shader={shader} data={} magic={:?} strings={:?}",
                shader_data.len(),
                shader_data.get(..4),
                strings.into_iter().take(16).collect_vec()
            );
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_mask_material_constants() {
        init_goliath_test_package_manager();
        for technique in [
            TagHash(0x80B14088),
            TagHash(0x80B1248B),
            TagHash(0x80B124CC),
            TagHash(0x80B140CC),
        ] {
            let entry = package_manager().get_entry(technique).expect("technique");
            let data = package_manager()
                .read_tag(technique)
                .expect("technique data");
            let preview = crate::material::MaterialTagPreview::load(&entry, &data)
                .expect("technique preview");
            let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
            eprintln!("technique {technique}");
            for stage in preview.stages.iter().filter(|stage| stage.stage == "PS") {
                eprintln!("  textures={:?}", stage.textures);
                eprintln!("  constants={:?}", stage.constants);
                eprintln!("  inline={:?}", stage.inline_constants);
                eprintln!(
                    "  cbuffer={:?}",
                    stage
                        .constant_buffer_preview
                        .as_ref()
                        .map(|preview| &preview.first_values)
                );
                eprintln!("  expressions={:?}", stage.bytecode.expressions);
                eprintln!("  externs={:?}", stage.bytecode.externs);
                eprintln!("  bindings={:?}", stage.bytecode.bindings);
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_model_uv_coverage() {
        init_goliath_test_package_manager();

        let max_models = std::env::var("QUICKTAG_UV_PROBE_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(400);
        let package_filter = std::env::var("QUICKTAG_UV_PROBE_PACKAGE_FILTER").ok();
        let tags = package_manager()
            .get_all_by_reference(CLASS_GEOMETRY_RESOURCE)
            .into_iter()
            .filter(|(tag, _entry)| {
                package_filter.as_ref().is_none_or(|filter| {
                    package_manager()
                        .package_paths
                        .get(&tag.pkg_id())
                        .is_some_and(|path| path.name.contains(filter))
                })
            })
            .take(max_models)
            .collect_vec();

        let mut parsed = 0usize;
        let mut declared = 0usize;
        let mut heuristic = Vec::new();
        let mut missing = Vec::new();
        let mut no_techniques = Vec::new();
        let mut no_textures = Vec::new();
        let cache = if std::env::var("QUICKTAG_UV_PROBE_FULL_CACHE").is_ok() {
            quicktag_scanner::load_tag_cache()
        } else {
            TagCache::default()
        };

        for (tag, entry) in tags {
            let Some((source, wireframe)) = parse_model_wireframe(tag, &entry) else {
                continue;
            };
            parsed += 1;
            match wireframe.uv_format.as_deref() {
                Some(uv) if uv.contains(" layout ") => declared += 1,
                Some(uv) => heuristic.push((tag, source.input_layout_index, uv.to_string())),
                None => missing.push((tag, source.input_layout_index)),
            }

            let techniques = find_model_technique_entries(&cache, tag, &entry);
            if techniques.is_empty() {
                no_techniques.push(tag);
                continue;
            }
            if find_model_textures(&cache, tag, &techniques).is_empty() {
                no_textures.push((tag, techniques.iter().map(|(tag, _)| *tag).collect_vec()));
            }
        }

        eprintln!(
            "goliath uv coverage parsed={parsed} declared={declared} heuristic={} missing={} no_techniques={} no_textures={}",
            heuristic.len(),
            missing.len(),
            no_techniques.len(),
            no_textures.len()
        );
        eprintln!(
            "heuristic first={:?}",
            heuristic
                .iter()
                .take(16)
                .map(|(tag, layout, uv)| format!("{tag}:{layout:?}:{uv}"))
                .collect_vec()
        );
        eprintln!(
            "missing first={:?}",
            missing
                .iter()
                .take(16)
                .map(|(tag, layout)| format!("{tag}:{layout:?}"))
                .collect_vec()
        );
        eprintln!(
            "no_techniques first={:?}",
            no_techniques
                .iter()
                .take(16)
                .map(|tag| format!("{tag}"))
                .collect_vec()
        );
        eprintln!(
            "no_textures first={:?}",
            no_textures
                .iter()
                .take(16)
                .map(|(tag, techniques)| format!("{tag}:{}", techniques.len()))
                .collect_vec()
        );
        for (tag, techniques) in no_textures.iter().take(16) {
            eprintln!(
                "no_texture tag={tag} techniques={:?}",
                techniques
                    .iter()
                    .take(16)
                    .map(|technique| format!("{technique}"))
                    .collect_vec()
            );
            for technique in techniques.iter().take(4) {
                let entry = package_manager()
                    .get_entry(*technique)
                    .expect("technique entry");
                let data = package_manager()
                    .read_tag(*technique)
                    .expect("technique data");
                eprintln!(
                    "  technique {technique} texture_bindings={:?}",
                    texture_bindings_for_technique(&entry, &data)
                );
            }
            if let Some(scan) = cache.hashes.get(tag) {
                eprintln!(
                    "  parents={:?}",
                    scan.references
                        .iter()
                        .take(16)
                        .map(|parent| {
                            let class_name = package_manager()
                                .get_entry(*parent)
                                .and_then(|entry| get_class_by_id(entry.reference))
                                .map(|class| class.name.to_string())
                                .unwrap_or_else(|| "unknown".to_string());
                            format!("{parent}:{class_name}")
                        })
                        .collect_vec()
                );
                for parent in scan.references.iter().take(8) {
                    let parent_textures =
                        find_related_tags(&cache, *parent, TagSearchKind::Texture, 4);
                    let parent_techniques =
                        find_related_tags(&cache, *parent, TagSearchKind::Technique, 4);
                    eprintln!(
                        "  parent {parent}: textures={:?} techniques={:?}",
                        parent_textures
                            .iter()
                            .take(8)
                            .map(|(tag, _)| format!("{tag}"))
                            .collect_vec(),
                        parent_techniques
                            .iter()
                            .take(8)
                            .map(|(tag, _)| format!("{tag}"))
                            .collect_vec()
                    );
                }
            }
        }

        assert!(parsed > 0, "probe found no parseable geometry resources");
        assert!(
            heuristic.is_empty(),
            "some parsed models still use heuristic UVs"
        );
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_marathon_tfx_unknown_opcodes() {
        init_goliath_test_package_manager();

        let mut unknown = std::collections::BTreeMap::<
            u8,
            (usize, Vec<(TagHash, &'static str, usize, Vec<u8>)>),
        >::new();
        let mut stages = 0usize;
        let mut complete = 0usize;

        for (tag, entry) in package_manager().get_all_by_reference(0x808031D8) {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            let Some(preview) = crate::material::MaterialTagPreview::load(&entry, &data) else {
                continue;
            };
            let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
            for stage in preview.stages {
                if stage.bytecode_len == 0 {
                    continue;
                }
                stages += 1;
                let Some(op) = stage.bytecode.ops.iter().find(|op| op.name == "unknown") else {
                    complete += 1;
                    continue;
                };
                let stage_index = match stage.stage {
                    "VS" => 0,
                    "GS" => 3,
                    "PS" => 4,
                    "CS" => 5,
                    _ => continue,
                };
                let Some(bytecode) = read_array(
                    &data,
                    0x58 + stage_index * 0x88 + 0x20,
                    1,
                    package_manager().version.endian(),
                ) else {
                    continue;
                };
                let context_start = op.offset.saturating_sub(8);
                let context_end = op.offset.saturating_add(24).min(bytecode.len());
                let row = unknown.entry(op.opcode).or_default();
                row.0 += 1;
                if row.1.len() < 8 {
                    row.1.push((
                        tag,
                        stage.stage,
                        op.offset,
                        bytecode[context_start..context_end].to_vec(),
                    ));
                }
            }
        }

        eprintln!("marathon_tfx stages={stages} complete={complete}");
        let rows = unknown
            .iter()
            .map(|(opcode, (count, examples))| (*count, *opcode, examples.first()))
            .sorted_by_key(|(count, opcode, _example)| (Reverse(*count), *opcode))
            .collect_vec();
        for (count, opcode, example) in rows.into_iter().take(40) {
            eprintln!("opcode=0x{opcode:02X} count={count} first={example:02X?}");
        }
        assert!(stages > 0);
        assert_eq!(
            complete, stages,
            "some Marathon TFX stages still stop on unknown opcodes"
        );
        assert!(unknown.is_empty());
    }

    fn init_goliath_test_package_manager() {
        use std::{path::PathBuf, sync::Arc};
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        assert!(
            packages.exists(),
            "packages path missing: {}",
            packages.display()
        );

        let pm = PackageManager::new(
            packages.to_string_lossy().to_string(),
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn loads_investment_decal_atlases_for_weapon_and_runner_geometry() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());

        for (root, expected_technique, expected_atlas) in [
            (
                TagHash(0x80B7B91F),
                TagHash(0x80A9A2E0),
                TagHash(0x80A60058),
            ),
            (
                TagHash(0x80B7B9C6),
                TagHash(0x80A9A2E0),
                TagHash(0x80A60058),
            ),
            (
                TagHash(0x80A9F88F),
                TagHash(0x80A9A2E0),
                TagHash(0x80A60058),
            ),
            (
                TagHash(0x80A9E83E),
                TagHash(0x80A9E7C9),
                TagHash(0x80A9BD8A),
            ),
        ] {
            let entry = package_manager().get_entry(root).expect("model root");
            let model_tags = selected_model_geometry_tags(&cache, root, entry.reference);
            let model_entries = model_tags
                .iter()
                .filter_map(|model_tag| {
                    package_manager()
                        .get_entry(*model_tag)
                        .map(|entry| (*model_tag, entry))
                })
                .collect_vec();
            let techniques = model_entries
                .iter()
                .flat_map(|(model_tag, model_entry)| {
                    find_model_technique_entries(&cache, *model_tag, model_entry)
                })
                .unique_by(|(technique, _entry)| *technique)
                .collect_vec();
            let textures = model_tags
                .iter()
                .flat_map(|model_tag| find_model_textures(&cache, *model_tag, &techniques))
                .unique_by(|(texture, _entry)| *texture)
                .collect_vec();
            let mut matching_ranges = 0;

            for (model_tag, model_entry) in model_entries {
                let Some((_source, mut wireframe)) = parse_model_wireframe(model_tag, &model_entry)
                else {
                    continue;
                };
                assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
                for range in wireframe
                    .material_ranges
                    .iter()
                    .filter(|range| range.technique == Some(expected_technique))
                {
                    matching_ranges += 1;
                    assert_eq!(
                        range.textures.color,
                        Some(expected_atlas),
                        "{root}: investment decal atlas was not promoted"
                    );
                    assert!(
                        !range.textures.aux.contains(&expected_atlas),
                        "{root}: investment decal atlas remained technical aux data"
                    );
                }
            }

            assert!(
                matching_ranges > 0,
                "{root}: expected investment decal technique {expected_technique}"
            );
        }

        let resolve = |technique| {
            let entry = package_manager().get_entry(technique).expect("technique");
            let data = package_manager()
                .read_tag(technique)
                .expect("technique data");
            let bindings = texture_bindings_for_technique(&entry, &data);
            investment_decal_for_technique(technique, &bindings).expect("investment decal")
        };

        let InvestmentDecalResolution::Shader(selector) = resolve(TagHash(0x80B140A0)) else {
            panic!("runner selector/mask technique was not decoded")
        };
        assert_eq!(selector.mode, InvestmentDecalMode::SelectorMask);
        assert_eq!(selector.color, TagHash(0x80B14333));
        assert_eq!(selector.mask, TagHash(0x80B14331));
        assert_eq!(selector.selector_color_count, 2);

        let InvestmentDecalResolution::Shader(detail) = resolve(TagHash(0x80B1443E)) else {
            panic!("runner detail-selector technique was not decoded")
        };
        assert_eq!(detail.mode, InvestmentDecalMode::DetailSelectorMask);
        assert_eq!(detail.detail, Some(TagHash(0x80A60000)));
        assert_eq!(detail.color, TagHash(0x80B14474));
        assert_eq!(detail.mask, TagHash(0x80B14471));
        assert_eq!(detail.mask_mode, InvestmentDecalMaskMode::UvSplit);
        assert_eq!(detail.atlas_selector_max, 1);

        let InvestmentDecalResolution::Shader(multi_color) = resolve(TagHash(0x80B7A1DF)) else {
            panic!("runner multi-colour selector technique was not decoded")
        };
        assert_eq!(multi_color.selector_color_count, 5);
        assert_eq!(multi_color.mask_mode, InvestmentDecalMaskMode::Binary);
        assert_eq!(multi_color.atlas_selector_max, 0);
    }

    fn authored_geometry_dyes(
        geometry: &[TagHash],
        palette: &[GearDyeMaterial; 6],
    ) -> Vec<GearDyeMaterial> {
        geometry
            .iter()
            .flat_map(|geometry| {
                package_manager()
                    .read_tag(*geometry)
                    .into_iter()
                    .flat_map(|data| {
                        geometry_primary_index_ranges(&data, package_manager().version.endian())
                    })
            })
            .filter_map(|range| {
                palette
                    .get(range.gear_dye_change_color_index as usize)
                    .copied()
            })
            .collect()
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn does_not_inject_legacy_implicit_weapon_mods() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        for (model, expected_geometry, rejected_defaults) in [
            (
                TagHash(0x80B6E58A),
                vec![TagHash(0x80B6E588)],
                vec![TagHash(0x80A61EE2), TagHash(0x80A61D7E)],
            ),
            (
                TagHash(0x80B6DECC),
                vec![TagHash(0x80B6DEC9)],
                vec![TagHash(0x80A61E8C)],
            ),
        ] {
            let entry = package_manager().get_entry(model).expect("model Pattern");
            assert_eq!(
                selected_model_geometry_tags(&cache, model, entry.reference),
                expected_geometry
            );
            let preview = GeometryTagPreview::load_model_with_weapon_mods(
                cache.clone(),
                model,
                &entry,
                model,
                &[],
            )
            .expect("unmodified weapon preview");
            let GeometryPreviewKind::Model(preview) = preview.kind else {
                panic!("weapon Pattern must load as a model");
            };
            assert_eq!(preview.geometry_parts, expected_geometry);
            assert!(
                rejected_defaults
                    .iter()
                    .all(|default| !preview.geometry_parts.contains(default)),
                "{model} still contains implicit default mods: {:?}",
                preview.geometry_parts
            );
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn resolves_authored_weapon_defaults_and_replaces_only_occupied_family() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let socket_index = WeaponModSocketIndex::new();
        for (weapon, expected) in [
            (
                TagHash(0x80A7AD83),
                vec![TagHash(0x80A61EE7), TagHash(0x80A61D82)],
            ),
            (TagHash(0x80A7ACEC), vec![TagHash(0x80A61E8F)]),
            (
                TagHash(0x80A7AD6C),
                vec![
                    TagHash(0x80A61D31),
                    TagHash(0x80A9A431),
                    TagHash(0x80A61D13),
                ],
            ),
            (
                TagHash(0x80A7C70D),
                vec![TagHash(0x80A9A231), TagHash(0x80A9A1DA)],
            ),
            (
                TagHash(0x80A7C86E),
                vec![
                    TagHash(0x80A9A0A3),
                    TagHash(0x80A9A036),
                    TagHash(0x80A617B7),
                ],
            ),
        ] {
            let socket = socket_index
                .owner_for(&cache, weapon, &expected)
                .unwrap_or_else(|| panic!("no authored socket owner for {weapon}"));
            assert_eq!(
                weapon_default_mod_patterns(&cache, weapon, socket)
                    .into_iter()
                    .collect::<rustc_hash::FxHashSet<_>>(),
                expected
                    .iter()
                    .copied()
                    .collect::<rustc_hash::FxHashSet<_>>(),
                "wrong authored defaults for {weapon}"
            );
            assert!(expected.iter().all(|default| {
                weapon_mod_attachment_pose(&cache, socket, *default).is_some()
                    && !pattern_nearest_geometry_tags(&cache, *default).is_empty()
            }));
        }

        let bully_socket = socket_index
            .owner_for(&cache, TagHash(0x80A7AD83), &[TagHash(0x80A60874)])
            .expect("Bully socket owner");
        let bully_defaults = weapon_unoccupied_default_mod_patterns(
            &cache,
            TagHash(0x80A7AD83),
            bully_socket,
            &[TagHash(0x80A60874)],
        );
        assert!(bully_defaults.contains(&TagHash(0x80A61EE7)));
        assert!(!bully_defaults.contains(&TagHash(0x80A61D82)));

        let misriah_socket = socket_index
            .owner_for(&cache, TagHash(0x80A7ACEC), &[TagHash(0x80A601B6)])
            .expect("Misriah socket owner");
        assert!(
            weapon_unoccupied_default_mod_patterns(
                &cache,
                TagHash(0x80A7ACEC),
                misriah_socket,
                &[TagHash(0x80A601B6)],
            )
            .is_empty(),
            "equipped Misriah grip must replace authored default grip"
        );

        let attachments = bully_defaults
            .into_iter()
            .chain([TagHash(0x80A60874)])
            .map(|model_tag| WeaponModPreviewAttachment {
                model_tag,
                rarity: None,
                unique_id: 0.5,
            })
            .collect_vec();
        let equipped_geometry = pattern_nearest_geometry_tags(&cache, TagHash(0x80A60874));
        assert!(!equipped_geometry.is_empty());
        let weapon = TagHash(0x80A7AD83);
        let entry = package_manager().get_entry(weapon).expect("Bully Pattern");
        let preview = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            cache.clone(),
            weapon,
            &entry,
            weapon,
            bully_socket,
            &attachments,
        )
        .expect("Bully preview");
        let GeometryPreviewKind::Model(preview) = preview.kind else {
            panic!("Bully Pattern must load as model");
        };
        assert!(preview.geometry_parts.contains(&TagHash(0x80A61EE2)));
        assert!(!preview.geometry_parts.contains(&TagHash(0x80A61D7E)));
        assert!(
            equipped_geometry
                .iter()
                .all(|geometry| preview.geometry_parts.contains(geometry))
        );
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn resolves_current_brrt_mod_attachment_poses() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let weapon_pattern = TagHash(0x80A7B0B0);
        let modifications = [TagHash(0x80A61F24), TagHash(0x80A6071A)];
        let socket = WeaponModSocketIndex::new()
            .owner_for(&cache, weapon_pattern, &modifications)
            .expect("BRRT authored socket owner");
        let poses = modifications.map(|modification| {
            let pose = weapon_mod_attachment_pose(&cache, socket, modification)
                .unwrap_or_else(|| panic!("no pose for {modification}"));
            assert!(weapon_mod_authored_families(&cache, modification).contains(&pose.family_id));
            assert!(pose.translation.iter().all(|value| value.is_finite()));
            pose
        });
        assert_ne!(poses[0].family_id, poses[1].family_id);

        let base = TagHash(0x80AA0CA3);
        let attachment_geometry = modifications
            .iter()
            .flat_map(|modification| weapon_mod_geometry_tags(&cache, *modification))
            .collect_vec();
        let entry = package_manager()
            .get_entry(base)
            .expect("BRRT current skin pattern");
        let attachments = modifications.map(|model_tag| WeaponModPreviewAttachment {
            model_tag,
            rarity: None,
            unique_id: 0.5,
        });
        let preview = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            cache.clone(),
            base,
            &entry,
            weapon_pattern,
            socket,
            &attachments,
        )
        .expect("assembled BRRT preview");
        let GeometryPreviewKind::Model(model) = preview.kind else {
            panic!("expected model preview");
        };
        for expected in attachment_geometry {
            assert!(
                model.geometry_parts.contains(&expected),
                "assembled preview is missing {expected}: {:?}",
                model.geometry_parts
            );
        }
        let wireframe = model.wireframe.expect("assembled preview has no geometry");
        assert!(!wireframe.vertices.is_empty());
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn applies_authored_dont_let_up_mod_material_channels() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let skin = TagHash(0x80AA0CA3);
        let owner = TagHash(0x80A7D43D);
        let attachments = [TagHash(0x80A61CF0), TagHash(0x80A6071A)];
        let palette = weapon_skin_gear_dye_palette(&cache, skin).expect("Don't let up dye palette");
        assert_eq!(palette[1].color, [0.760525, 0.658375, 0.03434, 1.0]);
        assert_ne!(
            palette[0], palette[1],
            "this regression needs distinct fallback and authored channels"
        );

        // The adjacent Goliath part bytes are independently authored: +0x1c
        // selects mesh detail while +0x1d selects the change-color channel.
        // Darksight has three high-detail dyed parts; Precision Barrel has one.
        for (geometry, expected) in [
            (TagHash(0x80A61CE0), vec![0, 0, 0]),
            (TagHash(0x80A60716), vec![0]),
        ] {
            let data = package_manager()
                .read_tag(geometry)
                .unwrap_or_else(|error| panic!("{geometry}: {error}"));
            let endian = package_manager().version.endian();
            let candidates = geometry_index_range_candidates(&data, endian);
            let selected = geometry_primary_index_ranges(&data, endian);
            assert!(
                candidates.iter().any(|range| range.lod_category == 7),
                "{geometry} regression fixture must contain a lower-detail mesh"
            );
            assert!(
                selected
                    .iter()
                    .all(|range| is_highest_detail_lod(range.lod_category)),
                "{geometry} must not select a lower-detail mesh"
            );
            let channels = selected
                .into_iter()
                .filter_map(|range| {
                    (range.gear_dye_change_color_index < 6)
                        .then_some(range.gear_dye_change_color_index)
                })
                .collect_vec();
            assert_eq!(channels, expected, "{geometry} authored dye channels");
        }

        let entry = package_manager().get_entry(skin).expect("skin pattern");
        let GeometryPreviewKind::Model(model) = GeometryTagPreview::load_model_with_weapon_mods(
            cache.clone(),
            skin,
            &entry,
            owner,
            &attachments,
        )
        .expect("assembled Don't let up preview")
        .kind
        else {
            panic!("expected model preview");
        };
        for expected in [TagHash(0x80A61CE0), TagHash(0x80A60716)] {
            assert!(
                model.geometry_parts.contains(&expected),
                "assembled preview is missing {expected}: {:?}",
                model.geometry_parts
            );
        }
        let ranges = model
            .wireframe
            .expect("assembled preview has no geometry")
            .material_ranges;
        let dyed = ranges
            .iter()
            .filter_map(|range| range.textures.gear_dye)
            .collect_vec();
        assert_eq!(
            dyed,
            [vec![palette[0]; 3], vec![palette[0]]].concat(),
            "mods must consume their mesh-authored channel from the selected skin palette"
        );
        assert!(
            ranges
                .iter()
                .filter(|range| range.textures.gear_dye.is_some())
                .all(|range| matches!(
                    range.textures.gear_dye_palette,
                    Some(actual) if actual == palette
                )),
            "dyed materials need the selected skin's six-color palette for packed material IDs"
        );
        assert_eq!(
            ranges
                .iter()
                .filter(|range| range.textures.gear_dye.is_some() && range.textures.color.is_some())
                .count(),
            2,
            "both textured mod materials must retain their authored color maps"
        );
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn follows_engine_material_region_bindings_for_weapon_mods() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();

        // The compiled general-purpose gear shader samples its packed material
        // region IDs from PS t3. These two Atrax fixtures previously regressed
        // because an unrelated adjacent texture replaced that authored binding.
        for (geometry, expected_control) in [
            (TagHash(0x80A60D31), TagHash(0x80A6183A)),
            (TagHash(0x80A60C4B), TagHash(0x80A61800)),
        ] {
            let entry = package_manager()
                .get_entry(geometry)
                .unwrap_or_else(|| panic!("missing geometry {geometry}"));
            let mut wireframe = parse_model_wireframe(geometry, &entry)
                .unwrap_or_else(|| panic!("failed to parse {geometry}"))
                .1;
            let techniques = find_model_technique_entries(&cache, geometry, &entry);
            let textures = find_model_textures(&cache, geometry, &techniques);
            assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
            let dyed = wireframe
                .material_ranges
                .iter()
                .filter(|range| {
                    range.gear_dye_change_color_index.is_some() && range.textures.color.is_some()
                })
                .collect_vec();
            assert!(!dyed.is_empty(), "{geometry} has no dyed material range");
            assert!(
                dyed.iter()
                    .all(|range| range.textures.control == Some(expected_control)),
                "{geometry} did not retain its authored PS t3 material-ID map: {dyed:#?}"
            );
        }

        // The same shader's TFX writes six GearDye channels to material IDs
        // 1..6 in this non-sequential order. This is the engine-authored LUT,
        // not a display-name, skin, or weapon-family heuristic.
        let technique = TagHash(0x80A60C3B);
        let entry = package_manager()
            .get_entry(technique)
            .expect("gear technique");
        let data = package_manager()
            .read_tag(technique)
            .expect("gear technique data");
        let preview = crate::material::MaterialTagPreview::load(&entry, &data)
            .expect("gear technique preview");
        let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
        let pixel = preview
            .stages
            .iter()
            .find(|stage| stage.stage == "PS")
            .expect("pixel stage");
        let material_id_parameters = [
            GEAR_DYE_COLOR_PARAMETERS[0],
            GEAR_DYE_COLOR_PARAMETERS[2],
            GEAR_DYE_COLOR_PARAMETERS[3],
            GEAR_DYE_COLOR_PARAMETERS[1],
            GEAR_DYE_COLOR_PARAMETERS[4],
            GEAR_DYE_COLOR_PARAMETERS[5],
        ];
        for (material_id, parameter) in material_id_parameters.into_iter().enumerate() {
            let target = format!("output[{}]", material_id + 8);
            let expression = pixel
                .bytecode
                .expressions
                .iter()
                .find(|expression| expression.target == target)
                .unwrap_or_else(|| panic!("missing {target}"));
            assert!(
                expression
                    .expression
                    .contains(&format!("0x{parameter:08X}")),
                "material ID {} resolved the wrong GearDye channel: {}",
                material_id + 1,
                expression.expression
            );
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn resolves_authored_weapon_skin_dye_palettes() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let cases = [
            (
                TagHash(0x80B6CC5D),
                [
                    [0.473531, 0.027321, 0.014444],
                    [0.181164, 0.174647, 0.181164],
                    [0.473532, 0.027321, 0.014444],
                    [0.730461, 0.0185, 0.0185],
                    [0.03434, 0.033105, 0.03434],
                    [0.723055, 0.693872, 0.708376],
                ],
            ),
            (
                TagHash(0x80B6E792),
                [
                    [0.412543, 0.027321, 0.016807],
                    [0.046665, 0.046665, 0.046665],
                    [0.401978, 0.05448, 0.045186],
                    [0.283149, 0.030713, 0.021219],
                    [0.401978, 0.082283, 0.074214],
                    [0.283149, 0.030713, 0.021219],
                ],
            ),
        ];
        for component in [TagHash(0x80B6CC5C), TagHash(0x80B6E791)] {
            let data = package_manager()
                .read_tag(component)
                .expect("dye component");
            assert!(
                decode_weapon_skin_gear_dye_palette(&data).is_some(),
                "failed to decode {component} directly"
            );
        }
        let expected_roughness = [
            [-1.4, 1.0, 1.0],
            [-2.0, 1.2, 1.2],
            [-1.4, 1.3, 1.3],
            [-2.0, 1.2, 1.2],
            [1.4, 0.3, 0.3],
            [-2.0, 1.2, 1.2],
        ];
        for (pattern, expected_colors) in cases {
            let palette = weapon_skin_gear_dye_palette(&cache, pattern)
                .unwrap_or_else(|| panic!("no authored palette for {pattern}"));
            for slot in 0..6 {
                for channel in 0..3 {
                    assert!(
                        (palette[slot].color[channel] - expected_colors[slot][channel]).abs()
                            < 0.000002,
                        "{pattern} slot {slot} color: {:?}",
                        palette[slot].color
                    );
                    assert!(
                        (palette[slot].roughness_remap[channel]
                            - expected_roughness[slot][channel])
                            .abs()
                            < 0.000002,
                        "{pattern} slot {slot} roughness: {:?}",
                        palette[slot].roughness_remap
                    );
                }
                assert_eq!(palette[slot].metal_remap, [1.0, 0.0, 0.0, 1.0]);
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_atrax_pattern_object_channels() {
        init_goliath_test_package_manager();
        let wanted = [
            0x1B3D64F0, 0x1B3D64F1, 0x1B3D64F3, 0x1B3D64F4, 0x1B3D64F6, 0x1B3D64F7, 0xC8939EB8,
            0xC8939EBA, 0xC8939EBB, 0xC8939EBC, 0xC8939EBD, 0xC8939EBF, 0x3CC0E328, 0x3CC0E32A,
            0x3CC0E32B, 0x3CC0E32C, 0x3CC0E32D, 0x3CC0E32F, 0x840253AF, 0x840253A1, 0xA500D3AF,
            0x6F1709EA, 0xDECA4D7E, 0x7B2426A3, 0x040174F7, 0x51E7A18D, 0x89464871,
        ];
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            if wanted.contains(&hash) {
                eprintln!("OBJECT_CHANNEL_WORD {hash:08X}={word}");
            }
        });
        let cache = quicktag_scanner::load_tag_cache();

        let dump_component_nodes = |label: &str, component: TagHash| {
            let data = package_manager()
                .read_tag(component)
                .expect("pattern component");
            let endian = package_manager().version.endian();
            let mut pointer_offset = 0x10usize;
            let mut seen = rustc_hash::FxHashSet::default();
            for index in 0..64 {
                let Some(relative) = data
                    .get(pointer_offset..pointer_offset + 8)
                    .map(|bytes| read_i64(bytes, endian))
                else {
                    break;
                };
                if relative == 0 || relative == i64::MAX {
                    break;
                }
                let Some(node_offset) = (pointer_offset as i64)
                    .checked_add(relative)
                    .and_then(|offset| usize::try_from(offset).ok())
                else {
                    break;
                };
                if !seen.insert(node_offset) || node_offset < 4 || node_offset + 0x10 > data.len() {
                    break;
                }
                let class = read_u32_at(&data, node_offset - 4, endian).unwrap_or_default();
                eprintln!(
                    "COMPONENT_NODE label={label} component={component} index={index} node=0x{node_offset:X} data=0x{:X} class={class:08X}",
                    node_offset + 0x10
                );
                pointer_offset = node_offset;
            }
        };
        for component in [
            TagHash(0x80A7C7B5),
            TagHash(0x80A61AE0),
            TagHash(0x80A60FA2),
            TagHash(0x80B6D72D),
        ] {
            dump_component_nodes("atrax", component);
            let data = package_manager()
                .read_tag(component)
                .expect("pattern component");
            eprintln!(
                "COMPONENT_HEADER component={component} words={:?}",
                data.chunks_exact(4)
                    .take(40)
                    .map(|bytes| format!(
                        "{:08X}",
                        read_u32(bytes, package_manager().version.endian())
                    ))
                    .collect_vec()
            );
        }

        for weapon in [
            TagHash(0x80A7C7B5),
            TagHash(0x80A7D43D),
            TagHash(0x80A96FA4),
        ] {
            for node in descendant_pattern_nodes(&cache, weapon, 12) {
                let poses = weapon_attachment_poses(node);
                if poses.is_empty() {
                    continue;
                }
                let data = package_manager().read_tag(node).expect("socket component");
                eprintln!(
                    "SOCKET_TABLE weapon={weapon} node={node} poses={:?} arrays={:?} children={:?} parents={:?}",
                    poses
                        .iter()
                        .map(|pose| (pose.family_id, pose.variant_id, pose.bone_index))
                        .collect_vec(),
                    scan_arrays(&data, package_manager().version.endian())
                        .into_iter()
                        .map(|array| (array.class, array.count, array.data_offset))
                        .collect_vec(),
                    cache
                        .hashes
                        .get(&node)
                        .into_iter()
                        .flat_map(|scan| scan.file_hashes.iter().map(|child| child.hash))
                        .collect_vec(),
                    cache
                        .hashes
                        .get(&node)
                        .into_iter()
                        .flat_map(|scan| scan.references.iter().copied())
                        .collect_vec(),
                );
                for array in scan_arrays(&data, package_manager().version.endian()) {
                    if matches!(array.class, CLASS_PATTERN_CHANNEL_BINDINGS | 0x8080B1F3) {
                        eprintln!(
                            "  SOCKET_SWITCH class={:08X} count={} words={:?}",
                            array.class,
                            array.count,
                            data[array.data_offset..array.end_offset]
                                .chunks_exact(4)
                                .take(if array.class == CLASS_PATTERN_CHANNEL_BINDINGS {
                                    6
                                } else {
                                    16
                                })
                                .map(|bytes| format!(
                                    "{:08X}",
                                    read_u32(bytes, package_manager().version.endian())
                                ))
                                .collect_vec()
                        );
                        if array.class == 0x8080B1F3 {
                            for (index, record) in
                                array_records(&data, array, 0x10).into_iter().enumerate()
                            {
                                let record_offset = array.data_offset + index * 0x10;
                                let relative =
                                    read_i64(&record[0..8], package_manager().version.endian());
                                if let Some(target) = (record_offset as i64)
                                    .checked_add(relative)
                                    .and_then(|target| usize::try_from(target).ok())
                                {
                                    eprintln!(
                                        "  SOCKET_RESOURCE weapon={weapon} node={node} index={index} class={:08X} target=0x{target:X} words={:?}",
                                        read_u32_at(record, 8, package_manager().version.endian())
                                            .unwrap_or_default(),
                                        data[target..(target + 0x100).min(data.len())]
                                            .chunks_exact(4)
                                            .map(|bytes| format!(
                                                "{:08X}",
                                                read_u32(bytes, package_manager().version.endian())
                                            ))
                                            .collect_vec()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        for (geometry, socket_variant) in [
            (TagHash(0x80A60D31), 0x840253AF_u32),
            (TagHash(0x80A60C4B), 0x840253A1_u32),
        ] {
            let geometry_data = package_manager().read_tag(geometry).expect("mod geometry");
            eprintln!("MOD_VARIANT geometry={geometry} socket_variant={socket_variant:08X}");
            for part in
                geometry_index_range_candidates(&geometry_data, package_manager().version.endian())
            {
                eprintln!(
                    "  part={} base={} variant_shader={} dye={} lod={}",
                    part.part_index,
                    part.technique,
                    part.variant_shader_index,
                    part.gear_dye_change_color_index,
                    part.lod_category
                );
            }
            for technique in
                geometry_index_range_candidates(&geometry_data, package_manager().version.endian())
                    .into_iter()
                    .map(|part| part.technique)
                    .unique()
            {
                let entry = package_manager()
                    .get_entry(technique)
                    .expect("mod technique");
                let data = package_manager()
                    .read_tag(technique)
                    .expect("mod technique data");
                let preview = crate::material::MaterialTagPreview::load(&entry, &data)
                    .expect("mod technique preview");
                let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
                eprintln!(
                    "  technique={technique} default7={:?} constants7={:?} inline7={:?} bindings={:?} expressions={:?}",
                    preview
                        .stages
                        .iter()
                        .find(|stage| stage.stage == "PS")
                        .and_then(|stage| stage.constant_buffer_preview.as_ref())
                        .and_then(|buffer| buffer.first_values.get(7)),
                    preview
                        .stages
                        .iter()
                        .find(|stage| stage.stage == "PS")
                        .and_then(|stage| stage.constants.get(7)),
                    preview
                        .stages
                        .iter()
                        .find(|stage| stage.stage == "PS")
                        .and_then(|stage| stage.inline_constants.get(7)),
                    preview
                        .stages
                        .iter()
                        .filter(|stage| stage.stage == "PS")
                        .flat_map(|stage| stage.bytecode.bindings.iter())
                        .filter(|binding| binding.source.contains("object_channel"))
                        .map(|binding| (&binding.kind, &binding.source, binding.slot))
                        .collect_vec(),
                    preview
                        .stages
                        .iter()
                        .filter(|stage| stage.stage == "PS")
                        .flat_map(|stage| stage.bytecode.expressions.iter())
                        .filter(|expression| {
                            expression.expression.contains("object_channel")
                                || expression.target.starts_with("output[")
                        })
                        .map(|expression| (&expression.target, &expression.expression))
                        .collect_vec()
                );
            }
            for parent in cache
                .hashes
                .get(&geometry)
                .into_iter()
                .flat_map(|scan| scan.references.iter().copied())
            {
                let Some(parent_entry) = package_manager().get_entry(parent) else {
                    continue;
                };
                eprintln!("  parent={parent} class={:08X}", parent_entry.reference);
                if !matches!(
                    parent_entry.reference,
                    CLASS_ENTITY_RESOURCE | CLASS_PATTERN_COMPONENT
                ) {
                    continue;
                }
                let data = package_manager().read_tag(parent).expect("model component");
                let endian = package_manager().version.endian();
                let resource_offset: usize = data
                    .get(0x18..0x20)
                    .map(|bytes| read_i64(bytes, endian))
                    .and_then(|relative| (0x18_i64 + relative).try_into().ok())
                    .unwrap_or_default();
                eprintln!(
                    "    resource=0x{resource_offset:X} model224={:?} model264={:?} socket_variant_offsets={:?}",
                    read_tag_at(&data, resource_offset + 0x224, endian),
                    read_tag_at(&data, resource_offset + 0x264, endian),
                    data.windows(4)
                        .enumerate()
                        .filter_map(|(offset, bytes)| {
                            (read_u32(bytes, endian) == socket_variant).then_some(offset)
                        })
                        .collect_vec()
                );
                eprintln!(
                    "    arrays={:?}",
                    scan_arrays(&data, endian)
                        .into_iter()
                        .map(|array| (array.class, array.count, array.data_offset))
                        .collect_vec()
                );
                for array in scan_arrays(&data, endian) {
                    if matches!(array.class, CLASS_PATTERN_CHANNEL_BINDINGS | 0x8080BAD0) {
                        let stride = if array.class == CLASS_PATTERN_CHANNEL_BINDINGS {
                            0x18
                        } else {
                            0x08
                        };
                        eprintln!(
                            "    material switches class={:08X} records={:?}",
                            array.class,
                            array_records(&data, array, stride)
                                .into_iter()
                                .map(|record| record
                                    .chunks_exact(4)
                                    .map(|bytes| format!("{:08X}", read_u32(bytes, endian)))
                                    .collect_vec())
                                .collect_vec()
                        );
                    }
                }
                for map_offset in [0x3c0, 0x400] {
                    let records12 = read_array(&data, resource_offset + map_offset, 12, endian)
                        .map(|records| {
                            records
                                .chunks_exact(12)
                                .map(|record| {
                                    (
                                        read_u32_at(record, 0, endian),
                                        read_u32_at(record, 4, endian),
                                        read_u32_at(record, 8, endian),
                                    )
                                })
                                .collect_vec()
                        });
                    let tags = read_tag_array(&data, resource_offset + map_offset, endian);
                    eprintln!("    vec+0x{map_offset:X}: records12={records12:?} tags={tags:?}");
                }
                let techniques = read_tag_array(&data, resource_offset + 0x400, endian);
                for technique in techniques {
                    let Some(entry) = package_manager().get_entry(technique) else {
                        continue;
                    };
                    let Ok(technique_data) = package_manager().read_tag(technique) else {
                        continue;
                    };
                    eprintln!(
                        "    variant technique={technique} textures={:?} constants={:?}",
                        texture_bindings_for_technique(&entry, &technique_data)
                            .into_iter()
                            .filter(|binding| binding.stage == "PS")
                            .map(|binding| (binding.slot, binding.tag))
                            .collect_vec(),
                        crate::material::material_constants_for_technique(&entry, &technique_data)
                    );
                }
            }
        }
        for mod_root in [TagHash(0x80A60D38), TagHash(0x80A60C4F)] {
            for node in descendant_pattern_nodes(&cache, mod_root, 8) {
                let data = package_manager().read_tag(node).expect("mod component");
                let endian = package_manager().version.endian();
                for offset in (0..data.len().saturating_sub(4)).step_by(4) {
                    if read_u32_at(&data, offset, endian) != Some(CLASS_WEAPON_MOD_VISUAL_BINDING) {
                        continue;
                    }
                    eprintln!(
                        "MOD_VISUAL root={mod_root} node={node} offset=0x{offset:X} words={:?}",
                        (0..24)
                            .filter_map(|index| read_u32_at(&data, offset + index * 4, endian))
                            .map(|value| format!("{value:08X}"))
                            .collect_vec()
                    );
                }
                for array in scan_arrays(&data, endian)
                    .into_iter()
                    .filter(|array| array.class == CLASS_PATTERN_CHANNEL_BINDINGS)
                {
                    eprintln!(
                        "MOD_CHANNELS root={mod_root} node={node} count={} words={:?}",
                        array.count,
                        data[array.data_offset..array.end_offset]
                            .chunks_exact(4)
                            .take(64)
                            .map(|bytes| format!("{:08X}", read_u32(bytes, endian)))
                            .collect_vec()
                    );
                }
            }
        }
        for root in [
            TagHash(0x80B6D750),
            TagHash(0x80A7C7B5),
            TagHash(0x80A60D38),
            TagHash(0x80A60C4F),
        ] {
            for node in descendant_pattern_nodes(&cache, root, 8) {
                let Some(entry) = package_manager().get_entry(node) else {
                    continue;
                };
                let Ok(data) = package_manager().read_tag(node) else {
                    continue;
                };
                for parameter in [
                    0xC8939EBF_u32,
                    0xC8939EBA,
                    0xC8939EBC,
                    0xC8939EBD,
                    0xC8939EBB,
                    0xC8939EB8,
                ] {
                    for offset in (0..data.len().saturating_sub(3))
                        .step_by(4)
                        .filter(|offset| {
                            read_u32_at(&data, *offset, package_manager().version.endian())
                                == Some(parameter)
                        })
                    {
                        let arrays = scan_arrays(&data, package_manager().version.endian());
                        eprintln!(
                            "VARIANT_RAW root={root} node={node} parameter={parameter:08X} offset=0x{offset:X} arrays={:?}",
                            arrays
                                .into_iter()
                                .filter(|array| offset >= array.data_offset
                                    && offset < array.end_offset)
                                .map(|array| (array.class, array.count, array.data_offset))
                                .collect_vec()
                        );
                    }
                }
                let arrays = scan_arrays(&data, package_manager().version.endian());
                for array in arrays
                    .iter()
                    .copied()
                    .filter(|array| array.class == 0x8080AF86)
                {
                    for (index, record) in array_records(&data, array, 0x70).into_iter().enumerate()
                    {
                        let Some(parameter) =
                            read_u32_at(record, 0, package_manager().version.endian())
                        else {
                            continue;
                        };
                        if ![
                            0xC8939EBF_u32,
                            0xC8939EBA,
                            0xC8939EBC,
                            0xC8939EBD,
                            0xC8939EBB,
                            0xC8939EB8,
                        ]
                        .contains(&parameter)
                        {
                            continue;
                        }
                        let record_offset = array.data_offset + index * 0x70;
                        let bytecode = read_array(
                            &data,
                            record_offset + 0x08,
                            1,
                            package_manager().version.endian(),
                        )
                        .unwrap_or_default();
                        let constants = read_array(
                            &data,
                            record_offset + 0x18,
                            0x10,
                            package_manager().version.endian(),
                        )
                        .unwrap_or_default();
                        eprintln!(
                            "VARIANT_EXPRESSION root={root} node={node} parameter={parameter:08X} bytecode={} constants={:?} words={:?}",
                            bytecode.iter().map(|byte| format!("{byte:02X}")).join(" "),
                            constants
                                .chunks_exact(0x10)
                                .map(|constant| read_vec4_f32(
                                    constant,
                                    0,
                                    package_manager().version.endian()
                                )
                                .unwrap())
                                .collect_vec(),
                            record
                                .chunks_exact(4)
                                .map(|bytes| format!(
                                    "{:08X}",
                                    read_u32(bytes, package_manager().version.endian())
                                ))
                                .collect_vec()
                        );
                    }
                }
                for array in arrays
                    .iter()
                    .copied()
                    .filter(|array| array.class == 0x8080AF14)
                {
                    for (index, record) in array_records(&data, array, 0x28).into_iter().enumerate()
                    {
                        let Some(parameter) =
                            read_u32_at(record, 0x20, package_manager().version.endian())
                        else {
                            continue;
                        };
                        if [
                            0xC8939EBF_u32,
                            0xC8939EBA,
                            0xC8939EBC,
                            0xC8939EBD,
                            0xC8939EBB,
                            0xC8939EB8,
                        ]
                        .contains(&parameter)
                        {
                            let provider = arrays
                                .iter()
                                .copied()
                                .find(|candidate| candidate.class == 0x8080AF13)
                                .and_then(|provider_array| {
                                    array_records(&data, provider_array, 0x30)
                                        .get(index)
                                        .copied()
                                })
                                .map(|provider| {
                                    provider
                                        .chunks_exact(4)
                                        .map(|bytes| {
                                            format!(
                                                "{:08X}",
                                                read_u32(bytes, package_manager().version.endian())
                                            )
                                        })
                                        .collect_vec()
                                });
                            eprintln!(
                                "VARIANT_SELECTOR root={root} node={node} index={index} parameter={parameter:08X} words={:?} provider={provider:?}",
                                record
                                    .chunks_exact(4)
                                    .map(|bytes| format!(
                                        "{:08X}",
                                        read_u32(bytes, package_manager().version.endian())
                                    ))
                                    .collect_vec()
                            );
                        }
                    }
                }
                let singleton_vectors = arrays
                    .iter()
                    .copied()
                    .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
                    .collect_vec();
                let vectors = singleton_vectors
                    .iter()
                    .copied()
                    .map(|array| {
                        read_serialized_dye_vector(&data, array, package_manager().version.endian())
                    })
                    .collect::<Option<Vec<_>>>();
                let Some(vectors) = vectors else {
                    continue;
                };
                for array in arrays
                    .iter()
                    .copied()
                    .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
                {
                    for record in array_records(&data, array, 0x0c) {
                        let Some(scope) =
                            read_u32_at(record, 0x00, package_manager().version.endian())
                        else {
                            continue;
                        };
                        let Some(parameter) =
                            read_u32_at(record, 0x04, package_manager().version.endian())
                        else {
                            continue;
                        };
                        let Some(index) =
                            read_u32_at(record, 0x08, package_manager().version.endian())
                                .map(|value| value as usize)
                        else {
                            continue;
                        };
                        if let Some(value) = vectors.get(index) {
                            eprintln!(
                                "OBJECT_CHANNEL root={root} node={node} class={:08X} scope={scope:08X} parameter={parameter:08X} index={index} value={value:?}",
                                entry.reference
                            );
                        }
                    }
                }
            }
        }

        let variant_parameters = [
            0xC8939EBF, 0xC8939EBA, 0xC8939EBC, 0xC8939EBD, 0xC8939EBB, 0xC8939EB8,
        ];
        for (component, _entry) in package_manager().get_all_by_reference(CLASS_PATTERN_COMPONENT) {
            let Ok(data) = package_manager().read_tag(component) else {
                continue;
            };
            let arrays = scan_arrays(&data, package_manager().version.endian());
            let vectors = arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
                .map(|array| {
                    read_serialized_dye_vector(&data, array, package_manager().version.endian())
                })
                .collect::<Option<Vec<_>>>();
            let Some(vectors) = vectors else {
                continue;
            };
            let values = arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
                .flat_map(|array| array_records(&data, array, 0x0c))
                .filter_map(|record| {
                    let parameter = read_u32_at(record, 0x04, package_manager().version.endian())?;
                    variant_parameters.contains(&parameter).then_some(())?;
                    let index =
                        read_u32_at(record, 0x08, package_manager().version.endian())? as usize;
                    Some((parameter, *vectors.get(index)?))
                })
                .collect_vec();
            if values.is_empty() {
                continue;
            }
            eprintln!(
                "VARIANT_CHANNEL component={component} values={values:?} parents={:?}",
                cache
                    .hashes
                    .get(&component)
                    .into_iter()
                    .flat_map(|scan| scan.references.iter().copied())
                    .map(|parent| {
                        (
                            parent,
                            package_manager()
                                .get_entry(parent)
                                .map(|entry| entry.reference),
                        )
                    })
                    .collect_vec()
            );
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages and shader decompiler"]
    #[cfg(feature = "decompile-shaders")]
    fn probes_goliath_gear_dye_pixel_shader() {
        init_goliath_test_package_manager();
        let shader = TagHash(0x80A6007C);
        let entry = package_manager().get_entry(shader).expect("shader header");
        let data = package_manager()
            .read_tag(TagHash(entry.reference))
            .expect("shader bytecode");
        let decompiled = hlsldecompiler::decompile(&data).expect("decompile pixel shader");
        std::fs::create_dir_all("target/quicktag-model-probe").unwrap();
        std::fs::write(
            "target/quicktag-model-probe/gear-dye-80A6007C.hlsl",
            &decompiled,
        )
        .unwrap();
        eprintln!("GEAR_DYE_SHADER_BEGIN\n{decompiled}\nGEAR_DYE_SHADER_END");
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn extracts_goliath_gear_dye_pixel_shader() {
        init_goliath_test_package_manager();
        std::fs::create_dir_all("target/quicktag-model-probe").unwrap();
        for (shader, stage) in [(TagHash(0x80A6007C), "ps"), (TagHash(0x80A60079), "vs")] {
            let entry = package_manager().get_entry(shader).expect("shader header");
            let data = package_manager()
                .read_tag(TagHash(entry.reference))
                .expect("shader bytecode");
            std::fs::write(
                format!("target/quicktag-model-probe/gear-dye-{shader}-{stage}.bin"),
                data,
            )
            .unwrap();
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn decodes_every_authored_gear_dye_component() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let endian = package_manager().version.endian();
        let mut same_shape = 0;
        let mut root_checks = 0;
        let mut candidates = vec![];
        let mut failures = vec![];
        for (tag, _entry) in package_manager().get_all_by_reference(CLASS_PATTERN_COMPONENT) {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            let arrays = scan_arrays(&data, package_manager().version.endian());
            let vector_count = arrays
                .iter()
                .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
                .count();
            if vector_count == 29 {
                same_shape += 1;
                let binding_parameters = arrays
                    .iter()
                    .copied()
                    .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
                    .flat_map(|array| array_records(&data, array, 0x0c))
                    .filter_map(|record| read_u32_at(record, 0x04, endian))
                    .collect::<rustc_hash::FxHashSet<_>>();
                if GEAR_DYE_COLOR_PARAMETERS
                    .iter()
                    .any(|parameter| binding_parameters.contains(parameter))
                {
                    candidates.push(tag);
                    let Some(expected) = decode_weapon_skin_gear_dye_palette(&data) else {
                        failures.push(tag);
                        continue;
                    };
                    for (depth, root) in ancestor_pattern_roots(&cache, tag) {
                        if depth != 1 {
                            continue;
                        }
                        root_checks += 1;
                        assert_eq!(
                            weapon_skin_gear_dye_palette(&cache, root),
                            Some(expected),
                            "selected pattern {root} resolved the wrong dye component instead of {tag}"
                        );
                    }
                }
            }
        }
        assert_eq!(same_shape, 223, "unexpected 29-vector component count");
        assert_eq!(candidates.len(), 214, "unexpected authored gear-dye count");
        assert_eq!(
            root_checks, 247,
            "unexpected direct skin/default pattern coverage"
        );
        assert!(
            failures.is_empty(),
            "binding-driven dye decode failed for {failures:?}"
        );

        for (tag, expected) in [
            (TagHash(0x80B6C2A0), [0.042311, 0.042311, 0.042311]),
            (TagHash(0x80B6CA95), [0.152284, 0.496933, 0.015574]),
        ] {
            let data = package_manager().read_tag(tag).expect("dye component");
            let palette = decode_weapon_skin_gear_dye_palette(&data).expect("authored palette");
            for (actual, expected) in palette[0].color.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 0.000002, "{tag}: {palette:?}");
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn applies_authored_basic_weapon_dye_to_mods() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let base = TagHash(0x80AA0E95);
        let palette = weapon_skin_gear_dye_palette(&cache, base)
            .expect("basic weapon must resolve its authored default palette");
        assert_eq!(palette[0].color, [0.042247, 0.042247, 0.042247, 1.0]);
        let entry = package_manager()
            .get_entry(base)
            .expect("basic weapon pattern");
        let preview = GeometryTagPreview::load_model_with_weapon_mods(
            cache,
            base,
            &entry,
            TagHash(0x80A96FA4),
            &[
                TagHash(0x80A60313),
                TagHash(0x80A6068A),
                TagHash(0x80A600AC),
            ],
        )
        .expect("basic weapon with three attached mods");
        let GeometryPreviewKind::Model(model) = preview.kind else {
            panic!("basic weapon preview must be a model");
        };
        for expected in [
            TagHash(0x80A6030F),
            TagHash(0x80A60687),
            TagHash(0x80A60099),
        ] {
            assert!(
                model.geometry_parts.contains(&expected),
                "missing {expected}"
            );
        }
        let ranges = model
            .wireframe
            .expect("basic weapon wireframe")
            .material_ranges;
        assert!(ranges.iter().any(|range| range.textures.color.is_some()));
        let applied = ranges
            .iter()
            .filter_map(|range| range.textures.gear_dye)
            .collect_vec();
        let attachment_geometry = [
            TagHash(0x80A6030F),
            TagHash(0x80A60687),
            TagHash(0x80A60099),
        ];
        assert_eq!(
            applied,
            authored_geometry_dyes(&attachment_geometry, &palette),
            "basic skin attachments must consume their mesh-authored palette channels"
        );
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn resolves_current_misriah_mod_attachment_poses() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let weapon_pattern = TagHash(0x80A7ACEC);
        let modifications = [
            TagHash(0x80A6032C),
            TagHash(0x80A60B69),
            TagHash(0x80A60496),
        ];
        let socket = WeaponModSocketIndex::new()
            .owner_for(&cache, weapon_pattern, &modifications)
            .expect("Misriah authored socket owner");
        let poses = modifications.map(|modification| {
            let pose = weapon_mod_attachment_pose(&cache, socket, modification)
                .unwrap_or_else(|| panic!("no pose for {modification}"));
            assert!(weapon_mod_authored_families(&cache, modification).contains(&pose.family_id));
            assert!(pose.translation.iter().all(|value| value.is_finite()));
            pose
        });
        assert_eq!(
            poses
                .iter()
                .map(|pose| pose.family_id)
                .collect::<rustc_hash::FxHashSet<_>>()
                .len(),
            modifications.len()
        );

        let base = TagHash(0x80B7D13F);
        let attachment_geometry = modifications
            .iter()
            .flat_map(|modification| weapon_mod_geometry_tags(&cache, *modification))
            .collect_vec();
        let entry = package_manager()
            .get_entry(base)
            .expect("Misriah base pattern");
        let attachments = modifications.map(|model_tag| WeaponModPreviewAttachment {
            model_tag,
            rarity: None,
            unique_id: 0.5,
        });
        let preview = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            cache.clone(),
            base,
            &entry,
            weapon_pattern,
            socket,
            &attachments,
        )
        .expect("assembled Misriah preview");
        let GeometryPreviewKind::Model(model) = preview.kind else {
            panic!("expected model preview");
        };
        for expected in attachment_geometry {
            assert!(
                model.geometry_parts.contains(&expected),
                "assembled preview is missing {expected}: {:?}",
                model.geometry_parts
            );
        }
        let wireframe = model.wireframe.expect("assembled preview has no geometry");
        assert!(!wireframe.vertices.is_empty());
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_reference_export_attachment_transforms() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let targets = [
            ("darksight", 0.2069903_f32, 0.1242157_f32),
            ("flechette", 0.177225_f32, 0.13954_f32),
        ];

        for tag in cache.hashes.keys().copied().sorted() {
            if !matches!(tag.0, 0x80A7D43D | 0x80A970D2) {
                continue;
            }
            let Some(entry) = package_manager().get_entry(tag) else {
                continue;
            };
            if entry.file_size > 2 * 1024 * 1024 {
                continue;
            }
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            for (name, x, z) in targets {
                for offset in (0..data.len().saturating_sub(3)).step_by(4) {
                    let value = read_f32(
                        &data[offset..offset + 4],
                        package_manager().version.endian(),
                    );
                    if (value - x).abs() > 0.000001 {
                        continue;
                    }
                    let nearby_z = (offset.saturating_sub(0x40)
                        ..(offset + 0x80).min(data.len().saturating_sub(3)))
                        .step_by(4)
                        .find(|candidate| {
                            let value = read_f32(
                                &data[*candidate..*candidate + 4],
                                package_manager().version.endian(),
                            );
                            (value - z).abs() < 0.000001
                        });
                    let Some(z_offset) = nearby_z else {
                        continue;
                    };
                    let class = get_class_by_id(entry.reference)
                        .map(|class| class.name.into_owned())
                        .unwrap_or_else(|| format!("{:08X}", entry.reference));
                    let parents = cache
                        .hashes
                        .get(&tag)
                        .into_iter()
                        .flat_map(|scan| scan.references.iter().copied())
                        .filter_map(|parent| {
                            let entry = package_manager().get_entry(parent)?;
                            Some((parent, entry.reference))
                        })
                        .sorted()
                        .collect_vec();
                    eprintln!(
                        "ATTACH_TRANSFORM {name} tag={tag} class={:08X}:{class} size={} x@{offset:X} z@{z_offset:X} parents={parents:08X?}",
                        entry.reference, entry.file_size,
                    );
                }
            }
        }

        let mut words = rustc_hash::FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            words.entry(hash).or_insert_with(|| word.to_owned());
        });
        if let Ok(extra_words) =
            std::fs::read_to_string("alkahest/crates/alkahest/wordlist_channels.txt")
        {
            for word in extra_words.lines() {
                let hash = quicktag_core::util::fnv1(word.as_bytes());
                if [
                    0x138DE801_u32,
                    0xA9D208CC,
                    0xA590EEC6,
                    0xD4FB5E33,
                    0xEAA8E3CF,
                ]
                .contains(&hash)
                {
                    eprintln!("QUICKDRAW_CHANNEL_WORD {hash:08X}={word}");
                }
            }
        }
        for tag in [TagHash(0x80A7D43D), TagHash(0x80A970D2)] {
            let data = package_manager()
                .read_tag(tag)
                .expect("attachment component");
            let scan = &cache.hashes[&tag];
            eprintln!(
                "ATTACH_COMPONENT tag={tag} arrays={:?}",
                scan_arrays(&data, package_manager().version.endian())
                    .into_iter()
                    .map(|array| (
                        array.class,
                        array.count,
                        array.data_offset,
                        array.end_offset
                    ))
                    .collect_vec()
            );
            eprintln!(
                "ATTACH_COMPONENT_REFS tag={tag} refs={:?} refs64={:?} words={:?}",
                scan.file_hashes
                    .iter()
                    .map(|reference| (reference.offset, reference.hash))
                    .collect_vec(),
                scan.file_hashes64
                    .iter()
                    .map(|reference| (
                        reference.offset,
                        reference.hash,
                        tag64_to_hash32(reference.hash)
                    ))
                    .collect_vec(),
                scan.wordlist_hashes
                    .iter()
                    .filter_map(|word| Some((word.offset, word.hash, words.get(&word.hash)?)))
                    .collect_vec(),
            );
            for (index, record) in data[0x300..0x93c].chunks_exact(0x30).enumerate() {
                let wide = u64::from_le_bytes(record[0x28..0x30].try_into().unwrap());
                let visual = tag64_to_hash32(tiger_pkg::TagHash64(wide));
                let rotation = std::array::from_fn::<_, 4, _>(|axis| {
                    f32::from_bits(u32::from_le_bytes(
                        record[axis * 4..0x04 + axis * 4].try_into().unwrap(),
                    ))
                });
                let translation = std::array::from_fn::<_, 4, _>(|axis| {
                    f32::from_bits(u32::from_le_bytes(
                        record[0x10 + axis * 4..0x14 + axis * 4].try_into().unwrap(),
                    ))
                });
                eprintln!(
                    "ATTACH_RECORD tag={tag} index={index} wide={wide:016X} visual={visual:?} rotation={rotation:?} translation={translation:?}"
                );
            }
            for (start, end) in [(0x560, 0x610), (0x7A0, 0x850)] {
                eprintln!(
                    "ATTACH_COMPONENT_WORDS tag={tag} range={start:X}..{end:X} {:?}",
                    data[start..end.min(data.len())]
                        .chunks_exact(4)
                        .enumerate()
                        .map(|(index, bytes)| {
                            let raw = u32::from_le_bytes(bytes.try_into().unwrap());
                            format!(
                                "+{:03X}={raw:08X}/{}",
                                start + index * 4,
                                f32::from_bits(raw)
                            )
                        })
                        .collect_vec()
                );
            }
        }

        for (name, identifier, expected_visual) in [
            ("darksight", 0x6F17_09EA_u32, TagHash(0x80A61398)),
            ("flechette", 0xA500_D3AF_u32, TagHash(0x80A61CF0)),
        ] {
            for tag in cache.hashes.keys().copied().sorted() {
                let Some(entry) = package_manager().get_entry(tag) else {
                    continue;
                };
                if entry.file_size > 128 * 1024
                    || !matches!(
                        entry.reference,
                        CLASS_PATTERN | CLASS_PATTERN_COMPONENT | 0x8080BA53 | 0x8080BEF0
                    )
                {
                    continue;
                }
                let Ok(data) = package_manager().read_tag(tag) else {
                    continue;
                };
                for (offset, _bytes) in data.chunks_exact(4).enumerate().filter(|(_, bytes)| {
                    u32::from_le_bytes((*bytes).try_into().unwrap()) == identifier
                }) {
                    eprintln!(
                        "ATTACH_IDENTIFIER name={name} identifier={identifier:08X} expected={expected_visual} tag={tag} class={:08X} offset={:X}",
                        entry.reference,
                        offset * 4,
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_mod_anchor_tags() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let anchors = [
            TagHash(0x80AA0E95),
            TagHash(0x80AA0ECF),
            TagHash(0x80A60B32),
            TagHash(0x80A60B36),
            TagHash(0x80A9A0C1),
            TagHash(0x80A9A0C3),
            TagHash(0x80A61398),
            TagHash(0x80A61CF0),
            TagHash(0x80A7EF31),
            TagHash(0x80B6E792),
        ];

        for anchor in anchors {
            let entry = package_manager().get_entry(anchor).expect("anchor entry");
            let package = package_manager()
                .package_paths
                .get(&anchor.pkg_id())
                .map(|path| path.name.clone())
                .unwrap_or_else(|| "?".to_owned());
            let class = get_class_by_id(entry.reference)
                .map(|class| class.name.into_owned())
                .unwrap_or_else(|| "unknown".to_owned());
            let transform = (entry.reference == CLASS_GEOMETRY_RESOURCE)
                .then(|| package_manager().read_tag(anchor).ok())
                .flatten()
                .and_then(|data| {
                    read_geometry_position_transform(&data, package_manager().version.endian())
                });
            eprintln!(
                "ANCHOR tag={anchor} class={:08X}:{class} size={} package={package} transform={transform:?}",
                entry.reference, entry.file_size
            );

            let Some(scan) = cache.hashes.get(&anchor) else {
                eprintln!("  no scan");
                continue;
            };
            let parents = scan
                .references
                .iter()
                .copied()
                .filter_map(|tag| {
                    package_manager().get_entry(tag).map(|entry| {
                        let name = get_class_by_id(entry.reference)
                            .map(|class| class.name.into_owned())
                            .unwrap_or_else(|| "unknown".to_owned());
                        (tag, entry.reference, name)
                    })
                })
                .sorted()
                .collect_vec();
            let children = scan
                .file_hashes
                .iter()
                .filter_map(|reference| {
                    package_manager().get_entry(reference.hash).map(|entry| {
                        let name = get_class_by_id(entry.reference)
                            .map(|class| class.name.into_owned())
                            .unwrap_or_else(|| "unknown".to_owned());
                        (reference.hash, entry.reference, name)
                    })
                })
                .unique()
                .sorted()
                .collect_vec();
            eprintln!("  parents={parents:?}");
            let pattern_children = children
                .iter()
                .filter(|(_tag, class, _name)| *class == CLASS_PATTERN_COMPONENT)
                .map(|(tag, _class, _name)| tag.to_string())
                .collect_vec();
            eprintln!("  pattern_children={pattern_children:?}");
            let geometry = pattern_geometry_tags(&cache, anchor);
            eprintln!(
                "  geometry={} {:?}",
                geometry.len(),
                geometry
                    .iter()
                    .map(|tag| {
                        let transform = package_manager().read_tag(*tag).ok().and_then(|data| {
                            read_geometry_position_transform(
                                &data,
                                package_manager().version.endian(),
                            )
                        });
                        format!("{tag}:{transform:?}")
                    })
                    .collect_vec()
            );
            if matches!(
                anchor.0,
                0x80AA0E95
                    | 0x80AA0ECF
                    | 0x80A60B32
                    | 0x80A60B36
                    | 0x80A9A0C1
                    | 0x80A9A0C3
                    | 0x80A61398
                    | 0x80A61CF0
                    | 0x80A7EF31
                    | 0x80B6E792
            ) {
                let nearest = pattern_nearest_geometry_tags(&cache, anchor);
                eprintln!(
                    "  nearest_geometry={:?}",
                    nearest.iter().map(|tag| tag.to_string()).collect_vec()
                );
                for tag in nearest {
                    let geometry_entry = package_manager().get_entry(tag).expect("geometry");
                    let model = load_model_preview_from_tags(
                        cache.clone(),
                        tag,
                        &geometry_entry,
                        "Geometry",
                        vec![tag],
                        &[],
                    );
                    let materials = model
                        .wireframe
                        .into_iter()
                        .flat_map(|wireframe| wireframe.material_ranges)
                        .map(|range| {
                            (
                                range.technique.map(|tag| tag.to_string()),
                                range.textures.color.map(|tag| tag.to_string()),
                                range.textures.control.map(|tag| tag.to_string()),
                                range.textures.color_tint,
                            )
                        })
                        .unique()
                        .collect_vec();
                    eprintln!("  material geometry={tag} {materials:?}");
                }
            }

            let mut frontier = vec![(anchor, 0usize)];
            let mut seen = rustc_hash::FxHashSet::default();
            seen.insert(anchor);
            while let Some((child, depth)) = frontier.pop() {
                if depth == 2 {
                    continue;
                }
                for parent in cache
                    .hashes
                    .get(&child)
                    .into_iter()
                    .flat_map(|scan| scan.references.iter().copied())
                    .sorted()
                {
                    if !seen.insert(parent) {
                        continue;
                    }
                    let Some(parent_entry) = package_manager().get_entry(parent) else {
                        continue;
                    };
                    if matches!(
                        parent_entry.reference,
                        CLASS_PATTERN | CLASS_PATTERN_COMPONENT | CLASS_GEOMETRY_RESOURCE
                    ) {
                        let name = get_class_by_id(parent_entry.reference)
                            .map(|class| class.name.into_owned())
                            .unwrap_or_else(|| "unknown".to_owned());
                        eprintln!(
                            "  up depth={} child={child} parent={parent} class={:08X}:{name}",
                            depth + 1,
                            parent_entry.reference
                        );
                    }
                    frontier.push((parent, depth + 1));
                }
            }
        }

        let base = cache.hashes.get(&anchors[0]).expect("base scan");
        let skin = cache.hashes.get(&anchors[1]).expect("skin scan");
        let base_refs = base
            .file_hashes
            .iter()
            .map(|reference| (reference.offset, reference.hash))
            .collect::<std::collections::BTreeMap<_, _>>();
        let skin_refs = skin
            .file_hashes
            .iter()
            .map(|reference| (reference.offset, reference.hash))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut transitions = rustc_hash::FxHashSet::default();
        for offset in base_refs.keys().chain(skin_refs.keys()).unique().sorted() {
            let before = base_refs.get(offset);
            let after = skin_refs.get(offset);
            if before != after {
                transitions.insert((before.copied(), after.copied()));
            }
        }
        eprintln!(
            "SKIN_REF_TRANSITIONS {:?}",
            transitions
                .into_iter()
                .map(|(base, skin)| (
                    base.map(|tag| tag.to_string()),
                    skin.map(|tag| tag.to_string())
                ))
                .sorted()
                .collect_vec()
        );
        for tag in [
            TagHash(0x80A60B32),
            TagHash(0x80A60B36),
            TagHash(0x80A9A0C1),
            TagHash(0x80A9A0C3),
        ] {
            eprintln!(
                "REAL_MOD tag={tag} nearest={:?} roots={:?}",
                pattern_nearest_geometry_tags(&cache, tag),
                ancestor_pattern_roots(&cache, tag)
            );
        }
        for geometry in [
            TagHash(0x80A61D7E),
            TagHash(0x80A9A580),
            TagHash(0x80A60B32),
            TagHash(0x80A9A0C1),
            TagHash(0x80A60B66),
        ] {
            for component in cache
                .hashes
                .get(&geometry)
                .into_iter()
                .flat_map(|scan| scan.references.iter().copied())
                .filter(|tag| {
                    package_manager()
                        .get_entry(*tag)
                        .is_some_and(|entry| entry.reference == CLASS_PATTERN_COMPONENT)
                })
            {
                let data = package_manager().read_tag(component).expect("component");
                let offsets = cache.hashes[&component]
                    .file_hashes
                    .iter()
                    .filter(|reference| reference.hash == geometry)
                    .map(|reference| reference.offset as usize)
                    .collect_vec();
                eprintln!(
                    "SOCKET geometry={geometry} component={component} size={} offsets={offsets:X?}",
                    data.len()
                );
                for child in cache.hashes[&component]
                    .file_hashes
                    .iter()
                    .map(|reference| reference.hash)
                    .unique()
                {
                    let Some(entry) = package_manager().get_entry(child) else {
                        continue;
                    };
                    if !matches!(entry.reference, 0x8080BA53 | 0x8080BAF8) {
                        continue;
                    }
                    let child_data = package_manager().read_tag(child).unwrap_or_default();
                    eprintln!(
                        "  VALUE tag={child} class={:08X} len={} words={:?}",
                        entry.reference,
                        child_data.len(),
                        child_data
                            .chunks_exact(4)
                            .map(|bytes| {
                                let raw = u32::from_le_bytes(bytes.try_into().unwrap());
                                format!("{raw:08X}/{}", f32::from_bits(raw))
                            })
                            .collect_vec()
                    );
                }
                for offset in offsets {
                    let start = offset.saturating_sub(64);
                    let end = (offset + 80).min(data.len());
                    let words = data[start..end]
                        .chunks_exact(4)
                        .enumerate()
                        .map(|(word, bytes)| {
                            let raw = u32::from_le_bytes(bytes.try_into().unwrap());
                            format!(
                                "+{:04X}={raw:08X}/{}",
                                start + word * 4,
                                f32::from_bits(raw)
                            )
                        })
                        .collect_vec();
                    eprintln!("  WORDS {words:?}");
                }
            }
        }
        let technique = TagHash(0x80A60B20);
        let technique_entry = package_manager().get_entry(technique).unwrap();
        let technique_data = package_manager().read_tag(technique).unwrap();
        let preview = crate::material::MaterialTagPreview::load(&technique_entry, &technique_data)
            .expect("magazine technique");
        let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
        for stage in preview.stages {
            eprintln!(
                "MAG_TECH stage={} shader={:?} textures={:?} bindings={:?} constants={:?} externs={:?}",
                stage.stage,
                stage.shader,
                stage.textures,
                stage.bytecode.bindings,
                stage.constants,
                stage.bytecode.externs
            );
        }
        for parent in cache.hashes[&TagHash(0x80A617EC)]
            .references
            .iter()
            .copied()
        {
            let Some(entry) = package_manager().get_entry(parent) else {
                continue;
            };
            let name = get_class_by_id(entry.reference)
                .map(|class| class.name.into_owned())
                .unwrap_or_else(|| "unknown".to_owned());
            eprintln!(
                "DECAL_PARENT texture=80A617EC parent={parent} class={:08X}:{name}",
                entry.reference
            );
        }
        let mut frontier = vec![(TagHash(0x80A60B56), 0usize)];
        let mut seen = rustc_hash::FxHashSet::default();
        seen.insert(TagHash(0x80A60B56));
        while let Some((child, depth)) = frontier.pop() {
            if depth >= 5 {
                continue;
            }
            for parent in cache
                .hashes
                .get(&child)
                .into_iter()
                .flat_map(|scan| scan.references.iter().copied())
            {
                if !seen.insert(parent) {
                    continue;
                }
                let Some(entry) = package_manager().get_entry(parent) else {
                    continue;
                };
                let name = get_class_by_id(entry.reference)
                    .map(|class| class.name.into_owned())
                    .unwrap_or_else(|| "unknown".to_owned());
                eprintln!(
                    "DECAL_UP depth={} child={child} parent={parent} class={:08X}:{name}",
                    depth + 1,
                    entry.reference
                );
                frontier.push((parent, depth + 1));
            }
        }
        let magazine_entry = package_manager().get_entry(TagHash(0x80A60B32)).unwrap();
        let mut magazine_wireframe = parse_model_wireframe(TagHash(0x80A60B32), &magazine_entry)
            .unwrap()
            .1;
        let magazine_techniques =
            find_model_technique_entries(&cache, TagHash(0x80A60B32), &magazine_entry);
        let magazine_textures =
            find_model_textures(&cache, TagHash(0x80A60B32), &magazine_techniques);
        assign_wireframe_material_textures(&mut magazine_wireframe, &cache, &magazine_textures);
        for range in magazine_wireframe.material_ranges {
            eprintln!(
                "MAG_RANGE stage={:?} tech={:?} start={} count={} color={:?} control={:?} aux={:?}",
                range.render_stage,
                range.technique,
                range.index_start,
                range.index_count,
                range.textures.color,
                range.textures.control,
                range.textures.aux
            );
        }
        let magazine_data = package_manager().read_tag(TagHash(0x80A60B32)).unwrap();
        for range in
            geometry_index_range_candidates(&magazine_data, package_manager().version.endian())
        {
            eprintln!(
                "MAG_CAND part={} stage={:?} tech={} variant={} start={} count={} lod={} flags={:08X} dye={}",
                range.part_index,
                range.render_stage,
                range.technique,
                range.variant_shader_index,
                range.index_start,
                range.index_count,
                range.lod_category,
                range.flags,
                range.gear_dye_change_color_index
            );
        }
        let root = TagHash(0x80A97015);
        let mut frontier = vec![(root, 0usize)];
        let mut seen = rustc_hash::FxHashSet::default();
        seen.insert(root);
        while let Some((node, depth)) = frontier.pop() {
            if depth >= 6 {
                continue;
            }
            for child in cache.hashes.get(&node).into_iter().flat_map(|scan| {
                scan.file_hashes
                    .iter()
                    .map(|reference| reference.hash)
                    .chain(
                        scan.file_hashes64
                            .iter()
                            .filter_map(|reference| tag64_to_hash32(reference.hash)),
                    )
            }) {
                let Some(entry) = package_manager().get_entry(child) else {
                    continue;
                };
                if !matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                    || !seen.insert(child)
                {
                    continue;
                }
                if entry.reference == CLASS_PATTERN {
                    let nearest = pattern_nearest_geometry_tags(&cache, child);
                    if !nearest.is_empty() {
                        eprintln!(
                            "ROOT_BRANCH depth={} pattern={child} geometry={:?}",
                            depth + 1,
                            nearest.iter().map(|tag| tag.to_string()).collect_vec()
                        );
                    }
                }
                frontier.push((child, depth + 1));
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_inventory_mod_render_paths() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        for start in [TagHash(0x80A9A582), TagHash(0x80A61D82)] {
            let mut queue = std::collections::VecDeque::from([(start, vec![start])]);
            let mut seen = rustc_hash::FxHashSet::default();
            seen.insert(start);
            let mut hits = vec![];
            while let Some((node, path)) = queue.pop_front() {
                if path.len() > 14 {
                    continue;
                }
                let Some(scan) = cache.hashes.get(&node) else {
                    continue;
                };
                let neighbors = scan
                    .references
                    .iter()
                    .copied()
                    .chain(scan.file_hashes.iter().map(|reference| reference.hash))
                    .chain(
                        scan.file_hashes64
                            .iter()
                            .filter_map(|reference| tag64_to_hash32(reference.hash)),
                    )
                    .unique()
                    .collect_vec();
                for next in neighbors {
                    if !seen.insert(next) {
                        continue;
                    }
                    let Some(entry) = package_manager().get_entry(next) else {
                        continue;
                    };
                    let mut next_path = path.clone();
                    next_path.push(next);
                    if entry.reference == CLASS_GEOMETRY_RESOURCE {
                        hits.push(next_path.clone());
                        if hits.len() >= 32 {
                            break;
                        }
                    }
                    queue.push_back((next, next_path));
                }
                if hits.len() >= 32 {
                    break;
                }
            }
            for path in hits {
                let annotated = path
                    .into_iter()
                    .map(|tag| {
                        let class = package_manager()
                            .get_entry(tag)
                            .map(|entry| entry.reference)
                            .unwrap_or_default();
                        format!("{tag}:{class:08X}")
                    })
                    .join(" -> ");
                eprintln!("MOD_PATH {start}: {annotated}");
            }
        }

        for tag in [TagHash(0x80AA0E95), TagHash(0x80A9A582)] {
            let scan = cache.hashes.get(&tag).expect("tag scan");
            for reference in &scan.file_hashes {
                let class = package_manager()
                    .get_entry(reference.hash)
                    .map(|entry| entry.reference)
                    .unwrap_or_default();
                eprintln!(
                    "MOD_DIRECT {tag} +{:X} -> {}:{class:08X}",
                    reference.offset, reference.hash
                );
            }
            let data = package_manager().read_tag(tag).expect("tag payload");
            for array in scan_arrays(&data, package_manager().version.endian()) {
                eprintln!(
                    "MOD_ARRAY {tag} class={:08X} count={} data={:X} end={:X}",
                    array.class, array.count, array.data_offset, array.end_offset
                );
                if array.class == 0x8080BAC2 {
                    for (index, record) in array_records(&data, array, 0x28)
                        .into_iter()
                        .enumerate()
                        .filter(|(_index, record)| {
                            read_u32_at(record, 0x20, package_manager().version.endian())
                                == Some(0x80A60015)
                        })
                    {
                        let words = record
                            .chunks_exact(4)
                            .map(|bytes| read_u32(bytes, package_manager().version.endian()))
                            .map(|raw| format!("{raw:08X}"))
                            .collect_vec();
                        eprintln!("MOD_ROOT_DESCRIPTOR {tag} index={index} words={words:?}");
                    }
                }
            }
        }
        for tag in [TagHash(0x80A9BC45), TagHash(0x80A9BC46)] {
            eprintln!(
                "MOD_COMPONENT {tag} nearest={:?}",
                pattern_nearest_geometry_tags(&cache, tag)
            );
            let scan = cache.hashes.get(&tag).expect("component scan");
            for reference in &scan.file_hashes {
                let class = package_manager()
                    .get_entry(reference.hash)
                    .map(|entry| entry.reference)
                    .unwrap_or_default();
                eprintln!(
                    "MOD_COMPONENT_REF {tag} +{:X} -> {}:{class:08X}",
                    reference.offset, reference.hash
                );
            }
        }
        for owner in cache.hashes[&TagHash(0x80A60015)]
            .references
            .iter()
            .copied()
            .filter(|tag| {
                package_manager()
                    .get_entry(*tag)
                    .is_some_and(|entry| entry.reference == CLASS_PATTERN_COMPONENT)
            })
        {
            let data = package_manager().read_tag(owner).expect("descriptor owner");
            for reference in cache.hashes[&owner]
                .file_hashes
                .iter()
                .filter(|reference| reference.hash == TagHash(0x80A60015))
            {
                let end = reference.offset as usize + 4;
                let start = end.saturating_sub(0x28);
                let words = data[start..end]
                    .chunks_exact(4)
                    .map(|bytes| read_u32(bytes, package_manager().version.endian()))
                    .map(|raw| format!("{raw:08X}"))
                    .collect_vec();
                eprintln!(
                    "MOD_DESCRIPTOR_RECORD owner={owner} offset={:X} words={words:?}",
                    reference.offset
                );
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_magazine_component_pose() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        for (component, geometry) in [
            (TagHash(0x80A9BC45), TagHash(0x80AA0E92)),
            (TagHash(0x80A9A357), TagHash(0x80A61D7E)),
            (TagHash(0x80A9B013), TagHash(0x80A9A580)),
            (TagHash(0x80A6076C), TagHash(0x80A60B32)),
        ] {
            let data = package_manager().read_tag(component).expect("component");
            let geometry_offsets = cache.hashes[&component]
                .file_hashes
                .iter()
                .filter(|reference| reference.hash == geometry)
                .map(|reference| reference.offset as usize)
                .collect_vec();
            let entry = package_manager()
                .get_entry(geometry)
                .expect("geometry entry");
            let wireframe = parse_model_wireframe(geometry, &entry)
                .expect("wireframe")
                .1;
            eprintln!(
                "POSE component={component} geometry={geometry} offsets={geometry_offsets:X?} bounds={:?}..{:?}",
                wireframe.min, wireframe.max
            );
            for array in scan_arrays(&data, package_manager().version.endian()) {
                if geometry_offsets
                    .iter()
                    .any(|offset| *offset >= array.data_offset && *offset < array.end_offset)
                {
                    let start = array.data_offset;
                    let end = array.end_offset.min(data.len());
                    eprintln!(
                        "POSE_ARRAY class={:08X} count={} data={start:X} end={end:X} bytes={}",
                        array.class,
                        array.count,
                        data[start..end]
                            .iter()
                            .map(|byte| format!("{byte:02X}"))
                            .join(" ")
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_updated_weapon_pattern_words() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let mut wordlist = rustc_hash::FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            wordlist.entry(hash).or_insert_with(|| word.to_owned());
        });
        for (name, model) in [
            ("BRRT SMG", TagHash(0x80A7B0B0)),
            ("Bully SMG", TagHash(0x80A7AD83)),
            ("Longshot", TagHash(0x80A7AD6C)),
            ("Misriah 2442", TagHash(0x80A7ACEC)),
            ("V11 Punch", TagHash(0x80A7AEC7)),
            ("V22 Volt Thrower", TagHash(0x80A7ADA4)),
        ] {
            let words = descendant_pattern_nodes(&cache, model, 8)
                .into_iter()
                .filter_map(|node| cache.hashes.get(&node))
                .flat_map(|scan| scan.wordlist_hashes.iter().map(|word| word.hash))
                .unique()
                .filter_map(|hash| Some((hash, wordlist.get(&hash)?.clone())))
                .sorted_by_key(|(_hash, word)| word.clone())
                .collect_vec();
            eprintln!("WEAPON_PATTERN_WORDS name={name} model={model} words={words:X?}");
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_updated_weapon_socket_ancestry() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        for (name, model) in [
            ("Bully SMG", TagHash(0x80A7AD83)),
            ("Misriah 2442", TagHash(0x80A7ACEC)),
            ("V99 Channel Rifle", TagHash(0x80A7C86E)),
            ("Biotoxic Disinjector", TagHash(0x80A7E10C)),
            ("BR33 Volley Rifle", TagHash(0x80A7AA89)),
            ("BRRT SMG", TagHash(0x80A7B0B0)),
            ("Copperhead RF", TagHash(0x80A7ADC2)),
            ("Demolition HMG", TagHash(0x80A7B89F)),
            ("Longshot", TagHash(0x80A7AD6C)),
            ("M77 Assault Rifle", TagHash(0x80A7C262)),
            ("Outland", TagHash(0x80A7C70D)),
            ("Stryder M1T", TagHash(0x80A7B354)),
            ("Twin Tap HBR", TagHash(0x80A7B336)),
            ("V00 ZEUS RG", TagHash(0x80A7AD4A)),
            ("WSTR Combat Shotgun", TagHash(0x80A7AC1F)),
        ] {
            let rooted = rooted_pattern_equivalent(&cache, model);
            let ancestors = std::iter::once((0usize, model))
                .chain(rooted.map(|root| (0, root)))
                .chain(ancestor_pattern_roots(&cache, model))
                .chain(
                    rooted
                        .into_iter()
                        .flat_map(|root| ancestor_pattern_roots(&cache, root)),
                )
                .unique_by(|(_depth, candidate)| *candidate)
                .map(|(depth, candidate)| {
                    let tables = descendant_pattern_nodes_with_depth(&cache, candidate, 12)
                        .into_iter()
                        .filter_map(|(node, node_depth)| {
                            let poses = weapon_attachment_poses(node);
                            (!poses.is_empty()).then_some((
                                node_depth,
                                node,
                                poses.iter().filter(|pose| pose.bone_index != 0).count(),
                                poses.len(),
                            ))
                        })
                        .collect_vec();
                    (depth, candidate, tables)
                })
                .filter(|(_depth, _candidate, tables)| !tables.is_empty())
                .collect_vec();
            eprintln!(
                "UPDATED_SOCKET_ANCESTRY name={name} model={model} rooted={rooted:?} candidates={ancestors:?}"
            );
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_weapon_socket_resources() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let start = TagHash(0x80AA0E95);
        let targets = [0x80809779, 0x808081DD, 0x80808B66, 0x80806D8A];
        let mut queue = std::collections::VecDeque::from([(start, vec![start])]);
        let mut seen = rustc_hash::FxHashSet::default();
        seen.insert(start);
        while let Some((node, path)) = queue.pop_front() {
            if path.len() > 12 || seen.len() > 50_000 {
                continue;
            }
            let Some(scan) = cache.hashes.get(&node) else {
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
                .chain(scan.references.iter().copied())
                .unique()
            {
                if !seen.insert(child) {
                    continue;
                }
                let Some(entry) = package_manager().get_entry(child) else {
                    continue;
                };
                let mut child_path = path.clone();
                child_path.push(child);
                if targets.contains(&entry.reference) {
                    eprintln!(
                        "SOCKET_RESOURCE class={:08X} path={}",
                        entry.reference,
                        child_path.iter().format(" -> ")
                    );
                }
                queue.push_back((child, child_path));
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_magazine_vertex_streams() {
        init_goliath_test_package_manager();
        for geometry in [TagHash(0x80A60B32), TagHash(0x80A61D7E)] {
            let data = package_manager().read_tag(geometry).expect("geometry");
            for array in scan_arrays(&data, package_manager().version.endian())
                .into_iter()
                .filter(|array| array.class == CLASS_GEOMETRY_BUFFER_SET)
            {
                for record in array_records(&data, array, 0x80) {
                    let refs = (0..0x20)
                        .step_by(4)
                        .map(|offset| {
                            read_u32_at(record, offset, package_manager().version.endian())
                        })
                        .collect_vec();
                    eprintln!(
                        "STREAMS geometry={geometry} layout={:?} refs={refs:08X?}",
                        geometry_buffer_set_input_layout_id(record)
                    );
                    for tag in refs
                        .into_iter()
                        .flatten()
                        .map(TagHash)
                        .filter(|tag| package_manager().get_entry(*tag).is_some())
                    {
                        let entry = package_manager().get_entry(tag).unwrap();
                        let payload = package_manager().read_tag(tag).unwrap_or_default();
                        eprintln!(
                            "STREAM tag={tag} class={:08X} len={} head={}",
                            entry.reference,
                            payload.len(),
                            payload
                                .iter()
                                .take(32)
                                .map(|byte| format!("{byte:02X}"))
                                .join(" ")
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_weapon_pattern_owners() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let weapon = TagHash(0x80AA0E95);
        for owner in cache.hashes[&weapon].references.iter().copied() {
            let entry = package_manager().get_entry(owner).expect("owner");
            let data = package_manager().read_tag(owner).unwrap_or_default();
            eprintln!(
                "WEAPON_OWNER tag={owner} class={:08X} len={} refs={:?} arrays={:?}",
                entry.reference,
                data.len(),
                cache
                    .hashes
                    .get(&owner)
                    .into_iter()
                    .flat_map(|scan| scan.file_hashes.iter())
                    .map(|reference| (
                        reference.offset,
                        reference.hash,
                        package_manager()
                            .get_entry(reference.hash)
                            .map(|entry| entry.reference)
                    ))
                    .collect_vec(),
                scan_arrays(&data, package_manager().version.endian())
                    .into_iter()
                    .map(|array| (
                        array.class,
                        array.count,
                        array.data_offset,
                        array.end_offset
                    ))
                    .collect_vec()
            );
            eprintln!(
                "WEAPON_OWNER_BYTES {owner} {}",
                data.iter().map(|byte| format!("{byte:02X}")).join(" ")
            );
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_root_hierarchy_records() {
        init_goliath_test_package_manager();
        let data = package_manager()
            .read_tag(TagHash(0x80A9700D))
            .expect("root component");
        for offset in (0x4B0..data.len()).step_by(8) {
            let bytes = &data[offset..(offset + 8).min(data.len())];
            eprintln!(
                "ROOT_WORD {offset:04X} {}",
                bytes.iter().map(|byte| format!("{byte:02X}")).join(" ")
            );
        }
        for offset in [0x188usize, 0x1B8, 0x1E8, 0x218, 0x248, 0x480] {
            let record = &data[offset..offset + 0x30];
            eprintln!(
                "ROOT_RECORD {offset:04X} {:?}",
                record
                    .chunks_exact(4)
                    .map(|bytes| {
                        let raw = u32::from_le_bytes(bytes.try_into().unwrap());
                        format!("{raw:08X}/{}", f32::from_bits(raw))
                    })
                    .collect_vec()
            );
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_weapon_translation_candidates() {
        init_goliath_test_package_manager();
        for tag in [
            TagHash(0x80AA0E95),
            TagHash(0x80A9BC45),
            TagHash(0x80A9700D),
            TagHash(0x80A97015),
            TagHash(0x80AA30DD),
        ] {
            let data = package_manager().read_tag(tag).unwrap_or_default();
            for offset in (0..data.len().saturating_sub(4)).step_by(4) {
                let raw = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let value = f32::from_bits(raw);
                if value.is_finite() && (0.08..0.18).contains(&value.abs()) {
                    eprintln!("TRANSLATION_CANDIDATE tag={tag} offset={offset:X} value={value}");
                    if tag == TagHash(0x80A97015) {
                        let start = offset.saturating_sub(0x40);
                        let end = (offset + 0x44).min(data.len());
                        eprintln!(
                            "TRANSLATION_CONTEXT {}",
                            data[start..end]
                                .chunks_exact(4)
                                .enumerate()
                                .map(|(index, bytes)| {
                                    let raw = u32::from_le_bytes(bytes.try_into().unwrap());
                                    format!(
                                        "+{:X}={raw:08X}/{}",
                                        start + index * 4,
                                        f32::from_bits(raw)
                                    )
                                })
                                .join(" ")
                        );
                        for array in scan_arrays(&data, package_manager().version.endian())
                            .into_iter()
                            .filter(|array| {
                                offset >= array.data_offset && offset < array.end_offset
                            })
                        {
                            eprintln!(
                                "TRANSLATION_ARRAY class={:08X} count={} data={:X} end={:X}",
                                array.class, array.count, array.data_offset, array.end_offset
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_goliath_weapon_pattern_mod_nodes() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let mut words = rustc_hash::FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            words.entry(hash).or_insert_with(|| word.to_owned());
        });
        if let Ok(extra_words) =
            std::fs::read_to_string("alkahest/crates/alkahest/wordlist_channels.txt")
        {
            for word in extra_words.lines() {
                words
                    .entry(quicktag_core::util::fnv1(word.as_bytes()))
                    .or_insert_with(|| word.to_owned());
            }
        }
        for word in words.values().unique() {
            for candidate in [
                format!("{word}0"),
                format!("{word}_0"),
                format!("{word}.0"),
                format!("parent.{word}0"),
                format!("parent.{word}_0"),
            ] {
                if quicktag_core::util::fnv1(candidate.as_bytes()) == 0xD5754C50 {
                    eprintln!("resolved D5754C50 channel prefix: {candidate}");
                }
            }
        }

        for selected in [
            TagHash(0x80B6CD7A),
            TagHash(0x80AA03CA),
            TagHash(0x80AA0E95),
            TagHash(0x80AA0ECF),
            TagHash(0x80A9A582),
            TagHash(0x80A61D82),
        ] {
            let mut ancestors = vec![selected];
            let mut seen = rustc_hash::FxHashSet::default();
            let mut roots = vec![];
            seen.insert(selected);
            while let Some(child) = ancestors.pop() {
                for parent in cache
                    .hashes
                    .get(&child)
                    .into_iter()
                    .flat_map(|scan| scan.references.iter().copied())
                {
                    let class = package_manager()
                        .get_entry(parent)
                        .map(|entry| entry.reference)
                        .unwrap_or_default();
                    if matches!(class, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                        && seen.insert(parent)
                    {
                        if class == CLASS_PATTERN {
                            roots.push(parent);
                        }
                        ancestors.push(parent);
                    }
                }
            }
            roots.sort();
            roots.dedup();
            eprintln!("selected={selected} roots={roots:?}");

            for root in roots {
                let mut frontier = vec![root];
                let mut visited = rustc_hash::FxHashSet::default();
                while let Some(node) = frontier.pop() {
                    if !visited.insert(node) {
                        continue;
                    }
                    let Some(entry) = package_manager().get_entry(node) else {
                        continue;
                    };
                    let scan = cache.hashes.get(&node);
                    let child_classes = scan
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
                        .filter_map(|child| {
                            package_manager()
                                .get_entry(child)
                                .map(|entry| (entry.reference, child))
                        })
                        .fold(
                            rustc_hash::FxHashMap::<u32, rustc_hash::FxHashSet<TagHash>>::default(),
                            |mut classes, (class, child)| {
                                classes.entry(class).or_default().insert(child);
                                classes
                            },
                        );
                    for config in child_classes
                        .get(&0x8080BA53)
                        .into_iter()
                        .flat_map(|tags| tags.iter().copied())
                    {
                        let data = package_manager().read_tag(config).unwrap_or_default();
                        let config_scan = cache.hashes.get(&config);
                        let config_words = config_scan
                            .into_iter()
                            .flat_map(|scan| scan.wordlist_hashes.iter())
                            .filter_map(|hash| {
                                words
                                    .get(&hash.hash)
                                    .map(|word| (hash.offset, word.clone()))
                            })
                            .collect_vec();
                        let refs = config_scan
                            .into_iter()
                            .flat_map(|scan| scan.file_hashes.iter())
                            .map(|reference| {
                                let class = package_manager()
                                    .get_entry(reference.hash)
                                    .map(|entry| entry.reference)
                                    .unwrap_or_default();
                                (reference.offset, reference.hash, class)
                            })
                            .collect_vec();
                        eprintln!(
                            "    config={config} len={} words={config_words:?} refs={refs:?} arrays={:?}",
                            data.len(),
                            scan_arrays(&data, package_manager().version.endian())
                                .into_iter()
                                .map(|array| (
                                    array.class,
                                    array.count,
                                    array.data_offset,
                                    array.end_offset
                                ))
                                .collect_vec(),
                        );
                        if matches!(config.0, 0x80A9BDA3 | 0x80A9BB64) {
                            for (line, bytes) in data.chunks(16).enumerate() {
                                eprintln!(
                                    "      {config} {:04X}: {}",
                                    line * 16,
                                    bytes.iter().map(|byte| format!("{byte:02X}")).join(" ")
                                );
                            }
                        }
                    }
                    let names = scan
                        .into_iter()
                        .flat_map(|scan| scan.wordlist_hashes.iter())
                        .filter_map(|hash| words.get(&hash.hash).map(|word| (hash.offset, word)))
                        .filter(|(_offset, word)| {
                            let word = word.to_ascii_lowercase();
                            word.contains("mod")
                                || word.contains("optic")
                                || word.contains("muzzle")
                                || word.contains("magazine")
                                || word.contains("grip")
                                || word.contains("socket")
                                || word.contains("attach")
                                || word.contains("variant")
                        })
                        .collect_vec();
                    let children = scan
                        .into_iter()
                        .flat_map(|scan| scan.file_hashes.iter().map(|reference| reference.hash))
                        .filter(|child| {
                            package_manager().get_entry(*child).is_some_and(|entry| {
                                matches!(
                                    entry.reference,
                                    CLASS_PATTERN
                                        | CLASS_PATTERN_COMPONENT
                                        | CLASS_GEOMETRY_RESOURCE
                                )
                            })
                        })
                        .unique()
                        .collect_vec();
                    let arrays = package_manager()
                        .read_tag(node)
                        .ok()
                        .map(|data| {
                            scan_arrays(&data, package_manager().version.endian())
                                .into_iter()
                                .map(|array| {
                                    (
                                        array.class,
                                        array.count,
                                        array.data_offset,
                                        array.end_offset,
                                    )
                                })
                                .collect_vec()
                        })
                        .unwrap_or_default();
                    if matches!(node.0, 0x80B6CD7B | 0x80B6D76B) {
                        let data = package_manager().read_tag(node).unwrap_or_default();
                        for array in scan_arrays(&data, package_manager().version.endian())
                            .into_iter()
                            .filter(|array| {
                                matches!(
                                    array.class,
                                    0x8080AF13 | 0x8080AF14 | 0x8080BF07 | 0x8080AF7B
                                )
                            })
                        {
                            let stride = match array.class {
                                0x8080AF13 => 0x30,
                                0x8080AF14 => 0x28,
                                0x8080BF07 => 0x10,
                                0x8080AF7B => 0x4,
                                _ => unreachable!(),
                            };
                            for (index, record) in
                                array_records(&data, array, stride).into_iter().enumerate()
                            {
                                let fields = record
                                    .chunks_exact(4)
                                    .enumerate()
                                    .map(|(field, bytes)| {
                                        let value =
                                            read_u32(bytes, package_manager().version.endian());
                                        let meaning =
                                            words.get(&value).cloned().unwrap_or_default();
                                        format!("+{:02X}={value:08X}:{meaning}", field * 4)
                                    })
                                    .collect_vec();
                                eprintln!(
                                    "    selector node={node} class={:08X} index={index}: {fields:?}",
                                    array.class
                                );
                            }
                        }
                    }
                    if node == root {
                        let data = package_manager().read_tag(node).unwrap_or_default();
                        let mut descriptors = rustc_hash::FxHashSet::default();
                        for array in scan_arrays(&data, package_manager().version.endian())
                            .into_iter()
                            .filter(|array| {
                                matches!(array.class, 0x8080BA61 | 0x8080BAC2 | 0x8080BAC0)
                            })
                        {
                            let stride = match array.class {
                                0x8080BA61 => 0x38,
                                0x8080BAC2 => 0x28,
                                0x8080BAC0 => 0x18,
                                _ => unreachable!(),
                            };
                            for (index, record) in
                                array_records(&data, array, stride).into_iter().enumerate()
                            {
                                if array.class == 0x8080BAC2 {
                                    if let Some(tag) = read_u32_at(
                                        record,
                                        0x20,
                                        package_manager().version.endian(),
                                    )
                                    .map(TagHash)
                                    .filter(|tag| {
                                        package_manager()
                                            .get_entry(*tag)
                                            .is_some_and(|entry| entry.reference == 0x8080BAF8)
                                    }) {
                                        descriptors.insert(tag);
                                    }
                                }
                                let fields = record
                                    .chunks_exact(4)
                                    .enumerate()
                                    .map(|(field, bytes)| {
                                        let value =
                                            read_u32(bytes, package_manager().version.endian());
                                        let meaning = words
                                            .get(&value)
                                            .cloned()
                                            .or_else(|| {
                                                package_manager().get_entry(TagHash(value)).map(
                                                    |entry| format!("tag:{:08X}", entry.reference),
                                                )
                                            })
                                            .unwrap_or_default();
                                        format!("+{:02X}={value:08X}:{meaning}", field * 4)
                                    })
                                    .collect_vec();
                                eprintln!(
                                    "    root_record root={root} class={:08X} index={index}: {fields:?}",
                                    array.class
                                );
                            }
                        }
                        for descriptor in descriptors.into_iter().sorted() {
                            let payload =
                                package_manager().read_tag(descriptor).unwrap_or_default();
                            let names = cache
                                .hashes
                                .get(&descriptor)
                                .into_iter()
                                .flat_map(|scan| scan.wordlist_hashes.iter())
                                .filter_map(|hash| {
                                    words
                                        .get(&hash.hash)
                                        .map(|word| (hash.offset, word.clone()))
                                })
                                .collect_vec();
                            eprintln!(
                                "    descriptor root={root} tag={descriptor} names={names:?} bytes={}",
                                payload.iter().map(|byte| format!("{byte:02X}")).join(" ")
                            );
                        }
                    }
                    eprintln!(
                        "  node={node} class={:08X} len={} names={names:?} children={children:?} child_classes={child_classes:?} arrays={arrays:?}",
                        entry.reference, entry.file_size,
                    );
                    frontier.extend(children.iter().copied().filter(|child| {
                        package_manager().get_entry(*child).is_some_and(|entry| {
                            matches!(entry.reference, CLASS_PATTERN | CLASS_PATTERN_COMPONENT)
                        })
                    }));
                }
                let geometry = pattern_geometry_tags(&cache, root);
                eprintln!("  root={root} geometry_count={}", geometry.len());
                for tag in geometry {
                    let transform = package_manager().read_tag(tag).ok().and_then(|data| {
                        read_geometry_position_transform(&data, package_manager().version.endian())
                    });
                    eprintln!("    geometry={tag} transform={transform:?}");
                }
            }
        }
    }

    #[test]
    fn parses_vertex_buffer_header_little_endian() {
        let data = [
            0x80, 0x00, 0x00, 0x00, // data_size
            0x20, 0x00, // stride
            0x03, 0x00, // vtype
            0xEF, 0xBE, 0xAD, 0xDE, // marker
        ];

        let header = VertexBufferHeader::parse(&data, Endian::Little).unwrap();
        assert_eq!(header.data_size, 0x80);
        assert_eq!(header.stride, 0x20);
        assert_eq!(header.vtype, 3);
        assert_eq!(header.deadbeef, 0xDEADBEEF);
    }

    #[test]
    fn parses_index_buffer_header_little_endian() {
        let data = [
            0x7F, 0x01, // unk0, is_32bit
            0x34, 0x12, // unk1
            0x00, 0x00, 0x00, 0x00, // zero
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // data_size
            0xEF, 0xBE, 0xAD, 0xDE, // marker
            0x00, 0x00, 0x00, 0x00, // zero1
        ];

        let header = IndexBufferHeader::parse(&data, Endian::Little).unwrap();
        assert_eq!(header.unk0, 0x7F);
        assert!(header.is_32bit);
        assert_eq!(header.unk1, 0x1234);
        assert_eq!(header.data_size, 0x40);
        assert_eq!(header.deadbeef, 0xDEADBEEF);
    }

    #[test]
    fn expands_triangle_strips_across_restart_indices() {
        assert_eq!(
            preview_triangles_from_indices(&[0, 1, 2, 3, 0xFFFF, 4, 5, 6], 5),
            vec![0, 1, 2, 2, 1, 3, 4, 5, 6]
        );
    }

    #[test]
    fn recognizes_all_alkahest_high_detail_lods() {
        for lod in [0, 1, 2, 3, 10] {
            assert!(is_highest_detail_lod(lod));
        }
        for lod in [4, 7, 8, 9] {
            assert!(!is_highest_detail_lod(lod));
        }
    }

    #[test]
    fn selects_visible_marathon_render_stages() {
        let boundaries = [0, 6, 6, 9, 9, 12, 12, 15, 18, 18];
        assert_eq!(
            preview_part_indices_from_boundaries(&boundaries, 18),
            Some((0..9).chain(12..18).collect())
        );
        assert!(preview_part_indices_from_boundaries(&[0, 3, 2], 3).is_none());

        let stages = preview_part_stages_from_boundaries(&boundaries, 18).expect("stage map");
        assert_eq!(stages[0], Some(0));
        assert_eq!(stages[4], Some(0));
        assert_eq!(stages[8], Some(2));
        assert_eq!(stages[12], Some(6));
        assert_eq!(stages[14], Some(6));
        assert_eq!(stages[15], Some(7));
        assert_eq!(stages[10], None);
    }

    #[test]
    fn applies_geometry_position_dequantization() {
        let mut wireframe = WireframePreview {
            source: "test".into(),
            position_format: "i16x4.xyz @ +0",
            uv_format: None,
            vertices: vec![[-32767.0, 0.0, 32767.0]],
            normals: None,
            procedural_positions: Some(vec![[-32767.0, 0.0, 32767.0]]),
            procedural_normals: None,
            tangents: None,
            uvs: None,
            normal_format: None,
            tangent_format: None,
            indices: vec![],
            material_ranges: vec![],
            min: [-32767.0, 0.0, 32767.0],
            max: [-32767.0, 0.0, 32767.0],
            vertex_count_total: 1,
            index_count_total: 0,
        };
        apply_geometry_position_transform(
            &mut wireframe,
            GeometryPositionTransform {
                scale: [2.0, 4.0, 8.0],
                offset: [10.0, 20.0, 30.0],
                procedural_scale: 0.25,
            },
        );
        assert_eq!(wireframe.vertices, vec![[8.0, 20.0, 38.0]]);
        assert_eq!(wireframe.procedural_positions, Some(vec![[-1.0, 0.0, 1.0]]));
        assert_eq!(wireframe.min, [8.0, 20.0, 38.0]);
    }

    #[test]
    fn socket_pose_preserves_gear_dye_procedural_coordinates() {
        let geometry = TagHash(1);
        let source = MeshSourcePreview {
            kind: "test",
            buffer_index: 0,
            technique: None,
            index_start: 0,
            index_count: 0,
            primitive_type: 0,
            lod_category: 0,
            input_layout_index: None,
            index_buffer: TagHash(0),
            vertex0_buffer: TagHash(0),
            vertex1_buffer: TagHash(0),
            color_buffer: TagHash(0),
            uv_transform: None,
            shader_constants: vec![],
        };
        let wireframe = WireframePreview {
            source: "test".into(),
            position_format: "f32x3",
            uv_format: None,
            vertices: vec![[1.0, 0.0, 0.0]],
            normals: Some(vec![[1.0, 0.0, 0.0]]),
            procedural_positions: None,
            procedural_normals: None,
            tangents: None,
            uvs: None,
            normal_format: None,
            tangent_format: None,
            indices: vec![],
            material_ranges: vec![],
            min: [1.0, 0.0, 0.0],
            max: [1.0, 0.0, 0.0],
            vertex_count_total: 1,
            index_count_total: 0,
        };
        let half_sqrt = std::f32::consts::FRAC_1_SQRT_2;
        let attachment = ResolvedWeaponModAttachment {
            geometry,
            pose: WeaponModAttachmentPose {
                family_id: 0,
                variant_id: 0,
                bone_index: 0,
                rotation: [0.0, 0.0, half_sqrt, half_sqrt],
                translation: [5.0, 6.0, 7.0],
            },
            rarity: Some(WeaponModRarity::Enhanced),
            unique_id: 0.25,
        };
        let mut parts = vec![(geometry, source, wireframe)];

        apply_weapon_mod_attachment_poses(&mut parts, &[attachment]);

        let transformed = &parts[0].2;
        for (actual, expected) in transformed.vertices[0].into_iter().zip([5.0, 7.0, 7.0]) {
            assert!((actual - expected).abs() < 0.000_01);
        }
        assert_eq!(
            transformed.procedural_positions,
            Some(vec![[1.0, 0.0, 0.0]])
        );
        assert_eq!(transformed.procedural_normals, Some(vec![[1.0, 0.0, 0.0]]));
    }

    #[test]
    fn reads_relative_array_payload() {
        let mut data = vec![0u8; 0x40];
        data[0x08..0x10].copy_from_slice(&2u64.to_le_bytes());
        data[0x10..0x18].copy_from_slice(&0x10i64.to_le_bytes());
        data[0x20..0x28].copy_from_slice(&2u64.to_le_bytes());
        data[0x28..0x2c].copy_from_slice(&0x80806D37u32.to_le_bytes());
        data[0x30..0x34].copy_from_slice(&0x11223344u32.to_le_bytes());
        data[0x34..0x38].copy_from_slice(&0x55667788u32.to_le_bytes());

        let array = read_array(&data, 0x08, 4, Endian::Little).unwrap();
        assert_eq!(array, &data[0x30..0x38]);
    }

    #[test]
    fn reads_static_mesh_technique_tags() {
        let mut data = vec![0u8; 0xa0];
        let technique0 = TagHash::new(0x0102, 0x0304);
        let technique1 = TagHash::new(0x0506, 0x0708);
        let technique2 = TagHash::new(0x0a0b, 0x0c0d);
        data[0x10..0x18].copy_from_slice(&2u64.to_le_bytes());
        data[0x18..0x20].copy_from_slice(&0x28i64.to_le_bytes());
        data[0x40..0x48].copy_from_slice(&2u64.to_le_bytes());
        data[0x50..0x54].copy_from_slice(&technique0.0.to_le_bytes());
        data[0x54..0x58].copy_from_slice(&technique1.0.to_le_bytes());

        data[0x20..0x28].copy_from_slice(&1u64.to_le_bytes());
        data[0x28..0x30].copy_from_slice(&0x38i64.to_le_bytes());
        data[0x60..0x68].copy_from_slice(&1u64.to_le_bytes());
        data[0x90..0x94].copy_from_slice(&technique2.0.to_le_bytes());

        let tags = static_mesh_technique_tags(&data, Endian::Little);

        assert_eq!(tags, vec![technique0, technique1, technique2]);
    }

    #[test]
    fn reads_dynamic_mesh_part_technique_tags() {
        let mut data = vec![0u8; 0xa0];
        let technique0 = TagHash::new(0x0102, 0x0304);
        let technique1 = TagHash::new(0x0506, 0x0708);
        data[0x20..0x28].copy_from_slice(&2u64.to_le_bytes());
        data[0x28..0x30].copy_from_slice(&0x18i64.to_le_bytes());
        data[0x40..0x48].copy_from_slice(&2u64.to_le_bytes());
        data[0x50..0x54].copy_from_slice(&technique0.0.to_le_bytes());
        data[0x74..0x78].copy_from_slice(&technique1.0.to_le_bytes());

        let tags = dynamic_mesh_technique_tags(&data, Endian::Little);

        assert_eq!(tags, vec![technique0, technique1]);
    }

    #[test]
    fn scans_candidate_tag_hashes_in_blob() {
        let little = TagHash::new(0x0102, 0x0304);
        let big = TagHash::new(0x0506, 0x0708);
        let mut data = vec![0u8; 0x10];
        data[0x04..0x08].copy_from_slice(&little.0.to_le_bytes());
        data[0x0c..0x10].copy_from_slice(&big.0.to_be_bytes());

        let little_tags = candidate_tag_hashes_in_blob(&data, Endian::Little);
        let big_tags = candidate_tag_hashes_in_blob(&data, Endian::Big);

        assert!(little_tags.contains(&little));
        assert!(big_tags.contains(&big));
    }

    #[test]
    fn reads_half_float_values() {
        assert_eq!(read_f16(&0x3c00u16.to_le_bytes(), Endian::Little), 1.0);
        assert_eq!(read_f16(&0xc000u16.to_le_bytes(), Endian::Little), -2.0);
        assert_eq!(read_f16(&0x3800u16.to_le_bytes(), Endian::Little), 0.5);
    }

    #[test]
    fn finds_half_uv_candidates() {
        let mut data = vec![0u8; 0x18];
        data[0x04..0x06].copy_from_slice(&0x0000u16.to_le_bytes());
        data[0x06..0x08].copy_from_slice(&0x0000u16.to_le_bytes());
        data[0x0c..0x0e].copy_from_slice(&0x3c00u16.to_le_bytes());
        data[0x0e..0x10].copy_from_slice(&0x3800u16.to_le_bytes());
        data[0x14..0x16].copy_from_slice(&0x4000u16.to_le_bytes());
        data[0x16..0x18].copy_from_slice(&0x3c00u16.to_le_bytes());

        let candidate = candidate_f16x2_uv(&data, 0x08, 0x04, Endian::Little).unwrap();

        assert_eq!(candidate.format, UvFormat::F16x2);
        assert_eq!(candidate.offset, 0x04);
        assert_eq!(candidate.valid_vertices, 3);
        assert_eq!(candidate.min, [0.0, 0.0]);
        assert_eq!(candidate.max, [2.0, 1.0]);
    }

    #[test]
    fn decodes_input_layout_snorm_uvs() {
        let mut data = vec![0u8; 0x10];
        data[0x00..0x02].copy_from_slice(&0x4000i16.to_le_bytes());
        data[0x02..0x04].copy_from_slice(&(-0x4000i16).to_le_bytes());
        data[0x08..0x0a].copy_from_slice(&0x7fffi16.to_le_bytes());
        data[0x0a..0x0c].copy_from_slice(&0i16.to_le_bytes());

        let uvs = decode_input_layout_uvs(
            &data,
            0x08,
            Endian::Little,
            InputLayoutTexcoord {
                buffer_index: 0,
                offset: 0,
                format: InputLayoutFormat::R16G16Snorm,
            },
            2,
        );

        assert_eq!(uvs.len(), 2);
        assert!((uvs[0][0] - 0.50001526).abs() < 0.00001);
        assert!((uvs[0][1] + 0.50001526).abs() < 0.00001);
        assert_eq!(uvs[1], [1.0, 0.0]);
    }

    #[test]
    fn decodes_and_normalizes_packed_snorm_tangent() {
        let mut data = vec![0u8; 0x10];
        data[0x04..0x06].copy_from_slice(&0x4000i16.to_le_bytes());
        data[0x06..0x08].copy_from_slice(&0i16.to_le_bytes());
        data[0x08..0x0a].copy_from_slice(&0x4000i16.to_le_bytes());
        data[0x0a..0x0c].copy_from_slice(&(-1i16).to_le_bytes());

        let vectors = decode_input_layout_vectors(
            &data,
            0x10,
            Endian::Little,
            InputLayoutVector {
                buffer_index: 0,
                offset: 4,
                format: InputLayoutFormat::R16G16B16A16Snorm,
            },
            1,
        )
        .expect("packed tangent");
        let tangent = normalize_input_layout_vector(vectors[0], true).expect("unit tangent");

        assert!((tangent[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.0001);
        assert_eq!(tangent[1], 0.0);
        assert!((tangent[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.0001);
        assert_eq!(tangent[3], -1.0);
    }

    #[test]
    fn propagates_uv_transform_to_shader_constant() {
        let constants = shader_constants_from_uv_transform(Some(UvTransformPreview {
            scale: [2.0, 3.0],
            offset: [0.25, 0.5],
        }));

        assert_eq!(constants.len(), 1);
        assert_eq!(constants[0].name, "uv_scale_offset");
        assert_eq!(constants[0].value, [2.0, 3.0, 0.25, 0.5]);
        assert_eq!(constants[0].source, "mesh instance UV transform");
    }

    #[test]
    fn reads_geometry_buffer_set_generate_gbuffer_layout() {
        let mut mesh = vec![0u8; 0x80];
        mesh[0x62] = 0x57;
        mesh[0x64] = 0x07;

        assert_eq!(geometry_buffer_set_input_layout_id(&mesh), Some(7));
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_weapon_attachment_channel_sources() {
        init_goliath_test_package_manager();
        let cache = quicktag_scanner::load_tag_cache();
        let endian = package_manager().version.endian();
        let needles = [
            0xA500D3AF_u32,
            0x6F1709EA,
            0x840253AF,
            0x840253A1,
            0x840254D0,
            0xDECA4D7E,
            0x7B2426A3,
            0x040174F7,
            0xC8939EBF,
            0xC8939EBA,
            0xC8939EBC,
            0xC8939EBD,
            0xC8939EBB,
            0xC8939EB8,
        ];
        for root in [
            TagHash(0x80A7C7B5),
            TagHash(0x80A7D43D),
            TagHash(0x80A60D38),
            TagHash(0x80A61CF0),
        ] {
            for node in descendant_pattern_nodes(&cache, root, 12) {
                let Ok(data) = package_manager().read_tag(node) else {
                    continue;
                };
                for offset in (0..data.len().saturating_sub(4)).step_by(4) {
                    let Some(value) = read_u32_at(&data, offset, endian) else {
                        continue;
                    };
                    if needles.contains(&value) {
                        eprintln!(
                            "ATTACHMENT_CHANNEL_HIT root={root} node={node} class={:08X} offset=0x{offset:X} value={value:08X} words={:?}",
                            package_manager()
                                .get_entry(node)
                                .map(|entry| entry.reference)
                                .unwrap_or_default(),
                            (offset.saturating_sub(0x20)..(offset + 0x40).min(data.len()))
                                .step_by(4)
                                .filter_map(|field| read_u32_at(&data, field, endian))
                                .map(|word| format!("{word:08X}"))
                                .collect_vec()
                        );
                    }
                }
            }
        }
        for root in [
            TagHash(0x80A60D38),
            TagHash(0x80A60C4F),
            TagHash(0x80A61CF0),
            TagHash(0x80A6071A),
        ] {
            let root_data = package_manager().read_tag(root).expect("mod pattern root");
            for array in scan_arrays(&root_data, endian)
                .into_iter()
                .filter(|array| array.class == 0x8080BA61)
            {
                for (index, record) in array_records(&root_data, array, 0x38)
                    .into_iter()
                    .enumerate()
                {
                    eprintln!(
                        "MOD_ROOT_BINDING root={root} index={index} words={:?}",
                        record
                            .chunks_exact(4)
                            .map(|bytes| format!("{:08X}", read_u32(bytes, endian)))
                            .collect_vec()
                    );
                }
            }
            for node in descendant_pattern_nodes(&cache, root, 8) {
                let Ok(data) = package_manager().read_tag(node) else {
                    continue;
                };
                for array in scan_arrays(&data, endian)
                    .into_iter()
                    .filter(|array| array.class == 0x8080AF86)
                {
                    for (channel_index, record) in
                        array_records(&data, array, 0x70).into_iter().enumerate()
                    {
                        let Some(parameter) = read_u32_at(record, 0, endian) else {
                            continue;
                        };
                        if [
                            0x5A116CBA_u32,
                            0xF714F29F,
                            0x3AF60B50,
                            0xC17A8BB6,
                            0xC8939EBF,
                            0xC8939EBA,
                            0xC8939EBC,
                            0xC8939EBD,
                            0xC8939EBB,
                            0xC8939EB8,
                        ]
                        .contains(&parameter)
                        {
                            eprintln!(
                                "MOD_CHANNEL_RECORD root={root} node={node} index={channel_index} parameter={parameter:08X} words={:?}",
                                record
                                    .chunks_exact(4)
                                    .map(|bytes| format!("{:08X}", read_u32(bytes, endian)))
                                    .collect_vec()
                            );
                        }
                    }
                }
                let arrays = scan_arrays(&data, endian);
                let vectors = arrays
                    .iter()
                    .copied()
                    .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
                    .map(|array| read_serialized_dye_vector(&data, array, endian))
                    .collect::<Option<Vec<_>>>();
                if let Some(vectors) = vectors {
                    for array in arrays
                        .into_iter()
                        .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
                    {
                        for record in array_records(&data, array, 0x0c) {
                            let Some(parameter) = read_u32_at(record, 4, endian) else {
                                continue;
                            };
                            let Some(index) = read_u32_at(record, 8, endian)
                                .and_then(|index| usize::try_from(index).ok())
                            else {
                                continue;
                            };
                            if let Some(value) = vectors.get(index) {
                                eprintln!(
                                    "MOD_LOCAL_VECTOR root={root} node={node} parameter={parameter:08X} value={value:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_quickdraw_grip_age_material() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let root = TagHash(0x80A601B6);
        let endian = package_manager().version.endian();
        let mut words = rustc_hash::FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            words.entry(hash).or_insert_with(|| word.to_owned());
        });

        eprintln!("QUICKDRAW root={root}");
        for node in descendant_pattern_nodes(&cache, root, 10) {
            let entry = package_manager().get_entry(node).expect("pattern node");
            let data = package_manager().read_tag(node).expect("pattern node data");
            let arrays = scan_arrays(&data, endian);
            eprintln!(
                "  NODE tag={node} class={:08X} len=0x{:X} arrays={:?}",
                entry.reference,
                data.len(),
                arrays
                    .iter()
                    .map(|array| (array.class, array.count, array.data_offset))
                    .collect_vec()
            );

            for array in arrays
                .iter()
                .copied()
                .filter(|array| array.class == 0x8080AF86)
            {
                for (index, record) in array_records(&data, array, 0x70).into_iter().enumerate() {
                    let parameter = read_u32_at(record, 0, endian).unwrap_or_default();
                    let record_offset = array.data_offset + index * 0x70;
                    let bytecode =
                        read_array(&data, record_offset + 0x08, 1, endian).unwrap_or_default();
                    let constants = read_array(&data, record_offset + 0x18, 0x10, endian)
                        .unwrap_or_default()
                        .chunks_exact(0x10)
                        .filter_map(|constant| read_vec4_f32(constant, 0, endian))
                        .collect_vec();
                    eprintln!(
                        "    OBJECT_CHANNEL index={index} parameter={parameter:08X} name={:?} bytecode={} constants={constants:?}",
                        words.get(&parameter),
                        bytecode.iter().map(|byte| format!("{byte:02X}")).join(" ")
                    );
                }
            }

            let vectors = arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_VECTOR4 && array.count == 1)
                .map(|array| read_serialized_dye_vector(&data, array, endian))
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default();
            for array in arrays
                .iter()
                .copied()
                .filter(|array| array.class == CLASS_PATTERN_VECTOR_BINDINGS)
            {
                for record in array_records(&data, array, 0x0c) {
                    let parameter = read_u32_at(record, 4, endian).unwrap_or_default();
                    let value = read_u32_at(record, 8, endian)
                        .and_then(|index| vectors.get(index as usize))
                        .copied();
                    eprintln!(
                        "    LOCAL_VECTOR parameter={parameter:08X} name={:?} value={value:?}",
                        words.get(&parameter),
                    );
                }
            }
        }

        let geometries = pattern_nearest_geometry_tags(&cache, root);
        eprintln!("  GEOMETRIES {geometries:?}");
        for geometry in geometries {
            let entry = package_manager().get_entry(geometry).expect("mod geometry");
            let techniques = find_model_technique_entries(&cache, geometry, &entry);
            let textures = find_model_textures(&cache, geometry, &techniques);
            let (_source, mut wireframe) =
                parse_model_wireframe(geometry, &entry).expect("mod wireframe");
            assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
            eprintln!(
                "  GEOMETRY {geometry} techniques={:?} textures={:?}",
                techniques.iter().map(|(tag, _)| *tag).collect_vec(),
                textures.iter().map(|(tag, _)| *tag).collect_vec(),
            );
            for range in &wireframe.material_ranges {
                let Some(technique) = range.technique else {
                    continue;
                };
                eprintln!(
                    "    RANGE indices={}+{} dye={:?} technique={technique} material={:?}",
                    range.index_start,
                    range.index_count,
                    range.gear_dye_change_color_index,
                    range.textures,
                );
                let technique_entry = package_manager().get_entry(technique).expect("technique");
                let technique_data = package_manager()
                    .read_tag(technique)
                    .expect("technique data");
                for binding in texture_bindings_for_technique(&technique_entry, &technique_data)
                    .into_iter()
                    .filter(|binding| binding.stage == "PS")
                {
                    let descriptor = Texture::load_data_d2(binding.tag, false)
                        .map(|(desc, _, _)| {
                            format!("{}x{} {:?}", desc.width, desc.height, desc.format)
                        })
                        .unwrap_or_else(|_| "non-texture".to_owned());
                    eprintln!("      PS t{}={} {descriptor}", binding.slot, binding.tag);
                }
                let preview =
                    crate::material::MaterialTagPreview::load(&technique_entry, &technique_data)
                        .expect("technique preview");
                let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
                if let Some(pixel) = preview.stages.iter().find(|stage| stage.stage == "PS") {
                    eprintln!(
                        "      SHADER {:?} bindings={:?} expressions={:?} inline={:?}",
                        pixel.shader,
                        pixel
                            .bytecode
                            .bindings
                            .iter()
                            .map(|binding| (&binding.kind, binding.slot, &binding.source))
                            .collect_vec(),
                        pixel
                            .bytecode
                            .expressions
                            .iter()
                            .map(|expression| (&expression.target, &expression.expression))
                            .collect_vec(),
                        pixel.constants,
                    );
                    for age in -2..=6 {
                        let channels = std::collections::HashMap::from([
                            (0x138D_E801, [age as f32; 4]),
                            (0xD358_3E54, [0.0; 4]),
                        ]);
                        let (_bindings, expressions) =
                            crate::material::interpret_tfx_stack_with_object_channels(
                                &pixel.bytecode.ops,
                                &pixel.constants,
                                &channels,
                            );
                        eprintln!(
                            "      AGE {age}: {:?}",
                            expressions
                                .iter()
                                .filter(|expression| {
                                    matches!(
                                        expression.target.as_str(),
                                        "output[24]" | "output[42]" | "output[49]"
                                    )
                                })
                                .map(|expression| (&expression.target, expression.value))
                                .collect_vec()
                        );
                    }
                    for expression in pixel.bytecode.expressions.iter().filter(|expression| {
                        matches!(
                            expression.target.as_str(),
                            "output[24]" | "output[42]" | "output[49]"
                        )
                    }) {
                        let Some(index) = pixel
                            .bytecode
                            .ops
                            .iter()
                            .position(|op| op.offset == expression.op_offset)
                        else {
                            continue;
                        };
                        eprintln!("      AGE_OUTPUT_OPS target={}:", expression.target);
                        for source in &pixel.bytecode.ops[index.saturating_sub(12)..=index] {
                            eprintln!(
                                "        {:04X} {:02X} {} {}",
                                source.offset, source.opcode, source.name, source.detail
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn resolves_authored_three_tier_weapon_mod_condition() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());

        for (root, expected_technique) in [
            (TagHash(0x80A601B6), TagHash(0x80A601A2)),
            (TagHash(0x80A60313), TagHash(0x80A602FF)),
        ] {
            let mut wear_materials = Vec::new();
            let mut resolved_techniques = Vec::new();
            for geometry in pattern_nearest_geometry_tags(&cache, root) {
                let entry = package_manager().get_entry(geometry).expect("mod geometry");
                let technique_entries = find_model_technique_entries(&cache, geometry, &entry);
                let textures = find_model_textures(&cache, geometry, &technique_entries);
                let (_source, mut wireframe) =
                    parse_model_wireframe(geometry, &entry).expect("mod wireframe");
                assign_wireframe_material_textures(&mut wireframe, &cache, &textures);
                resolved_techniques.extend(
                    wireframe
                        .material_ranges
                        .iter()
                        .filter_map(|range| range.technique),
                );
                wear_materials.extend(
                    wireframe
                        .material_ranges
                        .into_iter()
                        .filter_map(|range| range.textures.mod_wear),
                );
            }

            assert!(
                resolved_techniques.contains(&expected_technique),
                "{root} resolved {resolved_techniques:?}, expected own material {expected_technique}"
            );
            assert!(!wear_materials.is_empty(), "{root} has no wear material");
            for wear in wear_materials {
                assert_eq!(wear.scratches_projection, [4.0, -4.0, 0.0, 0.0]);
                assert_eq!(wear.scratches_remap_base, [0.0, 0.0, 0.0, 1.0]);
                assert_eq!(
                    wear.scratches_remap_scale,
                    [0.21404114, 0.21404114, 0.21404114, 0.0]
                );
                assert_eq!(wear.condition_blend, 0.5);
                assert_eq!(wear.grime_projection_unique_delta, [0.0, 0.0, 1.0, 1.0]);
                assert_eq!(wear.damage_projection_unique_delta, [0.0, 0.0, 1.0, 1.0]);
                assert_eq!(
                    wear.condition_controls,
                    [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    "{root} must use authored Enhanced/Deluxe/Superior TFX outputs"
                );
                for projection in [
                    wear.scratches_projection,
                    wear.grime_projection,
                    wear.damage_projection,
                ] {
                    assert!(projection.iter().all(|value| value.is_finite()));
                    assert!(projection[0].abs() > 0.05 && projection[1].abs() > 0.05);
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_weapon_mod_unique_id_projection() {
        init_goliath_test_package_manager();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());

        for (root, expected_technique) in [
            (TagHash(0x80A601B6), TagHash(0x80A601A2)),
            (TagHash(0x80A60313), TagHash(0x80A602FF)),
        ] {
            let geometry = pattern_nearest_geometry_tags(&cache, root)
                .into_iter()
                .next()
                .expect("mod geometry");
            let geometry_entry = package_manager()
                .get_entry(geometry)
                .expect("geometry entry");
            let techniques = find_model_technique_entries(&cache, geometry, &geometry_entry);
            assert!(techniques.iter().any(|(tag, _)| *tag == expected_technique));
            let technique_entry = package_manager()
                .get_entry(expected_technique)
                .expect("technique entry");
            let technique_data = package_manager()
                .read_tag(expected_technique)
                .expect("technique data");
            let preview = MaterialTagPreview::load(&technique_entry, &technique_data)
                .expect("technique preview");
            let MaterialPreviewKind::Technique(preview) = preview.kind;
            let pixel = preview
                .stages
                .iter()
                .find(|stage| stage.stage == "PS")
                .expect("pixel stage");

            eprintln!("UNIQUE_ID root={root} technique={expected_technique}");
            for seed in [0.0_f32, 0.125, 0.25, 0.5, 0.75, 1.0] {
                let channels = std::collections::HashMap::from([
                    (WEAPON_MOD_AGE_CHANNEL, [1.0; 4]),
                    (UNIQUE_ID_CHANNEL, [seed; 4]),
                ]);
                let (_bindings, expressions) = interpret_tfx_stack_with_object_channels(
                    &pixel.bytecode.ops,
                    &pixel.constants,
                    &channels,
                );
                let values = expressions
                    .iter()
                    .filter(|expression| {
                        matches!(
                            expression.target.as_str(),
                            "output[39]"
                                | "output[40]"
                                | "output[24]"
                                | "output[42]"
                                | "output[49]"
                        )
                    })
                    .map(|expression| (&expression.target, expression.value))
                    .collect_vec();
                eprintln!("  seed={seed:.3} {values:?}");
            }
        }

        for technique in [
            TagHash::new(0x130, 7672),
            TagHash::new(0x130, 3910),
            TagHash::new(0x130, 3194),
            TagHash::new(0x130, 4713),
            TagHash::new(0x130, 954),
        ] {
            let entry = package_manager()
                .get_entry(technique)
                .expect("technique entry");
            let data = package_manager()
                .read_tag(technique)
                .expect("technique data");
            let bindings = texture_bindings_for_technique(&entry, &data);
            let preview = MaterialTagPreview::load(&entry, &data).expect("technique preview");
            let MaterialPreviewKind::Technique(preview) = preview.kind;
            let pixel = preview
                .stages
                .iter()
                .find(|stage| stage.stage == "PS")
                .expect("pixel stage");
            eprintln!(
                "ALT_WEAR technique={technique} shader={:?} bindings={bindings:?} inline_len={} c14={:?} c15={:?} c16={:?} c50={:?}",
                pixel.shader,
                pixel.inline_constants.len(),
                pixel.inline_constants.get(14),
                pixel.inline_constants.get(15),
                pixel.inline_constants.get(16),
                pixel.inline_constants.get(50),
            );
            let age_targets = pixel
                .bytecode
                .expressions
                .iter()
                .filter(|expression| expression.expression.contains("object_channel(0x138DE801)"))
                .filter_map(|expression| {
                    expression
                        .target
                        .strip_prefix("output[")?
                        .strip_suffix(']')?
                        .parse::<usize>()
                        .ok()
                })
                .collect_vec();
            if let [first, _, third] = age_targets.as_slice() {
                eprintln!(
                    "  dynamic_inline scratch={:?} remap_base={:?} remap_scale={:?} blend={:?}",
                    pixel.inline_constants.get(first.saturating_sub(10)),
                    pixel.inline_constants.get(first.saturating_sub(9)),
                    pixel.inline_constants.get(first.saturating_sub(8)),
                    pixel.inline_constants.get(third + 1),
                );
            }
            eprintln!(
                "  age_expressions={:?}",
                pixel
                    .bytecode
                    .expressions
                    .iter()
                    .filter(|expression| {
                        expression.expression.contains("object_channel(0x138DE801)")
                            || expression.expression.contains("object_channel(0xD3583E54)")
                    })
                    .collect_vec()
            );
            for tier in [1.0_f32, 2.0, 3.0] {
                let channels = std::collections::HashMap::from([
                    (WEAPON_MOD_AGE_CHANNEL, [tier; 4]),
                    (UNIQUE_ID_CHANNEL, [0.25; 4]),
                ]);
                let (_bindings, expressions) = interpret_tfx_stack_with_object_channels(
                    &pixel.bytecode.ops,
                    &pixel.constants,
                    &channels,
                );
                let values = expressions
                    .iter()
                    .filter(|expression| {
                        matches!(
                            expression.target.as_str(),
                            "output[24]"
                                | "output[39]"
                                | "output[40]"
                                | "output[42]"
                                | "output[49]"
                        )
                    })
                    .map(|expression| (&expression.target, expression.value))
                    .collect_vec();
                eprintln!("  tier={tier} {values:?}");
            }
            eprintln!(
                "  wear={:?}",
                weapon_mod_wear_material(technique, &bindings)
            );
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages and GPU"]
    fn exports_quickdraw_grip_age_textures() {
        init_goliath_test_package_manager();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU adapter");
        let required_features = adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            ..Default::default()
        }))
        .expect("GPU device");
        let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let renderer = eframe::egui_wgpu::Renderer::new(
            &device,
            target_format,
            eframe::egui_wgpu::RendererOptions::default(),
        );
        let render_state = eframe::egui_wgpu::RenderState {
            adapter,
            available_adapters: vec![],
            device,
            queue,
            target_format,
            renderer: Arc::new(eframe::egui::mutex::RwLock::new(renderer)),
        };

        let textures = [
            TagHash(0x80A61633),
            TagHash(0x80A615D5),
            TagHash(0x80A4050F),
            TagHash(0x80A61670),
            TagHash(0x80A60055),
            TagHash(0x80A6149D),
            TagHash(0x80A61508),
            TagHash(0x80A6149F),
            TagHash(0x80A46D44),
        ];
        let tile = 256_u32;
        let columns = 3_u32;
        let rows = textures.len().div_ceil(columns as usize) as u32;
        let mut sheet = image::RgbaImage::new(columns * tile, rows * tile);
        let output = std::path::Path::new("target/quicktag-mod-age-probe");
        std::fs::create_dir_all(output).expect("probe directory");
        for (index, texture) in textures.into_iter().enumerate() {
            let loaded = Texture::load(&render_state, texture, false).expect("texture upload");
            let image = loaded.to_image(&render_state, 0).expect("texture capture");
            image
                .save(output.join(format!("{index}-t{index}-{texture}.png")))
                .expect("texture export");
            let thumbnail = image.thumbnail(tile, tile).to_rgba8();
            let x = index as u32 % columns * tile + (tile - thumbnail.width()) / 2;
            let y = index as u32 / columns * tile + (tile - thumbnail.height()) / 2;
            image::imageops::overlay(&mut sheet, &thumbnail, x.into(), y.into());
        }
        sheet
            .save(output.join("quickdraw-material-sheet.png"))
            .expect("contact sheet");
    }
}

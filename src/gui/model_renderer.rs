#[path = "model_surfaces.rs"]
mod model_surfaces;
use model_surfaces::{LoadedNativeSurface, NativeSurfaceSource};
#[path = "model_decals.rs"]
mod model_decals;
use model_decals::NativeDecalSource;


use std::{
    collections::HashMap,
    ops::Range,
    path::Path,
    sync::{Arc, LazyLock, Mutex},
    time::Instant,
};

use eframe::{
    egui,
    egui_wgpu::{
        CallbackResources, CallbackTrait, ScreenDescriptor,
        wgpu::{self, util::DeviceExt},
    },
};
use itertools::Itertools;
use rayon::prelude::*;
use tiger_pkg::{TagHash, package_manager};

use crate::{
    geometry::{
        AlphaMaskMaterial, AuthoredGeometryInput, AuthoredIndexBufferRef,
        AuthoredInputLayoutDescriptor, AuthoredStageInputLayout, AuthoredVertexStreamRef,
        CharacterSurfaceMaterial, ForwardCoatingMaterial, GearDyeMaterial, GearPatternMaterial,
        GeometryPositionTransform, InvestmentDecalMaskMode, InvestmentDecalMaterial,
        InvestmentDecalMode, RunnerLayeredSurfaceMaterial, RunnerOcclusionMaterial,
        SharedAtlasDetailMaterial, TransmissionMaterial, UvTransformPreview,
        WeaponModAttachmentPose, WeaponModConditionMaterial, WeaponSurfaceConditionMaterial,
        WireframeMaterialTextures, WireframePreview,
    },
    material::{TechniqueRenderState, is_sticker_proxy_technique, render_state_for_technique},
    render::{
        TigerDrawPacket,
        evidence::{EvidenceLevel, ProvenanceId, ProvenanceRecord, ProvenanceStore, SourceSpan},
        material::MaterialIR,
        pass_plan::{DrawPassPlan, RenderPassKind},
        technique::{ShaderStage, TechniqueDescriptor, VertexAbiDescriptor},
    },
    texture::{
        Texture, TextureType,
        cache::{MaterialTextureKey, TextureCache},
        linear_texture_format, srgb_texture_format,
    },
};





fn create_model_texture_view(
    texture: &wgpu::Texture,
    descriptor: &wgpu::TextureViewDescriptor<'_>,
) -> wgpu::TextureView {
    let view = texture.create_view(descriptor);
    view
}

fn create_model_bind_group(
    device: &wgpu::Device,
    descriptor: &wgpu::BindGroupDescriptor<'_>,
) -> wgpu::BindGroup {
    let group = device.create_bind_group(descriptor);
    group
}

const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DISTORTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
// RGBA8 keeps five logical targets under WebGPU's portable 32-byte/sample cap.
// These are research/debug contracts; compatibility HDR remains RGBA16F.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SURFACE_PROPERTIES_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
// Authored emission is HDR: the engine's packed intensity reaches 64.
const SURFACE_EMISSIVE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SURFACE_FLAGS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
// Alkahest's Tiger shadow view uses a 2048x2048 shadow surface. Keep the
// compatibility renderer at the same scale instead of retaining a 4K preview
// allocation that consumed 64 MiB by itself.
const SHADOW_MAP_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
    ambient_occlusion: f32,
    procedural_position: [f32; 3],
    procedural_normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AuthoredShadowUniform {
    geometry_scale: [f32; 4],
    geometry_offset: [f32; 4],
    attachment_rotation: [f32; 4],
    attachment_translation_scale: [f32; 4],
    uv_transform: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct AuthoredStageMetadata {
    raw_stage: u8,
    input_layout_id: u8,
    part_index: usize,
    source_index_start: u32,
    source_index_count: u32,
    primitive_type: u8,
    variant_shader_index: u16,
    flags: u32,
    lod_run: u8,
}

#[derive(Clone)]
struct ModelDraw {
    indices: Range<u32>,
    packet: TigerDrawPacket,
    material: Option<MaterialTextureKey>,
    solid_color: Option<[f32; 4]>,
    solid_surface: Option<[f32; 2]>,
    iridescence_id: Option<f32>,
    transmission: Option<TransmissionMaterial>,
    forward_coating: Option<ForwardCoatingMaterial>,
    character_surface: Option<CharacterSurfaceMaterial>,
    runner_layered_surface: Option<RunnerLayeredSurfaceMaterial>,
    runner_occlusion: Option<RunnerOcclusionMaterial>,
    alpha_mask: Option<AlphaMaskMaterial>,
    shared_atlas_detail: Option<SharedAtlasDetailMaterial>,
    control: Option<TagHash>,
    roughness_channel: u8,
    mask_palette: Option<[[f32; 4]; 2]>,
    gear_dye: Option<GearDyeMaterial>,
    gear_dye_default: Option<[f32; 4]>,
    gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    gear_worn_dye_palette: Option<[[f32; 4]; 6]>,
    gear_dye_detail_palette: Option<[[f32; 4]; 6]>,
    mod_wear: Option<WeaponModConditionMaterial>,
    surface_condition: Option<WeaponSurfaceConditionMaterial>,
    gear_pattern: Option<GearPatternMaterial>,
    investment_decal: Option<InvestmentDecalMaterial>,
    authored_shared_atlas: bool,
    sampler: Option<TagHash>,
    sticker_proxy: bool,
    pipeline: ModelPipelineKey,
    center: [f32; 3],
    /// Uniform packed-position scale from Tiger's scope_skinning[5].w.
    procedural_scale: f32,
    authored_source: Option<usize>,
    authored_stage: Option<AuthoredStageMetadata>,
    /// Whether Quicktag's raw-stream compatibility VS can honor this
    /// authored stage's vertex contract. Shader families that consume runtime
    /// vertex resources (for example SV_VertexID -> t2 packed positions) must
    /// use the reconstructed-position fallback until that resource ABI exists.
    authored_native_vertex_supported: bool,
    native_c827: Option<[[f32; 4]; 9]>,
    native_surface: Option<NativeSurfaceSource>,
    native_decal: Option<NativeDecalSource>,
    native_rejection: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ModelPipelineKey {
    blend: u8,
    depth_stencil: u8,
    rasterizer: u8,
    depth_bias: u8,
    family: crate::render::material::MaterialFamily,
    pass: RenderPassKind,
    technique: Option<TagHash>,
    vertex_layout_bytes: u16,
    target_signature: u16,
}

impl Default for ModelPipelineKey {
    fn default() -> Self {
        Self {
            blend: 0,
            depth_stencil: 2,
            rasterizer: 2,
            depth_bias: 0,
            family: crate::render::material::MaterialFamily::Unknown,
            pass: RenderPassKind::UnknownCompatibility,
            technique: None,
            vertex_layout_bytes: std::mem::size_of::<ModelVertex>() as u16,
            target_signature: 0x501,
        }
    }
}

impl ModelPipelineKey {
    fn select(selection: TechniqueRenderState) -> Self {
        let defaults = if selection.blend.is_some_and(blend_enabled) {
            Self {
                blend: 8,
                depth_stencil: 15,
                rasterizer: 2,
                depth_bias: 1,
                ..Self::default()
            }
        } else {
            Self::default()
        };
        Self {
            blend: selection.blend.unwrap_or(defaults.blend),
            depth_stencil: selection.depth_stencil.unwrap_or(defaults.depth_stencil),
            rasterizer: selection.rasterizer.unwrap_or(defaults.rasterizer),
            depth_bias: selection.depth_bias.unwrap_or(defaults.depth_bias),
            ..defaults
        }
    }

    fn select_for_draw(
        selection: TechniqueRenderState,
        family: crate::render::material::MaterialFamily,
        plan: &DrawPassPlan,
        technique: Option<TagHash>,
    ) -> Self {
        let mut key = Self::select(selection);
        key.family = family;
        key.pass = plan
            .passes
            .iter()
            .copied()
            .find(|pass| *pass != RenderPassKind::Shadow)
            .unwrap_or(RenderPassKind::UnknownCompatibility);
        if matches!(
            key.pass,
            RenderPassKind::Distortion | RenderPassKind::ForwardCoating
        ) {
            // Marathon Distortion writes a transmission/distortion target. Preview
            // composites it after deferred lighting without replacing depth.
            key.blend = 8;
            key.depth_stencil = 15;
            key.depth_bias = 1;
        }
        key.technique = technique;
        key
    }

    fn material_flags(self) -> Self {
        Self {
            pass: RenderPassKind::MaterialFlags,
            blend: 0,
            family: crate::render::material::MaterialFamily::Unknown,
            technique: None,
            ..self
        }
    }

    fn material_emissive(self) -> Self {
        Self {
            pass: RenderPassKind::MaterialEmissive,
            blend: 0,
            family: crate::render::material::MaterialFamily::Unknown,
            technique: None,
            ..self
        }
    }

    fn gpu_equivalent(self, other: Self) -> bool {
        self.blend == other.blend
            && self.depth_stencil == other.depth_stencil
            && self.rasterizer == other.rasterizer
            && self.depth_bias == other.depth_bias
            && self.pass == other.pass
            && self.vertex_layout_bytes == other.vertex_layout_bytes
            && self.target_signature == other.target_signature
    }
}

struct GpuAuthoredVertexStream {
    stream_index: u8,
    header_tag: TagHash,
    data_tag: TagHash,
    stride: u16,
    element_count: u32,
    buffer: wgpu::Buffer,
}

struct GpuAuthoredIndexBuffer {
    header_tag: TagHash,
    data_tag: TagHash,
    format: wgpu::IndexFormat,
    index_count: u32,
    buffer: wgpu::Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StrictShadowPipelineKey {
    layout_id: u8,
    primitive_type: u8,
    rasterizer: u8,
    index_32bit: bool,
    /// Package vertex-buffer strides are part of the IA contract. Layout
    /// element extents are not a substitute because authored buffers may pad.
    stream_strides: [u16; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StrictDepthPipelineKey {
    layout_id: u8,
    primitive_type: u8,
    rasterizer: u8,
    depth_stencil: u8,
    depth_bias: u8,
    index_32bit: bool,
    stream_strides: [u16; 8],
}

struct GpuAuthoredGeometryInput {
    geometry: TagHash,
    vertex_streams: Vec<GpuAuthoredVertexStream>,
    index_buffer: Option<GpuAuthoredIndexBuffer>,
    stage_layouts: Vec<AuthoredStageInputLayout>,
    position_transform: Option<GeometryPositionTransform>,
    uv_transform: Option<UvTransformPreview>,
    attachment_pose: Option<WeaponModAttachmentPose>,
    static_meshes: Vec<crate::render::static_mesh::StaticMesh>,
    static_mesh_parts: HashMap<usize, usize>,
    vertex_colors: Option<GpuAuthoredVertexColors>,
}

struct GpuAuthoredVertexColors {
    data_tag: TagHash,
    count: u32,
    buffer: wgpu::Buffer,
}

impl GpuAuthoredGeometryInput {
    fn static_mesh(&self, stage: AuthoredStageMetadata) -> Option<(usize, &crate::render::static_mesh::StaticMesh)> {
        let index = *self.static_mesh_parts.get(&stage.part_index)?;
        Some((index, self.static_meshes.get(index)?))
    }

    fn stage_layout(&self, raw_stage: u8) -> Option<&AuthoredStageInputLayout> {
        self.stage_layouts
            .iter()
            .find(|layout| layout.raw_stage == raw_stage)
    }

    fn shadow_uniform(&self) -> AuthoredShadowUniform {
        let position = self
            .position_transform
            .unwrap_or(GeometryPositionTransform {
                scale: [1.0; 3],
                offset: [0.0; 3],
                procedural_scale: 1.0,
            });
        let attachment = self.attachment_pose.unwrap_or(WeaponModAttachmentPose {
            family_id: 0,
            variant_id: 0,
            bone_index: 0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0; 3],
            scale: 1.0,
        });
        let uv = self.uv_transform.unwrap_or(UvTransformPreview {
            scale: [1.0; 2],
            offset: [0.0; 2],
        });
        AuthoredShadowUniform {
            geometry_scale: [position.scale[0], position.scale[1], position.scale[2], 0.0],
            geometry_offset: [
                position.offset[0],
                position.offset[1],
                position.offset[2],
                0.0,
            ],
            attachment_rotation: attachment.rotation,
            attachment_translation_scale: [
                attachment.translation[0],
                attachment.translation[1],
                attachment.translation[2],
                attachment.scale,
            ],
            uv_transform: [uv.scale[0], uv.scale[1], uv.offset[0], uv.offset[1]],
        }
    }
}

fn authored_stream_strides(
    source: &GpuAuthoredGeometryInput,
    descriptor: &AuthoredInputLayoutDescriptor,
) -> Option<[u16; 8]> {
    let mut strides = [0u16; 8];
    for stream in &descriptor.streams {
        let slot = usize::from(stream.stream_index);
        let output = strides.get_mut(slot)?;
        let gpu_stream = source
            .vertex_streams
            .iter()
            .find(|candidate| candidate.stream_index == stream.stream_index)?;
        if gpu_stream.stride == 0 {
            return None;
        }
        *output = gpu_stream.stride;
    }
    Some(strides)
}

fn authored_vertex_format(format: u8) -> Option<wgpu::VertexFormat> {
    match format {
        0x02 => Some(wgpu::VertexFormat::Float32x2),
        0x03 => Some(wgpu::VertexFormat::Float32x3),
        0x04 => Some(wgpu::VertexFormat::Float32x4),
        0x0A => Some(wgpu::VertexFormat::Snorm16x2),
        0x0B => Some(wgpu::VertexFormat::Snorm16x4),
        _ => None,
    }
}

fn authored_vertex_format_size(format: u8) -> Option<u64> {
    match format {
        0x02 => Some(8),
        0x03 => Some(12),
        0x04 => Some(16),
        0x0A => Some(4),
        0x0B => Some(8),
        _ => None,
    }
}

fn create_gpu_authored_vertex_stream(
    device: &wgpu::Device,
    stream: &AuthoredVertexStreamRef,
    buffer_cache: &mut HashMap<TagHash, wgpu::Buffer>,
) -> Option<GpuAuthoredVertexStream> {
    let byte_count = usize::try_from(stream.data_size).ok()?;
    let required = u64::from(stream.stride).checked_mul(u64::from(stream.element_count))?;
    if stream.stride == 0 || byte_count == 0 || required > u64::from(stream.data_size) {
        log::error!("Invalid authored vertex view {}: stride={} elements={} bytes={}",
            stream.header_tag, stream.stride, stream.element_count, stream.data_size);
        return None;
    }
    let buffer = if let Some(buffer) = buffer_cache.get(&stream.data_tag) {
        buffer.clone()
    } else {
        let data = package_manager().read_tag(stream.data_tag).ok()?;
        if data.len() < byte_count {
            log::error!("Truncated authored vertex payload {}: expected {byte_count}, got {}",
                stream.data_tag, data.len());
            return None;
        }
        let label = format!(
            "quicktag_authored_stream_{}_{}",
            stream.header_tag, stream.stream_index
        );
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&label),
            contents: &data[..byte_count],
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        buffer_cache.insert(stream.data_tag, buffer.clone());
        buffer
    };
    Some(GpuAuthoredVertexStream {
        stream_index: stream.stream_index,
        header_tag: stream.header_tag,
        data_tag: stream.data_tag,
        stride: stream.stride,
        element_count: stream.element_count,
        buffer,
    })
}

fn create_gpu_authored_index_buffer(
    device: &wgpu::Device,
    index: &AuthoredIndexBufferRef,
    buffer_cache: &mut HashMap<TagHash, wgpu::Buffer>,
) -> Option<GpuAuthoredIndexBuffer> {
    let byte_count = usize::try_from(index.data_size).ok()?;
    let stride = if index.is_32bit { 4 } else { 2 };
    if byte_count == 0 || u64::from(index.index_count).checked_mul(stride)? > index.data_size {
        log::error!("Invalid authored index view {}: elements={} bytes={}",
            index.header_tag, index.index_count, index.data_size);
        return None;
    }
    let buffer = if let Some(buffer) = buffer_cache.get(&index.data_tag) {
        buffer.clone()
    } else {
        let data = package_manager().read_tag(index.data_tag).ok()?;
        if data.len() < byte_count {
            log::error!("Truncated authored index payload {}: expected {byte_count}, got {}",
                index.data_tag, data.len());
            return None;
        }
        let label = format!("quicktag_authored_indices_{}", index.header_tag);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&label),
            contents: &data[..byte_count],
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        buffer_cache.insert(index.data_tag, buffer.clone());
        buffer
    };
    Some(GpuAuthoredIndexBuffer {
        header_tag: index.header_tag,
        data_tag: index.data_tag,
        format: if index.is_32bit {
            wgpu::IndexFormat::Uint32
        } else {
            wgpu::IndexFormat::Uint16
        },
        index_count: index.index_count,
        buffer,
    })
}

fn create_gpu_authored_geometry_input(
    device: &wgpu::Device,
    source: &AuthoredGeometryInput,
    required_stages: &[u8],
    native_static_required: &[AuthoredStageMetadata],
    vertex_color_required: bool,
    vertex_buffer_cache: &mut HashMap<TagHash, wgpu::Buffer>,
    index_buffer_cache: &mut HashMap<TagHash, wgpu::Buffer>,
    rigid_float_palette: &mut Option<wgpu::Buffer>,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Option<GpuAuthoredGeometryInput> {
    let required_streams = source
        .stage_layouts
        .iter()
        .filter(|layout| required_stages.contains(&layout.raw_stage))
        .filter_map(|layout| layout.descriptor.as_ref())
        .flat_map(|descriptor| descriptor.streams.iter().map(|stream| stream.stream_index))
        .unique()
        .collect_vec();

    let vertex_streams = source
        .vertex_streams
        .iter()
        .filter(|stream| required_streams.contains(&stream.stream_index))
        .map(|stream| create_gpu_authored_vertex_stream(device, stream, vertex_buffer_cache))
        .collect::<Option<Vec<_>>>()?;
    if required_streams.iter().any(|slot| !vertex_streams.iter().any(|stream| stream.stream_index == *slot)) {
        log::error!("Missing required authored vertex stream for geometry {}", source.geometry);
        return None;
    }
    let index_buffer = if required_stages.is_empty() { None } else {
        Some(create_gpu_authored_index_buffer(device, source.index_buffer.as_ref()?, index_buffer_cache)?)
    };
    if !native_static_required.is_empty() {
        let indices = index_buffer.as_ref()?;
        if native_static_required.iter().any(|stage| {
            stage.source_index_count == 0 || stage.source_index_start.checked_add(stage.source_index_count)
                .is_none_or(|end| end > indices.index_count)
        }) {
            log::error!("Invalid native strip format/range for geometry {}", source.geometry);
            return None;
        }
    }

    let mut static_mesh_parts = HashMap::new();
    // Direct float IA has no compute stage or generated storage buffers.
    if source.stage_layouts.iter().any(|stage|stage.layout_id==13 && required_stages.contains(&stage.raw_stage)) {
        let geometry = vertex_streams.iter().find(|stream|stream.stream_index == 0)?;
        let uv = vertex_streams.iter().find(|stream|stream.stream_index == 1)?;
        if geometry.stride != 48 || uv.stride != 4 || geometry.element_count != uv.element_count
            || source.uv_transform.is_none() || source.position_transform.is_none()
            || source.attachment_pose.is_some() {
            log::error!("Invalid direct-IA native inputs for geometry {}", source.geometry);
            return None;
        }
    }
    let static_meshes = if !native_static_required.is_empty() {
        use crate::render::authored_program::{DescriptorAbi, resolve_package_program};
        // Producer selection follows the geometry's authored compute parts,
        // independently of its visible VS. Never infer a CS from layout7.
        if source.attachment_pose.is_some() { return None; }
        let mut producers = Vec::new();
        let mut producer_constants = Vec::<Option<Vec<[f32; 4]>>>::new();
        for stage in native_static_required {
            let tag=crate::geometry::geometry_compute_technique(source.geometry, stage.part_index,
                stage.source_index_start..stage.source_index_start.checked_add(stage.source_index_count)?)?;
            let technique = TechniqueDescriptor::load(tag)?;
            let cs = technique.stages.iter().find(|s| s.stage == ShaderStage::Compute)?;
            let abi=resolve_package_program(cs.shader?, ShaderStage::Compute).ok()??.descriptor_abi;
            if !matches!(
                abi,
                DescriptorAbi::BodyMeshComputeStorage
                    | DescriptorAbi::BodyMesh15RowComputeStorage
                    | DescriptorAbi::HeadMeshComputeStorage
                    | DescriptorAbi::HeadMeshEC0BComputeStorage
                    | DescriptorAbi::HeadMesh15RowA60035ComputeStorage
                    | DescriptorAbi::HeadMesh15RowB8BDComputeStorage
                    | DescriptorAbi::HairMeshComputeStorage
                    | DescriptorAbi::HairMesh133RowComputeStorage
                    | DescriptorAbi::BodyProceduralComputeStorage
                    | DescriptorAbi::BodyMeshB4CBComputeStorage
                    | DescriptorAbi::Cloth45RowComputeStorage
                    | DescriptorAbi::Cloth46RowComputeStorage
                    | DescriptorAbi::BodyMeshAA060BComputeStorage
                    | DescriptorAbi::BodyMeshB152BEComputeStorage
            ) {
                return None;
            }
            let constants = if let Some((rows,_)) = abi.deformation_constant_contract() {
                let state=cs.runtime_state(runtime_inputs);
                if cs.constant_buffer_slot != Some(0) || state.constant_registers.len()!=rows
                    || state.constant_registers[14..].iter().flatten().any(|v|!v.is_finite())
                    || state.unresolved_dependencies.iter().filter_map(|value|
                        value.strip_prefix("output[").and_then(|s|s.split_once(']'))
                            .and_then(|(row,_)|row.parse::<usize>().ok()))
                        .any(|row| !matches!(row,0..=4 | 7..=13)) { return None; }
                Some(state.constant_registers)
            } else if matches!(abi, DescriptorAbi::HairMesh133RowComputeStorage
                | DescriptorAbi::BodyProceduralComputeStorage) {
                let state = cs.runtime_state(runtime_inputs);
                let rows = if abi == DescriptorAbi::BodyProceduralComputeStorage { 81 } else { 133 };
                if cs.constant_buffer_slot != Some(0) || state.constant_registers.len() != rows
                    || state.unresolved_dependencies.iter().filter_map(|value|
                        value.strip_prefix("output[").and_then(|s|s.split_once(']'))
                            .and_then(|(row,_)|row.parse::<usize>().ok()))
                        .any(|row| !matches!(row, 0 | 2 | 7..=20)) {
                    return None;
                }
                Some(state.constant_registers)
            } else { None };
            // All source buffers/object inputs are shared within this geometry.
            // Reuse a producer only when its exact shader ABI and complete
            // evaluated constant image match, including signed zero/NaN bits.
            let index = if let Some(index) = producers.iter().enumerate().find_map(|(index, &producer)| {
                let equal = match (&producer_constants[index], &constants) {
                    (None, None) => true,
                    (Some(before), Some(after)) => before.len() == after.len()
                        && before.iter().flatten().zip(after.iter().flatten())
                            .all(|(a,b)| a.to_bits() == b.to_bits()),
                    _ => false,
                };
                (producer == abi && equal).then_some(index)
            }) {
                index
            } else {
                producers.push(abi);
                producer_constants.push(constants);
                producers.len() - 1
            };
            static_mesh_parts.insert(stage.part_index, index);
        }
        let float_source=native_static_required.iter().any(|stage|stage.input_layout_id==13);
        if native_static_required.iter().any(|stage|(stage.input_layout_id==13)!=float_source) {return None;}
        if float_source && producers.iter().any(|p|!matches!(p,DescriptorAbi::BodyMeshComputeStorage|DescriptorAbi::BodyMesh15RowComputeStorage)) {return None;}
        let stride=if float_source {48}else{24};
        let stream = vertex_streams.iter().find(|s| s.stream_index == 0 && s.stride == stride)?;
        let uv = vertex_streams.iter().find(|s| s.stream_index == 1 && s.stride == 4)?;
        if stream.element_count == 0 || uv.element_count < stream.element_count
            || stream.buffer.size() != u64::from(stream.element_count) * u64::from(stride) {
            log::error!("Invalid native layout7 vertex views for geometry {}", source.geometry);
            return None;
        }
        let palette=if float_source {
            if source.skinning_buffer.is_some() {return None;}
            let reference=source.vertex_streams.iter().find(|s|s.stream_index==0)?;
            let raw=package_manager().read_tag(reference.data_tag).ok()?;
            if raw.len()!=stream.element_count as usize*48 || raw.chunks_exact(48).any(|v| {
                f32::from_le_bytes(v[12..16].try_into().unwrap())!=1.0
                    || v.chunks_exact(4).any(|w|!f32::from_le_bytes(w.try_into().unwrap()).is_finite())
            }) {return None;}
            let p=source.position_transform?;
            if p.scale.iter().any(|&s|s!=p.procedural_scale) {return None;}
            Vec::new()
        } else {
            let palette_ref=source.skinning_buffer.as_ref()?;
            let palette=package_manager().read_tag(palette_ref.data_tag).ok()?;
            if palette.is_empty() || palette.len()%4!=0 || palette.len()!=palette_ref.data_size as usize {return None;}
            palette
        };
        if source.position_transform.is_none() || source.uv_transform.is_none() {
            log::error!("Invalid native producer palette/transform for geometry {}", source.geometry);
            return None;
        }
        if producers.iter().any(|p|matches!(p,DescriptorAbi::HairMeshComputeStorage
            | DescriptorAbi::HairMesh133RowComputeStorage | DescriptorAbi::BodyProceduralComputeStorage
            | DescriptorAbi::Cloth45RowComputeStorage | DescriptorAbi::Cloth46RowComputeStorage
            | DescriptorAbi::BodyMeshAA060BComputeStorage | DescriptorAbi::BodyMeshB152BEComputeStorage)) {
            let reference = source.vertex_streams.iter().find(|s| s.stream_index == 0)?;
            let raw = package_manager().read_tag(reference.data_tag).ok()?;
            if !crate::render::static_mesh::hair_static_inputs_valid(&raw,palette.len()) {
                log::error!("Invalid static packed palette/frame inputs for geometry {}", source.geometry);
                return None;
            }
            for (&producer,constants) in producers.iter().zip(&mut producer_constants) {
                if producer==DescriptorAbi::BodyMeshB152BEComputeStorage {
                    let constants=constants.as_mut()?;
                    constants[16][0]=crate::render::static_mesh::packed_rotation_static_radius(&raw,constants)?;
                }
            }
        }
        if float_source && rigid_float_palette.is_none() {
            // Shared once per model, zero-initialized by WGPU. No 3.75MiB CPU
            // upload per geometry; values are output-inert under proved mode0.
            *rigid_float_palette=Some(device.create_buffer(&wgpu::BufferDescriptor {
                label:Some("bounded viewer Float48 inactive palette image"),size:245_760*16,
                usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false,
            }));
        }
        producers.into_iter().zip(producer_constants).map(|(producer, constants)| crate::render::static_mesh::StaticMesh::new(device, &stream.buffer, &palette, stream.element_count, producer,u32::from(stride),
            if float_source {rigid_float_palette.as_ref()}else{None}, constants.as_deref())).collect()
    } else { Vec::new() };

    let vertex_colors = if vertex_color_required {
        let color = source.color_buffer.as_ref()?;
        if color.stride != 4 || color.vertex_type != 5
            || color.element_count.checked_mul(4) != Some(color.data_size) { return None; }
        let data = package_manager().read_tag(color.data_tag).ok()?;
        let decoded = crate::render::color_vertex::decode(&data, color.element_count)?;
        Some(GpuAuthoredVertexColors {
            data_tag: color.data_tag, count: color.element_count,
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("authored RGBA8 UNORM vertex-color static view"),
                contents: bytemuck::cast_slice(&decoded),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        })
    } else { None };
    Some(GpuAuthoredGeometryInput {
        geometry: source.geometry,
        vertex_streams,
        index_buffer,
        stage_layouts: source.stage_layouts.clone(),
        position_transform: source.position_transform,
        uv_transform: source.uv_transform,
        attachment_pose: source.attachment_pose,
        static_meshes,
        static_mesh_parts,
        vertex_colors,
    })
}

pub(crate) struct GpuModelPreview {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    authored_inputs: Vec<Option<GpuAuthoredGeometryInput>>,
    draws: Vec<ModelDraw>,
    authored_shadow_draws: Vec<ModelDraw>,
    vertices: Vec<ModelVertex>,
    indices: Vec<u32>,
    vertex_abi: VertexAbiDescriptor,
    provenance: ProvenanceStore,
}

impl GpuModelPreview {
    fn authored_index_format(&self, stable_index: usize) -> wgpu::IndexFormat {
        let source = self.draws[stable_index].authored_source.expect("native source");
        self.authored_inputs[source].as_ref().expect("resident native source")
            .index_buffer.as_ref().expect("native index buffer").format
    }

    pub(crate) fn inspection_lines(&self) -> Vec<String> {
        let mut lines = self
            .draws
            .iter()
            .enumerate()
            .map(|(index, draw)| {
                let tfx = draw
                    .packet
                    .technique
                    .as_ref()
                    .map(|technique| {
                        technique
                            .stages
                            .iter()
                            .map(|stage| format!("{}:{:?}", stage.raw_stage_label, stage.tfx_execution.status))
                            .join(",")
                    })
                    .unwrap_or_else(|| "missing".into());
                let source = self
                    .provenance
                    .get(draw.packet.source)
                    .map(|record| format!("{:?}", record.evidence))
                    .unwrap_or_else(|| "missing".into());
                let authored_input = draw
                    .authored_source
                    .and_then(|source| self.authored_inputs.get(source))
                    .and_then(Option::as_ref);
                let authored_layout = draw
                    .packet
                    .raw_render_stage
                    .and_then(|stage| authored_input?.stage_layout(stage));
                let authored_status = match (authored_input, authored_layout) {
                    (Some(input), Some(layout)) => format!(
                        "geometry={} layout={} streams={}",
                        input.geometry,
                        layout.layout_id,
                        input.vertex_streams.len()
                    ),
                    (Some(input), None) => format!("geometry={} layout=missing", input.geometry),
                    (None, _) => "missing".into(),
                };
                let evaluation=draw.native_surface.as_ref().map(|s|format!("source {:?}",s.program))
                    .or_else(||draw.native_decal.as_ref().map(|s|format!("source {:?}",s.program)))
                    .or_else(||draw.native_c827.map(|_|"source C827".to_string()))
                    .unwrap_or_else(||format!("generic; {}",draw.native_rejection.unwrap_or("no authored material contract")));
                format!(
                    "#{index} lod={:?} stage={:?} tech={:?} family={:?} evaluation=[{}] passes={:?} authored=[{}] source={} tfx=[{}] warnings={:?}",
                    draw.packet.raw_lod_category,
                    draw.packet.raw_render_stage,
                    draw.packet.technique_hash,
                    draw.packet.material.family(),
                    evaluation,
                    draw.packet.pass_plan.passes,
                    authored_status,
                    source,
                    tfx,
                    draw.packet.pass_plan.warnings,
                )
            })
            .collect_vec();

        lines.extend(
            self.authored_shadow_draws
                .iter()
                .enumerate()
                .map(|(index, draw)| {
                    let authored = draw.authored_stage.expect("authored shadow metadata");
                    format!(
                        "shadow#{index} stage={:?} part={} layout={} source_indices={}+{} primitive={} lod={:?} variant_shader={} flags=0x{:08X} lod_run={} tech={:?} gpu_indices={:?}",
                        draw.packet.raw_render_stage,
                        authored.part_index,
                        authored.input_layout_id,
                        authored.source_index_start,
                        authored.source_index_count,
                        authored.primitive_type,
                        draw.packet.raw_lod_category,
                        authored.variant_shader_index,
                        authored.flags,
                        authored.lod_run,
                        draw.packet.technique_hash,
                        draw.indices,
                    )
                }),
        );
        lines
    }

    pub(crate) fn create(
        device: &wgpu::Device,
        wireframe: &WireframePreview,
        fallback_color: Option<tiger_pkg::TagHash>,
        runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
    ) -> Option<Self> {
        Self::create_with_geometry(device, wireframe, fallback_color, runtime_inputs, None)
    }

    /// Reevaluate package programs without decoding, AO baking or reuploading mesh inputs.
    pub(crate) fn with_channels(
        &self,
        device: &wgpu::Device,
        wireframe: &WireframePreview,
        runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
    ) -> Option<Self> {
        let updated = Self::create_with_geometry(device, wireframe, None, runtime_inputs, Some(self))?;
        if self.draws.iter().zip(&updated.draws).any(|(before, after)|
            (before.native_surface.is_some() && after.native_surface.is_none())
                || (before.native_decal.is_some() && after.native_decal.is_none())
                || (before.native_c827.is_some() && after.native_c827.is_none())) {
            log::warn!("Channel edit requires a material branch without a supported preview contract");
            return None;
        }
        Some(updated)
    }

    fn create_with_geometry(
        device: &wgpu::Device,
        wireframe: &WireframePreview,
        fallback_color: Option<TagHash>,
        runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
        retained: Option<&Self>,
    ) -> Option<Self> {
        let uvs = wireframe.uvs.as_ref()?;
        if wireframe.vertices.is_empty() || wireframe.indices.len() < 3 {
            return None;
        }

        let vertices = if let Some(retained) = retained {
            retained.vertices.clone()
        } else {
        let generated_normals;
        let normals = if let Some(normals) = wireframe
            .normals
            .as_ref()
            .filter(|normals| normals.len() == wireframe.vertices.len())
        {
            normals.as_slice()
        } else {
            generated_normals = smooth_normals(&wireframe.vertices, &wireframe.indices);
            generated_normals.as_slice()
        };
        let tangents = wireframe
            .tangents
            .as_ref()
            .filter(|tangents| tangents.len() == wireframe.vertices.len());
        let ambient_occlusion =
            vertex_ambient_occlusion(&wireframe.vertices, &wireframe.indices, normals);
        let procedural_positions = wireframe
            .procedural_positions
            .as_ref()
            .filter(|positions| positions.len() == wireframe.vertices.len());
        let procedural_normals = wireframe
            .procedural_normals
            .as_ref()
            .filter(|normals| normals.len() == wireframe.vertices.len());
        wireframe
            .vertices
            .iter()
            .enumerate()
            .map(|(source_index, position)| ModelVertex {
                position: *position,
                normal: normals[source_index],
                uv: uvs.get(source_index).copied().unwrap_or_default(),
                tangent: tangents
                    .and_then(|tangents| tangents.get(source_index).copied())
                    .unwrap_or_default(),
                ambient_occlusion: ambient_occlusion[source_index],
                procedural_position: procedural_positions
                    .and_then(|positions| positions.get(source_index).copied())
                    .unwrap_or(*position),
                procedural_normal: procedural_normals
                    .and_then(|normals| normals.get(source_index).copied())
                    .unwrap_or(normals[source_index]),
            })
            .collect::<Vec<_>>()
        };
        let mut draws = retained.map(|preview| preview.draws.clone())
            .unwrap_or_else(|| model_draws(wireframe, fallback_color));
        for draw in &mut draws {
            let scoped=draw.authored_source.and_then(|index|wireframe.authored_inputs.get(index))
                .map(|source|runtime_inputs.for_geometry(source.geometry));
            let surface=model_surfaces::resolve_source(draw.packet.technique.as_ref(),draw.authored_stage,draw.pipeline,
                scoped.as_ref().unwrap_or(runtime_inputs));
            let decal=model_decals::resolve_source(draw.packet.technique.as_ref(),draw.authored_stage,draw.pipeline,
                scoped.as_ref().unwrap_or(runtime_inputs));
            draw.native_rejection=match draw.packet.raw_render_stage {
                Some(0)=>surface.as_ref().err().copied(),
                Some(2)=>decal.as_ref().err().copied(),
                _=>None,
            };
            draw.native_surface=surface.ok();
            if draw.native_surface.as_ref().is_some_and(|surface|
                surface.vertex_program.auxiliary_vertex_contract().is_some())
                && !draw.authored_source.and_then(|index|wireframe.authored_inputs.get(index))
                    .is_some_and(model_surfaces::packed_static_source_valid) {
                draw.native_surface=None;
                draw.native_rejection=Some("invalid static packed source inputs");
            }
            draw.native_decal=decal.ok();
            if draw.native_surface.as_ref().is_some_and(|surface|
                surface.vertex_program == crate::render::authored_program::DescriptorAbi::VertexColorStorage)
                && !draw.authored_source.and_then(|index|wireframe.authored_inputs.get(index))
                    .zip(draw.authored_stage).is_some_and(|(source,stage)|
                        model_surfaces::vertex_color_source_valid(source,stage)) {
                draw.native_surface=None;
                draw.native_rejection=Some("unsupported vertex-color source/producer inputs");
            }
            draw.native_c827=native_c827_constants(draw.packet.technique.as_ref(),draw.authored_stage,draw.pipeline,
                scoped.as_ref().unwrap_or(runtime_inputs));
        }
        if draws.is_empty() {
            return None;
        }
        let (authored_shadow_indices, mut authored_shadow_draws) = retained
            .map(|preview| (Vec::new(), preview.authored_shadow_draws.clone()))
            .unwrap_or_else(|| model_authored_shadow_draws(wireframe, wireframe.indices.len()));
        let mut provenance = ProvenanceStore::default();
        for draw in &mut draws {
            let source_tag = draw.packet.technique_hash.unwrap_or(TagHash(0));
            draw.packet.source = provenance.insert(ProvenanceRecord {
                evidence: draw
                    .packet
                    .technique_hash
                    .map_or(EvidenceLevel::Probable, |_| EvidenceLevel::Confirmed),
                source_spans: vec![SourceSpan {
                    tag: source_tag,
                    offset: u64::from(draw.indices.start) * 4,
                    size: Some((draw.indices.end - draw.indices.start) * 4),
                }],
                technique: draw.packet.technique_hash,
                shader_stage: None,
                notes: vec![format!("wireframe source {}", wireframe.source)],
            });
        }

        for draw in &mut authored_shadow_draws {
            let source_tag = draw.packet.technique_hash.unwrap_or(TagHash(0));
            let authored = draw.authored_stage.expect("authored shadow metadata");
            draw.packet.source = provenance.insert(ProvenanceRecord {
                evidence: EvidenceLevel::Confirmed,
                source_spans: vec![SourceSpan {
                    tag: source_tag,
                    offset: u64::from(authored.source_index_start),
                    size: Some(authored.source_index_count),
                }],
                technique: draw.packet.technique_hash,
                shader_stage: Some("ShadowGenerate"),
                notes: vec![format!(
                    "authored ShadowGenerate part={} layout={} variant_shader={} primitive={} flags=0x{:08X} lod_run={}",
                    authored.part_index,
                    authored.input_layout_id,
                    authored.variant_shader_index,
                    authored.primitive_type,
                    authored.flags,
                    authored.lod_run,
                )],
            });
        }

        let vertex_buffer = retained.map(|preview| preview.vertex_buffer.clone()).unwrap_or_else(|| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quicktag_model_preview_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        let mut gpu_indices = wireframe.indices.clone();
        gpu_indices.extend_from_slice(&authored_shadow_indices);
        let index_buffer = retained.map(|preview| preview.index_buffer.clone()).unwrap_or_else(|| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quicktag_model_preview_indices"),
            contents: bytemuck::cast_slice(&gpu_indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        let mut required_stages = vec![Vec::<u8>::new(); wireframe.authored_inputs.len()];
        for draw in draws
            .iter()
            .filter(|draw| {
                draw.native_c827.is_some() || draw.native_surface.is_some() || draw.native_decal.is_some() || (draw.authored_native_vertex_supported
                    && draw
                        .packet
                        .pass_plan
                        .passes
                        .contains(&RenderPassKind::DepthOnly))
            })
            .chain(
                authored_shadow_draws
                    .iter()
                    .filter(|draw| draw.authored_native_vertex_supported),
            )
        {
            let (Some(source_index), Some(authored)) = (draw.authored_source, draw.authored_stage)
            else {
                continue;
            };
            if let Some(stages) = required_stages.get_mut(source_index)
                && !stages.contains(&authored.raw_stage)
            {
                stages.push(authored.raw_stage);
            }
        }

        let mut vertex_buffer_cache = HashMap::<TagHash, wgpu::Buffer>::new();
        let mut index_buffer_cache = HashMap::<TagHash, wgpu::Buffer>::new();
        if let Some(retained) = retained {
            for source in retained.authored_inputs.iter().flatten() {
                for stream in &source.vertex_streams {
                    vertex_buffer_cache.insert(stream.data_tag, stream.buffer.clone());
                }
                if let Some(index) = &source.index_buffer {
                    index_buffer_cache.insert(index.data_tag, index.buffer.clone());
                }
            }
        }
        let mut rigid_float_palette=None;
        let authored_inputs = wireframe
            .authored_inputs
            .iter()
            .enumerate()
            .map(|(source_index, source)| {
                create_gpu_authored_geometry_input(
                    device,
                    source,
                    required_stages
                        .get(source_index)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                    &draws.iter().filter(|draw| (draw.native_c827.is_some() || draw.native_surface.is_some() || draw.native_decal.is_some()) && draw.authored_source == Some(source_index))
                        .filter(|draw|!draw.native_surface.as_ref().is_some_and(|s|s.vertex_program==crate::render::authored_program::DescriptorAbi::RigidVertexDirectIa))
                        .filter_map(|draw|draw.authored_stage).collect_vec(),
                    draws.iter().any(|draw| draw.authored_source == Some(source_index)
                        && draw.native_surface.as_ref().is_some_and(|s| s.vertex_program
                            == crate::render::authored_program::DescriptorAbi::VertexColorStorage)),
                    &mut vertex_buffer_cache,
                    &mut index_buffer_cache,
                    &mut rigid_float_palette,
                    &runtime_inputs.for_geometry(source.geometry),
                )
            })
            .collect::<Vec<_>>();
        if let Some(draw) = draws.iter().filter(|d| d.native_c827.is_some() || d.native_surface.is_some() || d.native_decal.is_some())
            .filter(|d| !d.native_surface.as_ref().is_some_and(|surface|surface.vertex_program == crate::render::authored_program::DescriptorAbi::RigidVertexDirectIa)).find(|draw| {
            let mesh=draw.authored_source.and_then(|i| authored_inputs.get(i)).and_then(Option::as_ref).and_then(|s| s.static_mesh(draw.authored_stage?));
            mesh.is_none() || draw.native_surface.as_ref().is_some_and(|surface|
                if let Some(producer)=surface.vertex_program.static_vertex_producer() {
                    mesh.unwrap().1.producer != producer
                } else { matches!(mesh.unwrap().1.producer,
                    crate::render::authored_program::DescriptorAbi::HairMeshComputeStorage
                    | crate::render::authored_program::DescriptorAbi::HairMesh133RowComputeStorage) })
        }) {
            log::error!("Native preview input rejected: technique={:?} source={:?} stage={:?}",
                draw.packet.technique_hash, draw.authored_source, draw.authored_stage);
            return None;
        }

        Some(Self {
            vertex_buffer,
            index_buffer,
            authored_inputs,
            draws,
            authored_shadow_draws,
            vertices,
            indices: wireframe.indices.clone(),
            vertex_abi: VertexAbiDescriptor::from_wireframe(wireframe),
            provenance,
        })
    }
}

struct ModelDrawSource<'a> {
    indices: Range<u32>,
    raw_lod_category: Option<u8>,
    render_stage: Option<u8>,
    technique: Option<TagHash>,
    procedural_scale: f32,
    authored_source: Option<usize>,
    texture: Option<TagHash>,
    textures: &'a WireframeMaterialTextures,
    center: [f32; 3],
    authored_stage: Option<AuthoredStageMetadata>,
}

fn shader_payload_contains_ascii(shader: TagHash, needle: &[u8]) -> bool {
    let Some(entry) = package_manager().get_entry(shader) else {
        return false;
    };
    let Ok(data) = package_manager().read_tag(TagHash(entry.reference)) else {
        return false;
    };
    data.windows(needle.len()).any(|window| window == needle)
}

fn authored_native_vertex_supported(technique: Option<&TechniqueDescriptor>) -> bool {
    let Some(vertex_shader) = technique
        .into_iter()
        .flat_map(|technique| technique.stages.iter())
        .find(|stage| stage.stage == ShaderStage::Vertex)
        .and_then(|stage| stage.shader)
    else {
        return false;
    };

    // Marathon's common depth-only/ShadowGenerate VS family can ignore the
    // IA POSITION entirely. It indexes a runtime-generated packed-position
    // buffer (t2) with SV_VertexID instead. Quicktag does not reconstruct that
    // resource yet, so binding raw POSITION to our compatibility VS produces
    // the wrong caster geometry. Reconstructed ModelVertex positions already
    // contain the CPU dequantization/assembly transform and are the faithful
    // fallback until the runtime resource ABI is implemented.
    !shader_payload_contains_ascii(vertex_shader, b"SV_VertexID")
}

fn native_c827_constants(technique: Option<&TechniqueDescriptor>, authored: Option<AuthoredStageMetadata>, pipeline: ModelPipelineKey,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs) -> Option<[[f32;4];9]> {
    use crate::render::authored_program::{DescriptorAbi, resolve_package_program};
    let authored = authored?;
    if authored.input_layout_id != 7 || authored.primitive_type != 5 || authored.raw_stage != 2
        || pipeline.blend != 76 || pipeline.depth_stencil != 15 || pipeline.depth_bias != 1 || pipeline.rasterizer != 2 { return None; }
    let technique = technique?;
    let vs = technique.stages.iter().find(|s| s.stage == ShaderStage::Vertex)?;
    let ps = technique.stages.iter().find(|s| s.stage == ShaderStage::Pixel)?;
    if resolve_package_program(vs.shader?, ShaderStage::Vertex).ok()??.descriptor_abi != DescriptorAbi::SharedLayout7VertexScalarStorage
        || resolve_package_program(ps.shader?, ShaderStage::Pixel).ok()??.descriptor_abi != DescriptorAbi::C827Pixel { return None; }
    let state = ps.runtime_state(runtime_inputs);
    if ps.constant_buffer_slot != Some(0) || !state.unresolved_dependencies.is_empty() { return None; }
    state.constant_registers.try_into().ok()
}

fn model_draw_from_source(
    source: ModelDrawSource<'_>,
    technique_descriptors: &mut HashMap<TagHash, Option<TechniqueDescriptor>>,
) -> ModelDraw {
    let material_ir = MaterialIR::classify(source.textures);
    let material_inputs = material_ir.inputs().clone();
    let color = material_inputs
        .forward_coating
        .map(|coating| coating.detail)
        .or(material_inputs.color)
        .or_else(|| {
            (source.render_stage == Some(crate::render::adapter::GoliathAdapter::TRANSPARENT_STAGE))
                // Transparent-stage direct resources are collected in
                // material-role order. The first auxiliary texture is the
                // authored model-specific surface map; later entries are
                // shared shader utility resources.
                .then(|| material_inputs.aux.first().copied())
                .flatten()
        })
        .or(source.texture);
    let technique = source.technique.and_then(|tag| {
        technique_descriptors
            .entry(tag)
            .or_insert_with(|| TechniqueDescriptor::load(tag))
            .clone()
    });
    let render_state = technique
        .as_ref()
        .map(|technique| technique.render_state)
        .unwrap_or_else(|| {
            source
                .technique
                .map(render_state_for_technique)
                .unwrap_or_default()
        });
    let pass_plan = DrawPassPlan::derive(source.render_stage, render_state, &material_ir);
    let mut pipeline = ModelPipelineKey::select_for_draw(
        render_state,
        material_ir.family(),
        &pass_plan,
        source.technique,
    );
    let glass = material_inputs.transmission.filter(|transmission| transmission.absorption.is_some()
        && source.render_stage == Some(crate::render::adapter::GoliathAdapter::TRANSPARENT_STAGE));
    if glass.is_some() {
        pipeline.blend = ABSORPTION_BLEND;
    }
    let authored_native_vertex_supported =
        source.authored_stage.is_some() && authored_native_vertex_supported(technique.as_ref());

    ModelDraw {
        indices: source.indices.clone(),
        packet: TigerDrawPacket {
            indices: source.indices,
            raw_lod_category: source.raw_lod_category,
            raw_render_stage: source.render_stage,
            technique_hash: source.technique,
            technique,
            material: material_ir,
            pass_plan,
            source: ProvenanceId(u32::MAX),
        },
        material: (color.is_some() || material_inputs.normal.is_some() || material_inputs.emissive.is_some()).then_some(MaterialTextureKey {
            color,
            normal: material_inputs.normal,
            emissive: material_inputs.emissive,
            color_tint: material_inputs.color_tint,
            emissive_strength: material_inputs.emissive_strength,
        }),
        solid_color: material_inputs.solid_color,
        solid_surface: material_inputs.solid_surface,
        iridescence_id: material_inputs.iridescence_id,
        transmission: glass.or((source.render_stage
            == Some(crate::render::adapter::GoliathAdapter::DISTORTION_STAGE))
        .then_some(material_inputs.transmission)
        .flatten()),
        forward_coating: material_inputs.forward_coating,
        character_surface: material_inputs.character_surface,
        runner_layered_surface: material_inputs.runner_layered_surface,
        runner_occlusion: material_inputs.runner_occlusion,
        alpha_mask: material_inputs.alpha_mask,
        shared_atlas_detail: material_inputs.shared_atlas_detail,
        control: material_inputs.control,
        roughness_channel: material_inputs.roughness_channel,
        mask_palette: material_inputs.mask_palette,
        gear_dye: material_inputs.gear_dye,
        gear_dye_default: material_inputs.gear_dye_default,
        gear_dye_palette: material_inputs.gear_dye_palette,
        gear_worn_dye_palette: material_inputs.gear_worn_dye_palette,
        gear_dye_detail_palette: material_inputs.gear_dye_detail_palette,
        mod_wear: material_inputs.mod_wear,
        surface_condition: material_inputs.surface_condition,
        gear_pattern: material_inputs.gear_pattern,
        investment_decal: material_inputs.investment_decal,
        authored_shared_atlas: material_inputs.authored_shared_atlas,
        sampler: material_inputs.sampler,
        sticker_proxy: source.technique.is_some_and(is_sticker_proxy_technique),
        pipeline,
        center: source.center,
        procedural_scale: source.procedural_scale,
        authored_source: source.authored_source,
        authored_stage: source.authored_stage,
        authored_native_vertex_supported,
        native_c827: None,
        native_surface: None,
        native_decal: None,
        native_rejection: None,
    }
}

fn model_draws(
    wireframe: &WireframePreview,
    fallback_color: Option<tiger_pkg::TagHash>,
) -> Vec<ModelDraw> {
    let index_len = wireframe.indices.len();
    let mut ranges = wireframe.material_ranges.iter().collect_vec();
    ranges.sort_by_key(|range| range.index_start);
    let mut draws = Vec::new();
    let mut technique_descriptors = HashMap::<TagHash, Option<TechniqueDescriptor>>::new();

    for range in ranges {
        let start = range.index_start.min(index_len);
        let end = range
            .index_start
            .saturating_add(range.index_count)
            .min(index_len);
        if start < end {
            let authored_stage = range.authored_draw.and_then(|authored| {
                let raw_stage = range.render_stage?;
                let source_index = range.authored_source?;
                let input_layout_id = wireframe
                    .authored_inputs
                    .get(source_index)?
                    .stage_layouts
                    .iter()
                    .find(|layout| layout.raw_stage == raw_stage)?
                    .layout_id;
                Some(AuthoredStageMetadata {
                    raw_stage,
                    input_layout_id,
                    part_index: authored.part_index,
                    source_index_start: authored.source_index_start,
                    source_index_count: authored.source_index_count,
                    primitive_type: authored.primitive_type,
                    variant_shader_index: authored.variant_shader_index,
                    flags: authored.flags,
                    lod_run: authored.lod_run,
                })
            });
            draws.push(model_draw_from_source(
                ModelDrawSource {
                    indices: start as u32..end as u32,
                    raw_lod_category: range.raw_lod_category,
                    render_stage: range.render_stage,
                    technique: range.technique,
                    procedural_scale: range.procedural_scale,
                    authored_source: range.authored_source,
                    texture: range.texture,
                    textures: &range.textures,
                    center: draw_range_center(wireframe, start, end),
                    authored_stage,
                },
                &mut technique_descriptors,
            ));
        }
    }

    if draws.is_empty() && wireframe.material_ranges.is_empty() {
        let material_ir = MaterialIR::classify(&WireframeMaterialTextures::default());
        draws.push(ModelDraw {
            indices: 0..index_len as u32,
            packet: TigerDrawPacket {
                indices: 0..index_len as u32,
                raw_lod_category: None,
                raw_render_stage: None,
                technique_hash: None,
                technique: None,
                pass_plan: DrawPassPlan::derive(
                    None,
                    TechniqueRenderState::default(),
                    &material_ir,
                ),
                material: material_ir,
                source: ProvenanceId(u32::MAX),
            },
            material: fallback_color.map(default_material),
            solid_color: None,
            solid_surface: None,
            iridescence_id: None,
            transmission: None,
            forward_coating: None,
            character_surface: None,
            runner_layered_surface: None,
            runner_occlusion: None,
            alpha_mask: None,
            shared_atlas_detail: None,
            control: None,
            roughness_channel: 0,
            mask_palette: None,
            gear_dye: None,
            gear_dye_default: None,
            gear_dye_palette: None,
            gear_worn_dye_palette: None,
            gear_dye_detail_palette: None,
            mod_wear: None,
            surface_condition: None,
            gear_pattern: None,
            investment_decal: None,
            authored_shared_atlas: false,
            sampler: None,
            sticker_proxy: false,
            pipeline: ModelPipelineKey::default(),
            center: draw_range_center(wireframe, 0, index_len),
            procedural_scale: draw_range_procedural_scale(wireframe, 0, index_len),
            authored_source: None,
            authored_stage: None,
            authored_native_vertex_supported: false,
            native_c827: None,
            native_surface: None,
            native_decal: None,
            native_rejection: None,
        });
    }

    draws.sort_by_key(|draw| blend_enabled(draw.pipeline.blend));
    draws
}

fn model_authored_shadow_draws(
    wireframe: &WireframePreview,
    gpu_index_base: usize,
) -> (Vec<u32>, Vec<ModelDraw>) {
    let mut shadow_indices = Vec::new();
    let mut draws = Vec::new();
    let mut technique_descriptors = HashMap::<TagHash, Option<TechniqueDescriptor>>::new();

    for range in wireframe.authored_shadow_ranges.iter().filter(|range| {
        range.render_stage == crate::render::adapter::GoliathAdapter::AUTHORED_SHADOW_STAGE
    }) {
        if range.indices.is_empty() {
            continue;
        }
        let local_start = shadow_indices.len();
        shadow_indices.extend_from_slice(&range.indices);
        let local_end = shadow_indices.len();
        let indices = (gpu_index_base + local_start) as u32..(gpu_index_base + local_end) as u32;
        draws.push(model_draw_from_source(
            ModelDrawSource {
                indices,
                raw_lod_category: Some(range.raw_lod_category),
                render_stage: Some(range.render_stage),
                technique: range.technique,
                procedural_scale: range.procedural_scale,
                authored_source: range.authored_source,
                texture: range.texture,
                textures: &range.textures,
                center: draw_indices_center(&wireframe.vertices, &range.indices),
                authored_stage: Some(AuthoredStageMetadata {
                    raw_stage: range.render_stage,
                    input_layout_id: range.input_layout_id,
                    part_index: range.part_index,
                    source_index_start: range.source_index_start,
                    source_index_count: range.source_index_count,
                    primitive_type: range.primitive_type,
                    variant_shader_index: range.variant_shader_index,
                    flags: range.flags,
                    lod_run: range.lod_run,
                }),
            },
            &mut technique_descriptors,
        ));
    }

    (shadow_indices, draws)
}

fn draw_range_procedural_scale(wireframe: &WireframePreview, start: usize, end: usize) -> f32 {
    let Some(procedural) = wireframe.procedural_positions.as_ref() else {
        return 1.0;
    };
    let indices =
        &wireframe.indices[start.min(wireframe.indices.len())..end.min(wireframe.indices.len())];
    let Some(&anchor_index) = indices.first() else {
        return 1.0;
    };
    let anchor_index = anchor_index as usize;
    let (Some(anchor), Some(procedural_anchor)) = (
        wireframe.vertices.get(anchor_index),
        procedural.get(anchor_index),
    ) else {
        return 1.0;
    };
    let mut farthest = (0.0_f32, 0.0_f32);
    for &index in indices.iter().step_by(17) {
        let index = index as usize;
        let (Some(position), Some(procedural_position)) =
            (wireframe.vertices.get(index), procedural.get(index))
        else {
            continue;
        };
        let procedural_distance = procedural_position
            .iter()
            .zip(procedural_anchor)
            .map(|(value, anchor)| (value - anchor).powi(2))
            .sum::<f32>()
            .sqrt();
        if procedural_distance <= farthest.0 {
            continue;
        }
        let model_distance = position
            .iter()
            .zip(anchor)
            .map(|(value, anchor)| (value - anchor).powi(2))
            .sum::<f32>()
            .sqrt();
        farthest = (procedural_distance, model_distance);
    }
    if farthest.0 > 0.00001 {
        farthest.1 / farthest.0
    } else {
        1.0
    }
}

fn draw_range_center(wireframe: &WireframePreview, start: usize, end: usize) -> [f32; 3] {
    draw_indices_center(
        &wireframe.vertices,
        &wireframe.indices[start.min(wireframe.indices.len())..end.min(wireframe.indices.len())],
    )
}

fn draw_indices_center(vertices: &[[f32; 3]], indices: &[u32]) -> [f32; 3] {
    let mut sum = [0.0_f64; 3];
    let mut count = 0usize;
    for index in indices.iter().step_by(3) {
        let Some(position) = vertices.get(*index as usize) else {
            continue;
        };
        sum[0] += position[0] as f64;
        sum[1] += position[1] as f64;
        sum[2] += position[2] as f64;
        count += 1;
    }
    if count == 0 {
        return [0.0; 3];
    }
    [
        (sum[0] / count as f64) as f32,
        (sum[1] / count as f64) as f32,
        (sum[2] / count as f64) as f32,
    ]
}

fn default_material(color: tiger_pkg::TagHash) -> MaterialTextureKey {
    MaterialTextureKey {
        color: Some(color),
        normal: None,
        emissive: None,
        color_tint: [255; 4],
        emissive_strength: 0,
    }
}

fn smooth_normals(vertices: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0_f32; 3]; vertices.len()];
    for triangle in indices.chunks_exact(3) {
        let Some(a) = vertices.get(triangle[0] as usize) else {
            continue;
        };
        let Some(b) = vertices.get(triangle[1] as usize) else {
            continue;
        };
        let Some(c) = vertices.get(triangle[2] as usize) else {
            continue;
        };
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let face = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for index in triangle {
            if let Some(normal) = normals.get_mut(*index as usize) {
                normal[0] += face[0];
                normal[1] += face[1];
                normal[2] += face[2];
            }
        }
    }

    for normal in &mut normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > f32::EPSILON {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            *normal = [0.0, 0.0, 1.0];
        }
    }
    normals
}

/// Approximates Alkahest's baked static vertex-AO buffer for entity assets,
/// which do not carry the world-level AO allocation. Vertices on hard folds
/// receive less ambient light; flat authored surfaces remain fully exposed.
fn vertex_ambient_occlusion(
    vertices: &[[f32; 3]],
    indices: &[u32],
    normals: &[[f32; 3]],
) -> Vec<f32> {
    let mut alignment = vec![0.0_f32; vertices.len()];
    let mut count = vec![0_u32; vertices.len()];
    for triangle in indices.chunks_exact(3) {
        let (Some(a), Some(b), Some(c)) = (
            vertices.get(triangle[0] as usize),
            vertices.get(triangle[1] as usize),
            vertices.get(triangle[2] as usize),
        ) else {
            continue;
        };
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut face = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let length = (face[0] * face[0] + face[1] * face[1] + face[2] * face[2]).sqrt();
        if length <= f32::EPSILON {
            continue;
        }
        face.iter_mut().for_each(|value| *value /= length);
        for index in triangle.iter().map(|index| *index as usize) {
            let Some(normal) = normals.get(index) else {
                continue;
            };
            alignment[index] +=
                (normal[0] * face[0] + normal[1] * face[1] + normal[2] * face[2]).abs();
            count[index] += 1;
        }
    }
    alignment
        .into_iter()
        .zip(count)
        .map(|(sum, count)| {
            if count == 0 {
                1.0
            } else {
                (sum / count as f32).clamp(0.35, 1.0)
            }
        })
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    center: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    uv_transform: [f32; 4],
    light_direction: [f32; 4],
    light_parameters: [f32; 4],
    /// View-space light origin relative to the model camera frame.
    light_position: [f32; 4],
    postprocess0: [f32; 4],
    postprocess1: [f32; 4],
    postprocess2: [f32; 4],
    postprocess3: [f32; 4],
    postprocess4: [f32; 4],
    postprocess5: [f32; 4],
    fidelity: [f32; 4],
    shadow_parameters: [f32; 4],
    /// Appended last: shaders that do not light may declare the struct
    /// without these three rows.
    light_color: [f32; 4],
    ambient_sky: [f32; 4],
    ambient_ground: [f32; 4],
}

/// The viewer owns its orthographic camera. These are native VS matrix columns,
/// not captured game View scope values. Rebuilt on every camera/frame update.
fn native_view_image(scene: &SceneUniform) -> [[f32;4];27] {
    let [radius, yaw, pitch, zoom] = scene.params0;
    let (sy,cy) = yaw.sin_cos();
    let (sp,cp) = pitch.sin_cos();
    let rotate = |p: [f32;3]| {
        let x = p[0]*cy + p[1]*sy;
        let z = -p[0]*sy + p[1]*cy;
        [x, p[2]*cp-z*sp, p[2]*sp+z*cp]
    };
    let scale = 0.84 * zoom / radius.max(0.0001);
    let project = |p: [f32;3]| {
        let r = rotate(p);
        [-r[0]*scale*scene.params1[0], r[1]*scale, -r[2]*0.21/radius.max(0.0001), 0.0]
    };
    let mut view = [[0.0;4];27];
    for axis in 0..3 { let mut p=[0.0;3]; p[axis]=1.0; view[axis]=project(p); }
    let center=project([scene.center[0],scene.center[1],scene.center[2]]);
    view[19]=[-center[0]+scene.params1[1], -center[1]+scene.params1[2], 0.5-center[2], 1.0];
    // C827 View20..22 are relative-world offsets, not another clip matrix.
    // The viewer uses world coordinates directly, so those offsets stay zero.
    view[7]=[scene.center[0],scene.center[1],scene.center[2],1.0];
    view
}

fn c827_static_skinning(source: &GpuAuthoredGeometryInput, scene: &SceneUniform) -> [[f32;4];31] {
    let p=source.position_transform.expect("native static position transform");
    let uv=source.uv_transform.expect("native static UV transform");
    let mut skin=[[0.0;4];31];
    for start in [0,8] { for axis in 0..4 { skin[start+axis][axis]=1.0; } }
    skin[6]=[uv.scale[0]*scene.uv_transform[0], uv.scale[1]*scene.uv_transform[1],
        uv.offset[0]*scene.uv_transform[0]+scene.uv_transform[2], uv.offset[1]*scene.uv_transform[1]+scene.uv_transform[3]];
    // Exact paired C827 VS reads offset.xyz and scale.w. Geometry metadata's
    // separate scale/offset fields must be packed in that shader order.
    skin[12]=[p.offset[0],p.offset[1],p.offset[2],p.procedural_scale];
    skin[13]=skin[12];
    skin[14]=[f32::from_bits(2),f32::from_bits(2),0.0,0.0];
    // Viewer-owned object-space texture frequency. Native dynamic writer is
    // unobserved; this fixed static-preview input is explicit in the ledger.
    skin[5][3]=1.0;
    skin
}

fn native_surface_object_image(
    vertex: crate::render::authored_program::DescriptorAbi,
    source: &GpuAuthoredGeometryInput, scene: &SceneUniform,
) -> [[f32;4];31] {
    let mut object=c827_static_skinning(source,scene);
    if vertex == crate::render::authored_program::DescriptorAbi::VertexColorStorage {
        // Explicit static preview policy: C107's UMin selects actual authored
        // records and clamps to this buffer's final record, including sentinel.
        object[4][3] = f32::from_bits(source.vertex_colors.as_ref()
            .expect("authored vertex-color image").count - 1);
    }
    if vertex == crate::render::authored_program::DescriptorAbi::RigidVertexDirectIa {
        // Raw float positions are retained. IA13 VS applies the geometry
        // placement through cb1[0..3] exactly once, not via a compute producer.
        let p=source.position_transform.expect("direct-IA placement");
        for start in [0,8] {
            for axis in 0..3 {
                object[start+axis]=[0.0;4];
                object[start+axis][axis]=p.scale[axis];
            }
            object[start+3]=[p.offset[0],p.offset[1],p.offset[2],1.0];
        }
    }
    object
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelEnvironment {
    pub lighting_model: LightingModel,
    pub tfx_time_seconds: f32,
    pub tfx_paused: bool,
    pub tfx_speed: f32,
    pub time_of_day: f32,
    pub sun_intensity: f32,
    pub fog_density: f32,
    pub bloom_strength: f32,
    pub exposure: f32,
    pub auto_exposure: bool,
    pub vertex_ao_strength: f32,
    pub cubemap: Option<TagHash>,
    pub distortion: f32,
    pub hiz_culling: bool,
    pub water_strength: f32,
    pub water_level: f32,
    pub decal_strength: f32,
    pub road_decal_strength: f32,
    pub decal: Option<TagHash>,
    pub fxaa: bool,
    pub ssao_strength: f32,
    pub tone_mapping: bool,
    pub ambient_intensity: f32,
    /// Linear colour of the key light.
    pub light_color: [f32; 3],
    /// Linear ambient colours for surfaces facing up and down. Tiger's global
    /// light weights them by the squared half-range of the world normal's Z.
    pub ambient_sky_color: [f32; 3],
    pub ambient_ground_color: [f32; 3],
    pub specular_ibl_intensity: f32,
    /// Beam target in rig units when scaling is enabled, otherwise model units.
    pub light_target: [f32; 3],
    /// Unit world-space offset locating the light source on its orbit sphere.
    pub light_orbit_position: [f32; 3],
    /// Orbit center in rig units when scaling is enabled, otherwise model units.
    pub light_orbit_center: [f32; 3],
    pub light_orbit_radius: f32,
    /// Maximum distance at which the spotlight contributes direct light.
    pub light_range: f32,
    /// Spotlight half-angle in degrees, measured from the beam axis.
    pub light_cone_angle: f32,
    pub light_size: f32,
    /// Express rig distances relative to the base model's largest extent.
    pub light_scale_with_model: bool,
    /// Base weapon with default mods, captured independently of equipped mods.
    pub light_model_frame: Option<ModelCameraFrame>,
    pub light_gizmo: bool,
    pub shadow_strength: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gamma: f32,
    /// Semantic/debug channel selector. 0 final; 1..8 legacy lighting/MRT;
    /// 9..20 package material layers; 21..35 forward-coating channels.
    pub diagnostic_pass: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum LightingModel {
    TigerGgxCompatibility,
    #[default]
    TigerGgxApproximation,
    DebugLambert,
    SurfaceNormals,
    SurfaceProperties,
    SurfaceEmissive,
    SurfaceFlags,
    SurfaceAlbedo,
}

impl Default for ModelEnvironment {
    fn default() -> Self {
        Self {
            lighting_model: LightingModel::TigerGgxApproximation,
            tfx_time_seconds: 0.0,
            tfx_paused: false,
            tfx_speed: 1.0,
            time_of_day: 0.35,
            sun_intensity: 5.0,
            fog_density: 0.0,
            bloom_strength: 0.0,
            exposure: 1.0,
            auto_exposure: false,
            vertex_ao_strength: 0.5,
            cubemap: None,
            distortion: 0.0,
            hiz_culling: false,
            water_strength: 0.0,
            water_level: 0.68,
            decal_strength: 0.0,
            road_decal_strength: 0.0,
            decal: None,
            fxaa: true,
            ssao_strength: 10.0,
            tone_mapping: true,
            ambient_intensity: 0.3,
            light_color: [1.0; 3],
            ambient_sky_color: [1.0; 3],
            ambient_ground_color: [1.0; 3],
            specular_ibl_intensity: 0.2,
            // The default source faces the world origin, preserving the
            // historical key-light orientation while using finite lighting.
            light_target: [-0.183, 0.017, -0.483],
            light_orbit_position: [0.2562, 0.3389, 0.9053],
            light_orbit_center: [0.174, -0.045, -0.117],
            light_orbit_radius: 1.0,
            light_range: 4.0,
            light_cone_angle: 70.0,
            light_size: 5.0,
            light_scale_with_model: true,
            light_model_frame: None,
            light_gizmo: false,
            shadow_strength: 1.0,
            brightness: 1.4,
            contrast: 1.01,
            saturation: 1.2,
            gamma: 0.85,
            diagnostic_pass: 0,
        }
    }
}

fn first_person_key_light(
    time_of_day: f32,
    cast_direction: [f32; 3],
    shadow_strength: f32,
) -> ([f32; 4], f32) {
    let sun_angle = time_of_day.rem_euclid(1.0) * std::f32::consts::TAU;
    let sun_height = (sun_angle - std::f32::consts::FRAC_PI_2).sin();
    let length = cast_direction
        .into_iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let direction = if length > 0.0001 {
        cast_direction.map(|value| -value / length)
    } else {
        [0.42, 0.89, 0.16]
    };
    (
        [
            direction[0],
            direction[1],
            direction[2],
            shadow_strength.clamp(0.0, 1.0),
        ],
        sun_height,
    )
}

pub(crate) fn light_source_position(environment: &ModelEnvironment) -> [f32; 3] {
    let mut orbit_direction = environment.light_orbit_position;
    let orbit_length = orbit_direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if orbit_length <= 0.0001 {
        orbit_direction = ModelEnvironment::default().light_orbit_position;
    } else {
        orbit_direction
            .iter_mut()
            .for_each(|value| *value /= orbit_length);
    }
    std::array::from_fn(|axis| {
        environment.light_orbit_center[axis]
            + orbit_direction[axis] * environment.light_orbit_radius.max(0.0)
    })
}

pub(crate) fn light_cast_direction(environment: &ModelEnvironment) -> [f32; 3] {
    let light_position = light_source_position(environment);
    std::array::from_fn(|axis| environment.light_target[axis] - light_position[axis])
}

fn light_cone_cosines(half_angle_degrees: f32) -> (f32, f32) {
    let outer_angle = half_angle_degrees.clamp(1.0, 89.0).to_radians();
    let inner_angle = (outer_angle * 0.72).max(0.5_f32.to_radians());
    (outer_angle.cos(), inner_angle.cos())
}

pub(crate) fn model_direction_to_view(direction: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let xz = direction[0] * cy + direction[1] * sy;
    let zz = -direction[0] * sy + direction[1] * cy;
    [xz, direction[2] * cp - zz * sp, direction[2] * sp + zz * cp]
}

pub(crate) fn fixed_light_direction_to_view(direction: [f32; 3]) -> [f32; 3] {
    model_direction_to_view(direction, 0.0, 0.0)
}

pub(crate) fn fixed_view_direction_to_light(direction: [f32; 3]) -> [f32; 3] {
    view_direction_to_model(direction, 0.0, 0.0)
}

pub(crate) fn view_direction_to_model(direction: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let z_up_y = direction[1] * cp + direction[2] * sp;
    let yawed_z = -direction[1] * sp + direction[2] * cp;
    [
        direction[0] * cy - yawed_z * sy,
        direction[0] * sy + yawed_z * cy,
        z_up_y,
    ]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    tint: [f32; 4],
    params: [f32; 4],
    blend: [f32; 4],
    solid_surface: [f32; 4],
    palette_base: [f32; 4],
    palette_delta: [f32; 4],
    palette_tertiary: [f32; 4],
    roughness_remap: [f32; 4],
    metal_remap: [f32; 4],
    wear_params: [f32; 4],
    wear_scratches_projection: [f32; 4],
    wear_grime_projection: [f32; 4],
    wear_damage_projection: [f32; 4],
    wear_scratches_remap_base: [f32; 4],
    wear_scratches_remap_scale: [f32; 4],
    wear_surface_params: [f32; 4],
    gear_palette_default: [f32; 4],
    gear_palette_colors: [[f32; 4]; 6],
    gear_worn_dye_colors: [[f32; 4]; 6],
    gear_dye_detail_colors: [[f32; 4]; 6],
    gear_palette_roughness: [[f32; 4]; 6],
    gear_palette_metal: [[f32; 4]; 6],
    decal_params: [f32; 4],
    decal_detail_transform: [f32; 4],
    decal_detail_base: [f32; 4],
    decal_detail_scale: [f32; 4],
    decal_mask_params: [f32; 4],
    decal_mask_remap: [f32; 4],
    decal_selector_colors: [[f32; 4]; 5],
    sampler_params: [f32; 4],
    pattern_projection: [f32; 4],
    pattern_params: [f32; 4],
    pattern_stripe: [f32; 4],
    pattern_contour: [f32; 4],
    pattern_contour_remap: [f32; 4],
    pattern_colors: [[f32; 4]; 2],
    transmission_colors: [[f32; 4]; 2],
    transmission_surfaces: [[f32; 4]; 2],
    transmission_params: [f32; 4],
    coating_colors: [[f32; 4]; 2],
    coating_projection: [f32; 4],
    coating_params0: [f32; 4],
    coating_params1: [f32; 4],
    coating_environment_params: [f32; 4],
    coating_environment_extra: [f32; 4],
    coating_specular_colors: [[f32; 4]; 2],
    coating_specular_params: [[f32; 4]; 2],
    character_detail_transform: [f32; 4],
    character_detail_base: [f32; 4],
    character_detail_scale: [f32; 4],
    character_params: [f32; 4],
    character_extra: [[f32; 4]; 2],
    character_palette: [[f32; 4]; 2],
    character_procedural: [[f32; 4]; 11],
    runner_layered_params: [f32; 4],
    runner_layered_constants: [[f32; 4]; 24],
    runner_color_constants: [[f32; 4]; 7],
    alpha_mask_params: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ModelSamplerDesc {
    filter: u32,
    address_u: u32,
    address_v: u32,
    address_w: u32,
    mip_lod_bias: f32,
    max_anisotropy: u32,
    border_color: [f32; 4],
    min_lod: f32,
    max_lod: f32,
}

struct LoadedMaterial {
    key: Option<MaterialTextureKey>,
    solid_color: Option<[f32; 4]>,
    solid_surface: Option<[f32; 2]>,
    iridescence_id: Option<f32>,
    transmission: Option<TransmissionMaterial>,
    forward_coating: Option<ForwardCoatingMaterial>,
    character_surface: Option<CharacterSurfaceMaterial>,
    runner_layered_surface: Option<RunnerLayeredSurfaceMaterial>,
    runner_occlusion: Option<RunnerOcclusionMaterial>,
    alpha_mask: Option<AlphaMaskMaterial>,
    shared_atlas_detail: Option<SharedAtlasDetailMaterial>,
    blend: u8,
    control_tag: Option<TagHash>,
    roughness_channel: u8,
    mask_palette: Option<[[f32; 4]; 2]>,
    dye_palette: Option<[[f32; 4]; 3]>,
    gear_dye: Option<GearDyeMaterial>,
    gear_dye_default: Option<[f32; 4]>,
    gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    gear_worn_dye_palette: Option<[[f32; 4]; 6]>,
    gear_dye_detail_palette: Option<[[f32; 4]; 6]>,
    mod_wear: Option<WeaponModConditionMaterial>,
    surface_condition: Option<WeaponSurfaceConditionMaterial>,
    gear_pattern: Option<GearPatternMaterial>,
    investment_decal: Option<InvestmentDecalMaterial>,
    sampler_tag: Option<TagHash>,
    sampler: Option<ModelSamplerDesc>,
    color: Option<Arc<Texture>>,
    normal: Option<Arc<Texture>>,
    emissive: Option<Arc<Texture>>,
    control: Option<Arc<Texture>>,
    wear_scratches: Option<Arc<Texture>>,
    wear_grime: Option<Arc<Texture>>,
    wear_damage: Option<Arc<Texture>>,
    pattern_field: Option<Arc<Texture>>,
    character_surface_map: Option<Arc<Texture>>,
    character_detail_color: Option<Arc<Texture>>,
    character_procedural_map: Option<Arc<Texture>>,
    runner_surface_map: Option<Arc<Texture>>,
    runner_material_response_map: Option<Arc<Texture>>,
    runner_procedural_map: Option<Arc<Texture>>,
    runner_color_overlay_map: Option<Arc<Texture>>,
    runner_occlusion_map: Option<Arc<Texture>>,
    runner_detail_normal_a: Option<Arc<Texture>>,
    runner_detail_normal_b: Option<Arc<Texture>>,
    runner_detail_normal_c: Option<Arc<Texture>>,
    runner_detail_normal_d: Option<Arc<Texture>>,
    coating_environment_map: Option<Arc<Texture>>,
    coating_environment_sampler: Option<ModelSamplerDesc>,
    procedural_scale: f32,
}

#[derive(Clone)]
struct PreparedDraw {
    indices: Range<u32>,
    material_index: usize,
    pipeline: ModelPipelineKey,
    view_depth: f32,
    stable_index: usize,
    passes: Vec<RenderPassKind>,
    authored_source: Option<usize>,
    authored_stage: Option<AuthoredStageMetadata>,
    authored_native_vertex_supported: bool,
    native_c827: bool,
    native_surface: bool,
    native_decal: bool,
}

fn is_generic_opaque_receiver(draw: &PreparedDraw) -> bool {
    !draw.native_surface && !draw.native_decal && !draw.native_c827
        && draw.passes.iter().any(|pass| matches!(pass,
            RenderPassKind::OpaqueCompatibility | RenderPassKind::AlphaTestedCompatibility
                | RenderPassKind::UnknownCompatibility))
}

fn select_shadow_draws(
    visible_draws: &[PreparedDraw],
    authored_shadow_draws: &[PreparedDraw],
) -> Vec<PreparedDraw> {
    if !authored_shadow_draws.is_empty() {
        authored_shadow_draws.to_vec()
    } else {
        visible_draws
            .iter()
            .filter(|draw| draw.passes.contains(&RenderPassKind::Shadow))
            .cloned()
            .collect()
    }
}

pub(crate) struct ModelPaintCallback {
    preview: Arc<GpuModelPreview>,
    target_format: wgpu::TextureFormat,
    target_size: [u32; 2],
    scene: SceneUniform,
    export_camera: Option<ModelExportCamera>,
    export_focus: Option<(f32, [f32; 2])>,
    materials: Vec<LoadedMaterial>,
    native_surfaces: Vec<LoadedNativeSurface>,
    native_decals: Vec<LoadedNativeSurface>,
    native_resources_pending: bool,
    native_resource_failures: Vec<TagHash>,
    draws: Vec<PreparedDraw>,
    shadow_draws: Vec<PreparedDraw>,
    cubemap: Option<Arc<Texture>>,
    decal: Option<Arc<Texture>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ModelCameraFrame {
    pub(crate) center: [f32; 3],
    pub(crate) radius: f32,
}

/// Converts lighting controls to model coordinates independently of camera fit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelLightTransform {
    pub(crate) center: [f32; 3],
    pub(crate) scale: f32,
}

impl ModelLightTransform {
    pub(crate) fn new(environment: &ModelEnvironment, wireframe: &WireframePreview) -> Self {
        if environment.light_scale_with_model {
            let frame = environment
                .light_model_frame
                .unwrap_or_else(|| ModelCameraFrame::from_wireframe(wireframe));
            Self {
                center: frame.center,
                scale: frame.radius,
            }
        } else {
            Self {
                center: [0.0; 3],
                scale: 1.0,
            }
        }
    }

    pub(crate) fn to_model(self, point: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|axis| self.center[axis] + point[axis] * self.scale)
    }

    pub(crate) fn to_rig(self, point: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|axis| (point[axis] - self.center[axis]) / self.scale)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ModelExportCamera {
    frame: ModelCameraFrame,
    zoom: f32,
    pan: [f32; 2],
}

impl ModelExportCamera {
    pub(crate) fn from_wireframes(
        base: &WireframePreview,
        envelope: &WireframePreview,
        aspect: f32,
        yaw: f32,
        pitch: f32,
    ) -> Self {
        const FRAME_FILL: f32 = 0.88;
        let frame = ModelCameraFrame::from_wireframe(base);
        let base_bounds = projected_export_bounds(
            base.vertices.iter().copied(),
            frame.center,
            aspect,
            yaw,
            pitch,
        );
        let projected_center = [
            (base_bounds[0] + base_bounds[2]) * 0.5,
            (base_bounds[1] + base_bounds[3]) * 0.5,
        ];
        let zoom = fitted_export_zoom_for_positions_around(
            envelope.vertices.iter().copied(),
            frame.center,
            frame.radius,
            aspect,
            yaw,
            pitch,
            FRAME_FILL,
            projected_center,
        );
        let scale = 0.84 * zoom / frame.radius;
        let pan = [projected_center[0] * scale, -projected_center[1] * scale];
        Self { frame, zoom, pan }
    }
}

impl ModelCameraFrame {
    pub(crate) fn from_wireframe(wireframe: &WireframePreview) -> Self {
        let center = [
            (wireframe.min[0] + wireframe.max[0]) * 0.5,
            (wireframe.min[1] + wireframe.max[1]) * 0.5,
            (wireframe.min[2] + wireframe.max[2]) * 0.5,
        ];
        let extent = [
            wireframe.max[0] - wireframe.min[0],
            wireframe.max[1] - wireframe.min[1],
            wireframe.max[2] - wireframe.min[2],
        ];
        Self {
            center,
            radius: extent.into_iter().fold(0.0_f32, f32::max).max(0.0001),
        }
    }
}

fn shadow_bounding_radius(vertices: &[ModelVertex], center: [f32; 3], camera_radius: f32) -> f32 {
    let radius = vertices
        .iter()
        .map(|vertex| {
            let delta = [
                vertex.position[0] - center[0],
                vertex.position[1] - center[1],
                vertex.position[2] - center[2],
            ];
            delta
                .into_iter()
                .map(|axis| axis * axis)
                .sum::<f32>()
                .sqrt()
        })
        .fold(0.0_f32, f32::max)
        .max(camera_radius * 0.0001)
        .max(0.0001);
    // Five percent keeps filtered samples off the shadow-map edge without
    // throwing away the map resolution on long, narrow weapon geometry.
    radius * 1.05
}

impl ModelPaintCallback {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        preview: Arc<GpuModelPreview>,
        texture_cache: &TextureCache,
        wireframe: &WireframePreview,
        uv_transform: Option<UvTransformPreview>,
        camera_frame: Option<ModelCameraFrame>,
        yaw: f32,
        pitch: f32,
        zoom: f32,
        pan: egui::Vec2,
        show_stickers: bool,
        rect: egui::Rect,
        pixels_per_point: f32,
        environment: ModelEnvironment,
    ) -> Self {
        let camera_frame =
            camera_frame.unwrap_or_else(|| ModelCameraFrame::from_wireframe(wireframe));
        let center = camera_frame.center;
        let radius = camera_frame.radius;
        let shadow_radius = shadow_bounding_radius(&preview.vertices, center, radius);
        let aspect = (rect.height() / rect.width().max(1.0)).max(0.05);
        let transform = uv_transform.unwrap_or(UvTransformPreview {
            scale: [1.0; 2],
            offset: [0.0; 2],
        });
        let mut target_size = bounded_target_size(
            rect.width() * pixels_per_point,
            rect.height() * pixels_per_point,
        );

        let mut materials = Vec::<LoadedMaterial>::new();
        let mut native_surfaces = Vec::new();
        let mut native_decals = Vec::new();
        let mut native_resources_pending = false;
        let mut native_resource_failures = Vec::new();
        let mut draws = Vec::with_capacity(preview.draws.len());
        let mut authored_shadow_draws = Vec::with_capacity(preview.authored_shadow_draws.len());
        let visible_draw_count = preview.draws.len();
        let draw_sources = preview
            .draws
            .iter()
            .enumerate()
            .map(|(index, draw)| (false, index, draw))
            .chain(
                preview
                    .authored_shadow_draws
                    .iter()
                    .enumerate()
                    .map(|(index, draw)| (true, visible_draw_count + index, draw)),
            );
        for (is_authored_shadow, stable_index, draw) in draw_sources {
            if !is_authored_shadow && draw.sticker_proxy && !show_stickers {
                continue;
            }
            if !is_authored_shadow && draw.native_c827.is_some() {
                draws.push(PreparedDraw {
                    indices: draw.indices.clone(), material_index: usize::MAX,
                    pipeline: draw.pipeline, view_depth: model_view_depth(draw.center,center,yaw,pitch),
                    stable_index, passes: draw.packet.pass_plan.passes.clone(),
                    authored_source: draw.authored_source, authored_stage: draw.authored_stage,
                    authored_native_vertex_supported: false, native_c827: true, native_surface: false, native_decal: false,
                });
                continue;
            }
            let solid_color = draw.solid_color;
            let native_surface = !is_authored_shadow
                && (environment.lighting_model == LightingModel::SurfaceAlbedo || environment.diagnostic_pass == 1
                    || (environment.diagnostic_pass == 0 && matches!(environment.lighting_model,
                        LightingModel::TigerGgxCompatibility | LightingModel::TigerGgxApproximation | LightingModel::DebugLambert
                        | LightingModel::SurfaceNormals | LightingModel::SurfaceProperties | LightingModel::SurfaceEmissive)))
                && draw.native_surface.is_some();
            if native_surface {
                let spec = draw.native_surface.as_ref().expect("audited surface");
                // Retain complete shader resources through the existing bounded
                // texture cache. A pending resource is not replaced by a dummy.
                // Request every resource before checking readiness: short-circuiting
                // Option collection would serialize uploads across UI frames.
                let requested = spec.textures.iter().map(|&tag| texture_cache.get_or_load_material(tag).map(|t|t.0))
                    .collect::<Vec<_>>();
                let textures = requested.into_iter().collect::<Option<Vec<_>>>();
                if let Some(textures) = textures {
                    native_surfaces.push(LoadedNativeSurface { stable_index, textures });
                } else {
                    native_resources_pending=true;
                    native_resource_failures.extend(spec.textures.iter().copied()
                        .filter(|&tag| texture_cache.material_texture_failed(tag)));
                }
            }
            let native_decal = !is_authored_shadow && draw.native_decal.is_some()
                && (environment.lighting_model == LightingModel::SurfaceAlbedo || environment.diagnostic_pass == 1
                    || (environment.diagnostic_pass == 0 && matches!(environment.lighting_model,
                        LightingModel::TigerGgxCompatibility | LightingModel::TigerGgxApproximation | LightingModel::DebugLambert
                        | LightingModel::SurfaceNormals | LightingModel::SurfaceProperties | LightingModel::SurfaceEmissive)));
            if native_decal {
                let spec=draw.native_decal.as_ref().unwrap();
                let requested=spec.textures.iter().map(|&tag|texture_cache.get_or_load_material(tag).map(|t|t.0)).collect::<Vec<_>>();
                if let Some(textures)=requested.into_iter().collect::<Option<Vec<_>>>() {
                    native_decals.push(LoadedNativeSurface {stable_index,textures});
                } else {
                    native_resources_pending=true;
                    native_resource_failures.extend(spec.textures.iter().copied()
                        .filter(|&tag| texture_cache.material_texture_failed(tag)));
                }
            }
            let material_index = materials
                .iter()
                .position(|material| {
                    material.key == draw.material
                        && material.solid_color == solid_color
                        && material.solid_surface == draw.solid_surface
                        && material.iridescence_id == draw.iridescence_id
                        && material.transmission == draw.transmission
                        && material.forward_coating == draw.forward_coating
                        && material.character_surface == draw.character_surface
                        && material.runner_layered_surface == draw.runner_layered_surface
                        && material.runner_occlusion == draw.runner_occlusion
                        && material.alpha_mask == draw.alpha_mask
                        && material.shared_atlas_detail == draw.shared_atlas_detail
                        && material.blend == draw.pipeline.blend
                        && material.control_tag == draw.control
                        && material.roughness_channel == draw.roughness_channel
                        && material.mask_palette == draw.mask_palette
                        && material.dye_palette == draw_dye_palette(draw)
                        && material.gear_dye == draw.gear_dye
                        && material.gear_dye_default == draw.gear_dye_default
                        && material.gear_dye_palette == draw.gear_dye_palette
                        && material.gear_worn_dye_palette == draw.gear_worn_dye_palette
                        && material.gear_dye_detail_palette == draw.gear_dye_detail_palette
                        && material.mod_wear == draw.mod_wear
                        && material.surface_condition == draw.surface_condition
                        && material.gear_pattern == draw.gear_pattern
                        && material.investment_decal == draw.investment_decal
                        && material.sampler_tag == draw.sampler
                        && material.procedural_scale == draw.procedural_scale
                })
                .unwrap_or_else(|| {
                    let color = draw
                        .forward_coating
                        .map(|coating| coating.detail)
                        .or_else(|| draw.material.and_then(|material| material.color))
                        .map(|tag| texture_cache.get_or_default_material(tag).0)
                        .and_then(usable_2d_texture);
                    let normal = draw
                        .material
                        .and_then(|material| material.normal)
                        .and_then(|tag| texture_cache.get_or_load_material(tag))
                        .map(|loaded| loaded.0)
                        .and_then(usable_2d_texture);
                    let emissive = draw
                        .material
                        .and_then(|material| material.emissive)
                        .and_then(|tag| texture_cache.get_or_load_material(tag))
                        .map(|loaded| loaded.0)
                        .and_then(usable_2d_texture);
                    let control = draw
                        .control
                        .or_else(|| {
                            draw.character_surface
                                .filter(|surface| surface.mode == 2)
                                .map(|surface| surface.selector)
                        })
                        .and_then(|tag| texture_cache.get_or_load_material(tag))
                        .map(|loaded| loaded.0)
                        .and_then(usable_2d_texture);
                    let load_wear = |tag: TagHash| {
                        texture_cache
                            .get_or_load_material(tag)
                            .map(|loaded| loaded.0)
                            .and_then(usable_2d_texture)
                    };
                    let runner_procedural_wear = draw
                        .runner_layered_surface
                        .and_then(|surface| surface.procedural_wear);
                    let wear_scratches = draw
                        .mod_wear
                        .map(|wear| wear.scratches)
                        .or_else(|| draw.surface_condition.map(|surface| surface.response))
                        .or_else(|| runner_procedural_wear.map(|wear| wear[0]))
                        .and_then(load_wear);
                    let wear_grime = draw
                        .mod_wear
                        .map(|wear| wear.grime)
                        .or_else(|| runner_procedural_wear.map(|wear| wear[1]))
                        .and_then(load_wear);
                    let wear_damage = draw
                        .mod_wear
                        .map(|wear| wear.damage)
                        .or_else(|| draw.surface_condition.map(|condition| condition.breakup))
                        .or_else(|| runner_procedural_wear.map(|wear| wear[2]))
                        .and_then(load_wear);
                    let pattern_field = draw
                        .gear_pattern
                        .and_then(|pattern| load_wear(pattern.field))
                        .or_else(|| {
                            draw.shared_atlas_detail
                                .and_then(|surface| load_wear(surface.detail))
                        })
                        .or_else(|| {
                            draw.surface_condition
                                .and_then(|surface| load_wear(surface.detail))
                        });
                    let character_surface_map = draw
                        .character_surface
                        .and_then(|surface| load_wear(surface.surface));
                    let character_detail_color = draw
                        .character_surface
                        .and_then(|surface| load_wear(surface.detail_color));
                    let character_procedural_map = draw
                        .character_surface
                        .and_then(|surface| surface.procedural)
                        .and_then(load_wear);
                    let runner_surface_map = draw
                        .runner_layered_surface
                        .map(|surface| surface.surface)
                        .and_then(load_wear);
                    let runner_material_response_map = draw
                        .runner_layered_surface
                        .and_then(|surface| surface.material_response)
                        .and_then(load_wear);
                    let runner_procedural_map = draw
                        .runner_layered_surface
                        .and_then(|surface| surface.procedural)
                        .and_then(load_wear);
                    let runner_color_overlay_map = draw
                        .runner_layered_surface
                        .and_then(|surface| surface.color_overlay)
                        .and_then(load_wear);
                    let runner_occlusion_map = draw
                        .runner_occlusion
                        .and_then(|occlusion| load_wear(occlusion.texture));
                    let runner_detail_normal_a = draw
                        .runner_layered_surface
                        .and_then(|surface| load_wear(surface.detail_normal_a));
                    let runner_detail_normal_b = draw
                        .runner_layered_surface
                        .and_then(|surface| load_wear(surface.detail_normal_b));
                    let runner_detail_normal_c = draw
                        .runner_layered_surface
                        .and_then(|surface| surface.detail_normal_c)
                        .and_then(load_wear);
                    let runner_detail_normal_d = draw
                        .runner_layered_surface
                        .and_then(|surface| surface.detail_normal_d)
                        .and_then(load_wear);
                    let coating_environment_map = draw
                        .forward_coating
                        .and_then(|coating| texture_cache.get_or_load_material(coating.environment))
                        .map(|loaded| loaded.0)
                        .filter(|texture| texture.desc.kind() == TextureType::TextureCube);
                    let coating_environment_sampler = draw
                        .forward_coating
                        .and_then(|coating| load_model_sampler_desc(coating.environment_sampler));
                    let dye_palette = draw_dye_palette(draw);
                    materials.push(LoadedMaterial {
                        key: draw.material,
                        solid_color,
                        solid_surface: draw.solid_surface,
                        iridescence_id: draw.iridescence_id,
                        transmission: draw.transmission,
                        forward_coating: draw.forward_coating,
                        character_surface: draw.character_surface,
                        runner_layered_surface: draw.runner_layered_surface,
                        runner_occlusion: draw.runner_occlusion,
                        alpha_mask: draw.alpha_mask,
                        shared_atlas_detail: draw.shared_atlas_detail,
                        blend: draw.pipeline.blend,
                        control_tag: draw.control,
                        roughness_channel: draw.roughness_channel,
                        mask_palette: draw.mask_palette,
                        dye_palette,
                        gear_dye: draw.gear_dye,
                        gear_dye_default: draw.gear_dye_default,
                        gear_dye_palette: draw.gear_dye_palette,
                        gear_worn_dye_palette: draw.gear_worn_dye_palette,
                        gear_dye_detail_palette: draw.gear_dye_detail_palette,
                        mod_wear: draw.mod_wear,
                        surface_condition: draw.surface_condition,
                        gear_pattern: draw.gear_pattern,
                        investment_decal: draw.investment_decal,
                        sampler_tag: draw.sampler,
                        sampler: draw.sampler.and_then(load_model_sampler_desc),
                        color,
                        normal,
                        emissive,
                        control,
                        wear_scratches,
                        wear_grime,
                        wear_damage,
                        pattern_field,
                        character_surface_map,
                        character_detail_color,
                        character_procedural_map,
                        runner_surface_map,
                        runner_material_response_map,
                        runner_procedural_map,
                        runner_color_overlay_map,
                        runner_occlusion_map,
                        runner_detail_normal_a,
                        runner_detail_normal_b,
                        runner_detail_normal_c,
                        runner_detail_normal_d,
                        coating_environment_map,
                        coating_environment_sampler,
                        procedural_scale: draw.procedural_scale,
                    });
                    materials.len() - 1
                });
            let pipeline = draw.pipeline;
            let passes = draw.packet.pass_plan.passes.clone();
            let prepared = PreparedDraw {
                indices: draw.indices.clone(),
                material_index,
                pipeline,
                view_depth: model_view_depth(draw.center, center, yaw, pitch),
                stable_index,
                passes,
                authored_source: draw.authored_source,
                authored_stage: draw.authored_stage,
                authored_native_vertex_supported: draw.authored_native_vertex_supported,
                native_c827: draw.native_c827.is_some(),
                native_surface,
                native_decal,
            };
            if is_authored_shadow {
                authored_shadow_draws.push(prepared);
            } else {
                draws.push(prepared);
            }
        }
        let shadow_draws = select_shadow_draws(&draws, &authored_shadow_draws);

        draws.sort_by(|left, right| {
            let left_blended = blend_enabled(left.pipeline.blend);
            let right_blended = blend_enabled(right.pipeline.blend);
            left_blended.cmp(&right_blended).then_with(|| {
                if left_blended && right_blended {
                    let stage2=|draw:&PreparedDraw|draw.authored_stage.is_some_and(|stage|stage.raw_stage==2);
                    stage2(left).cmp(&stage2(right)).then_with(|| {
                        if stage2(left) && stage2(right) {std::cmp::Ordering::Equal}
                        else {left.view_depth.total_cmp(&right.view_depth)}
                    })
                        .then_with(|| left.stable_index.cmp(&right.stable_index))
                } else {
                    std::cmp::Ordering::Equal
                }
            })
        });
        if environment.hiz_culling {
            hiz_cull_draws(
                &preview,
                &mut draws,
                center,
                radius,
                aspect,
                yaw,
                pitch,
                [
                    pan.x * 2.0 / rect.width().max(1.0),
                    -pan.y * 2.0 / rect.height().max(1.0),
                ],
                zoom,
            );
        }
        let exposure = environment.exposure
            * if environment.auto_exposure {
                estimate_material_exposure(texture_cache, &materials)
            } else {
                1.0
            };
        let cubemap = environment
            .cubemap
            .into_iter()
            .chain(
                materials
                    .iter()
                    .filter_map(|material| material.key.and_then(|key| key.color)),
            )
            .find_map(|tag| {
                texture_cache
                    .get_or_load_material(tag)
                    .map(|loaded| loaded.0)
            })
            .filter(|texture| texture.desc.kind() == TextureType::TextureCube);
        let decal = environment
            .decal
            .and_then(|tag| {
                texture_cache
                    .get_or_load_material(tag)
                    .map(|loaded| loaded.0)
            })
            .and_then(usable_2d_texture);

        let (light_direction, sun_height) = first_person_key_light(
            environment.time_of_day,
            light_cast_direction(&environment),
            environment.shadow_strength,
        );
        let mut light_direction = light_direction;
        let world_light_direction = light_direction[..3]
            .try_into()
            .expect("light direction xyz");
        light_direction[..3].copy_from_slice(&fixed_light_direction_to_view(world_light_direction));
        let light_transform = ModelLightTransform::new(&environment, wireframe);
        let light_position_world = light_transform.to_model(light_source_position(&environment));
        let light_position_view = fixed_light_direction_to_view(std::array::from_fn(|axis| {
            light_position_world[axis] - center[axis]
        }));
        let (outer_cone_cosine, inner_cone_cosine) =
            light_cone_cosines(environment.light_cone_angle);
        let light_range = environment.light_range.max(0.05) * light_transform.scale;
        if draws.iter().any(|draw|draw.native_surface || draw.native_decal) {
            // Bound the complete graph, including four retained FP32 native
            // outputs. Other views retain their existing resolution budget.
            let mixed_receivers=draws.iter().any(|draw|draw.native_decal || draw.native_c827)
                && draws.iter().any(is_generic_opaque_receiver);
            let bytes_per_pixel=123+(if draws.iter().any(|draw|draw.native_decal) {16}else{0})
                +(if mixed_receivers {80}else{0});
            let max_pixels=(MAX_MODEL_TARGET_BYTES-u64::from(SHADOW_MAP_SIZE).pow(2)*4)/bytes_per_pixel;
            let pixels=u64::from(target_size[0])*u64::from(target_size[1]);
            if pixels>max_pixels {
                let scale=(max_pixels as f64/pixels as f64).sqrt();
                target_size=target_size.map(|v|((v as f64*scale).floor() as u32).max(1));
            }
        }
        Self {
            preview,
            target_format: texture_cache.render_state.target_format,
            target_size,
            export_camera: None,
            export_focus: None,
            scene: SceneUniform {
                // center.w remains available to diagnostics and camera tools;
                // spotlight projection uses the explicit source position.
                center: [center[0], center[1], center[2], shadow_radius],
                params0: [radius, yaw, pitch, zoom],
                params1: [
                    aspect,
                    pan.x * 2.0 / rect.width().max(1.0),
                    -pan.y * 2.0 / rect.height().max(1.0),
                    0.0,
                ],
                uv_transform: [
                    transform.scale[0],
                    transform.scale[1],
                    transform.offset[0],
                    transform.offset[1],
                ],
                light_direction,
                light_parameters: [
                    environment.light_size.clamp(0.0, 10.0),
                    light_range,
                    outer_cone_cosine,
                    inner_cone_cosine,
                ],
                light_position: [
                    light_position_view[0],
                    light_position_view[1],
                    light_position_view[2],
                    light_transform.scale,
                ],
                postprocess0: [
                    exposure,
                    environment.bloom_strength,
                    environment.fog_density,
                    environment.sun_intensity,
                ],
                postprocess1: [
                    target_size[0] as f32,
                    target_size[1] as f32,
                    sun_height,
                    environment.vertex_ao_strength,
                ],
                postprocess2: [
                    environment.diagnostic_pass as f32,
                    environment.distortion,
                    environment.fxaa as u8 as f32,
                    environment.ssao_strength,
                ],
                postprocess3: [
                    environment.water_strength,
                    environment.water_level,
                    environment.decal_strength,
                    environment.road_decal_strength,
                ],
                postprocess4: [
                    (!texture_cache.render_state.target_format.is_srgb()) as u8 as f32,
                    environment.tone_mapping as u8 as f32,
                    environment.ambient_intensity,
                    environment.specular_ibl_intensity,
                ],
                postprocess5: [
                    environment.brightness,
                    environment.contrast,
                    environment.saturation,
                    environment.gamma,
                ],
                fidelity: [
                    ((!native_surfaces.is_empty() || !native_decals.is_empty()) && environment.diagnostic_pass == 0 && matches!(environment.lighting_model,
                        LightingModel::TigerGgxCompatibility | LightingModel::TigerGgxApproximation | LightingModel::DebugLambert
                        | LightingModel::SurfaceNormals | LightingModel::SurfaceProperties | LightingModel::SurfaceEmissive)) as u8 as f32,
                    match environment.lighting_model {
                        LightingModel::TigerGgxCompatibility => 0.0,
                        LightingModel::TigerGgxApproximation => 1.0,
                        LightingModel::DebugLambert => 2.0,
                        LightingModel::SurfaceNormals => 3.0,
                        LightingModel::SurfaceProperties => 4.0,
                        LightingModel::SurfaceEmissive => 5.0,
                        LightingModel::SurfaceFlags => 6.0,
                        LightingModel::SurfaceAlbedo => 7.0,
                    },
                    environment.tfx_time_seconds,
                    environment.tfx_speed,
                ],
                shadow_parameters: [0.0; 4],
                light_color: [environment.light_color[0], environment.light_color[1], environment.light_color[2], 0.0],
                ambient_sky: [environment.ambient_sky_color[0], environment.ambient_sky_color[1], environment.ambient_sky_color[2], 0.0],
                ambient_ground: [environment.ambient_ground_color[0], environment.ambient_ground_color[1], environment.ambient_ground_color[2], 0.0],
            },
            materials,
            native_surfaces,
            native_decals,
            native_resources_pending,
            native_resource_failures,
            draws,
            shadow_draws,
            cubemap,
            decal,
        }
    }

    /// Export current assembled model with deterministic framing. Interactive
    /// pan/zoom never leak into the file. Weapon callers provide a camera fitted
    /// to the base weapon plus its complete compatible-mod envelope, so changing
    /// the equipped mod cannot move or rescale the weapon.
    pub(crate) fn with_export_camera(mut self, camera: Option<ModelExportCamera>) -> Self {
        self.export_camera = camera;
        self
    }

    /// Magnify an export around a point of the fitted frame, given in
    /// normalized device coordinates of that frame.
    pub(crate) fn with_export_focus(mut self, zoom: f32, focus: [f32; 2]) -> Self {
        self.export_focus = Some((zoom, focus));
        self
    }

    /// False while authored material textures are still loading. A failed
    /// load counts as ready so export reports the failure instead of waiting.
    pub(crate) fn resources_ready(&self) -> bool {
        !self.native_resources_pending || !self.native_resource_failures.is_empty()
    }

    pub(crate) fn show_loading_status(&self, ui: &egui::Ui, rect: egui::Rect) {
        if !self.native_resources_pending { return; }
        let (message, color) = if self.native_resource_failures.is_empty() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
            ("Loading material textures…".to_string(), egui::Color32::WHITE)
        } else {
            let tags=self.native_resource_failures.iter().unique().map(ToString::to_string).join(", ");
            (format!("Material textures failed to load: {tags}"), egui::Color32::LIGHT_RED)
        };
        ui.painter().text(rect.center(),egui::Align2::CENTER_CENTER,message,
            egui::FontId::proportional(14.0),color);
    }

    pub(crate) fn export_image(
        self,
        render_state: &eframe::egui_wgpu::RenderState,
        path: &Path,
        output_size: [u32; 2],
        format: image::ImageFormat,
    ) -> anyhow::Result<()> {
        std::fs::write(
            path,
            self.export_image_bytes(render_state, output_size, format)?,
        )?;
        Ok(())
    }

    pub(crate) fn export_image_bytes(
        mut self,
        render_state: &eframe::egui_wgpu::RenderState,
        output_size: [u32; 2],
        format: image::ImageFormat,
    ) -> anyhow::Result<Vec<u8>> {
        const FRAME_FILL: f32 = 0.88;
        anyhow::ensure!(self.native_resource_failures.is_empty(), "Authored material texture loading failed: {:?}", self.native_resource_failures);
        anyhow::ensure!(!self.native_resources_pending, "Authored material textures are not ready for export");
        anyhow::ensure!(
            output_size.into_iter().all(|dimension| dimension > 0),
            "model export dimensions must be non-zero"
        );
        let max_dimension = render_state.device.limits().max_texture_dimension_2d;
        anyhow::ensure!(
            output_size
                .into_iter()
                .all(|dimension| dimension <= max_dimension),
            "model export {}x{} exceeds GPU texture limit {max_dimension}",
            output_size[0],
            output_size[1]
        );
        self.target_size = output_size;
        if let Some(camera) = self.export_camera {
            self.apply_export_camera(camera);
        } else {
            self.fit_export_camera(FRAME_FILL);
        }
        if let Some((zoom, focus)) = self.export_focus {
            self.scene.params0[3] *= zoom;
            self.scene.params1[1] -= focus[0] * zoom;
            self.scene.params1[2] -= focus[1] * zoom;
        }
        self.target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        self.scene.postprocess4[0] = 0.0;
        self.scene.postprocess4[1] = if self.scene.postprocess4[1] > 0.5 {
            2.0
        } else {
            -1.0
        };

        let mut resources = CallbackResources::default();
        let descriptor = ScreenDescriptor {
            size_in_pixels: self.target_size,
            pixels_per_point: 1.0,
        };
        let mut prepare_encoder =
            render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("quicktag_model_export_prepare"),
                });
        self.prepare(
            &render_state.device,
            &render_state.queue,
            &descriptor,
            &mut prepare_encoder,
            &mut resources,
        );
        render_state.queue.submit(Some(prepare_encoder.finish()));

        let pipelines = resources
            .get::<ModelPipelineResources>()
            .ok_or_else(|| anyhow::anyhow!("model export pipeline was not prepared"))?;
        let frame = resources
            .get::<ModelFrameResources>()
            .ok_or_else(|| anyhow::anyhow!("model export frame was not prepared"))?;
        let size = self.target_size;
        let output = render_state
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("quicktag_model_export_output"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        let output_view = create_model_texture_view(&output, &Default::default());
        let unpadded_bytes_per_row = size[0] * 4;
        let bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
        let readback = render_state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quicktag_model_export_readback"),
            size: u64::from(bytes_per_row) * u64::from(size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("quicktag_model_export_copy"),
                });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_export_present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipelines.present_pipeline);
            pass.set_bind_group(0, &frame.present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        render_state.queue.submit(Some(encoder.finish()));
        render_state.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        render_state.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("model export readback callback failed: {error}"))??;
        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((size[0] * size[1] * 4) as usize);
        for row in mapped.chunks_exact(bytes_per_row as usize) {
            pixels.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        readback.unmap();
        let image = image::RgbaImage::from_raw(size[0], size[1], pixels)
            .ok_or_else(|| anyhow::anyhow!("invalid model export readback dimensions"))?;
        let mut output = std::io::Cursor::new(Vec::new());
        image.write_to(&mut output, format)?;
        Ok(output.into_inner())
    }

    fn fit_export_camera(&mut self, frame_fill: f32) {
        self.scene.params0[3] = fitted_export_zoom(
            &self.preview.vertices,
            self.scene.center,
            self.scene.params0[0],
            self.scene.params1[0],
            self.scene.params0[1],
            self.scene.params0[2],
            frame_fill,
        );
        self.scene.params1[1] = 0.0;
        self.scene.params1[2] = 0.0;
    }

    fn apply_export_camera(&mut self, camera: ModelExportCamera) {
        // Lighting is anchored in renderer/view space. Reframing or rotating the
        // model must not translate the light with the mesh.
        let shadow_radius = shadow_bounding_radius(
            &self.preview.vertices,
            camera.frame.center,
            camera.frame.radius,
        );
        self.scene.center = [
            camera.frame.center[0],
            camera.frame.center[1],
            camera.frame.center[2],
            shadow_radius,
        ];
        self.scene.params0[0] = camera.frame.radius;
        self.scene.params0[3] = camera.zoom;
        self.scene.params1[1] = camera.pan[0];
        self.scene.params1[2] = camera.pan[1];
    }

    fn frame_resource_key(&self) -> FrameResourceKey {
        let materials = self
            .materials
            .iter()
            .fold(0xcbf29ce484222325_u64, |hash, material| {
                let texture_id = |texture: &Option<Arc<Texture>>| {
                    texture
                        .as_ref()
                        .map_or(0, |texture| Arc::as_ptr(texture) as usize as u64)
                };
                let key = material.key;
                let tint = key.map_or(0, |key| u32::from_le_bytes(key.color_tint));
                let mut values = vec![
                    key.and_then(|key| key.color).map_or(0, |tag| u64::from(tag.0)),
                    key.and_then(|key| key.normal)
                        .map_or(0, |tag| u64::from(tag.0)),
                    key.and_then(|key| key.emissive)
                        .map_or(0, |tag| u64::from(tag.0)),
                    u64::from(tint),
                    key.map_or(0, |key| u64::from(key.emissive_strength)),
                    material.control_tag.map_or(0, |tag| u64::from(tag.0)),
                    material
                        .runner_occlusion
                        .map_or(0, |occlusion| u64::from(occlusion.texture.0)),
                    material.sampler_tag.map_or(0, |tag| u64::from(tag.0)),
                    u64::from(material.blend),
                    u64::from(material.roughness_channel),
                    texture_id(&material.color),
                    texture_id(&material.normal),
                    texture_id(&material.emissive),
                    texture_id(&material.control),
                    texture_id(&material.wear_scratches),
                    texture_id(&material.wear_grime),
                    texture_id(&material.wear_damage),
                    texture_id(&material.pattern_field),
                    texture_id(&material.character_surface_map),
                    texture_id(&material.character_detail_color),
                    texture_id(&material.character_procedural_map),
                    texture_id(&material.runner_surface_map),
                    texture_id(&material.runner_material_response_map),
                    texture_id(&material.runner_procedural_map),
                    texture_id(&material.runner_color_overlay_map),
                    texture_id(&material.runner_occlusion_map),
                    texture_id(&material.runner_detail_normal_a),
                    texture_id(&material.runner_detail_normal_b),
                    texture_id(&material.runner_detail_normal_c),
                    texture_id(&material.runner_detail_normal_d),
                ];
                if let Some(surface) = material.character_surface {
                    values.extend([
                        u64::from(surface.mode),
                        u64::from(surface.surface.0),
                        u64::from(surface.selector.0),
                        u64::from(surface.detail_color.0),
                        u64::from(surface.detail_normal.0),
                    ]);
                    values.extend(
                        surface
                            .detail_transform
                            .into_iter()
                            .chain(surface.detail_base)
                            .chain(surface.detail_scale)
                            .chain([surface.detail_gate])
                            .chain(surface.extra.into_iter().flatten())
                            .chain(surface.palette.into_iter().flatten())
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(alpha) = material.alpha_mask {
                    values.extend([
                        u64::from(alpha.texture.0),
                        u64::from(alpha.threshold.to_bits()),
                    ]);
                }
                if let Some(detail) = material.shared_atlas_detail {
                    values.push(u64::from(detail.detail.0));
                    values.extend(
                        detail
                            .projection
                            .into_iter()
                            .chain([detail.exponent])
                            .chain(detail.base)
                            .chain(detail.scale)
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(surface) = material.runner_layered_surface {
                    values.extend([
                        u64::from(surface.surface.0),
                        u64::from(surface.detail_normal_a.0),
                        u64::from(surface.detail_normal_b.0),
                        surface.detail_normal_c.map_or(0, |tag| u64::from(tag.0)),
                        surface.detail_normal_d.map_or(0, |tag| u64::from(tag.0)),
                        surface.material_response.map_or(0, |tag| u64::from(tag.0)),
                        u64::from(surface.mode),
                    ]);
                    values.extend(
                        surface
                            .constants
                            .into_iter()
                            .flatten()
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(decal) = material.investment_decal {
                    values.extend([
                        u64::from(decal.color.0),
                        u64::from(decal.mask.0),
                        decal.detail.map_or(0, |tag| u64::from(tag.0)),
                        match decal.mode {
                            InvestmentDecalMode::SelectorMask => 1,
                            InvestmentDecalMode::DetailSelectorMask => 2,
                            InvestmentDecalMode::SceneNormalColorMask => 3,
                        },
                        match decal.mask_mode {
                            InvestmentDecalMaskMode::Threshold => 0,
                            InvestmentDecalMaskMode::UvSplit => 1,
                            InvestmentDecalMaskMode::Binary => 2,
                        },
                        u64::from(decal.selector_color_count),
                        u64::from(decal.atlas_selector_max),
                    ]);
                    values.extend(
                        decal
                            .selector_colors
                            .into_iter()
                            .flatten()
                            .chain(decal.detail_transform)
                            .chain(decal.detail_base)
                            .chain(decal.detail_scale)
                            .chain(decal.grayscale_remap)
                            .chain(decal.positive_mask_remap)
                            .chain(decal.negative_mask_remap)
                            .chain([decal.mask_threshold, decal.output_gate])
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(wear) = material.mod_wear {
                    values.extend([
                        u64::from(wear.scratches.0),
                        u64::from(wear.grime.0),
                        u64::from(wear.damage.0),
                        wear.rarity.map_or(0, |rarity| u64::from(rarity.tier())),
                    ]);
                    values.extend(
                        wear.scratches_projection
                            .into_iter()
                            .chain(wear.grime_projection)
                            .chain(wear.damage_projection)
                            .chain(wear.grime_projection_unique_delta)
                            .chain(wear.damage_projection_unique_delta)
                            .chain(wear.scratches_remap_base)
                            .chain(wear.scratches_remap_scale)
                            .chain([wear.unique_id, wear.condition_blend])
                            .chain(wear.condition_controls.into_iter().flatten())
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(condition) = material.surface_condition {
                    values.push(u64::from(condition.response.0));
                    values.push(u64::from(condition.detail.0));
                    values.push(u64::from(condition.breakup.0));
                    values.extend(
                        condition
                            .detail_projection
                            .into_iter()
                            .chain([
                                condition.detail_exponent,
                                condition.detail_roughness,
                                condition.detail_remap[0],
                                condition.detail_remap[1],
                            ])
                            .chain(condition.projection)
                            .chain([condition.phase])
                            .chain(condition.triangle)
                            .chain(condition.orientation)
                            .chain(condition.albedo)
                            .chain([condition.roughness, condition.normal_flatten])
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(pattern) = material.gear_pattern {
                    values.push(u64::from(pattern.field.0));
                    values.extend(
                        pattern
                            .projection
                            .into_iter()
                            .chain([pattern.normal_power, pattern.field_midpoint])
                            .chain(pattern.warp)
                            .chain(pattern.stripe)
                            .chain(pattern.contour)
                            .chain(pattern.contour_remap)
                            .chain(pattern.colors.into_iter().flatten())
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(color) = material.solid_color {
                    values.extend(color.map(|value| u64::from(value.to_bits())));
                }
                if let Some(surface) = material.solid_surface {
                    values.extend(surface.map(|value| u64::from(value.to_bits())));
                }
                if let Some(iridescence_id) = material.iridescence_id {
                    values.push(u64::from(iridescence_id.to_bits()));
                }
                if let Some(palette) = material.mask_palette {
                    values.extend(
                        palette
                            .into_iter()
                            .flatten()
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(palette) = material.dye_palette {
                    values.extend(
                        palette
                            .into_iter()
                            .flatten()
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                if let Some(dye) = material.gear_dye {
                    values.extend(
                        dye.roughness_remap
                            .into_iter()
                            .chain(dye.metal_remap)
                            .map(|value| u64::from(value.to_bits())),
                    );
                }
                // TFX register images are not part of the current WGSL ABI yet.
                // Do not invalidate/rebuild every material buffer and render
                // bundle as time-dependent TFX values advance each frame.
                values.into_iter().fold(hash, |hash, value| {
                    (hash ^ value).wrapping_mul(0x100000001b3)
                })
            });
        let materials = self.draws.iter().fold(materials, |hash, draw| {
            hash.wrapping_mul(0x100000001b3)
                ^ u64::from(draw.indices.start)
                ^ u64::from(draw.indices.end).rotate_left(13)
                ^ (draw.material_index as u64).rotate_left(29)
                ^ u64::from(draw.pipeline.rasterizer).rotate_left(41)
                ^ u64::from(draw.pipeline.blend).rotate_left(47)
                ^ u64::from(draw.pipeline.depth_stencil).rotate_left(53)
                ^ u64::from(draw.pipeline.depth_bias).rotate_left(59)
        });
        FrameResourceKey {
            preview: Arc::as_ptr(&self.preview) as usize,
            size: self.target_size,
            materials: self.native_surfaces.iter().chain(&self.native_decals).fold(materials, |hash, surface| {
                surface.textures.iter().fold(hash ^ surface.stable_index as u64, |hash,t| {
                    (hash ^ Arc::as_ptr(t) as usize as u64).wrapping_mul(0x100000001b3)
                })
            }) ^ (self.draws.iter().any(|d|d.native_surface) as u64).rotate_left(17),
            cubemap: self
                .cubemap
                .as_ref()
                .map_or(0, |texture| Arc::as_ptr(texture) as usize),
            decal: self
                .decal
                .as_ref()
                .map_or(0, |texture| Arc::as_ptr(texture) as usize),
        }
    }
}

fn fitted_export_zoom(
    vertices: &[ModelVertex],
    center: [f32; 4],
    radius: f32,
    aspect: f32,
    yaw: f32,
    pitch: f32,
    frame_fill: f32,
) -> f32 {
    fitted_export_zoom_for_positions(
        vertices.iter().map(|vertex| vertex.position),
        [center[0], center[1], center[2]],
        radius,
        aspect,
        yaw,
        pitch,
        frame_fill,
    )
}

fn fitted_export_zoom_for_positions(
    positions: impl IntoIterator<Item = [f32; 3]>,
    center: [f32; 3],
    radius: f32,
    aspect: f32,
    yaw: f32,
    pitch: f32,
    frame_fill: f32,
) -> f32 {
    fitted_export_zoom_for_positions_around(
        positions, center, radius, aspect, yaw, pitch, frame_fill, [0.0; 2],
    )
}

fn fitted_export_zoom_for_positions_around(
    positions: impl IntoIterator<Item = [f32; 3]>,
    center: [f32; 3],
    radius: f32,
    aspect: f32,
    yaw: f32,
    pitch: f32,
    frame_fill: f32,
    projected_center: [f32; 2],
) -> f32 {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let mut max_x = 0.0_f32;
    let mut max_y = 0.0_f32;
    for position in positions {
        let value = [
            position[0] - center[0],
            position[2] - center[2],
            position[1] - center[1],
        ];
        let yawed = [
            value[0] * cos_yaw + value[2] * sin_yaw,
            value[1],
            -value[0] * sin_yaw + value[2] * cos_yaw,
        ];
        let view_x = yawed[0];
        let view_y = yawed[1] * cos_pitch - yawed[2] * sin_pitch;
        max_x = max_x.max((view_x * aspect - projected_center[0]).abs());
        max_y = max_y.max((view_y - projected_center[1]).abs());
    }
    let extent = max_x.max(max_y).max(0.0001);
    frame_fill.clamp(0.1, 0.98) * radius / (0.84 * extent)
}

fn projected_export_bounds(
    positions: impl IntoIterator<Item = [f32; 3]>,
    center: [f32; 3],
    aspect: f32,
    yaw: f32,
    pitch: f32,
) -> [f32; 4] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for position in positions {
        let value = [
            position[0] - center[0],
            position[2] - center[2],
            position[1] - center[1],
        ];
        let yawed = [
            value[0] * cos_yaw + value[2] * sin_yaw,
            value[1],
            -value[0] * sin_yaw + value[2] * cos_yaw,
        ];
        let x = yawed[0] * aspect;
        let y = yawed[1] * cos_pitch - yawed[2] * sin_pitch;
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].min(y);
        bounds[2] = bounds[2].max(x);
        bounds[3] = bounds[3].max(y);
    }
    if bounds.iter().all(|value| value.is_finite()) {
        bounds
    } else {
        [0.0; 4]
    }
}

/// Render the interactive model at physical display resolution. The previous
/// 2x supersample multiplied every persistent MRT/copy/depth allocation by four
/// and could put a single 1080p model view above half a gigabyte of GPU memory.
/// Export remains explicitly sized by the export path, so interactive preview
/// supersampling is not part of the authored Tiger contract.
const MODEL_RENDER_SUPERSAMPLE: f32 = 1.0;
const MAX_MODEL_TARGET_DIMENSION: f32 = 2560.0;
const MAX_MODEL_TARGET_PIXELS: f32 = 3_686_400.0; // 2560x1440
const MAX_MODEL_TARGET_BYTES: u64 = 256 * 1024 * 1024;

fn estimated_model_target_bytes(size: [u32; 2], features: ModelTargetFeatures) -> u64 {
    let pixels = u64::from(size[0]) * u64::from(size[1]);
    // Always-resident full-resolution targets:
    // HDR compatibility color + lit color (16 B), four 4-byte logical MRTs
    // (16 B), HDR emission (8 B), and depth (4 B).
    let mut bytes = pixels.saturating_mul(44);
    if features.native_surface { bytes=bytes.saturating_add(pixels.saturating_mul(64)); }
    if features.native_decals { bytes=bytes.saturating_add(pixels.saturating_mul(16)); }
    if features.mixed_receivers { bytes=bytes.saturating_add(pixels.saturating_mul(80)); }
    if features.deferred_normal_copy {
        bytes = bytes.saturating_add(pixels.saturating_mul(4));
    }
    if features.distortion {
        // HDR scene copy + RGBA8 distortion payload.
        bytes = bytes.saturating_add(pixels.saturating_mul(12));
    }
    if features.bloom {
        // RGBA16F half-resolution + two quarter-resolution ping-pong targets.
        bytes = bytes.saturating_add(pixels.saturating_mul(3));
    }
    bytes.saturating_add(
        u64::from(SHADOW_MAP_SIZE)
            .saturating_mul(u64::from(SHADOW_MAP_SIZE))
            .saturating_mul(4),
    )
}

fn bounded_target_size(width: f32, height: f32) -> [u32; 2] {
    let desired = [
        width.max(1.0) * MODEL_RENDER_SUPERSAMPLE,
        height.max(1.0) * MODEL_RENDER_SUPERSAMPLE,
    ];
    let pixel_scale = (MAX_MODEL_TARGET_PIXELS / (desired[0] * desired[1]))
        .sqrt()
        .min(1.0);
    let dimension_scale = (MAX_MODEL_TARGET_DIMENSION / desired[0])
        .min(MAX_MODEL_TARGET_DIMENSION / desired[1])
        .min(1.0);
    let scale = pixel_scale.min(dimension_scale);
    let mut size = [
        (desired[0] * scale).round().max(1.0) as u32,
        (desired[1] * scale).round().max(1.0) as u32,
    ];
    // Rounding can cross the pixel budget by one row/column. Preserve aspect
    // while making the budget a hard GPU-memory ceiling.
    while size[0] as f32 * size[1] as f32 > MAX_MODEL_TARGET_PIXELS {
        if size[0] >= size[1] {
            size[0] -= 1;
        } else {
            size[1] -= 1;
        }
    }
    size
}

#[derive(Clone, Copy)]
struct MaterialLuminance {
    log: f32,
    linear: f32,
}

struct AdaptedExposure {
    scale: f32,
    updated: Instant,
}

const MAX_MATERIAL_ANALYSIS_CACHE_ENTRIES: usize = 4096;
const MAX_ADAPTED_EXPOSURE_STATES: usize = 256;

static MATERIAL_LUMINANCE: LazyLock<Mutex<HashMap<TagHash, Option<MaterialLuminance>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static ADAPTED_EXPOSURE: LazyLock<Mutex<HashMap<Vec<u32>, AdaptedExposure>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn estimate_material_exposure(texture_cache: &TextureCache, materials: &[LoadedMaterial]) -> f32 {
    let mut tags = materials
        .iter()
        .filter_map(|material| material.key.and_then(|key| key.color))
        .unique()
        .collect_vec();
    tags.sort_unstable_by_key(|tag| tag.0);
    let luminances = tags
        .iter()
        .filter_map(|tag| texture_luminance(texture_cache, *tag))
        .collect_vec();
    if luminances.is_empty() {
        return 1.0;
    }
    let target = exposure_target(&luminances);
    let key = tags.iter().map(|tag| tag.0).collect_vec();
    let now = Instant::now();
    let Ok(mut states) = ADAPTED_EXPOSURE.lock() else {
        return target;
    };
    if states.len() >= MAX_ADAPTED_EXPOSURE_STATES && !states.contains_key(&key) {
        states.clear();
    }
    let state = states.entry(key).or_insert(AdaptedExposure {
        scale: target,
        updated: now,
    });
    let delta_time = now.duration_since(state.updated).as_secs_f32().min(0.25);
    state.scale = adapt_exposure(state.scale, target, delta_time);
    state.updated = now;
    state.scale
}

fn exposure_target(luminances: &[MaterialLuminance]) -> f32 {
    let count = luminances.len() as f32;
    let geometric = (luminances.iter().map(|value| value.log).sum::<f32>() / count).exp2();
    let linear = luminances.iter().map(|value| value.linear).sum::<f32>() / count;
    let highlight_protected = geometric + (linear - geometric) * 0.3;
    (0.18 / highlight_protected.clamp(0.001, 65_000.0)).clamp(0.35, 3.0)
}

fn adapt_exposure(current: f32, target: f32, delta_time: f32) -> f32 {
    let speed = if target < current { 2.0 } else { 1.0 };
    current + (target - current) * (1.0 - (-speed * delta_time.max(0.0)).exp())
}

fn texture_luminance(texture_cache: &TextureCache, tag: TagHash) -> Option<MaterialLuminance> {
    if let Some(value) = MATERIAL_LUMINANCE
        .lock()
        .ok()
        .and_then(|cache| cache.get(&tag).copied())
    {
        return value;
    }
    let value = texture_cache
        .get_or_load_material(tag)
        .and_then(|(texture, _)| texture.to_image(&texture_cache.render_state, 0).ok())
        .map(|image| image.thumbnail(64, 64).to_rgba8())
        .and_then(|image| {
            let mut log_sum = 0.0_f64;
            let mut linear_sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for pixel in image.pixels() {
                let alpha = pixel[3] as f64 / 255.0;
                if alpha < 0.02 {
                    continue;
                }
                let linear = |channel: u8| {
                    let value = channel as f64 / 255.0;
                    if value <= 0.04045 {
                        value / 12.92
                    } else {
                        ((value + 0.055) / 1.055).powf(2.4)
                    }
                };
                let luminance = linear(pixel[0]) * 0.2126
                    + linear(pixel[1]) * 0.7152
                    + linear(pixel[2]) * 0.0722;
                log_sum += luminance.max(0.0001).log2() * alpha;
                linear_sum += luminance * alpha;
                weight_sum += alpha;
            }
            (weight_sum > 0.0).then_some(MaterialLuminance {
                log: (log_sum / weight_sum) as f32,
                linear: (linear_sum / weight_sum) as f32,
            })
        });
    if let Ok(mut cache) = MATERIAL_LUMINANCE.lock() {
        if cache.len() >= MAX_MATERIAL_ANALYSIS_CACHE_ENTRIES && !cache.contains_key(&tag) {
            cache.clear();
        }
        cache.insert(tag, value);
    }
    value
}

fn draw_dye_palette(draw: &ModelDraw) -> Option<[[f32; 4]; 3]> {
    draw.gear_dye.map(|dye| [dye.color; 3])
}

fn model_view_depth(position: [f32; 3], center: [f32; 3], yaw: f32, pitch: f32) -> f32 {
    let value = [
        position[0] - center[0],
        position[2] - center[2],
        position[1] - center[1],
    ];
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let yawed_y = value[1];
    let yawed_z = -value[0] * sin_yaw + value[2] * cos_yaw;
    yawed_y * sin_pitch + yawed_z * cos_pitch
}

const HIZ_SIZE: usize = 128;

fn project_hiz_vertex(
    position: [f32; 3],
    center: [f32; 3],
    radius: f32,
    aspect: f32,
    yaw: f32,
    pitch: f32,
    pan: [f32; 2],
    zoom: f32,
) -> [f32; 3] {
    let value = [
        position[0] - center[0],
        position[2] - center[2],
        position[1] - center[1],
    ];
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let yawed = [
        value[0] * cy + value[2] * sy,
        value[1],
        -value[0] * sy + value[2] * cy,
    ];
    let view = [
        yawed[0],
        yawed[1] * cp - yawed[2] * sp,
        yawed[1] * sp + yawed[2] * cp,
    ];
    let scale = 0.84 * zoom / radius.max(0.0001);
    let clip = [-view[0] * scale * aspect + pan[0], view[1] * scale + pan[1]];
    [
        clip[0] * 0.5 + 0.5,
        0.5 - clip[1] * 0.5,
        model_orthographic_depth(view[2], radius),
    ]
}

/// Keep model depth independent from image magnification. Zoom is an
/// orthographic projection scale, not camera dolly: applying it to depth drove
/// vertices into the 0/1 clamps and made surfaces disappear in layers.
fn model_orthographic_depth(view_depth: f32, radius: f32) -> f32 {
    (0.5 - view_depth * 0.21 / radius.max(0.0001)).clamp(0.0, 1.0)
}


#[allow(clippy::too_many_arguments)]
fn hiz_cull_draws(
    preview: &GpuModelPreview,
    draws: &mut Vec<PreparedDraw>,
    center: [f32; 3],
    radius: f32,
    aspect: f32,
    yaw: f32,
    pitch: f32,
    pan: [f32; 2],
    zoom: f32,
) {
    if draws.len() < 2 {
        return;
    }
    let projected = preview
        .vertices
        .iter()
        .map(|vertex| {
            project_hiz_vertex(
                vertex.position,
                center,
                radius,
                aspect,
                yaw,
                pitch,
                pan,
                zoom,
            )
        })
        .collect::<Vec<_>>();
    let mut base = vec![1.0_f32; HIZ_SIZE * HIZ_SIZE];
    for draw in draws
        .iter()
        .filter(|draw| !blend_enabled(draw.pipeline.blend))
    {
        for triangle in preview.indices[draw.indices.start as usize
            ..draw.indices.end.min(preview.indices.len() as u32) as usize]
            .chunks_exact(3)
        {
            let (Some(a), Some(b), Some(c)) = (
                projected.get(triangle[0] as usize),
                projected.get(triangle[1] as usize),
                projected.get(triangle[2] as usize),
            ) else {
                continue;
            };
            rasterize_hiz_triangle(&mut base, *a, *b, *c);
        }
    }
    let mut levels = vec![(HIZ_SIZE, HIZ_SIZE, base)];
    while levels
        .last()
        .is_some_and(|(width, height, _)| *width > 1 || *height > 1)
    {
        let (width, height, source) = levels.last().expect("hiz level");
        let next_width = (*width / 2).max(1);
        let next_height = (*height / 2).max(1);
        let mut next = vec![0.0_f32; next_width * next_height];
        for y in 0..next_height {
            for x in 0..next_width {
                let mut depth = 0.0_f32;
                for oy in 0..2 {
                    for ox in 0..2 {
                        let sx = (x * 2 + ox).min(*width - 1);
                        let sy = (y * 2 + oy).min(*height - 1);
                        // Standard forward-Z: retain the farthest depth. A
                        // range is occluded only when every covered sample is
                        // closer than its nearest point. Background (1.0)
                        // therefore keeps partially uncovered ranges visible.
                        depth = depth.max(source[sy * *width + sx]);
                    }
                }
                next[y * next_width + x] = depth;
            }
        }
        levels.push((next_width, next_height, next));
    }
    draws.retain(|draw| {
        if blend_enabled(draw.pipeline.blend) {
            return true;
        }
        hiz_draw_visible(&levels, &projected, &preview.indices, &draw.indices)
    });
}

fn rasterize_hiz_triangle(depth: &mut [f32], a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
    let to_pixel = |point: [f32; 3]| {
        [
            point[0] * HIZ_SIZE as f32,
            point[1] * HIZ_SIZE as f32,
            point[2],
        ]
    };
    let [a, b, c] = [to_pixel(a), to_pixel(b), to_pixel(c)];
    let edge = |p: [f32; 3], q: [f32; 3], x: f32, y: f32| {
        (x - p[0]) * (q[1] - p[1]) - (y - p[1]) * (q[0] - p[0])
    };
    let area = edge(a, b, c[0], c[1]);
    if area.abs() < 0.00001 {
        return;
    }
    let min_x = a[0]
        .min(b[0])
        .min(c[0])
        .floor()
        .clamp(0.0, (HIZ_SIZE - 1) as f32) as usize;
    let max_x = a[0]
        .max(b[0])
        .max(c[0])
        .ceil()
        .clamp(0.0, (HIZ_SIZE - 1) as f32) as usize;
    let min_y = a[1]
        .min(b[1])
        .min(c[1])
        .floor()
        .clamp(0.0, (HIZ_SIZE - 1) as f32) as usize;
    let max_y = a[1]
        .max(b[1])
        .max(c[1])
        .ceil()
        .clamp(0.0, (HIZ_SIZE - 1) as f32) as usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let w0 = edge(b, c, px, py) / area;
            let w1 = edge(c, a, px, py) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let value = a[2] * w0 + b[2] * w1 + c[2] * w2;
                depth[y * HIZ_SIZE + x] = depth[y * HIZ_SIZE + x].min(value);
            }
        }
    }
}

fn hiz_draw_visible(
    levels: &[(usize, usize, Vec<f32>)],
    projected: &[[f32; 3]],
    indices: &[u32],
    range: &Range<u32>,
) -> bool {
    let mut min = [1.0_f32, 1.0];
    let mut max = [0.0_f32, 0.0];
    let mut nearest = 1.0_f32;
    let mut any = false;
    for index in &indices[range.start as usize..range.end.min(indices.len() as u32) as usize] {
        let Some(point) = projected.get(*index as usize) else {
            continue;
        };
        any = true;
        min[0] = min[0].min(point[0]);
        min[1] = min[1].min(point[1]);
        max[0] = max[0].max(point[0]);
        max[1] = max[1].max(point[1]);
        nearest = nearest.min(point[2]);
    }
    if !any || min[0] < 0.0 || min[1] < 0.0 || max[0] > 1.0 || max[1] > 1.0 {
        return true;
    }
    let extent = ((max[0] - min[0]) * HIZ_SIZE as f32)
        .max((max[1] - min[1]) * HIZ_SIZE as f32)
        .max(1.0);
    let mip = (extent.log2().ceil() as usize)
        .saturating_sub(1)
        .min(levels.len() - 1);
    let (width, height, data) = &levels[mip];
    let x0 = (min[0] * *width as f32)
        .floor()
        .clamp(0.0, (*width - 1) as f32) as usize;
    let y0 = (min[1] * *height as f32)
        .floor()
        .clamp(0.0, (*height - 1) as f32) as usize;
    let x1 = (max[0] * *width as f32)
        .floor()
        .clamp(0.0, (*width - 1) as f32) as usize;
    let y1 = (max[1] * *height as f32)
        .floor()
        .clamp(0.0, (*height - 1) as f32) as usize;
    let mut hzb_depth = 0.0_f32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            hzb_depth = hzb_depth.max(data[y * *width + x]);
        }
    }
    nearest <= hzb_depth + 0.003
}

fn usable_2d_texture(texture: Arc<Texture>) -> Option<Arc<Texture>> {
    (texture.desc.kind() == TextureType::Texture2D).then_some(texture)
}

struct ModelPipelineResources {
    target_format: wgpu::TextureFormat,
    scene_layout: wgpu::BindGroupLayout,
    shadow_scene_layout: wgpu::BindGroupLayout,
    strict_shadow_transform_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    coating_deferred_layout: wgpu::BindGroupLayout,
    present_layout: wgpu::BindGroupLayout,
    bloom_layout: wgpu::BindGroupLayout,
    lighting_layout: wgpu::BindGroupLayout,
    distortion_resolve_layout: wgpu::BindGroupLayout,
    material_sampler: wgpu::Sampler,
    present_sampler: wgpu::Sampler,
    _fallback_color: wgpu::Texture,
    fallback_color_view: wgpu::TextureView,
    model_shader: wgpu::ShaderModule,
    _authored_program_modules: crate::render::authored_program::AuthoredProgramModules,
    authored_surface_vertex_layout: wgpu::BindGroupLayout,
    authored_rigid_vertex_layout: wgpu::BindGroupLayout,
    authored_auxiliary_vertex_layout: wgpu::BindGroupLayout,
    authored_displacement_vertex_layout: wgpu::BindGroupLayout,
    authored_color_vertex_layout: wgpu::BindGroupLayout,
    authored_surface_pipelines: Vec<((crate::render::authored_program::DescriptorAbi, crate::render::authored_program::DescriptorAbi, u8, wgpu::IndexFormat), crate::render::body_draw::BodyDrawPipeline)>,
    authored_source_color: crate::render::surface_targets::SourceColorProjection,
    authored_viewer_material: crate::render::surface_targets::ViewerMaterialProjection,
    viewer_receiver_import: crate::render::surface_targets::ViewerReceiverImport,
    viewer_receiver_coverage: crate::render::surface_targets::ViewerReceiverCoverage,
    authored_compute_pipelines: Vec<(crate::render::authored_program::DescriptorAbi, crate::render::body_mesh::BodyMeshProducer)>,
    authored_c827_pipelines: Vec<(wgpu::IndexFormat,crate::render::c827_draw::C827DrawPipeline)>,
    authored_decal_pipelines: Vec<((crate::render::authored_program::DescriptorAbi, crate::render::authored_program::DescriptorAbi, wgpu::IndexFormat),crate::render::decal_draw::DecalDrawPipeline)>,
    authored_material_c827_pipelines: Vec<(wgpu::IndexFormat,crate::render::c827_draw::C827DrawPipeline)>,
    shadow_shader: wgpu::ShaderModule,
    authored_depth_shader: wgpu::ShaderModule,
    model_pipeline_layout: wgpu::PipelineLayout,
    strict_shadow_pipeline_layout: wgpu::PipelineLayout,
    strict_depth_pipeline_layout: wgpu::PipelineLayout,
    model_pipelines: Vec<(ModelPipelineKey, wgpu::RenderPipeline)>,
    present_pipeline: wgpu::RenderPipeline,
    lighting_pipeline: wgpu::RenderPipeline,
    distortion_resolve_pipeline: wgpu::RenderPipeline,
    bloom_bright_pipeline: wgpu::RenderPipeline,
    bloom_downsample_pipeline: wgpu::RenderPipeline,
    bloom_blur_horizontal_pipeline: wgpu::RenderPipeline,
    bloom_blur_vertical_pipeline: wgpu::RenderPipeline,
    // One depth-only pipeline per authored cull mode. A single back-face
    // pipeline makes two-sided and reversed-winding parts cast incorrectly.
    shadow_pipelines: [wgpu::RenderPipeline; 3],
    strict_shadow_pipelines: Vec<(StrictShadowPipelineKey, wgpu::RenderPipeline)>,
    strict_depth_pipelines: Vec<(StrictDepthPipelineKey, wgpu::RenderPipeline)>,
    depth_pipelines: [wgpu::RenderPipeline; 3],
    shadow_sampler: wgpu::Sampler,
    _fallback_cubemap: wgpu::Texture,
    fallback_cubemap_view: wgpu::TextureView,
    cubemap_sampler: wgpu::Sampler,
}





#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelTargetFeatures {
    native_surface: bool,
    native_decals: bool,
    mixed_receivers: bool,
    deferred_normal_copy: bool,
    distortion: bool,
    bloom: bool,
}

struct ModelTargetResources {
    native_surface: Option<crate::render::surface_targets::SurfaceTargets>,
    native_normal_snapshot: Option<(wgpu::Texture, wgpu::TextureView)>,
    mixed_receivers: Option<crate::render::surface_targets::ReceiverTargets>,
    size: [u32; 2],
    features: ModelTargetFeatures,
    _color: wgpu::Texture,
    color_view: wgpu::TextureView,
    lit_color: wgpu::Texture,
    lit_color_view: wgpu::TextureView,
    scene_color_copy: wgpu::Texture,
    scene_color_copy_view: wgpu::TextureView,
    scene_normal_copy: wgpu::Texture,
    scene_normal_copy_view: wgpu::TextureView,
    _distortion: wgpu::Texture,
    distortion_view: wgpu::TextureView,
    _surface_normal: wgpu::Texture,
    surface_normal_view: wgpu::TextureView,
    _surface_properties: wgpu::Texture,
    surface_properties_view: wgpu::TextureView,
    _surface_emissive: wgpu::Texture,
    surface_emissive_view: wgpu::TextureView,
    _surface_albedo: wgpu::Texture,
    surface_albedo_view: wgpu::TextureView,
    _surface_flags: wgpu::Texture,
    surface_flags_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    _shadow_depth: wgpu::Texture,
    shadow_depth_view: wgpu::TextureView,
    _bloom_half: wgpu::Texture,
    bloom_half_view: wgpu::TextureView,
    _bloom_quarter: wgpu::Texture,
    bloom_quarter_view: wgpu::TextureView,
    _bloom_blur: wgpu::Texture,
    bloom_blur_view: wgpu::TextureView,
}

struct ModelFrameResources {
    key: FrameResourceKey,
    scene_buffer: wgpu::Buffer,
    shadow_scene_bind_group: wgpu::BindGroup,
    scene_bind_group: wgpu::BindGroup,
    _authored_shadow_uniform_buffers: Vec<Option<wgpu::Buffer>>,
    authored_shadow_bind_groups: Vec<Option<wgpu::BindGroup>>,
    _material_buffers: Vec<wgpu::Buffer>,
    material_bind_groups: Vec<wgpu::BindGroup>,
    _coating_deferred_bind_group: wgpu::BindGroup,
    _coating_fallback_bind_group: wgpu::BindGroup,
    bloom_half_bind_group: wgpu::BindGroup,
    bloom_quarter_bind_group: wgpu::BindGroup,
    bloom_blur_horizontal_bind_group: wgpu::BindGroup,
    bloom_blur_vertical_bind_group: wgpu::BindGroup,
    present_bind_group: wgpu::BindGroup,
    lighting_bind_group: wgpu::BindGroup,
    distortion_resolve_bind_group: wgpu::BindGroup,
    opaque_bundles: Vec<wgpu::RenderBundle>,
    decal_runs: Vec<ModelDecalRun>,
    native_view_buffer: Option<wgpu::Buffer>,
    native_pixel_view_buffer: Option<wgpu::Buffer>,
    native_surfaces: Vec<(usize, crate::render::authored_program::DescriptorAbi, crate::render::body_draw::BodyDrawBindings)>,
    native_source_color: Option<wgpu::BindGroup>,
    native_viewer_material: Option<wgpu::BindGroup>,
    viewer_receiver_import: Option<wgpu::BindGroup>,
    viewer_receiver_coverage: Option<wgpu::BindGroup>,
    native_skinning_buffers: Vec<(usize, wgpu::Buffer)>,
    native_surface_constants: Vec<(usize, wgpu::Buffer)>,
    native_decal_constants: Vec<(usize, wgpu::Buffer)>,
    native_compute: Vec<((usize, usize), wgpu::BindGroup)>,
    native_c827: Vec<(usize, crate::render::c827_draw::C827DrawBindings)>,
    native_decals: Vec<(usize, NativeDecalBindings)>,
    additive_bundles: Vec<wgpu::RenderBundle>,
    transparent_bundles: Vec<wgpu::RenderBundle>,
    coating_bundles: Vec<wgpu::RenderBundle>,
    distortion_bundles: Vec<wgpu::RenderBundle>,
}

enum ModelDecalRun {
    Compatibility(Vec<wgpu::RenderBundle>),
    C827(usize),
    Authored(usize),
}

struct NativeDecalBindings {
    vertex_program: crate::render::authored_program::DescriptorAbi,
    program: crate::render::authored_program::DescriptorAbi,
    pixel: wgpu::BindGroup,
    vertex: wgpu::BindGroup,
    source: wgpu::Buffer,
    uv: wgpu::Buffer,
    indices: wgpu::Buffer,
    range: Range<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameResourceKey {
    preview: usize,
    size: [u32; 2],
    materials: u64,
    cubemap: usize,
    decal: usize,
}

fn create_model_render_bundles(
    device: &wgpu::Device,
    pipelines: &ModelPipelineResources,
    scene_bind_group: &wgpu::BindGroup,
    material_bind_groups: &[wgpu::BindGroup],
    coating_deferred_bind_group: &wgpu::BindGroup,
    preview: &GpuModelPreview,
    draws: &[PreparedDraw],
    accepted_passes: &[RenderPassKind],
) -> Vec<wgpu::RenderBundle> {
    draws
        .par_iter()
        .filter(|draw| {
            !draw.native_c827 && !draw.native_surface && !draw.native_decal && draw.passes
                .iter()
                .any(|pass| accepted_passes.contains(pass))
        })
        .filter_map(|draw| {
            let pipeline_key = draw.pipeline;
            let (_, pipeline) = pipelines
                .model_pipelines
                .iter()
                .find(|(key, _)| key.gpu_equivalent(pipeline_key))?;
            let material = material_bind_groups.get(draw.material_index)?;
            let distortion = accepted_passes
                .iter()
                .copied()
                .all(is_distortion_payload_pass);
            let transparent = accepted_passes.iter().copied().all(is_forward_pass);
            let depth_read_only = accepted_passes == [RenderPassKind::ForwardCoating];
            let color_formats = if distortion {
                vec![Some(DISTORTION_FORMAT)]
            } else if transparent {
                vec![Some(OFFSCREEN_FORMAT)]
            } else {
                vec![
                    Some(OFFSCREEN_FORMAT),
                    Some(SURFACE_FORMAT),
                    Some(SURFACE_PROPERTIES_FORMAT),
                    Some(SURFACE_FORMAT),
                ]
            };
            let mut bundle =
                device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some("quicktag_model_parallel_draw"),
                    color_formats: &color_formats,
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: DEPTH_FORMAT,
                        depth_read_only,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                });
            bundle.set_pipeline(pipeline);
            bundle.set_bind_group(0, scene_bind_group, &[]);
            bundle.set_bind_group(1, material, &[]);
            bundle.set_bind_group(2, coating_deferred_bind_group, &[]);
            bundle.set_vertex_buffer(0, preview.vertex_buffer.slice(..));
            bundle.set_index_buffer(preview.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            bundle.draw_indexed(draw.indices.clone(), 0, 0..1);
            Some(bundle.finish(&wgpu::RenderBundleDescriptor {
                label: Some("quicktag_model_parallel_draw"),
            }))
        })
        .collect()
}

impl ModelPaintCallback {
    fn target_features(&self) -> ModelTargetFeatures {
        ModelTargetFeatures {
            native_surface: self.draws.iter().any(|draw| draw.native_surface || draw.native_decal),
            native_decals: self.draws.iter().any(|draw| draw.native_decal),
            mixed_receivers: self.draws.iter().any(|draw| draw.native_decal || draw.native_c827)
                && self.draws.iter().any(|draw| draw.native_surface || draw.native_decal)
                && self.draws.iter().any(is_generic_opaque_receiver),
            deferred_normal_copy: self.draws.iter().any(|draw| {
                draw.passes.iter().any(|pass| {
                    matches!(
                        pass,
                        RenderPassKind::InvestmentDecalCompatibility
                            | RenderPassKind::ForwardCoating
                    )
                })
            }),
            distortion: self
                .draws
                .iter()
                .any(|draw| draw.passes.contains(&RenderPassKind::Distortion)),
            bloom: self.scene.postprocess0[1].abs() > 0.0001,
        }
    }
}

impl CallbackTrait for ModelPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Never change an admitted shader into a generic material while its
        // resources load. The next UI callback retries the same source contract.
        if self.native_resources_pending { return Vec::new(); }
        let needs_pipeline = callback_resources
            .get::<ModelPipelineResources>()
            .is_none_or(|resources| resources.target_format != self.target_format);
        if needs_pipeline {
            callback_resources.insert(create_pipeline_resources(
                device,
                _queue,
                self.target_format,
            ));
        }

        let strict_shadow_specs = self
            .shadow_draws
            .iter()
            .filter_map(|draw| {
                if !draw.authored_native_vertex_supported {
                    return None;
                }
                let source_index = draw.authored_source?;
                let authored = draw.authored_stage?;
                let source = self.preview.authored_inputs.get(source_index)?.as_ref()?;
                let layout = source
                    .stage_layout(crate::render::adapter::GoliathAdapter::AUTHORED_SHADOW_STAGE)?;
                if layout.layout_id != authored.input_layout_id {
                    return None;
                }
                let descriptor = layout.descriptor.clone()?;
                let stream_strides = authored_stream_strides(source, &descriptor)?;
                let index = source.index_buffer.as_ref()?;
                let end = authored
                    .source_index_start
                    .checked_add(authored.source_index_count)?;
                if end > index.index_count {
                    return None;
                }
                Some((
                    StrictShadowPipelineKey {
                        layout_id: layout.layout_id,
                        primitive_type: authored.primitive_type,
                        rasterizer: draw.pipeline.rasterizer,
                        index_32bit: matches!(index.format, wgpu::IndexFormat::Uint32),
                        stream_strides,
                    },
                    descriptor,
                ))
            })
            .unique_by(|(key, _descriptor)| *key)
            .collect_vec();

        let strict_depth_specs = self
            .draws
            .iter()
            .filter(|draw| draw.passes.contains(&RenderPassKind::DepthOnly))
            .filter_map(|draw| {
                if !draw.authored_native_vertex_supported {
                    return None;
                }
                let source_index = draw.authored_source?;
                let authored = draw.authored_stage?;
                let source = self.preview.authored_inputs.get(source_index)?.as_ref()?;
                let layout = source.stage_layout(authored.raw_stage)?;
                if layout.layout_id != authored.input_layout_id {
                    return None;
                }
                let descriptor = layout.descriptor.clone()?;
                let stream_strides = authored_stream_strides(source, &descriptor)?;
                let index = source.index_buffer.as_ref()?;
                let end = authored
                    .source_index_start
                    .checked_add(authored.source_index_count)?;
                if end > index.index_count {
                    return None;
                }
                Some((
                    StrictDepthPipelineKey {
                        layout_id: layout.layout_id,
                        primitive_type: authored.primitive_type,
                        rasterizer: draw.pipeline.rasterizer,
                        depth_stencil: draw.pipeline.depth_stencil,
                        depth_bias: draw.pipeline.depth_bias,
                        index_32bit: matches!(index.format, wgpu::IndexFormat::Uint32),
                        stream_strides,
                    },
                    descriptor,
                ))
            })
            .unique_by(|(key, _descriptor)| *key)
            .collect_vec();

        if let Some(resources) = callback_resources.get_mut::<ModelPipelineResources>() {
            for mesh in self
                .preview
                .authored_inputs
                .iter()
                .flatten()
                .flat_map(|source| source.static_meshes.iter())
            {
                if !resources
                    .authored_compute_pipelines
                    .iter()
                    .any(|(abi, _)| *abi == mesh.producer)
                {
                    let shader = resources
                        ._authored_program_modules
                        .module(mesh.producer)
                        .expect("admitted compute source");
                    resources.authored_compute_pipelines.push((
                        mesh.producer,
                        crate::render::body_mesh::BodyMeshProducer::new_for_abi(
                            device,
                            shader,
                            mesh.producer,
                        ),
                    ));
                }
            }
            for format in self.draws.iter().filter(|d|d.native_c827).map(|d|self.preview.authored_index_format(d.stable_index)).unique() {
                if !resources.authored_c827_pipelines.iter().any(|(f,_)|*f==format) {
                    resources.authored_c827_pipelines.push((format,create_authored_c827_pipeline(device,&resources._authored_program_modules,format)));
                }
                if !self.target_features().native_surface || resources.authored_material_c827_pipelines.iter().any(|(f,_)|*f==format) {continue;}
                use crate::render::authored_program::DescriptorAbi as D;
                resources.authored_material_c827_pipelines.push((format,crate::render::c827_draw::C827DrawPipeline::new_material(
                    device,resources._authored_program_modules.module(D::SharedLayout7VertexScalarStorage).unwrap(),
                    resources._authored_program_modules.module(D::C827Pixel).unwrap(),[wgpu::TextureFormat::Rgba32Float;4],
                    depth_stencil_state(15,1),rasterizer_cull_mode(2),format)));
            }
            for loaded in &self.native_decals {
                let spec=self.preview.draws[loaded.stable_index].native_decal.as_ref().unwrap();
                let format=self.preview.authored_index_format(loaded.stable_index);
                if resources.authored_decal_pipelines.iter().any(|(abi,_)|*abi==(spec.vertex_program,spec.program,format)) {continue;}
                let pipeline=crate::render::decal_draw::DecalDrawPipeline::new(
                    device, resources._authored_program_modules.module(spec.vertex_program).unwrap(),
                    resources._authored_program_modules.module(spec.program).unwrap(),
                    &resources.authored_surface_vertex_layout,model_decals::contract(spec.program).unwrap(),[wgpu::TextureFormat::Rgba32Float;4],
                    wgpu::FrontFace::Ccw,rasterizer_cull_mode(2),depth_stencil_state(15,1),format);
                resources.authored_decal_pipelines.push(((spec.vertex_program,spec.program,format),pipeline));
            }
            let missing_surface_keys = self.native_surfaces.iter().filter_map(|loaded| {
                let draw = &self.preview.draws[loaded.stable_index];
                let spec = draw.native_surface.as_ref()?;
                let key = (spec.vertex_program, spec.program, draw.pipeline.rasterizer,self.preview.authored_index_format(loaded.stable_index));
                (!resources.authored_surface_pipelines.iter().any(|(existing, _)| *existing == key)).then_some(key)
            }).unique().collect_vec();
            if !missing_surface_keys.is_empty() {
                let pipelines = create_authored_surface_pipelines(device, &resources._authored_program_modules,
                    &resources.authored_surface_vertex_layout, &resources.authored_rigid_vertex_layout,
                    &resources.authored_auxiliary_vertex_layout, &resources.authored_color_vertex_layout,
                    &resources.authored_displacement_vertex_layout,
                    &missing_surface_keys);
                assert_eq!(pipelines.len(), missing_surface_keys.len(), "every admitted material needs its exact cached pipeline");
                resources.authored_surface_pipelines.extend(pipelines);
            }
            let keys = self
                .draws
                .iter()
                .filter(|draw| !draw.native_c827 && !draw.native_surface && !draw.native_decal)
                .flat_map(|draw| {
                    [
                        Some(draw.pipeline),
                        (!draw.passes.iter().copied().any(is_forward_pass))
                            .then_some(draw.pipeline.material_flags()),
                        (!draw.passes.iter().copied().any(is_forward_pass))
                            .then_some(draw.pipeline.material_emissive()),
                    ]
                    .into_iter()
                    .flatten()
                })
                .unique()
                .collect_vec();
            for key in keys {
                if resources
                    .model_pipelines
                    .iter()
                    .any(|(existing, _pipeline)| existing.gpu_equivalent(key))
                {
                    continue;
                }
                let pipeline = create_model_pipeline(
                    device,
                    &resources.model_shader,
                    &resources.model_pipeline_layout,
                    key,
                );
                resources.model_pipelines.push((key, pipeline));
            }

            for (key, descriptor) in &strict_shadow_specs {
                if resources
                    .strict_shadow_pipelines
                    .iter()
                    .any(|(existing, _pipeline)| existing == key)
                {
                    continue;
                }
                if let Some(pipeline) = create_strict_shadow_pipeline(
                    device,
                    &resources.shadow_shader,
                    &resources.strict_shadow_pipeline_layout,
                    &descriptor,
                    *key,
                ) {
                    resources.strict_shadow_pipelines.push((*key, pipeline));
                }
            }

            for (key, descriptor) in &strict_depth_specs {
                if resources
                    .strict_depth_pipelines
                    .iter()
                    .any(|(existing, _pipeline)| existing == key)
                {
                    continue;
                }
                if let Some(pipeline) = create_strict_depth_pipeline(
                    device,
                    &resources.authored_depth_shader,
                    &resources.strict_depth_pipeline_layout,
                    descriptor,
                    *key,
                ) {
                    resources.strict_depth_pipelines.push((*key, pipeline));
                }
            }
        }

        let Some((
            scene_layout,
            shadow_scene_layout,
            strict_shadow_transform_layout,
            material_layout,
            present_layout,
            bloom_layout,
            lighting_layout,
            distortion_resolve_layout,
            material_sampler,
            present_sampler,
            fallback_color_view,
            shadow_sampler,
            fallback_cubemap_view,
            cubemap_sampler,
        )) = callback_resources
            .get::<ModelPipelineResources>()
            .map(|resources| {
                (
                    resources.scene_layout.clone(),
                    resources.shadow_scene_layout.clone(),
                    resources.strict_shadow_transform_layout.clone(),
                    resources.material_layout.clone(),
                    resources.present_layout.clone(),
                    resources.bloom_layout.clone(),
                    resources.lighting_layout.clone(),
                    resources.distortion_resolve_layout.clone(),
                    resources.material_sampler.clone(),
                    resources.present_sampler.clone(),
                    resources.fallback_color_view.clone(),
                    resources.shadow_sampler.clone(),
                    resources.fallback_cubemap_view.clone(),
                    resources.cubemap_sampler.clone(),
                )
            })
        else {
            return Vec::new();
        };

        let target_features = self.target_features();
        let needs_target =
            callback_resources
                .get::<ModelTargetResources>()
                .is_none_or(|resources| {
                    resources.size != self.target_size
                        || resources.features != target_features
                        || needs_pipeline
                });
        if needs_target {
            callback_resources.insert(create_target_resources(
                device,
                self.target_size,
                target_features,
            ));
        }

        let frame_key = self.frame_resource_key();
        let reuse_frame = !needs_pipeline
            && !needs_target
            && callback_resources
                .get::<ModelFrameResources>()
                .is_some_and(|frame| frame.key == frame_key);
        if reuse_frame {
            if let Some(frame) = callback_resources.get::<ModelFrameResources>() {
                _queue.write_buffer(&frame.scene_buffer, 0, bytemuck::bytes_of(&self.scene));
                if let Some(view)=&frame.native_view_buffer { _queue.write_buffer(view, 0, bytemuck::cast_slice(&native_view_image(&self.scene))); }
                if let Some(view)=&frame.native_pixel_view_buffer { _queue.write_buffer(view, 0, bytemuck::cast_slice(&model_surfaces::pixel_view_image(&self.scene))); }
                for (index,buffer) in &frame.native_skinning_buffers {
                    let source_index=self.preview.draws[*index].authored_source.expect("retained native draw source");
                    let source=self.preview.authored_inputs[source_index].as_ref().expect("retained native source");
                    let object=if let Some(surface)=&self.preview.draws[*index].native_surface {
                        native_surface_object_image(surface.vertex_program,source,&self.scene)
                    } else {c827_static_skinning(source,&self.scene)};
                    _queue.write_buffer(buffer,0,bytemuck::cast_slice(&object));
                }
                // Re-run every time-dependent TFX program at the viewer clock.
                let time=self.scene.fidelity[2];
                for (index,buffer) in &frame.native_surface_constants {
                    if let Some(constants)=self.preview.draws[*index].native_surface.as_ref()
                        .and_then(|surface|surface.animation.as_ref()).and_then(|animation|animation.at(time)) {
                        _queue.write_buffer(buffer,0,bytemuck::cast_slice(&constants));
                    }
                }
                for (index,buffer) in &frame.native_decal_constants {
                    if let Some(decal)=self.preview.draws[*index].native_decal.as_ref().filter(|decal|decal.animation.is_some()) {
                        _queue.write_buffer(buffer,0,bytemuck::cast_slice(&decal.globals(self.target_size,time)));
                    }
                }
            }
        } else {
            let native_view_buffer=self.draws.iter().any(|d| d.native_c827 || d.native_surface || d.native_decal).then(|| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer native VS View"), contents: bytemuck::cast_slice(&native_view_image(&self.scene)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }));
            let mut native_c827=Vec::new();
            let mut native_compute=Vec::new();
            let mut native_skinning_buffers=Vec::new();
            let mut native_surface_constants=Vec::new();
            let mut native_decal_constants=Vec::new();
            let native_pixel_view_buffer=self.draws.iter().any(|d| d.native_surface).then(|| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:Some("viewer authored PS View"),contents:bytemuck::cast_slice(&model_surfaces::pixel_view_image(&self.scene)),
                usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,
            }));
            let mut native_surfaces=Vec::new();
            let native_pipelines=callback_resources.get::<ModelPipelineResources>().expect("native pipeline cache");
            let native_source_color=callback_resources.get::<ModelTargetResources>().and_then(|target|
                target.native_surface.as_ref().map(|native|native_pipelines.authored_source_color.bind(device,native,
                    target.mixed_receivers.as_ref().map(|r|&r.coverage_view))));
            for loaded in &self.native_surfaces {
                let draw=&self.preview.draws[loaded.stable_index];
                let spec=draw.native_surface.as_ref().expect("audited surface source");
                let source_index=draw.authored_source.expect("native surface source");
                let source=self.preview.authored_inputs[source_index].as_ref().expect("retained raw source");
                let direct_ia=spec.vertex_program == crate::render::authored_program::DescriptorAbi::RigidVertexDirectIa;
                let mesh=if direct_ia { None } else {
                    let (mesh_index, mesh)=source.static_mesh(draw.authored_stage.expect("authored producer range")).expect("retained exact producer");
                    if !native_compute.iter().any(|(i,_)|*i==(source_index, mesh_index)) {
                        native_compute.push(((source_index, mesh_index),mesh.bind(device,&native_pipelines.authored_compute_pipelines.iter().find(|(abi,_)|*abi==mesh.producer).expect("exact compute pipeline").1)));
                    }
                    Some(mesh)
                };
                let skin=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:Some("viewer surface object scope"),contents:bytemuck::cast_slice(&native_surface_object_image(spec.vertex_program,source,&self.scene)),
                    usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST
                        | if spec.vertex_program == crate::render::authored_program::DescriptorAbi::VertexColorStorage {
                            wgpu::BufferUsages::COPY_SRC
                        } else { wgpu::BufferUsages::empty() },
                });
                let constants=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:Some("authored surface cb0"),contents:bytemuck::cast_slice(spec.animation.as_ref().and_then(|animation|animation.at(self.scene.fidelity[2])).as_ref().unwrap_or(&spec.constants)),usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,
                });
                let views=loaded.textures.iter().enumerate().map(|(slot,t)|match &t.full_cubemap_texture {
                    Some(cube) if spec.cube_slot == Some(slot as u32) => create_model_texture_view(cube,&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::Cube), array_layer_count: Some(6), ..Default::default()
                    }),
                    _ => create_model_texture_view(&t.handle,&Default::default()),
                }).collect_vec();
                let samplers=spec.samplers.iter().map(|desc|create_native_authored_sampler(device,desc)).collect_vec();
                let vertex=if spec.vertex_program == crate::render::authored_program::DescriptorAbi::DisplacementEC03VertexStorage {
                    let mesh = mesh.expect("exact EC03 static producer");
                    assert_eq!(mesh.producer, spec.vertex_program.static_vertex_producer().unwrap());
                    let cb0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("authored EC03 displacement image"),
                        contents: bytemuck::cast_slice(spec.vertex_constants.as_ref().expect("resolved EC03 CB0")),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    crate::render::displacement_vertex::bind(device, &native_pipelines.authored_displacement_vertex_layout,
                        crate::render::body_vertex::BodyVertexInputs {
                            skinning: &skin, view: native_view_buffer.as_ref().unwrap(),
                            positions: &mesh.positions, frames: &mesh.frames, previous_positions: &mesh.positions,
                        }, &cb0)
                } else if let Some((_,_,producer)) = spec.vertex_program.auxiliary_vertex_contract() {
                    let mesh = mesh.expect("exact static auxiliary-vertex producer");
                    assert_eq!(mesh.producer, producer);
                    let vertex_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("authored auxiliary vertex cb0"),
                        contents: bytemuck::cast_slice(spec.vertex_constants.as_ref().expect("audited auxiliary vertex image")),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    crate::render::auxiliary_vertex::bind(device, &native_pipelines.authored_auxiliary_vertex_layout,
                        [mesh.inert_vertex_auxiliary(), &mesh.positions, &mesh.frames, &mesh.positions,
                         &vertex_constants, native_view_buffer.as_ref().unwrap(), &skin])
                } else if spec.vertex_program == crate::render::authored_program::DescriptorAbi::VertexColorStorage {
                    let mesh = mesh.expect("source-paired vertex-color producer");
                    let colors = source.vertex_colors.as_ref().expect("actual authored colors");
                    crate::render::color_vertex::bind(device, &native_pipelines.authored_color_vertex_layout,
                        [&colors.buffer, &mesh.positions, &mesh.frames, &mesh.positions,
                         native_view_buffer.as_ref().unwrap(), &skin])
                } else if let Some(mesh)=mesh {
                    crate::render::body_vertex::bind(device,&native_pipelines.authored_surface_vertex_layout,
                        crate::render::body_vertex::BodyVertexInputs { skinning:&skin,view:native_view_buffer.as_ref().unwrap(),
                            positions:&mesh.positions,frames:&mesh.frames,previous_positions:&mesh.positions })
                } else {
                    crate::render::rigid_vertex::bind(device,&native_pipelines.authored_rigid_vertex_layout,
                        &skin,native_view_buffer.as_ref().unwrap())
                };
                let pipeline=&native_pipelines.authored_surface_pipelines.iter().find(|(key,_)|*key==(spec.vertex_program,spec.program,draw.pipeline.rasterizer,self.preview.authored_index_format(loaded.stable_index))).expect("exact surface pipeline").1;
                let stream=|index|&source.vertex_streams.iter().find(|s|s.stream_index==index).expect("raw surface IA").buffer;
                let authored=draw.authored_stage.unwrap();
                let index=source.index_buffer.as_ref().unwrap();
                let range=authored.source_index_start..authored.source_index_start.checked_add(authored.source_index_count).expect("surface strip overflow");
                assert!(range.end<=index.index_count);
                let bindings=pipeline.bind_dense(device,[&constants,&skin,native_pixel_view_buffer.as_ref().unwrap()],
                    &views.iter().collect_vec(),samplers.iter().collect_vec(),&vertex,stream(0),stream(1),&index.buffer,range);
                native_skinning_buffers.push((loaded.stable_index,skin));
                native_surface_constants.push((loaded.stable_index,constants));
                native_surfaces.push((loaded.stable_index,spec.program,bindings));
            }
            for draw in self.draws.iter().filter(|d| d.native_c827) {
                let source_index=draw.authored_source.expect("native draw source");
                let source=self.preview.authored_inputs[source_index].as_ref().expect("native raw source");
                let (mesh_index, mesh)=source.static_mesh(draw.authored_stage.expect("authored producer range")).expect("retained exact producer");
                if !native_compute.iter().any(|(i,_)| *i==(source_index, mesh_index)) {
                    native_compute.push(((source_index, mesh_index),mesh.bind(device,&native_pipelines.authored_compute_pipelines.iter().find(|(abi,_)|*abi==mesh.producer).expect("exact compute pipeline").1)));
                }
                let skin=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("viewer native Skinning prefix"), contents: bytemuck::cast_slice(&c827_static_skinning(source,&self.scene)),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let pixel=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("C827 authored external pixel constants"),
                    contents: bytemuck::cast_slice(self.preview.draws[draw.stable_index].native_c827.as_ref().expect("native pixel constants")),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let pipeline=&native_pipelines.authored_c827_pipelines.iter().find(|(f,_)|*f==self.preview.authored_index_format(draw.stable_index)).expect("exact C827 index format").1;
                let pixel_group=device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:Some("C827 retained pixel inputs"), layout:&pipeline.pixel_layout,
                    entries:&[wgpu::BindGroupEntry { binding:0,resource:pixel.as_entire_binding() }],
                });
                let vertex_group=crate::render::body_vertex::bind(device,&pipeline.vertex_layout,crate::render::body_vertex::BodyVertexInputs {
                    skinning:&skin,view:native_view_buffer.as_ref().expect("native camera buffer"),positions:&mesh.positions,frames:&mesh.frames,previous_positions:&mesh.positions,
                });
                native_skinning_buffers.push((draw.stable_index,skin));
                let stream=|index| source.vertex_streams.iter().find(|s| s.stream_index==index).expect("native IA stream").buffer.clone();
                let authored=draw.authored_stage.expect("native original range");
                let index=source.index_buffer.as_ref().expect("native strip indices");
                let end=authored.source_index_start.checked_add(authored.source_index_count).expect("C827 source range overflow");
                assert!(end<=index.index_count);
                native_c827.push((draw.stable_index,crate::render::c827_draw::C827DrawBindings {
                    pixel:pixel_group, vertex:vertex_group, source:stream(0), uv:stream(1),indices:index.buffer.clone(),
                    range:authored.source_index_start..end,
                }));
            }
            let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quicktag_model_scene_uniform"),
                contents: bytemuck::bytes_of(&self.scene),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let native_viewer_material=callback_resources.get::<ModelTargetResources>().and_then(|target|
                target.native_surface.as_ref().map(|native|native_pipelines.authored_viewer_material.bind(device,native,&scene_buffer,
                    target.mixed_receivers.as_ref().map(|r|&r.coverage_view))));
            let viewer_receiver_import=callback_resources.get::<ModelTargetResources>().and_then(|target|
                target.mixed_receivers.as_ref().map(|receiver|native_pipelines.viewer_receiver_import.bind(device,
                    &target.surface_normal_view,&target.surface_properties_view,&target.surface_albedo_view,&target.depth_view,
                    &receiver.original_native_normal_view,&scene_buffer)));
            let viewer_receiver_coverage=callback_resources.get::<ModelTargetResources>().and_then(|target|
                target.mixed_receivers.as_ref().map(|receiver|native_pipelines.viewer_receiver_coverage.bind_targets(device,
                    target.native_surface.as_ref().expect("mixed receivers require authored targets"),receiver)));
            let mut native_decals=Vec::new();
            if !self.native_decals.is_empty() {
                let normal=&callback_resources.get::<ModelTargetResources>().unwrap().native_normal_snapshot.as_ref().unwrap().1;
                let view=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:Some("viewer Decal viewport image"),contents:bytemuck::cast_slice(&model_decals::view(self.target_size)),
                    usage:wgpu::BufferUsages::UNIFORM,
                });
                for loaded in &self.native_decals {
                    let draw=&self.preview.draws[loaded.stable_index];
                    let source_index=draw.authored_source.unwrap();
                    let source=self.preview.authored_inputs[source_index].as_ref().unwrap();
                    let (mesh_index, mesh)=source.static_mesh(draw.authored_stage.expect("authored producer range")).expect("retained exact producer");
                    if !native_compute.iter().any(|(i,_)|*i==(source_index, mesh_index)) {
                        native_compute.push(((source_index, mesh_index),mesh.bind(device,&native_pipelines.authored_compute_pipelines.iter().find(|(abi,_)|*abi==mesh.producer).expect("exact compute pipeline").1)));
                    }
                    let spec=draw.native_decal.as_ref().unwrap();
                    let pipeline=&native_pipelines.authored_decal_pipelines.iter().find(|(abi,_)|*abi==(spec.vertex_program,spec.program,self.preview.authored_index_format(loaded.stable_index))).unwrap().1;
                    let globals=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label:Some("authored Decal constants and viewer texel transform"),contents:bytemuck::cast_slice(&spec.globals(self.target_size,self.scene.fidelity[2])),
                        usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,
                    });
                    let skin=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label:Some("viewer native Decal Skinning image"),contents:bytemuck::cast_slice(&c827_static_skinning(source,&self.scene)),
                        usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,
                    });
                    let vertex=crate::render::body_vertex::bind(device,&native_pipelines.authored_surface_vertex_layout,
                        crate::render::body_vertex::BodyVertexInputs {skinning:&skin,view:native_view_buffer.as_ref().unwrap(),
                            positions:&mesh.positions,frames:&mesh.frames,previous_positions:&mesh.positions});
                    let views=loaded.textures.iter().map(|t|create_model_texture_view(&t.handle,
                        &wgpu::TextureViewDescriptor {format:Some(t.desc.format),dimension:Some(wgpu::TextureViewDimension::D2),..Default::default()})).collect_vec();
                    let samplers=spec.samplers.iter().map(|desc|create_native_authored_sampler(device,desc)).collect_vec();
                    let pixel=pipeline.bind(device,crate::render::decal_draw::DecalPixelInputs {
                        globals:&globals,view:&view,scene_normal:normal,textures:&views.iter().collect_vec(),samplers:&samplers.iter().collect_vec(),
                    });
                    let stream=|index|source.vertex_streams.iter().find(|s|s.stream_index==index).unwrap().buffer.clone();
                    let authored=draw.authored_stage.unwrap();
                    let index=source.index_buffer.as_ref().unwrap();
                    let range=authored.source_index_start..authored.source_index_start.checked_add(authored.source_index_count).unwrap();
                    assert!(range.end<=index.index_count);
                    native_skinning_buffers.push((loaded.stable_index,skin));
                    native_decal_constants.push((loaded.stable_index,globals));
                    native_decals.push((loaded.stable_index,NativeDecalBindings {vertex_program:spec.vertex_program,program:spec.program,pixel,vertex,source:stream(0),uv:stream(1),
                        indices:index.buffer.clone(),range,
                    }));
                }
            }
            let Some(shadow_depth_view) = callback_resources
                .get::<ModelTargetResources>()
                .map(|target| target.shadow_depth_view.clone())
            else {
                return Vec::new();
            };
            let cubemap_view = self
                .cubemap
                .as_ref()
                .and_then(|texture| {
                    texture.full_cubemap_texture.as_ref().map(|handle| {
                        create_model_texture_view(
                            &handle,
                            &wgpu::TextureViewDescriptor {
                                format: Some(srgb_texture_format(texture.desc.format)),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                array_layer_count: Some(6),
                                ..Default::default()
                            },
                        )
                    })
                })
                .unwrap_or_else(|| fallback_cubemap_view.clone());
            let scene_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_scene_bind_group"),
                    layout: &scene_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: scene_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&shadow_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&cubemap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&cubemap_sampler),
                        },
                    ],
                },
            );
            let shadow_scene_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_shadow_scene_bind_group"),
                    layout: &shadow_scene_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scene_buffer.as_entire_binding(),
                    }],
                },
            );
            let mut authored_shadow_uniform_buffers =
                Vec::with_capacity(self.preview.authored_inputs.len());
            let mut authored_shadow_bind_groups =
                Vec::with_capacity(self.preview.authored_inputs.len());
            for (source_index, source) in self.preview.authored_inputs.iter().enumerate() {
                let Some(source) = source.as_ref().filter(|source| {
                    !source.vertex_streams.is_empty() && source.index_buffer.is_some()
                }) else {
                    authored_shadow_uniform_buffers.push(None);
                    authored_shadow_bind_groups.push(None);
                    continue;
                };
                let uniform = source.shadow_uniform();
                let label = format!("quicktag_authored_shadow_uniform_{source_index}");
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&label),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind_group = create_model_bind_group(
                    device,
                    &wgpu::BindGroupDescriptor {
                        label: Some(&format!(
                            "quicktag_authored_shadow_bind_group_{source_index}"
                        )),
                        layout: &strict_shadow_transform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        }],
                    },
                );
                authored_shadow_uniform_buffers.push(Some(buffer));
                authored_shadow_bind_groups.push(Some(bind_group));
            }

            let mut material_buffers = Vec::with_capacity(self.materials.len());
            let mut material_bind_groups = Vec::with_capacity(self.materials.len());
            for material in &self.materials {
                let tint = material
                    .key
                    .map(|key| key.color_tint.map(|value| value as f32 / 255.0))
                    .or(material.solid_color)
                    .unwrap_or([0.72, 0.75, 0.8, 1.0]);
                let uniform = MaterialUniform {
                    tint,
                    params: [
                        (material.normal.is_some() && material.investment_decal.is_none()) as u8
                            as f32,
                        material.emissive.is_some() as u8 as f32,
                        material
                            .key
                            .map(|key| key.emissive_strength as f32 / 255.0)
                            .unwrap_or(0.0),
                        material
                            .color
                            .as_ref()
                            .map(|texture| alpha_mode(texture.desc.format))
                            .unwrap_or(0.0),
                    ],
                    blend: [
                        material.blend as f32,
                        material.roughness_channel as f32,
                        (material.control.is_some() && material.alpha_mask.is_none()) as u8 as f32,
                        material.mask_palette.is_some() as u8 as f32,
                    ],
                    solid_surface: material
                        .solid_surface
                        .map(|surface| {
                            [
                                surface[0],
                                surface[1],
                                1.0,
                                material.iridescence_id.unwrap_or(0.0),
                            ]
                        })
                        .unwrap_or([0.0, 0.0, 0.0, material.iridescence_id.unwrap_or(0.0)]),
                    palette_base: material
                        .dye_palette
                        .map(|palette| palette[0])
                        .or_else(|| material.mask_palette.map(|palette| palette[0]))
                        .unwrap_or_default(),
                    palette_delta: material
                        .dye_palette
                        .map(|palette| palette[1])
                        .or_else(|| material.mask_palette.map(|palette| palette[1]))
                        .unwrap_or_default(),
                    palette_tertiary: material
                        .dye_palette
                        .map(|palette| palette[2])
                        .unwrap_or_default(),
                    roughness_remap: material
                        .gear_dye
                        .map(|dye| dye.roughness_remap)
                        .unwrap_or([1.0, 0.0, 0.0, 0.0]),
                    metal_remap: material
                        .gear_dye
                        .map(|dye| dye.metal_remap)
                        .unwrap_or([0.0, 0.0, 0.0, 0.0]),
                    wear_params: material
                        .mod_wear
                        .and_then(|wear| {
                            let rarity = wear.rarity?;
                            let controls = wear.condition_controls
                                [usize::from(rarity.tier().saturating_sub(1))];
                            Some([
                                controls[0],
                                controls[1],
                                controls[2],
                                (material.wear_scratches.is_some()
                                    && material.wear_grime.is_some()
                                    && material.wear_damage.is_some())
                                    as u8 as f32,
                            ])
                        })
                        .or_else(|| material.surface_condition.map(|_| [0.0, 0.0, 0.0, 2.0]))
                        .unwrap_or_default(),
                    wear_scratches_projection: material
                        .mod_wear
                        .map(|wear| wear.scratches_projection)
                        .or_else(|| {
                            material.surface_condition.map(|condition| {
                                [
                                    condition.phase,
                                    condition.orientation[0],
                                    condition.orientation[1],
                                    condition.normal_flatten,
                                ]
                            })
                        })
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    wear_grime_projection: material
                        .mod_wear
                        .map(|wear| {
                            std::array::from_fn(|axis| {
                                wear.grime_projection[axis]
                                    + wear.grime_projection_unique_delta[axis] * wear.unique_id
                            })
                        })
                        .or_else(|| {
                            material
                                .surface_condition
                                .map(|condition| condition.triangle)
                        })
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    wear_damage_projection: material
                        .mod_wear
                        .map(|wear| {
                            std::array::from_fn(|axis| {
                                wear.damage_projection[axis]
                                    + wear.damage_projection_unique_delta[axis] * wear.unique_id
                            })
                        })
                        .or_else(|| {
                            material
                                .surface_condition
                                .map(|condition| condition.projection)
                        })
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    wear_scratches_remap_base: material
                        .mod_wear
                        .map(|wear| wear.scratches_remap_base)
                        .or_else(|| {
                            material
                                .surface_condition
                                .map(|surface| surface.detail_projection)
                        })
                        .unwrap_or_default(),
                    wear_scratches_remap_scale: material
                        .mod_wear
                        .map(|wear| wear.scratches_remap_scale)
                        .or_else(|| {
                            material.surface_condition.map(|surface| {
                                [
                                    surface.detail_exponent,
                                    surface.detail_roughness,
                                    surface.detail_remap[0],
                                    surface.detail_remap[1],
                                ]
                            })
                        })
                        .unwrap_or([1.0; 4]),
                    wear_surface_params: material
                        .mod_wear
                        .map(|wear| [wear.condition_blend, 4.5947933, 0.0, 0.0])
                        .or_else(|| {
                            material.surface_condition.map(|condition| {
                                [
                                    condition.albedo[0],
                                    condition.albedo[1],
                                    condition.albedo[2],
                                    condition.roughness,
                                ]
                            })
                        })
                        .unwrap_or_default(),
                    gear_palette_default: material.gear_dye_default.unwrap_or_default(),
                    gear_palette_colors: material
                        .gear_dye_palette
                        .map(|palette| palette.map(|dye| dye.color))
                        .unwrap_or_default(),
                    gear_worn_dye_colors: material.gear_worn_dye_palette.unwrap_or_default(),
                    gear_dye_detail_colors: material.gear_dye_detail_palette.unwrap_or_default(),
                    gear_palette_roughness: material
                        .gear_dye_palette
                        .map(|palette| palette.map(|dye| dye.roughness_remap))
                        .unwrap_or_default(),
                    gear_palette_metal: material
                        .gear_dye_palette
                        .map(|palette| palette.map(|dye| dye.metal_remap))
                        .unwrap_or_default(),
                    decal_params: material
                        .investment_decal
                        .map(|decal| {
                            [
                                match decal.mode {
                                    InvestmentDecalMode::SelectorMask => 1.0,
                                    InvestmentDecalMode::DetailSelectorMask => 2.0,
                                    InvestmentDecalMode::SceneNormalColorMask => 3.0,
                                },
                                f32::from(decal.selector_color_count),
                                f32::from(decal.atlas_selector_max),
                                decal.output_gate,
                            ]
                        })
                        .unwrap_or_default(),
                    decal_detail_transform: material
                        .investment_decal
                        .map(|decal| decal.detail_transform)
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    decal_detail_base: material
                        .investment_decal
                        .map(|decal| decal.detail_base)
                        .unwrap_or([1.0; 4]),
                    decal_detail_scale: material
                        .investment_decal
                        .map(|decal| decal.detail_scale)
                        .unwrap_or_default(),
                    decal_mask_params: material
                        .investment_decal
                        .map(|decal| {
                            [
                                match decal.mask_mode {
                                    InvestmentDecalMaskMode::Threshold => 0.0,
                                    InvestmentDecalMaskMode::UvSplit => 1.0,
                                    InvestmentDecalMaskMode::Binary => 2.0,
                                },
                                decal.mask_threshold,
                                decal.grayscale_remap[0],
                                decal.grayscale_remap[1],
                            ]
                        })
                        .unwrap_or_default(),
                    decal_mask_remap: material
                        .investment_decal
                        .map(|decal| {
                            [
                                decal.positive_mask_remap[0],
                                decal.positive_mask_remap[1],
                                decal.negative_mask_remap[0],
                                decal.negative_mask_remap[1],
                            ]
                        })
                        .unwrap_or([0.0, 1.0, 0.0, 1.0]),
                    decal_selector_colors: material
                        .investment_decal
                        .map(|decal| decal.selector_colors)
                        .unwrap_or_default(),
                    // D3D11 sampler offset 0x10 is authored MipLODBias. Most
                    // Marathon surface samplers use -0.5. The old shader
                    // ignored it and forced +0.75, discarding 1.25 mip levels.
                    sampler_params: [
                        material
                            .sampler
                            .map(|sampler| sampler.mip_lod_bias)
                            .filter(|bias| bias.is_finite())
                            .unwrap_or(-0.5)
                            .clamp(-16.0, 15.99),
                        material.procedural_scale,
                        material.shared_atlas_detail.is_some() as u8 as f32,
                        material
                            .shared_atlas_detail
                            .map(|detail| detail.ambient_occlusion)
                            .unwrap_or(0.0),
                    ],
                    pattern_projection: material
                        .gear_pattern
                        .map(|pattern| pattern.projection)
                        .or_else(|| material.shared_atlas_detail.map(|detail| detail.projection))
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    pattern_params: material
                        .gear_pattern
                        .map(|pattern| {
                            [
                                material.pattern_field.is_some() as u8 as f32,
                                pattern.normal_power,
                                pattern.field_midpoint,
                                pattern.warp[0],
                            ]
                        })
                        .or_else(|| {
                            material
                                .shared_atlas_detail
                                .map(|detail| [0.0, detail.exponent, 0.0, 0.0])
                        })
                        .unwrap_or_default(),
                    pattern_stripe: material
                        .gear_pattern
                        .map(|pattern| pattern.stripe)
                        .or_else(|| {
                            material
                                .shared_atlas_detail
                                .map(|detail| [detail.base[0], detail.base[1], detail.base[2], 0.0])
                        })
                        .unwrap_or_default(),
                    pattern_contour: material
                        .gear_pattern
                        .map(|pattern| {
                            let mut contour = pattern.contour;
                            contour[3] = pattern.warp[1];
                            contour
                        })
                        .or_else(|| {
                            material.shared_atlas_detail.map(|detail| {
                                [detail.scale[0], detail.scale[1], detail.scale[2], 0.0]
                            })
                        })
                        .unwrap_or_default(),
                    pattern_contour_remap: material
                        .gear_pattern
                        .map(|pattern| pattern.contour_remap)
                        .unwrap_or_default(),
                    pattern_colors: material
                        .gear_pattern
                        .map(|pattern| pattern.colors)
                        .unwrap_or_default(),
                    transmission_colors: material
                        .transmission
                        .map(|transmission| transmission.colors)
                        .unwrap_or([[1.0; 4]; 2]),
                    transmission_surfaces: material
                        .transmission
                        .map(|transmission| transmission.surfaces)
                        .unwrap_or_default(),
                    transmission_params: [
                        material
                            .transmission
                            .map(|transmission| f32::from(transmission.color_count))
                            .unwrap_or(0.0),
                        material.transmission.and_then(|transmission| transmission.absorption)
                            .map_or(0.0, |absorption| absorption[0]),
                        material.transmission.and_then(|transmission| transmission.absorption)
                            .map_or(0.0, |absorption| absorption[1]),
                        0.0,
                    ],
                    coating_colors: material
                        .forward_coating
                        .map(|coating| coating.colors)
                        .unwrap_or_default(),
                    coating_projection: material
                        .forward_coating
                        .map(|coating| coating.projection)
                        .unwrap_or_default(),
                    coating_params0: material
                        .forward_coating
                        .map(|coating| {
                            [
                                coating.projection_exponent,
                                coating.incidence_remap[0],
                                coating.incidence_remap[1],
                                coating.coverage,
                            ]
                        })
                        .unwrap_or_default(),
                    coating_params1: material
                        .forward_coating
                        .map(|coating| {
                            [
                                coating.detail_remap[0],
                                coating.detail_remap[1],
                                coating.response_remap[0],
                                coating.response_remap[1],
                            ]
                        })
                        .unwrap_or_default(),
                    coating_environment_params: material
                        .forward_coating
                        .map(|coating| {
                            [
                                coating.environment_remap[0],
                                coating.environment_remap[1],
                                coating.environment_strength,
                                coating.environment_params[0],
                            ]
                        })
                        .unwrap_or_default(),
                    coating_environment_extra: material
                        .forward_coating
                        .map(|coating| {
                            [
                                coating.environment_params[1],
                                coating.environment_lod[0],
                                coating.environment_lod[1],
                                0.0,
                            ]
                        })
                        .unwrap_or_default(),
                    coating_specular_colors: material
                        .forward_coating
                        .map(|coating| coating.specular_colors)
                        .unwrap_or_default(),
                    coating_specular_params: material
                        .forward_coating
                        .map(|coating| {
                            [
                                [
                                    coating.specular_exponents[0],
                                    coating.specular_strengths[0],
                                    coating.environment_remap[0],
                                    coating.environment_remap[1],
                                ],
                                [
                                    coating.specular_exponents[1],
                                    coating.specular_strengths[1],
                                    coating.lobe_direction_scales[0],
                                    coating.lobe_direction_scales[1],
                                ],
                            ]
                        })
                        .unwrap_or_default(),
                    character_detail_transform: material
                        .character_surface
                        .map(|surface| surface.detail_transform)
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    character_detail_base: material
                        .character_surface
                        .map(|surface| surface.detail_base)
                        .unwrap_or([1.0; 4]),
                    character_detail_scale: material
                        .character_surface
                        .map(|surface| surface.detail_scale)
                        .unwrap_or_default(),
                    character_params: [
                        material.character_surface.is_some() as u8 as f32,
                        material
                            .character_surface
                            .map(|surface| surface.detail_gate)
                            .unwrap_or(0.0),
                        4.5947933,
                        material
                            .character_surface
                            .map(|surface| f32::from(surface.mode))
                            .unwrap_or(0.0),
                    ],
                    character_extra: material
                        .character_surface
                        .map(|surface| surface.extra)
                        .unwrap_or_default(),
                    character_palette: material
                        .character_surface
                        .map(|surface| surface.palette)
                        .unwrap_or([[1.0; 4]; 2]),
                    character_procedural: material
                        .character_surface
                        .map(|surface| surface.procedural_constants)
                        .unwrap_or([[0.0; 4]; 11]),
                    runner_layered_params: [
                        material.runner_layered_surface.is_some() as u8 as f32,
                        material
                            .runner_layered_surface
                            .map(|surface| f32::from(surface.mode))
                            .unwrap_or(0.0),
                        material.runner_occlusion.is_some() as u8 as f32,
                        material
                            .runner_occlusion
                            .map(|occlusion| f32::from(occlusion.channel))
                            .unwrap_or(0.0),
                    ],
                    runner_layered_constants: material
                        .runner_layered_surface
                        .map(|surface| surface.constants)
                        .unwrap_or([[0.0; 4]; 24]),
                    runner_color_constants: material
                        .runner_layered_surface
                        .map(|surface| surface.color_overlay_constants)
                        .unwrap_or([[0.0; 4]; 7]),
                    alpha_mask_params: material
                        .alpha_mask
                        .map(|alpha| [1.0, alpha.threshold, alpha.remap[0], alpha.remap[1]])
                        .unwrap_or_default(),
                };
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("quicktag_model_material_uniform"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                // Material role owns transfer semantics. Albedo/emissive use
                // compatible sRGB views; normal/control/wear maps use linear
                // views even when Tiger reused an sRGB-capable container.
                let color_view = material
                    .color
                    .as_ref()
                    .map(|texture| {
                        material_texture_view(texture, material.forward_coating.is_none())
                    })
                    .unwrap_or_else(|| fallback_color_view.clone());
                let normal_view = material
                    .normal
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let emissive_view = material
                    .emissive
                    .as_ref()
                    .map(|texture| material_texture_view(texture, true))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let control_view = material
                    .control
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let wear_scratches_view = material
                    .wear_scratches
                    .as_ref()
                    .map(|texture| {
                        material_texture_view(
                            texture,
                            material.mod_wear.is_some() && texture.desc.format.is_srgb(),
                        )
                    })
                    .unwrap_or_else(|| fallback_color_view.clone());
                let wear_grime_view = material
                    .wear_grime
                    .as_ref()
                    .map(|texture| {
                        material_texture_view(
                            texture,
                            material.mod_wear.is_none() || texture.desc.format.is_srgb(),
                        )
                    })
                    .unwrap_or_else(|| fallback_color_view.clone());
                let wear_damage_view = material
                    .wear_damage
                    .as_ref()
                    .map(|texture| {
                        material_texture_view(
                            texture,
                            material.mod_wear.is_some() && texture.desc.format.is_srgb(),
                        )
                    })
                    .unwrap_or_else(|| fallback_color_view.clone());
                let pattern_field_view = material
                    .pattern_field
                    .as_ref()
                    .or(material.runner_procedural_map.as_ref())
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let character_surface_view = material
                    .character_surface_map
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let character_detail_color_view = material
                    .character_detail_color
                    .as_ref()
                    .or(material.runner_color_overlay_map.as_ref())
                    .map(|texture| material_texture_view(texture, true))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let character_procedural_view = material
                    .character_procedural_map
                    .as_ref()
                    .or(material.runner_material_response_map.as_ref())
                    .or(material.runner_occlusion_map.as_ref())
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let runner_surface_view = material
                    .runner_surface_map
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let runner_detail_normal_a_view = material
                    .runner_detail_normal_a
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let runner_detail_normal_b_view = material
                    .runner_detail_normal_b
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let runner_detail_normal_c_view = material
                    .runner_detail_normal_c
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let runner_detail_normal_d_view = material
                    .runner_detail_normal_d
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let coating_environment_view = material
                    .coating_environment_map
                    .as_ref()
                    .and_then(|texture| {
                        texture.full_cubemap_texture.as_ref().map(|handle| {
                            create_model_texture_view(
                                &handle,
                                &wgpu::TextureViewDescriptor {
                                    format: Some(srgb_texture_format(texture.desc.format)),
                                    dimension: Some(wgpu::TextureViewDimension::Cube),
                                    array_layer_count: Some(6),
                                    ..Default::default()
                                },
                            )
                        })
                    })
                    .unwrap_or_else(|| fallback_cubemap_view.clone());
                let authored_sampler = material
                    .sampler
                    .map(|desc| create_model_sampler(device, desc));
                let sampler = authored_sampler.as_ref().unwrap_or(&material_sampler);
                let authored_coating_environment_sampler = material
                    .coating_environment_sampler
                    .map(|desc| create_model_sampler(device, desc));
                let coating_environment_sampler = authored_coating_environment_sampler
                    .as_ref()
                    .unwrap_or(&material_sampler);
                let bind_group = create_model_bind_group(
                    device,
                    &wgpu::BindGroupDescriptor {
                        label: Some("quicktag_model_material_bind_group"),
                        layout: &material_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&color_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&normal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&emissive_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&control_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(&wear_scratches_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(&wear_grime_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(&wear_damage_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::TextureView(&pattern_field_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    &character_surface_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 11,
                                resource: wgpu::BindingResource::TextureView(
                                    &character_detail_color_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 12,
                                resource: wgpu::BindingResource::TextureView(
                                    &character_procedural_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 13,
                                resource: wgpu::BindingResource::TextureView(&runner_surface_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 14,
                                resource: wgpu::BindingResource::TextureView(
                                    &runner_detail_normal_a_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 15,
                                resource: wgpu::BindingResource::TextureView(
                                    &runner_detail_normal_b_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 16,
                                resource: wgpu::BindingResource::TextureView(
                                    &runner_detail_normal_c_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 17,
                                resource: wgpu::BindingResource::TextureView(
                                    &runner_detail_normal_d_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 18,
                                resource: wgpu::BindingResource::TextureView(
                                    &coating_environment_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 19,
                                resource: wgpu::BindingResource::Sampler(
                                    coating_environment_sampler,
                                ),
                            },
                        ],
                    },
                );
                material_buffers.push(buffer);
                material_bind_groups.push(bind_group);
            }

            let Some((
                color_view,
                lit_color_view,
                surface_normal_view,
                surface_properties_view,
                surface_emissive_view,
                surface_flags_view,
                surface_albedo_view,
                depth_view,
                scene_color_copy_view,
                scene_normal_copy_view,
                distortion_view,
                shadow_depth_view,
                bloom_half_view,
                bloom_quarter_view,
                bloom_blur_view,
            )) = callback_resources
                .get::<ModelTargetResources>()
                .map(|target| {
                    (
                        target.color_view.clone(),
                        target.lit_color_view.clone(),
                        target.surface_normal_view.clone(),
                        target.surface_properties_view.clone(),
                        target.surface_emissive_view.clone(),
                        target.surface_flags_view.clone(),
                        target.surface_albedo_view.clone(),
                        target.depth_view.clone(),
                        target.scene_color_copy_view.clone(),
                        target.scene_normal_copy_view.clone(),
                        target.distortion_view.clone(),
                        target.shadow_depth_view.clone(),
                        target.bloom_half_view.clone(),
                        target.bloom_quarter_view.clone(),
                        target.bloom_blur_view.clone(),
                    )
                })
            else {
                return Vec::new();
            };
            let decal_view = self
                .decal
                .as_ref()
                .map(|texture| material_texture_view(texture, true))
                .unwrap_or_else(|| fallback_color_view.clone());
            let bloom_half_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_bloom_half_source"),
                    layout: &bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&lit_color_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                    ],
                },
            );
            let bloom_quarter_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_bloom_quarter_source"),
                    layout: &bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&bloom_half_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                    ],
                },
            );
            let bloom_blur_horizontal_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_bloom_blur_horizontal_source"),
                    layout: &bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&bloom_quarter_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                    ],
                },
            );
            let bloom_blur_vertical_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_bloom_blur_vertical_source"),
                    layout: &bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&bloom_blur_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                    ],
                },
            );
            let present_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_present_bind_group"),
                    layout: &present_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&lit_color_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: scene_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&decal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&bloom_half_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&bloom_quarter_view),
                        },
                    ],
                },
            );
            let lighting_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_lighting_bind_group"),
                    layout: &lighting_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&color_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&surface_normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&surface_properties_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&surface_emissive_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: scene_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&surface_flags_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&surface_albedo_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::TextureView(&shadow_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: wgpu::BindingResource::TextureView(&callback_resources.get::<ModelTargetResources>()
                                .and_then(|target|target.mixed_receivers.as_ref().map(|receiver|receiver.coverage_view.clone())
                                    .or_else(||target.native_surface.as_ref().map(|native|native.views[1].clone())))
                                .unwrap_or_else(||fallback_color_view.clone())),
                        },
                    ],
                },
            );
            let distortion_resolve_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_distortion_resolve_bind_group"),
                    layout: &distortion_resolve_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&scene_color_copy_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&distortion_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&present_sampler),
                        },
                    ],
                },
            );
            let coating_deferred_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_coating_deferred_bind_group"),
                    layout: &callback_resources
                        .get::<ModelPipelineResources>()
                        .expect("model pipelines exist while preparing frame resources")
                        .coating_deferred_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&scene_normal_copy_view),
                        },
                    ],
                },
            );
            let coating_fallback_bind_group = create_model_bind_group(
                device,
                &wgpu::BindGroupDescriptor {
                    label: Some("quicktag_model_coating_deferred_fallback_bind_group"),
                    layout: &callback_resources
                        .get::<ModelPipelineResources>()
                        .expect("model pipelines exist while preparing frame resources")
                        .coating_deferred_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&shadow_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&scene_normal_copy_view),
                        },
                    ],
                },
            );
            let make_bundles = |passes: &[RenderPassKind], coating_deferred| {
                callback_resources
                    .get::<ModelPipelineResources>()
                    .map(|pipelines| {
                        create_model_render_bundles(
                            device,
                            pipelines,
                            &scene_bind_group,
                            &material_bind_groups,
                            coating_deferred,
                            &self.preview,
                            &self.draws,
                            passes,
                        )
                    })
                    .unwrap_or_default()
            };
            let opaque_bundles = make_bundles(
                &[
                    RenderPassKind::OpaqueCompatibility,
                    RenderPassKind::AlphaTestedCompatibility,
                    RenderPassKind::UnknownCompatibility,
                ],
                &coating_fallback_bind_group,
            );
            let decal_bundles = make_bundles(
                &[
                    RenderPassKind::DecalCompatibility,
                    RenderPassKind::InvestmentDecalCompatibility,
                ],
                &coating_fallback_bind_group,
            );
            let mut generic=decal_bundles.into_iter();
            let mut decal_runs=Vec::new();
            for draw in self.draws.iter().filter(|d| d.passes.iter().any(|p| matches!(p,RenderPassKind::DecalCompatibility|RenderPassKind::InvestmentDecalCompatibility))) {
                if draw.native_c827 {
                    let index=native_c827.iter().position(|(i,_)| *i==draw.stable_index).expect("native decal retained binding");
                    decal_runs.push(ModelDecalRun::C827(index));
                } else if draw.native_decal {
                    let index=native_decals.iter().position(|(i,_)|*i==draw.stable_index).unwrap();
                    decal_runs.push(ModelDecalRun::Authored(index));
                } else {
                    let bundle = generic.next().expect("every compatibility decal must retain its own ordered bundle");
                    if let Some(ModelDecalRun::Compatibility(bundles))=decal_runs.last_mut() { bundles.push(bundle); }
                    else { decal_runs.push(ModelDecalRun::Compatibility(vec![bundle])); }
                }
            }
            assert!(generic.next().is_none());
            let additive_bundles = make_bundles(
                &[RenderPassKind::ForwardAdditive],
                &coating_fallback_bind_group,
            );
            let transparent_bundles = make_bundles(
                &[RenderPassKind::ForwardTransparent],
                &coating_fallback_bind_group,
            );
            let coating_bundles = make_bundles(
                &[RenderPassKind::ForwardCoating],
                &coating_deferred_bind_group,
            );
            let distortion_bundles =
                make_bundles(&[RenderPassKind::Distortion], &coating_fallback_bind_group);
            callback_resources.insert(ModelFrameResources {
                key: frame_key,
                scene_buffer,
                shadow_scene_bind_group,
                scene_bind_group,
                _authored_shadow_uniform_buffers: authored_shadow_uniform_buffers,
                authored_shadow_bind_groups,
                _material_buffers: material_buffers,
                material_bind_groups,
                _coating_deferred_bind_group: coating_deferred_bind_group,
                _coating_fallback_bind_group: coating_fallback_bind_group,
                bloom_half_bind_group,
                bloom_quarter_bind_group,
                bloom_blur_horizontal_bind_group,
                bloom_blur_vertical_bind_group,
                present_bind_group,
                lighting_bind_group,
                distortion_resolve_bind_group,
                opaque_bundles,
                decal_runs,
                native_view_buffer,
                native_pixel_view_buffer,
                native_surfaces,
                native_source_color,
                native_viewer_material,
                viewer_receiver_import,
                viewer_receiver_coverage,
                native_skinning_buffers,
                native_surface_constants,
                native_decal_constants,
                native_compute,
                native_c827,
                native_decals,
                additive_bundles,
                transparent_bundles,
                coating_bundles,
                distortion_bundles,
            });
        }

        let Some(pipelines) = callback_resources.get::<ModelPipelineResources>() else {
            return Vec::new();
        };
        let Some(target) = callback_resources.get::<ModelTargetResources>() else {
            return Vec::new();
        };
        let Some(frame) = callback_resources.get::<ModelFrameResources>() else {
            return Vec::new();
        };

        for ((index, mesh_index),bindings) in &frame.native_compute {
            let mesh=&self.preview.authored_inputs[*index].as_ref().expect("native geometry").static_meshes[*mesh_index];
            let producer=&pipelines.authored_compute_pipelines.iter().find(|(abi,_)|*abi==mesh.producer).expect("exact compute source").1;
            mesh.encode_once(egui_encoder,producer,bindings);
        }

        {
            let mut shadow_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_sun_shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.shadow_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            shadow_pass.set_bind_group(0, &frame.shadow_scene_bind_group, &[]);
            for draw in &self.shadow_draws {
                shadow_pass.set_bind_group(
                    1,
                    &frame.material_bind_groups[draw.material_index],
                    &[],
                );

                let rendered_authored = (|| {
                    if !draw.authored_native_vertex_supported {
                        return None;
                    }
                    let source_index = draw.authored_source?;
                    let authored = draw.authored_stage?;
                    let source = self.preview.authored_inputs.get(source_index)?.as_ref()?;
                    let stage_layout = source.stage_layout(
                        crate::render::adapter::GoliathAdapter::AUTHORED_SHADOW_STAGE,
                    )?;
                    if stage_layout.layout_id != authored.input_layout_id {
                        return None;
                    }
                    let descriptor = stage_layout.descriptor.as_ref()?;
                    let stream_strides = authored_stream_strides(source, descriptor)?;
                    let index = source.index_buffer.as_ref()?;
                    let transform_bind_group = frame
                        .authored_shadow_bind_groups
                        .get(source_index)?
                        .as_ref()?;
                    let end = authored
                        .source_index_start
                        .checked_add(authored.source_index_count)?;
                    if end > index.index_count {
                        return None;
                    }
                    let key = StrictShadowPipelineKey {
                        layout_id: stage_layout.layout_id,
                        primitive_type: authored.primitive_type,
                        rasterizer: draw.pipeline.rasterizer,
                        index_32bit: matches!(index.format, wgpu::IndexFormat::Uint32),
                        stream_strides,
                    };
                    let pipeline = pipelines
                        .strict_shadow_pipelines
                        .iter()
                        .find(|(existing, _pipeline)| *existing == key)
                        .map(|(_key, pipeline)| pipeline)?;

                    for stream in descriptor.streams.iter().filter(|stream| {
                        stream.elements.iter().any(|element| {
                            (element.semantic == 0 && element.semantic_index == 0)
                                || (element.semantic == 5 && element.semantic_index == 0)
                        })
                    }) {
                        let gpu_stream = source
                            .vertex_streams
                            .iter()
                            .find(|candidate| candidate.stream_index == stream.stream_index)?;
                        shadow_pass.set_vertex_buffer(
                            u32::from(stream.stream_index),
                            gpu_stream.buffer.slice(..),
                        );
                    }
                    shadow_pass.set_index_buffer(index.buffer.slice(..), index.format);
                    shadow_pass.set_pipeline(pipeline);
                    shadow_pass.set_bind_group(2, transform_bind_group, &[]);
                    shadow_pass.draw_indexed(authored.source_index_start..end, 0, 0..1);
                    Some(())
                })()
                .is_some();

                if !rendered_authored {
                    shadow_pass.set_pipeline(
                        &pipelines.shadow_pipelines
                            [shadow_pipeline_index(draw.pipeline.rasterizer)],
                    );
                    shadow_pass.set_vertex_buffer(0, self.preview.vertex_buffer.slice(..));
                    shadow_pass.set_index_buffer(
                        self.preview.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    shadow_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
                }
            }
        }

        let has_authored_depth = self
            .draws
            .iter()
            .any(|draw| draw.passes.contains(&RenderPassKind::DepthOnly));
        if has_authored_depth {
            let mut depth_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_authored_depth_prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            depth_pass.set_bind_group(0, &frame.scene_bind_group, &[]);
            for draw in self
                .draws
                .iter()
                .filter(|draw| draw.passes.contains(&RenderPassKind::DepthOnly))
            {
                let rendered_authored = (|| {
                    if !draw.authored_native_vertex_supported {
                        return None;
                    }
                    let source_index = draw.authored_source?;
                    let authored = draw.authored_stage?;
                    let source = self.preview.authored_inputs.get(source_index)?.as_ref()?;
                    let stage_layout = source.stage_layout(authored.raw_stage)?;
                    if stage_layout.layout_id != authored.input_layout_id {
                        return None;
                    }
                    let descriptor = stage_layout.descriptor.as_ref()?;
                    let stream_strides = authored_stream_strides(source, descriptor)?;
                    let index = source.index_buffer.as_ref()?;
                    let transform_bind_group =
                        frame.authored_shadow_bind_groups.get(source_index)?.as_ref()?;
                    let end = authored
                        .source_index_start
                        .checked_add(authored.source_index_count)?;
                    if end > index.index_count {
                        return None;
                    }
                    let key = StrictDepthPipelineKey {
                        layout_id: stage_layout.layout_id,
                        primitive_type: authored.primitive_type,
                        rasterizer: draw.pipeline.rasterizer,
                        depth_stencil: draw.pipeline.depth_stencil,
                        depth_bias: draw.pipeline.depth_bias,
                        index_32bit: matches!(index.format, wgpu::IndexFormat::Uint32),
                        stream_strides,
                    };
                    let pipeline = pipelines
                        .strict_depth_pipelines
                        .iter()
                        .find(|(existing, _pipeline)| *existing == key)
                        .map(|(_key, pipeline)| pipeline)?;

                    for stream in descriptor.streams.iter().filter(|stream| {
                        stream.elements.iter().any(|element| {
                            element.semantic == 0 && element.semantic_index == 0
                        })
                    }) {
                        let gpu_stream = source
                            .vertex_streams
                            .iter()
                            .find(|candidate| candidate.stream_index == stream.stream_index)?;
                        depth_pass.set_vertex_buffer(
                            u32::from(stream.stream_index),
                            gpu_stream.buffer.slice(..),
                        );
                    }
                    depth_pass.set_index_buffer(index.buffer.slice(..), index.format);
                    depth_pass.set_pipeline(pipeline);
                    depth_pass.set_bind_group(1, transform_bind_group, &[]);
                    depth_pass.draw_indexed(authored.source_index_start..end, 0, 0..1);
                    Some(())
                })()
                .is_some();

                if !rendered_authored {
                    depth_pass.set_pipeline(
                        &pipelines.depth_pipelines[shadow_pipeline_index(draw.pipeline.rasterizer)],
                    );
                    depth_pass.set_bind_group(
                        1,
                        &frame.material_bind_groups[draw.material_index],
                        &[],
                    );
                    depth_pass.set_bind_group(2, &frame._coating_fallback_bind_group, &[]);
                    depth_pass.set_vertex_buffer(0, self.preview.vertex_buffer.slice(..));
                    depth_pass.set_index_buffer(
                        self.preview.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    depth_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
                }
            }
        }

    let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("quicktag_model_offscreen_pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.006_049,
                            g: 0.009_721,
                            b: 0.022_174,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(clear_surface_attachment(&target.surface_normal_view)),
                Some(clear_surface_attachment(&target.surface_properties_view)),
                Some(clear_surface_attachment(&target.surface_albedo_view)),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if has_authored_depth {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.execute_bundles(frame.opaque_bundles.iter());
        drop(pass);

        if let Some(native)=&target.native_surface {
            let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:Some("live authored opaque surface MRTs"),
                color_attachments:&std::array::from_fn::<_,4,_>(|i|Some(clear_surface_attachment(&native.views[i]))),
                depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment {
                    view:&target.depth_view,depth_ops:Some(wgpu::Operations { load:wgpu::LoadOp::Load,store:wgpu::StoreOp::Store }),stencil_ops:None,
                }),timestamp_writes:None,occlusion_query_set:None,
            });
            for (stable_index,abi,bindings) in &frame.native_surfaces {
                let rasterizer=self.preview.draws[*stable_index].pipeline.rasterizer;
                let vertex=self.preview.draws[*stable_index].native_surface.as_ref().unwrap().vertex_program;
                pipelines.authored_surface_pipelines.iter().find(|(key,_)|*key==(vertex,*abi,rasterizer,self.preview.authored_index_format(*stable_index))).unwrap().1.encode(&mut pass,bindings);
            }
            drop(pass);
            // Decals need valid destinations on generic opaque parts too.
            // Retain native-only coverage and unchanged generic pixels.
            if let Some(receiver)=&target.mixed_receivers {
                receiver.copy_original_native_normal(egui_encoder,&native.textures[1]);
                let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("generic opaque receivers into authored decal attachments"),
                    color_attachments:&std::array::from_fn::<_,4,_>(|i|Some(load_surface_attachment(&native.views[i]))),
                    depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None,
                });
                pipelines.viewer_receiver_import.encode(&mut pass,frame.viewer_receiver_import.as_ref().unwrap());
                drop(pass);
                receiver.copy_opaque(egui_encoder,native);
            }
            if let Some((snapshot,_))=&target.native_normal_snapshot {
                egui_encoder.copy_texture_to_texture(native.textures[1].as_image_copy(),snapshot.as_image_copy(),
                    wgpu::Extent3d {width:target.size[0],height:target.size[1],depth_or_array_layers:1});
            }
            if frame.decal_runs.iter().any(|run|matches!(run,ModelDecalRun::Authored(_)|ModelDecalRun::C827(_))) {
                let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("authored ordered stage2 material composition"),
                    color_attachments:&std::array::from_fn::<_,4,_>(|i|Some(load_surface_attachment(&native.views[i]))),
                    depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment {
                        view:&target.depth_view,depth_ops:Some(wgpu::Operations {load:wgpu::LoadOp::Load,store:wgpu::StoreOp::Store}),stencil_ops:None,
                    }),timestamp_writes:None,occlusion_query_set:None,
                });
                for run in &frame.decal_runs {
                    match run {
                        ModelDecalRun::Authored(index) => {
                            let (stable_index,b)=&frame.native_decals[*index];
                            pipelines.authored_decal_pipelines.iter().find(|(abi,_)|*abi==(b.vertex_program,b.program,self.preview.authored_index_format(*stable_index))).unwrap().1.encode(&mut pass,&b.pixel,&b.vertex,
                                &b.source,&b.uv,&b.indices,b.range.clone());
                        },
                        ModelDecalRun::C827(index) => {
                            let (stable_index,b)=&frame.native_c827[*index];
                            pipelines.authored_material_c827_pipelines.iter().find(|(f,_)|*f==self.preview.authored_index_format(*stable_index)).unwrap().1.encode(&mut pass,b,false);
                        },
                        ModelDecalRun::Compatibility(_) => {},
                    }
                }
            }
            if let Some(receiver)=&target.mixed_receivers {
                let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("native and decal-modified receiver coverage"),
                    color_attachments:&[Some(clear_surface_attachment(&receiver.coverage_view))],
                    depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None,
                });
                pipelines.viewer_receiver_coverage.encode(&mut pass,frame.viewer_receiver_coverage.as_ref().unwrap());
            }
            if self.scene.postprocess2[0] == 0.0 && self.scene.fidelity[0] == 1.0 {
                let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("live authored materials to viewer lighting"),
                    color_attachments:&[
                        Some(load_surface_attachment(&target.surface_normal_view)),
                        Some(load_surface_attachment(&target.surface_properties_view)),
                        Some(load_surface_attachment(&target.surface_albedo_view)),
                    ], depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None,
                });
                pipelines.authored_viewer_material.encode(&mut pass,frame.native_viewer_material.as_ref().unwrap());
            }
            let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:Some("live authored RT0 Base Color consumer"),
                color_attachments:&[Some(load_surface_attachment(&target.surface_albedo_view))],
                depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None,
            });
            pipelines.authored_source_color.encode(&mut pass,frame.native_source_color.as_ref().unwrap(),false);
            drop(pass);
            if self.scene.postprocess2[0] == 1.0 {
                let mut pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("live authored RT0 Base Color diagnostic consumer"),
                    color_attachments:&[Some(load_surface_attachment(&target.color_view))],
                    depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None,
                });
                pipelines.authored_source_color.encode(&mut pass,frame.native_source_color.as_ref().unwrap(),true);
            }
        }

        // Stage-2 investment decals read the normal already written by the
        // opaque surface pass through an external screen-space texture. Keep
        // the source and destination separate: the normal attachment is still
        // loaded and written by the decal pass below.
        if target.features.deferred_normal_copy {
            egui_encoder.copy_texture_to_texture(
                target._surface_normal.as_image_copy(),
                target.scene_normal_copy.as_image_copy(),
                wgpu::Extent3d {
                    width: target.size[0],
                    height: target.size[1],
                    depth_or_array_layers: 1,
                },
            );
        }

        for run in &frame.decal_runs {
            if matches!(run,ModelDecalRun::Authored(_)) || (target.native_surface.is_some() && matches!(run,ModelDecalRun::C827(_))) { continue; }
            let mut special_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_ordered_decal_pass"),
                color_attachments: &[
                    Some(load_surface_attachment(&target.color_view)),
                    Some(load_surface_attachment(&target.surface_normal_view)),
                    Some(load_surface_attachment(&target.surface_properties_view)),
                    Some(load_surface_attachment(&target.surface_albedo_view)),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            match run {
                ModelDecalRun::Compatibility(bundles) => special_pass.execute_bundles(bundles.iter()),
                ModelDecalRun::Authored(_) => unreachable!("authored decals execute in the material pass"),
                ModelDecalRun::C827(index) => {
                    let (stable_index,b)=&frame.native_c827[*index];
                    pipelines.authored_c827_pipelines.iter().find(|(f,_)|*f==self.preview.authored_index_format(*stable_index)).unwrap().1.encode(&mut special_pass,b,false);
                },
            }
            drop(special_pass);
            if let ModelDecalRun::C827(index)=run {
                // Exact RT0 multiplier also consumes the viewer's separate
                // albedo color. This is a color projection, not an assertion
                // that native RT3 (motion) contains albedo.
                let mut albedo_pass=egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("C827 receiver albedo projection"),
                    color_attachments:&[Some(load_surface_attachment(&target.surface_albedo_view)),None,None,None],
                    depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment {
                        view:&target.depth_view, depth_ops:Some(wgpu::Operations { load:wgpu::LoadOp::Load,store:wgpu::StoreOp::Store }),stencil_ops:None,
                    }),timestamp_writes:None,occlusion_query_set:None,
                });
                let (stable_index,b)=&frame.native_c827[*index];
                pipelines.authored_c827_pipelines.iter().find(|(f,_)|*f==self.preview.authored_index_format(*stable_index)).unwrap().1.encode(&mut albedo_pass,b,true);
            }
        }

        {
            let mut emissive_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_material_emissive_pass"),
                color_attachments: &[Some(clear_surface_attachment(
                    &target.surface_emissive_view,
                ))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            emissive_pass.set_vertex_buffer(0, self.preview.vertex_buffer.slice(..));
            emissive_pass.set_index_buffer(
                self.preview.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for draw in self
                .draws
                .iter()
                .filter(|draw| !draw.native_c827 && !draw.native_surface && !draw.native_decal && !draw.passes.iter().copied().any(is_forward_pass))
            {
                let emissive_key = draw.pipeline.material_emissive();
                let Some((_, pipeline)) = pipelines
                    .model_pipelines
                    .iter()
                    .find(|(key, _)| key.gpu_equivalent(emissive_key))
                else {
                    continue;
                };
                emissive_pass.set_pipeline(pipeline);
                emissive_pass.set_bind_group(0, &frame.scene_bind_group, &[]);
                emissive_pass.set_bind_group(
                    1,
                    &frame.material_bind_groups[draw.material_index],
                    &[],
                );
                emissive_pass.set_bind_group(2, &frame._coating_fallback_bind_group, &[]);
                emissive_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }
        if self.scene.postprocess2[0] == 0.0 && self.scene.fidelity[0] == 1.0
            && let Some(inputs) = frame.native_viewer_material.as_ref()
        {
            let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("live authored emission to viewer lighting"),
                color_attachments: &[Some(load_surface_attachment(&target.surface_emissive_view))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pipelines.authored_viewer_material.encode_emissive(&mut pass, inputs);
        }

        {
            let mut flags_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_material_flags_pass"),
                color_attachments: &[Some(clear_surface_attachment(&target.surface_flags_view))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            flags_pass.set_vertex_buffer(0, self.preview.vertex_buffer.slice(..));
            flags_pass.set_index_buffer(
                self.preview.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for draw in self
                .draws
                .iter()
                .filter(|draw| !draw.native_c827 && !draw.native_surface && !draw.native_decal && !draw.passes.iter().copied().any(is_forward_pass))
            {
                let flag_key = draw.pipeline.material_flags();
                let Some((_, pipeline)) = pipelines
                    .model_pipelines
                    .iter()
                    .find(|(key, _)| key.gpu_equivalent(flag_key))
                else {
                    continue;
                };
                flags_pass.set_pipeline(pipeline);
                flags_pass.set_bind_group(0, &frame.scene_bind_group, &[]);
                flags_pass.set_bind_group(1, &frame.material_bind_groups[draw.material_index], &[]);
                flags_pass.set_bind_group(2, &frame._coating_fallback_bind_group, &[]);
                flags_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }

        {
            let mut lighting_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quicktag_model_deferred_lighting_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.lit_color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            lighting_pass.set_pipeline(&pipelines.lighting_pipeline);
            lighting_pass.set_bind_group(0, &frame.lighting_bind_group, &[]);
            lighting_pass.draw(0..3, 0..1);
        }

        for (label, bundles) in [
            (
                "quicktag_model_forward_coating_pass",
                frame.coating_bundles.as_slice(),
            ),
            (
                "quicktag_model_forward_additive_pass",
                frame.additive_bundles.as_slice(),
            ),
            (
                "quicktag_model_forward_transparent_pass",
                frame.transparent_bundles.as_slice(),
            ),
        ] {
            if bundles.is_empty() {
                continue;
            }
            let samples_attached_depth = label == "quicktag_model_forward_coating_pass";
            let mut transparent_pass =
                egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(load_surface_attachment(&target.lit_color_view))],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &target.depth_view,
                        // WebGPU permits the coating pass to sample a depth
                        // attachment only while the attachment is read-only.
                        depth_ops: (!samples_attached_depth).then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            transparent_pass.execute_bundles(bundles.iter());
        }

        if !frame.distortion_bundles.is_empty() {
            {
                let mut distortion_pass =
                    egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("quicktag_model_distortion_payload_pass"),
                        color_attachments: &[Some(clear_surface_attachment(
                            &target.distortion_view,
                        ))],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                distortion_pass.execute_bundles(frame.distortion_bundles.iter());
            }

            egui_encoder.copy_texture_to_texture(
                target.lit_color.as_image_copy(),
                target.scene_color_copy.as_image_copy(),
                wgpu::Extent3d {
                    width: target.size[0],
                    height: target.size[1],
                    depth_or_array_layers: 1,
                },
            );

            let mut distortion_resolve_pass =
                egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("quicktag_model_distortion_resolve_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target.lit_color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            distortion_resolve_pass.set_pipeline(&pipelines.distortion_resolve_pipeline);
            distortion_resolve_pass.set_bind_group(0, &frame.distortion_resolve_bind_group, &[]);
            distortion_resolve_pass.draw(0..3, 0..1);
        }

        for (label, pipeline, bind_group, view) in [
            (
                "quicktag_model_bloom_half_pass",
                &pipelines.bloom_bright_pipeline,
                &frame.bloom_half_bind_group,
                &target.bloom_half_view,
            ),
            (
                "quicktag_model_bloom_quarter_pass",
                &pipelines.bloom_downsample_pipeline,
                &frame.bloom_quarter_bind_group,
                &target.bloom_quarter_view,
            ),
            (
                "quicktag_model_bloom_horizontal_pass",
                &pipelines.bloom_blur_horizontal_pipeline,
                &frame.bloom_blur_horizontal_bind_group,
                &target.bloom_blur_view,
            ),
            (
                "quicktag_model_bloom_vertical_pass",
                &pipelines.bloom_blur_vertical_pipeline,
                &frame.bloom_blur_vertical_bind_group,
                &target.bloom_quarter_view,
            ),
        ] {
            let mut bloom_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            bloom_pass.set_pipeline(pipeline);
            bloom_pass.set_bind_group(0, bind_group, &[]);
            bloom_pass.draw(0..3, 0..1);
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if self.native_resources_pending { return; }
        let Some(pipelines) = callback_resources.get::<ModelPipelineResources>() else {
            return;
        };
        let Some(frame) = callback_resources.get::<ModelFrameResources>() else {
            return;
        };
        render_pass.set_pipeline(&pipelines.present_pipeline);
        render_pass.set_bind_group(0, &frame.present_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

fn alpha_mode(format: wgpu::TextureFormat) -> f32 {
    match format {
        wgpu::TextureFormat::Bc4RUnorm | wgpu::TextureFormat::Bc4RSnorm => -1.0,
        wgpu::TextureFormat::Bc1RgbaUnorm | wgpu::TextureFormat::Bc1RgbaUnormSrgb => 0.5,
        _ => 0.0,
    }
}

fn material_texture_view(texture: &Texture, srgb: bool) -> wgpu::TextureView {
    create_model_texture_view(
        &texture.handle,
        &wgpu::TextureViewDescriptor {
            format: Some(if srgb {
                srgb_texture_format(texture.desc.format)
            } else {
                linear_texture_format(texture.desc.format)
            }),
            ..Default::default()
        },
    )
}

fn create_native_authored_sampler(device: &wgpu::Device, desc: &ModelSamplerDesc) -> wgpu::Sampler {
    // D3D11_FILTER_MIN_MAG_MIP_POINT = 0; 0x15/0x55 use linear filters.
    // Adapters apply each authored mip bias in the original sample instruction.
    let filter = match desc.filter {
        0 => wgpu::FilterMode::Nearest,
        0x15 | 0x55 => wgpu::FilterMode::Linear,
        value => panic!("unsupported native authored filter: {value}"),
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("exact native authored sampler; shader embeds mip bias"),
        address_mode_u: authored_sampler_address_mode(desc.address_u),
        address_mode_v: authored_sampler_address_mode(desc.address_v),
        address_mode_w: authored_sampler_address_mode(desc.address_w),
        mag_filter: filter, min_filter: filter, mipmap_filter: filter,
        lod_min_clamp: desc.min_lod, lod_max_clamp: desc.max_lod,
        border_color: authored_sampler_border_color(desc).expect("admitted sampler border contract"),
        anisotropy_clamp: desc.max_anisotropy as u16, ..Default::default()
    })
}

// D3D11_TEXTURE_ADDRESS_MODE: explicit mapping for audited source samplers.
fn authored_sampler_address_mode(mode: u32) -> wgpu::AddressMode {
    match mode {
        1 => wgpu::AddressMode::Repeat,
        2 => wgpu::AddressMode::MirrorRepeat,
        3 => wgpu::AddressMode::ClampToEdge,
        4 => wgpu::AddressMode::ClampToBorder,
        _ => panic!("unsupported authored sampler address mode: {mode}"),
    }
}

fn authored_sampler_border_color(desc: &ModelSamplerDesc) -> Result<Option<wgpu::SamplerBorderColor>, &'static str> {
    if ![desc.address_u, desc.address_v, desc.address_w].contains(&4) { return Ok(None); }
    Ok(Some(match desc.border_color {
        [0.0, 0.0, 0.0, 0.0] => wgpu::SamplerBorderColor::TransparentBlack,
        [0.0, 0.0, 0.0, 1.0] => wgpu::SamplerBorderColor::OpaqueBlack,
        [1.0, 1.0, 1.0, 1.0] => wgpu::SamplerBorderColor::OpaqueWhite,
        _ => return Err("unsupported authored sampler border color"),
    }))
}

fn load_model_sampler_desc(tag: TagHash) -> Option<ModelSamplerDesc> {
    let entry = package_manager().get_entry(tag)?;
    if entry.file_type != 34 || entry.file_subtype != 1 {
        return None;
    }
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    decode_model_sampler_desc(&data)
}

fn decode_model_sampler_desc(data: &[u8]) -> Option<ModelSamplerDesc> {
    let read_u32 = |offset: usize| {
        data.get(offset..offset + 4)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_le_bytes)
    };
    let read_f32 = |offset: usize| read_u32(offset).map(f32::from_bits);
    Some(ModelSamplerDesc {
        filter: read_u32(0)?,
        address_u: read_u32(4)?,
        address_v: read_u32(8)?,
        address_w: read_u32(12)?,
        mip_lod_bias: read_f32(16)?,
        max_anisotropy: read_u32(20)?.clamp(1, 16),
        border_color: [read_f32(28)?, read_f32(32)?, read_f32(36)?, read_f32(40)?],
        min_lod: read_f32(44)?,
        max_lod: read_f32(48)?,
    })
}

fn create_model_sampler(device: &wgpu::Device, desc: ModelSamplerDesc) -> wgpu::Sampler {
    let address_mode = |mode| match mode {
        1 => wgpu::AddressMode::Repeat,
        2 => wgpu::AddressMode::MirrorRepeat,
        4 => wgpu::AddressMode::ClampToBorder,
        // Mirror-once remains outside the native material admission contract.
        _ => wgpu::AddressMode::ClampToEdge,
    };
    let anisotropic = desc.filter & 0x40 != 0;
    let trilinear = desc.filter & 0x4 != 0 && desc.filter & 0x10 != 0 && desc.filter & 0x1 != 0;
    let high_quality_anisotropic = anisotropic || trilinear;
    let filter_mode = |linear| {
        if anisotropic || linear {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        }
    };
    let min_lod = desc.min_lod.max(0.0);
    let max_lod = if desc.max_lod.is_finite() {
        desc.max_lod.clamp(min_lod, 32.0)
    } else {
        32.0
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quicktag_model_authored_sampler"),
        address_mode_u: address_mode(desc.address_u),
        address_mode_v: address_mode(desc.address_v),
        address_mode_w: address_mode(desc.address_w),
        mag_filter: filter_mode(desc.filter & 0x4 != 0),
        min_filter: filter_mode(desc.filter & 0x10 != 0),
        mipmap_filter: filter_mode(desc.filter & 0x1 != 0),
        lod_min_clamp: min_lod,
        lod_max_clamp: max_lod,
        compare: None,
        anisotropy_clamp: if high_quality_anisotropic {
            // Viewer quality mode: retain authored filtering/addressing while
            // upgrading linear samplers to portable 16x on oblique surfaces.
            16
        } else {
            1
        },
        border_color: authored_sampler_border_color(&desc).expect("authored sampler border contract"),
    })
}

fn create_pipeline_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
) -> ModelPipelineResources {
    assert!(
        device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS),
        "model renderer requires authored SPIR-V passthrough support"
    );
    let authored_program_modules =
        crate::render::authored_program::AuthoredProgramModules::new(device);
    let authored_surface_vertex_layout = crate::render::body_vertex::create_layout(device);
    let authored_rigid_vertex_layout = crate::render::rigid_vertex::create_layout(device);
    let authored_auxiliary_vertex_layout = crate::render::auxiliary_vertex::create_layout(device);
    let authored_displacement_vertex_layout = crate::render::displacement_vertex::create_layout(device);
    let authored_color_vertex_layout = crate::render::color_vertex::create_layout(device);
    // Material pipelines are created once when a fully resident admitted draw
    // first needs them. Opening another model never compiles unrelated shaders.
    let authored_surface_pipelines = Vec::new();
    let authored_source_color = crate::render::surface_targets::SourceColorProjection::new(device, [SURFACE_FORMAT,OFFSCREEN_FORMAT]);
    let authored_viewer_material = crate::render::surface_targets::ViewerMaterialProjection::new(device,
        [SURFACE_FORMAT,SURFACE_PROPERTIES_FORMAT,SURFACE_FORMAT], SURFACE_EMISSIVE_FORMAT);
    let viewer_receiver_import = crate::render::surface_targets::ViewerReceiverImport::new(device);
    let viewer_receiver_coverage = crate::render::surface_targets::ViewerReceiverCoverage::new(device);
    let authored_body_compute_pipeline =
        create_authored_body_compute_pipeline(device, &authored_program_modules);
    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quicktag_model_scene_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let shadow_scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quicktag_model_shadow_scene_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let strict_shadow_transform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quicktag_model_strict_shadow_transform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quicktag_model_material_layout"),
        entries: &[
            texture_entry(0),
            texture_entry(1),
            texture_entry(2),
            texture_entry(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture_entry(6),
            texture_entry(7),
            texture_entry(8),
            texture_entry(9),
            texture_entry(10),
            texture_entry(11),
            texture_entry(12),
            texture_entry(13),
            texture_entry(14),
            texture_entry(15),
            texture_entry(16),
            texture_entry(17),
            wgpu::BindGroupLayoutEntry {
                binding: 18,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 19,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    // Stage-8 coating and stage-2 investment decals read deferred screen
    // attachments. Keep those feedback inputs isolated from ordinary material
    // bindings: both source textures are populated by earlier passes.
    let coating_deferred_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quicktag_model_coating_deferred_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
    let present_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quicktag_model_present_layout"),
        entries: &[
            texture_entry(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture_entry(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            texture_entry(5),
            texture_entry(6),
        ],
    });
    let bloom_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quicktag_model_bloom_layout"),
        entries: &[
            texture_entry(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quicktag_model_lighting_layout"),
        entries: &[
            texture_entry(0),
            texture_entry(1),
            texture_entry(2),
            texture_entry(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            texture_entry(7),
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11, visibility: wgpu::ShaderStages::FRAGMENT, count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
            },
        ],
    });
    let distortion_resolve_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quicktag_model_distortion_resolve_layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quicktag_model_material_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: 16,
        ..Default::default()
    });
    let present_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quicktag_model_present_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let fallback_color = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("quicktag_model_neutral_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::default(),
        &[255, 255, 255, 255],
    );
    let fallback_color_view = create_model_texture_view(&fallback_color, &Default::default());
    let fallback_cubemap = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("quicktag_default_environment_cubemap"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::default(),
        &[
            55, 110, 255, 255, 55, 110, 255, 255, 55, 110, 255, 255, 55, 110, 255, 255, 55, 110,
            255, 255, 55, 110, 255, 255,
        ],
    );
    let fallback_cubemap_view = create_model_texture_view(
        &fallback_cubemap,
        &wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        },
    );
    let cubemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quicktag_environment_cubemap_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let model_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_shader"),
        source: wgpu::ShaderSource::Wgsl(MODEL_SHADER.into()),
    });
    let model_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quicktag_model_pipeline_layout"),
        bind_group_layouts: &[&scene_layout, &material_layout, &coating_deferred_layout],
        push_constant_ranges: &[],
    });
    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quicktag_model_shadow_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        compare: Some(wgpu::CompareFunction::LessEqual),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_shadow_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });
    let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quicktag_model_shadow_pipeline_layout"),
        bind_group_layouts: &[&shadow_scene_layout, &material_layout],
        push_constant_ranges: &[],
    });
    let strict_shadow_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quicktag_model_strict_shadow_pipeline_layout"),
            bind_group_layouts: &[
                &shadow_scene_layout,
                &material_layout,
                &strict_shadow_transform_layout,
            ],
            push_constant_ranges: &[],
        });
    let authored_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_authored_depth_shader"),
        source: wgpu::ShaderSource::Wgsl(AUTHORED_DEPTH_SHADER.into()),
    });
    let strict_depth_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quicktag_model_strict_depth_pipeline_layout"),
            bind_group_layouts: &[&scene_layout, &strict_shadow_transform_layout],
            push_constant_ranges: &[],
        });
    let create_shadow_pipeline = |label, cull_mode| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                // Keep caster bias deliberately small. Depth32Float gives us
                // enough precision that a large raster bias only creates
                // visible Peter-Panning at contact edges. The receiver shader
                // handles sample-to-sample plane slope in shadow-depth space.
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shadow_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview: None,
            cache: None,
        })
    };
    let shadow_pipelines = [
        create_shadow_pipeline("quicktag_model_shadow_pipeline_two_sided", None),
        create_shadow_pipeline(
            "quicktag_model_shadow_pipeline_cull_front",
            Some(wgpu::Face::Front),
        ),
        create_shadow_pipeline(
            "quicktag_model_shadow_pipeline_cull_back",
            Some(wgpu::Face::Back),
        ),
    ];
    let create_depth_pipeline = |label, cull_mode| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&model_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &model_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[model_vertex_layout()],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: None,
            multiview: None,
            cache: None,
        })
    };
    let depth_pipelines = [
        create_depth_pipeline("quicktag_model_depth_pipeline_two_sided", None),
        create_depth_pipeline(
            "quicktag_model_depth_pipeline_cull_front",
            Some(wgpu::Face::Front),
        ),
        create_depth_pipeline(
            "quicktag_model_depth_pipeline_cull_back",
            Some(wgpu::Face::Back),
        ),
    ];
    let present_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_present_shader"),
        source: wgpu::ShaderSource::Wgsl(PRESENT_SHADER.into()),
    });
    let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_bloom_shader"),
        source: wgpu::ShaderSource::Wgsl(BLOOM_SHADER.into()),
    });
    let lighting_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_lighting_shader"),
        source: wgpu::ShaderSource::Wgsl(LIGHTING_SHADER.into()),
    });
    let distortion_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("quicktag_model_distortion_resolve_shader"),
        source: wgpu::ShaderSource::Wgsl(DISTORTION_RESOLVE_SHADER.into()),
    });
    let lighting_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quicktag_model_lighting_pipeline_layout"),
        bind_group_layouts: &[&lighting_layout],
        push_constant_ranges: &[],
    });
    let lighting_pipeline = create_bloom_pipeline(
        device,
        &lighting_shader,
        &lighting_pipeline_layout,
        "fs_main",
    );
    let distortion_resolve_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quicktag_model_distortion_resolve_pipeline_layout"),
            bind_group_layouts: &[&distortion_resolve_layout],
            push_constant_ranges: &[],
        });
    let distortion_resolve_pipeline = create_bloom_pipeline(
        device,
        &distortion_resolve_shader,
        &distortion_resolve_pipeline_layout,
        "fs_main",
    );
    let bloom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quicktag_model_bloom_pipeline_layout"),
        bind_group_layouts: &[&bloom_layout],
        push_constant_ranges: &[],
    });
    let bloom_bright_pipeline =
        create_bloom_pipeline(device, &bloom_shader, &bloom_pipeline_layout, "fs_bright");
    let bloom_downsample_pipeline = create_bloom_pipeline(
        device,
        &bloom_shader,
        &bloom_pipeline_layout,
        "fs_downsample",
    );
    let bloom_blur_horizontal_pipeline = create_bloom_pipeline(
        device,
        &bloom_shader,
        &bloom_pipeline_layout,
        "fs_blur_horizontal",
    );
    let bloom_blur_vertical_pipeline = create_bloom_pipeline(
        device,
        &bloom_shader,
        &bloom_pipeline_layout,
        "fs_blur_vertical",
    );
    let present_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("quicktag_model_present_pipeline_layout"),
        bind_group_layouts: &[&present_layout],
        push_constant_ranges: &[],
    });
    let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("quicktag_model_present_pipeline"),
        layout: Some(&present_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &present_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &present_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });

    ModelPipelineResources {
        target_format,
        scene_layout,
        shadow_scene_layout,
        strict_shadow_transform_layout,
        material_layout,
        coating_deferred_layout,
        present_layout,
        bloom_layout,
        lighting_layout,
        distortion_resolve_layout,
        material_sampler,
        present_sampler,
        _fallback_color: fallback_color,
        fallback_color_view,
        model_shader,
        _authored_program_modules: authored_program_modules,
        authored_surface_vertex_layout,
        authored_rigid_vertex_layout,
        authored_auxiliary_vertex_layout,
        authored_displacement_vertex_layout,
        authored_color_vertex_layout,
        authored_surface_pipelines,
        authored_source_color,
        authored_viewer_material,
        viewer_receiver_import,
        viewer_receiver_coverage,
        authored_compute_pipelines: vec![(crate::render::authored_program::DescriptorAbi::BodyMeshComputeStorage, authored_body_compute_pipeline)],
        authored_c827_pipelines: Vec::new(),
        authored_decal_pipelines: Vec::new(),
        authored_material_c827_pipelines: Vec::new(),
        shadow_shader,
        authored_depth_shader,
        model_pipeline_layout,
        strict_shadow_pipeline_layout,
        strict_depth_pipeline_layout,
        model_pipelines: Vec::new(),
        present_pipeline,
        lighting_pipeline,
        distortion_resolve_pipeline,
        bloom_bright_pipeline,
        bloom_downsample_pipeline,
        bloom_blur_horizontal_pipeline,
        bloom_blur_vertical_pipeline,
        shadow_pipelines,
        strict_shadow_pipelines: Vec::new(),
        strict_depth_pipelines: Vec::new(),
        depth_pipelines,
        shadow_sampler,
        _fallback_cubemap: fallback_cubemap,
        fallback_cubemap_view,
        cubemap_sampler,
    }
}

fn create_strict_shadow_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    descriptor: &AuthoredInputLayoutDescriptor,
    key: StrictShadowPipelineKey,
) -> Option<wgpu::RenderPipeline> {
    let position = descriptor
        .streams
        .iter()
        .flat_map(|stream| stream.elements.iter().map(move |element| (stream, element)))
        .find(|(_stream, element)| element.semantic == 0 && element.semantic_index == 0)?;
    let uv = descriptor
        .streams
        .iter()
        .flat_map(|stream| stream.elements.iter().map(move |element| (stream, element)))
        .find(|(_stream, element)| element.semantic == 5 && element.semantic_index == 0)?;

    let vertex_entry = match position.1.format {
        0x03 => "vs_authored_vec3",
        0x04 | 0x0B => "vs_authored_vec4",
        _ => return None,
    };
    authored_vertex_format(uv.1.format)?;

    let max_stream = position.0.stream_index.max(uv.0.stream_index);
    let mut stream_storage =
        Vec::<(u8, u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)>::new();
    for stream_index in 0..=max_stream {
        let stream = descriptor
            .streams
            .iter()
            .find(|stream| stream.stream_index == stream_index)?;
        let stride = u64::from(*key.stream_strides.get(usize::from(stream_index))?);
        if stride == 0 {
            return None;
        }
        let mut attributes = Vec::new();
        for element in &stream.elements {
            let shader_location = if element.semantic == 0 && element.semantic_index == 0 {
                Some(0)
            } else if element.semantic == 5 && element.semantic_index == 0 {
                Some(2)
            } else {
                None
            };
            let Some(shader_location) = shader_location else {
                continue;
            };
            let format = authored_vertex_format(element.format)?;
            let attribute_end =
                u64::from(element.offset) + authored_vertex_format_size(element.format)?;
            if attribute_end > stride {
                return None;
            }
            attributes.push(wgpu::VertexAttribute {
                format,
                offset: u64::from(element.offset),
                shader_location,
            });
        }
        stream_storage.push((
            stream_index,
            stride,
            if stream.instanced {
                wgpu::VertexStepMode::Instance
            } else {
                wgpu::VertexStepMode::Vertex
            },
            attributes,
        ));
    }
    let vertex_layouts = stream_storage
        .iter()
        .map(
            |(_stream_index, stride, step_mode, attributes)| wgpu::VertexBufferLayout {
                array_stride: *stride,
                step_mode: *step_mode,
                attributes,
            },
        )
        .collect_vec();

    let (topology, strip_index_format) = match key.primitive_type {
        3 => (wgpu::PrimitiveTopology::TriangleList, None),
        5 => (
            wgpu::PrimitiveTopology::TriangleStrip,
            Some(if key.index_32bit {
                wgpu::IndexFormat::Uint32
            } else {
                wgpu::IndexFormat::Uint16
            }),
        ),
        _ => return None,
    };
    let label = format!(
        "quicktag_strict_shadow_layout{}_primitive{}_raster{}",
        key.layout_id, key.primitive_type, key.rasterizer
    );
    Some(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: Default::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format,
                cull_mode: rasterizer_cull_mode(key.rasterizer),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                // Tiger ShadowGenerate baseline: authored depth-bias preset 6.
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview: None,
            cache: None,
        }),
    )
}

fn create_strict_depth_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    descriptor: &AuthoredInputLayoutDescriptor,
    key: StrictDepthPipelineKey,
) -> Option<wgpu::RenderPipeline> {
    let position = descriptor
        .streams
        .iter()
        .flat_map(|stream| stream.elements.iter().map(move |element| (stream, element)))
        .find(|(_stream, element)| element.semantic == 0 && element.semantic_index == 0)?;

    let vertex_entry = match position.1.format {
        0x03 => "vs_authored_depth_vec3",
        0x04 | 0x0B => "vs_authored_depth_vec4",
        _ => return None,
    };

    let max_stream = position.0.stream_index;
    let mut stream_storage =
        Vec::<(u8, u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)>::new();
    for stream_index in 0..=max_stream {
        let stream = descriptor
            .streams
            .iter()
            .find(|stream| stream.stream_index == stream_index)?;
        let stride = u64::from(*key.stream_strides.get(usize::from(stream_index))?);
        if stride == 0 {
            return None;
        }
        let mut attributes = Vec::new();
        for element in &stream.elements {
            if element.semantic != 0 || element.semantic_index != 0 {
                continue;
            }
            let format = authored_vertex_format(element.format)?;
            let attribute_end =
                u64::from(element.offset) + authored_vertex_format_size(element.format)?;
            if attribute_end > stride {
                return None;
            }
            attributes.push(wgpu::VertexAttribute {
                format,
                offset: u64::from(element.offset),
                shader_location: 0,
            });
        }
        stream_storage.push((
            stream_index,
            stride,
            if stream.instanced {
                wgpu::VertexStepMode::Instance
            } else {
                wgpu::VertexStepMode::Vertex
            },
            attributes,
        ));
    }
    let vertex_layouts = stream_storage
        .iter()
        .map(
            |(_stream_index, stride, step_mode, attributes)| wgpu::VertexBufferLayout {
                array_stride: *stride,
                step_mode: *step_mode,
                attributes,
            },
        )
        .collect_vec();

    let (topology, strip_index_format) = match key.primitive_type {
        3 => (wgpu::PrimitiveTopology::TriangleList, None),
        5 => (
            wgpu::PrimitiveTopology::TriangleStrip,
            Some(if key.index_32bit {
                wgpu::IndexFormat::Uint32
            } else {
                wgpu::IndexFormat::Uint16
            }),
        ),
        _ => return None,
    };
    let label = format!(
        "quicktag_strict_depth_layout{}_primitive{}_raster{}_depth{}_bias{}",
        key.layout_id, key.primitive_type, key.rasterizer, key.depth_stencil, key.depth_bias
    );
    Some(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: Default::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format,
                cull_mode: rasterizer_cull_mode(key.rasterizer),
                ..Default::default()
            },
            depth_stencil: depth_stencil_state(key.depth_stencil, key.depth_bias),
            multisample: Default::default(),
            fragment: None,
            multiview: None,
            cache: None,
        }),
    )
}

fn create_bloom_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("quicktag_model_bloom_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: OFFSCREEN_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn create_authored_body_compute_pipeline(
    device: &wgpu::Device,
    modules: &crate::render::authored_program::AuthoredProgramModules,
) -> crate::render::body_mesh::BodyMeshProducer {
    use crate::render::authored_program::DescriptorAbi;

    crate::render::body_mesh::BodyMeshProducer::new(
        device,
        modules
            .module(DescriptorAbi::BodyMeshComputeStorage)
            .expect("embedded body mesh compute program"),
    )
}

fn create_authored_surface_pipelines(
    device: &wgpu::Device,
    modules: &crate::render::authored_program::AuthoredProgramModules,
    vertex_layout: &wgpu::BindGroupLayout,
    rigid_vertex_layout: &wgpu::BindGroupLayout,
    auxiliary_vertex_layout: &wgpu::BindGroupLayout,
    color_vertex_layout: &wgpu::BindGroupLayout,
    displacement_vertex_layout: &wgpu::BindGroupLayout,
    requested: &[(crate::render::authored_program::DescriptorAbi, crate::render::authored_program::DescriptorAbi, u8, wgpu::IndexFormat)],
) -> Vec<((crate::render::authored_program::DescriptorAbi, crate::render::authored_program::DescriptorAbi, u8, wgpu::IndexFormat), crate::render::body_draw::BodyDrawPipeline)> {
    use crate::render::{authored_program::DescriptorAbi as D, dense_pixel::SurfacePixelAbi as S};
    let contracts=[(D::BodyPixelDense,S::Body),(D::ChestPixelDense,S::Layered),(D::SleevesPixelDense,S::Layered),
        (D::HandsPixelDense,S::Hands),(D::HardwarePixelDense,S::Hardware),
        (D::FacePixelDense,S::Audited {rows:127,textures:7,volume:Some(5),cube:None,samplers:3}),
        (D::EyeDetailPixelDense,S::Audited {rows:71,textures:2,volume:Some(1),cube:None,samplers:2})].into_iter().map(|(ps,abi)|(D::BodyVertexScalarStorage,ps,abi)).chain(
            crate::render::runner_surface_programs::SURFACES.iter().flat_map(|s|s.vertex_programs.iter().map(move |&vertex|(vertex,s.program.descriptor_abi,
                S::Audited { rows:s.rows as u64,textures:s.texture_count,volume:s.volume_slot,cube:s.cube_slot,samplers:s.sampler_count })))).chain(
            [(D::HairVertexStorage, D::HairPixelDense,
              S::Audited { rows:167, textures:5, volume:Some(4), cube:None, samplers:2 }),
             (D::HairVertexStorage, D::HairSolidPixelDense,
              S::Audited { rows:170, textures:6, volume:Some(5), cube:None, samplers:2 })]);
    contracts.flat_map(|(vertex,program,abi)|[1u8,2].into_iter().flat_map(move |rasterizer|[wgpu::IndexFormat::Uint16,wgpu::IndexFormat::Uint32].map(|format|(vertex,program,abi,rasterizer,format))))
        .filter(|(vertex,program,_,rasterizer,format)|requested.contains(&(*vertex,*program,*rasterizer,*format)))
        .map(|(vertex,program,abi,rasterizer,format)| {
    let build=if matches!(vertex,D::RigidVertexDirectIa|D::FloatVertexScalarStorage) {crate::render::body_draw::BodyDrawPipeline::new_direct_ia}
        else {crate::render::body_draw::BodyDrawPipeline::new_dense};
    ((vertex,program,rasterizer,format), build(
        device,
        modules.module(vertex).expect("embedded audited layout7 vertex program"),
        modules.module(program).expect("embedded audited surface pixel program"),
        match vertex {
            D::RigidVertexDirectIa => rigid_vertex_layout,
            D::VertexColorStorage => color_vertex_layout,
            D::DisplacementEC03VertexStorage => displacement_vertex_layout,
            D::HairVertexStorage | D::Hair14580VertexStorage | D::Hair120RowVertexStorage | D::HairBA67VertexStorage
                | D::BodyProceduralVertexStorage | D::BodyC9C5VertexStorage | D::ClothVertexStorage | D::ClothB152VertexStorage => auxiliary_vertex_layout,
            _ => vertex_layout,
        }, abi,
        std::array::from_fn(|_|Some(surface_target(wgpu::TextureFormat::Rgba32Float))),
        wgpu::FrontFace::Ccw, (rasterizer==2).then_some(wgpu::Face::Back), depth_stencil_state(2, 0),format,
    )) }).collect()
}

fn create_authored_c827_pipeline(
    device: &wgpu::Device,
    modules: &crate::render::authored_program::AuthoredProgramModules,
    index_format: wgpu::IndexFormat,
) -> crate::render::c827_draw::C827DrawPipeline {
    use crate::render::authored_program::DescriptorAbi;
    crate::render::c827_draw::C827DrawPipeline::new(
        device,
        modules.module(DescriptorAbi::SharedLayout7VertexScalarStorage).expect("embedded C827 VS"),
        modules.module(DescriptorAbi::C827Pixel).expect("embedded C827 PS"),
        [OFFSCREEN_FORMAT, SURFACE_FORMAT, SURFACE_PROPERTIES_FORMAT, SURFACE_FORMAT],
        depth_stencil_state(15,1), rasterizer_cull_mode(2),index_format,
    )
}



#[allow(clippy::too_many_arguments)]

fn create_model_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    key: ModelPipelineKey,
) -> wgpu::RenderPipeline {
    let transparent = is_forward_pass(key.pass);
    let emissive_only = key.pass == RenderPassKind::MaterialEmissive;
    let flags_only = key.pass == RenderPassKind::MaterialFlags;
    let targets = if is_distortion_payload_pass(key.pass) {
        vec![Some(wgpu::ColorTargetState {
            format: DISTORTION_FORMAT,
            blend: blend_state(key.blend),
            write_mask: wgpu::ColorWrites::ALL,
        })]
    } else if transparent {
        vec![Some(wgpu::ColorTargetState {
            format: OFFSCREEN_FORMAT,
            blend: blend_state(key.blend),
            write_mask: wgpu::ColorWrites::ALL,
        })]
    } else if emissive_only {
        vec![Some(surface_target(SURFACE_EMISSIVE_FORMAT))]
    } else if flags_only {
        vec![Some(surface_target(SURFACE_FLAGS_FORMAT))]
    } else {
        vec![
            Some(wgpu::ColorTargetState {
                format: OFFSCREEN_FORMAT,
                blend: blend_state(key.blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(surface_target_with_mask(
                SURFACE_FORMAT,
                normal_surface_write_mask(key.pass),
            )),
            Some(surface_target(SURFACE_PROPERTIES_FORMAT)),
            Some(surface_target(SURFACE_FORMAT)),
        ]
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("quicktag_model_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 24,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 32,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 48,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 52,
                        shader_location: 5,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 64,
                        shader_location: 6,
                    },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: rasterizer_cull_mode(key.rasterizer),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: if flags_only {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            })
        } else {
            depth_stencil_state(key.depth_stencil, key.depth_bias)
        },
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(match key.pass {
                RenderPassKind::DecalCompatibility => "fs_main",
                RenderPassKind::InvestmentDecalCompatibility => "fs_investment_decal",
                RenderPassKind::ForwardAdditive => "fs_forward_transparent",
                RenderPassKind::ForwardTransparent => "fs_forward_transparent",
                RenderPassKind::ForwardCoating => "fs_forward_coating",
                RenderPassKind::Distortion => "fs_distortion",
                RenderPassKind::MaterialEmissive => "fs_material_emissive",
                RenderPassKind::MaterialFlags => "fs_material_flags",
                _ => "fs_main",
            }),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview: None,
        cache: None,
    })
}

fn model_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x4,
        4 => Float32,
        5 => Float32x3,
        6 => Float32x3
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

fn surface_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    surface_target_with_mask(format, wgpu::ColorWrites::ALL)
}

fn surface_target_with_mask(
    format: wgpu::TextureFormat,
    write_mask: wgpu::ColorWrites,
) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask,
    }
}

fn normal_surface_write_mask(pass: RenderPassKind) -> wgpu::ColorWrites {
    if matches!(
        pass,
        RenderPassKind::DecalCompatibility | RenderPassKind::InvestmentDecalCompatibility
    ) {
        // Decal RT1 alpha has no authored roughness override. Preserve opaque
        // roughness while still replacing RT1 RGB with the decal normal.
        wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE
    } else {
        wgpu::ColorWrites::ALL
    }
}

fn rasterizer_cull_mode(index: u8) -> Option<wgpu::Face> {
    match index {
        0 | 1 | 5 | 7 => None,
        3 | 8 => Some(wgpu::Face::Front),
        _ => Some(wgpu::Face::Back),
    }
}

fn shadow_pipeline_index(rasterizer: u8) -> usize {
    match rasterizer_cull_mode(rasterizer) {
        None => 0,
        Some(wgpu::Face::Front) => 1,
        Some(wgpu::Face::Back) => 2,
    }
}

/// Viewer blend for refractive glass: the fragment is the transmittance that
/// multiplies the lit scene behind it. Not an authored Tiger blend index.
const ABSORPTION_BLEND: u8 = 250;

fn blend_state(index: u8) -> Option<wgpu::BlendState> {
    let component = |src_factor, dst_factor| wgpu::BlendComponent {
        src_factor,
        dst_factor,
        operation: wgpu::BlendOperation::Add,
    };
    match index {
        0 | 1 | 57 => None,
        2 => Some(wgpu::BlendState {
            color: component(wgpu::BlendFactor::One, wgpu::BlendFactor::One),
            alpha: component(wgpu::BlendFactor::One, wgpu::BlendFactor::One),
        }),
        8 => Some(wgpu::BlendState {
            color: component(wgpu::BlendFactor::One, wgpu::BlendFactor::OneMinusSrcAlpha),
            alpha: component(wgpu::BlendFactor::One, wgpu::BlendFactor::OneMinusSrcAlpha),
        }),
        26 | 27 => Some(wgpu::BlendState {
            color: component(wgpu::BlendFactor::One, wgpu::BlendFactor::SrcAlpha),
            alpha: component(wgpu::BlendFactor::Zero, wgpu::BlendFactor::One),
        }),
        76 => Some(wgpu::BlendState {
            color: component(wgpu::BlendFactor::Dst, wgpu::BlendFactor::Zero),
            alpha: component(wgpu::BlendFactor::Zero, wgpu::BlendFactor::One),
        }),
        ABSORPTION_BLEND => Some(wgpu::BlendState {
            color: component(wgpu::BlendFactor::Zero, wgpu::BlendFactor::Src),
            alpha: component(wgpu::BlendFactor::Zero, wgpu::BlendFactor::One),
        }),
        _ => Some(wgpu::BlendState::ALPHA_BLENDING),
    }
}

fn blend_enabled(index: u8) -> bool {
    !matches!(index, 0 | 1 | 57)
}

fn is_forward_pass(pass: RenderPassKind) -> bool {
    matches!(
        pass,
        RenderPassKind::ForwardAdditive
            | RenderPassKind::ForwardTransparent
            | RenderPassKind::ForwardCoating
            | RenderPassKind::Distortion
    )
}

fn is_distortion_payload_pass(pass: RenderPassKind) -> bool {
    pass == RenderPassKind::Distortion
}

fn depth_stencil_state(index: u8, depth_bias: u8) -> Option<wgpu::DepthStencilState> {
    const DEPTH_INDICES: [u8; 89] = [
        0, 1, 2, 8, 2, 1, 1, 2, 2, 2, 2, 2, 2, 4, 6, 3, 7, 3, 9, 3, 3, 7, 7, 3, 3, 3, 6, 2, 3, 3,
        3, 1, 1, 1, 10, 11, 3, 12, 1, 1, 1, 3, 2, 6, 3, 3, 3, 3, 3, 3, 13, 1, 3, 7, 13, 13, 9, 3,
        1, 3, 1, 3, 1, 3, 3, 2, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 5, 5, 10, 1,
        14,
    ];
    let depth = DEPTH_INDICES.get(index as usize).copied().unwrap_or(2);
    let (write, compare) = match depth {
        0 | 1 => return None,
        2 | 8 => (true, wgpu::CompareFunction::LessEqual),
        3 | 9 | 13 => (false, wgpu::CompareFunction::LessEqual),
        4 => (true, wgpu::CompareFunction::GreaterEqual),
        5 => (true, wgpu::CompareFunction::Greater),
        6 => (false, wgpu::CompareFunction::GreaterEqual),
        7 => (false, wgpu::CompareFunction::Greater),
        10 => (true, wgpu::CompareFunction::Always),
        11 => (false, wgpu::CompareFunction::Never),
        12 => (false, wgpu::CompareFunction::Always),
        14 => (false, wgpu::CompareFunction::Equal),
        _ => (true, wgpu::CompareFunction::LessEqual),
    };
    Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: write,
        depth_compare: compare,
        stencil: Default::default(),
        bias: depth_bias_state(depth_bias),
    })
}

fn depth_bias_state(index: u8) -> wgpu::DepthBiasState {
    let values = [
        (0, 0.0),
        (0, 0.0),
        (5, 2.0),
        (10, 4.0),
        (15, 6.0),
        (20, 8.0),
        (2, 2.0),
        (-1, -2.0),
        (51, 2.0),
    ];
    let (constant, slope_scale) = values.get(index as usize).copied().unwrap_or_default();
    wgpu::DepthBiasState {
        constant,
        slope_scale,
        clamp: if constant == 0 { 0.0 } else { 10_000_000_000.0 },
    }
}

fn create_target_resources(
    device: &wgpu::Device,
    size: [u32; 2],
    features: ModelTargetFeatures,
) -> ModelTargetResources {
    debug_assert!(
        estimated_model_target_bytes(size, features) <= MAX_MODEL_TARGET_BYTES,
        "model target allocation exceeded the renderer memory budget"
    );
    let native_surface=features.native_surface.then(||crate::render::surface_targets::SurfaceTargets::new(device,size));
    let mixed_receivers=features.mixed_receivers.then(||crate::render::surface_targets::ReceiverTargets::new(device,size));
    let extent = wgpu::Extent3d {
        width: size[0],
        height: size[1],
        depth_or_array_layers: 1,
    };
    let native_normal_snapshot=features.native_decals.then(|| {
        let texture=device.create_texture(&wgpu::TextureDescriptor {
            label:Some("authored opaque normal snapshot before stage2"),size:extent,mip_level_count:1,sample_count:1,
            dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba32Float,
            usage:wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::COPY_SRC|wgpu::TextureUsages::TEXTURE_BINDING,view_formats:&[],
        });
        let view=texture.create_view(&Default::default()); (texture,view)
    });
    let optional_extent = |enabled: bool| {
        if enabled {
            extent
        } else {
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            }
        }
    };
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_color_target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = create_model_texture_view(&color, &Default::default());
    let lit_color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_lit_color_target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let lit_color_view = create_model_texture_view(&lit_color, &Default::default());
    let scene_color_copy = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_scene_color_copy"),
        size: optional_extent(features.distortion),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let scene_color_copy_view = create_model_texture_view(&scene_color_copy, &Default::default());
    let scene_normal_copy = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_scene_normal_copy"),
        size: optional_extent(features.deferred_normal_copy),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SURFACE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let scene_normal_copy_view = create_model_texture_view(&scene_normal_copy, &Default::default());
    let distortion = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_distortion_payload"),
        size: optional_extent(features.distortion),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DISTORTION_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let distortion_view = create_model_texture_view(&distortion, &Default::default());
    let surface_texture = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let surface_normal = surface_texture("quicktag_surface_normal_roughness", SURFACE_FORMAT);
    let surface_normal_view = create_model_texture_view(&surface_normal, &Default::default());
    let surface_properties = surface_texture(
        "quicktag_surface_material_properties",
        SURFACE_PROPERTIES_FORMAT,
    );
    let surface_properties_view =
        create_model_texture_view(&surface_properties, &Default::default());
    let surface_emissive = surface_texture("quicktag_surface_emissive", SURFACE_EMISSIVE_FORMAT);
    let surface_emissive_view = create_model_texture_view(&surface_emissive, &Default::default());
    let surface_albedo = surface_texture("quicktag_surface_albedo_opacity", SURFACE_FORMAT);
    let surface_albedo_view = create_model_texture_view(&surface_albedo, &Default::default());
    let surface_flags = surface_texture("quicktag_surface_flags", SURFACE_FLAGS_FORMAT);
    let surface_flags_view = create_model_texture_view(&surface_flags, &Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_depth_target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = create_model_texture_view(&depth, &Default::default());
    let shadow_depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quicktag_model_sun_shadow"),
        size: wgpu::Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_depth_view = create_model_texture_view(&shadow_depth, &Default::default());
    let bloom_texture = |label, width, height| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let bloom_half_size = if features.bloom {
        [(size[0] / 2).max(1), (size[1] / 2).max(1)]
    } else {
        [1, 1]
    };
    let bloom_quarter_size = if features.bloom {
        [(size[0] / 4).max(1), (size[1] / 4).max(1)]
    } else {
        [1, 1]
    };
    let bloom_half = bloom_texture(
        "quicktag_model_bloom_half",
        bloom_half_size[0],
        bloom_half_size[1],
    );
    let bloom_half_view = create_model_texture_view(&bloom_half, &Default::default());
    let bloom_quarter = bloom_texture(
        "quicktag_model_bloom_quarter",
        bloom_quarter_size[0],
        bloom_quarter_size[1],
    );
    let bloom_quarter_view = create_model_texture_view(&bloom_quarter, &Default::default());
    let bloom_blur = bloom_texture(
        "quicktag_model_bloom_blur_ping_pong",
        bloom_quarter_size[0],
        bloom_quarter_size[1],
    );
    let bloom_blur_view = create_model_texture_view(&bloom_blur, &Default::default());
    ModelTargetResources {
        native_surface,
        native_normal_snapshot,
        mixed_receivers,
        size,
        features,
        _color: color,
        color_view,
        lit_color,
        lit_color_view,
        scene_color_copy,
        scene_color_copy_view,
        scene_normal_copy,
        scene_normal_copy_view,
        _distortion: distortion,
        distortion_view,
        _surface_normal: surface_normal,
        surface_normal_view,
        _surface_properties: surface_properties,
        surface_properties_view,
        _surface_emissive: surface_emissive,
        surface_emissive_view,
        _surface_albedo: surface_albedo,
        surface_albedo_view,
        _surface_flags: surface_flags,
        surface_flags_view,
        _depth: depth,
        depth_view,
        _shadow_depth: shadow_depth,
        shadow_depth_view,
        _bloom_half: bloom_half,
        bloom_half_view,
        _bloom_quarter: bloom_quarter,
        bloom_quarter_view,
        _bloom_blur: bloom_blur,
        bloom_blur_view,
    }
}

fn clear_surface_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn load_surface_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    }
}

const LIGHTING_SHADER: &str = r#"
@group(0) @binding(0) var compatibility_hdr: texture_2d<f32>;
@group(0) @binding(1) var normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var material_properties: texture_2d<f32>;
@group(0) @binding(3) var surface_emissive: texture_2d<f32>;
@group(0) @binding(4) var surface_sampler: sampler;
@group(0) @binding(5) var<uniform> scene: SceneUniform;
@group(0) @binding(6) var surface_flags: texture_2d<u32>;
@group(0) @binding(7) var surface_albedo: texture_2d<f32>;
@group(0) @binding(8) var scene_depth: texture_depth_2d;
@group(0) @binding(9) var sun_shadow: texture_depth_2d;
@group(0) @binding(10) var sun_shadow_sampler: sampler_comparison;
@group(0) @binding(11) var authored_coverage: texture_2d<f32>;

struct SceneUniform {
    center: vec4<f32>, params0: vec4<f32>, params1: vec4<f32>, uv_transform: vec4<f32>,
    light_direction: vec4<f32>, light_parameters: vec4<f32>, light_position: vec4<f32>, postprocess0: vec4<f32>, postprocess1: vec4<f32>, postprocess2: vec4<f32>, postprocess3: vec4<f32>, postprocess4: vec4<f32>, postprocess5: vec4<f32>, fidelity: vec4<f32>, shadow_parameters: vec4<f32>, light_color: vec4<f32>, ambient_sky: vec4<f32>, ambient_ground: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return output;
}

fn view_direction_to_world(value: vec3<f32>) -> vec3<f32> {
    let cy = cos(scene.params0.y);
    let sy = sin(scene.params0.y);
    let cp = cos(scene.params0.z);
    let sp = sin(scene.params0.z);
    let yawed = vec3<f32>(
        value.x,
        value.y * cp + value.z * sp,
        -value.y * sp + value.z * cp,
    );
    let z_up = vec3<f32>(
        yawed.x * cy - yawed.z * sy,
        yawed.y,
        yawed.x * sy + yawed.z * cy,
    );
    return vec3<f32>(z_up.x, z_up.z, z_up.y);
}

fn world_direction_to_view(value: vec3<f32>) -> vec3<f32> {
    let z_up = vec3<f32>(value.x, value.z, value.y);
    let cy = cos(scene.params0.y);
    let sy = sin(scene.params0.y);
    let cp = cos(scene.params0.z);
    let sp = sin(scene.params0.z);
    let yawed = vec3<f32>(
        z_up.x * cy + z_up.z * sy,
        z_up.y,
        -z_up.x * sy + z_up.z * cy,
    );
    return vec3<f32>(
        yawed.x,
        yawed.y * cp - yawed.z * sp,
        yawed.y * sp + yawed.z * cp,
    );
}

fn reconstruct_object_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let scale = 0.84 * scene.params0.w / max(scene.params0.x, 0.0001);
    let view_position = vec3<f32>(
        -(clip.x - scene.params1.y) / max(scale * scene.params1.x, 0.0001),
        (clip.y - scene.params1.z) / max(scale, 0.0001),
        (0.5 - depth) * max(scene.params0.x, 0.0001) / 0.21,
    );
    return scene.center.xyz + view_direction_to_world(view_position);
}

fn light_clip(position: vec3<f32>) -> vec4<f32> {
    // Tiger/Alkahest-style directional shadow space. Marathon's exact
    // authored world-to-shadow matrix is still a follow-up target, but this
    // affine projection preserves the engine contract and avoids the old
    // finite-spotlight perspective distortion that produced shadow spikes.
    let light = normalize(view_direction_to_world(scene.light_direction.xyz));
    let up = select(
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(light.z) > 0.95,
    );
    let right = normalize(cross(up, light));
    let vertical = cross(light, right);
    let relative = (position - scene.center.xyz) / max(scene.center.w, 0.0001);
    return vec4<f32>(
        dot(relative, right),
        dot(relative, vertical),
        0.5 - dot(relative, light) * 0.5,
        1.0,
    );
}

fn spotlight_factor_world(position: vec3<f32>) -> f32 {
    let light_position = scene.center.xyz + view_direction_to_world(scene.light_position.xyz);
    let source_to_surface = position - light_position;
    let distance = length(source_to_surface);
    let source_to_surface_direction = source_to_surface / max(distance, 0.0001 * scene.light_position.w);
    let beam_axis = normalize(-view_direction_to_world(scene.light_direction.xyz));
    let cone = smoothstep(
        scene.light_parameters.z,
        scene.light_parameters.w,
        dot(source_to_surface_direction, beam_axis),
    );
    let range = max(scene.light_parameters.y, 0.05 * scene.light_position.w);
    let range_fade = 1.0 - smoothstep(range * 0.70, range, distance);
    let rig_distance = distance / scene.light_position.w;
    let distance_falloff = 1.0 / max(1.0, rig_distance * rig_distance);
    return cone * range_fade * distance_falloff;
}

const STRICT_SHADOW_SAMPLE_COUNT = 9u;
const STRICT_SHADOW_DISK = array<vec2<f32>, 9>(
    vec2<f32>( 0.0,  0.0),
    vec2<f32>(-0.45570818, -0.80781116),
    vec2<f32>( 0.89600897, -0.35694158),
    vec2<f32>(-0.19807512,  0.92104191),
    vec2<f32>( 0.25952172, -0.80905122),
    vec2<f32>( 0.59722185,  0.77891493),
    vec2<f32>(-0.80889258,  0.45309126),
    vec2<f32>(-0.91679038, -0.21303266),
    vec2<f32>( 0.67728304,  0.18318819),
);

fn strict_tiger_shadow(shadow_position: vec3<f32>, position_screen: vec2<f32>) -> f32 {
    let dimensions = vec2<f32>(textureDimensions(sun_shadow));
    let texel = 1.0 / dimensions;
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let angle = fract(magic.z * fract(dot(position_screen, magic.xy))) * 6.28318530718;
    let sine = sin(angle);
    let cosine = cos(angle);
    var visibility = 0.0;
    for (var sample = 0u; sample < STRICT_SHADOW_SAMPLE_COUNT; sample++) {
        let poisson = STRICT_SHADOW_DISK[sample];
        let rotated = vec2<f32>(
            cosine * poisson.x - sine * poisson.y,
            sine * poisson.x + cosine * poisson.y,
        );
        let sample_uv = shadow_position.xy + rotated * texel * 2.5;
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            visibility += 1.0;
            continue;
        }
        visibility += textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            sample_uv,
            shadow_position.z - 0.0001,
        );
    }
    return visibility / f32(STRICT_SHADOW_SAMPLE_COUNT);
}

const SHADOW_SAMPLE_COUNT = 32u;
const SHADOW_BLOCKER_SAMPLE_COUNT = 16u;
const SHADOW_DISK = array<vec2<f32>, 32>(
    vec2<f32>( 0.1250,  0.0000), vec2<f32>(-0.1596,  0.1462),
    vec2<f32>( 0.0244, -0.2784), vec2<f32>( 0.2012,  0.2625),
    vec2<f32>(-0.3693, -0.0653), vec2<f32>( 0.3498, -0.2225),
    vec2<f32>(-0.1170,  0.4352), vec2<f32>(-0.2231, -0.4296),
    vec2<f32>( 0.4841,  0.1768), vec2<f32>(-0.5036,  0.2079),
    vec2<f32>( 0.2428, -0.5188), vec2<f32>( 0.1794,  0.5720),
    vec2<f32>(-0.5408, -0.3134), vec2<f32>( 0.6344, -0.1395),
    vec2<f32>(-0.3871,  0.5507), vec2<f32>(-0.0894, -0.6902),
    vec2<f32>( 0.5491,  0.4628), vec2<f32>(-0.7389,  0.0306),
    vec2<f32>( 0.5390, -0.5363), vec2<f32>(-0.0361,  0.7798),
    vec2<f32>(-0.5128, -0.6145), vec2<f32>( 0.8124,  0.1093),
    vec2<f32>(-0.6883,  0.4789), vec2<f32>( 0.1881, -0.8361),
    vec2<f32>( 0.4350,  0.7592), vec2<f32>(-0.8504, -0.2713),
    vec2<f32>( 0.8261, -0.3817), vec2<f32>(-0.3579,  0.8552),
    vec2<f32>(-0.3194, -0.8880), vec2<f32>( 0.8499,  0.4467),
    vec2<f32>(-0.9440,  0.2488), vec2<f32>( 0.5366, -0.8345),
);

fn shadow_linear_depth(depth: f32) -> f32 {
    let near_plane = scene.shadow_parameters.y;
    let range = max(scene.shadow_parameters.z, near_plane + 0.01 * scene.light_position.w);
    let depth_scale = range / max(range - near_plane, 0.01 * scene.light_position.w);
    return depth_scale * near_plane / max(depth_scale - depth, 0.000001);
}

fn shadow_receiver_gradient(shadow_position: vec3<f32>) -> vec2<f32> {
    let dx = dpdx(shadow_position);
    let dy = dpdy(shadow_position);
    let determinant = dx.x * dy.y - dx.y * dy.x;
    if abs(determinant) < 0.00000001 {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(
        (dx.z * dy.y - dy.z * dx.y) / determinant,
        (dx.x * dy.z - dy.x * dx.z) / determinant,
    );
}

fn shadow_reference_depth(
    shadow_position: vec3<f32>,
    sample_offset: vec2<f32>,
    depth_gradient: vec2<f32>,
    n_dot_light: f32,
) -> f32 {
    // Receiver-plane bias follows the actual perspective depth slope at the
    // sample location. The remaining epsilon is only a few Depth32Float ULPs;
    // it is intentionally unrelated to shadow-map XY texel size.
    let epsilon = mix(0.00000025, 0.00000125, 1.0 - clamp(n_dot_light, 0.0, 1.0));
    return shadow_position.z + dot(depth_gradient, sample_offset) - epsilon;
}

fn pcss_shadow(shadow_position: vec3<f32>, n_dot_light: f32) -> f32 {
    let shadow_dimensions = vec2<i32>(textureDimensions(sun_shadow));
    let shadow_texel = 1.0 / vec2<f32>(shadow_dimensions);
    let depth_gradient = shadow_receiver_gradient(shadow_position);
    let softness = clamp(scene.params1.w, 0.0, 1.0);
    let center_reference = shadow_reference_depth(
        shadow_position,
        vec2<f32>(0.0),
        depth_gradient,
        n_dot_light,
    );
    if softness <= 0.001 {
        return textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            shadow_position.xy,
            center_reference,
        );
    }

    let receiver_depth = shadow_linear_depth(shadow_position.z);
    let outer_cosine = clamp(scene.light_parameters.z, 0.01, 0.9999);
    let cone_tangent =
        sqrt(max(1.0 - outer_cosine * outer_cosine, 0.0)) / outer_cosine;
    // Treat the softness control as an area-light radius in model-scale units.
    // This is deliberately small: contact hardening, not a screen-space blur,
    // determines how wide the final penumbra becomes.
    let source_radius = scene.shadow_parameters.x;
    let max_search_radius =
        max(shadow_texel.x, shadow_texel.y) * 12.0;
    let search_radius = min(
        0.5 * source_radius
            / max(receiver_depth * cone_tangent, 0.000001),
        max_search_radius,
    );

    var blocker_depth_sum = 0.0;
    var blocker_count = 0u;
    for (var sample = 0u; sample < SHADOW_BLOCKER_SAMPLE_COUNT; sample++) {
        let offset = SHADOW_DISK[sample] * search_radius;
        let sample_uv = shadow_position.xy + offset;
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            continue;
        }
        let sample_pixel = clamp(
            vec2<i32>(sample_uv * vec2<f32>(shadow_dimensions)),
            vec2<i32>(0),
            shadow_dimensions - vec2<i32>(1),
        );
        let blocker_depth = textureLoad(sun_shadow, sample_pixel, 0);
        let reference =
            shadow_reference_depth(shadow_position, offset, depth_gradient, n_dot_light);
        if blocker_depth < reference {
            blocker_depth_sum += shadow_linear_depth(blocker_depth);
            blocker_count += 1u;
        }
    }
    if blocker_count == 0u {
        return 1.0;
    }

    let average_blocker_depth = blocker_depth_sum / f32(blocker_count);
    let separation = max(receiver_depth - average_blocker_depth, 0.0);
    let penumbra_ratio =
        separation / max(average_blocker_depth, scene.shadow_parameters.y);
    let max_filter_radius =
        max(shadow_texel.x, shadow_texel.y) * 12.0;
    let filter_radius = min(
        0.5 * source_radius * penumbra_ratio
            / max(receiver_depth * cone_tangent, 0.000001),
        max_filter_radius,
    );
    if filter_radius <= max(shadow_texel.x, shadow_texel.y) * 0.35 {
        return textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            shadow_position.xy,
            center_reference,
        );
    }

    var visibility = 0.0;
    for (var sample = 0u; sample < SHADOW_SAMPLE_COUNT; sample++) {
        let offset = SHADOW_DISK[sample] * filter_radius;
        let sample_uv = shadow_position.xy + offset;
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            visibility += 1.0;
            continue;
        }
        visibility += textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            sample_uv,
            shadow_reference_depth(
                shadow_position,
                offset,
                depth_gradient,
                n_dot_light,
            ),
        );
    }
    return visibility / f32(SHADOW_SAMPLE_COUNT);
}

fn deferred_shadow(uv: vec2<f32>, normal: vec3<f32>) -> f32 {
    let dimensions = vec2<i32>(textureDimensions(scene_depth));
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
    let depth = textureLoad(scene_depth, pixel, 0);
    let position = reconstruct_object_position(uv, depth);
    let light_clip_position = light_clip(position);
    let shadow_position = vec3<f32>(
        light_clip_position.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5),
        light_clip_position.z,
    );
    if any(shadow_position.xy < vec2<f32>(0.0))
        || any(shadow_position.xy > vec2<f32>(1.0))
        || shadow_position.z <= 0.0
        || shadow_position.z >= 1.0 {
        return 1.0;
    }
    return strict_tiger_shadow(shadow_position, uv * scene.postprocess1.xy);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let compatibility = textureSample(compatibility_hdr, surface_sampler, input.uv);
    let packed_normal = textureSample(normal_roughness, surface_sampler, input.uv);
    let normal = normalize(packed_normal.rgb * 2.0 - vec3<f32>(1.0));
    let properties = textureSample(material_properties, surface_sampler, input.uv);
    let emissive_dimensions = vec2<i32>(textureDimensions(surface_emissive));
    let emissive_pixel = clamp(
        vec2<i32>(input.uv * vec2<f32>(emissive_dimensions)),
        vec2<i32>(0),
        emissive_dimensions - vec2<i32>(1),
    );
    let emissive = textureLoad(surface_emissive, emissive_pixel, 0).rgb;
    let surface = textureSample(surface_albedo, surface_sampler, input.uv);
    let scene_dimensions = vec2<i32>(textureDimensions(scene_depth));
    let scene_pixel = clamp(
        vec2<i32>(input.uv * vec2<f32>(scene_dimensions)),
        vec2<i32>(0),
        scene_dimensions - vec2<i32>(1),
    );
    let scene_depth_value = textureLoad(scene_depth, scene_pixel, 0);
    let position = reconstruct_object_position(input.uv, scene_depth_value);
    if u32(scene.postprocess2.x + 0.5) != 0u {
        return compatibility;
    }
    var authored = false;
    if scene.fidelity.x > 0.5 {
        authored = textureLoad(authored_coverage,scene_pixel,0).a != 0.0;
    }
    let model = u32(scene.fidelity.y + 0.5);
    // Authored opaque surfaces have no old forward material submission. Both
    // GGX viewer modes consume the decoded material ABI here; unrelated
    // Compatibility surfaces retain their existing forward result.
    if model == 1u || (model == 0u && authored) {
        if surface.a <= 0.0 {
            return compatibility;
        }
        let albedo = surface.rgb;
        let roughness = clamp(packed_normal.a, 0.045, 1.0);
        let key_roughness = clamp(
            roughness + (scene.light_parameters.x - 5.0) * 0.025,
            0.02,
            1.0,
        );
        let metalness = clamp(properties.r, 0.0, 1.0);
        let ao = clamp(properties.g, 0.0, 1.0);
        let view_position = world_direction_to_view(position - scene.center.xyz);
        let light = normalize(scene.light_position.xyz - view_position);
        let spotlight = spotlight_factor_world(position);
        let view = vec3<f32>(0.0, 0.0, 1.0);
        let halfway = normalize(light + view);
        let n_dot_l = max(dot(normal, light), 0.0);
        let n_dot_v = max(dot(normal, view), 0.001);
        let n_dot_h = max(dot(normal, halfway), 0.0);
        let v_dot_h = max(dot(view, halfway), 0.0);
        let alpha = key_roughness * key_roughness;
        let alpha2 = alpha * alpha;
        let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
        let distribution = alpha2 / max(3.14159265 * denominator * denominator, 0.0001);
        let k = (key_roughness + 1.0) * (key_roughness + 1.0) * 0.125;
        let visibility_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 0.0001);
        let visibility_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
        let f0 = mix(vec3<f32>(0.04), albedo, metalness);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
        let specular = distribution * visibility_l * visibility_v * fresnel * 0.25;
        let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metalness) * albedo / 3.14159265;
        let shadow = mix(1.0, deferred_shadow(input.uv, normal), scene.light_direction.w);
        let direct = (diffuse + specular)
            * scene.light_color.rgb
            * n_dot_l
            * scene.postprocess0.w
            * spotlight
            * shadow;
        // Tiger's global-light ambient: squared half-range of world-up.
        let up = view_direction_to_world(normal).z;
        let sky = clamp(0.5 + 0.5 * up, 0.0, 1.0);
        let ground = clamp(0.5 - 0.5 * up, 0.0, 1.0);
        let ambient = sky * sky * scene.ambient_sky.rgb + ground * ground * scene.ambient_ground.rgb;
        let ibl_diffuse = albedo * (1.0 - metalness) * scene.postprocess4.z * ao * ambient;
        let ibl_specular = f0 * mix(0.08, 1.0, 1.0 - roughness)
            * scene.postprocess4.w * ao;
        let color = (direct + ibl_diffuse + ibl_specular + emissive) * scene.postprocess0.x;
        return vec4<f32>(color, surface.a);
    }
    if model == 2u {
        let view_position = world_direction_to_view(position - scene.center.xyz);
        let light = normalize(scene.light_position.xyz - view_position);
        let lambert = max(dot(normal, light), 0.0) * spotlight_factor_world(position);
        let ao = properties.g;
        return vec4<f32>(surface.rgb * (0.18 * ao + 0.82 * lambert) + emissive, select(compatibility.a,surface.a,authored));
    }
    if model == 3u {
        return vec4<f32>(normal * 0.5 + vec3<f32>(0.5), compatibility.a);
    }
    if model == 4u {
        return vec4<f32>(properties.r, properties.g, packed_normal.a, 1.0);
    }
    if model == 5u {
        return vec4<f32>(emissive, 1.0);
    }
    if model == 6u {
        let dimensions = vec2<i32>(textureDimensions(surface_flags));
        let pixel = clamp(vec2<i32>(input.uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
        let flags = textureLoad(surface_flags, pixel, 0).r;
        return vec4<f32>(
            select(0.0, 1.0, (flags & 1u) != 0u),
            select(0.0, 1.0, (flags & 2u) != 0u),
            select(0.0, 1.0, (flags & 8u) != 0u),
            1.0,
        );
    }
    if model == 7u {
        return surface;
    }
    // Compatibility is an explicit migration path. Surface buffers remain
    // inspectable while family lighting moves here incrementally.
    return compatibility;
}
"#;

const DISTORTION_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var distortion_payload: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let payload_texel = 1.0 / vec2<f32>(textureDimensions(distortion_payload));
    let payload = textureSample(distortion_payload, linear_sampler, input.uv);
    let alpha = clamp(payload.a, 0.0, 1.0);
    let alpha_left = textureSample(
        distortion_payload,
        linear_sampler,
        input.uv - vec2<f32>(payload_texel.x, 0.0),
    ).a;
    let alpha_right = textureSample(
        distortion_payload,
        linear_sampler,
        input.uv + vec2<f32>(payload_texel.x, 0.0),
    ).a;
    let alpha_up = textureSample(
        distortion_payload,
        linear_sampler,
        input.uv - vec2<f32>(0.0, payload_texel.y),
    ).a;
    let alpha_down = textureSample(
        distortion_payload,
        linear_sampler,
        input.uv + vec2<f32>(0.0, payload_texel.y),
    ).a;
    // Filtered payload coverage supplies a stable local gradient. It produces
    // the small boundary refraction seen in Tiger without
    // moving interior scene details by tens of pixels.
    let gradient = vec2<f32>(alpha_right - alpha_left, alpha_down - alpha_up);
    let offset = gradient * payload_texel * 0.55;
    let refracted = textureSample(
        scene_color,
        linear_sampler,
        clamp(input.uv + offset, vec2<f32>(0.0), vec2<f32>(1.0)),
    );
    return vec4<f32>(payload.rgb + refracted.rgb * (1.0 - alpha), refracted.a);
}
"#;

const BLOOM_SHADER: &str = r#"
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return output;
}

fn filtered_source(uv: vec2<f32>) -> vec3<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source_texture));
    var color = textureSample(source_texture, source_sampler, uv).rgb * 4.0;
    color += textureSample(source_texture, source_sampler, uv + vec2<f32>( texel.x, 0.0)).rgb * 2.0;
    color += textureSample(source_texture, source_sampler, uv + vec2<f32>(-texel.x, 0.0)).rgb * 2.0;
    color += textureSample(source_texture, source_sampler, uv + vec2<f32>(0.0,  texel.y)).rgb * 2.0;
    color += textureSample(source_texture, source_sampler, uv + vec2<f32>(0.0, -texel.y)).rgb * 2.0;
    color += textureSample(source_texture, source_sampler, uv + texel).rgb;
    color += textureSample(source_texture, source_sampler, uv - texel).rgb;
    color += textureSample(source_texture, source_sampler, uv + vec2<f32>(texel.x, -texel.y)).rgb;
    color += textureSample(source_texture, source_sampler, uv + vec2<f32>(-texel.x, texel.y)).rgb;
    return color / 16.0;
}

@fragment
fn fs_bright(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = filtered_source(input.uv);
    let brightness = max(max(color.r, color.g), color.b);
    let knee = smoothstep(0.58, 0.90, brightness);
    return vec4<f32>(max(color - vec3<f32>(0.58), vec3<f32>(0.0)) * knee, 1.0);
}

@fragment
fn fs_downsample(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(filtered_source(input.uv), 1.0);
}

fn gaussian_blur(uv: vec2<f32>, axis: vec2<f32>) -> vec3<f32> {
    let texel = axis / vec2<f32>(textureDimensions(source_texture));
    var color = textureSample(source_texture, source_sampler, uv).rgb * 0.227027;
    color += textureSample(source_texture, source_sampler, uv + texel).rgb * 0.1945946;
    color += textureSample(source_texture, source_sampler, uv - texel).rgb * 0.1945946;
    color += textureSample(source_texture, source_sampler, uv + texel * 2.0).rgb * 0.1216216;
    color += textureSample(source_texture, source_sampler, uv - texel * 2.0).rgb * 0.1216216;
    color += textureSample(source_texture, source_sampler, uv + texel * 3.0).rgb * 0.054054;
    color += textureSample(source_texture, source_sampler, uv - texel * 3.0).rgb * 0.054054;
    color += textureSample(source_texture, source_sampler, uv + texel * 4.0).rgb * 0.016216;
    color += textureSample(source_texture, source_sampler, uv - texel * 4.0).rgb * 0.016216;
    return color;
}

@fragment
fn fs_blur_horizontal(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(gaussian_blur(input.uv, vec2<f32>(1.0, 0.0)), 1.0);
}

@fragment
fn fs_blur_vertical(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(gaussian_blur(input.uv, vec2<f32>(0.0, 1.0)), 1.0);
}
"#;

const SHADOW_SHADER: &str = r#"
struct SceneUniform {
    center: vec4<f32>, params0: vec4<f32>, params1: vec4<f32>, uv_transform: vec4<f32>,
    light_direction: vec4<f32>, light_parameters: vec4<f32>, light_position: vec4<f32>, postprocess0: vec4<f32>, postprocess1: vec4<f32>, postprocess2: vec4<f32>, postprocess3: vec4<f32>, postprocess4: vec4<f32>, postprocess5: vec4<f32>, fidelity: vec4<f32>, shadow_parameters: vec4<f32>,
}
@group(0) @binding(0) var<uniform> scene: SceneUniform;

struct MaterialUniform {
    tint: vec4<f32>, params: vec4<f32>, blend: vec4<f32>,
    solid_surface: vec4<f32>,
    palette_base: vec4<f32>, palette_delta: vec4<f32>, palette_tertiary: vec4<f32>,
    roughness_remap: vec4<f32>,
    metal_remap: vec4<f32>,
    wear_params: vec4<f32>,
    wear_scratches_projection: vec4<f32>,
    wear_grime_projection: vec4<f32>,
    wear_damage_projection: vec4<f32>,
    wear_scratches_remap_base: vec4<f32>,
    wear_scratches_remap_scale: vec4<f32>,
    wear_surface_params: vec4<f32>,
    gear_palette_default: vec4<f32>,
    gear_palette_colors: array<vec4<f32>, 6>,
    gear_worn_dye_colors: array<vec4<f32>, 6>,
    gear_dye_detail_colors: array<vec4<f32>, 6>,
    gear_palette_roughness: array<vec4<f32>, 6>,
    gear_palette_metal: array<vec4<f32>, 6>,
    decal_params: vec4<f32>,
    decal_detail_transform: vec4<f32>,
    decal_detail_base: vec4<f32>,
    decal_detail_scale: vec4<f32>,
    decal_mask_params: vec4<f32>,
    decal_mask_remap: vec4<f32>,
    decal_selector_colors: array<vec4<f32>, 5>,
    sampler_params: vec4<f32>,
    pattern_projection: vec4<f32>,
    pattern_params: vec4<f32>,
    pattern_stripe: vec4<f32>,
    pattern_contour: vec4<f32>,
    pattern_contour_remap: vec4<f32>,
    pattern_colors: array<vec4<f32>, 2>,
    transmission_colors: array<vec4<f32>, 2>,
    transmission_surfaces: array<vec4<f32>, 2>,
    transmission_params: vec4<f32>,
    coating_colors: array<vec4<f32>, 2>,
    coating_projection: vec4<f32>,
    coating_params0: vec4<f32>,
    coating_params1: vec4<f32>,
    coating_environment_params: vec4<f32>,
    coating_environment_extra: vec4<f32>,
    coating_specular_colors: array<vec4<f32>, 2>,
    coating_specular_params: array<vec4<f32>, 2>,
    character_detail_transform: vec4<f32>,
    character_detail_base: vec4<f32>,
    character_detail_scale: vec4<f32>,
    character_params: vec4<f32>,
    character_extra: array<vec4<f32>, 2>,
    character_palette: array<vec4<f32>, 2>,
    character_procedural: array<vec4<f32>, 11>,
    runner_layered_params: vec4<f32>,
    runner_layered_constants: array<vec4<f32>, 24>,
    runner_color_constants: array<vec4<f32>, 7>,
    alpha_mask_params: vec4<f32>,
}
@group(1) @binding(0) var color_texture: texture_2d<f32>;
@group(1) @binding(3) var control_texture: texture_2d<f32>;
@group(1) @binding(4) var material_sampler: sampler;
@group(1) @binding(5) var<uniform> material: MaterialUniform;

struct AuthoredShadowUniform {
    geometry_scale: vec4<f32>,
    geometry_offset: vec4<f32>,
    attachment_rotation: vec4<f32>,
    attachment_translation_scale: vec4<f32>,
    uv_transform: vec4<f32>,
}
@group(2) @binding(0) var<uniform> authored: AuthoredShadowUniform;

fn view_direction_to_world(value: vec3<f32>) -> vec3<f32> {
    let cy = cos(scene.params0.y);
    let sy = sin(scene.params0.y);
    let cp = cos(scene.params0.z);
    let sp = sin(scene.params0.z);
    let yawed = vec3<f32>(
        value.x,
        value.y * cp + value.z * sp,
        -value.y * sp + value.z * cp,
    );
    let z_up = vec3<f32>(
        yawed.x * cy - yawed.z * sy,
        yawed.y,
        yawed.x * sy + yawed.z * cy,
    );
    return vec3<f32>(z_up.x, z_up.z, z_up.y);
}

fn light_clip(position: vec3<f32>) -> vec4<f32> {
    // Tiger/Alkahest-style directional shadow space. Marathon's exact
    // authored world-to-shadow matrix is still a follow-up target, but this
    // affine projection preserves the engine contract and avoids the old
    // finite-spotlight perspective distortion that produced shadow spikes.
    let light = normalize(view_direction_to_world(scene.light_direction.xyz));
    let up = select(
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(light.z) > 0.95,
    );
    let right = normalize(cross(up, light));
    let vertical = cross(light, right);
    let relative = (position - scene.center.xyz) / max(scene.center.w, 0.0001);
    return vec4<f32>(
        dot(relative, right),
        dot(relative, vertical),
        0.5 - dot(relative, light) * 0.5,
        1.0,
    );
}

struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn rotate_authored(value: vec3<f32>, rotation: vec4<f32>) -> vec3<f32> {
    let twice_cross = 2.0 * cross(rotation.xyz, value);
    return value + rotation.w * twice_cross + cross(rotation.xyz, twice_cross);
}

fn authored_shadow_vertex(position: vec3<f32>, uv: vec2<f32>) -> ShadowVertexOutput {
    let geometry_position = position * authored.geometry_scale.xyz + authored.geometry_offset.xyz;
    let scaled = geometry_position * authored.attachment_translation_scale.w;
    let placed = rotate_authored(scaled, authored.attachment_rotation)
        + authored.attachment_translation_scale.xyz;
    var output: ShadowVertexOutput;
    output.position = light_clip(placed);
    output.uv = uv * authored.uv_transform.xy + authored.uv_transform.zw;
    return output;
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> ShadowVertexOutput {
    var output: ShadowVertexOutput;
    output.position = light_clip(position);
    output.uv = uv * scene.uv_transform.xy + scene.uv_transform.zw;
    return output;
}

@vertex
fn vs_authored_vec3(
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> ShadowVertexOutput {
    return authored_shadow_vertex(position, uv);
}

@vertex
fn vs_authored_vec4(
    @location(0) position: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> ShadowVertexOutput {
    return authored_shadow_vertex(position.xyz, uv);
}

@fragment
fn fs_main(input: ShadowVertexOutput) {
    let base_color = textureSampleBias(
        color_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    );
    if material.alpha_mask_params.x > 0.5 {
        let coverage_sample = textureSampleBias(
            control_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).r;
        let coverage = coverage_sample * material.alpha_mask_params.w
            + material.alpha_mask_params.z;
        if coverage < material.alpha_mask_params.y {
            discard;
        }
        return;
    }
    if material.decal_params.x > 0.5 {
        let raw_mask = textureSampleBias(
            control_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).r;
        var decal_mask = 0.0;
        if material.decal_params.x < 1.5 {
            decal_mask = select(0.0, 1.0, raw_mask > material.decal_mask_params.y);
        } else if material.decal_mask_params.x < 1.5 {
            if input.uv.y > 1.0 {
                decal_mask = select(
                    clamp(material.decal_mask_remap.z + raw_mask * material.decal_mask_remap.w, 0.0, 1.0),
                    clamp(material.decal_mask_remap.x + raw_mask * material.decal_mask_remap.y, 0.0, 1.0),
                    input.uv.x > 0.0,
                );
            } else {
                decal_mask = select(1.0, raw_mask, input.uv.x > 0.0);
            }
        } else {
            decal_mask = select(0.0, 1.0, raw_mask > 0.0);
        }
        if decal_mask * material.decal_params.w <= 0.001 {
            discard;
        }
        return;
    }
    let mask_material = material.params.w < -0.5;
    let alpha = select(base_color.a, base_color.r, mask_material);
    if (!mask_material && base_color.a < material.params.w)
        || (mask_material && alpha < 0.02) {
        discard;
    }
}
"#;

const AUTHORED_DEPTH_SHADER: &str = r#"
struct SceneUniform {
    center: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    uv_transform: vec4<f32>,
    light_direction: vec4<f32>,
    light_parameters: vec4<f32>,
    light_position: vec4<f32>,
    postprocess0: vec4<f32>,
    postprocess1: vec4<f32>,
    postprocess2: vec4<f32>,
    postprocess3: vec4<f32>,
    postprocess4: vec4<f32>,
    postprocess5: vec4<f32>,
    fidelity: vec4<f32>,
    shadow_parameters: vec4<f32>,
}

struct AuthoredGeometryUniform {
    geometry_scale: vec4<f32>,
    geometry_offset: vec4<f32>,
    attachment_rotation: vec4<f32>,
    attachment_translation_scale: vec4<f32>,
    uv_transform: vec4<f32>,
}

@group(0) @binding(0) var<uniform> scene: SceneUniform;
@group(1) @binding(0) var<uniform> authored_geometry: AuthoredGeometryUniform;

fn rotate_view(value: vec3<f32>) -> vec3<f32> {
    let z_up = vec3<f32>(value.x, value.z, value.y);
    let cy = cos(scene.params0.y);
    let sy = sin(scene.params0.y);
    let cp = cos(scene.params0.z);
    let sp = sin(scene.params0.z);
    let yawed = vec3<f32>(
        z_up.x * cy + z_up.z * sy,
        z_up.y,
        -z_up.x * sy + z_up.z * cy,
    );
    return vec3<f32>(
        yawed.x,
        yawed.y * cp - yawed.z * sp,
        yawed.y * sp + yawed.z * cp,
    );
}

fn rotate_authored_geometry(value: vec3<f32>, rotation: vec4<f32>) -> vec3<f32> {
    let twice_cross = 2.0 * cross(rotation.xyz, value);
    return value + rotation.w * twice_cross + cross(rotation.xyz, twice_cross);
}

fn authored_world_position(position: vec3<f32>) -> vec3<f32> {
    let geometry_position =
        position * authored_geometry.geometry_scale.xyz + authored_geometry.geometry_offset.xyz;
    let scaled = geometry_position * authored_geometry.attachment_translation_scale.w;
    return rotate_authored_geometry(scaled, authored_geometry.attachment_rotation)
        + authored_geometry.attachment_translation_scale.xyz;
}

fn model_clip_position(position: vec3<f32>) -> vec4<f32> {
    let view_position = rotate_view(position - scene.center.xyz);
    let scale = 0.84 * scene.params0.w / max(scene.params0.x, 0.0001);
    let depth = clamp(
        0.5 - view_position.z * 0.21 / max(scene.params0.x, 0.0001),
        0.0,
        1.0,
    );
    return vec4<f32>(
        -view_position.x * scale * scene.params1.x + scene.params1.y,
        view_position.y * scale + scene.params1.z,
        depth,
        1.0,
    );
}

@vertex
fn vs_authored_depth_vec3(
    @location(0) position: vec3<f32>,
) -> @builtin(position) vec4<f32> {
    return model_clip_position(authored_world_position(position));
}

@vertex
fn vs_authored_depth_vec4(
    @location(0) position: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    return model_clip_position(authored_world_position(position.xyz));
}
"#;

const MODEL_SHADER: &str = r#"
struct SceneUniform {
    center: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    uv_transform: vec4<f32>,
    light_direction: vec4<f32>,
    light_parameters: vec4<f32>,
    light_position: vec4<f32>,
    postprocess0: vec4<f32>,
    postprocess1: vec4<f32>,
    postprocess2: vec4<f32>,
    postprocess3: vec4<f32>,
    postprocess4: vec4<f32>,
    postprocess5: vec4<f32>,
    fidelity: vec4<f32>,
    shadow_parameters: vec4<f32>,
}

struct MaterialUniform {
    tint: vec4<f32>,
    params: vec4<f32>,
    blend: vec4<f32>,
    solid_surface: vec4<f32>,
    palette_base: vec4<f32>,
    palette_delta: vec4<f32>,
    palette_tertiary: vec4<f32>,
    roughness_remap: vec4<f32>,
    metal_remap: vec4<f32>,
    wear_params: vec4<f32>,
    wear_scratches_projection: vec4<f32>,
    wear_grime_projection: vec4<f32>,
    wear_damage_projection: vec4<f32>,
    wear_scratches_remap_base: vec4<f32>,
    wear_scratches_remap_scale: vec4<f32>,
    wear_surface_params: vec4<f32>,
    gear_palette_default: vec4<f32>,
    gear_palette_colors: array<vec4<f32>, 6>,
    gear_worn_dye_colors: array<vec4<f32>, 6>,
    gear_dye_detail_colors: array<vec4<f32>, 6>,
    gear_palette_roughness: array<vec4<f32>, 6>,
    gear_palette_metal: array<vec4<f32>, 6>,
    decal_params: vec4<f32>,
    decal_detail_transform: vec4<f32>,
    decal_detail_base: vec4<f32>,
    decal_detail_scale: vec4<f32>,
    decal_mask_params: vec4<f32>,
    decal_mask_remap: vec4<f32>,
    decal_selector_colors: array<vec4<f32>, 5>,
    sampler_params: vec4<f32>,
    pattern_projection: vec4<f32>,
    pattern_params: vec4<f32>,
    pattern_stripe: vec4<f32>,
    pattern_contour: vec4<f32>,
    pattern_contour_remap: vec4<f32>,
    pattern_colors: array<vec4<f32>, 2>,
    transmission_colors: array<vec4<f32>, 2>,
    transmission_surfaces: array<vec4<f32>, 2>,
    transmission_params: vec4<f32>,
    coating_colors: array<vec4<f32>, 2>,
    coating_projection: vec4<f32>,
    coating_params0: vec4<f32>,
    coating_params1: vec4<f32>,
    coating_environment_params: vec4<f32>,
    coating_environment_extra: vec4<f32>,
    coating_specular_colors: array<vec4<f32>, 2>,
    coating_specular_params: array<vec4<f32>, 2>,
    character_detail_transform: vec4<f32>,
    character_detail_base: vec4<f32>,
    character_detail_scale: vec4<f32>,
    character_params: vec4<f32>,
    character_extra: array<vec4<f32>, 2>,
    character_palette: array<vec4<f32>, 2>,
    character_procedural: array<vec4<f32>, 11>,
    runner_layered_params: vec4<f32>,
    runner_layered_constants: array<vec4<f32>, 24>,
    runner_color_constants: array<vec4<f32>, 7>,
    alpha_mask_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> scene: SceneUniform;
@group(0) @binding(1) var sun_shadow: texture_depth_2d;
@group(0) @binding(2) var sun_shadow_sampler: sampler_comparison;
@group(0) @binding(3) var environment_cubemap: texture_cube<f32>;
@group(0) @binding(4) var environment_sampler: sampler;
@group(1) @binding(0) var color_texture: texture_2d<f32>;
@group(1) @binding(1) var normal_texture: texture_2d<f32>;
@group(1) @binding(2) var emissive_texture: texture_2d<f32>;
@group(1) @binding(3) var control_texture: texture_2d<f32>;
@group(1) @binding(4) var material_sampler: sampler;
@group(1) @binding(5) var<uniform> material: MaterialUniform;
@group(1) @binding(6) var wear_scratches_texture: texture_2d<f32>;
@group(1) @binding(7) var wear_grime_texture: texture_2d<f32>;
@group(1) @binding(8) var wear_damage_texture: texture_2d<f32>;
// Weapon gear patterns and runner object-space procedural fields are mutually
// exclusive draw ABIs and therefore share Tiger's sixteenth texture binding.
@group(1) @binding(9) var pattern_or_runner_procedural_texture: texture_2d<f32>;
@group(1) @binding(10) var character_surface_texture: texture_2d<f32>;
@group(1) @binding(11) var character_detail_color_texture: texture_2d<f32>;
// Character procedural fields and runner response/AO are mutually exclusive
// material ABIs. Tiger's runner response stores normal response in R and AO in G.
@group(1) @binding(12) var procedural_or_response_texture: texture_2d<f32>;
@group(1) @binding(13) var runner_surface_texture: texture_2d<f32>;
@group(1) @binding(14) var runner_detail_normal_a_texture: texture_2d<f32>;
@group(1) @binding(15) var runner_detail_normal_b_texture: texture_2d<f32>;
@group(1) @binding(16) var runner_detail_normal_c_texture: texture_2d<f32>;
@group(1) @binding(17) var runner_detail_normal_d_texture: texture_2d<f32>;
@group(1) @binding(18) var coating_environment_texture: texture_cube<f32>;
@group(1) @binding(19) var coating_environment_sampler: sampler;
// Tiger stage-8 coating PS t0: deferred RT2 produced by the opaque surface
// pass. This is screen-space material data, not another authored 2D texture.
@group(2) @binding(0) var coating_scene_depth: texture_depth_2d;
// Stage-2 investment decal PS t2: packed screen-space surface normal copied
// after the opaque pass. It must not alias the normal render attachment.
@group(2) @binding(1) var investment_scene_normal: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) ambient_occlusion: f32,
    @location(5) procedural_position: vec3<f32>,
    @location(6) procedural_normal: vec3<f32>,
}

fn view_direction_to_world(value: vec3<f32>) -> vec3<f32> {
    let cy = cos(scene.params0.y);
    let sy = sin(scene.params0.y);
    let cp = cos(scene.params0.z);
    let sp = sin(scene.params0.z);
    let yawed = vec3<f32>(
        value.x,
        value.y * cp + value.z * sp,
        -value.y * sp + value.z * cp,
    );
    let z_up = vec3<f32>(
        yawed.x * cy - yawed.z * sy,
        yawed.y,
        yawed.x * sy + yawed.z * cy,
    );
    return vec3<f32>(z_up.x, z_up.z, z_up.y);
}

fn light_clip(position: vec3<f32>) -> vec4<f32> {
    // Tiger/Alkahest-style directional shadow space. Marathon's exact
    // authored world-to-shadow matrix is still a follow-up target, but this
    // affine projection preserves the engine contract and avoids the old
    // finite-spotlight perspective distortion that produced shadow spikes.
    let light = normalize(view_direction_to_world(scene.light_direction.xyz));
    let up = select(
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(light.z) > 0.95,
    );
    let right = normalize(cross(up, light));
    let vertical = cross(light, right);
    let relative = (position - scene.center.xyz) / max(scene.center.w, 0.0001);
    return vec4<f32>(
        dot(relative, right),
        dot(relative, vertical),
        0.5 - dot(relative, light) * 0.5,
        1.0,
    );
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_position: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_tangent: vec4<f32>,
    @location(4) ambient_occlusion: f32,
    @location(6) world_relative: vec3<f32>,
    @location(7) world_normal: vec3<f32>,
    @location(8) procedural_position: vec3<f32>,
    @location(9) procedural_normal: vec3<f32>,
}

fn rotate_view(value: vec3<f32>) -> vec3<f32> {
    let z_up = vec3<f32>(value.x, value.z, value.y);
    let cy = cos(scene.params0.y);
    let sy = sin(scene.params0.y);
    let cp = cos(scene.params0.z);
    let sp = sin(scene.params0.z);
    let yawed = vec3<f32>(
        z_up.x * cy + z_up.z * sy,
        z_up.y,
        -z_up.x * sy + z_up.z * cy,
    );
    return vec3<f32>(
        yawed.x,
        yawed.y * cp - yawed.z * sp,
        yawed.y * sp + yawed.z * cp,
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let view_position = rotate_view(input.position - scene.center.xyz);
    let view_normal = normalize(rotate_view(input.normal));
    let scale = 0.84 * scene.params0.w / max(scene.params0.x, 0.0001);
    // Orthographic zoom changes only screen-space magnification. Depth must
    // retain the fitted-model range or close zoom collapses surfaces onto the
    // clip planes and produces progressive disappearance/z-fighting.
    let depth = clamp(
        0.5 - view_position.z * 0.21 / max(scene.params0.x, 0.0001),
        0.0,
        1.0,
    );

    var output: VertexOutput;
    output.clip_position = vec4<f32>(
        -view_position.x * scale * scene.params1.x + scene.params1.y,
        view_position.y * scale + scene.params1.z,
        depth,
        1.0,
    );
    output.view_position = view_position;
    output.view_normal = view_normal;
    output.uv = input.uv * scene.uv_transform.xy + scene.uv_transform.zw;
    output.view_tangent = vec4<f32>(rotate_view(input.tangent.xyz), input.tangent.w);
    output.ambient_occlusion = input.ambient_occlusion;
    output.world_relative = input.position - scene.center.xyz;
    output.world_normal = input.normal;
    output.procedural_position = input.procedural_position;
    output.procedural_normal = input.procedural_normal;
    return output;
}

fn triplanar_surface(
    texture: texture_2d<f32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    projection: vec4<f32>,
) -> vec3<f32> {
    // Match compiled common-surface PS: TEXCOORD5 planes are YZ, XZ, XY;
    // TEXCOORD6 supplies normalized blend weights.
    var weights = pow(abs(normalize(normal)), vec3<f32>(4.0));
    weights /= max(weights.x + weights.y + weights.z, 0.0001);
    let x = textureSampleBias(
        texture,
        material_sampler,
        position.yz * projection.xy + projection.zw,
        material.sampler_params.x,
    ).rgb;
    let y = textureSampleBias(
        texture,
        material_sampler,
        position.xz * projection.xy + projection.zw,
        material.sampler_params.x,
    ).rgb;
    let z = textureSampleBias(
        texture,
        material_sampler,
        position.xy * projection.xy + projection.zw,
        material.sampler_params.x,
    ).rgb;
    return x * weights.x + y * weights.y + z * weights.z;
}

fn runner_triplanar_scalar(
    texture: texture_2d<f32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    projection: vec4<f32>,
    exponent: f32,
) -> f32 {
    var weights = pow(abs(normalize(normal)), vec3<f32>(exponent));
    weights /= max(weights.x + weights.y + weights.z, 0.0001);
    let x = textureSampleBias(
        texture, material_sampler,
        position.yz * projection.xy + projection.zw,
        material.sampler_params.x,
    ).r;
    let y = textureSampleBias(
        texture, material_sampler,
        position.xz * projection.xy + projection.zw,
        material.sampler_params.x,
    ).r;
    let z = textureSampleBias(
        texture, material_sampler,
        position.xy * projection.xy + projection.zw,
        material.sampler_params.x,
    ).r;
    return dot(vec3<f32>(x, y, z), weights);
}

fn runner_e4db_condition(input: VertexOutput) -> f32 {
    // 80A9E4DB t9/t11 share three authored UV transforms. t10 supplies the
    // low-amplitude RG distortion used before the six powered t11 samples.
    let warp_uv = input.procedural_position.xy * 5.0;
    let warp = (textureSampleBias(
        wear_grime_texture, material_sampler, warp_uv,
        material.sampler_params.x).rg - vec2<f32>(0.5))
        * material.runner_layered_constants[22].z;
    let uv0 = input.uv * material.runner_layered_constants[18].xy
        + material.runner_layered_constants[18].zw + warp;
    let uv1 = input.uv * material.runner_layered_constants[19].xy
        + material.runner_layered_constants[19].zw + warp;
    let uv2 = input.uv * material.runner_layered_constants[20].xy
        + material.runner_layered_constants[20].zw + warp;
    let scratch = clamp(2.0 * (
        textureSampleBias(wear_scratches_texture, material_sampler, uv0,
            material.sampler_params.x).r
        + textureSampleBias(wear_scratches_texture, material_sampler, uv1,
            material.sampler_params.x).r
        + textureSampleBias(wear_scratches_texture, material_sampler, uv2,
            material.sampler_params.x).r), 0.0, 1.0);
    let breakup0 = max(textureSampleBias(
        wear_damage_texture, material_sampler, uv0,
        material.sampler_params.x).r, 0.000001);
    let breakup1 = max(textureSampleBias(
        wear_damage_texture, material_sampler, uv1,
        material.sampler_params.x).r, 0.000001);
    let breakup2 = max(textureSampleBias(
        wear_damage_texture, material_sampler, uv2,
        material.sampler_params.x).r, 0.000001);
    let exponents = material.runner_layered_constants[21]
        + vec4<f32>(material.runner_layered_constants[22].xy, 0.0, 0.0);
    let breakup = clamp(
        pow(breakup0, exponents.x + exponents.y)
        * pow(breakup1, exponents.z + exponents.w)
        * pow(breakup2, material.runner_layered_constants[22].x
            + material.runner_layered_constants[22].y),
        0.0, 1.0);
    let authored = clamp(max(scratch, breakup), 0.0, 1.0);
    let first = material.runner_layered_constants[23].x
        + material.runner_layered_constants[23].y * authored;
    return clamp(material.runner_layered_constants[23].z
        + material.runner_layered_constants[23].w * first, 0.0, 1.0);
}

fn gear_pattern_plane(axis_position: f32, field_uv: vec2<f32>, warp: f32) -> f32 {
    let field = textureSampleBias(
        pattern_or_runner_procedural_texture,
        material_sampler,
        field_uv,
        material.sampler_params.x,
    ).r;
    let displaced = axis_position + warp * (field - material.pattern_params.z);
    let periodic = fract(
        displaced * material.pattern_stripe.x + material.pattern_stripe.y
    );
    let stripe = clamp(
        material.pattern_stripe.z
            + abs(periodic + material.pattern_contour.x) * material.pattern_stripe.w,
        0.0,
        1.0,
    );
    return stripe * material.pattern_contour.y;
}

fn apply_gear_pattern(albedo: vec3<f32>, input: VertexOutput) -> vec3<f32> {
    if material.pattern_params.x < 0.5 {
        return albedo;
    }

    let control = textureSampleBias(
        control_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    ).rgb;
    let low_r = control.r <= 0.5;
    let low_g = control.g <= 0.5;
    let low_b = control.b <= 0.5;
    let selector_two = low_r && low_g && !low_b;
    let selector_four = low_r && !low_g && low_b;
    if !selector_two && !selector_four {
        return albedo;
    }

    let warp = select(
        material.pattern_contour.w,
        material.pattern_params.w,
        selector_two,
    );
    // scope_skinning[5].w is an authored procedural-coordinate multiplier,
    // independent from the XYZ dequantization used for raster position.
    let position = input.procedural_position * material.sampler_params.y;
    let projection = material.pattern_projection;
    let line_x = gear_pattern_plane(
        position.y,
        position.yz * projection.xy + projection.zw,
        warp,
    );
    let line_y = gear_pattern_plane(
        position.x,
        position.xz * projection.xy + projection.zw,
        warp,
    );
    let line_z = gear_pattern_plane(
        position.x,
        position.xy * projection.xy + projection.zw,
        warp,
    );
    var weights = pow(
        abs(normalize(input.procedural_normal)),
        vec3<f32>(material.pattern_params.y),
    );
    weights /= max(weights.x + weights.y + weights.z, 0.0001);
    let lines = dot(vec3<f32>(line_x, line_y, line_z), weights);
    let contour_position = fract(lines - 1.0 + material.pattern_contour.z);
    let contour = clamp(
        material.pattern_contour_remap.x
            + abs(contour_position + material.pattern_contour_remap.z)
                * material.pattern_contour_remap.y,
        0.0,
        1.0,
    );
    let pattern_color = select(
        material.pattern_colors[1].rgb,
        material.pattern_colors[0].rgb,
        selector_two,
    );
    return mix(albedo, pattern_color, contour);
}









fn apply_weapon_mod_condition(albedo: vec3<f32>, input: VertexOutput) -> vec4<f32> {
    // wear_params = TFX outputs [24, 42, 49, resources-present]. Game data
    // evaluates these to Enhanced [1,1,1], Deluxe [0,0,1], Superior [0,0,0].
    // No rarity-wide brightness scalar exists.
    if material.wear_params.w < 0.5 {
        return vec4<f32>(albedo, 0.0);
    }

    // Common-surface VS multiplies raw packed POSITION by
    // scope_skinning[5].w before forwarding TEXCOORD5. This draw-local value
    // differs per geometry; omitting it turns authored broad wear into dense
    // tiling (for example 0.05579831 on Precision Barrel).
    let position = input.procedural_position * material.sampler_params.y;
    let scratches = triplanar_surface(
        wear_scratches_texture,
        position,
        input.procedural_normal,
        material.wear_scratches_projection,
    ).r;
    let grime = triplanar_surface(
        wear_grime_texture,
        position,
        input.procedural_normal,
        material.wear_grime_projection,
    );
    let damage = triplanar_surface(
        wear_damage_texture,
        position,
        input.procedural_normal,
        material.wear_damage_projection,
    ).r;

    // Literal thresholds/remaps from common weapon-part shader cbuffer
    // 45/47/48. Output 42 is middle response: 0 narrows Deluxe wear to most
    // damaged texels; 1 expands Enhanced wear through middle band.
    let low_response = clamp((damage - 0.4) * 5.0, 0.0, 1.0);
    let high_response = clamp((damage - 0.6) * 2.5, 0.0, 1.0);
    let damage_mask = select(
        mix(material.wear_params.y, 1.0, high_response),
        mix(0.0, material.wear_params.y, low_response),
        damage < 0.6,
    );

    // DXIL branch order:
    //  1. t7 selects where t6's authored worn surface replaces clean paint;
    //  2. output49 gates that branch (Enhanced+Deluxe, not Superior);
    //  3. output24 selects t5-remapped paint for Enhanced;
    //  4. cbuffer50 blends both branches. t6 is a surface color, never a
    //     multiplier over albedo. This avoids pseudo-darkening Deluxe.
    let worn_surface = mix(albedo, grime, damage_mask);
    let condition_surface = mix(albedo, worn_surface, material.wear_params.z);
    let scratch_remap = clamp(
        (material.wear_scratches_remap_base.xyz
            + material.wear_scratches_remap_scale.xyz * scratches)
            * material.wear_surface_params.y,
        vec3<f32>(0.0),
        vec3<f32>(4.0),
    );
    let scratch_surface = albedo * scratch_remap;
    let base_branch = mix(albedo, scratch_surface, material.wear_params.x);
    let conditioned = mix(
        base_branch,
        condition_surface,
        material.wear_surface_params.x,
    );
    let wear_mask = clamp(
        max(damage_mask * material.wear_params.z, scratches * material.wear_params.x)
            * material.wear_surface_params.x,
        0.0,
        1.0,
    );
    return vec4<f32>(clamp(conditioned, vec3<f32>(0.0), vec3<f32>(4.0)), wear_mask);
}

fn weapon_surface_condition_mask(input: VertexOutput) -> f32 {
    if material.wear_params.w < 1.5 || material.wear_params.w > 2.5 {
        return 0.0;
    }
    let uv = input.procedural_position.xy * material.wear_damage_projection.xy
        + material.wear_damage_projection.zw;
    let packed = textureSampleBias(
        wear_damage_texture,
        material_sampler,
        uv,
        material.sampler_params.x,
    );
    let phase = fract(packed.b + material.wear_scratches_projection.x);
    let triangle_phase = material.wear_grime_projection.x
        * (phase - 1.0 + packed.a)
        / max(material.wear_grime_projection.y, 0.0001);
    let triangle = 1.0 - abs(fract(triangle_phase) * 2.0 - 1.0);
    let breakup = clamp(
        (triangle * material.wear_grime_projection.z
            + material.wear_grime_projection.w)
            * (1.0 - phase),
        0.0,
        1.0,
    );
    let normal_z = normalize(input.procedural_normal).z;
    let facing = clamp(
        material.wear_scratches_projection.y
            + material.wear_scratches_projection.z * normal_z,
        0.0,
        1.0,
    );
    // Inventory preview drives common condition at one quarter. Gameplay's
    // full response crushes neutral weapon paint almost black.
    return 0.25 * clamp(1.0 - breakup * facing * 0.35, 0.0, 1.0);
}

fn apply_weapon_surface_condition(albedo: vec3<f32>, input: VertexOutput) -> vec3<f32> {
    let mask = weapon_surface_condition_mask(input);
    if material.wear_params.w < 1.5 || material.wear_params.w > 2.5 {
        return albedo;
    }
    let response = material.wear_surface_params;
    let conditioned_surface = pow(
        max(albedo * response.y, vec3<f32>(0.00001)),
        vec3<f32>(response.x),
    );
    let blend = pow(max(mask, 0.00001), response.z) * response.y;
    return mix(albedo, conditioned_surface, clamp(blend, 0.0, 1.0));
}

fn runner_detail_normal(
    texture_value: texture_2d<f32>,
    input_uv: vec2<f32>,
    row_a: vec4<f32>,
    row_b: vec4<f32>,
    remap: vec4<f32>,
) -> vec3<f32> {
    let detail_uv = vec2<f32>(
        dot(row_a.xy, input_uv) + row_a.z,
        dot(row_b.xy, input_uv) + row_b.z,
    );
    let detail_sample = textureSampleBias(
        texture_value,
        material_sampler,
        detail_uv,
        material.sampler_params.x,
    );
    let detail_xy = detail_sample.xy * remap.x + vec2<f32>(remap.y);
    return normalize(vec3<f32>(
        detail_xy,
        sqrt(max(1.0 - dot(detail_xy, detail_xy), 0.0)),
    ));
}

fn blend_runner_normal(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(base.xy + detail.xy, base.z * detail.z));
}

fn mapped_normal(input: VertexOutput) -> vec3<f32> {
    let base_normal = normalize(input.view_normal);
    if material.params.x < 0.5 {
        return base_normal;
    }

    var tangent: vec3<f32>;
    var bitangent: vec3<f32>;
    if length(input.view_tangent.xyz) > 0.5 {
        tangent = normalize(
            input.view_tangent.xyz - base_normal * dot(base_normal, input.view_tangent.xyz)
        );
        bitangent = normalize(cross(base_normal, tangent)) * input.view_tangent.w;
    } else {
        let dpdx_value = dpdx(input.view_position);
        let dpdy_value = dpdy(input.view_position);
        let duvdx = dpdx(input.uv);
        let duvdy = dpdy(input.uv);
        let determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
        if abs(determinant) < 0.000001 {
            return base_normal;
        }
        tangent = normalize((dpdx_value * duvdy.y - dpdy_value * duvdx.y) / determinant);
        bitangent = normalize((-dpdx_value * duvdy.x + dpdy_value * duvdx.x) / determinant);
    }
    // Marathon's material shaders remap normal-map RG, then reconstruct positive Z.
    // Inventory reference is driven by silhouette and authored hard edges, not
    // full-amplitude compressed micro-normal noise.
    let normal_strength = select(0.35, 1.0, material.runner_layered_params.x > 0.5);
    let sampled_xy = (textureSampleBias(
        normal_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    ).xy * 2.0 - 1.0) * normal_strength;
    let sampled_z = sqrt(max(1.0 - dot(sampled_xy, sampled_xy), 0.0));
    var sampled = normalize(vec3<f32>(sampled_xy, sampled_z));
    if material.runner_layered_params.x > 0.5 {
        var selector = textureSampleBias(
            control_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        );
        if material.runner_layered_params.y > 2.5 {
            selector = textureSampleBias(
                runner_surface_texture,
                material_sampler,
                input.uv,
                material.sampler_params.x,
            );
        }
        if material.runner_layered_params.y > 2.5
            && material.runner_layered_params.y < 3.5 {
            // Full10 ABI (80A9C244): t4 is unconditional; t5/t6/t7 are
            // selected by packed t2 G/B/R bands. t6 has two authored UV
            // transforms selected by literal c10.x.
            let detail_a = runner_detail_normal(
                runner_detail_normal_a_texture,
                input.uv,
                material.runner_layered_constants[1],
                material.runner_layered_constants[2],
                material.runner_layered_constants[3],
            );
            sampled = blend_runner_normal(sampled, detail_a);

            if selector.g > material.runner_layered_constants[7].x
                && selector.g < material.runner_layered_constants[8].x {
                let detail_b = runner_detail_normal(
                    runner_detail_normal_b_texture,
                    input.uv,
                    material.runner_layered_constants[4],
                    material.runner_layered_constants[5],
                    material.runner_layered_constants[6],
                );
                sampled = blend_runner_normal(sampled, detail_b);
            }

            if selector.b > material.runner_layered_constants[15].x
                && selector.b < material.runner_layered_constants[16].x {
                let use_alternate = material.runner_layered_constants[0].x < 0.0;
                let c_offset = select(9u, 12u, use_alternate);
                let detail_c = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[c_offset],
                    material.runner_layered_constants[c_offset + 1u],
                    material.runner_layered_constants[c_offset + 2u],
                );
                sampled = blend_runner_normal(sampled, detail_c);
            }

            if selector.r > material.runner_layered_constants[20].x
                && selector.r < material.runner_layered_constants[21].x {
                let detail_d = runner_detail_normal(
                    runner_detail_normal_d_texture,
                    input.uv,
                    material.runner_layered_constants[17],
                    material.runner_layered_constants[18],
                    material.runner_layered_constants[19],
                );
                sampled = blend_runner_normal(sampled, detail_d);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if (material.runner_layered_params.y > 5.5
                && material.runner_layered_params.y < 6.5)
            || (material.runner_layered_params.y > 22.5
                && material.runner_layered_params.y < 23.5) {
            // Local full9 ABI: t1 G gates t3, t1 B selects t4,
            // then t1 R selects either t5 transform.
            let detail_a = runner_detail_normal(
                runner_detail_normal_a_texture,
                input.uv,
                material.runner_layered_constants[1],
                material.runner_layered_constants[2],
                material.runner_layered_constants[3],
            );
            let detail_b = runner_detail_normal(
                runner_detail_normal_b_texture,
                input.uv,
                material.runner_layered_constants[5],
                material.runner_layered_constants[6],
                material.runner_layered_constants[7],
            );
            var selected_r = false;
            if selector.r > material.runner_layered_constants[16].x
                && selector.r < material.runner_layered_constants[17].x {
                let detail_c = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[10],
                    material.runner_layered_constants[11],
                    material.runner_layered_constants[12],
                );
                sampled = blend_runner_normal(sampled, detail_c);
                selected_r = true;
            }
            if !selected_r && selector.r >= material.runner_layered_constants[17].x {
                let detail_d = runner_detail_normal(
                    runner_detail_normal_d_texture,
                    input.uv,
                    material.runner_layered_constants[13],
                    material.runner_layered_constants[14],
                    material.runner_layered_constants[15],
                );
                sampled = blend_runner_normal(sampled, detail_d);
            }
            if selector.b > material.runner_layered_constants[8].x
                && selector.b < material.runner_layered_constants[9].x {
                sampled = blend_runner_normal(sampled, detail_b);
            }
            if abs(round(selector.g - material.runner_layered_constants[4].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, detail_a);
            }
            if material.runner_layered_params.y > 22.5 {
                let response = textureSampleBias(
                    procedural_or_response_texture, material_sampler, input.uv,
                    material.sampler_params.x);
                var response_strength = mix(0.9, 1.0, response.r);
                // E4DB extends mode 23 with its t9/t10/t11 procedural
                // condition stack. Non-E4DB mode-23 rows stay zero.
                if material.runner_layered_constants[18].x != 0.0 {
                    response_strength = min(
                        response_strength,
                        mix(0.9, 1.0, runner_e4db_condition(input)),
                    );
                }
                sampled = normalize(vec3<f32>(sampled.xy * response_strength, sampled.z));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if (material.runner_layered_params.y > 8.5
                && material.runner_layered_params.y < 9.5)
            || (material.runner_layered_params.y > 24.5
                && material.runner_layered_params.y < 25.5) {
            // 80A9C3F6: t7 base; t6 R-band, t5 B-band, t4 G-gate.
            if selector.r > material.runner_layered_constants[13].x
                && selector.r < material.runner_layered_constants[14].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[10],
                    material.runner_layered_constants[11],
                    material.runner_layered_constants[12],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if selector.b > material.runner_layered_constants[8].x
                && selector.b < material.runner_layered_constants[9].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_b_texture,
                    input.uv,
                    material.runner_layered_constants[5],
                    material.runner_layered_constants[6],
                    material.runner_layered_constants[7],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if abs(round(selector.g - material.runner_layered_constants[4].x)) > 0.5 {
                let detail = runner_detail_normal(
                    runner_detail_normal_a_texture,
                    input.uv,
                    material.runner_layered_constants[1],
                    material.runner_layered_constants[2],
                    material.runner_layered_constants[3],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if material.runner_layered_params.y > 24.5 {
                let response = textureSampleBias(
                    procedural_or_response_texture, material_sampler, input.uv,
                    material.sampler_params.x);
                sampled = normalize(vec3<f32>(sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 9.5
            && material.runner_layered_params.y < 10.5 {
            // Full11 ABI (80A9A9AD/80A9A9B1): t4 G/B/R select four
            // independently transformed normals over the remapped t9 base.
            let base_texel = textureSampleBias(
                normal_texture,
                material_sampler,
                input.uv,
                material.sampler_params.x,
            );
            let base_xy = base_texel.xy * material.runner_layered_constants[23].x
                + vec2<f32>(material.runner_layered_constants[23].y);
            sampled = normalize(vec3<f32>(
                base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0)),
            ));

            if selector.g > material.runner_layered_constants[6].x
                && selector.g < material.runner_layered_constants[7].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_a_texture,
                    input.uv,
                    material.runner_layered_constants[0],
                    material.runner_layered_constants[1],
                    material.runner_layered_constants[2],
                );
                sampled = blend_runner_normal(sampled, detail);
            } else if selector.g >= material.runner_layered_constants[7].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_b_texture,
                    input.uv,
                    material.runner_layered_constants[3],
                    material.runner_layered_constants[4],
                    material.runner_layered_constants[5],
                );
                sampled = blend_runner_normal(sampled, detail);
            }

            if selector.b > material.runner_layered_constants[14].x
                && selector.b < material.runner_layered_constants[15].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[8],
                    material.runner_layered_constants[9],
                    material.runner_layered_constants[10],
                );
                sampled = blend_runner_normal(sampled, detail);
            } else if selector.b >= material.runner_layered_constants[15].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[11],
                    material.runner_layered_constants[12],
                    material.runner_layered_constants[13],
                );
                sampled = blend_runner_normal(sampled, detail);
            }

            if selector.r > material.runner_layered_constants[19].x
                && selector.r < material.runner_layered_constants[20].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_d_texture,
                    input.uv,
                    material.runner_layered_constants[16],
                    material.runner_layered_constants[17],
                    material.runner_layered_constants[18],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if (material.runner_layered_params.y > 10.5
                && material.runner_layered_params.y < 11.5)
            || (material.runner_layered_params.y > 27.5
                && material.runner_layered_params.y < 28.5) {
            // 80A9B065: t1 A/G gate t3, B selects t4, R selects t5.
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5
                || abs(round(selector.g - material.runner_layered_constants[5].x)) > 0.5 {
                let detail = runner_detail_normal(
                    runner_detail_normal_a_texture,
                    input.uv,
                    material.runner_layered_constants[2],
                    material.runner_layered_constants[3],
                    material.runner_layered_constants[4],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_b_texture,
                    input.uv,
                    material.runner_layered_constants[6],
                    material.runner_layered_constants[7],
                    material.runner_layered_constants[8],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if selector.r > material.runner_layered_constants[14].x
                && selector.r < material.runner_layered_constants[15].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[11],
                    material.runner_layered_constants[12],
                    material.runner_layered_constants[13],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if material.runner_layered_params.y > 27.5 {
                let response = textureSampleBias(
                    procedural_or_response_texture, material_sampler, input.uv,
                    material.sampler_params.x);
                sampled = normalize(vec3<f32>(
                    sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 11.5
            && material.runner_layered_params.y < 12.5 {
            // 80A9DB9A: t2 A/G control t4, B selects t5, R gates t6.
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5
                || abs(round(selector.g - material.runner_layered_constants[2].x)) > 0.5 {
                let detail = runner_detail_normal(
                    runner_detail_normal_a_texture,
                    input.uv,
                    material.runner_layered_constants[3],
                    material.runner_layered_constants[4],
                    material.runner_layered_constants[5],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_b_texture,
                    input.uv,
                    material.runner_layered_constants[6],
                    material.runner_layered_constants[7],
                    material.runner_layered_constants[8],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            if selector.r > material.runner_layered_constants[14].x
                && selector.r < material.runner_layered_constants[15].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[11],
                    material.runner_layered_constants[12],
                    material.runner_layered_constants[13],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 12.5
            && material.runner_layered_params.y < 13.5 {
            // 80A9C27C: t1 A/B/R selects t3/t4/t5 over t6.
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if selector.b > material.runner_layered_constants[8].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            if selector.r > material.runner_layered_constants[13].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[10], material.runner_layered_constants[11],
                    material.runner_layered_constants[12]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 13.5
            && material.runner_layered_params.y < 14.5 {
            // 80A9DAC9: t2 G/B/R selects t4/t5/t6 over t7.
            if abs(round(selector.g - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if selector.b > material.runner_layered_constants[8].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            if selector.r > material.runner_layered_constants[13].x
                && selector.r < material.runner_layered_constants[14].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[10], material.runner_layered_constants[11],
                    material.runner_layered_constants[12]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 14.5
            && material.runner_layered_params.y < 15.5 {
            // 80A9AFBF: t2 A gates t4; B/G/R jointly select t5.
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if abs(round(selector.b - material.runner_layered_constants[5].x)) > 0.5
                && selector.g >= material.runner_layered_constants[6].x
                && selector.r > material.runner_layered_constants[11].x
                && selector.r < material.runner_layered_constants[12].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[8], material.runner_layered_constants[9],
                    material.runner_layered_constants[10]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 15.5
            && material.runner_layered_params.y < 16.5 {
            // 80AA0261/80AA0263: t1 A gates t3 while B selects t4.
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if selector.b > material.runner_layered_constants[8].x
                && selector.b < material.runner_layered_constants[9].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            // Engine t2.r is the authored material-response field. The DXIL
            // remaps it into the final tangent-normal amplitude.
            let response = textureSampleBias(
                procedural_or_response_texture, material_sampler, input.uv,
                material.sampler_params.x);
            sampled = normalize(vec3<f32>(sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 16.5
            && material.runner_layered_params.y < 17.5 {
            // 80A9B86A: t6 packs A/G/B/R selectors. The B and R bands
            // choose authored detail variants before A/G gate the stack.
            var b_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[12].x
                && selector.b < material.runner_layered_constants[13].x {
                b_detail = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]);
            } else if selector.b >= material.runner_layered_constants[13].x {
                b_detail = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[9], material.runner_layered_constants[10],
                    material.runner_layered_constants[11]);
            }
            var r_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[20].x
                && selector.r < material.runner_layered_constants[21].x {
                r_detail = runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[14], material.runner_layered_constants[15],
                    material.runner_layered_constants[16]);
            } else if selector.r >= material.runner_layered_constants[21].x {
                r_detail = runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[17], material.runner_layered_constants[18],
                    material.runner_layered_constants[19]);
            }
            var authored = blend_runner_normal(b_detail, r_detail);
            if abs(round(selector.g - material.runner_layered_constants[5].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 17.5
            && material.runner_layered_params.y < 18.5 {
            // 80A9D952: t3 G selects t5; B/R select three transformed
            // variants of the shared t6 detail field over the t7 base.
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.g > material.runner_layered_constants[4].x
                && selector.g < material.runner_layered_constants[5].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            var shared_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[12].x
                && selector.b < material.runner_layered_constants[13].x {
                shared_detail = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]);
            } else if selector.b >= material.runner_layered_constants[13].x {
                shared_detail = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[9], material.runner_layered_constants[10],
                    material.runner_layered_constants[11]);
            }
            if selector.r > material.runner_layered_constants[17].x
                && selector.r < material.runner_layered_constants[18].x {
                shared_detail = blend_runner_normal(shared_detail, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[14], material.runner_layered_constants[15],
                    material.runner_layered_constants[16]));
            }
            sampled = blend_runner_normal(sampled, blend_runner_normal(authored, shared_detail));
            // Engine t4.r owns the same normal-response output in this larger
            // generated permutation.
            let response = textureSampleBias(
                procedural_or_response_texture, material_sampler, input.uv,
                material.sampler_params.x);
            sampled = normalize(vec3<f32>(sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 18.5
            && material.runner_layered_params.y < 19.5 {
            // AFB8/AFBA: t9 is the base tangent normal. t8 is transformed by
            // c94/c95, remapped by c96, and selected by t2.r's c97/c98 band.
            let detail = runner_detail_normal(
                runner_detail_normal_a_texture,
                input.uv,
                material.runner_layered_constants[1],
                material.runner_layered_constants[2],
                material.runner_layered_constants[3],
            );
            let gate = select(
                0.0,
                1.0,
                selector.r > material.runner_layered_constants[14].x
                    && selector.r < material.runner_layered_constants[15].x,
            );
            sampled = normalize(mix(sampled, blend_runner_normal(sampled, detail), gate));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if (material.runner_layered_params.y > 19.5
                && material.runner_layered_params.y < 20.5)
            || (material.runner_layered_params.y > 26.5
                && material.runner_layered_params.y < 27.5) {
            // 80A9BD17/80A9DA29: t2 G/B choose t4/t5. R selects one of
            // two authored transforms of t6 before the t7 base normal.
            var r_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[16].x
                && selector.r < material.runner_layered_constants[17].x {
                if material.runner_layered_constants[0].x < 0.0 {
                    r_detail = runner_detail_normal(
                        runner_detail_normal_d_texture, input.uv,
                        material.runner_layered_constants[13], material.runner_layered_constants[14],
                        material.runner_layered_constants[15]);
                } else {
                    r_detail = runner_detail_normal(
                        runner_detail_normal_c_texture, input.uv,
                        material.runner_layered_constants[10], material.runner_layered_constants[11],
                        material.runner_layered_constants[12]);
                }
            } else if selector.r >= material.runner_layered_constants[17].x {
                r_detail = runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[13], material.runner_layered_constants[14],
                    material.runner_layered_constants[15]);
            }
            sampled = blend_runner_normal(sampled, r_detail);
            if selector.b > material.runner_layered_constants[8].x
                && selector.b < material.runner_layered_constants[9].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            if abs(round(selector.g - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if material.runner_layered_params.y > 26.5 {
                // 80A9C96F uses t4.r as authored normal response after the
                // same G/B/dual-R stack. t4.g also owns AO in material pass.
                let response = textureSampleBias(
                    procedural_or_response_texture, material_sampler, input.uv,
                    material.sampler_params.x);
                sampled = normalize(vec3<f32>(
                    sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 20.5
            && material.runner_layered_params.y < 21.5 {
            // 80A9E64F adds the A gate to the same dual-R generated ABI.
            var r_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[17].x
                && selector.r < material.runner_layered_constants[18].x {
                if material.runner_layered_constants[0].x < 0.0 {
                    r_detail = runner_detail_normal(
                        runner_detail_normal_d_texture, input.uv,
                        material.runner_layered_constants[14], material.runner_layered_constants[15],
                        material.runner_layered_constants[16]);
                } else {
                    r_detail = runner_detail_normal(
                        runner_detail_normal_c_texture, input.uv,
                        material.runner_layered_constants[11], material.runner_layered_constants[12],
                        material.runner_layered_constants[13]);
                }
            } else if selector.r >= material.runner_layered_constants[18].x {
                r_detail = runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[14], material.runner_layered_constants[15],
                    material.runner_layered_constants[16]);
            }
            sampled = blend_runner_normal(sampled, r_detail);
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]));
            }
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5
                || abs(round(selector.g - material.runner_layered_constants[2].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[3], material.runner_layered_constants[4],
                    material.runner_layered_constants[5]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 21.5
            && material.runner_layered_params.y < 22.5 {
            // 80A9C31E: t1.b selects t4; t1.g adds t3; t1.a gates the
            // resulting authored stack over the t5 base normal.
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                authored = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]);
            } else if selector.b >= material.runner_layered_constants[10].x {
                authored = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]);
            }
            if abs(round(selector.g - material.runner_layered_constants[5].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            let response = textureSampleBias(
                procedural_or_response_texture, material_sampler, input.uv,
                material.sampler_params.x);
            sampled = normalize(vec3<f32>(sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 25.5
            && material.runner_layered_params.y < 26.5 {
            // 80A9DCF8: selector B/R choose t4/t5. A/G gate authored stack.
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[6].x
                && selector.b < material.runner_layered_constants[7].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[3], material.runner_layered_constants[4],
                    material.runner_layered_constants[5]));
            }
            if selector.r > material.runner_layered_constants[11].x
                && selector.r < material.runner_layered_constants[12].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[8], material.runner_layered_constants[9],
                    material.runner_layered_constants[10]));
            }
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5
                || abs(round(selector.g - material.runner_layered_constants[2].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 28.5
            && material.runner_layered_params.y < 29.5 {
            // 80A9D2D7: B switches t4/t5, R switches t6/t7, then A/G gate
            // their authored composite over t8 base.
            var b_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                b_detail = runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]);
            } else if selector.b >= material.runner_layered_constants[10].x {
                b_detail = runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[3], material.runner_layered_constants[4],
                    material.runner_layered_constants[5]);
            }
            var r_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[17].x
                && selector.r < material.runner_layered_constants[18].x {
                r_detail = runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[11], material.runner_layered_constants[12],
                    material.runner_layered_constants[13]);
            } else if selector.r >= material.runner_layered_constants[18].x {
                r_detail = runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[14], material.runner_layered_constants[15],
                    material.runner_layered_constants[16]);
            }
            let authored = blend_runner_normal(b_detail, r_detail);
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5
                || abs(round(selector.g - material.runner_layered_constants[2].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 29.5
            && material.runner_layered_params.y < 30.5 {
            // 80A9D569: independent A/G details plus B/R-gated t8.
            if abs(round(selector.a - material.runner_layered_constants[4].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            if abs(round(selector.g - material.runner_layered_constants[8].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            if selector.b >= material.runner_layered_constants[9].x
                && selector.r > material.runner_layered_constants[14].x
                && selector.r < material.runner_layered_constants[15].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[11], material.runner_layered_constants[12],
                    material.runner_layered_constants[13]));
            }
            let response = textureSampleBias(
                procedural_or_response_texture, material_sampler, input.uv,
                material.sampler_params.x);
            sampled = normalize(vec3<f32>(
                sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 30.5
            && material.runner_layered_params.y < 31.5 {
            // Emerald full character surface (80A9F4E5). Engine remaps t7
            // with c32, then composes t4 from the t2 A gate, t5 from its B
            // band, and t6 from its R band.
            let base_texel = textureSampleBias(
                normal_texture,
                material_sampler,
                input.uv,
                material.sampler_params.x,
            );
            let base_xy = base_texel.xy * material.runner_layered_constants[13].x
                + vec2<f32>(material.runner_layered_constants[13].y);
            sampled = normalize(vec3<f32>(
                base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0)),
            ));

            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[6].x
                && selector.b < material.runner_layered_constants[7].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[3],
                    material.runner_layered_constants[4],
                    material.runner_layered_constants[5]));
            }
            if selector.r > material.runner_layered_constants[11].x
                && selector.r < material.runner_layered_constants[12].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[8],
                    material.runner_layered_constants[9],
                    material.runner_layered_constants[10]));
            }
            if abs(round(selector.a - material.runner_layered_constants[2].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[0],
                    material.runner_layered_constants[14],
                    material.runner_layered_constants[1]));
            }
            sampled = blend_runner_normal(sampled, authored);
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 31.5
            && material.runner_layered_params.y < 32.5 {
            // Emerald full13 sibling (80A9F500): t4/t5/t6/t7 are selected
            // by t2 A/G/B/R and composed over remapped t8.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[16].x
                + vec2<f32>(material.runner_layered_constants[16].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if abs(round(selector.a - material.runner_layered_constants[3].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[0], material.runner_layered_constants[1],
                    material.runner_layered_constants[2]));
            }
            if abs(round(selector.g - material.runner_layered_constants[7].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[4], material.runner_layered_constants[5],
                    material.runner_layered_constants[6]));
            }
            if selector.b > material.runner_layered_constants[11].x
                && selector.b < material.runner_layered_constants[12].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[8], material.runner_layered_constants[9],
                    material.runner_layered_constants[10]));
            }
            if selector.r > material.runner_layered_constants[14].x
                && selector.r < material.runner_layered_constants[15].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[8], material.runner_layered_constants[9],
                    material.runner_layered_constants[13]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 32.5
            && material.runner_layered_params.y < 33.5 {
            // Emerald compact sibling (80A9F518): t4 is A-gated; two t5
            // transforms are selected by t2 B/R over remapped t7.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[16].x
                + vec2<f32>(material.runner_layered_constants[16].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if abs(round(selector.a - material.runner_layered_constants[3].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[0], material.runner_layered_constants[1],
                    material.runner_layered_constants[2]));
            }
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[4], material.runner_layered_constants[5],
                    material.runner_layered_constants[6]));
            }
            if selector.r > material.runner_layered_constants[14].x
                && selector.r < material.runner_layered_constants[15].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[11], material.runner_layered_constants[12],
                    material.runner_layered_constants[13]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 33.5
            && material.runner_layered_params.y < 34.5 {
            // Procedural Emerald sibling (80A9F4B3): A/G/B/R select t4,
            // t5, t6 and a second t4 transform over remapped t7.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[21].x
                + vec2<f32>(material.runner_layered_constants[21].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if abs(round(selector.a - material.runner_layered_constants[3].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[0], material.runner_layered_constants[1],
                    material.runner_layered_constants[2]));
            }
            if abs(round(selector.g - material.runner_layered_constants[7].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[4], material.runner_layered_constants[5],
                    material.runner_layered_constants[6]));
            }
            if selector.b > material.runner_layered_constants[14].x
                && selector.b < material.runner_layered_constants[15].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[8], material.runner_layered_constants[9],
                    material.runner_layered_constants[10]));
            }
            if selector.r > material.runner_layered_constants[19].x
                && selector.r < material.runner_layered_constants[20].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[16], material.runner_layered_constants[17],
                    material.runner_layered_constants[18]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 34.5
            && material.runner_layered_params.y < 35.5 {
            // Compact character surface (80A9F528): t4 from A and t5 from R
            // are composed over the remapped t6 base.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[9].x
                + vec2<f32>(material.runner_layered_constants[9].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if abs(round(selector.a - material.runner_layered_constants[3].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[0], material.runner_layered_constants[1],
                    material.runner_layered_constants[2]));
            }
            if selector.r > material.runner_layered_constants[7].x
                && selector.r < material.runner_layered_constants[8].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[4], material.runner_layered_constants[5],
                    material.runner_layered_constants[6]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 35.5
            && material.runner_layered_params.y < 36.5 {
            // Local panel surface (80A9F589): transformed t3 is selected by
            // t1.r over the remapped t4 base.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[5].x
                + vec2<f32>(material.runner_layered_constants[5].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if selector.r > material.runner_layered_constants[3].x
                && selector.r < material.runner_layered_constants[4].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[0], material.runner_layered_constants[1],
                    material.runner_layered_constants[2]));
            }
            let response = textureSampleBias(
                procedural_or_response_texture, material_sampler, input.uv,
                material.sampler_params.x);
            sampled = normalize(vec3<f32>(sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 36.5
            && material.runner_layered_params.y < 37.5 {
            // 80B142A5 full13: t3 A/G/B/R selector; t5/t6 A/G;
            // t7/t8 B variants; t7 R; t9 remapped base.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[21].x
                + vec2<f32>(material.runner_layered_constants[21].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if selector.a > material.runner_layered_constants[4].x
                && selector.a < material.runner_layered_constants[5].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            if selector.g > material.runner_layered_constants[9].x
                && selector.g < material.runner_layered_constants[10].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]));
            }
            var b_detail = vec3<f32>(0.0, 0.0, 1.0);
            if selector.b > material.runner_layered_constants[17].x
                && selector.b < material.runner_layered_constants[18].x {
                b_detail = select(
                    runner_detail_normal(runner_detail_normal_c_texture, input.uv,
                        material.runner_layered_constants[11], material.runner_layered_constants[12],
                        material.runner_layered_constants[13]),
                    runner_detail_normal(runner_detail_normal_d_texture, input.uv,
                        material.runner_layered_constants[14], material.runner_layered_constants[15],
                        material.runner_layered_constants[16]),
                    material.runner_layered_constants[0].x < 0.0);
            } else if selector.b >= material.runner_layered_constants[18].x {
                b_detail = runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[14], material.runner_layered_constants[15],
                    material.runner_layered_constants[16]);
            }
            sampled = blend_runner_normal(sampled, b_detail);
            if selector.r > material.runner_layered_constants[19].x
                && selector.r < material.runner_layered_constants[20].x
                && material.runner_layered_constants[0].x >= 0.0 {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[11], material.runner_layered_constants[12],
                    material.runner_layered_constants[13]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 37.5
            && material.runner_layered_params.y < 38.5 {
            // 80B143A0: t1 G/B/R builds the authored detail stack and A
            // gates it over the remapped t6 base normal.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[16].x
                + vec2<f32>(material.runner_layered_constants[16].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[14].x
                && selector.r < material.runner_layered_constants[15].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[11], material.runner_layered_constants[12],
                    material.runner_layered_constants[13]));
            }
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[6], material.runner_layered_constants[7],
                    material.runner_layered_constants[8]));
            }
            if abs(round(selector.g - material.runner_layered_constants[5].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            if abs(round(selector.a - material.runner_layered_constants[1].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 38.5
            && material.runner_layered_params.y < 39.5 {
            // 80B1444E/80B14BF9: t2 B/R selects t5/t6, then G gates t4,
            // all over the remapped t7 base.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[15].x
                + vec2<f32>(material.runner_layered_constants[15].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[13].x
                && selector.r < material.runner_layered_constants[14].x {
                authored = runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[10], material.runner_layered_constants[11],
                    material.runner_layered_constants[12]);
            } else if selector.r >= material.runner_layered_constants[14].x {
                authored = runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[10], material.runner_layered_constants[11],
                    material.runner_layered_constants[12]);
            }
            if selector.b > material.runner_layered_constants[8].x
                && selector.b < material.runner_layered_constants[9].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            } else if selector.b >= material.runner_layered_constants[9].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            if abs(round(selector.g - material.runner_layered_constants[4].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            sampled = blend_runner_normal(sampled, authored);
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 39.5
            && material.runner_layered_params.y < 40.5 {
            // 80B14701: t1 G selects t3/t4, B selects transformed t5,
            // and R gates t6 over the remapped t7 base.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[22].x
                + vec2<f32>(material.runner_layered_constants[22].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if selector.r > material.runner_layered_constants[20].x
                && selector.r < material.runner_layered_constants[21].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_d_texture, input.uv,
                    material.runner_layered_constants[17], material.runner_layered_constants[18],
                    material.runner_layered_constants[19]));
            }
            if selector.b > material.runner_layered_constants[15].x
                && selector.b < material.runner_layered_constants[16].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[9], material.runner_layered_constants[10],
                    material.runner_layered_constants[11]));
            } else if selector.b >= material.runner_layered_constants[16].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[12], material.runner_layered_constants[13],
                    material.runner_layered_constants[14]));
            }
            if selector.g > material.runner_layered_constants[7].x
                && selector.g < material.runner_layered_constants[8].x {
                let selected = select(
                    runner_detail_normal(runner_detail_normal_a_texture, input.uv,
                        material.runner_layered_constants[1], material.runner_layered_constants[2],
                        material.runner_layered_constants[3]),
                    runner_detail_normal(runner_detail_normal_b_texture, input.uv,
                        material.runner_layered_constants[4], material.runner_layered_constants[5],
                        material.runner_layered_constants[6]),
                    material.runner_layered_constants[0].x < 0.0);
                authored = blend_runner_normal(authored, selected);
            } else if selector.g >= material.runner_layered_constants[8].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[4], material.runner_layered_constants[5],
                    material.runner_layered_constants[6]));
            }
            sampled = blend_runner_normal(sampled, authored);
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 41.5
            && material.runner_layered_params.y < 42.5 {
            // 80A9AD5D: A gates the complete stack, G gates t6, B selects
            // t7, and R selects t8. t9 is the remapped base normal.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[17].x
                + vec2<f32>(material.runner_layered_constants[17].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if abs(round(selector.g - material.runner_layered_constants[4].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            if selector.b > material.runner_layered_constants[8].x
                && selector.b < material.runner_layered_constants[9].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[5], material.runner_layered_constants[6],
                    material.runner_layered_constants[7]));
            }
            if selector.r > material.runner_layered_constants[13].x
                && selector.r < material.runner_layered_constants[14].x {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_c_texture, input.uv,
                    material.runner_layered_constants[10], material.runner_layered_constants[11],
                    material.runner_layered_constants[12]));
            }
            if abs(round(selector.a - material.runner_layered_constants[0].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 42.5
            && material.runner_layered_params.y < 43.5 {
            // 80A9AD68: A gates the complete stack, G gates t5, while the
            // generated B/R bands select t6 over the remapped t7 base.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[14].x
                + vec2<f32>(material.runner_layered_constants[14].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            var authored = vec3<f32>(0.0, 0.0, 1.0);
            if abs(round(selector.g - material.runner_layered_constants[4].x)) > 0.5 {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            let b_selected = selector.b > material.runner_layered_constants[5].x
                && selector.b < material.runner_layered_constants[6].x;
            let r_selected = selector.r > material.runner_layered_constants[10].x
                && selector.r < material.runner_layered_constants[11].x;
            if b_selected || r_selected {
                authored = blend_runner_normal(authored, runner_detail_normal(
                    runner_detail_normal_b_texture, input.uv,
                    material.runner_layered_constants[7], material.runner_layered_constants[8],
                    material.runner_layered_constants[9]));
            }
            if abs(round(selector.a - material.runner_layered_constants[0].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, authored);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 43.5
            && material.runner_layered_params.y < 44.5 {
            // 80A9B71C: generated code multiplies the t3 detail by the
            // authored A/G/B/R selector tests before composing with t4.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[9].x
                + vec2<f32>(material.runner_layered_constants[9].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            let selected = abs(round(selector.a - material.runner_layered_constants[0].x)) > 0.5
                && abs(round(selector.g - material.runner_layered_constants[1].x)) > 0.5
                && selector.b > material.runner_layered_constants[2].x
                && selector.b < material.runner_layered_constants[3].x
                && selector.r > material.runner_layered_constants[7].x
                && selector.r < material.runner_layered_constants[8].x;
            if selected {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[4], material.runner_layered_constants[5],
                    material.runner_layered_constants[6]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 44.5
            && material.runner_layered_params.y < 45.5 {
            // 80A9C430 woven fabric. t5 is the remapped base normal; t3 is
            // selected by the authored t1.g band, and tiled t4 supplies the
            // fine weave relief visible across the cloth panels.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[9].x
                + vec2<f32>(material.runner_layered_constants[9].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            if selector.g > material.runner_layered_constants[4].x
                && selector.g < material.runner_layered_constants[5].x {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[1], material.runner_layered_constants[2],
                    material.runner_layered_constants[3]));
            }
            sampled = blend_runner_normal(sampled, runner_detail_normal(
                runner_detail_normal_b_texture, input.uv,
                material.runner_layered_constants[6], material.runner_layered_constants[7],
                material.runner_layered_constants[8]));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 45.5
            && material.runner_layered_params.y < 46.5 {
            // 80A9C65C compact woven fabric: authored t4 base plus the
            // transformed fine t3 weave normal.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[4].x
                + vec2<f32>(material.runner_layered_constants[4].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            sampled = blend_runner_normal(sampled, runner_detail_normal(
                runner_detail_normal_a_texture, input.uv,
                material.runner_layered_constants[1], material.runner_layered_constants[2],
                material.runner_layered_constants[3]));
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 46.5
            && material.runner_layered_params.y < 47.5 {
            // 80A9CBEE: t2 alpha and green gates multiply the transformed
            // t4 armor relief before it is composed over t5.
            let base_texel = textureSampleBias(normal_texture, material_sampler, input.uv,
                material.sampler_params.x);
            let base_xy = base_texel.xy * material.runner_layered_constants[5].x
                + vec2<f32>(material.runner_layered_constants[5].y);
            sampled = normalize(vec3<f32>(base_xy,
                sqrt(max(1.0 - dot(base_xy, base_xy), 0.0))));
            let enabled = abs(round(selector.a - material.runner_layered_constants[0].x)) > 0.5
                && abs(round(selector.g - material.runner_layered_constants[1].x)) > 0.5;
            if enabled {
                sampled = blend_runner_normal(sampled, runner_detail_normal(
                    runner_detail_normal_a_texture, input.uv,
                    material.runner_layered_constants[2], material.runner_layered_constants[3],
                    material.runner_layered_constants[4]));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 7.5
            && material.runner_layered_params.y < 8.5 {
            // 80A9AFB6: t4 base normal already sampled above. Engine applies
            // transformed t3 only inside the authored t1.r selector band.
            if selector.r > material.runner_layered_constants[4].x
                && selector.r < material.runner_layered_constants[5].x {
                let detail = runner_detail_normal(
                    runner_detail_normal_a_texture,
                    input.uv,
                    material.runner_layered_constants[1],
                    material.runner_layered_constants[2],
                    material.runner_layered_constants[3],
                );
                sampled = blend_runner_normal(sampled, detail);
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        if material.runner_layered_params.y > 6.5 {
            // Expanded local full9 ABI (80A9D6FD). Engine order is R-selected
            // t5, B-selected t3, G-gated t4, then A-gated t3.
            let detail_a = runner_detail_normal(
                runner_detail_normal_a_texture,
                input.uv,
                material.runner_layered_constants[1],
                material.runner_layered_constants[2],
                material.runner_layered_constants[3],
            );
            let detail_b = runner_detail_normal(
                runner_detail_normal_b_texture,
                input.uv,
                material.runner_layered_constants[5],
                material.runner_layered_constants[6],
                material.runner_layered_constants[7],
            );
            var selected_r = false;
            if selector.r > material.runner_layered_constants[17].x
                && selector.r < material.runner_layered_constants[18].x {
                let detail_c = runner_detail_normal(
                    runner_detail_normal_c_texture,
                    input.uv,
                    material.runner_layered_constants[11],
                    material.runner_layered_constants[12],
                    material.runner_layered_constants[13],
                );
                sampled = blend_runner_normal(sampled, detail_c);
                selected_r = true;
            }
            if !selected_r && selector.r >= material.runner_layered_constants[18].x {
                let detail_d = runner_detail_normal(
                    runner_detail_normal_d_texture,
                    input.uv,
                    material.runner_layered_constants[14],
                    material.runner_layered_constants[15],
                    material.runner_layered_constants[16],
                );
                sampled = blend_runner_normal(sampled, detail_d);
            }
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                sampled = blend_runner_normal(sampled, detail_a);
            }
            if abs(round(selector.g - material.runner_layered_constants[8].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, detail_b);
            }
            if abs(round(selector.a - material.runner_layered_constants[4].x)) > 0.5 {
                sampled = blend_runner_normal(sampled, detail_a);
            }
            if material.runner_layered_params.y > 23.5 {
                let response = textureSampleBias(
                    procedural_or_response_texture, material_sampler, input.uv,
                    material.sampler_params.x);
                sampled = normalize(vec3<f32>(sampled.xy * mix(0.9, 1.0, response.r), sampled.z));
            }
            return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
        }
        let row_a = material.runner_layered_constants[1];
        let row_b = material.runner_layered_constants[2];
        let remap_a = material.runner_layered_constants[3];
        let uv_a = vec2<f32>(
            dot(row_a.xy, input.uv) + row_a.z,
            dot(row_b.xy, input.uv) + row_b.z,
        );
        let detail_a_sample = textureSampleBias(
            runner_detail_normal_a_texture,
            material_sampler,
            uv_a,
            material.sampler_params.x,
        );
        let detail_a_xy = detail_a_sample.xy * remap_a.x + vec2<f32>(remap_a.y);
        let detail_a = normalize(vec3<f32>(
            detail_a_xy,
            sqrt(max(1.0 - dot(detail_a_xy, detail_a_xy), 0.0)),
        ));
        let second_offset = select(6u, 4u, material.runner_layered_params.y > 1.5);
        let second_row_a = material.runner_layered_constants[second_offset];
        let second_row_b = material.runner_layered_constants[second_offset + 1u];
        let remap_b = material.runner_layered_constants[second_offset + 2u];
        let uv_b = vec2<f32>(
            dot(second_row_a.xy, input.uv) + second_row_a.z,
            dot(second_row_b.xy, input.uv) + second_row_b.z,
        );
        let detail_b_sample = textureSampleBias(
            runner_detail_normal_b_texture,
            material_sampler,
            uv_b,
            material.sampler_params.x,
        );
        let detail_b_xy = detail_b_sample.xy * remap_b.x + vec2<f32>(remap_b.y);
        let detail_b = normalize(vec3<f32>(
            detail_b_xy,
            sqrt(max(1.0 - dot(detail_b_xy, detail_b_xy), 0.0)),
        ));
        if material.runner_layered_params.y < 1.5 {
            // Full9: c29/c30 and c34/c35, gated by t1.b/t1.r bounds.
            let lower_a = material.runner_layered_constants[4].x;
            let upper_a = material.runner_layered_constants[5].x;
            if selector.b > lower_a && selector.b < upper_a {
                sampled = normalize(vec3<f32>(
                    sampled.xy + detail_a.xy,
                    sampled.z * detail_a.z,
                ));
            }
            let lower_b = material.runner_layered_constants[9].x;
            let upper_b = material.runner_layered_constants[10].x;
            if selector.r > lower_b && selector.r < upper_b {
                sampled = normalize(vec3<f32>(
                    sampled.xy + detail_b.xy,
                    sampled.z * detail_b.z,
                ));
            }
        } else if material.runner_layered_params.y > 3.5
            && material.runner_layered_params.y < 4.5 {
            // Procedural full9 ABI (80A9AFB4): t2 G/B select t4/t5.
            if selector.g > material.runner_layered_constants[4].x
                && selector.g < material.runner_layered_constants[5].x {
                sampled = blend_runner_normal(sampled, detail_a);
            }
            if selector.b > material.runner_layered_constants[9].x
                && selector.b < material.runner_layered_constants[10].x {
                sampled = blend_runner_normal(sampled, detail_b);
            }
        } else if material.runner_layered_params.y > 4.5
            && material.runner_layered_params.y < 5.5 {
            // Sibling switched ABI: t1.r below c15 enables the sole t2 detail.
            if selector.r < material.runner_layered_constants[0].x {
                sampled = blend_runner_normal(sampled, detail_a);
            }
        } else {
            // Switched dual-layer ABI: t1.r < c83 chooses t2, otherwise t3.
            let chosen = select(detail_b, detail_a, selector.r < material.runner_layered_constants[0].x);
            sampled = normalize(vec3<f32>(
                sampled.xy + chosen.xy,
                sampled.z * chosen.z,
            ));
        }
    }
    return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
}

fn fallback_surface(_albedo: vec3<f32>) -> vec2<f32> {
    // Unknown material families stay neutral rather than inventing surface
    // properties from albedo. Authored/Tiger evidence should replace this.
    return vec2<f32>(0.82, 0.0);
}

fn gear_palette_index(control: vec3<f32>) -> i32 {
    let low_r = control.r <= 0.5;
    let low_g = control.g <= 0.5;
    let low_b = control.b <= 0.5;
    if !low_r && low_g && low_b { return 0; }
    if low_r && !low_g && low_b { return 1; }
    if low_r && low_g && !low_b { return 2; }
    if !low_r && !low_g && low_b { return 3; }
    if low_r && !low_g && !low_b { return 4; }
    if !(low_r && low_g && low_b) { return 5; }
    return -1;
}

fn material_surface(input: VertexOutput, albedo: vec3<f32>) -> vec2<f32> {
    let uv = input.uv;
    if material.solid_surface.z > 0.5 {
        return material.solid_surface.xy;
    }
    if material.character_params.x > 0.5 {
        // Compatibility preview approximation. Audited body RT1.a is literal
        // .67, while RGB vector length carries a separate response. This fixed
        // viewer roughness does not decode the authored MRT material contract.
        return vec2<f32>(0.67, 0.0);
    }
    if (material.runner_layered_params.y > 15.5
            && material.runner_layered_params.y < 16.5)
        || (material.runner_layered_params.y > 17.5
            && material.runner_layered_params.y < 18.5) {
        // AA0261/AA0263 t2.g and D952 t4.g feed the final roughness output.
        // The generated shaders average it with their accumulated base
        // response; retain Quicktag's decoded class metalness unchanged.
        let response = textureSampleBias(
            procedural_or_response_texture,
            material_sampler,
            uv,
            material.sampler_params.x,
        );
        let class_surface = fallback_surface(albedo);
        return vec2<f32>(
            clamp((class_surface.x + clamp(response.g, 0.0, 1.0)) * 0.5, 0.04, 1.0),
            class_surface.y,
        );
    }
    if (material.runner_layered_params.y > 41.5
            && material.runner_layered_params.y < 43.5) {
        // AD5D/AD68 share the authored 80A613F1 procedural response field.
        // It is a powered scalar response, not colour or a decal. Preserve
        // its high-frequency material breakup in the roughness target.
        let response = textureSampleBias(
            pattern_or_runner_procedural_texture,
            material_sampler,
            uv,
            material.sampler_params.x,
        ).r;
        let class_surface = fallback_surface(albedo);
        return vec2<f32>(
            clamp(mix(class_surface.x, class_surface.x * 0.72, response), 0.04, 1.0),
            class_surface.y,
        );
    }
    if material.blend.z < 0.5 {
        return fallback_surface(albedo);
    }
    let control = textureSampleBias(
        control_texture,
        material_sampler,
        uv,
        material.sampler_params.x,
    );
    if material.wear_params.w > 1.5 && material.wear_params.w < 2.5 {
        // Common-weapon MRT ABI: projected t4 is a physical-detail field.
        // DXIL mixes t3.a toward class roughness with that field. It never
        // multiplies albedo; doing so clips bright skins to white.
        let detail = runner_triplanar_scalar(
            pattern_or_runner_procedural_texture,
            input.procedural_position * material.sampler_params.y,
            input.procedural_normal,
            material.wear_scratches_remap_base,
            material.wear_scratches_remap_scale.x,
        );
        let detail_mix = clamp(
            material.wear_scratches_remap_scale.z
                + material.wear_scratches_remap_scale.w * detail,
            0.0,
            1.0,
        );
        let response = textureSampleBias(
            wear_scratches_texture,
            material_sampler,
            uv,
            material.sampler_params.x,
        ).r;
        return vec2<f32>(
            clamp(mix(control.a, material.wear_scratches_remap_scale.y, detail_mix), 0.04, 1.0),
            clamp(response, 0.0, 1.0),
        );
    }
    var authored = control.r;
    if material.blend.y > 1.5 {
        authored = control.g;
    }
    if material.blend.y > 2.5 {
        authored = control.b;
    }
    if material.blend.y > 3.5 {
        authored = control.a;
    }
    // Tiger feeds one authored control channel through independent roughness
    // and metal remaps. The inventory match intentionally filters both maps:
    // large painted surfaces are rough dielectrics, not coated show-car metal.
    var roughness_remap = material.roughness_remap;
    var metal_remap = material.metal_remap;
    var package_gear_surface = false;
    if material.palette_tertiary.a > 0.5 {
        let palette_index = gear_palette_index(control.rgb);
        if palette_index >= 0 {
            roughness_remap = material.gear_palette_roughness[u32(palette_index)];
            metal_remap = material.gear_palette_metal[u32(palette_index)];
            package_gear_surface = true;
        }
    }
    let authored_roughness = authored * roughness_remap.x + roughness_remap.y;
    let authored_metalness = authored * metal_remap.x + metal_remap.y;
    if package_gear_surface {
        // GearDye supplies a complete per-region surface response. Preserve it
        // directly: the old generic matte clamps erased smooth metallic trim
        // and therefore its colored inventory-environment reflection.
        return vec2<f32>(
            clamp(authored_roughness, 0.04, 1.0),
            clamp(authored_metalness, 0.0, 1.0),
        );
    }
    let class_surface = fallback_surface(albedo);
    let roughness = max(mix(class_surface.x, authored_roughness, 0.45), 0.42);
    let luma = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
    let painted_limit = select(0.15, 0.80, luma > 0.32 && class_surface.y > 0.2);
    let metalness = min(clamp(authored_metalness * 0.70, 0.0, 0.80), painted_limit);
    return vec2<f32>(clamp(roughness, 0.42, 0.92), metalness);
}

fn fresnel_schlick(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cosine, 5.0);
}

fn grazing_boost(n_dot_v: f32, roughness: f32) -> f32 {
    let edge = pow(1.0 - clamp(n_dot_v, 0.0, 1.0), 4.0);
    // Stronger on smoother surfaces, weaker on rough plastic.
    return 1.0 + edge * mix(2.2, 0.8, roughness);
}

fn ggx_specular(
    normal: vec3<f32>,
    light: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let n_dot_l = max(dot(normal, light), 0.0);
    let n_dot_v = max(dot(normal, view), 0.0);
    let half_direction = normalize(light + view);
    let n_dot_h = max(dot(normal, half_direction), 0.0);
    let v_dot_h = max(dot(view, half_direction), 0.0);
    let alpha = max(roughness * roughness, 0.0064);
    let alpha2 = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let distribution = alpha2 / max(3.14159265 * denominator * denominator, 0.0001);
    let visibility_v = n_dot_l * sqrt(max(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2, 0.0));
    let visibility_l = n_dot_v * sqrt(max(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2, 0.0));
    let visibility = 0.5 / max(visibility_v + visibility_l, 0.0001);
    return distribution * visibility * fresnel_schlick(v_dot_h, f0);
}

fn clamp_specular_luminance(value: vec3<f32>, maximum: f32) -> vec3<f32> {
    let luminance = dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
    return value * min(1.0, maximum / max(luminance, 0.000001));
}

// Small Tiger/Alkahest-style comparison kernel for the engine-faithful path.
const STRICT_SHADOW_SAMPLE_COUNT = 9u;
const STRICT_SHADOW_DISK = array<vec2<f32>, 9>(
    vec2<f32>( 0.0,  0.0),
    vec2<f32>(-0.45570818, -0.80781116),
    vec2<f32>( 0.89600897, -0.35694158),
    vec2<f32>(-0.19807512,  0.92104191),
    vec2<f32>( 0.25952172, -0.80905122),
    vec2<f32>( 0.59722185,  0.77891493),
    vec2<f32>(-0.80889258,  0.45309126),
    vec2<f32>(-0.91679038, -0.21303266),
    vec2<f32>( 0.67728304,  0.18318819),
);

fn strict_tiger_shadow(shadow_position: vec3<f32>, position_screen: vec2<f32>) -> f32 {
    let dimensions = vec2<f32>(textureDimensions(sun_shadow));
    let texel = 1.0 / dimensions;
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let angle = fract(magic.z * fract(dot(position_screen, magic.xy))) * 6.28318530718;
    let sine = sin(angle);
    let cosine = cos(angle);
    var visibility = 0.0;
    for (var sample = 0u; sample < STRICT_SHADOW_SAMPLE_COUNT; sample++) {
        let poisson = STRICT_SHADOW_DISK[sample];
        let rotated = vec2<f32>(
            cosine * poisson.x - sine * poisson.y,
            sine * poisson.x + cosine * poisson.y,
        );
        let sample_uv = shadow_position.xy + rotated * texel * 2.5;
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            visibility += 1.0;
            continue;
        }
        visibility += textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            sample_uv,
            shadow_position.z - 0.0001,
        );
    }
    return visibility / f32(STRICT_SHADOW_SAMPLE_COUNT);
}

// Stable Vogel disk for blocker search and final comparison filtering.
// The radius is no longer a fixed screen-space blur: blocker distance drives
// the penumbra so contact shadows remain attached and naturally hard.
const SHADOW_SAMPLE_COUNT = 32u;
const SHADOW_BLOCKER_SAMPLE_COUNT = 16u;
const SHADOW_DISK = array<vec2<f32>, 32>(
    vec2<f32>( 0.1250,  0.0000), vec2<f32>(-0.1596,  0.1462),
    vec2<f32>( 0.0244, -0.2784), vec2<f32>( 0.2012,  0.2625),
    vec2<f32>(-0.3693, -0.0653), vec2<f32>( 0.3498, -0.2225),
    vec2<f32>(-0.1170,  0.4352), vec2<f32>(-0.2231, -0.4296),
    vec2<f32>( 0.4841,  0.1768), vec2<f32>(-0.5036,  0.2079),
    vec2<f32>( 0.2428, -0.5188), vec2<f32>( 0.1794,  0.5720),
    vec2<f32>(-0.5408, -0.3134), vec2<f32>( 0.6344, -0.1395),
    vec2<f32>(-0.3871,  0.5507), vec2<f32>(-0.0894, -0.6902),
    vec2<f32>( 0.5491,  0.4628), vec2<f32>(-0.7389,  0.0306),
    vec2<f32>( 0.5390, -0.5363), vec2<f32>(-0.0361,  0.7798),
    vec2<f32>(-0.5128, -0.6145), vec2<f32>( 0.8124,  0.1093),
    vec2<f32>(-0.6883,  0.4789), vec2<f32>( 0.1881, -0.8361),
    vec2<f32>( 0.4350,  0.7592), vec2<f32>(-0.8504, -0.2713),
    vec2<f32>( 0.8261, -0.3817), vec2<f32>(-0.3579,  0.8552),
    vec2<f32>(-0.3194, -0.8880), vec2<f32>( 0.8499,  0.4467),
    vec2<f32>(-0.9440,  0.2488), vec2<f32>( 0.5366, -0.8345),
);

fn shadow_linear_depth(depth: f32) -> f32 {
    let near_plane = scene.shadow_parameters.y;
    let range = max(scene.shadow_parameters.z, near_plane + 0.01 * scene.light_position.w);
    let depth_scale = range / max(range - near_plane, 0.01 * scene.light_position.w);
    return depth_scale * near_plane / max(depth_scale - depth, 0.000001);
}

fn shadow_receiver_gradient(shadow_position: vec3<f32>) -> vec2<f32> {
    let dx = dpdx(shadow_position);
    let dy = dpdy(shadow_position);
    let determinant = dx.x * dy.y - dx.y * dy.x;
    if abs(determinant) < 0.00000001 {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(
        (dx.z * dy.y - dy.z * dx.y) / determinant,
        (dx.x * dy.z - dy.x * dx.z) / determinant,
    );
}

fn shadow_reference_depth(
    shadow_position: vec3<f32>,
    sample_offset: vec2<f32>,
    depth_gradient: vec2<f32>,
    n_dot_light: f32,
) -> f32 {
    // Receiver-plane bias follows perspective depth variation for each tap.
    // The residual epsilon is tiny and expressed in depth units, not XY texels.
    let epsilon = mix(0.00000025, 0.00000125, 1.0 - clamp(n_dot_light, 0.0, 1.0));
    return shadow_position.z + dot(depth_gradient, sample_offset) - epsilon;
}

fn pcss_shadow(shadow_position: vec3<f32>, n_dot_light: f32) -> f32 {
    let shadow_dimensions = vec2<i32>(textureDimensions(sun_shadow));
    let shadow_texel = 1.0 / vec2<f32>(shadow_dimensions);
    let depth_gradient = shadow_receiver_gradient(shadow_position);
    let softness = clamp(scene.params1.w, 0.0, 1.0);
    let center_reference = shadow_reference_depth(
        shadow_position,
        vec2<f32>(0.0),
        depth_gradient,
        n_dot_light,
    );
    if softness <= 0.001 {
        return textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            shadow_position.xy,
            center_reference,
        );
    }

    let receiver_depth = shadow_linear_depth(shadow_position.z);
    let outer_cosine = clamp(scene.light_parameters.z, 0.01, 0.9999);
    let cone_tangent =
        sqrt(max(1.0 - outer_cosine * outer_cosine, 0.0)) / outer_cosine;
    let source_radius = scene.shadow_parameters.x;
    let max_search_radius = max(shadow_texel.x, shadow_texel.y) * 12.0;
    let search_radius = min(
        0.5 * source_radius / max(receiver_depth * cone_tangent, 0.000001),
        max_search_radius,
    );

    var blocker_depth_sum = 0.0;
    var blocker_count = 0u;
    for (var sample = 0u; sample < SHADOW_BLOCKER_SAMPLE_COUNT; sample++) {
        let offset = SHADOW_DISK[sample] * search_radius;
        let sample_uv = shadow_position.xy + offset;
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            continue;
        }
        let sample_pixel = clamp(
            vec2<i32>(sample_uv * vec2<f32>(shadow_dimensions)),
            vec2<i32>(0),
            shadow_dimensions - vec2<i32>(1),
        );
        let blocker_depth = textureLoad(sun_shadow, sample_pixel, 0);
        let reference =
            shadow_reference_depth(shadow_position, offset, depth_gradient, n_dot_light);
        if blocker_depth < reference {
            blocker_depth_sum += shadow_linear_depth(blocker_depth);
            blocker_count += 1u;
        }
    }
    if blocker_count == 0u {
        return 1.0;
    }

    let average_blocker_depth = blocker_depth_sum / f32(blocker_count);
    let separation = max(receiver_depth - average_blocker_depth, 0.0);
    let penumbra_ratio =
        separation / max(average_blocker_depth, scene.shadow_parameters.y);
    let max_filter_radius = max(shadow_texel.x, shadow_texel.y) * 12.0;
    let filter_radius = min(
        0.5 * source_radius * penumbra_ratio
            / max(receiver_depth * cone_tangent, 0.000001),
        max_filter_radius,
    );
    if filter_radius <= max(shadow_texel.x, shadow_texel.y) * 0.35 {
        return textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            shadow_position.xy,
            center_reference,
        );
    }

    var visibility = 0.0;
    for (var sample = 0u; sample < SHADOW_SAMPLE_COUNT; sample++) {
        let offset = SHADOW_DISK[sample] * filter_radius;
        let sample_uv = shadow_position.xy + offset;
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            visibility += 1.0;
            continue;
        }
        visibility += textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            sample_uv,
            shadow_reference_depth(
                shadow_position,
                offset,
                depth_gradient,
                n_dot_light,
            ),
        );
    }
    return visibility / f32(SHADOW_SAMPLE_COUNT);
}

fn spotlight_shadow(input: VertexOutput) -> f32 {
    let shadow_clip = light_clip(scene.center.xyz + input.world_relative);
    if shadow_clip.w <= 0.0001 {
        return 1.0;
    }
    let shadow_position = vec3<f32>(
        shadow_clip.xy / shadow_clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5),
        shadow_clip.z / shadow_clip.w,
    );
    if any(shadow_position.xy < vec2<f32>(0.0))
        || any(shadow_position.xy > vec2<f32>(1.0))
        || shadow_position.z <= 0.0
        || shadow_position.z >= 1.0 {
        return 1.0;
    }

    return strict_tiger_shadow(shadow_position, input.clip_position.xy);
}

fn investment_decal_mask(uv: vec2<f32>) -> f32 {
    let raw_mask = textureSampleBias(
        control_texture,
        material_sampler,
        uv,
        material.sampler_params.x,
    ).r;
    if material.decal_params.x > 2.5 {
        return select(0.0, 1.0, raw_mask > material.decal_mask_params.y);
    }
    if material.decal_params.x < 1.5 {
        return select(0.0, 1.0, raw_mask > material.decal_mask_params.y);
    }
    if material.decal_mask_params.x < 1.5 {
        if uv.y > 1.0 {
            return select(
                clamp(material.decal_mask_remap.z + raw_mask * material.decal_mask_remap.w, 0.0, 1.0),
                clamp(material.decal_mask_remap.x + raw_mask * material.decal_mask_remap.y, 0.0, 1.0),
                uv.x > 0.0,
            );
        }
        return select(1.0, raw_mask, uv.x > 0.0);
    }
    return select(0.0, 1.0, raw_mask > 0.0);
}

fn investment_decal_source(uv: vec2<f32>, atlas: vec3<f32>) -> vec3<f32> {
    // Scene-normal decals use the directly sampled sRGB t3 colour. Their UV
    // is ordinary mesh UV; there is no selector encoded in UV.x.
    if material.decal_params.x > 2.5 {
        return atlas;
    }
    let selector = i32(uv.x);
    if material.decal_params.x < 1.5 {
        if selector == 0 {
            return atlas;
        }
        let constant_index = selector - 1;
        if constant_index >= 0 && constant_index < i32(material.decal_params.y) {
            return material.decal_selector_colors[constant_index].rgb;
        }
        return vec3<f32>(0.0);
    }

    if uv.x > 0.0 {
        let atlas_selector_max = i32(material.decal_params.z);
        if selector >= 0 && selector <= atlas_selector_max {
            return atlas;
        }
        let constant_index = selector - atlas_selector_max - 1;
        if constant_index >= 0 && constant_index < i32(material.decal_params.y) {
            return material.decal_selector_colors[constant_index].rgb;
        }
        return vec3<f32>(0.0);
    }

    let grayscale = material.decal_mask_params.z
        + textureSampleBias(
            control_texture,
            material_sampler,
            uv,
            material.sampler_params.x,
        ).r
            * material.decal_mask_params.w;
    return vec3<f32>(grayscale);
}

fn investment_scene_normal_value(input: VertexOutput) -> vec3<f32> {
    let dimensions = vec2<i32>(textureDimensions(investment_scene_normal));
    let pixel = clamp(
        vec2<i32>(input.clip_position.xy),
        vec2<i32>(0),
        dimensions - vec2<i32>(1),
    );
    return normalize(
        textureLoad(investment_scene_normal, pixel, 0).rgb * 2.0 - vec3<f32>(1.0)
    );
}

struct FragmentOutput {
    @location(0) compatibility_hdr: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
    @location(2) material_properties: vec4<f32>,
    @location(3) albedo: vec4<f32>,
}

fn character_palette_procedural_mask(
    input: VertexOutput,
    uv_a: vec2<f32>,
    uv_b: vec2<f32>,
) -> f32 {
    // Arata palette-mask DXIL c6/c10..c15/c20/c22..c24. t3 supplies two
    // transformed mark samples; t2 supplies the object-space tri-planar field.
    let mark_a = max(textureSampleBias(
        control_texture, material_sampler, uv_a, material.sampler_params.x,
    ).r, 0.0);
    let mark_b = max(textureSampleBias(
        control_texture, material_sampler, uv_b, material.sampler_params.x,
    ).r, 0.0);
    let paired = clamp(
        pow(mark_a, material.character_procedural[8].x)
            * pow(mark_b, material.character_procedural[9].x),
        0.0,
        1.0,
    );

    let position = input.procedural_position;
    var weights = pow(
        abs(normalize(input.procedural_normal)),
        vec3<f32>(material.character_procedural[1].x),
    );
    weights /= max(weights.x + weights.y + weights.z, 0.0001);
    let projection = material.character_procedural[2];
    let field_x = textureSampleBias(
        procedural_or_response_texture,
        material_sampler,
        vec2<f32>(
            projection.x * position.y + projection.z,
            projection.y * position.z + projection.w,
        ),
        material.sampler_params.x,
    ).r;
    let field_y = textureSampleBias(
        procedural_or_response_texture,
        material_sampler,
        vec2<f32>(
            projection.x * position.x + projection.z,
            projection.y * position.z + projection.w,
        ),
        material.sampler_params.x,
    ).r;
    let field_z = textureSampleBias(
        procedural_or_response_texture,
        material_sampler,
        vec2<f32>(
            projection.x * position.x + projection.z,
            projection.y * position.y + projection.w,
        ),
        material.sampler_params.x,
    ).r;
    let field_scale = material.character_procedural[3].x;
    let field_base = material.character_procedural[4].x;
    let frequency = material.character_procedural[5].x;
    let phase = material.character_procedural[5].z;
    let stripe_remap = material.character_procedural[6];
    let stripe_scale = material.character_procedural[7].x;
    let stripe_x = clamp(
        stripe_remap.x
            + abs(fract((field_x - field_base) * field_scale + position.y)
                * frequency + phase) * stripe_remap.y,
        0.0,
        1.0,
    ) * stripe_scale;
    let stripe_y = clamp(
        stripe_remap.x
            + abs(fract((field_y - field_base) * field_scale + position.x)
                * frequency + phase) * stripe_remap.y,
        0.0,
        1.0,
    ) * stripe_scale;
    let stripe_z = clamp(
        stripe_remap.x
            + abs(fract((field_z - field_base) * field_scale + position.x)
                * frequency + phase) * stripe_remap.y,
        0.0,
        1.0,
    ) * stripe_scale;
    let field = dot(vec3<f32>(stripe_x, stripe_y, stripe_z), weights);
    let contour_position = fract(
        round(paired)
            * (field - 1.0 + paired * material.character_procedural[10].x),
    );
    let contour = material.character_procedural[0];
    return clamp(
        contour.x + abs(contour_position + contour.z) * contour.y,
        0.0,
        1.0,
    );
}

fn spotlight_factor(position: vec3<f32>) -> f32 {
    let source_to_surface = position - scene.light_position.xyz;
    let distance = length(source_to_surface);
    let source_to_surface_direction = source_to_surface / max(distance, 0.0001 * scene.light_position.w);
    let beam_axis = normalize(-scene.light_direction.xyz);
    let cone = smoothstep(
        scene.light_parameters.z,
        scene.light_parameters.w,
        dot(source_to_surface_direction, beam_axis),
    );
    let range = max(scene.light_parameters.y, 0.05 * scene.light_position.w);
    let range_fade = 1.0 - smoothstep(range * 0.70, range, distance);
    // Clamp the near-field denominator so the configured source does not
    // explode when a model surface crosses the light origin.
    let rig_distance = distance / scene.light_position.w;
    let distance_falloff = 1.0 / max(1.0, rig_distance * rig_distance);
    return cone * range_fade * distance_falloff;
}

fn shade_model(input: VertexOutput, investment_decal: bool) -> FragmentOutput {
    let base_color = textureSampleBias(
        color_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    );
    if !investment_decal && material.alpha_mask_params.x > 0.5 {
        let coverage_sample = textureSampleBias(
            control_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).r;
        let coverage = coverage_sample * material.alpha_mask_params.w
            + material.alpha_mask_params.z;
        if coverage < material.alpha_mask_params.y {
            discard;
        }
    }
    let mask_material = material.params.w < -0.5;
    var material_alpha = select(base_color.a, base_color.r, mask_material);
    if !investment_decal && ((!mask_material && base_color.a < material.params.w)
        || (mask_material && material_alpha < 0.02)) {
        discard;
    }
    let decal_blend = material.blend.x == 26.0 || material.blend.x == 27.0;
    var sampled_albedo = select(
        base_color.rgb,
        select(vec3<f32>(base_color.r), vec3<f32>(1.0), decal_blend),
        mask_material,
    );
    var semantic_dye = vec3<f32>(0.0);
    var semantic_worn_dye = vec3<f32>(0.0);
    var semantic_dye_detail = vec3<f32>(0.0);
    var semantic_dye_mask = vec3<f32>(0.0);
    if investment_decal {
        let decal_mask = investment_decal_mask(input.uv);
        material_alpha = decal_mask * material.decal_params.w;
        if material_alpha <= 0.001 {
            discard;
        }
        sampled_albedo = investment_decal_source(input.uv, base_color.rgb);
        if material.decal_params.x > 1.5 && material.decal_params.x < 2.5 {
            let detail_uv = input.uv * material.decal_detail_transform.xy
                + material.decal_detail_transform.zw;
            let detail_sample = textureSampleBias(
                normal_texture,
                material_sampler,
                detail_uv,
                material.sampler_params.x,
            ).r;
            let detail = material.decal_detail_base.rgb
                + material.decal_detail_scale.rgb * detail_sample;
            sampled_albedo *= detail * 4.5947933;
        }
    }
    if !investment_decal && material.character_params.x > 0.5 {
        let surface_sample = textureSampleBias(
            character_surface_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        );
        if material.character_params.w < 1.5 {
            // Literal common-character DXIL path: t1 detail is transformed by
            // c0, remapped by c3/c4, multiplied with t0, then selected by
            // round(t2.a - c5.x).
            let detail_uv = input.uv * material.character_detail_transform.xy
                + material.character_detail_transform.zw;
            let detail_sample = textureSampleBias(
                character_detail_color_texture,
                material_sampler,
                detail_uv,
                material.sampler_params.x,
            ).r;
            let detail = clamp(
                material.character_detail_base.rgb
                    + material.character_detail_scale.rgb * detail_sample,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            let detail_gate = clamp(
                round(surface_sample.a - material.character_params.y),
                0.0,
                1.0,
            );
            sampled_albedo = mix(
                sampled_albedo,
                sampled_albedo * detail * material.character_params.z,
                detail_gate,
            );
        } else {
            // Palette-mask family: c2/c3 and c4/c5 are two affine UV rows.
            // t1 and t3 build the authored mark; t4.g gates c0/c1 palette over
            // local t0. This retains Arata-family logos and panel colors that
            // were previously discarded with the auxiliary texture list.
            let uv_a = vec2<f32>(
                dot(material.character_detail_transform.xy, input.uv)
                    + material.character_detail_transform.z,
                dot(material.character_detail_base.xy, input.uv)
                    + material.character_detail_base.z,
            );
            let uv_b = vec2<f32>(
                dot(material.character_detail_scale.xy, input.uv)
                    + material.character_detail_scale.z,
                dot(material.character_extra[0].xy, input.uv)
                    + material.character_extra[0].z,
            );
            let mark_a = textureSampleBias(
                character_detail_color_texture,
                material_sampler,
                uv_a,
                material.sampler_params.x,
            ).r;
            let mark_b = textureSampleBias(
                character_detail_color_texture,
                material_sampler,
                uv_b,
                material.sampler_params.x,
            ).r;
            // DXIL %295..%519: transformed t1 marks plus the independent
            // t3/t2 object-space contour branch form the final palette mask.
            let procedural_mark = character_palette_procedural_mask(input, uv_a, uv_b);
            let palette_mask = clamp(mark_a + mark_b + procedural_mark, 0.0, 1.0);
            let palette_color = mix(
                material.character_palette[0].rgb,
                material.character_palette[1].rgb,
                palette_mask,
            );
            let palette_gate = clamp(
                round(surface_sample.g - material.character_params.y),
                0.0,
                1.0,
            );
            sampled_albedo = mix(sampled_albedo, palette_color, palette_gate);
        }
    }
    if !investment_decal
        && material.runner_layered_params.y > 18.5
        && material.runner_layered_params.y < 19.5 {
        // AFB8/AFBA procedural panel: t4 is sampled as a scalar pattern,
        // t5 supplies its RG modulation, and t6 is the local panel mask.
        let row_u = material.runner_layered_constants[4];
        let row_v = material.runner_layered_constants[5];
        let pattern_uv = vec2<f32>(
            dot(row_u.xy, input.uv) + row_u.z,
            dot(row_v.xy, input.uv) + row_v.z,
        );
        let pattern = textureSampleBias(
            runner_detail_normal_b_texture,
            material_sampler,
            pattern_uv,
            material.sampler_params.x,
        ).r;
        let procedural = textureSampleBias(
            procedural_or_response_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).r;
        let panel_mask = textureSampleBias(
            runner_detail_normal_c_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).r;
        let pattern_color = material.runner_layered_constants[9].rgb;
        sampled_albedo = mix(
            sampled_albedo,
            sampled_albedo * mix(vec3<f32>(1.0), pattern_color, pattern),
            clamp(panel_mask * procedural, 0.0, 1.0),
        );
    }
    if !investment_decal
        && material.runner_layered_params.y > 26.5
        && material.runner_layered_params.y < 27.5 {
        // 80A9C96F DXIL: t2 is an object-space tri-planar scalar field. It
        // modulates t0 albedo only in the authored high B selector band (or
        // the middle band when c14 selects the procedural alternative).
        let selector = textureSampleBias(
            runner_surface_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        );
        let middle_band = selector.b > material.runner_layered_constants[18].x
            && selector.b < material.runner_layered_constants[19].x;
        let procedural_selected = selector.b >= material.runner_layered_constants[19].x
            || (middle_band && material.runner_layered_constants[0].x < 0.0);
        if procedural_selected {
            let field = runner_triplanar_scalar(
                pattern_or_runner_procedural_texture,
                input.procedural_position,
                input.procedural_normal,
                material.runner_layered_constants[21],
                material.runner_layered_constants[20].x,
            );
            let field_color = material.runner_layered_constants[22].rgb
                + material.runner_layered_constants[23].rgb * field;
            sampled_albedo *= field_color * 4.5947933;
        }
    }
    if !investment_decal
        && material.runner_layered_params.y > 27.5
        && material.runner_layered_params.y < 28.5 {
        // 80A9D3FC DXIL: t0 is an unconditional object-space tri-planar
        // scalar field. c4 + c5 * field multiplies t1 albedo by Tiger's
        // authored linear-colour scale.
        let field = runner_triplanar_scalar(
            pattern_or_runner_procedural_texture,
            input.procedural_position,
            input.procedural_normal,
            material.runner_layered_constants[17],
            material.runner_layered_constants[16].x,
        );
        let field_color = material.runner_layered_constants[18].rgb
            + material.runner_layered_constants[19].rgb * field;
        sampled_albedo *= field_color * 4.5947933;
    }
    if !investment_decal
        && material.runner_layered_params.y > 29.5
        && material.runner_layered_params.y < 30.5 {
        // 80A9D569 DXIL: t1.r is an authored repeating colour mask. c1/c2
        // transform its UV, while round(t4.a - c3.x) gates the layer. This is
        // a colour operation; treating t1 as an unused auxiliary texture
        // removes runner emblems and fabric marks entirely.
        let selector = textureSampleBias(
            runner_surface_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        );
        let row_u = material.runner_color_constants[1];
        let row_v = material.runner_color_constants[2];
        let overlay_uv = vec2<f32>(
            dot(row_u.xy, input.uv) + row_u.z,
            dot(row_v.xy, input.uv) + row_v.z,
        );
        let overlay = textureSampleBias(
            character_detail_color_texture,
            material_sampler,
            overlay_uv,
            material.sampler_params.x,
        ).r;
        let gate = clamp(
            round(selector.a - material.runner_color_constants[3].x),
            0.0,
            1.0,
        );
        sampled_albedo = mix(
            sampled_albedo,
            material.runner_color_constants[0].rgb,
            clamp(overlay * gate, 0.0, 1.0),
        );
    }
    if !investment_decal
        && material.runner_layered_params.y > 36.5
        && material.runner_layered_params.y < 37.5 {
        // 80B142A5 DXIL colour branch. t2.r is the authored relief/pattern
        // field; c24/c25 transform it, c28+c29*t2 remaps its colour, and
        // round(t3.g-c30.x) selects the result over local t0 albedo.
        let selector = textureSampleBias(
            runner_surface_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        );
        let row_u = material.runner_color_constants[0];
        let row_v = material.runner_color_constants[1];
        let overlay_uv = vec2<f32>(
            dot(row_u.xy, input.uv) + row_u.z,
            dot(row_v.xy, input.uv) + row_v.z,
        );
        let relief = clamp(textureSampleBias(
            character_detail_color_texture,
            material_sampler,
            overlay_uv,
            material.sampler_params.x,
        ).r, 0.0, 1.0);
        let remapped = clamp(
            material.runner_color_constants[2].rgb
                + material.runner_color_constants[3].rgb * relief,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let layer = sampled_albedo * remapped * 4.5947933;
        let gate = clamp(
            round(selector.g - material.runner_color_constants[4].x),
            0.0,
            1.0,
        );
        sampled_albedo = mix(sampled_albedo, layer, gate);
    }
    if !investment_decal
        && material.runner_layered_params.y > 40.5
        && material.runner_layered_params.y < 41.5 {
        // 80A9B860 runner skin ABI. t1 is a shared cellular response LUT and
        // must never appear as albedo. Tiger derives skin colour from
        // c121/c122/c123 while t0 supplies local pore/recess information.
        let feature = clamp(textureSampleBias(
            runner_surface_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).r, 0.0, 1.0);
        let cellular = textureSampleBias(
            pattern_or_runner_procedural_texture,
            material_sampler,
            input.uv * material.runner_layered_constants[9].xy,
            material.sampler_params.x,
        ).r;
        let primary = clamp(
            material.runner_layered_constants[0].rgb
                + material.runner_layered_constants[1].rgb
                    * clamp(feature + (cellular - 0.5) * 0.08, 0.0, 1.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let recessed = clamp(
            material.runner_layered_constants[2].rgb,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let recessed_mask = pow(1.0 - feature, 2.0);
        sampled_albedo = mix(primary, recessed, recessed_mask);
    }
    if !investment_decal && material.sampler_params.z > 0.5 {
        // Shared-atlas detail ABI: compiled PS multiplies direct t0 RGB by a
        // triplanar linear t1 response after its c5/c6 affine remap.
        let detail = runner_triplanar_scalar(
            pattern_or_runner_procedural_texture,
            input.procedural_position,
            input.procedural_normal,
            material.pattern_projection,
            material.pattern_params.y,
        );
        let response = clamp(
            material.pattern_stripe.rgb + material.pattern_contour.rgb * detail,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        sampled_albedo *= response * 4.5947933;
    }
    let pre_pattern_albedo = sampled_albedo;
    sampled_albedo = apply_gear_pattern(sampled_albedo, input);
    if material.pattern_params.x > 0.5 {
        semantic_dye_detail = sampled_albedo;
        semantic_dye_mask = vec3<f32>(
            clamp(length(sampled_albedo - pre_pattern_albedo) * 4.0, 0.0, 1.0)
        );
    }
    if material.blend.w > 0.5 {
        let mask = dot(base_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        // Compact hair shaders use t0 as a strand/micro-shadow mask. Their
        // inline colors feed the full Tiger lighting model, not direct albedo.
        // Preserve strand detail over a neutral fiber base in the preview.
        sampled_albedo = mix(vec3<f32>(0.62), vec3<f32>(1.0), mask);
    }
    if material.palette_tertiary.a > 0.5 {
        // This is the literal scale in Marathon's compiled GearDye pixel
        // shader (DXIL 0x4012611180000000). Keep it per channel: collapsing
        // the authored color map to luminance destroys localized hue/detail.
        let detail = base_color.rgb * 4.5947933;
        // Tiger's GearDye pixel shader decodes the three control bits in this
        // exact order. ID 0 uses the technique default; IDs 1..6 select the
        // six authored object-channel outputs. The channel order is the TFX
        // ABI order, which intentionally differs from RGB binary order.
        let control = textureSampleBias(
            control_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).rgb;
        let low_r = control.r <= 0.5;
        let low_g = control.g <= 0.5;
        let low_b = control.b <= 0.5;
        var dye = material.gear_palette_default.rgb;
        var worn_dye = vec3<f32>(0.0);
        var dye_detail = vec3<f32>(0.0);
        if !low_r && low_g && low_b {
            dye = material.gear_palette_colors[0].rgb;
            worn_dye = material.gear_worn_dye_colors[0].rgb;
            dye_detail = material.gear_dye_detail_colors[0].rgb;
        } else if low_r && low_g && !low_b {
            dye = material.gear_palette_colors[2].rgb;
            worn_dye = material.gear_worn_dye_colors[2].rgb;
            dye_detail = material.gear_dye_detail_colors[2].rgb;
        } else if !low_r && !low_g && low_b {
            dye = material.gear_palette_colors[3].rgb;
            worn_dye = material.gear_worn_dye_colors[3].rgb;
            dye_detail = material.gear_dye_detail_colors[3].rgb;
        } else if low_r && !low_g && low_b {
            dye = material.gear_palette_colors[1].rgb;
            worn_dye = material.gear_worn_dye_colors[1].rgb;
            dye_detail = material.gear_dye_detail_colors[1].rgb;
        } else if low_r && !low_g && !low_b {
            dye = material.gear_palette_colors[4].rgb;
            worn_dye = material.gear_worn_dye_colors[4].rgb;
            dye_detail = material.gear_dye_detail_colors[4].rgb;
        } else if !(low_r && low_g && low_b) {
            dye = material.gear_palette_colors[5].rgb;
            worn_dye = material.gear_worn_dye_colors[5].rgb;
            dye_detail = material.gear_dye_detail_colors[5].rgb;
        }
        sampled_albedo = (dye + worn_dye + dye_detail) * detail;
        semantic_dye = dye;
        semantic_worn_dye = worn_dye;
        semantic_dye_detail = dye_detail;
        semantic_dye_mask = control.rgb;
    }
    let conditioned_dye = apply_weapon_mod_condition(sampled_albedo, input);
    sampled_albedo = conditioned_dye.rgb;
    // The user-facing Worn Dye view isolates the resolved worn surface for
    // rarity-conditioned mods. Materials without that physical ABI still show
    // their authored Worn Dye object-channel contribution.
    semantic_worn_dye = select(
        semantic_worn_dye,
        conditioned_dye.rgb,
        material.wear_params.w > 0.5 && material.wear_params.w < 1.5,
    );
    var semantic_wear_mask = conditioned_dye.a;
    sampled_albedo = apply_weapon_surface_condition(sampled_albedo, input);
    let albedo = sampled_albedo * material.tint.rgb;
    let condition_mask = weapon_surface_condition_mask(input);
    semantic_wear_mask = max(semantic_wear_mask, condition_mask);
    let mapped = mapped_normal(input);
    var normal = normalize(mix(
        mapped,
        normalize(input.view_normal),
        condition_mask * material.wear_scratches_projection.w,
    ));
    if investment_decal && material.decal_params.x > 2.5 {
        normal = investment_scene_normal_value(input);
    }
    let light = normalize(scene.light_position.xyz - input.view_position);
    let spotlight = spotlight_factor(input.view_position);
    let view_direction = vec3<f32>(0.0, 0.0, 1.0);
    let surface = material_surface(input, albedo);
    let roughness = clamp(
        surface.x
            + (1.0 - surface.x) * material.wear_surface_params.w * condition_mask,
        0.02,
        1.0,
    );
    let metalness = surface.y;
    let f0 = mix(vec3<f32>(0.03), albedo, metalness);
    let key_roughness = clamp(
        roughness + (scene.light_parameters.x - 5.0) * 0.025,
        0.02,
        1.0,
    );
    let key_specular = ggx_specular(normal, light, view_direction, key_roughness, f0);
    let reflection_direction = reflect(-view_direction, normal);
    let sampled_environment = textureSample(environment_cubemap, environment_sampler, reflection_direction).rgb;
    let environment_luma = dot(sampled_environment, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Inventory surfaces retain the authored environment hue. In particular,
    // pale grazing-angle weapon trim reflects the blue preview studio; forcing
    // the cubemap to luminance turns those package-authored specular strips white.
    let environment = mix(vec3<f32>(environment_luma), sampled_environment, 0.82)
        * vec3<f32>(1.0, 1.0, 0.97);
    let n_dot_v = max(dot(normal, view_direction), 0.0);
    let environment_fresnel = fresnel_schlick(n_dot_v, f0);
    var vertex_ao = mix(1.0, input.ambient_occlusion, scene.postprocess1.w);
    if material.sampler_params.w > 0.0 {
        // Some opaque shared-atlas shaders author RT2.g directly. Preserve
        // that MRT value instead of replacing it with Quicktag geometry AO.
        vertex_ao = material.sampler_params.w;
    }
    if material.runner_layered_params.z > 0.5 {
        // 80A9A9D3 writes RT2.g as the mean of its independent t2 scalar AO
        // and the geometry/procedural occlusion term.
        let runner_ao_sample = textureSampleBias(
            procedural_or_response_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        );
        let runner_ao = select(
            runner_ao_sample.r,
            runner_ao_sample.g,
            material.runner_layered_params.w > 0.5,
        );
        vertex_ao = 0.5 * (vertex_ao + clamp(runner_ao, 0.0, 1.0));
    }
    let sun_visibility = mix(1.0, spotlight_shadow(input), scene.light_direction.w);
    let n_dot_l = max(dot(normal, light), 0.0);
    // Near-neutral 5400–5900 K source. Diffuse establishes the form; there is
    // deliberately no camera fill, rim term, or Fresnel added to final colour.
    let key_color = vec3<f32>(1.0, 0.91, 0.80);
    // Tiger's extracted base textures are authored for a deferred irradiance
    // working range. Calibrate diffuse only; do not resurrect the old scale on
    // specular, which was responsible for the luminous cream contours.
    let diffuse_working_scale = 10.0;
    let direct_diffuse = albedo * (1.0 - metalness) * key_color
        * scene.postprocess0.w * n_dot_l * sun_visibility * spotlight * diffuse_working_scale;
    let up_factor = normal.y * 0.5 + 0.5;
    let hemi_irradiance = mix(
        vec3<f32>(0.075, 0.078, 0.082),
        vec3<f32>(0.19, 0.185, 0.175),
        up_factor,
    ) * scene.postprocess4.z;
    let unoccluded_indirect_diffuse = albedo * (1.0 - metalness) * hemi_irradiance
        * diffuse_working_scale;
    let indirect_diffuse = unoccluded_indirect_diffuse * vertex_ao;

    let grazing = grazing_boost(n_dot_v, roughness);

    let direct_specular = clamp_specular_luminance(
        key_specular
            * key_color
            * scene.postprocess0.w
            * n_dot_l
            * sun_visibility
            * spotlight
            * 0.24
            * grazing,
        0.22,
    );

    let specular_occlusion = clamp(
        pow(n_dot_v + vertex_ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + vertex_ao,
        0.0,
        1.0,
    );

    let indirect_specular = clamp_specular_luminance(
        environment
            * environment_fresnel
            * (1.0 - roughness * 0.72)
            * scene.postprocess4.w
            * specular_occlusion
            * grazing,
        0.14,
    );

    var emissive_sample = vec3<f32>(0.0);
    var emission_intensity = 0.0;
    var emissive_output = vec3<f32>(0.0);
    if material.params.y > 0.5 {
        emissive_sample = textureSampleBias(
            emissive_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).rgb;
        emission_intensity = select(1.0, material.params.z, material.params.z > 0.0);
        emissive_output = emissive_sample * emission_intensity;
    }

    let diagnostic_mode = u32(scene.postprocess2.x + 0.5);
    var color = direct_diffuse + indirect_diffuse + direct_specular + indirect_specular;
    if diagnostic_mode == 1u {
        color = albedo;
    } else if diagnostic_mode == 2u {
        color = direct_diffuse + unoccluded_indirect_diffuse;
    } else if diagnostic_mode == 3u {
        color = vec3<f32>(vertex_ao);
    } else if diagnostic_mode == 4u {
        color = direct_specular + indirect_specular;
    } else if diagnostic_mode == 6u {
        color = normal * 0.5 + vec3<f32>(0.5);
    } else if diagnostic_mode == 9u {
        color = semantic_dye;
    } else if diagnostic_mode == 10u {
        color = semantic_worn_dye;
    } else if diagnostic_mode == 11u {
        color = semantic_dye_detail;
    } else if diagnostic_mode == 12u {
        color = vec3<f32>(roughness);
    } else if diagnostic_mode == 13u {
        color = vec3<f32>(1.0 - roughness);
    } else if diagnostic_mode == 14u {
        color = vec3<f32>(emission_intensity);
    } else if diagnostic_mode == 15u {
        color = vec3<f32>(1.0 - material_alpha * material.tint.a);
    } else if diagnostic_mode == 16u {
        color = vec3<f32>(metalness);
    } else if diagnostic_mode == 17u {
        color = vec3<f32>(select(0.0, 1.0, material.transmission_params.x > 0.5));
    } else if diagnostic_mode == 18u {
        // No decoded iridescence payload means ID 0, never a guessed value.
        color = vec3<f32>(material.solid_surface.w);
    } else if diagnostic_mode == 19u {
        color = semantic_dye_mask;
    } else if diagnostic_mode == 20u {
        color = vec3<f32>(semantic_wear_mask);
    }
    if material.params.y > 0.5 {
        if diagnostic_mode == 0u || diagnostic_mode == 5u {
            color += emissive_output;
        }
    }
    let raw_channel_view = diagnostic_mode == 1u
        || diagnostic_mode == 3u
        || diagnostic_mode == 6u
        || diagnostic_mode >= 7u;
    // 0.6-style environment approximation for the asset viewer. Keep it in
    // linear space so translucent/decal state remains authored.
    let scene_radius = max(scene.params0.x, 0.0001);
    let fog_distance = length(input.view_position) / scene_radius;
    let normalized_height = input.world_relative.z / scene_radius;
    let height_density = exp(-max(normalized_height + 0.18, 0.0) * 2.4);
    let fog_amount = 1.0 - exp(-scene.postprocess0.z * fog_distance * height_density * 5.5);
    let day_factor = clamp(scene.postprocess1.z * 0.5 + 0.5, 0.0, 1.0);
    let horizon_color = mix(vec3<f32>(0.018, 0.025, 0.055), vec3<f32>(0.38, 0.52, 0.72), day_factor);
    let zenith_color = mix(vec3<f32>(0.008, 0.012, 0.032), vec3<f32>(0.12, 0.25, 0.48), day_factor);
    let height_mix = clamp(normalized_height * 0.7 + 0.35, 0.0, 1.0);
    let view_ray = normalize(vec3<f32>(-input.view_position.xy / scene_radius * 0.18, 1.0));
    let phase_g = 0.55;
    let phase_cos = clamp(dot(view_ray, light), -1.0, 1.0);
    let phase = (1.0 - phase_g * phase_g)
        / (12.5663706 * pow(max(1.0 + phase_g * phase_g - 2.0 * phase_g * phase_cos, 0.001), 1.5));
    let daylight = clamp(scene.postprocess1.z * 0.5 + 0.5, 0.0, 1.0);
    let sun_scatter = vec3<f32>(1.0, 0.58, 0.28) * phase * 2.8 * daylight;
    let fog_color = mix(horizon_color, zenith_color, height_mix) + sun_scatter;
    if !raw_channel_view {
        color = mix(color, fog_color, clamp(fog_amount, 0.0, 0.96));
    }
    if !raw_channel_view {
        color *= scene.postprocess0.x;
    }
    if diagnostic_mode == 7u {
        color = emissive_sample;
    } else if diagnostic_mode == 8u {
        let investment_decal = material.decal_params.x > 0.5;
        let mask_material = material.params.w < -0.5;
        color = vec3<f32>(
            select(0.0, 1.0, !investment_decal),
            select(0.0, 1.0, investment_decal),
            select(0.0, 1.0, mask_material),
        );
    }
    let output_alpha = material_alpha * material.tint.a;
    var compatibility_hdr = vec4<f32>(color, output_alpha);
    if decal_blend {
        // Tiger decal blend is One + Dst*SrcAlpha. Invert alpha and premultiply
        // final-color output to reproduce standard source-over compositing.
        compatibility_hdr = vec4<f32>(color * output_alpha, 1.0 - output_alpha);
    } else if material.blend.x == 8.0 {
        compatibility_hdr = vec4<f32>(color * output_alpha, output_alpha);
    } else {
        let opaque = material.blend.x == 0.0
            || material.blend.x == 1.0
            || material.blend.x == 57.0;
        compatibility_hdr = vec4<f32>(color, select(output_alpha, 1.0, opaque));
    }
    var output: FragmentOutput;
    output.compatibility_hdr = compatibility_hdr;
    // Authored stage-2 decal PS outputs alpha zero for RT1. Decal pipelines
    // mask RT1 alpha, preserving opaque roughness while writing resolved
    // scene or mesh normal RGB.
    output.normal_roughness = vec4<f32>(
        normal * 0.5 + vec3<f32>(0.5),
        select(roughness, 0.0, investment_decal),
    );
    output.material_properties = vec4<f32>(
        metalness,
        vertex_ao,
        select(0.0, 1.0, material.transmission_params.x > 0.5),
        material.solid_surface.w,
    );
    output.albedo = vec4<f32>(albedo, output_alpha);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    return shade_model(input, false);
}

@fragment
fn fs_investment_decal(input: VertexOutput) -> FragmentOutput {
    return shade_model(input, true);
}

@fragment
fn fs_forward_transparent(input: VertexOutput) -> @location(0) vec4<f32> {
    if material.transmission_params.y > 0.0 {
        // Refractive glass. The source blends `opacity` of
        // `tint * gain * scene` over the scene; that is the scene times this
        // transmittance. The shell is thin, so the near tint applies. The
        // source's blur, world lighting and fog are not reproduced.
        let opacity = material.transmission_params.y;
        let tinted = material.transmission_colors[0].rgb * material.transmission_params.z;
        return vec4<f32>(vec3<f32>(1.0 - opacity) + opacity * tinted, 1.0);
    }
    return shade_model(input, false).compatibility_hdr;
}

@fragment
fn fs_forward_coating(input: VertexOutput) -> @location(0) vec4<f32> {
    // Dedicated stage-8 coating ABI. Deferred t0 is scene depth. The compiled
    // shader linearizes it and subtracts the coating fragment's view depth;
    // c24 remaps that shell-to-surface separation into the c22/c23 colour mix.
    // t1 is linear object-space detail, and coverage is the authored TFX object
    // channel rather than either texture.
    let normal = normalize(input.view_normal);
    let view_direction = vec3<f32>(0.0, 0.0, 1.0);
    let raw_scene_depth = textureLoad(
        coating_scene_depth,
        vec2<i32>(input.clip_position.xy),
        0,
    );
    let scene_view_depth = (0.5 - raw_scene_depth)
        * max(scene.params0.x, 0.0001) / 0.21;
    // Quicktag's orthographic view z grows toward the camera; Tiger's decoded
    // positive view depth grows away from it. Reverse the subtraction while
    // preserving the same shell-to-surface distance.
    let depth_separation = input.view_position.z - scene_view_depth;
    let depth_incidence = clamp(
        material.coating_params0.y * depth_separation + material.coating_params0.z,
        0.0,
        1.0,
    );
    let incidence = depth_incidence;
    let authored_base_color = mix(
        material.coating_colors[0].rgb,
        material.coating_colors[1].rgb,
        incidence,
    );
    let base_color = select(
        authored_base_color / vec3<f32>(12.92),
        pow(
            (authored_base_color + vec3<f32>(0.055)) / vec3<f32>(1.055),
            vec3<f32>(2.4),
        ),
        authored_base_color > vec3<f32>(0.04045),
    );
    let detail_sample = runner_triplanar_scalar(
        color_texture,
        input.procedural_position * material.sampler_params.y,
        input.procedural_normal,
        material.coating_projection,
        material.coating_params0.x,
    );
    let detail = clamp(
        material.coating_params1.x + material.coating_params1.y * detail_sample,
        0.0,
        1.0,
    );
    let response = clamp(
        material.coating_params1.z + material.coating_params1.w * detail,
        0.0,
        1.0,
    );

    let light = normalize(scene.light_position.xyz - input.view_position);
    let spotlight = spotlight_factor(input.view_position);
    let n_dot_l = max(dot(normal, light), 0.0);
    let sun_visibility = mix(1.0, spotlight_shadow(input), scene.light_direction.w);
    // The coating PS is a narrow light-responsive lobe, not Lambert diffuse.
    // Its package response is sharply angular: preserve the lit face while
    // preventing an oblique inset face from receiving comparable irradiance.
    let coating_key_response = n_dot_l * smoothstep(0.40, 0.60, n_dot_l);
    // The generated coating shader has a narrow visibility response after its
    // player-centred lighting-grid lookup. Preserve that separation with the
    // renderer's stable PCF visibility: partially occluded rail faces must not
    // receive nearly the same ambient/key energy as their exposed neighbour.
    let coating_shadow_grid = smoothstep(0.74, 0.81, sun_visibility);
    let coating_shadow_sensitivity = smoothstep(0.68, 0.695, n_dot_l)
        * (1.0 - smoothstep(0.71, 0.725, n_dot_l));
    let coating_shadow_response = mix(
        1.0,
        coating_shadow_grid,
        coating_shadow_sensitivity,
    );
    let coating_visibility = mix(0.08, 1.0, coating_shadow_response);
    // Ambient environment light is independent of the spotlight's source,
    // cone, and shadow map. Only the key lobe follows source visibility.
    let ambient_illumination = 0.18 * scene.postprocess4.z;
    let key_illumination = coating_key_response
        * sun_visibility
        * spotlight
        * coating_visibility
        * scene.postprocess0.w
        * 0.82;
    let coating_key_color = mix(
        vec3<f32>(0.67, 1.14, 0.94),
        vec3<f32>(0.66, 1.05, 2.30),
        coating_shadow_sensitivity,
    );
    let base_illumination = ambient_illumination + key_illumination;
    let lit_base = base_color
        * (vec3<f32>(ambient_illumination) + coating_key_color * key_illumination);

    let reflection_direction = reflect(-view_direction, normal);
    // PS c31 chooses a lower mip floor from the detail response. The audited
    // coating permutations currently author c29=c30=0, so implicit derivative
    // LOD is the exact max(calculatedLOD, 0) operation used by the DXIL.
    let authored_lod_floor = mix(
        material.coating_environment_extra.y,
        material.coating_environment_extra.z,
        response,
    );
    let environment_sample = textureSampleBias(
        coating_environment_texture,
        coating_environment_sampler,
        reflection_direction,
        max(authored_lod_floor, 0.0),
    ).rgb;
    let environment_detail = material.coating_environment_params.x
        + material.coating_environment_params.y * detail;
    let environment_base = lit_base * material.coating_environment_params.w
        + vec3<f32>(material.coating_environment_extra.x);
    let environment = environment_sample
        * environment_detail
        * material.coating_environment_params.z
        * environment_base;

    // c39/c43 bend the interpolated bitangent toward the normal. The compiled
    // shader evaluates two coloured grazing lobes against the view vector; the
    // constants are not conventional roughness or metalness.
    let tangent = normalize(input.view_tangent.xyz);
    let bitangent = normalize(cross(normal, tangent) * input.view_tangent.w);
    let lobe_axis0 = normalize(
        bitangent + normal * material.coating_specular_params[1].z,
    );
    let lobe_axis1 = normalize(
        bitangent + normal * material.coating_specular_params[1].w,
    );
    let lobe_basis0 = sqrt(max(1.0 - pow(dot(lobe_axis0, view_direction), 2.0), 0.0));
    let lobe_basis1 = sqrt(max(1.0 - pow(dot(lobe_axis1, view_direction), 2.0), 0.0));
    let lobe0 = clamp(
        material.coating_specular_colors[0].rgb
            * pow(lobe_basis0, material.coating_specular_params[0].x)
            * material.coating_specular_params[0].y,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let lobe1 = clamp(
        material.coating_specular_colors[1].rgb
            * pow(lobe_basis1, material.coating_specular_params[1].x)
            * material.coating_specular_params[1].y,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let surface_color = lit_base + environment + lobe0 + lobe1;
    let coverage = clamp(material.coating_params0.w, 0.0, 1.0);

    let diagnostic_mode = u32(scene.postprocess2.x + 0.5);
    if diagnostic_mode == 1u || diagnostic_mode == 21u {
        return vec4<f32>(select(base_color, material.coating_colors[0].rgb, diagnostic_mode == 21u), 1.0);
    }
    if diagnostic_mode == 4u {
        return vec4<f32>(lobe0 + lobe1 + environment, 1.0);
    }
    if diagnostic_mode == 6u {
        return vec4<f32>(normal * 0.5 + vec3<f32>(0.5), 1.0);
    }
    if diagnostic_mode == 12u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if diagnostic_mode == 13u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if diagnostic_mode == 15u {
        return vec4<f32>(vec3<f32>(1.0 - coverage), 1.0);
    }
    if diagnostic_mode == 16u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if diagnostic_mode == 17u {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    if diagnostic_mode == 22u {
        return vec4<f32>(material.coating_colors[1].rgb, 1.0);
    }
    if diagnostic_mode == 23u {
        return vec4<f32>(vec3<f32>(incidence), 1.0);
    }
    if diagnostic_mode == 24u {
        return vec4<f32>(vec3<f32>(coverage), 1.0);
    }
    if diagnostic_mode == 25u {
        return vec4<f32>(vec3<f32>(response), 1.0);
    }
    if diagnostic_mode == 26u {
        return vec4<f32>(lobe0, 1.0);
    }
    if diagnostic_mode == 27u {
        return vec4<f32>(lobe1, 1.0);
    }
    if diagnostic_mode == 28u {
        return vec4<f32>(environment, 1.0);
    }
    if diagnostic_mode == 29u {
        return vec4<f32>(surface_color * coverage, coverage);
    }
    if diagnostic_mode == 30u {
        return vec4<f32>(vec3<f32>(n_dot_l), 1.0);
    }
    if diagnostic_mode == 31u {
        return vec4<f32>(vec3<f32>(base_illumination), 1.0);
    }
    if diagnostic_mode == 32u {
        return vec4<f32>(vec3<f32>(sun_visibility), 1.0);
    }
    if diagnostic_mode == 33u {
        return vec4<f32>(surface_color, 1.0);
    }
    if diagnostic_mode == 34u {
        return vec4<f32>(lit_base, 1.0);
    }
    if diagnostic_mode == 35u {
        return vec4<f32>(vec3<f32>(0.5 + depth_separation * 8.0), 1.0);
    }
    if diagnostic_mode >= 7u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(surface_color * coverage, coverage);
}

@fragment
fn fs_distortion(input: VertexOutput) -> @location(0) vec4<f32> {
    // Stage 8 writes a premultiplied material payload. Full-resolution depth
    // keeps relief ownership deterministic in the interactive preview.
    let distortion_map = textureSampleBias(
        color_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    );
    let normal = normalize(input.view_normal);
    let view_direction = vec3<f32>(0.0, 0.0, 1.0);
    let fresnel = pow(1.0 - max(dot(normal, view_direction), 0.0), 5.0);
    let authored_signal = dot(distortion_map.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let authored_color = mix(
        material.transmission_colors[0].rgb,
        material.transmission_colors[1].rgb,
        authored_signal,
    );
    let authored_surface = mix(
        material.transmission_surfaces[0],
        material.transmission_surfaces[1],
        authored_signal,
    );
    let diagnostic_mode = u32(scene.postprocess2.x + 0.5);
    if diagnostic_mode == 1u {
        return vec4<f32>(authored_color, 1.0);
    }
    if diagnostic_mode == 12u {
        return vec4<f32>(vec3<f32>(authored_surface.x), 1.0);
    }
    if diagnostic_mode == 13u {
        return vec4<f32>(vec3<f32>(1.0 - authored_surface.x), 1.0);
    }
    if diagnostic_mode == 15u {
        return vec4<f32>(vec3<f32>(1.0 - authored_signal), 1.0);
    }
    if diagnostic_mode == 16u {
        return vec4<f32>(vec3<f32>(authored_surface.y), 1.0);
    }
    if diagnostic_mode == 17u {
        return vec4<f32>(vec3<f32>(authored_signal), 1.0);
    }
    if diagnostic_mode >= 9u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    var base_color = select(
        vec3<f32>(0.055, 0.060, 0.070),
        authored_color,
        material.transmission_params.x > 0.5,
    );
    let light = normalize(scene.light_position.xyz - input.view_position);
    let spotlight = spotlight_factor(input.view_position);
    let n_dot_l = max(dot(normal, light), 0.0);
    let half_vector = normalize(light + view_direction);
    if material.transmission_params.x > 0.5 {
        // This material block is an opaque coated surface: its neighbouring
        // constants are roughness and metalness, not transmission opacity.
        // Stage 8 chooses the render target/resolve path; it does not make the
        // surface conventionally transparent. Alpha here is raster coverage.
        let roughness = clamp(authored_surface.x, 0.02, 1.0);
        let metalness = clamp(authored_surface.y, 0.0, 1.0);
        let specular_power = mix(128.0, 4.0, roughness);
        let f0 = mix(vec3<f32>(0.04), base_color, metalness);
        let direct_diffuse =
            base_color * (1.0 - metalness) * (0.18 + n_dot_l * 0.72 * spotlight);
        let direct_specular = f0
            * pow(max(dot(normal, half_vector), 0.0), specular_power)
            * mix(1.0, 0.28, roughness)
            * spotlight;
        let grazing_specular = f0 * fresnel * mix(0.42, 0.12, roughness);
        let surface_color = direct_diffuse + direct_specular + grazing_specular;
        let coverage = clamp(authored_signal, 0.0, 1.0);
        return vec4<f32>(surface_color * coverage, coverage);
    }

    // Unclassified stage-8 effects retain map-driven coverage until their
    // own shader ABI proves opaque-surface semantics.
    let coverage = authored_signal * 0.45;
    let surface_color = base_color
        * (0.055 + n_dot_l * 0.105 * spotlight + fresnel * 0.025 * spotlight);
    return vec4<f32>(surface_color * coverage, coverage);
}

@fragment
fn fs_material_emissive(input: VertexOutput) -> @location(0) vec4<f32> {
    if material.params.y > 0.5 {
        let intensity = select(1.0, material.params.z, material.params.z > 0.0);
        return vec4<f32>(textureSampleBias(
            emissive_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).rgb * intensity, intensity);
    }
    return vec4<f32>(0.0);
}

struct AuxiliaryOutput {
    @location(0) flags: u32,
}

@fragment
fn fs_material_flags(input: VertexOutput) -> AuxiliaryOutput {
    let investment_decal = material.decal_params.x > 0.5;
    let mask_material = material.params.w < -0.5;
    var output: AuxiliaryOutput;
    output.flags = select(1u, 2u, investment_decal)
        | select(0u, 4u, mask_material);
    return output;
}
"#;

const PRESENT_SHADER: &str = r#"
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> scene: SceneUniform;
@group(0) @binding(3) var decal_texture: texture_2d<f32>;
@group(0) @binding(4) var scene_depth: texture_depth_2d;
@group(0) @binding(5) var bloom_half: texture_2d<f32>;
@group(0) @binding(6) var bloom_quarter: texture_2d<f32>;

struct SceneUniform {
    center: vec4<f32>, params0: vec4<f32>, params1: vec4<f32>, uv_transform: vec4<f32>,
    light_direction: vec4<f32>, light_parameters: vec4<f32>, light_position: vec4<f32>, postprocess0: vec4<f32>, postprocess1: vec4<f32>, postprocess2: vec4<f32>, postprocess3: vec4<f32>, postprocess4: vec4<f32>, postprocess5: vec4<f32>, fidelity: vec4<f32>, shadow_parameters: vec4<f32>,
}

fn reconstruct_view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let scale = 0.84 * scene.params0.w / max(scene.params0.x, 0.0001);
    return vec3<f32>(
        -(clip.x - scene.params1.y) / max(scale * scene.params1.x, 0.0001),
        (clip.y - scene.params1.z) / max(scale, 0.0001),
        (0.5 - depth) * max(scene.params0.x, 0.0001) / 0.21,
    );
}

fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    let numerator = color * (2.51 * color + vec3<f32>(0.03));
    let denominator = color * (2.43 * color + vec3<f32>(0.59)) + vec3<f32>(0.14);
    return clamp(numerator / denominator, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn filmic_grade(color: vec3<f32>) -> vec3<f32> {
    // Lift before the curve so small cavities can still approach black. Blend
    // a gentler Reinhard response into ACES, then compress contrast globally.
    let lifted = max(color, vec3<f32>(0.0)) + vec3<f32>(0.004);
    let reinhard = lifted / (vec3<f32>(1.0) + lifted);
    var graded = mix(aces_fitted(lifted), reinhard, 0.25) * 0.90;
    let luma_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luma = dot(graded, luma_weights);
    // Neutral grade: stronger shadow desaturation, mild global/highlight
    // desaturation, no warm-highlight/cool-shadow split.
    let saturation = mix(0.76, 0.93, smoothstep(0.10, 0.48, luma));
    graded = vec3<f32>(luma) + (graded - vec3<f32>(luma)) * saturation;
    return clamp(graded, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, color <= vec3<f32>(0.0031308));
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diagnostic_mode = u32(scene.postprocess2.x + 0.5);
    if diagnostic_mode != 0u {
        // Semantic views are data inspection, not beauty renders. Present the
        // composed channel literally: no FXAA, bloom, SSAO, scene distortion,
        // decals, tone curve, grading, exposure, or display transfer.
        let depth_size = vec2<i32>(textureDimensions(scene_depth));
        let pixel = clamp(
            vec2<i32>(input.uv * vec2<f32>(depth_size)),
            vec2<i32>(0),
            depth_size - vec2<i32>(1),
        );
        let covered = textureLoad(scene_depth, pixel, 0) < 0.9999;
        let export_transparent = scene.postprocess4.y < -0.5 || scene.postprocess4.y > 1.5;
        let alpha = select(1.0, select(0.0, 1.0, covered), export_transparent);
        return vec4<f32>(textureSample(source_texture, source_sampler, input.uv).rgb, alpha);
    }
    let centered = input.uv - vec2<f32>(0.5);
    let radius = length(centered);
    let distortion_envelope = smoothstep(0.72, 0.05, radius);
    let distorted_uv = input.uv + normalize(centered + vec2<f32>(0.00001))
        * sin(radius * 42.0) * scene.postprocess2.y * distortion_envelope * 0.012;
    var color = textureSample(source_texture, source_sampler, distorted_uv).rgb;
    let texel = 1.0 / max(scene.postprocess1.xy, vec2<f32>(1.0));
    if scene.postprocess2.z > 0.5 {
        let luma_weights = vec3<f32>(0.299, 0.587, 0.114);
        let luma_m = dot(color, luma_weights);
        let luma_nw = dot(textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(-texel.x, -texel.y)).rgb, luma_weights);
        let luma_ne = dot(textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>( texel.x, -texel.y)).rgb, luma_weights);
        let luma_sw = dot(textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(-texel.x,  texel.y)).rgb, luma_weights);
        let luma_se = dot(textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>( texel.x,  texel.y)).rgb, luma_weights);
        let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
        let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
        var direction = vec2<f32>(
            -((luma_nw + luma_ne) - (luma_sw + luma_se)),
             ((luma_nw + luma_sw) - (luma_ne + luma_se)),
        );
        let reduction = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 0.0078125);
        direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + reduction), vec2<f32>(-8.0), vec2<f32>(8.0)) * texel;
        let rgb_a = 0.5 * (
            textureSample(source_texture, source_sampler, distorted_uv + direction * (1.0 / 3.0 - 0.5)).rgb
            + textureSample(source_texture, source_sampler, distorted_uv + direction * (2.0 / 3.0 - 0.5)).rgb
        );
        let rgb_b = rgb_a * 0.5 + 0.25 * (
            textureSample(source_texture, source_sampler, distorted_uv + direction * -0.5).rgb
            + textureSample(source_texture, source_sampler, distorted_uv + direction * 0.5).rgb
        );
        let luma_b = dot(rgb_b, luma_weights);
        color = select(rgb_b, rgb_a, luma_b < luma_min || luma_b > luma_max);
        // Approximate the reference's non-sharpened temporal/downsample resolve
        // with a normalized 3x3 Gaussian at roughly 0.5 output-pixel radius.
        // color = color * 0.620
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>( texel.x, 0.0)).rgb * 0.084
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(-texel.x, 0.0)).rgb * 0.084
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(0.0,  texel.y)).rgb * 0.084
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(0.0, -texel.y)).rgb * 0.084
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>( texel.x,  texel.y)).rgb * 0.011
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(-texel.x,  texel.y)).rgb * 0.011
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>( texel.x, -texel.y)).rgb * 0.011
        //     + textureSample(source_texture, source_sampler, distorted_uv + vec2<f32>(-texel.x, -texel.y)).rgb * 0.011;
    }
    if scene.postprocess2.w > 0.001 {
        let depth_size = vec2<i32>(textureDimensions(scene_depth));
        let pixel = clamp(vec2<i32>(distorted_uv * vec2<f32>(depth_size)), vec2<i32>(0), depth_size - vec2<i32>(1));
        let center_depth = textureLoad(scene_depth, pixel, 0);
        if center_depth < 0.9999 {
            let center_position = reconstruct_view_position(distorted_uv, center_depth);
            let has_right = pixel.x + 1 < depth_size.x;
            let has_down = pixel.y + 1 < depth_size.y;
            let right_pixel = select(max(pixel - vec2<i32>(1, 0), vec2<i32>(0)), pixel + vec2<i32>(1, 0), has_right);
            let down_pixel = select(max(pixel - vec2<i32>(0, 1), vec2<i32>(0)), pixel + vec2<i32>(0, 1), has_down);
            let right_uv = (vec2<f32>(right_pixel) + vec2<f32>(0.5)) / vec2<f32>(depth_size);
            let down_uv = (vec2<f32>(down_pixel) + vec2<f32>(0.5)) / vec2<f32>(depth_size);
            let right_position = reconstruct_view_position(right_uv, textureLoad(scene_depth, right_pixel, 0));
            let down_position = reconstruct_view_position(down_uv, textureLoad(scene_depth, down_pixel, 0));
            let tangent_x = select(center_position - right_position, right_position - center_position, has_right);
            let tangent_y = select(center_position - down_position, down_position - center_position, has_down);
            let surface_normal = normalize(cross(tangent_x, tangent_y));
            let offsets = array<vec2<i32>, 16>(
                vec2<i32>(-2, 0), vec2<i32>(2, 0), vec2<i32>(0, -2), vec2<i32>(0, 2),
                vec2<i32>(-2, -2), vec2<i32>(2, -2), vec2<i32>(-2, 2), vec2<i32>(2, 2),
                vec2<i32>(-18, 0), vec2<i32>(18, 0), vec2<i32>(0, -18), vec2<i32>(0, 18),
                vec2<i32>(-20, -12), vec2<i32>(20, -12), vec2<i32>(-20, 12), vec2<i32>(20, 12),
            );
            var fine_occlusion = 0.0;
            var fine_weight = 0.0;
            var structural_occlusion = 0.0;
            var structural_weight = 0.0;
            for (var index = 0; index < 16; index++) {
                let sample_pixel = clamp(pixel + offsets[index], vec2<i32>(0), depth_size - vec2<i32>(1));
                let sample_depth = textureLoad(scene_depth, sample_pixel, 0);
                if sample_depth < 0.9999 {
                    let sample_uv = (vec2<f32>(sample_pixel) + vec2<f32>(0.5)) / vec2<f32>(depth_size);
                    let sample_position = reconstruct_view_position(sample_uv, sample_depth);
                    let delta = sample_position - center_position;
                    let distance = length(delta);
                    if index < 8 {
                        let range_weight = 1.0 - smoothstep(scene.params0.x * 0.005, scene.params0.x * 0.055, distance);
                        let hemisphere = max(dot(delta / max(distance, 0.0001), surface_normal) - 0.05, 0.0);
                        fine_occlusion += hemisphere * range_weight;
                        fine_weight += range_weight;
                    } else {
                        let range_weight = 1.0 - smoothstep(scene.params0.x * 0.025, scene.params0.x * 0.22, distance);
                        let hemisphere = max(dot(delta / max(distance, 0.0001), surface_normal) - 0.12, 0.0);
                        structural_occlusion += hemisphere * range_weight;
                        structural_weight += range_weight;
                    }
                }
            }
            let fine = fine_occlusion / max(fine_weight, 1.0);
            let structural = structural_occlusion / max(structural_weight, 1.0);
            let ao = clamp(1.0 - (fine * 1.25 + structural * 0.35) * scene.postprocess2.w, 0.45, 1.0);
            let diagnostic_mode = u32(scene.postprocess2.x + 0.5);
            if diagnostic_mode == 3u {
                color *= ao;
            } else if diagnostic_mode == 0u || diagnostic_mode == 5u {
                // Screen AO is applied after forward composition, so keep its
                // unavoidable direct-light influence diagnostic-only small.
                color *= mix(1.0, ao, 0.15);
            }
        }
    }
    let bloom_near = textureSample(bloom_half, source_sampler, distorted_uv).rgb;
    let bloom_wide = textureSample(bloom_quarter, source_sampler, distorted_uv).rgb;
    color += (bloom_near * 0.62 + bloom_wide * 1.18) * scene.postprocess0.y;

    let water_level = clamp(scene.postprocess3.y, 0.05, 0.95);
    if distorted_uv.y > water_level && scene.postprocess3.x > 0.001 {
        let water_depth = (distorted_uv.y - water_level) / max(1.0 - water_level, 0.001);
        let ripple = sin(distorted_uv.x * 95.0 + water_depth * 22.0) * 0.004
            + sin(distorted_uv.x * 41.0 - water_depth * 37.0) * 0.003;
        let reflected_uv = vec2<f32>(
            clamp(distorted_uv.x + ripple * scene.postprocess3.x, 0.0, 1.0),
            clamp(water_level - (distorted_uv.y - water_level) * 0.72, 0.0, 1.0),
        );
        let reflection = textureSample(source_texture, source_sampler, reflected_uv).rgb;
        let fresnel = pow(clamp(water_depth, 0.0, 1.0), 2.0);
        let water_color = mix(vec3<f32>(0.02, 0.11, 0.16), reflection, 0.58 + fresnel * 0.32);
        color = mix(color, water_color, scene.postprocess3.x * (0.48 + fresnel * 0.42));
    }

    let decal_uv = (distorted_uv - vec2<f32>(0.5)) / vec2<f32>(0.42, 0.42) + vec2<f32>(0.5);
    if all(decal_uv >= vec2<f32>(0.0)) && all(decal_uv <= vec2<f32>(1.0)) {
        let decal = textureSample(decal_texture, source_sampler, decal_uv);
        color = mix(color, decal.rgb, decal.a * scene.postprocess3.z);
    }
    let road_uv = vec2<f32>(
        (distorted_uv.x - 0.5) / max(0.18 + distorted_uv.y * 0.42, 0.01) + 0.5,
        (distorted_uv.y - 0.55) / 0.42,
    );
    if all(road_uv >= vec2<f32>(0.0)) && all(road_uv <= vec2<f32>(1.0)) {
        let road = textureSample(decal_texture, source_sampler, road_uv);
        let road_fade = smoothstep(0.55, 0.82, distorted_uv.y);
        color = mix(color, road.rgb, road.a * road_fade * scene.postprocess3.w);
    }
    let depth_size = vec2<i32>(textureDimensions(scene_depth));
    let final_pixel = clamp(
        vec2<i32>(distorted_uv * vec2<f32>(depth_size)),
        vec2<i32>(0),
        depth_size - vec2<i32>(1),
    );
    // egui presents into a non-sRGB swapchain on Windows. Keep viewer
    // background in UI color space, but run rendered geometry through the
    // game's linear HDR -> filmic -> display transfer path.
    if textureLoad(scene_depth, final_pixel, 0) < 0.9999 {
        let diagnostic_mode = u32(scene.postprocess2.x + 0.5);
        if scene.postprocess4.y > 0.5
            && diagnostic_mode != 1u
            && diagnostic_mode != 3u
            && diagnostic_mode != 5u
            && diagnostic_mode != 6u
        {
            color = filmic_grade(max(color, vec3<f32>(0.0)));
        }
        let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
        color = vec3<f32>(luma) + (color - vec3<f32>(luma)) * scene.postprocess5.z;
        color = (color - vec3<f32>(0.5)) * scene.postprocess5.y + vec3<f32>(0.5);
        color *= scene.postprocess5.x;
        color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / max(scene.postprocess5.w, 0.05)));
    }
    if scene.postprocess4.x > 0.5 {
        color = linear_to_srgb(color);
    }
    let export_transparent = scene.postprocess4.y < -0.5 || scene.postprocess4.y > 1.5;
    let covered = textureLoad(scene_depth, final_pixel, 0) < 0.9999;
    return vec4<f32>(color, select(1.0, select(0.0, 1.0, covered), export_transparent));
}
"#;

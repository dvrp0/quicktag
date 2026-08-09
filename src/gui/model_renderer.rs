use std::{
    collections::HashMap,
    ops::Range,
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
        GearDyeMaterial, GearPatternMaterial, InvestmentDecalMaskMode, InvestmentDecalMaterial,
        InvestmentDecalMode, UvTransformPreview, WeaponModConditionMaterial,
        WireframeMaterialTextures, WireframePreview,
    },
    material::{TechniqueRenderState, is_sticker_proxy_technique, render_state_for_technique},
    render::{
        TigerDrawPacket,
        evidence::{
            EvidenceLevel, FidelityMode, ProvenanceId, ProvenanceRecord, ProvenanceStore,
            SourceSpan,
        },
        material::MaterialIR,
        pass_plan::{DrawPassPlan, RenderPassKind},
        technique::{TechniqueDescriptor, VertexAbiDescriptor},
    },
    texture::{
        Texture, TextureType,
        cache::{MaterialTextureKey, TextureCache},
        linear_texture_format, srgb_texture_format,
    },
};

const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// RGBA8 keeps five logical targets under WebGPU's portable 32-byte/sample cap.
// These are research/debug contracts; compatibility HDR remains RGBA16F.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SURFACE_PROPERTIES_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
const SURFACE_EMISSIVE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SURFACE_FLAGS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_MAP_SIZE: u32 = 4096;

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

#[derive(Clone)]
struct ModelDraw {
    indices: Range<u32>,
    packet: TigerDrawPacket,
    material: Option<MaterialTextureKey>,
    solid_color: Option<[f32; 4]>,
    solid_surface: Option<[f32; 2]>,
    control: Option<TagHash>,
    roughness_channel: u8,
    mask_palette: Option<[[f32; 4]; 2]>,
    gear_dye: Option<GearDyeMaterial>,
    gear_dye_default: Option<[f32; 4]>,
    gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    mod_wear: Option<WeaponModConditionMaterial>,
    gear_pattern: Option<GearPatternMaterial>,
    investment_decal: Option<InvestmentDecalMaterial>,
    authored_shared_atlas: bool,
    sampler: Option<TagHash>,
    sticker_proxy: bool,
    pipeline: ModelPipelineKey,
    center: [f32; 3],
    /// Uniform packed-position scale from Tiger's scope_skinning[5].w.
    procedural_scale: f32,
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

pub(crate) struct GpuModelPreview {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    draws: Vec<ModelDraw>,
    vertices: Vec<ModelVertex>,
    indices: Vec<u32>,
    vertex_abi: VertexAbiDescriptor,
    provenance: ProvenanceStore,
}

impl GpuModelPreview {
    pub(crate) fn inspection_lines(&self) -> Vec<String> {
        self.draws
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
                format!(
                    "#{index} lod={:?} stage={:?} tech={:?} family={:?} passes={:?} source={} tfx=[{}] warnings={:?}",
                    draw.packet.raw_lod_category,
                    draw.packet.raw_render_stage,
                    draw.packet.technique_hash,
                    draw.packet.material.family(),
                    draw.packet.pass_plan.passes,
                    source,
                    tfx,
                    draw.packet.pass_plan.warnings,
                )
            })
            .collect()
    }

    pub(crate) fn create(
        device: &wgpu::Device,
        wireframe: &WireframePreview,
        fallback_color: Option<tiger_pkg::TagHash>,
    ) -> Option<Self> {
        let uvs = wireframe.uvs.as_ref()?;
        if wireframe.vertices.is_empty() || wireframe.indices.len() < 3 {
            return None;
        }

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
        let vertices = wireframe
            .vertices
            .iter()
            .enumerate()
            .map(|(index, position)| ModelVertex {
                position: *position,
                normal: normals[index],
                uv: uvs.get(index).copied().unwrap_or_default(),
                tangent: tangents
                    .and_then(|tangents| tangents.get(index).copied())
                    .unwrap_or_default(),
                ambient_occlusion: ambient_occlusion[index],
                procedural_position: procedural_positions
                    .and_then(|positions| positions.get(index).copied())
                    .unwrap_or(*position),
                procedural_normal: procedural_normals
                    .and_then(|normals| normals.get(index).copied())
                    .unwrap_or(normals[index]),
            })
            .collect::<Vec<_>>();
        let mut draws = model_draws(wireframe, fallback_color);
        if draws.is_empty() {
            return None;
        }
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quicktag_model_preview_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quicktag_model_preview_indices"),
            contents: bytemuck::cast_slice(&wireframe.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Some(Self {
            vertex_buffer,
            index_buffer,
            draws,
            vertices,
            indices: wireframe.indices.clone(),
            vertex_abi: VertexAbiDescriptor::from_wireframe(wireframe),
            provenance,
        })
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
        let color = range.textures.color.or(range.texture);

        if start < end {
            let technique = range.technique.and_then(|tag| {
                technique_descriptors
                    .entry(tag)
                    .or_insert_with(|| TechniqueDescriptor::load(tag))
                    .clone()
            });
            let render_state = technique
                .as_ref()
                .map(|technique| technique.render_state)
                .unwrap_or_else(|| {
                    range
                        .technique
                        .map(render_state_for_technique)
                        .unwrap_or_default()
                });
            let material_ir = MaterialIR::classify(&range.textures);
            let pass_plan = DrawPassPlan::derive(range.render_stage, render_state, &material_ir);
            let pipeline = ModelPipelineKey::select_for_draw(
                render_state,
                material_ir.family(),
                &pass_plan,
                range.technique,
            );
            draws.push(ModelDraw {
                indices: start as u32..end as u32,
                packet: TigerDrawPacket {
                    indices: start as u32..end as u32,
                    raw_lod_category: range.raw_lod_category,
                    raw_render_stage: range.render_stage,
                    technique_hash: range.technique,
                    technique,
                    material: material_ir,
                    pass_plan,
                    source: ProvenanceId(u32::MAX),
                },
                material: color.map(|color| MaterialTextureKey {
                    color,
                    normal: range.textures.normal,
                    emissive: range.textures.emissive,
                    color_tint: range.textures.color_tint,
                    emissive_strength: range.textures.emissive_strength,
                }),
                solid_color: range.textures.solid_color,
                solid_surface: range.textures.solid_surface,
                control: range.textures.control,
                roughness_channel: range.textures.roughness_channel,
                mask_palette: range.textures.mask_palette,
                gear_dye: range.textures.gear_dye,
                gear_dye_default: range.textures.gear_dye_default,
                gear_dye_palette: range.textures.gear_dye_palette,
                mod_wear: range.textures.mod_wear,
                gear_pattern: range.textures.gear_pattern,
                investment_decal: range.textures.investment_decal,
                authored_shared_atlas: range.textures.authored_shared_atlas,
                sampler: range.textures.sampler,
                sticker_proxy: range.technique.is_some_and(is_sticker_proxy_technique),
                pipeline,
                center: draw_range_center(wireframe, start, end),
                procedural_scale: range.procedural_scale,
            });
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
            control: None,
            roughness_channel: 0,
            mask_palette: None,
            gear_dye: None,
            gear_dye_default: None,
            gear_dye_palette: None,
            mod_wear: None,
            gear_pattern: None,
            investment_decal: None,
            authored_shared_atlas: false,
            sampler: None,
            sticker_proxy: false,
            pipeline: ModelPipelineKey::default(),
            center: draw_range_center(wireframe, 0, index_len),
            procedural_scale: draw_range_procedural_scale(wireframe, 0, index_len),
        });
    }

    draws.sort_by_key(|draw| blend_enabled(draw.pipeline.blend));
    draws
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
    let mut sum = [0.0_f64; 3];
    let mut count = 0usize;
    for index in wireframe.indices
        [start.min(wireframe.indices.len())..end.min(wireframe.indices.len())]
        .iter()
        .step_by(3)
    {
        let Some(position) = wireframe.vertices.get(*index as usize) else {
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
        color,
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
    postprocess0: [f32; 4],
    postprocess1: [f32; 4],
    postprocess2: [f32; 4],
    postprocess3: [f32; 4],
    postprocess4: [f32; 4],
    postprocess5: [f32; 4],
    fidelity: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelEnvironment {
    pub fidelity_mode: FidelityMode,
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
    pub specular_ibl_intensity: f32,
    pub light_position: [f32; 3],
    pub light_orbit_radius: f32,
    pub light_size: f32,
    pub light_gizmo: bool,
    pub shadow_strength: f32,
    /// Normalized PCF radius: 0.0 is hard, 1.0 is the widest penumbra.
    pub shadow_softness: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gamma: f32,
    /// 0 final, 1 base colour, 2 diffuse, 3 AO, 4 specular, 5 pre-tone HDR,
    /// 6 mapped view normal.
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
            fidelity_mode: FidelityMode::PrettyPreview,
            lighting_model: LightingModel::TigerGgxApproximation,
            tfx_time_seconds: 0.0,
            tfx_paused: true,
            tfx_speed: 1.0,
            time_of_day: 0.35,
            sun_intensity: 0.35,
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
            ambient_intensity: 0.2,
            specular_ibl_intensity: 0.3,
            light_position: [0.061, 0.711, 0.701],
            light_orbit_radius: -1.0,
            light_size: 5.0,
            light_gizmo: false,
            shadow_strength: 1.0,
            shadow_softness: 0.5,
            brightness: 1.0,
            contrast: 1.01,
            saturation: 1.4,
            gamma: 1.0,
            diagnostic_pass: 0,
        }
    }
}

fn first_person_key_light(
    time_of_day: f32,
    position: [f32; 3],
    shadow_strength: f32,
) -> ([f32; 4], f32) {
    let sun_angle = time_of_day.rem_euclid(1.0) * std::f32::consts::TAU;
    let sun_height = (sun_angle - std::f32::consts::FRAC_PI_2).sin();
    let length = position
        .into_iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let direction = if length > 0.0001 {
        position.map(|value| value / length)
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ModelSamplerDesc {
    filter: u32,
    address_u: u32,
    address_v: u32,
    address_w: u32,
    mip_lod_bias: f32,
    max_anisotropy: u32,
    min_lod: f32,
    max_lod: f32,
}

struct LoadedMaterial {
    key: Option<MaterialTextureKey>,
    solid_color: Option<[f32; 4]>,
    solid_surface: Option<[f32; 2]>,
    blend: u8,
    control_tag: Option<TagHash>,
    roughness_channel: u8,
    mask_palette: Option<[[f32; 4]; 2]>,
    dye_palette: Option<[[f32; 4]; 3]>,
    gear_dye: Option<GearDyeMaterial>,
    gear_dye_default: Option<[f32; 4]>,
    gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    mod_wear: Option<WeaponModConditionMaterial>,
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
    procedural_scale: f32,
}

struct PreparedDraw {
    indices: Range<u32>,
    material_index: usize,
    pipeline: ModelPipelineKey,
    view_depth: f32,
    stable_index: usize,
    passes: Vec<RenderPassKind>,
}

pub(crate) struct ModelPaintCallback {
    preview: Arc<GpuModelPreview>,
    target_format: wgpu::TextureFormat,
    target_size: [u32; 2],
    scene: SceneUniform,
    materials: Vec<LoadedMaterial>,
    draws: Vec<PreparedDraw>,
    cubemap: Option<Arc<Texture>>,
    decal: Option<Arc<Texture>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ModelCameraFrame {
    pub(crate) center: [f32; 3],
    pub(crate) radius: f32,
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
        let target_size = bounded_target_size(
            rect.width() * pixels_per_point,
            rect.height() * pixels_per_point,
        );

        let mut materials = Vec::<LoadedMaterial>::new();
        let mut draws = Vec::with_capacity(preview.draws.len());
        for (stable_index, draw) in preview.draws.iter().enumerate() {
            if draw.sticker_proxy && !show_stickers {
                continue;
            }
            if draw.material.is_some_and(|material| {
                !draw.authored_shared_atlas
                    && is_debug_placeholder_texture(texture_cache, material.color)
            }) {
                continue;
            }
            let material_index = materials
                .iter()
                .position(|material| {
                    material.key == draw.material
                        && material.solid_color == draw.solid_color
                        && material.solid_surface == draw.solid_surface
                        && material.blend == draw.pipeline.blend
                        && material.control_tag == draw.control
                        && material.roughness_channel == draw.roughness_channel
                        && material.mask_palette == draw.mask_palette
                        && material.dye_palette == draw_dye_palette(draw)
                        && material.gear_dye == draw.gear_dye
                        && material.gear_dye_default == draw.gear_dye_default
                        && material.gear_dye_palette == draw.gear_dye_palette
                        && material.mod_wear == draw.mod_wear
                        && material.gear_pattern == draw.gear_pattern
                        && material.investment_decal == draw.investment_decal
                        && material.sampler_tag == draw.sampler
                        && material.procedural_scale == draw.procedural_scale
                })
                .unwrap_or_else(|| {
                    let color = draw
                        .material
                        .map(|material| texture_cache.get_or_default(material.color).0)
                        .and_then(usable_2d_texture);
                    let normal = draw
                        .material
                        .and_then(|material| material.normal)
                        .and_then(|tag| texture_cache.get_or_load(tag))
                        .map(|loaded| loaded.0)
                        .and_then(usable_2d_texture);
                    let emissive = draw
                        .material
                        .and_then(|material| material.emissive)
                        .and_then(|tag| texture_cache.get_or_load(tag))
                        .map(|loaded| loaded.0)
                        .and_then(usable_2d_texture);
                    let control = draw
                        .control
                        .and_then(|tag| texture_cache.get_or_load(tag))
                        .map(|loaded| loaded.0)
                        .and_then(usable_2d_texture);
                    let load_wear = |tag: TagHash| {
                        texture_cache
                            .get_or_load(tag)
                            .map(|loaded| loaded.0)
                            .and_then(usable_2d_texture)
                    };
                    let wear_scratches = draw.mod_wear.and_then(|wear| load_wear(wear.scratches));
                    let wear_grime = draw.mod_wear.and_then(|wear| load_wear(wear.grime));
                    let wear_damage = draw.mod_wear.and_then(|wear| load_wear(wear.damage));
                    let pattern_field = draw
                        .gear_pattern
                        .and_then(|pattern| load_wear(pattern.field));
                    let dye_palette = draw_dye_palette(draw);
                    materials.push(LoadedMaterial {
                        key: draw.material,
                        solid_color: draw.solid_color,
                        solid_surface: draw.solid_surface,
                        blend: draw.pipeline.blend,
                        control_tag: draw.control,
                        roughness_channel: draw.roughness_channel,
                        mask_palette: draw.mask_palette,
                        dye_palette,
                        gear_dye: draw.gear_dye,
                        gear_dye_default: draw.gear_dye_default,
                        gear_dye_palette: draw.gear_dye_palette,
                        mod_wear: draw.mod_wear,
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
                        procedural_scale: draw.procedural_scale,
                    });
                    materials.len() - 1
                });
            draws.push(PreparedDraw {
                indices: draw.indices.clone(),
                material_index,
                pipeline: draw.pipeline,
                view_depth: model_view_depth(draw.center, center, yaw, pitch),
                stable_index,
                passes: draw.packet.pass_plan.passes.clone(),
            });
        }
        draws.sort_by(|left, right| {
            let left_blended = blend_enabled(left.pipeline.blend);
            let right_blended = blend_enabled(right.pipeline.blend);
            left_blended.cmp(&right_blended).then_with(|| {
                if left_blended && right_blended {
                    left.view_depth
                        .total_cmp(&right.view_depth)
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
                    .filter_map(|material| material.key.map(|key| key.color)),
            )
            .find_map(|tag| texture_cache.get_or_load(tag).map(|loaded| loaded.0))
            .filter(|texture| texture.desc.kind() == TextureType::TextureCube);
        let decal = environment
            .decal
            .and_then(|tag| texture_cache.get_or_load(tag).map(|loaded| loaded.0))
            .and_then(usable_2d_texture);

        let (light_direction, sun_height) = first_person_key_light(
            environment.time_of_day,
            environment.light_position,
            environment.shadow_strength,
        );
        Self {
            preview,
            target_format: texture_cache.render_state.target_format,
            target_size,
            scene: SceneUniform {
                // center.w is the rotation-invariant bounding-sphere radius
                // used by the directional-light orthographic projection.
                center: [center[0], center[1], center[2], shadow_radius],
                params0: [radius, yaw, pitch, zoom],
                params1: [
                    aspect,
                    pan.x * 2.0 / rect.width().max(1.0),
                    -pan.y * 2.0 / rect.height().max(1.0),
                    environment.shadow_softness.clamp(0.0, 1.0),
                ],
                uv_transform: [
                    transform.scale[0],
                    transform.scale[1],
                    transform.offset[0],
                    transform.offset[1],
                ],
                light_direction,
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
                    (environment.fidelity_mode == FidelityMode::PrettyPreview) as u8 as f32,
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
            },
            materials,
            draws,
            cubemap,
            decal,
        }
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
                    key.map_or(0, |key| u64::from(key.color.0)),
                    key.and_then(|key| key.normal)
                        .map_or(0, |tag| u64::from(tag.0)),
                    key.and_then(|key| key.emissive)
                        .map_or(0, |tag| u64::from(tag.0)),
                    u64::from(tint),
                    key.map_or(0, |key| u64::from(key.emissive_strength)),
                    material.control_tag.map_or(0, |tag| u64::from(tag.0)),
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
                ];
                if let Some(decal) = material.investment_decal {
                    values.extend([
                        u64::from(decal.color.0),
                        u64::from(decal.mask.0),
                        decal.detail.map_or(0, |tag| u64::from(tag.0)),
                        match decal.mode {
                            InvestmentDecalMode::SelectorMask => 1,
                            InvestmentDecalMode::DetailSelectorMask => 2,
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
            materials,
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

/// Render the model above display resolution so fine atlas/decal strokes are
/// integrated before presentation. This is especially important while zoomed
/// out: texture mip filtering alone cannot antialias geometry-projected detail
/// once an authored stroke becomes narrower than one viewport pixel.
const MODEL_RENDER_SUPERSAMPLE: f32 = 2.0;
const MAX_MODEL_TARGET_DIMENSION: f32 = 4096.0;
const MAX_MODEL_TARGET_PIXELS: f32 = 8_294_400.0;

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

static MATERIAL_LUMINANCE: LazyLock<Mutex<HashMap<TagHash, Option<MaterialLuminance>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static ADAPTED_EXPOSURE: LazyLock<Mutex<HashMap<Vec<u32>, AdaptedExposure>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn estimate_material_exposure(texture_cache: &TextureCache, materials: &[LoadedMaterial]) -> f32 {
    let mut tags = materials
        .iter()
        .filter_map(|material| material.key.map(|key| key.color))
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
        .get_or_load(tag)
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
        cache.insert(tag, value);
    }
    value
}

static DEBUG_PLACEHOLDER_TEXTURES: LazyLock<Mutex<HashMap<TagHash, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn draw_dye_palette(draw: &ModelDraw) -> Option<[[f32; 4]; 3]> {
    draw.gear_dye.map(|dye| [dye.color; 3])
}

fn is_debug_placeholder_texture(texture_cache: &TextureCache, tag: TagHash) -> bool {
    if let Some(value) = DEBUG_PLACEHOLDER_TEXTURES
        .lock()
        .ok()
        .and_then(|cache| cache.get(&tag).copied())
    {
        return value;
    }
    let value = texture_cache
        .get_or_load(tag)
        .and_then(|(texture, _id)| texture.to_image(&texture_cache.render_state, 0).ok())
        .map(|image| debug_placeholder_pixels(&image.thumbnail(64, 64).to_rgba8()))
        .unwrap_or(false);
    if let Ok(mut cache) = DEBUG_PLACEHOLDER_TEXTURES.lock() {
        cache.insert(tag, value);
    }
    value
}

fn debug_placeholder_pixels(image: &image::RgbaImage) -> bool {
    let mut neon = 0usize;
    let mut neutral_dark = 0usize;
    let mut visible = 0usize;
    for pixel in image.pixels() {
        let [r, g, b, a] = pixel.0;
        if a < 16 {
            continue;
        }
        visible += 1;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        if max < 140 && max.saturating_sub(min) < 24 {
            neutral_dark += 1;
        }
        if b > 170 && g < 115 && (r > 120 || r < 80) {
            neon += 1;
        }
    }
    visible > 0 && neon * 100 >= visible && neutral_dark * 2 >= visible
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
        (0.5 - view[2] * scale * 0.25).clamp(0.0, 1.0),
    ]
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
    material_layout: wgpu::BindGroupLayout,
    present_layout: wgpu::BindGroupLayout,
    bloom_layout: wgpu::BindGroupLayout,
    lighting_layout: wgpu::BindGroupLayout,
    material_sampler: wgpu::Sampler,
    present_sampler: wgpu::Sampler,
    _fallback_color: wgpu::Texture,
    fallback_color_view: wgpu::TextureView,
    model_shader: wgpu::ShaderModule,
    model_pipeline_layout: wgpu::PipelineLayout,
    model_pipelines: Vec<(ModelPipelineKey, wgpu::RenderPipeline)>,
    present_pipeline: wgpu::RenderPipeline,
    lighting_pipeline: wgpu::RenderPipeline,
    bloom_bright_pipeline: wgpu::RenderPipeline,
    bloom_downsample_pipeline: wgpu::RenderPipeline,
    bloom_blur_horizontal_pipeline: wgpu::RenderPipeline,
    bloom_blur_vertical_pipeline: wgpu::RenderPipeline,
    // One depth-only pipeline per authored cull mode. A single back-face
    // pipeline makes two-sided and reversed-winding parts cast incorrectly.
    shadow_pipelines: [wgpu::RenderPipeline; 3],
    shadow_sampler: wgpu::Sampler,
    _fallback_cubemap: wgpu::Texture,
    fallback_cubemap_view: wgpu::TextureView,
    cubemap_sampler: wgpu::Sampler,
}

struct ModelTargetResources {
    size: [u32; 2],
    _color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _lit_color: wgpu::Texture,
    lit_color_view: wgpu::TextureView,
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
    _material_buffers: Vec<wgpu::Buffer>,
    material_bind_groups: Vec<wgpu::BindGroup>,
    bloom_half_bind_group: wgpu::BindGroup,
    bloom_quarter_bind_group: wgpu::BindGroup,
    bloom_blur_horizontal_bind_group: wgpu::BindGroup,
    bloom_blur_vertical_bind_group: wgpu::BindGroup,
    present_bind_group: wgpu::BindGroup,
    lighting_bind_group: wgpu::BindGroup,
    opaque_bundles: Vec<wgpu::RenderBundle>,
    decal_bundles: Vec<wgpu::RenderBundle>,
    transparent_bundles: Vec<wgpu::RenderBundle>,
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
    preview: &GpuModelPreview,
    draws: &[PreparedDraw],
    accepted_passes: &[RenderPassKind],
) -> Vec<wgpu::RenderBundle> {
    draws
        .par_iter()
        .filter(|draw| {
            draw.passes
                .iter()
                .any(|pass| accepted_passes.contains(pass))
        })
        .filter_map(|draw| {
            let (_, pipeline) = pipelines
                .model_pipelines
                .iter()
                .find(|(key, _)| *key == draw.pipeline)?;
            let material = material_bind_groups.get(draw.material_index)?;
            let transparent = accepted_passes == [RenderPassKind::ForwardTransparent];
            let color_formats = if transparent {
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
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                });
            bundle.set_pipeline(pipeline);
            bundle.set_bind_group(0, scene_bind_group, &[]);
            bundle.set_bind_group(1, material, &[]);
            bundle.set_vertex_buffer(0, preview.vertex_buffer.slice(..));
            bundle.set_index_buffer(preview.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            bundle.draw_indexed(draw.indices.clone(), 0, 0..1);
            Some(bundle.finish(&wgpu::RenderBundleDescriptor {
                label: Some("quicktag_model_parallel_draw"),
            }))
        })
        .collect()
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

        if let Some(resources) = callback_resources.get_mut::<ModelPipelineResources>() {
            let keys = self
                .draws
                .iter()
                .flat_map(|draw| {
                    [
                        Some(draw.pipeline),
                        (!draw.passes.contains(&RenderPassKind::ForwardTransparent))
                            .then_some(draw.pipeline.material_flags()),
                        (!draw.passes.contains(&RenderPassKind::ForwardTransparent))
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
                    .any(|(existing, _pipeline)| *existing == key)
                {
                    continue;
                }
                let shared_pipeline = resources
                    .model_pipelines
                    .iter()
                    .find(|(existing, _)| existing.gpu_equivalent(key))
                    .map(|(_, pipeline)| pipeline.clone());
                if let Some(pipeline) = shared_pipeline {
                    resources.model_pipelines.push((key, pipeline));
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
        }

        let Some((
            scene_layout,
            shadow_scene_layout,
            material_layout,
            present_layout,
            bloom_layout,
            lighting_layout,
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
                    resources.material_layout.clone(),
                    resources.present_layout.clone(),
                    resources.bloom_layout.clone(),
                    resources.lighting_layout.clone(),
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

        let needs_target = callback_resources
            .get::<ModelTargetResources>()
            .is_none_or(|resources| resources.size != self.target_size || needs_pipeline);
        if needs_target {
            callback_resources.insert(create_target_resources(device, self.target_size));
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
            }
        } else {
            let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quicktag_model_scene_uniform"),
                contents: bytemuck::bytes_of(&self.scene),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
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
                        handle.create_view(&wgpu::TextureViewDescriptor {
                            format: Some(srgb_texture_format(texture.desc.format)),
                            dimension: Some(wgpu::TextureViewDimension::Cube),
                            array_layer_count: Some(6),
                            ..Default::default()
                        })
                    })
                })
                .unwrap_or(fallback_cubemap_view);
            let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            });
            let shadow_scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quicktag_model_shadow_scene_bind_group"),
                layout: &shadow_scene_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                }],
            });

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
                        material.control.is_some() as u8 as f32,
                        material.mask_palette.is_some() as u8 as f32,
                    ],
                    solid_surface: material
                        .solid_surface
                        .map(|surface| [surface[0], surface[1], 1.0, 0.0])
                        .unwrap_or_default(),
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
                        .unwrap_or_default(),
                    wear_scratches_projection: material
                        .mod_wear
                        .map(|wear| wear.scratches_projection)
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    wear_grime_projection: material
                        .mod_wear
                        .map(|wear| {
                            std::array::from_fn(|axis| {
                                wear.grime_projection[axis]
                                    + wear.grime_projection_unique_delta[axis] * wear.unique_id
                            })
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
                        .unwrap_or([1.0, 1.0, 0.0, 0.0]),
                    wear_scratches_remap_base: material
                        .mod_wear
                        .map(|wear| wear.scratches_remap_base)
                        .unwrap_or_default(),
                    wear_scratches_remap_scale: material
                        .mod_wear
                        .map(|wear| wear.scratches_remap_scale)
                        .unwrap_or([1.0; 4]),
                    wear_surface_params: material
                        .mod_wear
                        .map(|wear| [wear.condition_blend, 4.5947933, 0.0, 0.0])
                        .unwrap_or_default(),
                    gear_palette_default: material.gear_dye_default.unwrap_or_default(),
                    gear_palette_colors: material
                        .gear_dye_palette
                        .map(|palette| palette.map(|dye| dye.color))
                        .unwrap_or_default(),
                    decal_params: material
                        .investment_decal
                        .map(|decal| {
                            [
                                match decal.mode {
                                    InvestmentDecalMode::SelectorMask => 1.0,
                                    InvestmentDecalMode::DetailSelectorMask => 2.0,
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
                        0.0,
                        0.0,
                        0.0,
                    ],
                    pattern_projection: material
                        .gear_pattern
                        .map(|pattern| pattern.projection)
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
                        .unwrap_or_default(),
                    pattern_stripe: material
                        .gear_pattern
                        .map(|pattern| pattern.stripe)
                        .unwrap_or_default(),
                    pattern_contour: material
                        .gear_pattern
                        .map(|pattern| {
                            let mut contour = pattern.contour;
                            contour[3] = pattern.warp[1];
                            contour
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
                    .map(|texture| material_texture_view(texture, true))
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
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let wear_grime_view = material
                    .wear_grime
                    .as_ref()
                    .map(|texture| material_texture_view(texture, true))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let wear_damage_view = material
                    .wear_damage
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let pattern_field_view = material
                    .pattern_field
                    .as_ref()
                    .map(|texture| material_texture_view(texture, false))
                    .unwrap_or_else(|| fallback_color_view.clone());
                let authored_sampler = material
                    .sampler
                    .map(|desc| create_model_sampler(device, desc));
                let sampler = authored_sampler.as_ref().unwrap_or(&material_sampler);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    ],
                });
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
            let bloom_half_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            });
            let bloom_quarter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            });
            let bloom_blur_horizontal_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                });
            let bloom_blur_vertical_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                });
            let present_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            });
            let lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                ],
            });
            let make_bundles = |passes: &[RenderPassKind]| {
                callback_resources
                    .get::<ModelPipelineResources>()
                    .map(|pipelines| {
                        create_model_render_bundles(
                            device,
                            pipelines,
                            &scene_bind_group,
                            &material_bind_groups,
                            &self.preview,
                            &self.draws,
                            passes,
                        )
                    })
                    .unwrap_or_default()
            };
            let opaque_bundles = make_bundles(&[
                RenderPassKind::OpaqueCompatibility,
                RenderPassKind::AlphaTestedCompatibility,
                RenderPassKind::UnknownCompatibility,
            ]);
            let decal_bundles = make_bundles(&[RenderPassKind::InvestmentDecalCompatibility]);
            let transparent_bundles = make_bundles(&[RenderPassKind::ForwardTransparent]);
            callback_resources.insert(ModelFrameResources {
                key: frame_key,
                scene_buffer,
                shadow_scene_bind_group,
                scene_bind_group,
                _material_buffers: material_buffers,
                material_bind_groups,
                bloom_half_bind_group,
                bloom_quarter_bind_group,
                bloom_blur_horizontal_bind_group,
                bloom_blur_vertical_bind_group,
                present_bind_group,
                lighting_bind_group,
                opaque_bundles,
                decal_bundles,
                transparent_bundles,
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
            shadow_pass.set_vertex_buffer(0, self.preview.vertex_buffer.slice(..));
            shadow_pass.set_index_buffer(
                self.preview.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for draw in self
                .draws
                .iter()
                .filter(|draw| draw.passes.contains(&RenderPassKind::Shadow))
            {
                shadow_pass.set_pipeline(
                    &pipelines.shadow_pipelines[shadow_pipeline_index(draw.pipeline.rasterizer)],
                );
                shadow_pass.set_bind_group(
                    1,
                    &frame.material_bind_groups[draw.material_index],
                    &[],
                );
                shadow_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
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
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.execute_bundles(frame.opaque_bundles.iter());
        drop(pass);

        for (label, bundles) in [(
            "quicktag_model_investment_decal_pass",
            frame.decal_bundles.as_slice(),
        )] {
            if bundles.is_empty() {
                continue;
            }
            let mut special_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
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
            special_pass.execute_bundles(bundles.iter());
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
                .filter(|draw| !draw.passes.contains(&RenderPassKind::ForwardTransparent))
            {
                let emissive_key = draw.pipeline.material_emissive();
                let Some((_, pipeline)) = pipelines
                    .model_pipelines
                    .iter()
                    .find(|(key, _)| *key == emissive_key)
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
                emissive_pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
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
                .filter(|draw| !draw.passes.contains(&RenderPassKind::ForwardTransparent))
            {
                let flag_key = draw.pipeline.material_flags();
                let Some((_, pipeline)) = pipelines
                    .model_pipelines
                    .iter()
                    .find(|(key, _)| *key == flag_key)
                else {
                    continue;
                };
                flags_pass.set_pipeline(pipeline);
                flags_pass.set_bind_group(0, &frame.scene_bind_group, &[]);
                flags_pass.set_bind_group(1, &frame.material_bind_groups[draw.material_index], &[]);
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

        if !frame.transparent_bundles.is_empty() {
            let mut transparent_pass =
                egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("quicktag_model_forward_transparent_pass"),
                    color_attachments: &[Some(load_surface_attachment(&target.lit_color_view))],
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
            transparent_pass.execute_bundles(frame.transparent_bundles.iter());
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
    texture.handle.create_view(&wgpu::TextureViewDescriptor {
        format: Some(if srgb {
            srgb_texture_format(texture.desc.format)
        } else {
            linear_texture_format(texture.desc.format)
        }),
        ..Default::default()
    })
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
        min_lod: read_f32(44)?,
        max_lod: read_f32(48)?,
    })
}

fn create_model_sampler(device: &wgpu::Device, desc: ModelSamplerDesc) -> wgpu::Sampler {
    let address_mode = |mode| match mode {
        1 => wgpu::AddressMode::Repeat,
        2 => wgpu::AddressMode::MirrorRepeat,
        // D3D mirror-once and border modes have no portable WebGPU equivalent.
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
        border_color: None,
    })
}

fn create_pipeline_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
) -> ModelPipelineResources {
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
                visibility: wgpu::ShaderStages::FRAGMENT,
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
                visibility: wgpu::ShaderStages::FRAGMENT,
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
    let fallback_color_view = fallback_color.create_view(&Default::default());
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
            97, 97, 95, 255, 97, 97, 95, 255, 97, 97, 95, 255, 97, 97, 95, 255, 97, 97, 95, 255,
            97, 97, 95, 255,
        ],
    );
    let fallback_cubemap_view = fallback_cubemap.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        array_layer_count: Some(6),
        ..Default::default()
    });
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
        bind_group_layouts: &[&scene_layout, &material_layout],
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
                // Raster slope bias removes most coplanar self-occlusion;
                // receiver bias below handles the remaining grazing angles.
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
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
        material_layout,
        present_layout,
        bloom_layout,
        lighting_layout,
        material_sampler,
        present_sampler,
        _fallback_color: fallback_color,
        fallback_color_view,
        model_shader,
        model_pipeline_layout,
        model_pipelines: Vec::new(),
        present_pipeline,
        lighting_pipeline,
        bloom_bright_pipeline,
        bloom_downsample_pipeline,
        bloom_blur_horizontal_pipeline,
        bloom_blur_vertical_pipeline,
        shadow_pipelines,
        shadow_sampler,
        _fallback_cubemap: fallback_cubemap,
        fallback_cubemap_view,
        cubemap_sampler,
    }
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

fn create_model_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    key: ModelPipelineKey,
) -> wgpu::RenderPipeline {
    let transparent = key.pass == RenderPassKind::ForwardTransparent;
    let emissive_only = key.pass == RenderPassKind::MaterialEmissive;
    let flags_only = key.pass == RenderPassKind::MaterialFlags;
    let targets = if transparent {
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
            Some(surface_target(SURFACE_FORMAT)),
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
                RenderPassKind::InvestmentDecalCompatibility => "fs_investment_decal",
                RenderPassKind::ForwardTransparent => "fs_forward_transparent",
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

fn surface_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
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
        _ => Some(wgpu::BlendState::ALPHA_BLENDING),
    }
}

fn blend_enabled(index: u8) -> bool {
    !matches!(index, 0 | 1 | 57)
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

fn create_target_resources(device: &wgpu::Device, size: [u32; 2]) -> ModelTargetResources {
    let extent = wgpu::Extent3d {
        width: size[0],
        height: size[1],
        depth_or_array_layers: 1,
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
    let color_view = color.create_view(&Default::default());
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
    let lit_color_view = lit_color.create_view(&Default::default());
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
    let surface_normal_view = surface_normal.create_view(&Default::default());
    let surface_properties = surface_texture(
        "quicktag_surface_material_properties",
        SURFACE_PROPERTIES_FORMAT,
    );
    let surface_properties_view = surface_properties.create_view(&Default::default());
    let surface_emissive = surface_texture("quicktag_surface_emissive", SURFACE_EMISSIVE_FORMAT);
    let surface_emissive_view = surface_emissive.create_view(&Default::default());
    let surface_albedo = surface_texture("quicktag_surface_albedo_opacity", SURFACE_FORMAT);
    let surface_albedo_view = surface_albedo.create_view(&Default::default());
    let surface_flags = surface_texture("quicktag_surface_flags", SURFACE_FLAGS_FORMAT);
    let surface_flags_view = surface_flags.create_view(&Default::default());
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
    let depth_view = depth.create_view(&Default::default());
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
    let shadow_depth_view = shadow_depth.create_view(&Default::default());
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
    let bloom_half_size = [(size[0] / 2).max(1), (size[1] / 2).max(1)];
    let bloom_quarter_size = [(size[0] / 4).max(1), (size[1] / 4).max(1)];
    let bloom_half = bloom_texture(
        "quicktag_model_bloom_half",
        bloom_half_size[0],
        bloom_half_size[1],
    );
    let bloom_half_view = bloom_half.create_view(&Default::default());
    let bloom_quarter = bloom_texture(
        "quicktag_model_bloom_quarter",
        bloom_quarter_size[0],
        bloom_quarter_size[1],
    );
    let bloom_quarter_view = bloom_quarter.create_view(&Default::default());
    let bloom_blur = bloom_texture(
        "quicktag_model_bloom_blur_ping_pong",
        bloom_quarter_size[0],
        bloom_quarter_size[1],
    );
    let bloom_blur_view = bloom_blur.create_view(&Default::default());
    ModelTargetResources {
        size,
        _color: color,
        color_view,
        _lit_color: lit_color,
        lit_color_view,
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

struct SceneUniform {
    center: vec4<f32>, params0: vec4<f32>, params1: vec4<f32>, uv_transform: vec4<f32>,
    light_direction: vec4<f32>, postprocess0: vec4<f32>, postprocess1: vec4<f32>, postprocess2: vec4<f32>, postprocess3: vec4<f32>, postprocess4: vec4<f32>, postprocess5: vec4<f32>, fidelity: vec4<f32>,
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

fn reconstruct_object_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let scale = 0.84 * scene.params0.w / max(scene.params0.x, 0.0001);
    let view_position = vec3<f32>(
        -(clip.x - scene.params1.y) / max(scale * scene.params1.x, 0.0001),
        (clip.y - scene.params1.z) / max(scale, 0.0001),
        (0.5 - depth) / max(scale * 0.25, 0.0001),
    );
    return scene.center.xyz + view_direction_to_world(view_position);
}

fn deferred_shadow(uv: vec2<f32>, normal: vec3<f32>) -> f32 {
    let dimensions = vec2<i32>(textureDimensions(scene_depth));
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
    let depth = textureLoad(scene_depth, pixel, 0);
    let position = reconstruct_object_position(uv, depth);
    let light_world = normalize(view_direction_to_world(scene.light_direction.xyz));
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(light_world.z) > 0.95);
    let right = normalize(cross(up, light_world));
    let vertical = cross(light_world, right);
    let relative = (position - scene.center.xyz) / max(scene.center.w, 0.0001);
    let shadow_position = vec3<f32>(
        vec2<f32>(dot(relative, right), dot(relative, vertical)) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5),
        0.5 - dot(relative, light_world) * 0.5,
    );
    if any(shadow_position.xy < vec2<f32>(0.0)) || any(shadow_position.xy > vec2<f32>(1.0)) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(sun_shadow));
    let receiver_bias = texel.x * (0.75 + (1.0 - max(dot(view_direction_to_world(normal), light_world), 0.0)) * 2.0);
    return textureSampleCompareLevel(
        sun_shadow,
        sun_shadow_sampler,
        shadow_position.xy,
        shadow_position.z - receiver_bias,
    );
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
    let model = u32(scene.fidelity.y + 0.5);
    if model == 1u {
        if surface.a <= 0.0 {
            return compatibility;
        }
        let albedo = surface.rgb;
        let roughness = clamp(packed_normal.a, 0.045, 1.0);
        let metalness = clamp(properties.r, 0.0, 1.0);
        let ao = clamp(properties.g, 0.0, 1.0);
        let light = normalize(scene.light_direction.xyz);
        let view = vec3<f32>(0.0, 0.0, 1.0);
        let halfway = normalize(light + view);
        let n_dot_l = max(dot(normal, light), 0.0);
        let n_dot_v = max(dot(normal, view), 0.001);
        let n_dot_h = max(dot(normal, halfway), 0.0);
        let v_dot_h = max(dot(view, halfway), 0.0);
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
        let distribution = alpha2 / max(3.14159265 * denominator * denominator, 0.0001);
        let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
        let visibility_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 0.0001);
        let visibility_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
        let f0 = mix(vec3<f32>(0.04), albedo, metalness);
        let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
        let specular = distribution * visibility_l * visibility_v * fresnel * 0.25;
        let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metalness) * albedo / 3.14159265;
        let shadow = mix(1.0, deferred_shadow(input.uv, normal), scene.light_direction.w);
        let direct = (diffuse + specular) * n_dot_l * scene.postprocess0.w * shadow;
        let ibl_diffuse = albedo * (1.0 - metalness) * scene.postprocess4.z * ao;
        let ibl_specular = f0 * mix(0.08, 1.0, 1.0 - roughness)
            * scene.postprocess4.w * ao;
        let color = (direct + ibl_diffuse + ibl_specular + emissive) * scene.postprocess0.x;
        return vec4<f32>(color, surface.a);
    }
    if model == 2u {
        let light = normalize(scene.light_direction.xyz);
        let lambert = max(dot(normal, light), 0.0);
        let ao = properties.g;
        return vec4<f32>(surface.rgb * (0.18 * ao + 0.82 * lambert) + emissive, compatibility.a);
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
    light_direction: vec4<f32>, postprocess0: vec4<f32>, postprocess1: vec4<f32>, postprocess2: vec4<f32>, postprocess3: vec4<f32>, postprocess4: vec4<f32>, postprocess5: vec4<f32>, fidelity: vec4<f32>,
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
}
@group(1) @binding(0) var color_texture: texture_2d<f32>;
@group(1) @binding(3) var control_texture: texture_2d<f32>;
@group(1) @binding(4) var material_sampler: sampler;
@group(1) @binding(5) var<uniform> material: MaterialUniform;

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
    // Shading light is authored in camera/view space. Convert only shadow
    // projection back into model/world space so both passes agree.
    let light = normalize(view_direction_to_world(scene.light_direction.xyz));
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(light.z) > 0.95);
    let right = normalize(cross(up, light));
    let vertical = cross(light, right);
    let relative = (position - scene.center.xyz) / max(scene.center.w, 0.0001);
    // `light` points from the surface toward the source. The source-facing
    // side must therefore have the smaller depth. The old +dot projection
    // inverted this ordering and let hidden back surfaces shadow front ones.
    let light_depth = 0.5 - dot(relative, light) * 0.5;
    return vec4<f32>(dot(relative, right), dot(relative, vertical), light_depth, 1.0);
}

struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

@fragment
fn fs_main(input: ShadowVertexOutput) {
    let base_color = textureSampleBias(
        color_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    );
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

const MODEL_SHADER: &str = r#"
struct SceneUniform {
    center: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    uv_transform: vec4<f32>,
    light_direction: vec4<f32>,
    postprocess0: vec4<f32>,
    postprocess1: vec4<f32>,
    postprocess2: vec4<f32>,
    postprocess3: vec4<f32>,
    postprocess4: vec4<f32>,
    postprocess5: vec4<f32>,
    fidelity: vec4<f32>,
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
@group(1) @binding(9) var pattern_field_texture: texture_2d<f32>;

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
    let light = normalize(view_direction_to_world(scene.light_direction.xyz));
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(light.z) > 0.95);
    let right = normalize(cross(up, light));
    let vertical = cross(light, right);
    let relative = (position - scene.center.xyz) / max(scene.center.w, 0.0001);
    let light_depth = 0.5 - dot(relative, light) * 0.5;
    return vec4<f32>(dot(relative, right), dot(relative, vertical), light_depth, 1.0);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_position: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_tangent: vec4<f32>,
    @location(4) ambient_occlusion: f32,
    @location(5) shadow_position: vec3<f32>,
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
    let depth = clamp(0.5 - view_position.z * scale * 0.25, 0.0, 1.0);

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
    let shadow_clip = light_clip(input.position);
    output.shadow_position = vec3<f32>(shadow_clip.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5), shadow_clip.z);
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
    // Match the compiled shader's object-space tri-planar projection. The XY
    // scale/offset comes from TFX output 40; using it also prevents camera-angle
    // shimmer and the incorrect high-frequency 40x tiling.
    var weights = pow(abs(normalize(normal)), vec3<f32>(4.0));
    weights /= max(weights.x + weights.y + weights.z, 0.0001);
    let x = textureSampleBias(
        texture,
        material_sampler,
        position.zy * projection.xy + projection.zw,
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

fn gear_pattern_plane(axis_position: f32, field_uv: vec2<f32>, warp: f32) -> f32 {
    let field = textureSampleBias(
        pattern_field_texture,
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
    let position = input.procedural_position;
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









fn apply_weapon_mod_condition(albedo: vec3<f32>, input: VertexOutput) -> vec3<f32> {
    // wear_params = TFX outputs [24, 42, 49, resources-present]. Game data
    // evaluates these to Enhanced [1,1,1], Deluxe [0,0,1], Superior [0,0,0].
    // No rarity-wide brightness scalar exists.
    if material.wear_params.w < 0.5 {
        return albedo;
    }

    let scratches = triplanar_surface(
        wear_scratches_texture,
        input.procedural_position,
        input.procedural_normal,
        material.wear_scratches_projection,
    ).r;
    let grime = triplanar_surface(
        wear_grime_texture,
        input.procedural_position,
        input.procedural_normal,
        material.wear_grime_projection,
    );
    let damage = triplanar_surface(
        wear_damage_texture,
        input.procedural_position,
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
    return clamp(conditioned, vec3<f32>(0.0), vec3<f32>(4.0));
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
    let sampled_xy = (textureSampleBias(
        normal_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    ).xy * 2.0 - 1.0) * 0.35;
    let sampled_z = sqrt(max(1.0 - dot(sampled_xy, sampled_xy), 0.0));
    let sampled = normalize(vec3<f32>(sampled_xy, sampled_z));
    return normalize(mat3x3<f32>(tangent, bitangent, base_normal) * sampled);
}

fn fallback_surface(albedo: vec3<f32>) -> vec2<f32> {
    // Strict Tiger uses a neutral dielectric fallback. Pretty Preview retains
    // legacy luma/chroma guesses, but never presents them as authored values.
    if scene.fidelity.x < 0.5 {
        return vec2<f32>(0.82, 0.0);
    }
    let luma = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma = max(albedo.r, max(albedo.g, albedo.b))
        - min(albedo.r, min(albedo.g, albedo.b));
    // Conservative class defaults for shader families without a proven
    // packed material channel: dark coatings, dusty coloured coatings, pale
    // metal, then rough dielectric polymer.
    if luma < 0.055 {
        return vec2<f32>(0.86, 0.0);
    }
    if luma < 0.18 && albedo.r > albedo.g * 1.12 && albedo.b > albedo.g * 1.06 {
        return vec2<f32>(0.80, 0.0);
    }
    if luma > 0.28 && chroma < 0.14 {
        return vec2<f32>(0.68, 0.08);
    }
    return vec2<f32>(0.82, 0.0);
}

fn material_surface(uv: vec2<f32>, albedo: vec3<f32>) -> vec2<f32> {
    if material.solid_surface.z > 0.5 {
        return material.solid_surface.xy;
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
    let authored_roughness = authored * material.roughness_remap.x + material.roughness_remap.y;
    let class_surface = fallback_surface(albedo);
    let roughness = max(mix(class_surface.x, authored_roughness, 0.45), 0.42);
    let authored_metalness = authored * material.metal_remap.x + material.metal_remap.y;
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

// Stable Vogel disk. Hardware comparison filtering turns each tap into a
// bilinear 2x2 PCF sample. More taps keep wide softness settings smooth while
// remaining deterministic, so the preview does not shimmer while orbiting.
const SHADOW_SAMPLE_COUNT = 24u;
const SHADOW_DISK = array<vec2<f32>, 24>(
    vec2<f32>( 0.1443,  0.0000), vec2<f32>(-0.1843,  0.1689),
    vec2<f32>( 0.0282, -0.3215), vec2<f32>( 0.2324,  0.3031),
    vec2<f32>(-0.4264, -0.0754), vec2<f32>( 0.4039, -0.2569),
    vec2<f32>(-0.1351,  0.5026), vec2<f32>(-0.2577, -0.4961),
    vec2<f32>( 0.5590,  0.2041), vec2<f32>(-0.5816,  0.2401),
    vec2<f32>( 0.2803, -0.5991), vec2<f32>( 0.2072,  0.6605),
    vec2<f32>(-0.6244, -0.3619), vec2<f32>( 0.7325, -0.1610),
    vec2<f32>(-0.4470,  0.6359), vec2<f32>(-0.1033, -0.7970),
    vec2<f32>( 0.6340,  0.5343), vec2<f32>(-0.8532,  0.0353),
    vec2<f32>( 0.6223, -0.6193), vec2<f32>(-0.0416,  0.9004),
    vec2<f32>(-0.5922, -0.7096), vec2<f32>( 0.9380,  0.1262),
    vec2<f32>(-0.7948,  0.5530), vec2<f32>( 0.2172, -0.9654),
);

fn directional_shadow(input: VertexOutput) -> f32 {
    if any(input.shadow_position.xy < vec2<f32>(0.0))
        || any(input.shadow_position.xy > vec2<f32>(1.0))
        || input.shadow_position.z <= 0.0
        || input.shadow_position.z >= 1.0 {
        return 1.0;
    }

    let shadow_texel = 1.0 / vec2<f32>(textureDimensions(sun_shadow));
    let light_world = normalize(view_direction_to_world(scene.light_direction.xyz));
    let geometric_normal = normalize(input.world_normal);
    let n_dot_light = max(dot(geometric_normal, light_world), 0.0);
    // Depth is normalized across the fitted bounding sphere. Expressing bias
    // in shadow texels keeps it stable for every weapon size and suppresses
    // triangle/segment acne without detaching genuine contact shadows.
    let receiver_bias = shadow_texel.x * (0.75 + (1.0 - n_dot_light) * 2.0);
    let softness = clamp(scene.params1.w, 0.0, 1.0);

    // A true hard-shadow endpoint is useful for inspection and cheaper than
    // running the full PCF kernel. The squared response gives finer control
    // near the sharp end while still allowing a broad penumbra at 1.0.
    if softness <= 0.001 {
        return textureSampleCompare(
            sun_shadow,
            sun_shadow_sampler,
            input.shadow_position.xy,
            input.shadow_position.z - receiver_bias,
        );
    }

    let filter_radius = 0.75 + softness * softness * 48.0;
    var visibility = 0.0;
    for (var sample = 0u; sample < SHADOW_SAMPLE_COUNT; sample++) {
        let sample_uv = input.shadow_position.xy
            + SHADOW_DISK[sample] * shadow_texel * filter_radius;
        // Out-of-frustum space contains no caster. Check every tap instead of
        // clamping it to an edge depth, which would stretch a silhouette into
        // a false shadow along the light-frustum boundary.
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            visibility += 1.0;
        } else {
            visibility += textureSampleCompare(
                sun_shadow,
                sun_shadow_sampler,
                sample_uv,
                input.shadow_position.z - receiver_bias,
            );
        }
    }
    return visibility / f32(SHADOW_SAMPLE_COUNT);
}

fn investment_decal_mask(uv: vec2<f32>) -> f32 {
    let raw_mask = textureSampleBias(
        control_texture,
        material_sampler,
        uv,
        material.sampler_params.x,
    ).r;
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

struct FragmentOutput {
    @location(0) compatibility_hdr: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
    @location(2) material_properties: vec4<f32>,
    @location(3) albedo: vec4<f32>,
}

fn shade_model(input: VertexOutput, investment_decal: bool) -> FragmentOutput {
    let base_color = textureSampleBias(
        color_texture,
        material_sampler,
        input.uv,
        material.sampler_params.x,
    );
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
    if investment_decal {
        let decal_mask = investment_decal_mask(input.uv);
        material_alpha = decal_mask * material.decal_params.w;
        if material_alpha <= 0.001 {
            discard;
        }
        sampled_albedo = investment_decal_source(input.uv, base_color.rgb);
        if material.decal_params.x > 1.5 {
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
    sampled_albedo = apply_gear_pattern(sampled_albedo, input);
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
        if !low_r && low_g && low_b {
            dye = material.gear_palette_colors[0].rgb;
        } else if low_r && low_g && !low_b {
            dye = material.gear_palette_colors[2].rgb;
        } else if !low_r && !low_g && low_b {
            dye = material.gear_palette_colors[3].rgb;
        } else if low_r && !low_g && low_b {
            dye = material.gear_palette_colors[1].rgb;
        } else if low_r && !low_g && !low_b {
            dye = material.gear_palette_colors[4].rgb;
        } else if !(low_r && low_g && low_b) {
            dye = material.gear_palette_colors[5].rgb;
        }
        sampled_albedo = dye * detail;
    }
    sampled_albedo = apply_weapon_mod_condition(sampled_albedo, input);
    let albedo = sampled_albedo * material.tint.rgb;
    let normal = mapped_normal(input);
    let light = normalize(scene.light_direction.xyz);
    let view_direction = vec3<f32>(0.0, 0.0, 1.0);
    let surface = material_surface(input.uv, albedo);
    let roughness = surface.x;
    let metalness = surface.y;
    let f0 = mix(vec3<f32>(0.03), albedo, metalness);
    let key_specular = ggx_specular(normal, light, view_direction, roughness, f0);
    let reflection_direction = reflect(-view_direction, normal);
    let sampled_environment = textureSample(environment_cubemap, environment_sampler, reflection_direction).rgb;
    let environment_luma = dot(sampled_environment, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Environment lighting is independent from the navy viewport clear colour.
    // Preserve only 4% of an explicitly selected cubemap's chroma.
    let environment = mix(vec3<f32>(environment_luma), sampled_environment, 0.04)
        * vec3<f32>(1.0, 1.0, 0.97);
    let n_dot_v = max(dot(normal, view_direction), 0.0);
    let environment_fresnel = fresnel_schlick(n_dot_v, f0);
    let vertex_ao = mix(1.0, input.ambient_occlusion, scene.postprocess1.w);
    let sun_visibility = mix(1.0, directional_shadow(input), scene.light_direction.w);
    let n_dot_l = max(dot(normal, light), 0.0);
    // Near-neutral 5400–5900 K source. Diffuse establishes the form; there is
    // deliberately no camera fill, rim term, or Fresnel added to final colour.
    let key_color = vec3<f32>(1.0, 0.91, 0.80);
    // Tiger's extracted base textures are authored for a deferred irradiance
    // working range. Calibrate diffuse only; do not resurrect the old scale on
    // specular, which was responsible for the luminous cream contours.
    let diffuse_working_scale = 10.0;
    let direct_diffuse = albedo * (1.0 - metalness) * key_color
        * scene.postprocess0.w * n_dot_l * sun_visibility * diffuse_working_scale;
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
    }
    var emissive_output = vec3<f32>(0.0);
    if material.params.y > 0.5 {
        emissive_output = textureSampleBias(
            emissive_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).rgb
            * min(material.params.z, 0.01);
        if diagnostic_mode == 0u || diagnostic_mode == 5u {
            color += emissive_output;
        }
    }
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
    color = mix(color, fog_color, clamp(fog_amount, 0.0, 0.96));
    if diagnostic_mode != 1u
        && diagnostic_mode != 3u
        && diagnostic_mode != 6u
    {
        color *= scene.postprocess0.x;
    }
    if diagnostic_mode == 7u {
        color = emissive_output;
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
    output.normal_roughness = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), roughness);
    output.material_properties = vec4<f32>(metalness, vertex_ao, 0.0, output_alpha);
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
    return shade_model(input, false).compatibility_hdr;
}

@fragment
fn fs_material_emissive(input: VertexOutput) -> @location(0) vec4<f32> {
    if material.params.y > 0.5 {
        return vec4<f32>(textureSampleBias(
            emissive_texture,
            material_sampler,
            input.uv,
            material.sampler_params.x,
        ).rgb * min(material.params.z, 0.01), 1.0);
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
    light_direction: vec4<f32>, postprocess0: vec4<f32>, postprocess1: vec4<f32>, postprocess2: vec4<f32>, postprocess3: vec4<f32>, postprocess4: vec4<f32>, postprocess5: vec4<f32>, fidelity: vec4<f32>,
}

fn reconstruct_view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let scale = 0.84 * scene.params0.w / max(scene.params0.x, 0.0001);
    return vec3<f32>(
        -(clip.x - scene.params1.y) / max(scale * scene.params1.x, 0.0001),
        (clip.y - scene.params1.z) / max(scale, 0.0001),
        (0.5 - depth) / max(scale * 0.25, 0.0001),
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
    return vec4<f32>(color, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::{
        BLOOM_SHADER, GpuModelPreview, LIGHTING_SHADER, LightingModel, MAX_MODEL_TARGET_PIXELS,
        MODEL_SHADER, MaterialLuminance, ModelEnvironment, ModelFrameResources, ModelPaintCallback,
        ModelPipelineKey, ModelPipelineResources, PRESENT_SHADER, SHADOW_SHADER, adapt_exposure,
        alpha_mode, blend_enabled, blend_state, bounded_target_size, create_model_pipeline,
        create_model_sampler, create_pipeline_resources, decode_model_sampler_desc,
        exposure_target, first_person_key_light, hiz_draw_visible, model_draws, model_view_depth,
        rasterizer_cull_mode, shadow_pipeline_index, smooth_normals, vertex_ambient_occlusion,
    };
    use crate::{
        geometry::{
            GeometryPreviewKind, GeometryTagPreview, WeaponModPreviewAttachment, WeaponModRarity,
            WireframeMaterialRange, WireframeMaterialTextures, WireframePreview,
        },
        render::{evidence::FidelityMode, pass_plan::RenderPassKind},
        texture::{Texture, cache::TextureCache},
    };
    use eframe::{
        egui,
        egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor, wgpu},
    };
    use either::Either::Left;
    use itertools::Itertools;
    use serde::Serialize;
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
    };
    use tiger_pkg::{GameVersion, MarathonVersion, PackageManager, TagHash, package_manager};

    #[derive(Serialize)]
    struct VisualBaselineMetrics {
        baseline: String,
        width: u32,
        height: u32,
        mean_absolute_error: f64,
        p99_channel_delta: u8,
        max_channel_delta: u8,
        changed_pixel_fraction: f64,
        changed_pixel_threshold: u8,
    }

    #[derive(Serialize)]
    struct VisualCaptureDraw {
        draw_index: usize,
        raw_lod: Option<u8>,
        raw_stage: Option<u8>,
        technique: Option<String>,
        family: String,
        passes: Vec<String>,
        provenance: String,
        tfx_states: Vec<String>,
        unknown_tfx_stages: usize,
        warnings: Vec<String>,
    }

    #[derive(Serialize)]
    struct VisualCaptureMetadata {
        schema: u32,
        adapter_version: &'static str,
        renderer_schema_version: u32,
        asset: &'static str,
        owner: &'static str,
        attachments: [&'static str; 2],
        gpu_name: String,
        gpu_backend: String,
        gpu_driver: String,
        gpu_driver_info: String,
        output_size: [u32; 2],
        yaw_degrees: f32,
        pitch_degrees: f32,
        scale: f32,
        fidelity: String,
        lighting_model: String,
        debug_channel: String,
        draws: Vec<VisualCaptureDraw>,
        unknown_tfx_stages: usize,
    }

    fn verify_visual_baseline(image: &image::RgbaImage, baseline_path: &Path, report_path: &Path) {
        let baseline = image::open(baseline_path)
            .unwrap_or_else(|error| panic!("baseline {}: {error}", baseline_path.display()))
            .to_rgba8();
        assert_eq!(
            image.dimensions(),
            baseline.dimensions(),
            "visual baseline dimensions changed for {}",
            baseline_path.display()
        );

        let changed_pixel_threshold = std::env::var("QUICKTAG_PROBE_CHANGED_THRESHOLD")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8_u8);
        let max_mae = std::env::var("QUICKTAG_PROBE_MAX_MAE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1.5_f64);
        let max_p99 = std::env::var("QUICKTAG_PROBE_MAX_P99")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8_u8);
        let max_changed_fraction = std::env::var("QUICKTAG_PROBE_MAX_CHANGED_FRACTION")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.02_f64);

        let mut channel_deltas = Vec::with_capacity((image.width() * image.height() * 3) as usize);
        let mut changed_pixels = 0_u64;
        for (actual, expected) in image.pixels().zip(baseline.pixels()) {
            let mut changed = false;
            for channel in 0..3 {
                let delta = actual[channel].abs_diff(expected[channel]);
                channel_deltas.push(delta);
                changed |= delta > changed_pixel_threshold;
            }
            changed_pixels += u64::from(changed);
        }
        channel_deltas.sort_unstable();
        let sum = channel_deltas
            .iter()
            .map(|delta| u64::from(*delta))
            .sum::<u64>();
        let mean_absolute_error = sum as f64 / channel_deltas.len().max(1) as f64;
        let p99_index = (channel_deltas.len().saturating_sub(1) * 99) / 100;
        let p99_channel_delta = channel_deltas.get(p99_index).copied().unwrap_or(0);
        let max_channel_delta = channel_deltas.last().copied().unwrap_or(0);
        let changed_pixel_fraction =
            changed_pixels as f64 / u64::from(image.width() * image.height()).max(1) as f64;
        let metrics = VisualBaselineMetrics {
            baseline: baseline_path.display().to_string(),
            width: image.width(),
            height: image.height(),
            mean_absolute_error,
            p99_channel_delta,
            max_channel_delta,
            changed_pixel_fraction,
            changed_pixel_threshold,
        };
        std::fs::write(
            report_path,
            serde_json::to_vec_pretty(&metrics).expect("serialize visual baseline metrics"),
        )
        .unwrap_or_else(|error| panic!("write {}: {error}", report_path.display()));
        eprintln!(
            "visual baseline {}: MAE={mean_absolute_error:.4}, p99={p99_channel_delta}, max={max_channel_delta}, changed>{changed_pixel_threshold}={:.3}%",
            baseline_path.display(),
            changed_pixel_fraction * 100.0,
        );
        assert!(
            mean_absolute_error <= max_mae
                && p99_channel_delta <= max_p99
                && changed_pixel_fraction <= max_changed_fraction,
            "visual regression against {}: MAE {mean_absolute_error:.4}/{max_mae:.4}, p99 {p99_channel_delta}/{max_p99}, changed fraction {changed_pixel_fraction:.5}/{max_changed_fraction:.5}; report {}",
            baseline_path.display(),
            report_path.display(),
        );
    }

    #[test]
    fn computes_triangle_normals() {
        let normals = smooth_normals(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[0, 1, 2],
        );
        assert!(
            normals
                .iter()
                .all(|normal| (normal[2] - 1.0).abs() < 0.0001)
        );
    }

    #[test]
    fn detects_neon_wip_texture_without_rejecting_colored_albedo() {
        let mut wip = image::RgbaImage::from_pixel(32, 32, image::Rgba([45, 45, 45, 255]));
        for x in 0..16 {
            wip.put_pixel(x, 8, image::Rgba([255, 0, 255, 255]));
            wip.put_pixel(x, 16, image::Rgba([0, 0, 255, 255]));
        }
        assert!(super::debug_placeholder_pixels(&wip));

        let purple = image::RgbaImage::from_pixel(32, 32, image::Rgba([180, 30, 220, 255]));
        assert!(!super::debug_placeholder_pixels(&purple));
    }

    #[test]
    fn ignores_invalid_triangle_indices() {
        let normals = smooth_normals(&[[0.0, 0.0, 0.0]], &[0, 1, 2]);
        assert_eq!(normals, vec![[0.0, 0.0, 1.0]]);
    }

    #[test]
    fn validates_model_preview_shaders() {
        for source in [
            MODEL_SHADER,
            PRESENT_SHADER,
            SHADOW_SHADER,
            BLOOM_SHADER,
            LIGHTING_SHADER,
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("WGSL should parse");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("WGSL should validate");
        }
    }

    #[test]
    fn reconstructs_tangent_normal_z_from_xy() {
        assert!(MODEL_SHADER.contains("1.0 - dot(sampled_xy, sampled_xy)"));
        assert!(!MODEL_SHADER.contains("normal_texture, material_sampler, input.uv).xyz * 2.0"));
    }

    #[test]
    fn flips_model_projection_horizontally() {
        assert!(MODEL_SHADER.contains("-view_position.x * scale"));
    }

    #[test]
    fn preserves_configured_default_key_light_direction() {
        let environment = ModelEnvironment::default();
        let (light, _height) = first_person_key_light(
            environment.time_of_day,
            environment.light_position,
            environment.shadow_strength,
        );
        assert!(light[0] > 0.0, "configured orbit X remains positive");
        assert!(light[1] > 0.0, "configured orbit Y remains positive");
        let length = environment
            .light_position
            .into_iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        for axis in 0..3 {
            assert!((light[axis] - environment.light_position[axis] / length).abs() < 0.001);
        }
        assert_eq!(light[3], 1.0);
        assert!(MODEL_SHADER.contains("view_direction_to_world(scene.light_direction.xyz)"));
        assert!(SHADOW_SHADER.contains("view_direction_to_world(scene.light_direction.xyz)"));
    }

    #[test]
    fn uses_requested_inventory_environment_defaults() {
        let environment = ModelEnvironment::default();
        assert_eq!(environment.fog_density, 0.0);
        assert_eq!(environment.bloom_strength, 0.0);
        assert_eq!(environment.sun_intensity, 0.35);
        assert_eq!(environment.exposure, 1.0);
        assert!(!environment.auto_exposure);
        assert_eq!(environment.ambient_intensity, 0.2);
        assert_eq!(environment.specular_ibl_intensity, 0.3);
        assert_eq!(environment.vertex_ao_strength, 0.5);
        assert_eq!(environment.light_position, [0.061, 0.711, 0.701]);
        assert_eq!(environment.light_orbit_radius, -1.0);
        assert_eq!(environment.light_size, 5.0);
        assert!(!environment.light_gizmo);
        assert_eq!(environment.shadow_strength, 1.0);
        assert_eq!(environment.shadow_softness, 0.5);
        assert_eq!(environment.brightness, 1.0);
        assert_eq!(environment.contrast, 1.01);
        assert_eq!(environment.saturation, 1.4);
        assert_eq!(environment.gamma, 1.0);
        assert_eq!(environment.ssao_strength, 10.0);
        assert_eq!(environment.distortion, 0.0);
        assert_eq!(environment.water_strength, 0.0);
        assert_eq!(environment.decal_strength, 0.0);
        assert_eq!(environment.road_decal_strength, 0.0);
        assert_eq!(environment.decal, None);
        assert_eq!(environment.cubemap, None);
        assert_eq!(environment.diagnostic_pass, 0);
        assert!(environment.tone_mapping);
    }

    #[test]
    fn uses_diffuse_dominant_matte_inventory_brdf() {
        assert!(MODEL_SHADER.contains("material.roughness_remap.x"));
        assert!(MODEL_SHADER.contains("material.metal_remap.x"));
        assert!(MODEL_SHADER.contains("let f0 = mix(vec3<f32>(0.03), albedo, metalness)"));
        assert!(MODEL_SHADER.contains("let direct_diffuse"));
        assert!(MODEL_SHADER.contains("key_specular"));
        assert!(MODEL_SHADER.contains("* 0.24"));
        assert!(MODEL_SHADER.contains("let specular_occlusion"));
        assert!(MODEL_SHADER.contains("directional_shadow(input)"));
        assert!(!MODEL_SHADER.contains("rim_"));
    }

    #[test]
    fn uses_front_to_back_light_depth_and_stable_shadow_filtering() {
        for shader in [MODEL_SHADER, SHADOW_SHADER] {
            assert!(shader.contains("0.5 - dot(relative, light) * 0.5"));
            assert!(shader.contains("max(scene.center.w, 0.0001)"));
        }
        assert!(MODEL_SHADER.contains("let receiver_bias = shadow_texel.x"));
        assert!(MODEL_SHADER.contains("const SHADOW_DISK"));
        assert!(MODEL_SHADER.contains("let softness = clamp(scene.params1.w"));
        assert!(MODEL_SHADER.contains("softness * softness * 48.0"));
        assert!(!MODEL_SHADER.contains("shadow_texel * 5.0"));
        assert!(!MODEL_SHADER.contains("mix(0.30, 1.0"));
    }

    #[test]
    fn shadow_pass_preserves_authored_face_culling() {
        assert_eq!(shadow_pipeline_index(0), 0);
        assert_eq!(shadow_pipeline_index(3), 1);
        assert_eq!(shadow_pipeline_index(2), 2);
    }

    #[test]
    fn exposes_required_render_diagnostics() {
        for mode in 1..=6 {
            assert!(MODEL_SHADER.contains(&format!("diagnostic_mode == {mode}u")));
        }
        assert!(PRESENT_SHADER.contains("diagnostic_mode != 5u"));
        assert!(PRESENT_SHADER.contains("diagnostic_mode != 6u"));
    }

    #[test]
    fn does_not_treat_investment_decal_stage_as_user_sticker() {
        let preview = |stage| WireframePreview {
            source: "test".into(),
            position_format: "f32x3 @ +0",
            uv_format: None,
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: None,
            procedural_positions: None,
            procedural_normals: None,
            tangents: None,
            uvs: None,
            normal_format: None,
            tangent_format: None,
            indices: vec![0, 1, 2],
            material_ranges: vec![WireframeMaterialRange {
                index_start: 0,
                index_count: 3,
                raw_lod_category: Some(2),
                render_stage: Some(stage),
                technique: None,
                gear_dye_change_color_index: None,
                procedural_scale: 1.0,
                texture: None,
                textures: WireframeMaterialTextures::default(),
            }],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 0.0],
            vertex_count_total: 3,
            index_count_total: 3,
        };
        let decal = model_draws(&preview(2), None);
        assert!(!decal[0].sticker_proxy);
        assert_eq!(decal[0].packet.raw_lod_category, Some(2));
        assert_eq!(decal[0].packet.raw_render_stage, Some(2));
        assert_eq!(decal[0].packet.technique_hash, None);
        assert!(!model_draws(&preview(1), None)[0].sticker_proxy);
    }

    #[test]
    fn renders_decoded_textureless_material_instead_of_hiding_it() {
        let mut runtime = WireframeMaterialTextures::default();
        runtime.solid_color = Some([0.7, 0.3, 0.1, 1.0]);
        runtime.solid_surface = Some([0.5, 0.25]);
        let wireframe = WireframePreview {
            source: "runtime surface".into(),
            position_format: "f32x3 @ +0",
            uv_format: None,
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: None,
            procedural_positions: None,
            procedural_normals: None,
            tangents: None,
            uvs: None,
            normal_format: None,
            tangent_format: None,
            indices: vec![0, 1, 2, 0, 2, 1],
            material_ranges: vec![
                WireframeMaterialRange {
                    index_start: 0,
                    index_count: 3,
                    raw_lod_category: Some(0),
                    render_stage: Some(0),
                    technique: None,
                    gear_dye_change_color_index: None,
                    procedural_scale: 1.0,
                    texture: None,
                    textures: runtime,
                },
                WireframeMaterialRange {
                    index_start: 3,
                    index_count: 3,
                    raw_lod_category: Some(0),
                    render_stage: Some(0),
                    technique: None,
                    gear_dye_change_color_index: None,
                    procedural_scale: 1.0,
                    texture: None,
                    textures: WireframeMaterialTextures::default(),
                },
            ],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 0.0],
            vertex_count_total: 3,
            index_count_total: 6,
        };
        let draws = model_draws(&wireframe, None);
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].indices, 0..3);
        assert_eq!(draws[0].solid_color, Some([0.7, 0.3, 0.1, 1.0]));
        assert_eq!(draws[0].solid_surface, Some([0.5, 0.25]));
    }

    #[test]
    fn does_not_render_unselected_material_range_gaps() {
        let wireframe = WireframePreview {
            source: "selected material ranges".into(),
            position_format: "f32x3 @ +0",
            uv_format: None,
            vertices: vec![[0.0, 0.0, 0.0]; 5],
            normals: None,
            procedural_positions: None,
            procedural_normals: None,
            tangents: None,
            uvs: None,
            normal_format: None,
            tangent_format: None,
            indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 4],
            material_ranges: vec![WireframeMaterialRange {
                index_start: 3,
                index_count: 3,
                raw_lod_category: Some(0),
                render_stage: Some(0),
                technique: None,
                gear_dye_change_color_index: None,
                procedural_scale: 1.0,
                texture: None,
                textures: WireframeMaterialTextures::default(),
            }],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
            vertex_count_total: 5,
            index_count_total: 9,
        };
        let draws = model_draws(&wireframe, Some(TagHash(0x80A00001)));
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].indices, 3..6);
    }

    #[test]
    fn creates_model_preview_pipelines() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU adapter required for model pipeline smoke test");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("GPU device required for model pipeline smoke test");

        let resources =
            create_pipeline_resources(&device, &queue, wgpu::TextureFormat::Bgra8UnormSrgb);
        let _sampler = create_model_sampler(
            &device,
            super::ModelSamplerDesc {
                filter: 0x55,
                address_u: 1,
                address_v: 2,
                address_w: 3,
                mip_lod_bias: -0.5,
                max_anisotropy: 8,
                min_lod: 0.0,
                max_lod: 12.0,
            },
        );
        for key in [
            ModelPipelineKey::default(),
            ModelPipelineKey {
                rasterizer: 1,
                ..Default::default()
            },
            ModelPipelineKey {
                blend: 8,
                depth_stencil: 15,
                depth_bias: 1,
                ..Default::default()
            },
            ModelPipelineKey {
                blend: 26,
                ..Default::default()
            },
            ModelPipelineKey {
                pass: RenderPassKind::InvestmentDecalCompatibility,
                blend: 26,
                ..Default::default()
            },
            ModelPipelineKey {
                pass: RenderPassKind::ForwardTransparent,
                blend: 8,
                ..Default::default()
            },
            ModelPipelineKey {
                pass: RenderPassKind::MaterialEmissive,
                ..Default::default()
            },
            ModelPipelineKey {
                pass: RenderPassKind::MaterialFlags,
                ..Default::default()
            },
        ] {
            let _pipeline = create_model_pipeline(
                &device,
                &resources.model_shader,
                &resources.model_pipeline_layout,
                key,
            );
        }
    }

    #[test]
    fn maps_two_sided_rasterizer_and_bc1_cutout() {
        assert_eq!(rasterizer_cull_mode(1), None);
        assert_eq!(rasterizer_cull_mode(2), Some(wgpu::Face::Back));
        assert_eq!(alpha_mode(wgpu::TextureFormat::Bc4RUnorm), -1.0);
        assert_eq!(alpha_mode(wgpu::TextureFormat::Bc1RgbaUnorm), 0.5);
        assert_eq!(alpha_mode(wgpu::TextureFormat::Bc7RgbaUnorm), 0.0);

        let transparent = ModelPipelineKey::select(crate::material::TechniqueRenderState {
            blend: Some(26),
            ..Default::default()
        });
        assert_eq!(transparent.blend, 26);
        assert_eq!(transparent.depth_stencil, 15);
        assert_eq!(transparent.depth_bias, 1);

        let opaque_alias = ModelPipelineKey::select(crate::material::TechniqueRenderState {
            blend: Some(57),
            ..Default::default()
        });
        assert!(!blend_enabled(57));
        assert_eq!(opaque_alias.depth_stencil, 2);
        assert_eq!(opaque_alias.depth_bias, 0);
        assert!(blend_state(57).is_none());
        assert!(blend_state(27).is_some());
        assert!(blend_state(76).is_some());
    }

    #[test]
    fn vertex_ao_preserves_flats_and_darkens_folds() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 0, 3, 1];
        let normals = smooth_normals(&vertices, &indices);
        let ao = vertex_ambient_occlusion(&vertices, &indices, &normals);
        assert!(ao[0] < 0.9, "shared right-angle fold should be occluded");
        assert_eq!(ao[2], 1.0, "single flat face should remain exposed");
    }

    #[test]
    fn autoexposure_protects_highlights() {
        let values = [
            MaterialLuminance {
                log: 0.01_f32.log2(),
                linear: 0.01,
            },
            MaterialLuminance {
                log: 0.01_f32.log2(),
                linear: 1.0,
            },
        ];
        let geometric_only = (0.18 / 0.01_f32).clamp(0.35, 3.0);

        assert!(exposure_target(&values) < geometric_only);
    }

    #[test]
    fn autoexposure_adapts_to_brightness_faster_than_darkness() {
        let toward_bright_scene = adapt_exposure(2.0, 1.0, 0.5);
        let toward_dark_scene = adapt_exposure(1.0, 2.0, 0.5);

        assert!(2.0 - toward_bright_scene > toward_dark_scene - 1.0);
        assert_eq!(adapt_exposure(1.25, 2.0, 0.0), 1.25);
    }

    #[test]
    fn model_target_respects_gpu_memory_budget() {
        let size = bounded_target_size(4096.0, 4096.0);
        assert!(size[0] as f32 * size[1] as f32 <= MAX_MODEL_TARGET_PIXELS);
        assert!((size[0] as f32 / size[1] as f32 - 1.0).abs() < 0.001);
        assert_eq!(bounded_target_size(320.0, 180.0), [640, 360]);
        assert_eq!(bounded_target_size(1280.0, 720.0), [2560, 1440]);
        assert_eq!(bounded_target_size(1920.0, 1080.0), [3840, 2160]);
    }

    #[test]
    fn hiz_rejects_ranges_behind_nearer_depth() {
        let levels = vec![(4, 4, vec![0.2_f32; 16])];
        let indices = vec![0, 1, 2];
        let hidden = vec![[0.25, 0.25, 0.8], [0.75, 0.25, 0.8], [0.5, 0.75, 0.8]];
        let front = vec![[0.25, 0.25, 0.1], [0.75, 0.25, 0.1], [0.5, 0.75, 0.1]];
        assert!(!hiz_draw_visible(&levels, &hidden, &indices, &(0..3)));
        assert!(hiz_draw_visible(&levels, &front, &indices, &(0..3)));

        let partially_uncovered = vec![(4, 4, {
            let mut depth = vec![0.2_f32; 16];
            depth[10] = 1.0;
            depth
        })];
        assert!(
            hiz_draw_visible(&partially_uncovered, &hidden, &indices, &(0..3)),
            "background coverage must conservatively keep a partially visible range"
        );
        assert!(!ModelEnvironment::default().hiz_culling);
    }

    #[test]
    fn decodes_d3d11_sampler_descriptor() {
        let mut data = vec![0_u8; 52];
        data[0..4].copy_from_slice(&0x55_u32.to_le_bytes());
        data[4..8].copy_from_slice(&1_u32.to_le_bytes());
        data[8..12].copy_from_slice(&2_u32.to_le_bytes());
        data[12..16].copy_from_slice(&3_u32.to_le_bytes());
        data[16..20].copy_from_slice(&(-0.5_f32).to_le_bytes());
        data[20..24].copy_from_slice(&8_u32.to_le_bytes());
        data[44..48].copy_from_slice(&0.0_f32.to_le_bytes());
        data[48..52].copy_from_slice(&12.0_f32.to_le_bytes());

        let sampler = decode_model_sampler_desc(&data).expect("sampler descriptor");
        assert_eq!(sampler.filter, 0x55);
        assert_eq!(
            [sampler.address_u, sampler.address_v, sampler.address_w],
            [1, 2, 3]
        );
        assert_eq!(sampler.max_anisotropy, 8);
        assert_eq!(sampler.mip_lod_bias, -0.5);
        assert_eq!([sampler.min_lod, sampler.max_lod], [0.0, 12.0]);
    }

    #[test]
    fn sorts_depth_in_same_space_as_model_shader() {
        let center = [0.0; 3];
        let front = model_view_depth([1.0, 0.0, 0.0], center, -std::f32::consts::FRAC_PI_2, 0.0);
        let back = model_view_depth([-1.0, 0.0, 0.0], center, -std::f32::consts::FRAC_PI_2, 0.0);

        assert!(
            front > back,
            "larger view Z is closer in preview depth mapping"
        );
    }

    fn crop_render_to_reference(
        render: &image::RgbaImage,
        reference_size: [u32; 2],
    ) -> image::RgbaImage {
        let background = render.get_pixel(0, 0).0;
        let mut min = [render.width(), render.height()];
        let mut max = [0_u32; 2];
        for (x, y, pixel) in render.enumerate_pixels() {
            if pixel.0[..3]
                .iter()
                .zip(background[..3].iter())
                .any(|(value, background)| value.abs_diff(*background) > 12)
            {
                min[0] = min[0].min(x);
                min[1] = min[1].min(y);
                max[0] = max[0].max(x);
                max[1] = max[1].max(y);
            }
        }
        if min[0] > max[0] || min[1] > max[1] {
            return render.clone();
        }

        let padding = ((max[0] - min[0] + 1).max(max[1] - min[1] + 1) / 24).max(4);
        let mut left = min[0].saturating_sub(padding);
        let mut top = min[1].saturating_sub(padding);
        let mut width = (max[0] + padding + 1).min(render.width()) - left;
        let mut height = (max[1] + padding + 1).min(render.height()) - top;
        let target_aspect = reference_size[0] as f32 / reference_size[1].max(1) as f32;
        if width as f32 / height as f32 > target_aspect {
            let wanted = (width as f32 / target_aspect).ceil() as u32;
            let extra = wanted.saturating_sub(height);
            top = top.saturating_sub(extra / 2);
            height = wanted.min(render.height() - top);
        } else {
            let wanted = (height as f32 * target_aspect).ceil() as u32;
            let extra = wanted.saturating_sub(width);
            left = left.saturating_sub(extra / 2);
            width = wanted.min(render.width() - left);
        }
        let crop = image::imageops::crop_imm(render, left, top, width, height).to_image();
        image::imageops::resize(
            &crop,
            reference_size[0],
            reference_size[1],
            image::imageops::FilterType::Lanczos3,
        )
    }

    fn dominant_neutral_rgb(
        image: &image::RgbaImage,
        x: std::ops::Range<u32>,
        y: std::ops::Range<u32>,
    ) -> [f32; 3] {
        let mut bins = [0_usize; 32];
        let mut samples = Vec::new();
        for row in y {
            for column in x.clone() {
                let [red, green, blue, _alpha] = image.get_pixel(column, row).0;
                let max = red.max(green).max(blue);
                let min = red.min(green).min(blue);
                if !(24..=235).contains(&max) || max.saturating_sub(min) > 18 {
                    continue;
                }
                let luma = (u16::from(red) + u16::from(green) + u16::from(blue)) / 3;
                bins[(luma / 8).min(31) as usize] += 1;
                samples.push((luma, [red, green, blue]));
            }
        }
        let dominant_bin = bins
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .map(|(index, _)| index as u16)
            .expect("neutral color sample");
        let mut sum = [0_u64; 3];
        let mut count = 0_u64;
        for (luma, color) in samples {
            if luma / 8 != dominant_bin {
                continue;
            }
            for channel in 0..3 {
                sum[channel] += u64::from(color[channel]);
            }
            count += 1;
        }
        assert!(
            count > 20,
            "neutral sample must contain a stable color cluster"
        );
        [
            sum[0] as f32 / count as f32,
            sum[1] as f32 / count as f32,
            sum[2] as f32 / count as f32,
        ]
    }

    fn rgb_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
        left.into_iter()
            .zip(right)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn bright_component_profile(image: &image::RgbaImage) -> (usize, usize) {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let mut bright = vec![false; width * height];
        for (x, y, pixel) in image.enumerate_pixels() {
            let [red, green, blue, _alpha] = pixel.0;
            let max = red.max(green).max(blue);
            let min = red.min(green).min(blue);
            bright[y as usize * width + x as usize] = min > 175 && max - min < 38;
        }

        let total = bright.iter().filter(|value| **value).count();
        let mut largest = 0usize;
        let mut visited = vec![false; bright.len()];
        for start in 0..bright.len() {
            if !bright[start] || visited[start] {
                continue;
            }
            let mut stack = vec![start];
            visited[start] = true;
            let mut size = 0usize;
            while let Some(index) = stack.pop() {
                size += 1;
                let x = index % width;
                let y = index / width;
                for neighbor in [
                    (x > 0).then_some(index - 1),
                    (x + 1 < width).then_some(index + 1),
                    (y > 0).then_some(index - width),
                    (y + 1 < height).then_some(index + width),
                ]
                .into_iter()
                .flatten()
                {
                    if bright[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
            largest = largest.max(size);
        }
        (total, largest)
    }

    fn dominant_orange_rgb(
        image: &image::RgbaImage,
        x: std::ops::Range<u32>,
        y: std::ops::Range<u32>,
    ) -> ([f32; 3], usize) {
        let mut bins = std::collections::BTreeMap::<[u8; 3], (usize, [u64; 3])>::new();
        for row in y {
            for column in x.clone() {
                let [red, green, blue, _alpha] = image.get_pixel(column, row).0;
                if red < 70 || red < green.saturating_add(30) || green < blue.saturating_add(15) {
                    continue;
                }
                let entry = bins
                    .entry([red / 8, green / 8, blue / 8])
                    .or_insert((0, [0; 3]));
                entry.0 += 1;
                for (sum, value) in entry.1.iter_mut().zip([red, green, blue]) {
                    *sum += u64::from(value);
                }
            }
        }
        let (_bin, (count, sum)) = bins
            .into_iter()
            .max_by_key(|(_bin, (count, _sum))| *count)
            .expect("orange color sample");
        (sum.map(|channel| channel as f32 / count as f32), count)
    }

    #[derive(Debug)]
    struct MaterialBandProfile {
        luma: [f32; 7],
        foreground: usize,
        black_fraction: f32,
        clipped_fraction: f32,
        purple_pixels: usize,
        cream_pixels: usize,
        green_pixels: usize,
        green_max: u8,
    }

    fn implementation_material_bands(image: &image::RgbaImage) -> MaterialBandProfile {
        let background = image.get_pixel(0, 0).0;
        let mut luma = Vec::new();
        let mut black = 0usize;
        let mut clipped = 0usize;
        let mut purple = 0usize;
        let mut cream = 0usize;
        let mut green = 0usize;
        let mut green_max = 0u8;
        for pixel in image.pixels() {
            let [red, green_channel, blue, _alpha] = pixel.0;
            let distance = [red, green_channel, blue]
                .into_iter()
                .zip(background)
                .map(|(value, background)| (f32::from(value) - f32::from(background)).powi(2))
                .sum::<f32>()
                .sqrt();
            if distance < 10.0 {
                continue;
            }
            let value = f32::from(red) * 0.2126
                + f32::from(green_channel) * 0.7152
                + f32::from(blue) * 0.0722;
            luma.push(value);
            black += ((16.0..=40.0).contains(&value)) as usize;
            clipped += (red > 210 || green_channel > 210 || blue > 210) as usize;
            purple += (red > green_channel.saturating_add(3)
                && blue > green_channel.saturating_add(2)
                && (45.0..=105.0).contains(&value)) as usize;
            cream += (red >= green_channel
                && green_channel >= blue
                && (105.0..=195.0).contains(&value)) as usize;
            if green_channel > red.saturating_add(35) && green_channel > blue.saturating_add(25) {
                green += 1;
                green_max = green_max.max(green_channel);
            }
        }
        assert!(luma.len() > 1_000, "material-band segmentation failed");
        luma.sort_by(f32::total_cmp);
        let quantile = |percent: usize| luma[(luma.len() - 1) * percent / 100];
        let count = luma.len();
        MaterialBandProfile {
            luma: [1, 5, 25, 50, 75, 95, 99].map(quantile),
            foreground: count,
            black_fraction: black as f32 / count as f32,
            clipped_fraction: clipped as f32 / count as f32,
            purple_pixels: purple,
            cream_pixels: cream,
            green_pixels: green,
            green_max,
        }
    }

    /// Background-independent comparison for inventory screenshots. Geometry
    /// is registered by `crop_render_to_reference`; this profile then compares
    /// foreground RGB, luminance, and chroma distributions. It intentionally
    /// ignores game UI badges in the upper/lower-right corners.
    fn foreground_visual_similarity(
        reference: &image::RgbaImage,
        implementation: &image::RgbaImage,
    ) -> (f32, [f32; 5], [f32; 5], usize, usize) {
        fn foreground_samples(image: &image::RgbaImage) -> Vec<[f32; 5]> {
            let patch = (image.width().min(image.height()) / 20).clamp(2, 16);
            let mut background = [0_u64; 3];
            let mut background_count = 0_u64;
            for y in 0..patch {
                for x in 0..patch {
                    for (sum, channel) in background.iter_mut().zip(image.get_pixel(x, y).0) {
                        *sum += u64::from(channel);
                    }
                    background_count += 1;
                }
            }
            let background = background.map(|sum| sum as f32 / background_count as f32);
            let background_luma =
                background[0] * 0.2126 + background[1] * 0.7152 + background[2] * 0.0722;
            let mut samples = Vec::new();
            for (x, y, pixel) in image.enumerate_pixels() {
                let normalized = [
                    x as f32 / image.width().max(1) as f32,
                    y as f32 / image.height().max(1) as f32,
                ];
                if normalized[0] > 0.82 && normalized[1] > 0.78
                    || normalized[0] > 0.90 && normalized[1] < 0.10
                {
                    continue;
                }
                let rgb = [
                    f32::from(pixel[0]),
                    f32::from(pixel[1]),
                    f32::from(pixel[2]),
                ];
                let luma = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
                let distance = rgb
                    .into_iter()
                    .zip(background)
                    .map(|(value, background)| (value - background).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if distance < 24.0 || luma < background_luma + 8.0 {
                    continue;
                }
                samples.push([
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    luma,
                    rgb.into_iter().fold(f32::MIN, f32::max)
                        - rgb.into_iter().fold(f32::MAX, f32::min),
                ]);
            }
            assert!(samples.len() > 1_000, "foreground segmentation failed");
            samples
        }

        fn profile(mut samples: Vec<[f32; 5]>, sample_count: usize) -> Vec<f32> {
            // Different inventory backgrounds hide different amounts of the
            // almost-black silhouette. Match the brighter common population;
            // black retention is validated separately against absolute bands.
            samples.sort_by(|left, right| right[3].total_cmp(&left[3]));
            samples.truncate(sample_count);
            let mut channels = std::array::from_fn::<Vec<f32>, 5, _>(|channel| {
                samples.iter().map(|sample| sample[channel]).collect()
            });
            let mut result = Vec::with_capacity(5 * 19);
            for channel in &mut channels {
                channel.sort_by(f32::total_cmp);
                for percentile in 1..=19 {
                    let index = ((channel.len() - 1) * percentile / 20).min(channel.len() - 1);
                    result.push(channel[index]);
                }
            }
            result
        }

        let reference_samples = foreground_samples(reference);
        let implementation_samples = foreground_samples(implementation);
        let reference_count = reference_samples.len();
        let implementation_count = implementation_samples.len();
        let common_count = reference_count.min(implementation_count);
        let reference = profile(reference_samples, common_count);
        let implementation = profile(implementation_samples, common_count);
        let channel_similarity = std::array::from_fn(|channel| {
            let start = channel * 19;
            let mean_delta = reference[start..start + 19]
                .iter()
                .zip(&implementation[start..start + 19])
                .map(|(reference, implementation)| (reference - implementation).abs())
                .sum::<f32>()
                / 19.0;
            (1.0 - mean_delta / 255.0).clamp(0.0, 1.0) * 100.0
        });
        let channel_bias = std::array::from_fn(|channel| {
            let start = channel * 19;
            implementation[start..start + 19]
                .iter()
                .zip(&reference[start..start + 19])
                .map(|(implementation, reference)| implementation - reference)
                .sum::<f32>()
                / 19.0
        });
        let luma_quantiles =
            |profile: &[f32]| [0, 4, 9, 14, 18].map(|index| profile[3 * 19 + index]);
        eprintln!(
            "foreground luma q05/q25/q50/q75/q95: reference={:.1?} implementation={:.1?}",
            luma_quantiles(&reference),
            luma_quantiles(&implementation),
        );
        let mean_delta = reference
            .iter()
            .zip(&implementation)
            .map(|(reference, implementation)| (reference - implementation).abs())
            .sum::<f32>()
            / (5 * 19) as f32;
        (
            (1.0 - mean_delta / 255.0).clamp(0.0, 1.0) * 100.0,
            channel_similarity,
            channel_bias,
            reference_count,
            implementation_count,
        )
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires installed Marathon packages and GPU"]
    async fn renders_weapon_mod_skin_visual_comparisons() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let manager = PackageManager::new(
            packages.to_string_lossy().to_string(),
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(manager));
        quicktag_core::classes::initialize_reference_names();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU adapter");
        let adapter_info = adapter.get_info();
        let required_features = adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            ..Default::default()
        }))
        .expect("GPU device");
        // egui-wgpu deliberately prefers a non-sRGB framebuffer. Mirror the
        // runtime target so the harness exercises the present transfer curve,
        // not only the linear offscreen material pass.
        let target_format = wgpu::TextureFormat::Bgra8Unorm;
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
        let texture_cache = TextureCache::new(render_state.clone());
        let output = std::path::Path::new("target/quicktag-model-probe");
        std::fs::create_dir_all(output).expect("probe directory");
        let mut renders = Vec::new();
        let mut visual_failures = Vec::new();
        let requested_case = std::env::var("QUICKTAG_MODEL_PROBE_CASE").ok();
        let diagnostic_name = std::env::var("QUICKTAG_PROBE_PASS")
            .unwrap_or_else(|_| "final".into())
            .to_ascii_lowercase();
        let (diagnostic_pass, probe_lighting_model) = match diagnostic_name.as_str() {
            "final" => (0, LightingModel::TigerGgxApproximation),
            "base-color" | "base_colour" | "base-colour" => {
                (1, LightingModel::TigerGgxCompatibility)
            }
            "diffuse" | "diffuse-only" => (2, LightingModel::TigerGgxCompatibility),
            "ao" | "ao-only" => (3, LightingModel::TigerGgxCompatibility),
            "specular" | "specular-only" => (4, LightingModel::TigerGgxCompatibility),
            "pre-tone" | "pre_tone" | "hdr" => (5, LightingModel::TigerGgxCompatibility),
            "normal" | "normals" => (6, LightingModel::TigerGgxCompatibility),
            "mrt-albedo" => (0, LightingModel::SurfaceAlbedo),
            "mrt-normal" => (0, LightingModel::SurfaceNormals),
            "mrt-properties" => (0, LightingModel::SurfaceProperties),
            "mrt-emissive" => (7, LightingModel::TigerGgxCompatibility),
            "mrt-flags" => (8, LightingModel::TigerGgxCompatibility),
            value => panic!("unknown QUICKTAG_PROBE_PASS {value}"),
        };
        let tuning_probe = std::env::var("QUICKTAG_PROBE_TUNE")
            .ok()
            .and_then(|value| value.parse::<u8>().ok())
            .is_some_and(|value| value != 0);

        for (name, weapon, weapon_owner, mods, expected_dye_colors, yaw) in [
            (
                "yokais-claw-misriah-full",
                TagHash(0x80B6CC5D),
                TagHash(0x80A96FA4),
                vec![
                    TagHash(0x80A60313),
                    TagHash(0x80A6068A),
                    TagHash(0x80A600AC),
                ],
                vec![
                    [0.473531, 0.027321, 0.014444, 1.0],
                    [0.181164, 0.174647, 0.181164, 1.0],
                    [0.473532, 0.027321, 0.014444, 1.0],
                    [0.730461, 0.0185, 0.0185, 1.0],
                    [0.03434, 0.033105, 0.03434, 1.0],
                    [0.723055, 0.693872, 0.708376, 1.0],
                ],
                0.0,
            ),
            (
                "dont-let-up-brrt-darksight-precision",
                TagHash(0x80AA0CA3),
                TagHash(0x80A7D43D),
                vec![TagHash(0x80A61CF0), TagHash(0x80A6071A)],
                vec![[0.760525, 0.658375, 0.03434, 1.0]],
                0.0,
            ),
            (
                "revamp-br33-vibrant-sport-deluxe",
                TagHash(0x80A9FF17),
                TagHash(0x80A7AA89),
                vec![TagHash(0x80A60FED), TagHash(0x80A60608)],
                vec![],
                -24.0_f32.to_radians(),
            ),
            (
                "dont-let-up-brrt-darksight-precision-yaw-left",
                TagHash(0x80AA0CA3),
                TagHash(0x80A7D43D),
                vec![TagHash(0x80A61CF0), TagHash(0x80A6071A)],
                vec![[0.760525, 0.658375, 0.03434, 1.0]],
                -0.42,
            ),
            (
                "dont-let-up-brrt-darksight-precision-yaw-right",
                TagHash(0x80AA0CA3),
                TagHash(0x80A7D43D),
                vec![TagHash(0x80A61CF0), TagHash(0x80A6071A)],
                vec![[0.760525, 0.658375, 0.03434, 1.0]],
                0.42,
            ),
            (
                "dont-let-up-brrt-darksight-precision-yaw-near-a",
                TagHash(0x80AA0CA3),
                TagHash(0x80A7D43D),
                vec![TagHash(0x80A61CF0), TagHash(0x80A6071A)],
                vec![[0.760525, 0.658375, 0.03434, 1.0]],
                0.18,
            ),
            (
                "dont-let-up-brrt-darksight-precision-yaw-near-b",
                TagHash(0x80AA0CA3),
                TagHash(0x80A7D43D),
                vec![TagHash(0x80A61CF0), TagHash(0x80A6071A)],
                vec![[0.760525, 0.658375, 0.03434, 1.0]],
                0.20,
            ),
            (
                "atrax-sting-v11-rangefinder-suppression",
                TagHash(0x80B6D750),
                TagHash(0x80A7C7B5),
                vec![TagHash(0x80A60D38), TagHash(0x80A60C4F)],
                vec![
                    [0.138432, 0.138432, 0.138432, 1.0],
                    [0.028426, 0.527115, 0.177888, 1.0],
                    [0.031623, 0.031623, 0.031623, 1.0],
                    [0.508881, 0.508881, 0.508881, 1.0],
                    [0.031896, 0.031896, 0.031896, 1.0],
                    [0.520996, 0.520996, 0.520996, 1.0],
                ],
                0.0,
            ),
            (
                "midnight-decay-misriah-precision-quickdraw-slick",
                TagHash(0x80AA09BF),
                TagHash(0x80A96FA4),
                vec![
                    TagHash(0x80A60313),
                    TagHash(0x80A601B6),
                    TagHash(0x80A6068A),
                ],
                vec![
                    [0.03434, 0.033105, 0.03434, 1.0],
                    [0.291771, 0.291771, 0.291771, 1.0],
                    [0.068478, 0.082283, 0.099899, 1.0],
                    [0.064803, 0.064803, 0.064803, 1.0],
                    [0.03434, 0.038204, 0.045186, 1.0],
                    [0.171441, 0.171441, 0.171441, 1.0],
                ],
                0.0,
            ),
            (
                "midnight-decay-misriah-precision-only",
                TagHash(0x80AA09BF),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A60313)],
                vec![],
                0.0,
            ),
            (
                "midnight-decay-misriah-quickdraw-only",
                TagHash(0x80AA09BF),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A601B6)],
                vec![],
                0.0,
            ),
            (
                "midnight-decay-misriah-slick-only",
                TagHash(0x80AA09BF),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A6068A)],
                vec![],
                0.0,
            ),
            (
                "retro-remix-misriah-precision-clean",
                TagHash(0x80B6DF0D),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A60313)],
                vec![],
                0.0,
            ),
            (
                "retro-remix-misriah-precision-enhanced",
                TagHash(0x80B6DF0D),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A60313)],
                vec![],
                0.0,
            ),
            (
                "retro-remix-misriah-precision-deluxe",
                TagHash(0x80B6DF0D),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A60313)],
                vec![],
                0.0,
            ),
            (
                "retro-remix-misriah-precision-superior",
                TagHash(0x80B6DF0D),
                TagHash(0x80A96FA4),
                vec![TagHash(0x80A60313)],
                vec![],
                0.0,
            ),
            (
                "vox-nocturna-misriah-ingame-lighting",
                TagHash(0x80B6DECC),
                TagHash(0x80A96FA4),
                vec![
                    TagHash(0x80A60313),
                    TagHash(0x80A60B69),
                    TagHash(0x80A607D9),
                ],
                vec![],
                -22.6_f32.to_radians(),
            ),
            (
                "vox-nocturna-v85-flat-panel",
                TagHash(0x80A9F63F),
                TagHash(0x80A9F63F),
                vec![],
                vec![],
                -24.0_f32.to_radians(),
            ),
            (
                "m77-investment-decal",
                TagHash(0x80B7B91F),
                TagHash(0x80B7B91F),
                vec![],
                vec![],
                -15.7_f32.to_radians(),
            ),
            (
                "syntax-disrupt-v75-decal",
                TagHash(0x80A9B86C),
                TagHash(0x80A9B86C),
                vec![],
                vec![],
                -24.0_f32.to_radians(),
            ),
            (
                "arata-vectus-v66-detail",
                TagHash(0x80B7C5BD),
                TagHash(0x80B7C5BD),
                vec![],
                vec![],
                -23.5_f32.to_radians(),
            ),
            (
                "runner-achromatic-rush-decal",
                TagHash(0x80B141C0),
                TagHash(0x80B141C0),
                vec![],
                vec![],
                0.0,
            ),
            (
                "runner-detail-selector-decal",
                TagHash(0x80B14470),
                TagHash(0x80B14470),
                vec![],
                vec![],
                0.0,
            ),
        ] {
            if requested_case
                .as_deref()
                .is_some_and(|requested| requested != name)
            {
                continue;
            }
            let lighting_reference_case = name == "vox-nocturna-misriah-ingame-lighting";
            let flat_panel_reference_case = name == "vox-nocturna-v85-flat-panel";
            let investment_decal_reference_case = name == "m77-investment-decal";
            let revamp_baseline_case = name == "revamp-br33-vibrant-sport-deluxe";
            let probe_f32 = |key: &str, fallback: f32| {
                std::env::var(key)
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(fallback)
            };
            let yaw = probe_f32("QUICKTAG_PROBE_YAW_DEGREES", yaw.to_degrees()).to_radians();
            let Some(entry) = package_manager().get_entry(weapon) else {
                eprintln!("skipping stale visual fixture {name}: missing {weapon}");
                continue;
            };
            let rarity = if name.ends_with("-enhanced") {
                Some(WeaponModRarity::Enhanced)
            } else if name.ends_with("-deluxe") {
                Some(WeaponModRarity::Deluxe)
            } else if name.ends_with("-superior") {
                Some(WeaponModRarity::Superior)
            } else {
                None
            };
            let expects_attached_dyes = !expected_dye_colors.is_empty();
            let attachments = mods
                .iter()
                .copied()
                .map(|model_tag| WeaponModPreviewAttachment {
                    model_tag,
                    rarity,
                    unique_id: match rarity {
                        Some(WeaponModRarity::Enhanced) => 0.137,
                        Some(WeaponModRarity::Deluxe) => 0.619,
                        Some(WeaponModRarity::Superior) => 0.853,
                        None => 0.5,
                    },
                })
                .collect_vec();
            let preview = GeometryTagPreview::load_model_with_weapon_mod_attachments(
                cache.clone(),
                weapon,
                &entry,
                weapon,
                weapon_owner,
                &attachments,
            )
            .expect("model preview");
            let GeometryPreviewKind::Model(model) = preview.kind else {
                panic!("weapon preview must be model");
            };
            let wireframe = model.wireframe.as_ref().expect("wireframe");
            if matches!(name, "syntax-disrupt-v75-decal" | "arata-vectus-v66-detail")
                && std::env::var_os("QUICKTAG_MODEL_MATERIAL_REPORT").is_some()
            {
                let mut report = format!(
                    "{name} geometry parts: {:?}\n{name} mesh source: {:?}\n{name} uv transform: {:?}\n{name} model bounds: {:?}..{:?}\n",
                    model.geometry_parts,
                    model.mesh_source,
                    model.preview_uv_transform(),
                    wireframe.min,
                    wireframe.max,
                );
                if let Some(positions) = wireframe.procedural_positions.as_ref() {
                    let minimum: [f32; 3] = std::array::from_fn(|axis| {
                        positions
                            .iter()
                            .map(|position| position[axis])
                            .fold(f32::INFINITY, f32::min)
                    });
                    let maximum: [f32; 3] = std::array::from_fn(|axis| {
                        positions
                            .iter()
                            .map(|position| position[axis])
                            .fold(f32::NEG_INFINITY, f32::max)
                    });
                    use std::fmt::Write as _;
                    writeln!(report, "{name} procedural bounds: {minimum:?}..{maximum:?}").unwrap();
                }
                let frame = super::ModelCameraFrame::from_wireframe(wireframe);
                let pitch = 0.05_f32;
                let scale = 0.84 * 3.1 / frame.radius.max(0.0001);
                let aspect = 640.0 / 1024.0;
                for (index, range) in wireframe.material_ranges.iter().enumerate() {
                    use std::fmt::Write as _;
                    let range_end = range
                        .index_start
                        .saturating_add(range.index_count)
                        .min(wireframe.indices.len());
                    let range_uvs = wireframe.indices[range.index_start.min(range_end)..range_end]
                        .iter()
                        .filter_map(|vertex| wireframe.uvs.as_ref()?.get(*vertex as usize).copied())
                        .collect_vec();
                    let uv_min: [f32; 2] = std::array::from_fn(|axis| {
                        range_uvs
                            .iter()
                            .map(|uv| uv[axis])
                            .fold(f32::INFINITY, f32::min)
                    });
                    let uv_max: [f32; 2] = std::array::from_fn(|axis| {
                        range_uvs
                            .iter()
                            .map(|uv| uv[axis])
                            .fold(f32::NEG_INFINITY, f32::max)
                    });
                    let range_procedural_normals = wireframe.indices
                        [range.index_start.min(range_end)..range_end]
                        .iter()
                        .filter_map(|vertex| {
                            wireframe
                                .procedural_normals
                                .as_ref()?
                                .get(*vertex as usize)
                                .copied()
                        })
                        .collect_vec();
                    let procedural_normal_min: [f32; 3] = std::array::from_fn(|axis| {
                        range_procedural_normals
                            .iter()
                            .map(|normal| normal[axis])
                            .fold(f32::INFINITY, f32::min)
                    });
                    let procedural_normal_max: [f32; 3] = std::array::from_fn(|axis| {
                        range_procedural_normals
                            .iter()
                            .map(|normal| normal[axis])
                            .fold(f32::NEG_INFINITY, f32::max)
                    });
                    let screen_points = wireframe.indices
                        [range.index_start.min(range_end)..range_end]
                        .iter()
                        .filter_map(|vertex| wireframe.vertices.get(*vertex as usize))
                        .map(|position| {
                            let value = [
                                position[0] - frame.center[0],
                                position[1] - frame.center[1],
                                position[2] - frame.center[2],
                            ];
                            let z_up = [value[0], value[2], value[1]];
                            let (cy, sy) = (yaw.cos(), yaw.sin());
                            let (cp, sp) = (pitch.cos(), pitch.sin());
                            let yawed = [
                                z_up[0] * cy + z_up[2] * sy,
                                z_up[1],
                                -z_up[0] * sy + z_up[2] * cy,
                            ];
                            let view = [yawed[0], yawed[1] * cp - yawed[2] * sp];
                            [
                                (-view[0] * scale * aspect + 1.0) * 512.0,
                                (1.0 - view[1] * scale) * 320.0,
                            ]
                        })
                        .collect_vec();
                    let screen_min: [f32; 2] = std::array::from_fn(|axis| {
                        screen_points
                            .iter()
                            .map(|point| point[axis])
                            .fold(f32::INFINITY, f32::min)
                    });
                    let screen_max: [f32; 2] = std::array::from_fn(|axis| {
                        screen_points
                            .iter()
                            .map(|point| point[axis])
                            .fold(f32::NEG_INFINITY, f32::max)
                    });
                    writeln!(
                        report,
                        "{name} range {index}: idx={}..{} screen={screen_min:?}..{screen_max:?} uv={uv_min:?}..{uv_max:?} procedural_normal={procedural_normal_min:?}..{procedural_normal_max:?} procedural_scale={} stage={:?} technique={:?} change_color={:?} color={:?} normal={:?} control={:?} solid={:?} dye={:?} palette={:?} investment={:?} aux={:?}",
                        range.index_start,
                        range.index_start + range.index_count,
                        range.procedural_scale,
                        range.render_stage,
                        range.technique,
                        range.gear_dye_change_color_index,
                        range.textures.color,
                        range.textures.normal,
                        range.textures.control,
                        range.textures.solid_color,
                        range.textures.gear_dye,
                        range.textures.gear_dye_palette.is_some(),
                        range.textures.investment_decal,
                        range.textures.aux,
                    )
                    .unwrap();
                    if let Some(technique) = range.technique
                        && let Some(entry) = package_manager().get_entry(technique)
                        && let Ok(data) = package_manager().read_tag(technique)
                    {
                        let direct = crate::material::texture_bindings_for_technique(&entry, &data)
                            .into_iter()
                            .map(|binding| (binding.stage, binding.slot, binding.tag))
                            .collect_vec();
                        let tfx =
                            crate::material::tfx_texture_bindings_for_technique(&entry, &data)
                                .into_iter()
                                .map(|binding| {
                                    (
                                        binding.stage,
                                        binding.slot,
                                        binding.source_scope,
                                        binding.source_offset,
                                    )
                                })
                                .collect_vec();
                        writeln!(report, "  bindings direct={direct:?} tfx={tfx:?}").unwrap();
                        if let Some(preview) =
                            crate::material::MaterialTagPreview::load(&entry, &data)
                        {
                            let crate::material::MaterialPreviewKind::Technique(preview) =
                                preview.kind;
                            for stage in preview.stages {
                                writeln!(
                                    report,
                                    "  {} shader={:?} inline={:?} expressions={:?}",
                                    stage.stage,
                                    stage.shader,
                                    stage.inline_constants,
                                    stage.bytecode.expressions,
                                )
                                .unwrap();
                                writeln!(
                                    report,
                                    "  {} indexed inline={:?}",
                                    stage.stage,
                                    stage
                                        .inline_constants
                                        .iter()
                                        .copied()
                                        .enumerate()
                                        .collect_vec(),
                                )
                                .unwrap();
                                writeln!(
                                    report,
                                    "  {} tfx constants={:?} ops={:?}",
                                    stage.stage, stage.constants, stage.bytecode.ops,
                                )
                                .unwrap();
                                if let Some(shader) = stage.shader
                                    && let Some(shader_entry) = package_manager().get_entry(shader)
                                    && let Ok(shader_data) =
                                        package_manager().read_tag(TagHash(shader_entry.reference))
                                {
                                    std::fs::write(
                                        output.join(format!(
                                            "{name}-{technique}-{shader}-{}.bin",
                                            stage.stage.to_ascii_lowercase()
                                        )),
                                        shader_data,
                                    )
                                    .unwrap();
                                }
                            }
                        }
                    }
                }
                std::fs::write(output.join(format!("{name}.materials.txt")), report).unwrap();
            }
            let texture_tags = wireframe
                .material_ranges
                .iter()
                .flat_map(|range| {
                    range
                        .textures
                        .color
                        .into_iter()
                        .chain(range.texture)
                        .chain(range.textures.normal)
                        .chain(range.textures.emissive)
                        .chain(range.textures.control)
                        .chain(range.textures.aux.iter().copied())
                        .chain(
                            range
                                .textures
                                .mod_wear
                                .into_iter()
                                .flat_map(|wear| [wear.scratches, wear.grime, wear.damage]),
                        )
                })
                .unique()
                .collect_vec();
            for tag in &texture_tags {
                let texture = Texture::load(&render_state, *tag, true)
                    .unwrap_or_else(|error| panic!("texture {tag}: {error}"));
                if name.starts_with("atrax-sting")
                    || name == "syntax-disrupt-v75-decal"
                    || name == "arata-vectus-v66-detail"
                    || matches!(
                        tag.0,
                        0x80AA0ED7
                            | 0x80AA0ED3
                            | 0x80A9A4F5
                            | 0x80A9A501
                            | 0x80A6149D
                            | 0x80A617EF
                            | 0x80A617BB
                            | 0x80A617EA
                            | 0x80A617EC
                    )
                {
                    texture
                        .to_image(&render_state, 0)
                        .expect("probe texture readback")
                        .save(output.join(format!("texture-{tag}.png")))
                        .expect("save probe texture");
                }
                let id = render_state.renderer.write().register_native_texture(
                    &render_state.device,
                    &texture.view,
                    wgpu::FilterMode::Linear,
                );
                texture_cache
                    .cache
                    .write()
                    .insert(*tag, Left(Some((Arc::new(texture), id))));
            }
            let fallback = wireframe
                .material_ranges
                .iter()
                .find_map(|range| range.textures.color.or(range.texture));
            let gpu = Arc::new(
                GpuModelPreview::create(&render_state.device, wireframe, fallback)
                    .expect("GPU model"),
            );
            let explicit_index_count: usize = wireframe
                .material_ranges
                .iter()
                .map(|range| {
                    let start = range.index_start.min(wireframe.indices.len());
                    let end = range
                        .index_start
                        .saturating_add(range.index_count)
                        .min(wireframe.indices.len());
                    end.saturating_sub(start)
                })
                .sum();
            let drawn_index_count: usize = gpu.draws.iter().map(|draw| draw.indices.len()).sum();
            assert_eq!(
                drawn_index_count, explicit_index_count,
                "excluded stage/LOD gaps must not become fallback draws"
            );
            if revamp_baseline_case {
                assert!(
                    gpu.vertex_abi
                        .attributes
                        .iter()
                        .any(|attribute| attribute.semantic == "POSITION0" && attribute.decoded),
                    "80A9FF17 vertex ABI lost POSITION0"
                );
                let metadata = gpu
                    .draws
                    .iter()
                    .map(|draw| {
                        (
                            draw.packet.raw_lod_category,
                            draw.packet.raw_render_stage,
                            draw.packet.technique_hash,
                            draw.indices.clone(),
                        )
                    })
                    .collect_vec();
                eprintln!("80A9FF17 raw draw metadata: {metadata:?}");
                assert!(
                    metadata
                        .iter()
                        .all(|(lod, stage, technique, _)| lod.is_some()
                            && stage.is_some()
                            && technique.is_some()),
                    "80A9FF17 must retain raw LOD, stage, and technique on every GPU draw: {metadata:?}"
                );
                assert!(
                    gpu.draws.iter().all(|draw| {
                        draw.packet
                            .technique
                            .as_ref()
                            .is_some_and(|technique| !technique.stages.is_empty())
                            && !draw.packet.pass_plan.passes.is_empty()
                            && draw.packet.indices == draw.indices
                            && gpu.provenance.get(draw.packet.source).is_some()
                    }),
                    "80A9FF17 draw packets require per-stage technique ABI and validated pass plans"
                );
                eprintln!(
                    "80A9FF17 families/plans: {:?}",
                    gpu.draws
                        .iter()
                        .map(|draw| (
                            draw.packet.material.family(),
                            &draw.packet.pass_plan.passes,
                            draw.packet
                                .technique
                                .as_ref()
                                .map(|technique| technique.stages.len())
                        ))
                        .collect_vec()
                );
            }
            let size = if lighting_reference_case {
                [997_u32, 326_u32]
            } else {
                [1024_u32, 640_u32]
            };
            let rect = egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            );
            let mut verification_environment = if lighting_reference_case {
                let defaults = ModelEnvironment::default();
                ModelEnvironment {
                    time_of_day: probe_f32("QUICKTAG_PROBE_TIME", defaults.time_of_day),
                    sun_intensity: probe_f32("QUICKTAG_PROBE_SUN", defaults.sun_intensity),
                    fog_density: probe_f32("QUICKTAG_PROBE_FOG", defaults.fog_density),
                    bloom_strength: probe_f32("QUICKTAG_PROBE_BLOOM", defaults.bloom_strength),
                    exposure: probe_f32("QUICKTAG_PROBE_EXPOSURE", defaults.exposure),
                    auto_exposure: std::env::var("QUICKTAG_PROBE_AUTO_EXPOSURE")
                        .ok()
                        .and_then(|value| value.parse::<u8>().ok())
                        .map_or(defaults.auto_exposure, |value| value != 0),
                    vertex_ao_strength: probe_f32(
                        "QUICKTAG_PROBE_VERTEX_AO",
                        defaults.vertex_ao_strength,
                    ),
                    ssao_strength: probe_f32("QUICKTAG_PROBE_SSAO", defaults.ssao_strength),
                    tone_mapping: std::env::var("QUICKTAG_PROBE_TONEMAP")
                        .ok()
                        .and_then(|value| value.parse::<u8>().ok())
                        .map_or(defaults.tone_mapping, |value| value != 0),
                    ambient_intensity: probe_f32(
                        "QUICKTAG_PROBE_AMBIENT",
                        defaults.ambient_intensity,
                    ),
                    specular_ibl_intensity: probe_f32(
                        "QUICKTAG_PROBE_SPECULAR_IBL",
                        probe_f32("QUICKTAG_PROBE_FILL", defaults.specular_ibl_intensity),
                    ),
                    diagnostic_pass,
                    ..defaults
                }
            } else if flat_panel_reference_case {
                ModelEnvironment {
                    diagnostic_pass,
                    ..ModelEnvironment::default()
                }
            } else {
                ModelEnvironment {
                    sun_intensity: 0.0,
                    fog_density: 0.0,
                    bloom_strength: 0.0,
                    exposure: 1.0,
                    auto_exposure: false,
                    vertex_ao_strength: 0.0,
                    ssao_strength: 0.0,
                    tone_mapping: false,
                    ambient_intensity: 1.5,
                    specular_ibl_intensity: 0.0,
                    diagnostic_pass,
                    ..ModelEnvironment::default()
                }
            };
            verification_environment.fidelity_mode = match std::env::var("QUICKTAG_PROBE_FIDELITY")
                .unwrap_or_else(|_| "strict".into())
                .to_ascii_lowercase()
                .as_str()
            {
                "strict" | "strict-tiger" => FidelityMode::StrictTiger,
                "pretty" | "pretty-preview" => FidelityMode::PrettyPreview,
                value => panic!("unknown QUICKTAG_PROBE_FIDELITY {value}"),
            };
            verification_environment.lighting_model = probe_lighting_model;
            let callback = ModelPaintCallback::new(
                gpu,
                &texture_cache,
                wireframe,
                model.preview_uv_transform(),
                None,
                yaw,
                if lighting_reference_case {
                    probe_f32("QUICKTAG_PROBE_PITCH_DEGREES", 16.2).to_radians()
                } else if revamp_baseline_case {
                    probe_f32("QUICKTAG_PROBE_PITCH_DEGREES", 14.0).to_radians()
                } else if flat_panel_reference_case || investment_decal_reference_case {
                    if investment_decal_reference_case {
                        15.2_f32.to_radians()
                    } else {
                        14.8_f32.to_radians()
                    }
                } else {
                    0.05
                },
                3.1,
                egui::Vec2::ZERO,
                false,
                rect,
                1.0,
                verification_environment,
            );
            if revamp_baseline_case {
                let draws = callback
                    .preview
                    .draws
                    .iter()
                    .enumerate()
                    .map(|(draw_index, draw)| {
                        let tfx_states = draw
                            .packet
                            .technique
                            .as_ref()
                            .map(|technique| {
                                technique
                                    .stages
                                    .iter()
                                    .map(|stage| {
                                        format!(
                                            "{}:{:?}",
                                            stage.raw_stage_label, stage.tfx_execution.status
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let unknown_tfx_stages = tfx_states
                            .iter()
                            .filter(|state| {
                                state.contains("Partial")
                                    || state.contains("StoppedAtUnknown")
                                    || state.contains("Invalid")
                            })
                            .count();
                        VisualCaptureDraw {
                            draw_index,
                            raw_lod: draw.packet.raw_lod_category,
                            raw_stage: draw.packet.raw_render_stage,
                            technique: draw.packet.technique_hash.map(|tag| tag.to_string()),
                            family: format!("{:?}", draw.packet.material.family()),
                            passes: draw
                                .packet
                                .pass_plan
                                .passes
                                .iter()
                                .map(|pass| format!("{pass:?}"))
                                .collect(),
                            provenance: callback
                                .preview
                                .provenance
                                .get(draw.packet.source)
                                .map(|record| format!("{:?}", record.evidence))
                                .unwrap_or_else(|| "Missing".into()),
                            tfx_states,
                            unknown_tfx_stages,
                            warnings: draw
                                .packet
                                .pass_plan
                                .warnings
                                .iter()
                                .map(|warning| format!("{warning:?}"))
                                .collect(),
                        }
                    })
                    .collect::<Vec<_>>();
                let capture = VisualCaptureMetadata {
                    schema: 2,
                    adapter_version: crate::render::adapter::GoliathAdapter::ADAPTER_VERSION,
                    renderer_schema_version: 2,
                    asset: "80A9FF17",
                    owner: "80A7AA89",
                    attachments: ["80A60FED", "80A60608"],
                    gpu_name: adapter_info.name.clone(),
                    gpu_backend: format!("{:?}", adapter_info.backend),
                    gpu_driver: adapter_info.driver.clone(),
                    gpu_driver_info: adapter_info.driver_info.clone(),
                    output_size: [1024, 640],
                    yaw_degrees: yaw.to_degrees(),
                    pitch_degrees: callback.scene.params0[2].to_degrees(),
                    scale: 3.1,
                    fidelity: format!("{:?}", verification_environment.fidelity_mode),
                    lighting_model: format!("{:?}", verification_environment.lighting_model),
                    debug_channel: diagnostic_name.clone(),
                    unknown_tfx_stages: draws.iter().map(|draw| draw.unknown_tfx_stages).sum(),
                    draws,
                };
                std::fs::write(
                    output.join(format!("{name}-{diagnostic_name}.capture.json")),
                    serde_json::to_vec_pretty(&capture).expect("serialize capture metadata"),
                )
                .expect("write capture metadata");
            }
            if lighting_reference_case {
                eprintln!(
                    "Vox Nocturna lighting probe: pass={diagnostic_name} yaw={:.2} pitch={:.2} exposure={:.4} environment={verification_environment:?}",
                    yaw.to_degrees(),
                    probe_f32("QUICKTAG_PROBE_PITCH_DEGREES", 16.2),
                    callback.scene.postprocess0[0],
                );
            }
            if let Some(rarity) = rarity {
                eprintln!(
                    "{name} wear materials: {:?}",
                    callback
                        .materials
                        .iter()
                        .filter_map(|material| {
                            Some((
                                material.mod_wear?,
                                material.wear_scratches.is_some(),
                                material.wear_grime.is_some(),
                                material.wear_damage.is_some(),
                            ))
                        })
                        .collect_vec()
                );
                assert!(
                    callback.materials.iter().any(|material| {
                        material.mod_wear.is_some_and(|wear| {
                            wear.rarity == Some(rarity)
                                && [
                                    wear.scratches_projection,
                                    wear.grime_projection,
                                    wear.damage_projection,
                                ]
                                .into_iter()
                                .all(|projection| {
                                    projection.iter().all(|value| value.is_finite())
                                        && (0.05..=100.0).contains(&projection[0].abs())
                                        && (0.05..=100.0).contains(&projection[1].abs())
                                })
                                && wear.condition_controls
                                    == [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
                        }) && material.wear_scratches.is_some()
                            && material.wear_grime.is_some()
                            && material.wear_damage.is_some()
                    }),
                    "{name} did not carry authored projections and three-tier TFX controls into the wear material"
                );
            }
            if name == "arata-vectus-v66-detail" {
                assert!(
                    callback.materials.iter().any(|material| {
                        material.gear_pattern.is_some()
                            && material.pattern_field.is_some()
                            && material.control.is_some()
                    }),
                    "Arata Vectus must decode control-map-selected procedural gear pattern"
                );
            }
            let dye_palettes = callback
                .materials
                .iter()
                .filter_map(|material| material.dye_palette)
                .collect_vec();
            let gear_dye_palettes = callback
                .materials
                .iter()
                .filter_map(|material| material.gear_dye_palette)
                .collect_vec();
            if expects_attached_dyes {
                assert!(
                    !dye_palettes.is_empty(),
                    "{name}: attached mods need skin dyes"
                );
                assert!(
                    !gear_dye_palettes.is_empty(),
                    "{name}: attached mods need the full skin palette"
                );
            }
            assert!(gear_dye_palettes.iter().all(|palette| {
                palette.iter().all(|dye| {
                    dye.color[..3]
                        .iter()
                        .all(|channel| channel.is_finite() && (0.0..=4.0).contains(channel))
                })
            }));
            assert!(
                gear_dye_palettes
                    .iter()
                    .all(|palette| *palette == gear_dye_palettes[0]),
                "{name} applied different synthesized palettes to its attachments"
            );
            for expected in expected_dye_colors {
                assert!(
                    gear_dye_palettes.iter().any(|palette| {
                        palette.iter().any(|dye| {
                            dye.color
                                .into_iter()
                                .zip(expected)
                                .all(|(actual, expected)| (actual - expected).abs() < 0.000_01)
                        })
                    }),
                    "{name} lost authored skin color {expected:?} while applying it to mods"
                );
            }
            let mut resources = CallbackResources::default();
            let descriptor = ScreenDescriptor {
                size_in_pixels: size,
                pixels_per_point: 1.0,
            };
            let mut encoder =
                render_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("weapon_mod_probe"),
                    });
            callback.prepare(
                &render_state.device,
                &render_state.queue,
                &descriptor,
                &mut encoder,
                &mut resources,
            );
            render_state.queue.submit(Some(encoder.finish()));
            for _ in 0..2 {
                let mut reuse_encoder =
                    render_state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("weapon_mod_probe_reused_frame"),
                        });
                callback.prepare(
                    &render_state.device,
                    &render_state.queue,
                    &descriptor,
                    &mut reuse_encoder,
                    &mut resources,
                );
                render_state.queue.submit(Some(reuse_encoder.finish()));
            }
            let pipelines = resources
                .get::<ModelPipelineResources>()
                .expect("model pipelines");
            let frame = resources.get::<ModelFrameResources>().expect("model frame");
            let presented = render_state
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("weapon_mod_probe_presented"),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: target_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
            let presented_view = presented.create_view(&Default::default());
            let unpadded_bytes_per_row = size[0] * 4;
            let bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
            let buffer = render_state.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("weapon_mod_probe_readback"),
                size: u64::from(bytes_per_row) * u64::from(size[1]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                render_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("weapon_mod_probe_copy"),
                    });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("weapon_mod_probe_present"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &presented_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
                presented.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
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
            render_state
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("GPU wait");
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            render_state
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("map wait");
            let mapped = slice.get_mapped_range();
            let mut pixels = Vec::with_capacity(
                (u64::from(unpadded_bytes_per_row) * u64::from(size[1])) as usize,
            );
            for row in mapped.chunks_exact(bytes_per_row as usize) {
                pixels.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
            }
            drop(mapped);
            buffer.unmap();
            if matches!(
                target_format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            ) {
                for pixel in pixels.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
            }
            let image =
                image::RgbaImage::from_raw(size[0], size[1], pixels).expect("readback image");
            let background = image.get_pixel(0, 0).0[..3].to_vec();
            let visible = image
                .pixels()
                .filter(|pixel| {
                    pixel.0[..3]
                        .iter()
                        .zip(&background)
                        .any(|(value, background)| value.abs_diff(*background) > 8)
                })
                .count();
            assert!(visible > 5_000, "render must be nonblank");
            if lighting_reference_case {
                let suffix = std::env::var("QUICKTAG_PROBE_OUTPUT_SUFFIX")
                    .ok()
                    .filter(|value| !value.is_empty())
                    .map(|value| format!("-{value}"))
                    .unwrap_or_default();
                image
                    .save(output.join(format!("{name}-{diagnostic_name}{suffix}-untested.png")))
                    .expect("save lighting tuning render before acceptance checks");
            }
            if lighting_reference_case && diagnostic_pass == 0 {
                let profile = implementation_material_bands(&image);
                eprintln!("Vox Nocturna material bands: {profile:?}");
                if !tuning_probe {
                    assert!(
                        (8.0..=18.5).contains(&profile.luma[0]),
                        "deepest cavities must remain in the 8–18 target band"
                    );
                    assert!(
                        profile.black_fraction >= 0.08,
                        "at least 8% of visible model must retain 16–40 black-polymer luminance"
                    );
                    assert!(
                        (135.0..=185.0).contains(&profile.luma[6]),
                        "99th-percentile highlight must remain in the pale-rail target band"
                    );
                    assert!(
                        profile.clipped_fraction <= 0.005,
                        "no more than 0.5% of model pixels may exceed sRGB 210"
                    );
                    assert!(
                        profile.purple_pixels > profile.foreground / 200,
                        "dusty purple material must remain visible"
                    );
                    assert!(
                        profile.cream_pixels > profile.foreground / 200,
                        "cream rail/decals must remain visible"
                    );
                    assert!(
                        profile.green_pixels > 10 && profile.green_max <= 185,
                        "green internal component must stay saturated without exceeding 185"
                    );
                }
            }
            if name == "dont-let-up-brrt-darksight-precision" && diagnostic_pass == 0 {
                let (mut dark, mut yellow) = (0usize, 0usize);
                for y in 130..230 {
                    for x in 120..270 {
                        let [red, green, blue, _alpha] = image.get_pixel(x, y).0;
                        let max = red.max(green).max(blue);
                        let min = red.min(green).min(blue);
                        if (30..180).contains(&max) && max.saturating_sub(min) < 28 {
                            dark += 1;
                        }
                        if red > 140
                            && green > 120
                            && blue < 190
                            && red.saturating_sub(blue) > 45
                            && green.saturating_sub(blue) > 30
                        {
                            yellow += 1;
                        }
                    }
                }
                eprintln!("Darksight material regions: neutral={dark}, yellow={yellow}");
                if dark <= 1_000 || yellow <= 150 || dark <= yellow {
                    visual_failures.push(format!(
                        "Darksight must keep a predominantly neutral body with localized yellow accents; neutral={dark}, yellow={yellow}"
                    ));
                }
            }
            if name == "atrax-sting-v11-rangefinder-suppression" {
                let (mut neutral, mut green) = (0usize, 0usize);
                // Fixed-camera crop containing only the two attached mods. The
                // engine's packed material-ID map must preserve their neutral
                // housings while selecting Atrax's authored green region.
                for y in 160..450 {
                    for x in 90..285 {
                        let [red, green_channel, blue, _alpha] = image.get_pixel(x, y).0;
                        let max = red.max(green_channel).max(blue);
                        let min = red.min(green_channel).min(blue);
                        if (45..170).contains(&max) && max.saturating_sub(min) < 30 {
                            neutral += 1;
                        }
                        if green_channel > 55
                            && green_channel as u16 * 100 > red as u16 * 135
                            && green_channel as u16 * 100 > blue as u16 * 120
                        {
                            green += 1;
                        }
                    }
                }
                assert!(
                    neutral > 10_000 && green > 300,
                    "Atrax mods must keep neutral housings with a localized green region; neutral={neutral}, green={green}"
                );

                // Keep these values as diagnostics. Absolute apparent
                // brightness depends on the game's camera and lighting, so it
                // is not an engine-level attachment-palette invariant.
                let barrel = dominant_neutral_rgb(&image, 95..205, 170..235);
                let optic = dominant_neutral_rgb(&image, 435..530, 55..95);
                let body = dominant_neutral_rgb(&image, 300..550, 95..185);
                let barrel_distance = rgb_distance(barrel, body);
                let optic_distance = rgb_distance(optic, body);
                eprintln!(
                    "Atrax neutral similarity: barrel={barrel:?}, optic={optic:?}, body={body:?}, distances=({barrel_distance:.2}, {optic_distance:.2})"
                );
            }
            if name == "midnight-decay-misriah-precision-quickdraw-slick" {
                // The supplied in-game reference has nearly identical neutral
                // front-body/front-choke samples (#171A18 and #181818), while
                // the rear choke sits about 27 luma levels above the body. Use
                // relative colors so the harness remains independent of the
                // game's exposure and screenshot compression.
                let body = dominant_neutral_rgb(&image, 240..520, 235..285);
                let front_choke = dominant_neutral_rgb(&image, 95..170, 290..350);
                let rear_choke = dominant_neutral_rgb(&image, 170..238, 290..350);
                let quickdraw_grip = dominant_neutral_rgb(&image, 240..405, 295..330);
                let slick_mag = dominant_neutral_rgb(&image, 340..475, 226..260);
                let front_distance = rgb_distance(front_choke, body);
                let luma = |color: [f32; 3]| color.into_iter().sum::<f32>() / 3.0;
                let front_luma_delta = (luma(front_choke) - luma(body)).abs();
                let rear_luma_delta = (luma(rear_choke) - luma(body)).abs();
                let grip_luma_delta = luma(quickdraw_grip) - luma(body);
                let mag_luma_delta = luma(slick_mag) - luma(body);
                eprintln!(
                    "Midnight Decay neutral verification: body={body:?}, front_choke={front_choke:?}, rear_choke={rear_choke:?}, quickdraw_grip={quickdraw_grip:?}, slick_mag={slick_mag:?}, front_distance={front_distance:.2}, front_luma_delta={front_luma_delta:.2}, rear_luma_delta={rear_luma_delta:.2}, grip_luma_delta={grip_luma_delta:.2}, mag_luma_delta={mag_luma_delta:.2}"
                );
                if front_luma_delta > 18.0 || !(15.0..=40.0).contains(&rear_luma_delta) {
                    visual_failures.push(format!(
                        "Midnight Decay mod colors diverge from the reference relationship: body={body:?}, front choke={front_choke:?} (RGB distance {front_distance:.2}, luma delta {front_luma_delta:.2}), rear choke={rear_choke:?} (luma delta {rear_luma_delta:.2})"
                    ));
                }
            }
            if flat_panel_reference_case {
                let crop_x = 260..510;
                let crop_y = 205..380;
                let (orange, orange_count) =
                    dominant_orange_rgb(&image, crop_x.clone(), crop_y.clone());
                let target = [177.0, 87.0, 36.0];
                let distance = rgb_distance(orange, target);
                let white_count = crop_y
                    .flat_map(|row| crop_x.clone().map(move |column| (column, row)))
                    .filter(|(column, row)| {
                        let [red, green, blue, _alpha] = image.get_pixel(*column, *row).0;
                        red.min(green).min(blue) > 210
                            && red.max(green).max(blue) - red.min(green).min(blue) < 24
                    })
                    .count();
                eprintln!(
                    "V85 Vox Nocturna panel: rendered={orange:.1?}, target=#B15724, distance={distance:.1}, orange={orange_count}, white={white_count}"
                );
                assert!(
                    orange_count > 1_000 && distance < 80.0,
                    "procedural panel must render near annotated orange #B15724"
                );
                assert!(
                    white_count < orange_count / 3,
                    "procedural panel regressed to broad white fallback"
                );
            }
            if investment_decal_reference_case {
                let mut stencil_pixels = 0usize;
                let mut dark_panel_pixels = 0usize;
                // Fixed-camera crop around the M77 lower-frame decal. The
                // authored atlas contains sparse white marks over the dark
                // frame; a missing atlas turns nearly this whole quad white.
                for y in 390..411 {
                    for x in 515..536 {
                        let [red, green, blue, _alpha] = image.get_pixel(x, y).0;
                        let max = red.max(green).max(blue);
                        let min = red.min(green).min(blue);
                        if red > 110 && green > 110 && blue > 110 && max - min < 40 {
                            stencil_pixels += 1;
                        }
                        if (26..105).contains(&max) && max - min < 45 {
                            dark_panel_pixels += 1;
                        }
                    }
                }
                eprintln!(
                    "M77 investment decal: stencil={stencil_pixels}, dark panel={dark_panel_pixels}"
                );
                assert!(
                    (8..=100).contains(&stencil_pixels) && dark_panel_pixels > 300,
                    "M77 decal must remain a sparse stencil over the dark frame, not a solid white fallback"
                );
            }
            if name == "syntax-disrupt-v75-decal" {
                let mut authored_blue = 0usize;
                let mut white_fallback = 0usize;
                // Fixed-camera crop over the textureless TFX panel. Its
                // channel-zero engine state is saturated blue/cyan; the old
                // missing-material path rendered this entire region white.
                for y in 232..254 {
                    for x in 475..496 {
                        let [red, green, blue, _alpha] = image.get_pixel(x, y).0;
                        if blue > 120 && green > 80 && red < 100 {
                            authored_blue += 1;
                        }
                        if red > 220 && green > 220 && blue > 220 {
                            white_fallback += 1;
                        }
                    }
                }
                let mut dark_atlas_marks = 0usize;
                let mut white_atlas_marks = 0usize;
                // The large CyAc lettering is geometry painted by the sole
                // sRGB atlas bound to its opaque PS t0. Generic atlas
                // filtering used to discard it and leave white geometry.
                for y in 266..440 {
                    for x in 373..454 {
                        let [red, green, blue, _alpha] = image.get_pixel(x, y).0;
                        let max = red.max(green).max(blue);
                        let min = red.min(green).min(blue);
                        if (85..130).contains(&max) && max - min < 12 {
                            dark_atlas_marks += 1;
                        }
                        if min > 220 && max - min < 24 {
                            white_atlas_marks += 1;
                        }
                    }
                }
                eprintln!(
                    "Syntax Disrupt materials: TFX authored={authored_blue}, TFX white={white_fallback}, atlas dark={dark_atlas_marks}, atlas white={white_atlas_marks}"
                );
                assert!(
                    authored_blue > 300 && white_fallback < 10,
                    "textureless TFX panel must use its authored object-channel colour, not white fallback"
                );
                assert!(
                    dark_atlas_marks > 500 && white_atlas_marks < 10,
                    "opaque sole-atlas lettering must stay dark instead of falling back to white"
                );
            }
            if name == "arata-vectus-v66-detail" {
                let mut red = 0usize;
                let mut contour = 0usize;
                let mut transitions = 0usize;
                let classify = |pixel: &image::Rgba<u8>| {
                    let [red, green, blue, _alpha] = pixel.0;
                    let max = red.max(green).max(blue);
                    let min = red.min(green).min(blue);
                    if min > 155 && max - min < 55 {
                        2u8
                    } else if red > 95
                        && red.saturating_sub(green) > 50
                        && red.saturating_sub(blue) > 40
                    {
                        1u8
                    } else {
                        0u8
                    }
                };
                // Front red receiver only. Authored contour alternates red and
                // pale strokes densely; missing shader branch leaves nearly
                // the whole crop red except sparse ordinary decals/struts.
                for y in 245..435 {
                    let mut previous = 0u8;
                    for x in 155..395 {
                        let class = classify(image.get_pixel(x, y));
                        red += usize::from(class == 1);
                        contour += usize::from(class == 2);
                        transitions +=
                            usize::from(previous != 0 && class != 0 && previous != class);
                        previous = class;
                    }
                }
                eprintln!(
                    "Arata procedural contour: red={red}, pale={contour}, transitions={transitions}"
                );
                assert!(
                    red > 4_000 && contour > 2_000 && transitions > 250,
                    "Arata front receiver must retain dense pale contour strokes over red base"
                );
            }
            if matches!(
                name,
                "runner-achromatic-rush-decal" | "runner-detail-selector-decal"
            ) {
                let (bright_pixels, largest_component) = bright_component_profile(&image);
                eprintln!(
                    "Runner investment decals: bright={bright_pixels}, largest component={largest_component}"
                );
                assert!(
                    bright_pixels
                        > if name == "runner-achromatic-rush-decal" {
                            500
                        } else {
                            100
                        }
                        && largest_component < 2_000,
                    "runner decals must preserve authored marks without returning solid white atlas quads"
                );
            }
            let suffix = std::env::var("QUICKTAG_PROBE_OUTPUT_SUFFIX")
                .ok()
                .filter(|value| !value.is_empty())
                .map(|value| format!("-{value}"))
                .unwrap_or_default();
            let output_stem = if diagnostic_name != "final" {
                format!("{name}-{diagnostic_name}")
            } else {
                name.to_string()
            };
            let output_path = output.join(format!("{output_stem}{suffix}.png"));
            image.save(&output_path).expect("save render");
            if let Some(baseline_path) = std::env::var_os("QUICKTAG_PROBE_BASELINE") {
                let report_path = output.join(format!("{output_stem}{suffix}.metrics.json"));
                verify_visual_baseline(&image, Path::new(&baseline_path), &report_path);
            }
            renders.push((name, image));
        }
        let rarity_render = |rarity: &str| {
            renders
                .iter()
                .find(|(name, _)| *name == format!("retro-remix-misriah-precision-{rarity}"))
                .map(|(_, image)| image)
        };
        if let (Some(clean), Some(enhanced), Some(deluxe), Some(superior)) = (
            rarity_render("clean"),
            rarity_render("enhanced"),
            rarity_render("deluxe"),
            rarity_render("superior"),
        ) {
            let mean_pixel_delta = |left: &image::RgbaImage, right: &image::RgbaImage| {
                left.pixels()
                    .zip(right.pixels())
                    .map(|(left, right)| {
                        left.0[..3]
                            .iter()
                            .zip(&right.0[..3])
                            .map(|(left, right)| left.abs_diff(*right) as f32)
                            .sum::<f32>()
                            / 3.0
                    })
                    .sum::<f32>()
                    / (left.width() * left.height()) as f32
            };
            let deluxe_to_superior = mean_pixel_delta(deluxe, superior);
            let enhanced_to_clean = mean_pixel_delta(enhanced, superior);
            let changed_pixels = |left: &image::RgbaImage, right: &image::RgbaImage| {
                left.pixels()
                    .zip(right.pixels())
                    .filter(|(left, right)| {
                        left.0[..3]
                            .iter()
                            .zip(&right.0[..3])
                            .any(|(left, right)| left.abs_diff(*right) > 3)
                    })
                    .count()
            };
            let deluxe_changed = changed_pixels(deluxe, superior);
            let enhanced_changed = changed_pixels(enhanced, superior);
            eprintln!(
                "Retro_Remix Precision Choke condition deltas: Deluxe/Superior={deluxe_to_superior:.3} ({deluxe_changed} px), Enhanced/Superior={enhanced_to_clean:.3} ({enhanced_changed} px)"
            );
            assert!(
                mean_pixel_delta(clean, superior) <= f32::EPSILON,
                "Superior TFX outputs are zero, so it must match no-condition clean render exactly"
            );
            assert!(
                deluxe_to_superior > 0.01 && deluxe_changed > 100,
                "Deluxe must reveal intermediate spatial wear instead of sharing Superior clean material"
            );
            assert!(
                enhanced_to_clean > deluxe_to_superior && enhanced_changed > deluxe_changed,
                "Enhanced must expose more authored wear than Deluxe; deltas {enhanced_to_clean:.3}/{deluxe_to_superior:.3}, changed {enhanced_changed}/{deluxe_changed}"
            );
        }
        // Some Windows drivers block while tearing down a test-only device after mapped readback.
        std::mem::forget(texture_cache);
        std::mem::forget(render_state);
        assert!(visual_failures.is_empty(), "{}", visual_failures.join("\n"));

        for (reference_env, render_name, slug, title) in [
            (
                "QUICKTAG_YOKAIS_CLAW_REFERENCE",
                "yokais-claw-misriah-full",
                "yokais-claw",
                "Yokai's Claw",
            ),
            (
                "QUICKTAG_DONT_LET_UP_REFERENCE",
                "dont-let-up-brrt-darksight-precision",
                "dont-let-up-brrt",
                "BRRT SMG: Don't let up.",
            ),
            (
                "QUICKTAG_ATRAX_STING_REFERENCE",
                "atrax-sting-v11-rangefinder-suppression",
                "atrax-sting-v11",
                "V11 Punch: Atrax Sting",
            ),
            (
                "QUICKTAG_MIDNIGHT_DECAY_REFERENCE",
                "midnight-decay-misriah-precision-quickdraw-slick",
                "midnight-decay-misriah",
                "Misriah 2442: Midnight Decay",
            ),
            (
                "QUICKTAG_PRECISION_CHOKE_SUPERIOR_REFERENCE",
                "retro-remix-misriah-precision-superior",
                "retro-remix-misriah-precision-superior",
                "Misriah 2442: Retro_Remix — Superior Precision Choke",
            ),
            (
                "QUICKTAG_PRECISION_CHOKE_ENHANCED_REFERENCE",
                "retro-remix-misriah-precision-enhanced",
                "retro-remix-misriah-precision-enhanced",
                "Misriah 2442: Retro_Remix — Enhanced Precision Choke",
            ),
            (
                "QUICKTAG_VOX_NOCTURNA_LIGHTING_REFERENCE",
                "vox-nocturna-misriah-ingame-lighting",
                "vox-nocturna-misriah-ingame-lighting",
                "Misriah 2442: Vox Nocturna — in-game lighting",
            ),
        ] {
            if render_name == "vox-nocturna-misriah-ingame-lighting"
                && (diagnostic_pass != 0 || tuning_probe)
            {
                continue;
            }
            let Ok(reference_path) = std::env::var(reference_env).map(PathBuf::from) else {
                continue;
            };
            let mut reference = image::open(&reference_path)
                .unwrap_or_else(|error| panic!("reference {}: {error}", reference_path.display()))
                .to_rgba8();
            // RENDER.md's supplied capture is a vertical before/after panel:
            // game reference above, old Quicktag output below. Compare only
            // the upper source panel when that composite aspect is detected.
            if slug == "vox-nocturna-misriah-ingame-lighting"
                && reference.height() > reference.width() / 2
            {
                reference = image::imageops::crop_imm(
                    &reference,
                    0,
                    0,
                    reference.width(),
                    reference.height() / 2,
                )
                .to_image();
            }
            let reference_name = format!("{slug}-image-1-in-game-reference.png");
            let implementation_name = format!("{slug}-image-2-quicktag-implementation.png");
            reference
                .save(output.join(&reference_name))
                .expect("save normalized reference");
            let implementation = renders
                .iter()
                .find(|(name, _image)| *name == render_name)
                .unwrap_or_else(|| panic!("missing {render_name} comparison render"))
                .1
                .clone();
            let implementation =
                crop_render_to_reference(&implementation, [reference.width(), reference.height()]);
            implementation
                .save(output.join(&implementation_name))
                .expect("save normalized implementation");
            if slug == "vox-nocturna-misriah-ingame-lighting" {
                let (similarity, channels, bias, reference_pixels, implementation_pixels) =
                    foreground_visual_similarity(&reference, &implementation);
                eprintln!(
                    "Vox Nocturna in-game lighting similarity: {similarity:.2}% RGB/luma/chroma={channels:.2?} bias={bias:.2?} (foreground {reference_pixels}/{implementation_pixels} px)"
                );
                assert!(
                    similarity >= 95.0,
                    "Vox Nocturna lighting similarity {similarity:.2}% is below 95%"
                );
            }
            std::fs::write(
                output.join(format!("{slug}-comparison.html")),
                format!(
                    r#"<!doctype html>
<meta charset="utf-8">
<title>{title} mod skin verification</title>
<style>
body {{ margin: 0; background: #0f1520; color: #e8edf5; font: 16px system-ui, sans-serif; }}
main {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }}
figure {{ margin: 0; padding: 12px; background: #182131; border: 1px solid #35435a; }}
figcaption {{ margin-bottom: 10px; font-weight: 700; }}
img {{ display: block; width: 100%; height: auto; image-rendering: auto; }}
</style>
<main>
  <figure><figcaption>Image 1 — In-game reference: {title}</figcaption><img src="{reference_name}"></figure>
  <figure><figcaption>Quicktag implementation — fixed camera, authored mod dyes</figcaption><img src="{implementation_name}"></figure>
</main>
"#
                ),
            )
            .expect("write labeled visual comparison");
        }

        // One supplied screenshot may contain all three engine tiers. Split it
        // into labeled panels so texture shape/placement can be checked beside
        // fixed-camera Quicktag output without conflating whole-image exposure.
        if let Ok(reference_path) =
            std::env::var("QUICKTAG_PRECISION_CHOKE_TIERS_REFERENCE").map(PathBuf::from)
        {
            let reference = image::open(&reference_path)
                .unwrap_or_else(|error| panic!("reference {}: {error}", reference_path.display()))
                .to_rgba8();
            let boundaries = [
                0,
                reference.width() / 3,
                reference.width() * 2 / 3,
                reference.width(),
            ];
            let mut figures = String::new();
            for (index, tier) in ["enhanced", "deluxe", "superior"].into_iter().enumerate() {
                let width = boundaries[index + 1] - boundaries[index];
                let panel = image::imageops::crop_imm(
                    &reference,
                    boundaries[index],
                    0,
                    width,
                    reference.height(),
                )
                .to_image();
                let reference_name = format!("precision-choke-{tier}-in-game.png");
                panel
                    .save(output.join(&reference_name))
                    .expect("save tier reference panel");

                let render_name = format!("retro-remix-misriah-precision-{tier}");
                let implementation = renders
                    .iter()
                    .find(|(name, _image)| *name == render_name)
                    .unwrap_or_else(|| panic!("missing {render_name} comparison render"));
                let implementation =
                    crop_render_to_reference(&implementation.1, [width, reference.height()]);
                let implementation_name = format!("precision-choke-{tier}-quicktag.png");
                implementation
                    .save(output.join(&implementation_name))
                    .expect("save tier implementation panel");
                figures.push_str(&format!(
                    "<figure><figcaption>In game — {tier}</figcaption><img src=\"{reference_name}\"></figure><figure><figcaption>Quicktag — {tier}</figcaption><img src=\"{implementation_name}\"></figure>"
                ));
            }
            std::fs::write(
                output.join("precision-choke-three-tier-comparison.html"),
                format!(
                    r#"<!doctype html>
<meta charset="utf-8">
<title>Precision Choke three-tier wear verification</title>
<style>
body {{ margin: 0; background: #0f1520; color: #e8edf5; font: 16px system-ui, sans-serif; }}
main {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px; }}
figure {{ margin: 0; padding: 10px; background: #182131; border: 1px solid #35435a; }}
figcaption {{ margin-bottom: 8px; font-weight: 700; text-transform: capitalize; }}
img {{ display: block; width: 100%; height: auto; }}
</style>
<main>{figures}</main>"#
                ),
            )
            .expect("write three-tier comparison");
        }
    }
}

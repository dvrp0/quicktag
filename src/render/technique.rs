use tiger_pkg::{TagHash, package_manager};

use crate::render::tfx::{TfxExecutionResult, TfxRuntimeInputs, execute_preview};
use crate::{
    geometry::WireframePreview,
    material::{
        MaterialPreviewKind, MaterialTagPreview, TechniquePreview, TechniqueRenderState,
        TfxBytecodePreview, WideHashPreview,
    },
    texture::{Texture, TextureDesc},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Geometry,
    Pixel,
    Compute,
    Unknown,
}

impl ShaderStage {
    pub fn from_label(label: &str) -> Self {
        match label {
            "VS" => Self::Vertex,
            "GS" => Self::Geometry,
            "PS" => Self::Pixel,
            "CS" => Self::Compute,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimensionAbi {
    D2,
    D2Array,
    D3,
    CubeCandidate,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpaceIntent {
    Linear,
    Srgb,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct TextureAbi {
    pub dimension: TextureDimensionAbi,
    pub format: String,
    pub color_space: ColorSpaceIntent,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
}

impl TextureAbi {
    fn from_desc(desc: TextureDesc) -> Self {
        let format = format!("{:?}", desc.format);
        let color_space = if format.ends_with("Srgb") {
            ColorSpaceIntent::Srgb
        } else {
            // Storage format is known, authored shader intent is not. Linear
            // formats may still contain color sampled through an sRGB view.
            ColorSpaceIntent::Unknown
        };
        let dimension = if desc.depth > 1 {
            TextureDimensionAbi::D3
        } else if desc.array_size == 6 {
            TextureDimensionAbi::CubeCandidate
        } else if desc.array_size > 1 {
            TextureDimensionAbi::D2Array
        } else {
            TextureDimensionAbi::D2
        };
        Self {
            dimension,
            format,
            color_space,
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
            array_size: desc.array_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceBindingDescriptor {
    pub slot: u32,
    pub raw: WideHashPreview,
    pub resolved: Option<TagHash>,
    pub texture_abi: Option<TextureAbi>,
    pub required: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct SamplerBindingDescriptor {
    pub ordinal: u32,
    pub raw: WideHashPreview,
    pub resolved: Option<TagHash>,
}

#[derive(Debug, Clone)]
pub struct ShaderSignature {
    pub shader: Option<TagHash>,
    pub texture_count: u32,
    pub sampler_count: u32,
    pub constant_count: u32,
    pub inline_constant_count: u32,
    pub tfx_byte_count: u32,
}

#[derive(Debug, Clone)]
pub struct TechniqueStageDescriptor {
    pub stage: ShaderStage,
    pub raw_stage_label: &'static str,
    pub shader: Option<TagHash>,
    pub signature: ShaderSignature,
    pub resources: Vec<ResourceBindingDescriptor>,
    pub samplers: Vec<SamplerBindingDescriptor>,
    pub constants: Vec<[f32; 4]>,
    pub inline_constants: Vec<[f32; 4]>,
    pub constant_buffer_slot: Option<i32>,
    pub constant_buffer: Option<TagHash>,
    pub tfx: TfxBytecodePreview,
    pub tfx_execution: TfxExecutionResult,
}

#[derive(Debug, Clone)]
pub struct TechniqueDescriptor {
    pub technique_hash: TagHash,
    pub bind_mode: u32,
    pub render_state: TechniqueRenderState,
    pub raw_used_scopes: u64,
    pub raw_compatible_scopes: u64,
    pub stages: Vec<TechniqueStageDescriptor>,
}

impl TechniqueDescriptor {
    pub fn load(tag: TagHash) -> Option<Self> {
        let entry = package_manager().get_entry(tag)?;
        let data = package_manager().read_tag(tag).ok()?;
        let preview = MaterialTagPreview::load(&entry, &data)?;
        let MaterialPreviewKind::Technique(preview) = preview.kind;
        Some(Self::from_preview(tag, preview))
    }

    pub fn from_preview(tag: TagHash, preview: TechniquePreview) -> Self {
        let render_state = TechniqueRenderState::from_raw(preview.state_selection);
        let stages = preview
            .stages
            .into_iter()
            .map(|stage| {
                let resources = stage
                    .textures
                    .iter()
                    .map(|binding| {
                        let resolved = binding.texture.resolved.or_else(|| {
                            binding
                                .texture
                                .raw32
                                .is_some()
                                .then_some(binding.texture.raw32)
                        });
                        let texture_abi = resolved
                            .and_then(|texture| Texture::load_desc(texture).ok())
                            .map(TextureAbi::from_desc);
                        ResourceBindingDescriptor {
                            slot: binding.slot,
                            raw: binding.texture,
                            resolved,
                            texture_abi,
                            required: None,
                        }
                    })
                    .collect::<Vec<_>>();
                let samplers = stage
                    .samplers
                    .iter()
                    .enumerate()
                    .map(|(ordinal, sampler)| SamplerBindingDescriptor {
                        ordinal: ordinal as u32,
                        raw: *sampler,
                        resolved: sampler
                            .resolved
                            .or_else(|| sampler.raw32.is_some().then_some(sampler.raw32)),
                    })
                    .collect::<Vec<_>>();
                let signature = ShaderSignature {
                    shader: stage.shader,
                    texture_count: resources.len() as u32,
                    sampler_count: samplers.len() as u32,
                    constant_count: stage.constants.len() as u32,
                    inline_constant_count: stage.inline_constants.len() as u32,
                    tfx_byte_count: stage.bytecode_len as u32,
                };
                let tfx_execution = execute_preview(
                    &stage.bytecode,
                    &stage.constants,
                    &TfxRuntimeInputs::default(),
                );
                TechniqueStageDescriptor {
                    stage: ShaderStage::from_label(stage.stage),
                    raw_stage_label: stage.stage,
                    shader: stage.shader,
                    signature,
                    resources,
                    samplers,
                    constants: stage.constants,
                    inline_constants: stage.inline_constants,
                    constant_buffer_slot: stage.constant_buffer_slot,
                    constant_buffer: stage.constant_buffer,
                    tfx: stage.bytecode,
                    tfx_execution,
                }
            })
            .collect();
        Self {
            technique_hash: tag,
            bind_mode: preview.bind_mode,
            render_state,
            raw_used_scopes: preview.used_scopes,
            raw_compatible_scopes: preview.compatible_scopes,
            stages,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexAttributeAbi {
    pub semantic: &'static str,
    pub format: String,
    pub decoded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexAbiDescriptor {
    pub attributes: Vec<VertexAttributeAbi>,
    pub has_skinning: bool,
    pub has_morph_or_soft_deformation: bool,
    pub unsupported_requirements: Vec<String>,
}

impl VertexAbiDescriptor {
    pub fn from_wireframe(wireframe: &WireframePreview) -> Self {
        let mut attributes = vec![VertexAttributeAbi {
            semantic: "POSITION0",
            format: wireframe.position_format.to_string(),
            decoded: true,
        }];
        for (semantic, format, decoded) in [
            (
                "NORMAL0",
                wireframe.normal_format.clone(),
                wireframe.normals.is_some(),
            ),
            (
                "TANGENT0",
                wireframe.tangent_format.clone(),
                wireframe.tangents.is_some(),
            ),
            (
                "TEXCOORD0",
                wireframe.uv_format.clone(),
                wireframe.uvs.is_some(),
            ),
        ] {
            if let Some(format) = format {
                attributes.push(VertexAttributeAbi {
                    semantic,
                    format,
                    decoded,
                });
            }
        }
        let has_skinning = wireframe.position_format.contains("CPU bind-pose skinning");
        let mut unsupported_requirements = vec![
            "additional UV/color streams not yet preserved".into(),
            "morph/soft-deformation streams not yet preserved".into(),
        ];
        if !has_skinning {
            unsupported_requirements.push("bone indices/weights not yet preserved".into());
        }
        Self {
            attributes,
            has_skinning,
            has_morph_or_soft_deformation: false,
            unsupported_requirements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_labels_are_typed_without_losing_unknowns() {
        assert_eq!(ShaderStage::from_label("PS"), ShaderStage::Pixel);
        assert_eq!(ShaderStage::from_label("MS"), ShaderStage::Unknown);
    }
}

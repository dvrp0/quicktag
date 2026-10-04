use tiger_pkg::{TagHash, package_manager};

use crate::render::tfx::{
    TfxExecutionResult, TfxRuntimeBinding, TfxRuntimeInputs, TfxValue, execute_preview,
};
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
pub struct IndexedResourceDescriptor {
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
    pub indexed_resources: Vec<IndexedResourceDescriptor>,
    pub constants: Vec<[f32; 4]>,
    pub inline_constants: Vec<[f32; 4]>,
    pub external_constants: Option<Vec<[f32; 4]>>,
    pub constant_buffer_slot: Option<i32>,
    pub constant_buffer: Option<TagHash>,
    pub tfx: TfxBytecodePreview,
    pub texture_metadata: std::collections::BTreeMap<u8, crate::texture::TextureExpressionMetadata>,
    pub tfx_execution: TfxExecutionResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TechniqueRuntimeBinding {
    pub kind: &'static str,
    pub stage: ShaderStage,
    pub slot: u8,
    pub source: String,
    pub resolved: Option<TagHash>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TechniqueStageRuntimeState {
    pub stage: ShaderStage,
    pub constant_buffer_slot: Option<i32>,
    pub constant_registers: Vec<[f32; 4]>,
    pub bindings: Vec<TechniqueRuntimeBinding>,
    pub unresolved_dependencies: Vec<String>,
    pub status: crate::material::TfxDecodeStatus,
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

impl TechniqueStageDescriptor {
    pub fn execute(&self, inputs: &TfxRuntimeInputs) -> TfxExecutionResult {
        execute_preview(&self.tfx, &self.constants, inputs, &self.texture_metadata)
    }

    pub fn runtime_state(&self, inputs: &TfxRuntimeInputs) -> TechniqueStageRuntimeState {
        let execution = self.execute(inputs);
        // An authored external buffer replaces inline data, as in Tiger's
        // dynamic-constant initialization. TFX writes apply afterward.
        let mut constant_registers = if self.constant_buffer.is_some() {
            self.external_constants.clone().unwrap_or_default()
        } else {
            self.inline_constants.clone()
        };
        for (target, value) in &execution.outputs {
            let Some(index) = target
                .strip_prefix("output[")
                .and_then(|target| target.strip_suffix(']'))
                .and_then(|index| index.parse::<usize>().ok())
            else {
                continue;
            };
            let TfxValue::Vector(value) = value else {
                continue;
            };
            if constant_registers.len() <= index {
                constant_registers.resize(index + 1, [0.0; 4]);
            }
            constant_registers[index] = *value;
        }

        let mut bindings = self
            .resources
            .iter()
            .filter_map(|resource| {
                Some(TechniqueRuntimeBinding {
                    kind: "texture",
                    stage: self.stage,
                    slot: u8::try_from(resource.slot).ok()?,
                    source: "authored_resource".into(),
                    resolved: resource.resolved,
                })
            })
            .collect::<Vec<_>>();
        for binding in &execution.bindings {
            let stage = ShaderStage::from_label(binding.stage);
            let resolved = resolve_tfx_runtime_binding(self, binding, inputs);
            if let Some(existing) = bindings
                .iter_mut()
                .find(|existing| existing.kind == binding.kind && existing.slot == binding.slot)
            {
                *existing = TechniqueRuntimeBinding {
                    kind: binding.kind,
                    stage,
                    slot: binding.slot,
                    source: binding.source.clone(),
                    resolved,
                };
            } else {
                bindings.push(TechniqueRuntimeBinding {
                    kind: binding.kind,
                    stage,
                    slot: binding.slot,
                    source: binding.source.clone(),
                    resolved,
                });
            }
        }
        bindings.sort_by_key(|binding| (binding.stage as u8, binding.slot));

        let mut unresolved_dependencies = execution
            .dependencies
            .iter()
            .filter(|dependency| !dependency.resolved)
            .map(|dependency| {
                format!(
                    "{}+0x{:X} at byte {}",
                    dependency.scope, dependency.byte_offset, dependency.byte_offset
                )
            })
            .collect::<Vec<_>>();
        unresolved_dependencies.extend(
            bindings
                .iter()
                .filter(|binding| binding.resolved.is_none())
                .map(|binding| {
                    format!(
                        "{} {:?} slot {} <- {}",
                        binding.kind, binding.stage, binding.slot, binding.source
                    )
                }),
        );
        if self.constant_buffer.is_some() && self.external_constants.is_none() {
            unresolved_dependencies.push("authored constant buffer unavailable".into());
        }
        unresolved_dependencies.extend(execution.outputs.iter().filter_map(|(target, value)| {
            (target.starts_with("output[") && !matches!(value, TfxValue::Vector(_)))
                .then(|| format!("{} <- unresolved numeric value", target))
        }));
        unresolved_dependencies.sort();
        unresolved_dependencies.dedup();

        TechniqueStageRuntimeState {
            stage: self.stage,
            constant_buffer_slot: self.constant_buffer_slot,
            constant_registers,
            bindings,
            unresolved_dependencies,
            status: execution.status,
        }
    }
}

fn resolve_tfx_runtime_binding(
    stage: &TechniqueStageDescriptor,
    binding: &TfxRuntimeBinding,
    inputs: &TfxRuntimeInputs,
) -> Option<TagHash> {
    if binding.kind == "sampler" {
        let index = binding
            .source
            .strip_prefix("sampler[")?
            .strip_suffix(']')?
            .parse::<usize>()
            .ok()?;
        return stage.indexed_resources.get(index).and_then(|sampler| {
            sampler
                .resolved
                .or_else(|| sampler.raw.raw32.is_some().then_some(sampler.raw.raw32))
        });
    }

    for prefix in ["extern_texture(", "extern_uav(", "extern_resource("] {
        let Some(inner) = binding
            .source
            .strip_prefix(prefix)
            .and_then(|source| source.strip_suffix(')'))
        else {
            continue;
        };
        let (scope, offset) = inner.split_once("+0x")?;
        let offset = u32::from_str_radix(offset, 16).ok()?;
        return inputs
            .extern_resources
            .get(&(scope.to_string(), offset))
            .copied();
    }

    None
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
                let indexed_resources = stage
                    .indexed_resources
                    .iter()
                    .enumerate()
                    .map(|(ordinal, sampler)| IndexedResourceDescriptor {
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
                    sampler_count: stage.bytecode.bindings.iter().filter(|b|b.kind=="sampler").count() as u32,
                    constant_count: stage.constants.len() as u32,
                    inline_constant_count: stage.inline_constants.len() as u32,
                    tfx_byte_count: stage.bytecode_len as u32,
                };
                let texture_metadata = stage.bytecode.ops.iter().filter(|op| {
                    matches!(op.name,"push_tex_tiling_params"|"push_tex_tile_layer_count")
                }).filter_map(|op| {
                    let index=op.detail.split_once(" fields=0x")?.0.strip_prefix("index=")?.parse::<u8>().ok()?;
                    let resource=indexed_resources.get(usize::from(index))?.resolved?;
                    let metadata=Texture::validated_descriptor_d2(resource).ok()?.expression_metadata?;
                    Some((index,metadata))
                }).collect();
                let tfx_execution = execute_preview(
                    &stage.bytecode,
                    &stage.constants,
                    &TfxRuntimeInputs::default(),
                    &texture_metadata,
                );
                TechniqueStageDescriptor {
                    stage: ShaderStage::from_label(stage.stage),
                    raw_stage_label: stage.stage,
                    shader: stage.shader,
                    signature,
                    resources,
                    indexed_resources,
                    constants: stage.constants,
                    inline_constants: stage.inline_constants,
                    external_constants: stage.constant_buffer_preview.map(|buffer| buffer.values),
                    constant_buffer_slot: stage.constant_buffer_slot,
                    constant_buffer: stage.constant_buffer,
                    tfx: stage.bytecode,
                    texture_metadata,
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
            "additional UV/color attributes are not decoded into the visible vertex ABI".into(),
            "morph/soft-deformation behavior is not implemented in the visible vertex ABI".into(),
        ];
        if !has_skinning {
            unsupported_requirements
                .push("bone indices/weights are not decoded into the visible vertex ABI".into());
        }
        Self {
            attributes,
            has_skinning,
            has_morph_or_soft_deformation: false,
            unsupported_requirements,
        }
    }
}

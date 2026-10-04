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

    pub fn runtime_states(&self, inputs: &TfxRuntimeInputs) -> Vec<TechniqueStageRuntimeState> {
        self.stages
            .iter()
            .map(|stage| stage.runtime_state(inputs))
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Marathon packages; focused CPU numeric-origin probe, no GPU"]
    fn probes_remaining_runner_numeric_origins() {
        let manager=tiger_pkg::PackageManager::new(
            r"D:\SteamLibrary\steamapps\common\Marathon\packages",
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),None).unwrap();
        tiger_pkg::initialize_package_manager(&std::sync::Arc::new(manager));
        quicktag_core::classes::initialize_reference_names();
        let cache=quicktag_scanner::load_tag_cache();
        let candidates=[
            (0x80A9ADA4,0x80A9ADE1,0x80A9ADD9,vec![28]),
            (0x80A9E079,0x80A9E0F6,0x80A9E0EE,vec![24]),
            (0x80B143C3,0x80B143B3,0x80B143AC,vec![5,93,110]),
            (0x80B1454D,0x80B145C2,0x80B145BA,vec![5]),
        ];
        let mut records=Vec::new();
        for (root,technique,shader,rows) in candidates {
            let descriptor=TechniqueDescriptor::load(TagHash(technique)).unwrap();
            let stage=descriptor.stages.iter().find(|stage|stage.stage==ShaderStage::Pixel).unwrap();
            assert_eq!(stage.shader,Some(TagHash(shader)));
            let inputs=TfxRuntimeInputs::for_model_preview(&cache,TagHash(root));
            let execution=stage.execute(&inputs);
            let selected=rows.into_iter().map(|row| {
                let target=format!("output[{row}]");
                serde_json::json!({"row":row,"value":format!("{:?}",execution.outputs.get(&target)),
                    "source_expression":stage.tfx.expressions.iter().filter(|e|e.target==target)
                        .map(|e|e.expression.clone()).collect::<Vec<_>>()})
            }).collect::<Vec<_>>();
            records.push(serde_json::json!({"root":format!("{root:08X}"),"technique":format!("{technique:08X}"),
                "shader":format!("{shader:08X}"),"rows":selected,
                "ops":stage.tfx.ops.iter().map(|op|serde_json::json!({"name":op.name,"detail":op.detail})).collect::<Vec<_>>(),
                "indexed_resources":stage.indexed_resources.iter().map(|resource|resource.resolved.map(|tag|tag.to_string())).collect::<Vec<_>>(),
                "texture_metadata_indices":stage.texture_metadata.keys().collect::<Vec<_>>() }));
        }
        let output=std::path::Path::new("target/runner-view-owner-audit/nonview-numeric-source.json");
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        std::fs::write(output,serde_json::to_vec_pretty(&records).unwrap()).unwrap();
        eprintln!("NUMERIC_ORIGINS {}",output.display());
    }

    #[test]
    #[ignore = "requires installed Marathon packages; CPU metadata/TFX probe only"]
    fn probes_marathon_indexed_texture_metadata() {
        let manager=tiger_pkg::PackageManager::new(
            r"D:\SteamLibrary\steamapps\common\Marathon\packages",
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),None).unwrap();
        tiger_pkg::initialize_package_manager(&std::sync::Arc::new(manager));
        quicktag_core::classes::initialize_reference_names();
        let technique=TechniqueDescriptor::load(TagHash(0x80B15819)).unwrap();
        let stage=technique.stages.iter().find(|stage|stage.stage==ShaderStage::Pixel).unwrap();
        assert_eq!(stage.indexed_resources.len(),4);
        assert_eq!(stage.indexed_resources[0].resolved,Some(TagHash(0x80A60020)));
        for index in 1..=3 {
            assert_eq!(stage.indexed_resources[index].resolved,Some(TagHash(0x80B15810)));
            let meta=stage.texture_metadata.get(&(index as u8)).unwrap();
            assert_eq!(meta.tiling_params.map(f32::to_bits),[0x3eaaaaab,0x3ea957fb,0x3eaaaaab,0x3de38e39]);
            assert_eq!(meta.tile_count,9);
        }
        let mut inputs=TfxRuntimeInputs::default();
        let initial=stage.execute(&inputs);
        assert_eq!(initial.outputs["output[4]"],TfxValue::Vector([f32::from_bits(0x3eaaaaab),f32::from_bits(0x3ea957fb),0.0,0.0]));
        inputs.time_seconds=-1.0;
        let phase=((0.555_f32*0.5)*9.0).fract();
        assert_eq!(stage.execute(&inputs).outputs["output[4]"],TfxValue::Vector([
            f32::from_bits(0x3eaaaaab),f32::from_bits(0x3ea957fb),f32::from_bits(0x3eaaaaab)*phase,f32::from_bits(0x3de38e39)*phase]));
        let mut missing=stage.clone();missing.texture_metadata.remove(&2);
        assert!(matches!(missing.execute(&inputs).outputs["output[4]"],TfxValue::Unknown(_)));
        assert!(stage.texture_metadata[&3].evaluate("push_tex_tile_layer_count",0x55).is_none(),"unproved layer-count lanes stay unresolved");
        assert_eq!(stage.runtime_state(&inputs).bindings.iter().find(|b|b.kind=="texture"&&b.slot==3).unwrap().resolved,Some(TagHash(0x80B15810)));
    }

    #[test]
    #[ignore = "requires Marathon packages and render_globals_probe named pipeline export"]
    fn exports_marathon_material_consumer_bindings() {
        export_named_consumer_bindings(
            &["debug_specular_smoothness", "debug_world_normal", "debug_texture_ao",
              "debug_emissive", "debug_emissive_intensity", "debug_transmission",
              "debug_colored_overcoat_id", "debug_source_color", "debug_metalness"],
            "debug-consumer-bindings.json", 18,
        );
    }

    #[test]
    #[ignore = "requires Marathon packages and render_globals_probe named pipeline export"]
    fn exports_marathon_lighting_consumer_bindings() {
        export_named_consumer_bindings(
            &["global_lighting", "global_lighting_and_shading", "deferred_shading",
              "deferred_shading_no_atm", "global_lighting_and_shading_gel"],
            "lighting-consumer-bindings.json", 10,
        );
    }

    fn export_named_consumer_bindings(names: &[&str], artifact: &str, expected: usize) {
        use sha2::{Digest, Sha256};
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into());
        let manager = std::sync::Arc::new(PackageManager::new(
            packages, GameVersion::Marathon(MarathonVersion::Marathon), None).unwrap());
        tiger_pkg::initialize_package_manager(&manager);
        quicktag_core::classes::initialize_reference_names();
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
            "target/cryo-runtime-audit/render-globals/named-pipelines.json").unwrap()).unwrap();
        let mut reports = Vec::new();
        for globals in manifest["globals"].as_array().unwrap() {
            for array in globals["arrays"].as_array().unwrap() {
                for record in array["records"].as_array().unwrap() {
                    let name = record["name"].as_str().unwrap();
                    if !names.contains(&name) { continue; }
                    assert_eq!(record["target_status"], "verified-target-class");
                    let tag = TagHash(u32::from_str_radix(record["tag"].as_str().unwrap(), 16).unwrap());
                    let bytes = manager.read_tag(tag).unwrap();
                    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
                    let descriptor = TechniqueDescriptor::load(tag).unwrap();
                    let ps = descriptor.stages.iter().find(|s| s.stage == ShaderStage::Pixel).unwrap();
                    let unresolved = ps.runtime_state(&TfxRuntimeInputs::default());
                    let sampler_resources: Vec<_> = ps.indexed_resources.iter().map(|s|
                        serde_json::json!({"ordinal":s.ordinal,"tag":s.resolved.map(|v|v.to_string())})).collect();
                    let bindings: Vec<_> = ps.tfx.bindings.iter().map(|b| serde_json::json!({
                        "kind":b.kind,"stage":b.stage,"slot":b.slot,"source":b.source
                    })).collect();
                    let operations: Vec<_> = ps.tfx.ops.iter().map(|op| serde_json::json!({
                        "offset":op.offset,"opcode":op.opcode,"name":op.name,"detail":op.detail,"extern_scope_id":op.extern_scope_id
                    })).collect();
                    let externs: Vec<_> = ps.tfx.externs.iter().map(|e| serde_json::json!({
                        "offset":e.op_offset,"scope":e.scope,"scope_id":e.scope_id,
                        "value_type":e.value_type,"byte_offset":e.byte_offset,"hint":e.hint
                    })).collect();
                    let expressions: Vec<_> = ps.tfx.expressions.iter().map(|e| serde_json::json!({
                        "offset":e.op_offset,"target":e.target,"expression":e.expression,"value":e.value
                    })).collect();
                    reports.push(serde_json::json!({"globals":globals["tag"],"name":name,"technique":tag.to_string(),
                        "technique_sha256":record["sha256"],"shader":ps.shader.map(|s|s.to_string()),
                        "source_shaders":record["shaders"],"raw_used_scopes":descriptor.raw_used_scopes,
                        "raw_compatible_scopes":descriptor.raw_compatible_scopes,
                        "tfx_status":format!("{:?}",ps.tfx.status),"tfx_unknown_ops":ps.tfx.unknown_ops,
                        "tfx_byte_count":ps.tfx.total_bytes,"bindings":bindings,"operations":operations,
                        "externs":externs,"expressions":expressions,"inline_constants":ps.inline_constants,
                        "external_constants":ps.external_constants,"constant_buffer_slot":ps.constant_buffer_slot,
                        "constant_buffer":ps.constant_buffer.map(|t|t.to_string()),
                        "sampler_resources":sampler_resources,
                        "unresolved_with_empty_runtime":unresolved.unresolved_dependencies,
                        "constants_with_empty_runtime":unresolved.constant_registers
                    }));
                }
            }
        }
        assert_eq!(reports.len(), expected);
        std::fs::write(format!("target/cryo-runtime-audit/render-globals/{artifact}"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema":"quicktag-marathon-material-consumer-bindings-v1","consumers":reports,
                "limits":"Actual packaged TFX sources and program identity. Extern offsets are not resolved native resources; active attachment formats, native values and shader execution remain separate proofs."
            })).unwrap()).unwrap();
    }

    fn constant_stage() -> TechniqueStageDescriptor {
        let tfx = TfxBytecodePreview::default();
        TechniqueStageDescriptor {
            stage: ShaderStage::Pixel,
            raw_stage_label: "PS",
            shader: None,
            signature: ShaderSignature {
                shader: None,
                texture_count: 0,
                sampler_count: 0,
                constant_count: 0,
                inline_constant_count: 1,
                tfx_byte_count: 0,
            },
            resources: vec![],
            indexed_resources: vec![],
            constants: vec![],
            inline_constants: vec![[99.0; 4]],
            external_constants: None,
            constant_buffer_slot: Some(0),
            constant_buffer: None,
            tfx_execution: execute_preview(&tfx, &[], &TfxRuntimeInputs::default(), &Default::default()),
            texture_metadata: Default::default(),
            tfx,
        }
    }

    #[test]
    fn authored_external_constants_replace_inline_without_truncation() {
        let mut stage = constant_stage();
        stage.constant_buffer = Some(TagHash(0x80A9C827));
        let values = (0..12).map(|i| [i as f32; 4]).collect::<Vec<_>>();
        stage.external_constants = Some(values.clone());
        let state = stage.runtime_state(&TfxRuntimeInputs::default());
        assert_eq!(state.constant_registers, values);
        assert!(state.unresolved_dependencies.is_empty());
    }

    #[test]
    fn missing_authored_buffer_is_reported_without_inline_substitution() {
        let mut stage = constant_stage();
        stage.constant_buffer = Some(TagHash(0x80A9C827));
        let state = stage.runtime_state(&TfxRuntimeInputs::default());
        assert!(state.constant_registers.is_empty());
        assert_eq!(
            state.unresolved_dependencies,
            ["authored constant buffer unavailable"]
        );
        stage.constant_buffer = None;
        assert_eq!(
            stage
                .runtime_state(&TfxRuntimeInputs::default())
                .constant_registers,
            vec![[99.0; 4]]
        );
    }

    #[test]
    fn tfx_overlays_external_registers_and_reports_unresolved_outputs() {
        use crate::material::TfxBytecodeOpPreview;
        let mut stage = constant_stage();
        stage.constant_buffer = Some(TagHash(0x80A9C827));
        stage.external_constants = Some(vec![[3.0; 4]; 9]);
        stage.tfx.ops = vec![
            TfxBytecodeOpPreview {
                offset: 0,
                opcode: 0x4a,
                name: "push_extern_float",
                detail: "Frame+0x0".into(),
                extern_scope_id: Some(0),
            },
            TfxBytecodeOpPreview {
                offset: 3,
                opcode: 0x53,
                name: "pop_output",
                detail: "element=8".into(),
                extern_scope_id: None,
            },
        ];
        let state = stage.runtime_state(&TfxRuntimeInputs {
            time_seconds: 2.5,
            ..Default::default()
        });
        assert_eq!(state.constant_registers[0], [3.0; 4]);
        assert_eq!(state.constant_registers[8], [2.5; 4]);
        stage.tfx.ops[0].detail = "Frame+0x40".into();
        let state = stage.runtime_state(&TfxRuntimeInputs::default());
        assert_eq!(state.constant_registers[8], [3.0; 4]);
        assert!(
            state
                .unresolved_dependencies
                .iter()
                .any(|dependency| dependency.starts_with("output[8] <-"))
        );
    }

    #[test]
    fn stage_labels_are_typed_without_losing_unknowns() {
        assert_eq!(ShaderStage::from_label("PS"), ShaderStage::Pixel);
        assert_eq!(ShaderStage::from_label("MS"), ShaderStage::Unknown);
    }
}

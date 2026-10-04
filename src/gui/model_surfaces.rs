//! Authored source-color material admission and explicit viewer scopes.
//! Native runtime values and lit composition remain separate fidelity gates.
use super::{
    AuthoredStageMetadata, ModelPipelineKey, ModelSamplerDesc, SceneUniform, ShaderStage, TagHash,
    TechniqueDescriptor, Texture, load_model_sampler_desc, native_view_image,
};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct NativeSurfaceSource {
    pub(super) vertex_program: crate::render::authored_program::DescriptorAbi,
    pub(super) program: crate::render::authored_program::DescriptorAbi,
    pub(super) constants: Vec<[f32; 4]>,
    pub(super) vertex_constants: Option<Vec<[f32; 4]>>,
    pub(super) textures: Vec<TagHash>,
    pub(super) samplers: Vec<ModelSamplerDesc>,
    pub(super) metadata_rows: Vec<(usize, Option<usize>)>,
}

pub(super) struct LoadedNativeSurface {
    pub(super) stable_index: usize,
    pub(super) textures: Vec<Arc<Texture>>,
}

/// A source shader contract is insufficient without a valid static input image.
/// Withhold that draw before GPU geometry construction, rather than rejecting
/// every otherwise renderable part of the model.
pub(super) fn packed_static_source_valid(source: &crate::geometry::AuthoredGeometryInput) -> bool {
    if source.attachment_pose.is_some() || source.position_transform.is_none() || source.uv_transform.is_none() {
        return false;
    }
    let Some(stream)=source.vertex_streams.iter().find(|v|v.stream_index==0 && v.stride==24) else{return false;};
    let Some(palette)=source.skinning_buffer.as_ref() else{return false;};
    if !source.vertex_streams.iter().any(|v|v.stream_index==1 && v.stride==4 && v.element_count>=stream.element_count) {
        return false;
    }
    let Ok(raw)=tiger_pkg::package_manager().read_tag(stream.data_tag) else{return false;};
    let Ok(bytes)=tiger_pkg::package_manager().read_tag(palette.data_tag) else{return false;};
    bytes.len()==palette.data_size as usize && !bytes.is_empty() && bytes.len()%4==0
        && raw.len()==stream.element_count as usize*24
        && crate::render::static_mesh::hair_static_inputs_valid(&raw,bytes.len())
}

pub(super) fn vertex_color_source_valid(source: &crate::geometry::AuthoredGeometryInput,
    stage: AuthoredStageMetadata) -> bool {
    use crate::render::authored_program::{DescriptorAbi as D, resolve_package_program};
    if !packed_static_source_valid(source) { return false; }
    let Some(color) = source.color_buffer.as_ref() else { return false; };
    if color.stride != 4 || color.vertex_type != 5 || color.element_count == 0
        || color.element_count.checked_mul(4) != Some(color.data_size) { return false; }
    let Ok(bytes) = tiger_pkg::package_manager().read_tag(color.data_tag) else { return false; };
    if crate::render::color_vertex::decode(&bytes, color.element_count).is_none() { return false; }
    let Some(end) = stage.source_index_start.checked_add(stage.source_index_count) else { return false; };
    let Some(tag) = crate::geometry::geometry_compute_technique(source.geometry, stage.part_index,
        stage.source_index_start..end) else { return false; };
    let Some(technique) = TechniqueDescriptor::load(tag) else { return false; };
    technique.stages.iter().find(|s| s.stage == ShaderStage::Compute).and_then(|cs| cs.shader)
        .and_then(|tag| resolve_package_program(tag, ShaderStage::Compute).ok().flatten())
        .is_some_and(|p| matches!(p.descriptor_abi,D::BodyMeshB4CBComputeStorage
            | D::Cloth45RowComputeStorage | D::BodyMeshAA060BComputeStorage))
}

/// Source materials accept only independently audited VS/PS pairs.
/// An unresolved metadata row is safe only behind its proved zero output gate.
/// This does not resolve the game's metadata selector or inherited draw state.
pub(super) fn resolve_source(
    technique: Option<&TechniqueDescriptor>,
    authored: Option<AuthoredStageMetadata>,
    pipeline: ModelPipelineKey,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Result<NativeSurfaceSource, &'static str> {
    use crate::render::authored_program::{DescriptorAbi as D, resolve_package_program};
    let authored = authored.ok_or("missing authored stage")?;
    if !matches!((authored.input_layout_id, authored.primitive_type), (7,5) | (13,3))
        || authored.raw_stage != 0
        || pipeline.blend != 0
        || pipeline.depth_stencil != 2
        || pipeline.depth_bias != 0
        || !matches!(pipeline.rasterizer, 1 | 2)
    {
        return Err("unsupported opaque stage/layout/topology/render state");
    }
    let technique = technique.ok_or("missing technique")?;
    let vs = technique
        .stages
        .iter()
        .find(|s| s.stage == ShaderStage::Vertex).ok_or("missing vertex stage")?;
    let ps = technique
        .stages
        .iter()
        .find(|s| s.stage == ShaderStage::Pixel).ok_or("missing pixel stage")?;
    let vertex_program = resolve_package_program(vs.shader.ok_or("missing vertex shader")?, ShaderStage::Vertex)
        .map_err(|_|"vertex source identity failed")?.ok_or("unregistered vertex program")?.descriptor_abi;
    if !matches!(vertex_program, D::BodyVertexScalarStorage | D::RigidSignVertexScalarStorage
        | D::EyesVertexStorage | D::RunnerVertexScalarStorage | D::RigidVertexDirectIa | D::FloatVertexScalarStorage
        | D::HairVertexStorage | D::Hair14580VertexStorage | D::Hair120RowVertexStorage | D::HairBA67VertexStorage
        | D::BodyProceduralVertexStorage | D::VertexColorStorage | D::ClothVertexStorage | D::ClothB152VertexStorage
        | D::DisplacementEC03VertexStorage | D::BodyC9C5VertexStorage) {
        return Err("unsupported vertex runtime ABI");
    }
    if matches!(vertex_program,D::RigidVertexDirectIa|D::FloatVertexScalarStorage) != (authored.input_layout_id == 13) {
        return Err("vertex source does not match the authored IA/topology contract");
    }
    let program = resolve_package_program(ps.shader.ok_or("missing pixel shader")?, ShaderStage::Pixel)
        .map_err(|_|"pixel source identity failed")?.ok_or("unregistered pixel program")?
        .descriptor_abi;
    let required_vertex = match program {
        D::RunnerSurface(index) => crate::render::runner_surface_programs::SURFACES.get(index as usize).ok_or("missing surface contract")?.vertex_programs,
        D::HairPixelDense | D::HairSolidPixelDense => &[D::HairVertexStorage],
        _ => &[D::BodyVertexScalarStorage],
    };
    if !required_vertex.contains(&vertex_program) { return Err("vertex/pixel contract mismatch"); }
    let (rows, count, volume, metadata_rows, unresolved_dependencies, sampler_count) = match program {
        D::RunnerSurface(index) => {
            let contract=crate::render::runner_surface_programs::SURFACES.get(index as usize).ok_or("missing surface contract")?;
            (contract.rows,contract.texture_count,contract.volume_slot,contract.metadata_rows.to_vec(),contract.unresolved_dependencies.iter().map(|s|s.to_string()).collect::<Vec<_>>(),contract.sampler_count)
        },
        D::BodyPixelDense => (125, 8, Some(7), vec![(75, Some(84))], vec!["output[75] <- unresolved numeric value".to_string()], 3),
        D::ChestPixelDense => (129, 9, Some(8), vec![(76, Some(85))], vec!["output[76] <- unresolved numeric value".to_string()], 3),
        D::SleevesPixelDense => (129, 9, Some(8), vec![(79, Some(88))], vec!["output[79] <- unresolved numeric value".to_string()], 3),
        D::HandsPixelDense => (138, 9, Some(7), vec![(81, Some(90))], vec!["output[81] <- unresolved numeric value".to_string()], 3),
        D::HardwarePixelDense => (117, 7, Some(6), vec![(70, Some(79))], vec!["output[70] <- unresolved numeric value".to_string()], 3),
        D::FacePixelDense => (127, 7, Some(5), vec![(73, Some(82))], vec!["output[73] <- unresolved numeric value".to_string()], 3),
        D::EyeDetailPixelDense => (71, 2, Some(1), vec![(40, Some(49))], vec!["output[40] <- unresolved numeric value".to_string()], 2),
        // Original A9D5 row133 affects RT0/RT1 only through row142.x. The
        // source-proved zero gate is required for every admitted technique.
        D::HairPixelDense => (167, 5, Some(4), vec![(133, Some(142))], vec!["output[133] <- unresolved numeric value".to_string()], 2),
        D::HairSolidPixelDense => (170, 6, Some(5), vec![(136, Some(145))], vec!["output[136] <- unresolved numeric value".to_string()], 2),
        _ => return Err("unsupported pixel runtime ABI"),
    };
    let state = ps.runtime_state(runtime_inputs);
    if ps.constant_buffer_slot != Some(0) { return Err("unexpected constant buffer slot"); }
    if state.constant_registers.len() != rows { return Err("constant row count mismatch"); }
    let mut constants = state.constant_registers;
    if state.unresolved_dependencies != unresolved_dependencies {
        #[cfg(test)]
        eprintln!("Surface dependency contract rejected shader={:?} actual={:?} audited={:?}",
            ps.shader, state.unresolved_dependencies, unresolved_dependencies);
        return Err("unresolved runtime dependencies differ from audited source proof");
    }
    for &(unresolved_row, gate) in &metadata_rows {
        if unresolved_row >= rows || gate.is_some_and(|g| g >= rows || metadata_rows.iter().any(|&(row,_)| row == g)) {
            return Err("invalid metadata proof contract");
        }
        if gate.is_some_and(|g| constants[g][0] != 0.0) {
            return Err("metadata zero gate is active");
        }
        // All four components independently proved output/control-flow inert.
        constants[unresolved_row] = [0.0; 4];
    }
    let vertex_constants = if let Some((written_rows, declared_rows)) = vertex_program.vertex_constant_contract() {
        let mut state = vs.runtime_state(runtime_inputs);
        if vs.constant_buffer_slot != Some(0) || state.constant_registers.len() != written_rows
            || !state.unresolved_dependencies.is_empty()
        {
            return Err("unresolved auxiliary vertex constant image");
        }
        // Exact source contracts prove the declared tail unused. Preserve
        // every authored write; pad only the source-unused declaration range.
        state.constant_registers.resize(declared_rows, [0.0; 4]);
        Some(state.constant_registers)
    } else {
        None
    };
    let textures = (0..count)
        .map(|slot| {
            let binding = state
                .bindings
                .iter()
                .find(|b| b.kind == "texture" && u32::from(b.slot) == slot)?;
            let tag = binding.resolved?;
            let desc = Texture::load_desc(tag).ok()?;
            if desc.array_size != 1 || (desc.depth > 1) != (Some(slot) == volume) {
                return None;
            }
            Some(tag)
        })
        .collect::<Option<Vec<_>>>().ok_or("missing/unresolved texture or incompatible texture dimensions")?;
    let samplers = (1..=sampler_count as u8)
        .map(|slot| {
            let binding = state
                .bindings
                .iter()
                .find(|b| b.kind == "sampler" && b.slot == slot)?;
            let desc = load_model_sampler_desc(binding.resolved?)?;
            // Exact supported filter/address contract; no viewer 16x upgrade and
            // no address-mode approximation. The adapter already applies -0.5.
            if !matches!(desc.filter, 0 | 0x15 | 0x55)
                || (desc.filter == 0 && desc.max_anisotropy != 1)
                || desc.mip_lod_bias != -0.5
                || ![desc.address_u, desc.address_v, desc.address_w]
                    .iter()
                    .all(|v| matches!(v, 1..=4))
                || super::authored_sampler_border_color(&desc).is_err()
                || desc.min_lod != 0.0
                || desc.max_lod < 32.0
            {
                return None;
            }
            Some(desc)
        })
        .collect::<Option<Vec<_>>>().ok_or("missing/unresolved sampler or unsupported sampler contract")?;
    Ok(NativeSurfaceSource {
        vertex_program,
        program,
        constants,
        vertex_constants,
        textures,
        samplers,
        metadata_rows,
    })
}

pub(super) fn pixel_view_image(scene: &SceneUniform) -> [[f32; 4]; 29] {
    let vertex = native_view_image(scene);
    let mut view = [[0.0; 4]; 29];
    for start in [0, 20] {
        view[start..start + 3].copy_from_slice(&vertex[..3]);
        view[start + 3] = vertex[19];
    }
    let distance = 5.0 * scene.params0[0].max(0.0001);
    let (sy, cy) = scene.params0[1].sin_cos();
    let (sp, cp) = scene.params0[2].sin_cos();
    view[7] = [
        scene.center[0] - sy * cp * distance,
        scene.center[1] + cy * cp * distance,
        scene.center[2] + sp * distance,
        1.0,
    ];
    view
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    #[ignore = "requires installed Marathon packages"]
    fn audits_cross_skin_authored_material_admission() {
        let manager = tiger_pkg::PackageManager::new(
            std::env::var("QUICKTAG_MARATHON_PACKAGES").unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into()),
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon), None,
        ).expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(manager));
        quicktag_core::classes::initialize_reference_names();
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        let output = std::path::PathBuf::from(std::env::var("QUICKTAG_MATERIAL_PROBE_OUTPUT")
            .unwrap_or_else(|_|"target/cross-skin-audit".into()));
        std::fs::create_dir_all(&output).unwrap();
        let mut reports = vec![];
        let cases=if let Ok(root)=std::env::var("QUICKTAG_MATERIAL_PROBE_ROOT") {
            root.split(',').map(|value|("requested-model",u32::from_str_radix(value.trim().trim_start_matches("0x"),16).expect("hexadecimal Pattern root"))).collect::<Vec<_>>()
        }else{vec![
            ("thief-cryo-shift", 0x80A9D134),
            ("assassin-digital-prowl", 0x80A9DC79),
            ("assassin-cryo-shift", 0x80A9DD33),
            ("vandal-weaverunner", 0x80A9DAD3),
            ("destroyer-cyber-red", 0x80A9D869),
        ]};
        for (name, root) in cases {
            let root = TagHash(root);
            let shell = crate::geometry::RunnerShellAssembly::resolve(&cache, root).expect("shell");
            let preview = shell.load(cache.clone()).expect("preview");
            let crate::geometry::GeometryPreviewKind::Model(model) = preview.kind else { panic!("model"); };
            let wireframe = model.wireframe.unwrap();
            let inputs = crate::render::tfx::TfxRuntimeInputs::for_model_preview(&cache, root);
            // Export the normal loader's authoritative buffer references. A
            // float4 shader SRV alone does not establish its package view format.
            let authored_sources = wireframe.authored_inputs.iter().map(|input| {
                let color = input.color_buffer.as_ref().map(|buffer| {
                    let header = tiger_pkg::package_manager().read_tag(buffer.header_tag).unwrap();
                    let data = tiger_pkg::package_manager().read_tag(buffer.data_tag).unwrap();
                    assert_eq!(data.len(), buffer.data_size as usize);
                    let artifact = format!("{}-color-{}.bin", input.geometry, buffer.data_tag);
                    std::fs::write(output.join(&artifact), &data).unwrap();
                    serde_json::json!({
                        "header":buffer.header_tag.to_string(), "data":buffer.data_tag.to_string(),
                        "header_bytes":header, "stride":buffer.stride, "vertex_type":buffer.vertex_type,
                        "element_count":buffer.element_count, "bytes":data.len(),
                        "sha256":format!("{:x}",Sha256::digest(&data)), "artifact":artifact,
                    })
                });
                serde_json::json!({"geometry":input.geometry.to_string(),"color_buffer":color})
            }).collect::<Vec<_>>();
            let mut draws = vec![];
            for (index, draw) in super::super::model_draws(&wireframe, None).iter().enumerate() {
                let geometry = draw.authored_source.and_then(|i|wireframe.authored_inputs.get(i)).map(|g|g.geometry);
                let scoped_inputs = geometry.map(|g|inputs.for_geometry(g)).unwrap_or_else(||inputs.clone());
                let mut stages = vec![];
                if let Some(technique) = &draw.packet.technique {
                    for stage in &technique.stages {
                        let Some(shader) = stage.shader else { continue; };
                        let entry = tiger_pkg::package_manager().get_entry(shader).unwrap();
                        let payload = tiger_pkg::package_manager().read_tag(TagHash(entry.reference)).unwrap();
                        let digest = format!("{:x}", Sha256::digest(&payload));
                        std::fs::write(output.join(format!("{shader}-{:?}.dxil", stage.stage)), &payload).unwrap();
                        let state = stage.runtime_state(&scoped_inputs);
                        stages.push(serde_json::json!({
                            "stage":format!("{:?}",stage.stage), "shader":shader.to_string(),
                            "payload_sha256":digest,
                            "registered":crate::render::authored_program::resolve_package_program(shader,stage.stage).unwrap().map(|p|format!("{:?}",p.descriptor_abi)),
                            "constant_slot":stage.constant_buffer_slot, "rows":state.constant_registers,
                            "unresolved":state.unresolved_dependencies,
                            "tfx_ops":stage.tfx.ops.iter().map(|op|serde_json::json!({"offset":op.offset,"opcode":op.opcode,"name":op.name,"detail":op.detail})).collect::<Vec<_>>(),
                            "tfx_constants":stage.constants,
                            "numeric_origins":stage.execute(&scoped_inputs).outputs.iter().filter(|(target,_)|target.starts_with("output[")).map(|(target,value)|serde_json::json!({"target":target,"value":format!("{value:?}")})).collect::<Vec<_>>(),
                            "bindings":state.bindings.iter().map(|b|serde_json::json!({"kind":b.kind,"slot":b.slot,"source":b.source,"resolved":b.resolved.map(|t|t.to_string())})).collect::<Vec<_>>(),
                        }));
                    }
                }
                let source = resolve_source(draw.packet.technique.as_ref(),draw.authored_stage,draw.pipeline,&scoped_inputs);
                let producer = geometry.zip(draw.authored_stage).and_then(|(geometry,stage)| {
                    let tag=crate::geometry::geometry_compute_technique(geometry,stage.part_index,stage.source_index_start..stage.source_index_start.checked_add(stage.source_index_count)?)?;
                    let technique=TechniqueDescriptor::load(tag)?;
                    let cs=technique.stages.iter().find(|s|s.stage==ShaderStage::Compute)?;
                    let shader=cs.shader?;
                    let program=crate::render::authored_program::resolve_package_program(shader,ShaderStage::Compute).ok()?;
                    Some(serde_json::json!({"technique":tag.to_string(),"shader":shader.to_string(),"registered":program.map(|p|format!("{:?}",p.descriptor_abi))}))
                });
                draws.push(serde_json::json!({
                    "index":index,"technique":draw.packet.technique_hash.map(|t|t.to_string()),
                    "geometry":geometry.map(|g|g.to_string()),
                    "pattern_path":shell.parts.iter().find(|p|geometry.is_some_and(|g|p.geometry.contains(&g))).map(|p|p.pattern_path.iter().map(|t|t.to_string()).collect::<Vec<_>>()),
                    "compute_producer":producer,
                    "authored":format!("{:?}",draw.authored_stage),"pipeline":format!("{:?}",draw.pipeline),
                    "native_admitted":source.ok().map(|s|format!("{:?}",s.program)),"stages":stages,
                }));
            }
            eprintln!("{name}: {} draws, {} admitted", draws.len(),draws.iter().filter(|d|!d["native_admitted"].is_null()).count());
            reports.push(serde_json::json!({"name":name,"root":root.to_string(),"draws":draws,"authored_sources":authored_sources,
                "object_scopes":crate::geometry::pattern_object_channel_evidence(&cache,root)}));
        }
        std::fs::write(output.join("material-admission.json"),serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
    }
}

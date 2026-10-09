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
    pub(super) animation: Option<AnimatedConstants>,
    pub(super) vertex_constants: Option<Vec<[f32; 4]>>,
    pub(super) textures: Vec<TagHash>,
    pub(super) cube_slot: Option<u32>,
    pub(super) samplers: Vec<ModelSamplerDesc>,
}

/// Pixel constants whose TFX program reads the Frame clock. The renderer
/// re-runs the program each frame instead of freezing them at load.
#[derive(Clone)]
pub(super) struct AnimatedConstants {
    stage: Arc<crate::render::technique::TechniqueStageDescriptor>,
    inputs: Arc<crate::render::tfx::TfxRuntimeInputs>,
    allowed: fn(&str) -> bool,
}

impl AnimatedConstants {
    pub(super) fn new(
        stage: &crate::render::technique::TechniqueStageDescriptor,
        inputs: &crate::render::tfx::TfxRuntimeInputs,
        allowed: fn(&str) -> bool,
    ) -> Option<Self> {
        // Frame+0x0 is game time and Frame+0x4 render time.
        stage.tfx.externs.iter()
            .any(|external| external.scope == "Frame" && matches!(external.byte_offset, 0 | 4))
            .then(|| Self { stage: Arc::new(stage.clone()), inputs: Arc::new(inputs.clone()), allowed })
    }

    pub(super) fn at(&self, time_seconds: f32) -> Option<Vec<[f32; 4]>> {
        let mut inputs = (*self.inputs).clone();
        inputs.time_seconds = time_seconds;
        let state = self.stage.runtime_state(&inputs);
        let mut constants = state.constant_registers;
        zero_unresolved_rows(&mut constants, &state.unresolved_dependencies, self.allowed).ok()?;
        Some(constants)
    }
}

/// Constant rows whose TFX expression cannot be evaluated are written as zero
/// for every program alike. Any other unresolved input must be one the caller
/// supplies itself.
pub(super) fn zero_unresolved_rows(
    constants: &mut [[f32; 4]],
    unresolved: &[String],
    allowed: fn(&str) -> bool,
) -> Result<(), &'static str> {
    for dependency in unresolved {
        let row = dependency.strip_prefix("output[").and_then(|rest| rest.split_once(']'))
            .and_then(|(row, _)| row.parse::<usize>().ok());
        match row {
            Some(row) if row < constants.len() => constants[row] = [0.0; 4],
            _ if allowed(dependency) => {}
            _ => return Err("unresolved runtime input is not a constant row"),
        }
    }
    Ok(())
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

/// Source materials accept only registered VS/PS pairs.
/// Constant rows whose TFX expression cannot be evaluated are zero for every
/// program alike. This does not resolve inherited draw state.
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
        | D::DisplacementEC03VertexStorage | D::ShellOffsetD8B4VertexStorage | D::ShellOffsetC7F6VertexStorage | D::BodyC9C5VertexStorage) {
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
    let (rows, count, volume, cube, sampler_count) = match program {
        D::RunnerSurface(index) => {
            let contract=crate::render::runner_surface_programs::SURFACES.get(index as usize).ok_or("missing surface contract")?;
            (contract.rows,contract.texture_count,contract.volume_slot,contract.cube_slot,contract.sampler_count)
        },
        D::BodyPixelDense => (125, 8, Some(7), None, 3),
        D::ChestPixelDense => (129, 9, Some(8), None, 3),
        D::SleevesPixelDense => (129, 9, Some(8), None, 3),
        D::HandsPixelDense => (138, 9, Some(7), None, 3),
        D::HardwarePixelDense => (117, 7, Some(6), None, 3),
        D::FacePixelDense => (127, 7, Some(5), None, 3),
        D::EyeDetailPixelDense => (71, 2, Some(1), None, 2),
        D::HairPixelDense => (167, 5, Some(4), None, 2),
        D::HairSolidPixelDense => (170, 6, Some(5), None, 2),
        _ => return Err("unsupported pixel runtime ABI"),
    };
    let state = ps.runtime_state(runtime_inputs);
    if ps.constant_buffer_slot != Some(0) { return Err("unexpected constant buffer slot"); }
    if state.constant_registers.len() != rows { return Err("constant row count mismatch"); }
    let mut constants = state.constant_registers;
    // A texture slot the technique leaves empty is not a missing input: the
    // shader reads zero from it, as from any unbound resource.
    fn empty_texture_slot(dependency: &str) -> bool {
        dependency.starts_with("texture Pixel slot ") && dependency.ends_with("<- authored_resource")
    }
    zero_unresolved_rows(&mut constants, &state.unresolved_dependencies, empty_texture_slot)?;
    let animation = AnimatedConstants::new(ps, runtime_inputs, empty_texture_slot);
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
            let Some(tag) = binding.resolved else {
                // An empty slot: plain 2D only, the shape the stand-in texel has.
                return (binding.source == "authored_resource" && Some(slot) != cube && Some(slot) != volume)
                    .then_some(TagHash::NONE);
            };
            let desc = Texture::load_desc(tag).ok()?;
            let layers = if Some(slot) == cube { 6 } else { 1 };
            if desc.array_size != layers || (desc.depth > 1) != (Some(slot) == volume) {
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
        animation,
        vertex_constants,
        textures,
        cube_slot: cube,
        samplers,
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

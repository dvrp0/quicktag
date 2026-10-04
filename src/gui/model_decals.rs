//! Exact stage-2 materials over authored opaque receivers.
//! Viewport and immutable packed-normal snapshot are viewer-owned externs.
use super::{
    AuthoredStageMetadata, ModelPipelineKey, ModelSamplerDesc, ShaderStage, TagHash,
    TechniqueDescriptor, Texture, load_model_sampler_desc,
};

#[derive(Clone)]
pub(super) struct NativeDecalSource {
    pub program: crate::render::authored_program::DescriptorAbi,
    pub vertex_program: crate::render::authored_program::DescriptorAbi,
    pub constants: Vec<[f32; 4]>,
    pub textures: Vec<TagHash>,
    pub samplers: Vec<ModelSamplerDesc>,
}

pub(super) fn resolve_source(
    technique: Option<&TechniqueDescriptor>,
    authored: Option<AuthoredStageMetadata>,
    pipeline: ModelPipelineKey,
    inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Result<NativeDecalSource, &'static str> {
    use crate::render::authored_program::{DescriptorAbi as D, resolve_package_program};
    let authored = authored.ok_or("missing authored stage")?;
    if authored.raw_stage != 2
        || authored.input_layout_id != 7
        || authored.primitive_type != 5
        || !matches!(pipeline.blend, 26 | 27)
        || pipeline.depth_stencil != 15
        || pipeline.depth_bias != 1
        || pipeline.rasterizer != 2
    {
        return Err("unsupported decal stage/layout/topology/render state");
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
    let program = resolve_package_program(ps.shader.ok_or("missing pixel shader")?, ShaderStage::Pixel)
        .map_err(|_|"pixel source identity failed")?.ok_or("unregistered pixel program")?
        .descriptor_abi;
    let required_vertex = match program {
        D::RunnerDecal(index) => crate::render::runner_decal_programs::DECALS.get(index as usize).ok_or("missing decal contract")?.vertex_program,
        _ => D::SharedLayout7VertexScalarStorage,
    };
    if vertex_program != required_vertex { return Err("vertex/pixel contract mismatch"); }
    let pixel_contract = contract(program).ok_or("unsupported pixel runtime ABI")?;
    if pipeline.blend != pixel_contract.blend {
        return Err("decal blend contract mismatch");
    }
    let (rows, texture_slots, sampler_slots, biases): (usize, &[u8], &[u8], &[f32]) = match program
    {
        D::D0F2PixelDense => (9, &[3, 4], &[1], &[-0.5]),
        D::RunnerDecal(index) => {
            let contract = crate::render::runner_decal_programs::DECALS.get(index as usize).ok_or("missing decal contract")?;
            (
                contract.rows,
                contract.textures,
                contract.samplers,
                contract.sampler_biases,
            )
        }
        _ => return Err("unsupported pixel runtime ABI"),
    };
    let state = ps.runtime_state(inputs);
    if ps.constant_buffer_slot != Some(0) || state.constant_registers.len() != rows {
        return Err("constant buffer slot/row count mismatch");
    }
    let (expected_dependencies, metadata_rows): (&[&str], &[(usize, Option<usize>)]) = match program {
        D::RunnerDecal(index) => {
            let contract=&crate::render::runner_decal_programs::DECALS[index as usize];
            (contract.unresolved_dependencies, contract.metadata_rows)
        },
        _ => (&["Decal+0x30 at byte 48", "Decal+0x8 at byte 8", "output[0] <- unresolved numeric value", "texture Pixel slot 2 <- extern_texture(Decal+0x8)"], &[]),
    };
    if state.unresolved_dependencies.iter().map(String::as_str).collect::<Vec<_>>() != expected_dependencies {
        return Err("unresolved runtime dependencies differ from audited decal contract");
    }
    if pixel_contract.scene_normal && !state.bindings.iter().any(|b| {
        b.kind == "texture" && b.slot == 2 && b.source == "extern_texture(Decal+0x8)" && b.resolved.is_none()
    }) { return Err("missing scene-normal extern binding"); }
    let mut constants=state.constant_registers;
    for &(row,gate) in metadata_rows {
        if row>=rows || gate.is_some_and(|g|g>=rows || metadata_rows.iter().any(|&(r,_)|r==g)) {
            return Err("invalid decal metadata proof");
        }
        if gate.is_some_and(|g|constants[g][0]!=0.0) { return Err("decal metadata gate active"); }
        constants[row]=[0.0;4];
    }
    let textures = texture_slots
        .iter()
        .map(|&slot| {
            let tag = state
                .bindings
                .iter()
                .find(|b| b.kind == "texture" && b.slot == slot)?
                .resolved?;
            let desc = Texture::load_desc(tag).ok()?;
            (desc.depth == 1 && desc.array_size == 1).then_some(tag)
        })
        .collect::<Option<Vec<_>>>().ok_or("missing/unresolved texture or incompatible texture dimensions")?;
    let samplers = sampler_slots
        .iter()
        .zip(biases)
        .map(|(&slot, &bias)| {
            let sampler = load_model_sampler_desc(
                state
                    .bindings
                    .iter()
                    .find(|b| b.kind == "sampler" && b.slot == slot)?
                    .resolved?,
            )?;
            if !matches!(sampler.filter, 0 | 0x15 | 0x55)
                || (sampler.filter == 0 && sampler.max_anisotropy != 1)
                || sampler.mip_lod_bias != bias
                || ![sampler.address_u, sampler.address_v, sampler.address_w]
                    .iter()
                    .all(|v| matches!(v, 1..=4))
                || super::authored_sampler_border_color(&sampler).is_err()
                || sampler.min_lod != 0.0
                || sampler.max_lod < 32.0
            {
                return None;
            }
            Some(sampler)
        })
        .collect::<Option<Vec<_>>>().ok_or("missing/unresolved sampler or unsupported sampler contract")?;
    Ok(NativeDecalSource {
        program,
        vertex_program,
        constants,
        textures,
        samplers,
    })
}

impl NativeDecalSource {
    pub fn globals(&self, size: [u32; 2]) -> Vec<[f32; 4]> {
        let mut constants = self.constants.clone();
        // Original PS reads trunc(row0.xy * View16.zw * SV_Position.xy + row0.zw).
        // Paired viewer viewport rows address the same receiver pixel.
        if contract(self.program).unwrap().scene_normal {
            constants[0] = [size[0] as f32, size[1] as f32, 0.0, 0.0];
        }
        constants
    }
}

pub(super) fn contract(
    program: crate::render::authored_program::DescriptorAbi,
) -> Option<crate::render::decal_draw::DecalPixelContract> {
    use crate::render::{authored_program::DescriptorAbi as D, decal_draw::DecalPixelContract};
    match program {
        D::D0F2PixelDense => Some(DecalPixelContract::D0F2),
        D::RunnerDecal(index) => {
            let c = crate::render::runner_decal_programs::DECALS.get(index as usize)?;
            Some(DecalPixelContract {
                constant_rows: c.rows,
                uses_view: c.uses_view,
                scene_normal: c.scene_normal,
                texture_count: c.textures.len() as u32,
                sampler_count: c.samplers.len() as u32,
                blend: c.blend,
            })
        }
        _ => None,
    }
}

pub(super) fn view(size: [u32; 2]) -> [[f32; 4]; 29] {
    let mut view = [[0.0; 4]; 29];
    view[16] = [0.0, 0.0, 1.0 / size[0] as f32, 1.0 / size[1] as f32];
    view
}

pub mod adapter;
pub mod authored_program;
pub(crate) mod runner_surface_programs;
pub(crate) mod runner_decal_programs;
pub(crate) mod body_draw;
pub(crate) mod body_material;
pub(crate) mod body_mesh;
pub(crate) mod body_vertex;
pub(crate) mod displacement_vertex;
pub(crate) mod auxiliary_vertex;
pub(crate) mod color_vertex;
pub(crate) mod rigid_vertex;
pub(crate) mod c827_draw;
pub(crate) mod decal_draw;
pub(crate) mod dense_pixel;
pub mod evidence;
pub(crate) mod global_light;
pub(crate) mod layered_material;
pub mod material;
pub mod pass_plan;
pub mod preview_scene;
pub mod stage;
pub(crate) mod static_mesh;
pub(crate) mod surface_targets;
pub mod technique;
pub mod tfx;
pub(crate) mod channels;
#[cfg(test)]
mod channel_probe;

#[cfg(test)]
mod authored_shader_probe;
#[cfg(test)]
mod compute_shader_probe;

use std::ops::Range;

use tiger_pkg::TagHash;

use self::{
    evidence::ProvenanceId, material::MaterialIR, pass_plan::DrawPassPlan,
    technique::TechniqueDescriptor,
};

/// Renderer-facing draw contract. Raw Tiger identifiers remain authoritative;
/// semantic descriptors/plans are derived and replaceable.
#[derive(Debug, Clone)]
pub struct TigerDrawPacket {
    pub indices: Range<u32>,
    pub raw_lod_category: Option<u8>,
    pub raw_render_stage: Option<u8>,
    pub technique_hash: Option<TagHash>,
    pub technique: Option<TechniqueDescriptor>,
    pub material: MaterialIR,
    pub pass_plan: DrawPassPlan,
    pub source: ProvenanceId,
}

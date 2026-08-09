pub mod evidence;
pub mod material;
pub mod pass_plan;
pub mod preview_scene;
pub mod technique;
pub mod tfx;

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

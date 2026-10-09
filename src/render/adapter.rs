use crate::{
    material::TechniqueRenderState,
    render::{
        evidence::EvidenceLevel,
        stage::MarathonRenderStage,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawRenderStageId(pub u8);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageInterpretation {
    pub raw: RawRenderStageId,
    pub stage: Option<MarathonRenderStage>,
    pub semantic_hint: Option<&'static str>,
    pub evidence: EvidenceLevel,
}

pub struct GoliathAdapter;

impl GoliathAdapter {
    // Compatibility aliases for callers that still operate on raw stage IDs.
    // The semantic source of truth is MarathonRenderStage.
    pub const PRIMARY_STAGE: u8 = MarathonRenderStage::GenerateGbuffer.raw();
    pub const DECAL_STAGE: u8 = MarathonRenderStage::Decals.raw();
    pub const INVESTMENT_DECAL_STAGE: u8 = MarathonRenderStage::InvestmentDecals.raw();
    pub const AUTHORED_SHADOW_STAGE: u8 = MarathonRenderStage::ShadowGenerate.raw();
    pub const ADDITIVE_STAGE: u8 = MarathonRenderStage::DecalsAdditive.raw();
    pub const TRANSPARENT_STAGE: u8 = MarathonRenderStage::Transparents.raw();
    pub const DISTORTION_STAGE: u8 = MarathonRenderStage::Distortion.raw();
    pub const OCCLUSION_STAGE: u8 = MarathonRenderStage::LightShaftOcclusion.raw();
    pub const DEPTH_ONLY_STAGE: u8 = MarathonRenderStage::DepthPrepass.raw();
    pub const AUXILIARY_STAGE: u8 = MarathonRenderStage::PostprocessTransparentStencil.raw();
    pub const FORWARD_SPECIAL_STAGE: u8 = MarathonRenderStage::Reticle.raw();

    pub fn stage(raw: u8) -> StageInterpretation {
        let stage = MarathonRenderStage::try_from(raw).ok();
        StageInterpretation {
            raw: RawRenderStageId(raw),
            stage,
            semantic_hint: stage.and_then(MarathonRenderStage::semantic_name),
            evidence: stage.map_or(EvidenceLevel::Unknown, MarathonRenderStage::evidence),
        }
    }

    pub fn shadow_participation(
        raw_stage: Option<u8>,
        state: TechniqueRenderState,
    ) -> EvidenceLevel {
        if raw_stage == Some(Self::PRIMARY_STAGE)
            && state.blend.is_none_or(|blend| matches!(blend, 0 | 1 | 57))
        {
            EvidenceLevel::Probable
        } else {
            EvidenceLevel::Unknown
        }
    }
}

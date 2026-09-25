use crate::{
    material::TechniqueRenderState,
    render::{
        evidence::EvidenceLevel,
        stage::{MARATHON_RENDER_STAGE_COUNT, MarathonRenderStage},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawLodCategory(pub u8);

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
    pub const ADAPTER_VERSION: &'static str = "goliath-render-v2";
    pub const OBSERVED_STAGE_RANGE_COUNT: usize = MARATHON_RENDER_STAGE_COUNT;

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

    pub fn lod_membership(raw: Option<u8>) -> Option<RawLodCategory> {
        raw.map(RawLodCategory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_goliath_stages_remain_raw() {
        let stage = GoliathAdapter::stage(25);
        assert_eq!(stage.raw, RawRenderStageId(25));
        assert_eq!(stage.stage, None);
        assert_eq!(stage.evidence, EvidenceLevel::Unknown);
        assert!(stage.semantic_hint.is_none());
    }

    #[test]
    fn marathon_specific_stage_remains_semantically_unknown() {
        let stage = GoliathAdapter::stage(MarathonRenderStage::MarathonSpecific3.raw());
        assert_eq!(stage.stage, Some(MarathonRenderStage::MarathonSpecific3));
        assert_eq!(stage.evidence, EvidenceLevel::Unknown);
        assert!(stage.semantic_hint.is_none());
    }

    #[test]
    fn package_backed_stage_aliases_use_formalized_abi() {
        assert_eq!(GoliathAdapter::AUTHORED_SHADOW_STAGE, 4);
        assert_eq!(GoliathAdapter::ADDITIVE_STAGE, 7);
        assert_eq!(GoliathAdapter::TRANSPARENT_STAGE, 8);
        assert_eq!(GoliathAdapter::DISTORTION_STAGE, 9);
        assert_eq!(GoliathAdapter::OCCLUSION_STAGE, 10);
        assert_eq!(GoliathAdapter::DEPTH_ONLY_STAGE, 13);
        assert_eq!(GoliathAdapter::FORWARD_SPECIAL_STAGE, 17);

        let stage = GoliathAdapter::stage(GoliathAdapter::DISTORTION_STAGE);
        assert_eq!(stage.stage, Some(MarathonRenderStage::Distortion));
        assert_eq!(stage.semantic_hint, Some("distortion"));
        assert_eq!(stage.evidence, EvidenceLevel::StronglyCorrelated);
    }
}

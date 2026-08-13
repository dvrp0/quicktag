use crate::{material::TechniqueRenderState, render::evidence::EvidenceLevel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawLodCategory(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawRenderStageId(pub u8);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageInterpretation {
    pub raw: RawRenderStageId,
    pub semantic_hint: Option<&'static str>,
    pub evidence: EvidenceLevel,
}

pub struct GoliathAdapter;

impl GoliathAdapter {
    pub const ADAPTER_VERSION: &'static str = "goliath-render-v1";
    pub const OBSERVED_STAGE_RANGE_COUNT: usize = 25;
    pub const DISTORTION_STAGE: u8 = 8;
    pub const PRIMARY_STAGE: u8 = 0;
    pub const DECAL_STAGE: u8 = 1;
    pub const INVESTMENT_DECAL_STAGE: u8 = 2;
    pub const AUTHORED_SHADOW_STAGE: u8 = 4;
    pub const ADDITIVE_STAGE: u8 = 6;
    pub const TRANSPARENT_STAGE: u8 = 7;
    pub const OCCLUSION_STAGE: u8 = 9;
    pub const DEPTH_ONLY_STAGE: u8 = 13;
    pub const AUXILIARY_STAGE: u8 = 15;
    pub const FORWARD_SPECIAL_STAGE: u8 = 17;

    pub fn stage(raw: u8) -> StageInterpretation {
        StageInterpretation {
            raw: RawRenderStageId(raw),
            semantic_hint: match raw {
                Self::PRIMARY_STAGE => Some("observed primary entity stage"),
                Self::DECAL_STAGE => Some("decal"),
                Self::INVESTMENT_DECAL_STAGE => Some("investment decal"),
                Self::AUTHORED_SHADOW_STAGE => Some("observed shadow-only geometry"),
                Self::ADDITIVE_STAGE => Some("additive"),
                Self::TRANSPARENT_STAGE => Some("transparent"),
                Self::DISTORTION_STAGE => Some("distortion"),
                Self::OCCLUSION_STAGE => Some("observed max-blend forward payload"),
                Self::DEPTH_ONLY_STAGE => Some("observed depth-only geometry"),
                Self::AUXILIARY_STAGE => Some("observed auxiliary geometry"),
                Self::FORWARD_SPECIAL_STAGE => Some("observed forward special"),
                _ => None,
            },
            evidence: match raw {
                Self::DISTORTION_STAGE
                | Self::AUTHORED_SHADOW_STAGE
                | Self::OCCLUSION_STAGE
                | Self::DEPTH_ONLY_STAGE
                | Self::AUXILIARY_STAGE
                | Self::FORWARD_SPECIAL_STAGE => EvidenceLevel::Confirmed,
                Self::PRIMARY_STAGE
                | Self::DECAL_STAGE
                | Self::INVESTMENT_DECAL_STAGE
                | Self::ADDITIVE_STAGE
                | Self::TRANSPARENT_STAGE => EvidenceLevel::StronglyCorrelated,
                _ => EvidenceLevel::Unknown,
            },
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
        let stage = GoliathAdapter::stage(24);
        assert_eq!(stage.raw, RawRenderStageId(24));
        assert_eq!(stage.evidence, EvidenceLevel::Unknown);
        assert!(stage.semantic_hint.is_none());
    }

    #[test]
    fn identifies_authored_distortion_stage() {
        let stage = GoliathAdapter::stage(GoliathAdapter::DISTORTION_STAGE);
        assert_eq!(stage.semantic_hint, Some("distortion"));
        assert_eq!(stage.evidence, EvidenceLevel::Confirmed);
    }
}

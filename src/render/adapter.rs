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

    pub fn stage(raw: u8) -> StageInterpretation {
        StageInterpretation {
            raw: RawRenderStageId(raw),
            semantic_hint: match raw {
                0 => Some("observed primary entity stage"),
                Self::DISTORTION_STAGE => Some("distortion"),
                _ => None,
            },
            evidence: match raw {
                Self::DISTORTION_STAGE => EvidenceLevel::Confirmed,
                0 => EvidenceLevel::StronglyCorrelated,
                _ => EvidenceLevel::Unknown,
            },
        }
    }

    pub fn shadow_participation(
        raw_stage: Option<u8>,
        state: TechniqueRenderState,
    ) -> EvidenceLevel {
        if raw_stage == Some(0) && state.blend.is_none_or(|blend| matches!(blend, 0 | 1 | 57)) {
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

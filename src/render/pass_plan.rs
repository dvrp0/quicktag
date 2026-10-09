use crate::{
    material::TechniqueRenderState,
    render::{
        adapter::GoliathAdapter,
        evidence::EvidenceLevel,
        material::{MaterialFamily, MaterialIR},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPassKind {
    Shadow,
    DepthOnly,
    OpaqueCompatibility,
    AlphaTestedCompatibility,
    DecalCompatibility,
    InvestmentDecalCompatibility,
    ForwardAdditive,
    ForwardTransparent,
    ForwardCoating,
    Distortion,
    Auxiliary,
    UnknownCompatibility,
    MaterialEmissive,
    MaterialFlags,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawPassPlan {
    pub evidence: EvidenceLevel,
    pub passes: Vec<RenderPassKind>,
    pub warnings: Vec<String>,
}

impl DrawPassPlan {
    pub fn derive(
        raw_stage: Option<u8>,
        state: TechniqueRenderState,
        material: &MaterialIR,
    ) -> Self {
        let stage = raw_stage.map(GoliathAdapter::stage);
        if raw_stage == Some(GoliathAdapter::AUTHORED_SHADOW_STAGE) {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::Shadow],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::DEPTH_ONLY_STAGE) {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::DepthOnly],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::AUXILIARY_STAGE) {
            return Self {
                evidence: stage.map_or(EvidenceLevel::Unknown, |stage| stage.evidence),
                passes: vec![RenderPassKind::Auxiliary],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::OCCLUSION_STAGE) {
            return Self {
                evidence: stage.map_or(EvidenceLevel::Unknown, |stage| stage.evidence),
                passes: vec![RenderPassKind::ForwardTransparent],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::DISTORTION_STAGE) {
            return Self {
                evidence: stage.map_or(EvidenceLevel::Unknown, |stage| stage.evidence),
                passes: vec![if material.family() == MaterialFamily::ForwardCoating {
                    RenderPassKind::ForwardCoating
                } else {
                    RenderPassKind::Distortion
                }],
                warnings: vec![],
            };
        }
        if material.family() == MaterialFamily::InvestmentDecal {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::InvestmentDecalCompatibility],
                warnings: (raw_stage != Some(GoliathAdapter::INVESTMENT_DECAL_STAGE))
                    .then(|| format!("investment decal observed on raw stage {raw_stage:?}"))
                    .into_iter()
                    .collect(),
            };
        }
        if matches!(
            raw_stage,
            Some(GoliathAdapter::DECAL_STAGE | GoliathAdapter::INVESTMENT_DECAL_STAGE)
        ) {
            return Self {
                evidence: EvidenceLevel::StronglyCorrelated,
                passes: vec![RenderPassKind::DecalCompatibility],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::ADDITIVE_STAGE) {
            return Self {
                evidence: EvidenceLevel::StronglyCorrelated,
                passes: vec![RenderPassKind::ForwardAdditive],
                warnings: vec![],
            };
        }
        let resource_free_surface = match material {
            MaterialIR::Unknown(_) => true,
            MaterialIR::Surface(surface) => {
                surface.inputs.color.is_none()
                    && surface.inputs.normal.is_none()
                    && surface.inputs.emissive.is_none()
                    && surface.inputs.control.is_none()
                    && surface.inputs.solid_color.is_none()
            }
            MaterialIR::Decal(_)
            | MaterialIR::ForwardCoating(_)
            | MaterialIR::ForwardSpecial(_) => false,
        };
        if raw_stage == Some(GoliathAdapter::FORWARD_SPECIAL_STAGE) && resource_free_surface {
            // Goliath stage 17 includes camera-facing helper payloads whose
            // shader depends entirely on external engine scopes. Rendering an
            // unclassified, resource-free payload as an ordinary transparent
            // surface exposes its authored quad as a white card.
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::Auxiliary],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::TRANSPARENT_STAGE)
            || raw_stage == Some(GoliathAdapter::FORWARD_SPECIAL_STAGE)
        {
            return Self {
                evidence: stage.map_or(EvidenceLevel::Unknown, |stage| stage.evidence),
                passes: vec![RenderPassKind::ForwardTransparent],
                warnings: vec![],
            };
        }
        let blended = state
            .blend
            .is_some_and(|index| !matches!(index, 0 | 1 | 57));
        if blended {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::ForwardTransparent],
                warnings: vec![],
            };
        }
        if material.family() == MaterialFamily::CompactHair {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![
                    RenderPassKind::Shadow,
                    RenderPassKind::AlphaTestedCompatibility,
                ],
                warnings: vec![],
            };
        }
        if raw_stage.is_none() {
            return Self {
                evidence: EvidenceLevel::Unknown,
                passes: vec![RenderPassKind::UnknownCompatibility],
                warnings: vec!["missing raw render stage".into()],
            };
        }
        let shadow_evidence = GoliathAdapter::shadow_participation(raw_stage, state);
        let mut passes = vec![];
        if shadow_evidence != EvidenceLevel::Unknown {
            passes.push(RenderPassKind::Shadow);
        }
        passes.push(RenderPassKind::OpaqueCompatibility);
        Self {
            evidence: shadow_evidence,
            passes,
            warnings: vec![],
        }
    }
}

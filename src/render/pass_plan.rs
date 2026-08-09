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
    OpaqueCompatibility,
    AlphaTestedCompatibility,
    InvestmentDecalCompatibility,
    ForwardTransparent,
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
        if material.family() == MaterialFamily::InvestmentDecal {
            return Self {
                evidence: EvidenceLevel::StronglyCorrelated,
                passes: vec![RenderPassKind::InvestmentDecalCompatibility],
                warnings: (raw_stage != Some(2))
                    .then(|| format!("investment decal observed on raw stage {raw_stage:?}"))
                    .into_iter()
                    .collect(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::WireframeMaterialTextures, render::material::MaterialIR};

    #[test]
    fn unknown_stage_is_explicit() {
        let material = MaterialIR::classify(&WireframeMaterialTextures::default());
        let plan = DrawPassPlan::derive(None, TechniqueRenderState::default(), &material);
        assert_eq!(plan.passes, [RenderPassKind::UnknownCompatibility]);
        assert!(!plan.warnings.is_empty());
    }
}

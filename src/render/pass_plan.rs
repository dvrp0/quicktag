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
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::Auxiliary],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::OCCLUSION_STAGE) {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::ForwardTransparent],
                warnings: vec![],
            };
        }
        if raw_stage == Some(GoliathAdapter::DISTORTION_STAGE) {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::Distortion],
                warnings: vec![],
            };
        }
        if material.family() == MaterialFamily::InvestmentDecal {
            return Self {
                evidence: EvidenceLevel::Confirmed,
                passes: vec![RenderPassKind::InvestmentDecalCompatibility],
                warnings: (raw_stage != Some(GoliathAdapter::DECAL_STAGE))
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
                surface.color_texture.is_none()
                    && surface.normal_texture.is_none()
                    && surface.emissive_texture.is_none()
                    && surface.control_texture.is_none()
                    && surface.solid_color.is_none()
            }
            MaterialIR::Decal(_) | MaterialIR::ForwardSpecial(_) => false,
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

    #[test]
    fn distortion_stage_uses_forward_distortion_pass() {
        let material = MaterialIR::classify(&WireframeMaterialTextures::default());
        let plan = DrawPassPlan::derive(
            Some(GoliathAdapter::DISTORTION_STAGE),
            TechniqueRenderState::default(),
            &material,
        );
        assert_eq!(plan.passes, [RenderPassKind::Distortion]);
        assert_eq!(plan.evidence, EvidenceLevel::Confirmed);
    }

    #[test]
    fn authored_auxiliary_stages_never_fall_through_to_opaque() {
        let material = MaterialIR::classify(&WireframeMaterialTextures::default());
        for (stage, expected) in [
            (
                GoliathAdapter::AUTHORED_SHADOW_STAGE,
                RenderPassKind::Shadow,
            ),
            (GoliathAdapter::DEPTH_ONLY_STAGE, RenderPassKind::DepthOnly),
            (GoliathAdapter::AUXILIARY_STAGE, RenderPassKind::Auxiliary),
            (
                GoliathAdapter::OCCLUSION_STAGE,
                RenderPassKind::ForwardTransparent,
            ),
        ] {
            let plan =
                DrawPassPlan::derive(stage.into(), TechniqueRenderState::default(), &material);
            assert_eq!(plan.passes, [expected]);
        }
    }

    #[test]
    fn resource_free_forward_special_payload_is_not_a_visible_white_card() {
        let material = MaterialIR::classify(&WireframeMaterialTextures::default());
        let plan = DrawPassPlan::derive(
            Some(GoliathAdapter::FORWARD_SPECIAL_STAGE),
            TechniqueRenderState {
                blend: Some(8),
                ..Default::default()
            },
            &material,
        );
        assert_eq!(plan.passes, [RenderPassKind::Auxiliary]);

        let mut palette_contaminated = WireframeMaterialTextures::default();
        palette_contaminated.gear_dye_palette = Some(
            [crate::geometry::GearDyeMaterial {
                color: [1.0; 4],
                roughness_remap: [0.0; 4],
                metal_remap: [0.0; 4],
            }; 6],
        );
        let material = MaterialIR::classify(&palette_contaminated);
        let plan = DrawPassPlan::derive(
            Some(GoliathAdapter::FORWARD_SPECIAL_STAGE),
            TechniqueRenderState {
                blend: Some(8),
                ..Default::default()
            },
            &material,
        );
        assert_eq!(plan.passes, [RenderPassKind::Auxiliary]);
    }

    #[test]
    fn textured_forward_special_payload_remains_visible() {
        let mut textures = WireframeMaterialTextures::default();
        textures.color = Some(tiger_pkg::TagHash(0x80A60058));
        let material = MaterialIR::classify(&textures);
        let plan = DrawPassPlan::derive(
            Some(GoliathAdapter::FORWARD_SPECIAL_STAGE),
            TechniqueRenderState {
                blend: Some(8),
                ..Default::default()
            },
            &material,
        );
        assert_eq!(plan.passes, [RenderPassKind::ForwardTransparent]);
    }

    #[test]
    fn stage_two_non_investment_material_remains_an_authored_decal() {
        let material = MaterialIR::classify(&WireframeMaterialTextures::default());
        let plan = DrawPassPlan::derive(
            Some(GoliathAdapter::INVESTMENT_DECAL_STAGE),
            TechniqueRenderState::default(),
            &material,
        );
        assert_eq!(plan.passes, [RenderPassKind::DecalCompatibility]);
    }
}

use crate::geometry::{
    GearDyeMaterial, GearPatternMaterial, InvestmentDecalMaterial, WeaponModWearMaterial,
    WireframeMaterialTextures,
};
use crate::render::evidence::EvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialFamily {
    StandardSurface,
    TexturelessSolid,
    GoliathGearSurface,
    StaticGearPattern,
    WeaponModCondition,
    InvestmentDecal,
    CompactHair,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MaterialExpression {
    Constant([f32; 4]),
    Texture {
        tag: tiger_pkg::TagHash,
        role: &'static str,
    },
    Layer {
        index: usize,
        inputs: Vec<tiger_pkg::TagHash>,
    },
    UnknownRawChannels(Vec<tiger_pkg::TagHash>),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MaterialDependencyGraph {
    pub nodes: Vec<MaterialExpression>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SurfaceIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub dependencies: MaterialDependencyGraph,
    pub color_texture: Option<tiger_pkg::TagHash>,
    pub normal_texture: Option<tiger_pkg::TagHash>,
    pub emissive_texture: Option<tiger_pkg::TagHash>,
    pub control_texture: Option<tiger_pkg::TagHash>,
    pub gear_dye: Option<GearDyeMaterial>,
    pub pattern: Option<GearPatternMaterial>,
    pub condition: Option<WeaponModWearMaterial>,
    pub solid_color: Option<[f32; 4]>,
    pub solid_surface: Option<[f32; 2]>,
}

#[derive(Debug, Clone)]
pub struct DecalIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub investment: InvestmentDecalMaterial,
}

#[derive(Debug, Clone)]
pub struct ForwardSpecialIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub mask_texture: Option<tiger_pkg::TagHash>,
    pub palette: [[f32; 4]; 2],
}

#[derive(Debug, Clone)]
pub struct UnknownMaterialIR {
    pub evidence: EvidenceLevel,
    pub bound_resources: Vec<tiger_pkg::TagHash>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum MaterialIR {
    Surface(SurfaceIR),
    Decal(DecalIR),
    ForwardSpecial(ForwardSpecialIR),
    Unknown(UnknownMaterialIR),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassificationCandidate {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub matched: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassificationReport {
    pub selected: MaterialFamily,
    pub candidates: Vec<ClassificationCandidate>,
    pub rejected: Vec<(MaterialFamily, &'static str)>,
    pub conflicts: Vec<String>,
}

impl MaterialIR {
    pub fn family(&self) -> MaterialFamily {
        match self {
            Self::Surface(surface) => surface.family,
            Self::Decal(decal) => decal.family,
            Self::ForwardSpecial(special) => special.family,
            Self::Unknown(_) => MaterialFamily::Unknown,
        }
    }

    pub fn classification_report(textures: &WireframeMaterialTextures) -> ClassificationReport {
        let mut candidates = Vec::new();
        let mut add = |family, evidence, matched| {
            candidates.push(ClassificationCandidate {
                family,
                evidence,
                matched,
            });
        };
        if textures.investment_decal.is_some() {
            add(
                MaterialFamily::InvestmentDecal,
                EvidenceLevel::Confirmed,
                vec!["investment decal ABI"],
            );
        }
        if textures.mask_palette.is_some() {
            add(
                MaterialFamily::CompactHair,
                EvidenceLevel::Confirmed,
                vec!["compact hair palette ABI"],
            );
        }
        if textures.gear_pattern.is_some() {
            add(
                MaterialFamily::StaticGearPattern,
                EvidenceLevel::Confirmed,
                vec!["gear pattern payload"],
            );
        }
        if textures.mod_wear.is_some() {
            add(
                MaterialFamily::WeaponModCondition,
                EvidenceLevel::Confirmed,
                vec!["weapon condition payload"],
            );
        }
        if textures.gear_dye.is_some() || textures.gear_dye_palette.is_some() {
            add(
                MaterialFamily::GoliathGearSurface,
                EvidenceLevel::StronglyCorrelated,
                vec!["Goliath Gear palette/remap"],
            );
        }
        if textures.solid_color.is_some() {
            add(
                MaterialFamily::TexturelessSolid,
                EvidenceLevel::StronglyCorrelated,
                vec!["authored solid constants"],
            );
        }
        if textures.color.is_some()
            || textures.normal.is_some()
            || textures.emissive.is_some()
            || textures.control.is_some()
        {
            add(
                MaterialFamily::StandardSurface,
                EvidenceLevel::Probable,
                vec!["surface texture ABI"],
            );
        }
        let precedence = [
            MaterialFamily::InvestmentDecal,
            MaterialFamily::CompactHair,
            MaterialFamily::StaticGearPattern,
            MaterialFamily::WeaponModCondition,
            MaterialFamily::GoliathGearSurface,
            MaterialFamily::TexturelessSolid,
            MaterialFamily::StandardSurface,
        ];
        let selected = precedence
            .into_iter()
            .find(|family| {
                candidates
                    .iter()
                    .any(|candidate| candidate.family == *family)
            })
            .unwrap_or(MaterialFamily::Unknown);
        let conflicts = (candidates.len() > 1)
            .then(|| {
                format!(
                    "{} signatures matched; precedence selected {selected:?}",
                    candidates.len()
                )
            })
            .into_iter()
            .collect();
        let rejected = precedence
            .into_iter()
            .filter(|family| {
                !candidates
                    .iter()
                    .any(|candidate| candidate.family == *family)
            })
            .map(|family| (family, "required deterministic signature absent"))
            .collect();
        ClassificationReport {
            selected,
            candidates,
            rejected,
            conflicts,
        }
    }

    pub fn classify(textures: &WireframeMaterialTextures) -> Self {
        let report = Self::classification_report(textures);
        if let Some(investment) = textures.investment_decal {
            return Self::Decal(DecalIR {
                family: MaterialFamily::InvestmentDecal,
                evidence: EvidenceLevel::Confirmed,
                investment,
            });
        }
        if report.selected == MaterialFamily::CompactHair {
            return Self::ForwardSpecial(ForwardSpecialIR {
                family: MaterialFamily::CompactHair,
                evidence: EvidenceLevel::Confirmed,
                mask_texture: textures.color,
                palette: textures.mask_palette.unwrap_or_default(),
            });
        }
        let (family, evidence) = if textures.gear_pattern.is_some() {
            (MaterialFamily::StaticGearPattern, EvidenceLevel::Confirmed)
        } else if textures.mod_wear.is_some() {
            (MaterialFamily::WeaponModCondition, EvidenceLevel::Confirmed)
        } else if textures.gear_dye.is_some() || textures.gear_dye_palette.is_some() {
            (
                MaterialFamily::GoliathGearSurface,
                EvidenceLevel::StronglyCorrelated,
            )
        } else if textures.solid_color.is_some() {
            (
                MaterialFamily::TexturelessSolid,
                EvidenceLevel::StronglyCorrelated,
            )
        } else if textures.color.is_some()
            || textures.normal.is_some()
            || textures.emissive.is_some()
            || textures.control.is_some()
        {
            (MaterialFamily::StandardSurface, EvidenceLevel::Probable)
        } else {
            return Self::Unknown(UnknownMaterialIR {
                evidence: EvidenceLevel::Unknown,
                bound_resources: textures.aux.clone(),
                reasons: vec!["no confirmed surface or decal signature".into()],
            });
        };
        Self::Surface(SurfaceIR {
            family,
            evidence,
            dependencies: dependency_graph(textures),
            color_texture: textures.color,
            normal_texture: textures.normal,
            emissive_texture: textures.emissive,
            control_texture: textures.control,
            gear_dye: textures.gear_dye,
            pattern: textures.gear_pattern,
            condition: textures.mod_wear,
            solid_color: textures.solid_color,
            solid_surface: textures.solid_surface,
        })
    }
}

fn dependency_graph(textures: &WireframeMaterialTextures) -> MaterialDependencyGraph {
    let mut graph = MaterialDependencyGraph::default();
    for (tag, role) in [
        (textures.color, "base_color"),
        (textures.normal, "normal"),
        (textures.emissive, "emissive"),
        (textures.control, "raw_control_rgba"),
    ] {
        if let Some(tag) = tag {
            graph.nodes.push(MaterialExpression::Texture { tag, role });
        }
    }
    for (index, layer) in textures.layers.iter().enumerate() {
        let inputs = [layer.color, layer.normal, layer.emissive]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        graph
            .nodes
            .push(MaterialExpression::Layer { index, inputs });
    }
    if !textures.aux.is_empty() {
        graph
            .nodes
            .push(MaterialExpression::UnknownRawChannels(textures.aux.clone()));
        graph
            .warnings
            .push("unclassified bound resources preserved".into());
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_deterministic_and_decal_first() {
        let mut textures = WireframeMaterialTextures::default();
        textures.color = Some(tiger_pkg::TagHash(0x80A00001));
        assert_eq!(
            MaterialIR::classify(&textures).family(),
            MaterialFamily::StandardSurface
        );

        textures.mask_palette = Some([[0.0; 4]; 2]);
        let report = MaterialIR::classification_report(&textures);
        assert_eq!(report.selected, MaterialFamily::CompactHair);
        assert!(!report.conflicts.is_empty());
        assert_eq!(
            MaterialIR::classify(&textures).family(),
            MaterialFamily::CompactHair
        );
    }
}

use crate::geometry::{
    AlphaMaskMaterial, CharacterSurfaceMaterial, ForwardCoatingMaterial, GearDyeMaterial,
    GearPatternMaterial, InvestmentDecalMaterial, RunnerLayeredSurfaceMaterial,
    RunnerOcclusionMaterial, SharedAtlasDetailMaterial, TransmissionMaterial,
    WeaponModConditionMaterial, WeaponSurfaceConditionMaterial, WireframeMaterialLayer,
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
    ForwardCoating,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfirmedGearSlot {
    pub color: [f32; 4],
    pub roughness_remap: [f32; 4],
    pub metalness_remap: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoliathGearDyeIR {
    pub slots: [ConfirmedGearSlot; 6],
    pub raw_parameters: Vec<[f32; 4]>,
    pub evidence: EvidenceLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlChannelExpression {
    pub raw_channel: char,
    pub expression: &'static str,
    pub evidence: EvidenceLevel,
}

#[derive(Debug, Clone)]
pub struct SurfaceIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub dependencies: MaterialDependencyGraph,
    pub control_expressions: Vec<ControlChannelExpression>,
    pub inputs: MaterialInputs,
    pub goliath_gear: Option<GoliathGearDyeIR>,
}

/// Lossless renderer-facing material payload. Classification may add semantic
/// meaning, but must never discard a decoded package input.
#[derive(Debug, Clone)]
pub struct MaterialInputs {
    pub color: Option<tiger_pkg::TagHash>,
    pub normal: Option<tiger_pkg::TagHash>,
    pub emissive: Option<tiger_pkg::TagHash>,
    pub control: Option<tiger_pkg::TagHash>,
    pub character_surface: Option<CharacterSurfaceMaterial>,
    pub runner_layered_surface: Option<RunnerLayeredSurfaceMaterial>,
    pub runner_occlusion: Option<RunnerOcclusionMaterial>,
    pub alpha_mask: Option<AlphaMaskMaterial>,
    pub shared_atlas_detail: Option<SharedAtlasDetailMaterial>,
    pub roughness_channel: u8,
    pub sampler: Option<tiger_pkg::TagHash>,
    pub aux: Vec<tiger_pkg::TagHash>,
    pub layers: Vec<WireframeMaterialLayer>,
    pub color_tint: [u8; 4],
    pub mask_palette: Option<[[f32; 4]; 2]>,
    pub gear_dye: Option<GearDyeMaterial>,
    pub gear_dye_default: Option<[f32; 4]>,
    pub gear_dye_palette: Option<[GearDyeMaterial; 6]>,
    pub gear_worn_dye_palette: Option<[[f32; 4]; 6]>,
    pub gear_dye_detail_palette: Option<[[f32; 4]; 6]>,
    pub mod_wear: Option<WeaponModConditionMaterial>,
    pub surface_condition: Option<WeaponSurfaceConditionMaterial>,
    pub gear_pattern: Option<GearPatternMaterial>,
    pub authored_shared_atlas: bool,
    pub investment_decal: Option<InvestmentDecalMaterial>,
    pub emissive_strength: u8,
    pub solid_color: Option<[f32; 4]>,
    pub solid_surface: Option<[f32; 2]>,
    pub iridescence_id: Option<f32>,
    pub transmission: Option<TransmissionMaterial>,
    pub forward_coating: Option<ForwardCoatingMaterial>,
}

impl From<&WireframeMaterialTextures> for MaterialInputs {
    fn from(value: &WireframeMaterialTextures) -> Self {
        Self {
            color: value.color,
            normal: value.normal,
            emissive: value.emissive,
            control: value.control,
            character_surface: value.character_surface,
            runner_layered_surface: value.runner_layered_surface,
            runner_occlusion: value.runner_occlusion,
            alpha_mask: value.alpha_mask,
            shared_atlas_detail: value.shared_atlas_detail,
            roughness_channel: value.roughness_channel,
            sampler: value.sampler,
            aux: value.aux.clone(),
            layers: value.layers.clone(),
            color_tint: value.color_tint,
            mask_palette: value.mask_palette,
            gear_dye: value.gear_dye,
            gear_dye_default: value.gear_dye_default,
            gear_dye_palette: value.gear_dye_palette,
            gear_worn_dye_palette: value.gear_worn_dye_palette,
            gear_dye_detail_palette: value.gear_dye_detail_palette,
            mod_wear: value.mod_wear,
            surface_condition: value.surface_condition,
            gear_pattern: value.gear_pattern,
            authored_shared_atlas: value.authored_shared_atlas,
            investment_decal: value.investment_decal,
            emissive_strength: value.emissive_strength,
            solid_color: value.solid_color,
            solid_surface: value.solid_surface,
            iridescence_id: value.iridescence_id,
            transmission: value.transmission,
            forward_coating: value.forward_coating,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecalIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub investment: InvestmentDecalMaterial,
    pub inputs: MaterialInputs,
}

#[derive(Debug, Clone)]
pub struct ForwardSpecialIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub mask_texture: Option<tiger_pkg::TagHash>,
    pub palette: [[f32; 4]; 2],
    pub inputs: MaterialInputs,
}

#[derive(Debug, Clone)]
pub struct ForwardCoatingIR {
    pub family: MaterialFamily,
    pub evidence: EvidenceLevel,
    pub coating: ForwardCoatingMaterial,
    pub inputs: MaterialInputs,
}

#[derive(Debug, Clone)]
pub struct UnknownMaterialIR {
    pub evidence: EvidenceLevel,
    pub bound_resources: Vec<tiger_pkg::TagHash>,
    pub reasons: Vec<String>,
    pub inputs: MaterialInputs,
}

#[derive(Debug, Clone)]
pub enum MaterialIR {
    Surface(SurfaceIR),
    Decal(DecalIR),
    ForwardCoating(ForwardCoatingIR),
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
            Self::ForwardCoating(coating) => coating.family,
            Self::ForwardSpecial(special) => special.family,
            Self::Unknown(_) => MaterialFamily::Unknown,
        }
    }

    pub fn inputs(&self) -> &MaterialInputs {
        match self {
            Self::Surface(surface) => &surface.inputs,
            Self::Decal(decal) => &decal.inputs,
            Self::ForwardCoating(coating) => &coating.inputs,
            Self::ForwardSpecial(special) => &special.inputs,
            Self::Unknown(unknown) => &unknown.inputs,
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
        if textures.forward_coating.is_some() {
            add(
                MaterialFamily::ForwardCoating,
                EvidenceLevel::Confirmed,
                vec!["forward coating PS/TFX ABI"],
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
            MaterialFamily::ForwardCoating,
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
                inputs: textures.into(),
            });
        }
        if let Some(coating) = textures.forward_coating {
            return Self::ForwardCoating(ForwardCoatingIR {
                family: MaterialFamily::ForwardCoating,
                evidence: EvidenceLevel::Confirmed,
                coating,
                inputs: textures.into(),
            });
        }
        if report.selected == MaterialFamily::CompactHair {
            return Self::ForwardSpecial(ForwardSpecialIR {
                family: MaterialFamily::CompactHair,
                evidence: EvidenceLevel::Confirmed,
                mask_texture: textures.color,
                palette: textures.mask_palette.unwrap_or_default(),
                inputs: textures.into(),
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
                inputs: textures.into(),
            });
        };
        Self::Surface(SurfaceIR {
            family,
            evidence,
            dependencies: dependency_graph(textures),
            control_expressions: control_expressions(textures),
            inputs: textures.into(),
            goliath_gear: textures.gear_dye_palette.map(|palette| GoliathGearDyeIR {
                slots: palette.map(|slot| ConfirmedGearSlot {
                    color: slot.color,
                    roughness_remap: slot.roughness_remap,
                    metalness_remap: slot.metal_remap,
                }),
                raw_parameters: palette
                    .into_iter()
                    .flat_map(|slot| [slot.color, slot.roughness_remap, slot.metal_remap])
                    .collect(),
                evidence: EvidenceLevel::Confirmed,
            }),
        })
    }
}

fn control_expressions(textures: &WireframeMaterialTextures) -> Vec<ControlChannelExpression> {
    if textures.gear_dye_palette.is_some() {
        return ['R', 'G', 'B']
            .map(|raw_channel| ControlChannelExpression {
                raw_channel,
                expression: "raw <= 0.5 selects Goliath Gear palette bit",
                evidence: EvidenceLevel::Confirmed,
            })
            .into();
    }
    if textures.control.is_some() {
        return ['R', 'G', 'B', 'A']
            .map(|raw_channel| ControlChannelExpression {
                raw_channel,
                expression: "raw channel preserved; semantic unknown",
                evidence: EvidenceLevel::Unknown,
            })
            .into();
    }
    vec![]
}

fn dependency_graph(textures: &WireframeMaterialTextures) -> MaterialDependencyGraph {
    let mut graph = MaterialDependencyGraph::default();
    let mut push_texture = |tag: Option<tiger_pkg::TagHash>, role| {
        if let Some(tag) = tag {
            graph.nodes.push(MaterialExpression::Texture { tag, role });
        }
    };
    for (tag, role) in [
        (textures.color, "base_color"),
        (textures.normal, "normal"),
        (textures.emissive, "emissive"),
        (textures.control, "raw_control_rgba"),
    ] {
        push_texture(tag, role);
    }
    if let Some(character) = textures.character_surface {
        for (tag, role) in [
            (Some(character.surface), "character_surface"),
            (Some(character.selector), "character_selector"),
            (Some(character.detail_color), "character_detail_color"),
            (Some(character.detail_normal), "character_detail_normal"),
            (character.procedural, "character_procedural"),
        ] {
            push_texture(tag, role);
        }
    }
    if let Some(runner) = textures.runner_layered_surface {
        for (tag, role) in [
            (Some(runner.surface), "runner_surface"),
            (runner.material_response, "runner_material_response"),
            (Some(runner.detail_normal_a), "runner_detail_normal_a"),
            (Some(runner.detail_normal_b), "runner_detail_normal_b"),
            (runner.detail_normal_c, "runner_detail_normal_c"),
            (runner.detail_normal_d, "runner_detail_normal_d"),
            (runner.procedural, "runner_procedural"),
            (runner.color_overlay, "runner_color_overlay"),
        ] {
            push_texture(tag, role);
        }
        if let Some([scratches, distortion, breakup]) = runner.procedural_wear {
            for (tag, role) in [
                (scratches, "runner_wear_scratches"),
                (distortion, "runner_wear_distortion"),
                (breakup, "runner_wear_breakup"),
            ] {
                push_texture(Some(tag), role);
            }
        }
    }
    if let Some(occlusion) = textures.runner_occlusion {
        push_texture(Some(occlusion.texture), "runner_occlusion");
    }
    if let Some(alpha_mask) = textures.alpha_mask {
        push_texture(Some(alpha_mask.texture), "alpha_mask");
    }
    if let Some(detail) = textures.shared_atlas_detail {
        push_texture(Some(detail.detail), "shared_atlas_detail");
    }
    if let Some(pattern) = textures.gear_pattern {
        push_texture(Some(pattern.field), "gear_pattern_field");
    }
    if let Some(wear) = textures.mod_wear {
        for (tag, role) in [
            (wear.scratches, "weapon_wear_scratches"),
            (wear.grime, "weapon_wear_grime"),
            (wear.damage, "weapon_wear_damage"),
        ] {
            push_texture(Some(tag), role);
        }
    }
    if let Some(condition) = textures.surface_condition {
        for (tag, role) in [
            (condition.response, "weapon_surface_response"),
            (condition.detail, "weapon_surface_detail"),
            (condition.breakup, "weapon_surface_breakup"),
        ] {
            push_texture(Some(tag), role);
        }
    }
    if let Some(decal) = textures.investment_decal {
        for (tag, role) in [
            (Some(decal.color), "investment_decal_color"),
            (Some(decal.mask), "investment_decal_mask"),
            (decal.detail, "investment_decal_detail"),
        ] {
            push_texture(tag, role);
        }
    }
    if let Some(coating) = textures.forward_coating {
        push_texture(Some(coating.detail), "forward_coating_detail");
        push_texture(Some(coating.environment), "forward_coating_environment");
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
    let known_tags = graph
        .nodes
        .iter()
        .flat_map(|node| match node {
            MaterialExpression::Texture { tag, .. } => vec![*tag],
            MaterialExpression::Layer { inputs, .. }
            | MaterialExpression::UnknownRawChannels(inputs) => inputs.clone(),
            MaterialExpression::Constant(_) => vec![],
        })
        .collect::<Vec<_>>();
    let unknown = textures
        .aux
        .iter()
        .copied()
        .filter(|tag| !known_tags.contains(tag))
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        graph
            .nodes
            .push(MaterialExpression::UnknownRawChannels(unknown));
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

    #[test]
    fn dependency_graph_names_known_resources_and_only_warns_for_unknowns() {
        let color = tiger_pkg::TagHash(0x80A00001);
        let alpha = tiger_pkg::TagHash(0x80A00002);
        let unknown = tiger_pkg::TagHash(0x80A00003);
        let mut textures = WireframeMaterialTextures::default();
        textures.color = Some(color);
        textures.alpha_mask = Some(AlphaMaskMaterial {
            texture: alpha,
            remap: [0.0, 1.0],
            threshold: 0.5,
        });
        textures.aux = vec![alpha, unknown];

        let MaterialIR::Surface(surface) = MaterialIR::classify(&textures) else {
            panic!("color texture must classify as a surface");
        };
        assert!(
            surface
                .dependencies
                .nodes
                .contains(&MaterialExpression::Texture {
                    tag: alpha,
                    role: "alpha_mask",
                })
        );
        assert!(
            surface
                .dependencies
                .nodes
                .contains(&MaterialExpression::UnknownRawChannels(vec![unknown]))
        );
        assert_eq!(surface.dependencies.warnings.len(), 1);
    }

    #[test]
    fn coating_is_selected_only_from_decoded_shader_evidence() {
        let detail = tiger_pkg::TagHash(0x80A60055);
        let environment = tiger_pkg::TagHash(0x80A60056);
        let mut textures = WireframeMaterialTextures::default();
        textures.color = Some(detail);
        textures.aux = vec![detail, environment];

        assert_eq!(
            MaterialIR::classify(&textures).family(),
            MaterialFamily::StandardSurface,
            "resource hashes alone must never opt a skin into coating"
        );

        textures.forward_coating = Some(ForwardCoatingMaterial {
            detail,
            environment,
            environment_sampler: tiger_pkg::TagHash(0x80A60057),
            colors: [[0.4; 4], [0.8; 4]],
            incidence_remap: [1.0, 0.0],
            coverage: 0.95,
            projection: [40.0, 40.0, 0.0, 0.0],
            projection_exponent: 40.0,
            detail_remap: [0.32, 0.68],
            response_remap: [-1.3, 2.3],
            environment_lod: [0.0, 0.0],
            environment_remap: [0.0, 1.0],
            environment_strength: 0.775,
            environment_params: [1.0, 1.0, 0.0, 0.0],
            specular_colors: [[0.4; 4], [0.2; 4]],
            specular_exponents: [300.0, 20.0],
            specular_strengths: [1.0, 0.6],
            lobe_direction_scales: [0.08, 0.54],
        });
        assert_eq!(
            MaterialIR::classify(&textures).family(),
            MaterialFamily::ForwardCoating
        );
    }
}

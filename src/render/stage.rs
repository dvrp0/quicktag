use super::evidence::EvidenceLevel;

pub const MARATHON_RENDER_STAGE_COUNT: usize = 25;

/// Marathon/Goliath's dynamic-mesh render-stage table.
///
/// Package evidence shows a 25-stage table (26 boundaries). Relative to the
/// Destiny 2 Tiger table reconstructed by Alkahest, Marathon has one additional
/// slot at raw index 3. Multiple independent anchors then line up with a +1
/// shift from ShadowGenerate onward:
///
/// - raw 4: VS-only, high-overlap caster pass -> ShadowGenerate
/// - raw 13: VS-only, high-overlap depth pass -> DepthPrepass
/// - raw 17: authored camera-facing weapon helper cards -> Reticle
/// - raw 24: compute-only techniques -> ComputeSkinning
///
/// The semantic purpose of raw stage 3 is still unknown and is intentionally
/// not named beyond MarathonSpecific3.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MarathonRenderStage {
    GenerateGbuffer = 0,
    Decals = 1,
    InvestmentDecals = 2,
    MarathonSpecific3 = 3,
    ShadowGenerate = 4,
    LightingApply = 5,
    LightProbeApply = 6,
    DecalsAdditive = 7,
    Transparents = 8,
    Distortion = 9,
    LightShaftOcclusion = 10,
    SkinPrepass = 11,
    LensFlares = 12,
    DepthPrepass = 13,
    WaterReflection = 14,
    PostprocessTransparentStencil = 15,
    Impulse = 16,
    Reticle = 17,
    WaterRipples = 18,
    MaskSunLight = 19,
    Volumetrics = 20,
    Cubemaps = 21,
    PostprocessScreen = 22,
    WorldForces = 23,
    ComputeSkinning = 24,
}

impl MarathonRenderStage {
    pub const ALL: [Self; MARATHON_RENDER_STAGE_COUNT] = [
        Self::GenerateGbuffer,
        Self::Decals,
        Self::InvestmentDecals,
        Self::MarathonSpecific3,
        Self::ShadowGenerate,
        Self::LightingApply,
        Self::LightProbeApply,
        Self::DecalsAdditive,
        Self::Transparents,
        Self::Distortion,
        Self::LightShaftOcclusion,
        Self::SkinPrepass,
        Self::LensFlares,
        Self::DepthPrepass,
        Self::WaterReflection,
        Self::PostprocessTransparentStencil,
        Self::Impulse,
        Self::Reticle,
        Self::WaterRipples,
        Self::MaskSunLight,
        Self::Volumetrics,
        Self::Cubemaps,
        Self::PostprocessScreen,
        Self::WorldForces,
        Self::ComputeSkinning,
    ];

    pub const fn raw(self) -> u8 {
        self as u8
    }

    pub const fn semantic_name(self) -> Option<&'static str> {
        Some(match self {
            Self::GenerateGbuffer => "generate_gbuffer",
            Self::Decals => "decals",
            Self::InvestmentDecals => "investment_decals",
            Self::MarathonSpecific3 => return None,
            Self::ShadowGenerate => "shadow_generate",
            Self::LightingApply => "lighting_apply",
            Self::LightProbeApply => "light_probe_apply",
            Self::DecalsAdditive => "decals_additive",
            Self::Transparents => "transparents",
            Self::Distortion => "distortion",
            Self::LightShaftOcclusion => "light_shaft_occlusion",
            Self::SkinPrepass => "skin_prepass",
            Self::LensFlares => "lens_flares",
            Self::DepthPrepass => "depth_prepass",
            Self::WaterReflection => "water_reflection",
            Self::PostprocessTransparentStencil => "postprocess_transparent_stencil",
            Self::Impulse => "impulse",
            Self::Reticle => "reticle",
            Self::WaterRipples => "water_ripples",
            Self::MaskSunLight => "mask_sun_light",
            Self::Volumetrics => "volumetrics",
            Self::Cubemaps => "cubemaps",
            Self::PostprocessScreen => "postprocess_screen",
            Self::WorldForces => "world_forces",
            Self::ComputeSkinning => "compute_skinning",
        })
    }

    pub const fn evidence(self) -> EvidenceLevel {
        match self {
            Self::InvestmentDecals
            | Self::ShadowGenerate
            | Self::DepthPrepass
            | Self::ComputeSkinning => EvidenceLevel::Confirmed,

            Self::GenerateGbuffer
            | Self::Transparents
            | Self::Distortion
            | Self::LightShaftOcclusion
            | Self::WaterReflection
            | Self::PostprocessTransparentStencil
            | Self::Reticle
            | Self::WaterRipples => EvidenceLevel::StronglyCorrelated,

            Self::Decals
            | Self::LightingApply
            | Self::LightProbeApply
            | Self::DecalsAdditive
            | Self::SkinPrepass
            | Self::LensFlares
            | Self::Impulse
            | Self::MaskSunLight
            | Self::Volumetrics
            | Self::Cubemaps
            | Self::PostprocessScreen
            | Self::WorldForces => EvidenceLevel::Probable,

            Self::MarathonSpecific3 => EvidenceLevel::Unknown,
        }
    }
}

impl TryFrom<u8> for MarathonRenderStage {
    type Error = u8;

    fn try_from(raw: u8) -> Result<Self, Self::Error> {
        Self::ALL.get(raw as usize).copied().ok_or(raw)
    }
}

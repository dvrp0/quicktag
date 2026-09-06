use rustc_hash::FxHashMap;
use tiger_pkg::{TagHash, package_manager};

const ICON_DEFINITION_REFERENCE: u32 = 0x8080_5335;
const ICON_CONTAINER_REFERENCE: u32 = 0x8080_5350;
const ICON_IDENTIFIER_OFFSET: usize = 0x10;
const ICON_CONTAINER_OFFSET: usize = 0x18;
const ICON_TEXTURE_OFFSET: usize = 0x90;

#[derive(Clone, Debug, Default)]
pub(super) struct ImplantDetails {
    pub(super) effects: Vec<super::item_effect::ItemEffect>,
    pub(super) stats: Vec<ImplantStat>,
}

#[derive(Clone, Debug)]
pub(super) struct ImplantStat {
    pub(super) name: String,
    pub(super) value: f32,
}

pub(super) struct ImplantIconResolver {
    textures: FxHashMap<u32, TagHash>,
    implant_textures: FxHashMap<u32, TagHash>,
}

impl ImplantIconResolver {
    pub(super) fn load() -> Self {
        let mut definitions = package_manager().get_all_by_reference(ICON_DEFINITION_REFERENCE);
        definitions.sort_unstable_by_key(|(tag, _)| *tag);
        let textures = definitions
            .into_iter()
            .filter_map(|(definition, _)| {
                let data = package_manager().read_tag(definition).ok()?;
                let identifier = read_u32(&data, ICON_IDENTIFIER_OFFSET)?;
                let container = TagHash(read_u32(&data, ICON_CONTAINER_OFFSET)?);
                (package_manager().get_entry(container)?.reference == ICON_CONTAINER_REFERENCE)
                    .then_some(())?;
                let container = package_manager().read_tag(container).ok()?;
                let texture = TagHash(read_u32(&container, ICON_TEXTURE_OFFSET)?);
                package_manager().get_entry(texture)?;
                Some((identifier, texture))
            })
            .collect::<FxHashMap<_, _>>();
        let implant_textures = textures
            .iter()
            .filter_map(|(identifier, texture)| {
                let desc = crate::texture::Texture::load_desc(*texture).ok()?;
                (desc.width == 64 && desc.height == 64).then_some((*identifier, *texture))
            })
            .collect();
        Self {
            textures,
            implant_textures,
        }
    }

    pub(super) fn resolve_implant(&self, display_identifier: &str) -> Option<TagHash> {
        let (authored_identifier, category) = match display_identifier {
            "back_in_action" => ("sleight_of_hand", false),
            "energy_harvester" => ("energizer", false),
            "frenzy_matrix" => ("frenzy", false),
            "head_implants" => ("head", true),
            "leg_implants" => ("legs", true),
            "regen_matrix" => ("regen", false),
            "torso_implants" => ("torso", true),
            "triage_cloak" => ("going_dark", false),
            _ => {
                let authored_hash = match display_identifier {
                    "bionic_legs" => 0xE022_6868,
                    "counter_intel" => 0xCA7B_D8D9,
                    "dynamo" => 0x4266_8A81,
                    "explosive_and_melee_resistance" => 0x66E9_525E,
                    "petty_theft" => 0xF973_6C7D,
                    "ping" => 0x7B2B_17E6,
                    "splash_guard" => 0xC9F2_EFE1,
                    "targeting_brace" => 0xE8BE_DC7C,
                    _ => quicktag_core::util::fnv1(display_identifier.as_bytes()),
                };
                return self.implant_textures.get(&authored_hash).copied();
            }
        };
        let authored_hash = quicktag_core::util::fnv1(authored_identifier.as_bytes());
        if category {
            self.textures.get(&authored_hash).copied()
        } else {
            self.implant_textures.get(&authored_hash).copied()
        }
    }
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

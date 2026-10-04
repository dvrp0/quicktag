use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
};

use eframe::egui::{self, Color32, RichText};
use itertools::Itertools;
use quicktag_strings::localized::{
    LocalizedLanguage, LocalizedStringPart, LocalizedStringResolver, StringCache,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use tiger_pkg::{GameVersion, TagHash, package_manager};

use crate::{
    geometry::WeaponModRarity,
    texture::{Texture, cache::TextureCache},
};

use super::implant_stats::{ImplantDetails, ImplantIconResolver, ImplantStatsResolver};
use super::item_effect::{ItemEffect, ItemEffectResolver};
use super::profile_texture::resolve_profile_textures;
use super::sticker_texture::resolve_sticker_texture;
use super::weapon_stats::{
    WeaponModDetails, WeaponModStatChange, WeaponModWeaponStats, WeaponStatResolver, WeaponStats,
};
use super::{TOASTS, ViewAction, common::ResponseExt};

const GEAR_DISPLAY_REFERENCE: u32 = 0x80806ef6;
const DISPLAY_TO_HASH_REFERENCE: u32 = 0x80806ef0;
const HASH_TO_DEFINITION_REFERENCE: u32 = 0x80809685;
const GEAR_DEFINITION_REFERENCE: u32 = 0x8080968b;
const INTERNAL_CATEGORY_NAMESPACE_REGISTRY_REFERENCE: u32 = 0x808095d4;
const INTERNAL_CATEGORY_REGISTRY_REFERENCE: u32 = 0x808095d6;
const INTERNAL_CATEGORY_NAMESPACE_TABLE_OFFSET: usize = 0x8;
const INTERNAL_CATEGORY_INDEX_TABLE_OFFSET: usize = 0x8;
const INTERNAL_CATEGORY_LOOKUP_TABLE_OFFSET: usize = 0x18;
// Goliath's July investment-definition layout inserted one pointer-sized field
// before the authored category table.  Reading the former 0x168 header yields
// an empty range for every item even though the live table begins at 0x170.
const INTERNAL_CATEGORY_LIST_OFFSET: usize = 0x170;
const PATTERN_GLOBAL_TABLE_REFERENCE: u32 = 0x80806cac;
const PATTERN_ASSIGNMENT_TABLE_REFERENCE: u32 = 0x8080b61c;
const INVESTMENT_COSMETIC_MAP_REFERENCE: u32 = 0x80803081;
const PATTERN_TRANSLATION_BLOCK_MARKER: u32 = 0x808091d3;
const MODEL_PATTERN_REFERENCE: u32 = 0x8080baad;
const MODEL_PATTERN_COMPONENT_REFERENCE: u32 = 0x8080badb;
const EMPTY_HASH: u32 = 0x811c9dc5;
const RARITY_MARKER: u32 = 0x808092a2;
const RARITY_STANDARD_HASH: u32 = 0xb45bd4fa;
const RARITY_ENHANCED_HASH: u32 = 0x6ca1c197;
const RARITY_DELUXE_HASH: u32 = 0x43d7f7f0;
const RARITY_SUPERIOR_HASH: u32 = 0xeeeb4d28;
const RARITY_PRESTIGE_HASH: u32 = 0x9067f1ea;
const RARITY_CONTRABAND_HASH: u32 = 0x10ba2aa7;
const RARITY_DYNAMIC_HASH: u32 = 0x8489fd3c;
const RARITY_QUEST_HASH: u32 = 0x6b5d0665;
const RARITY_UNIQUE_HASH: u32 = 0xdd0dcef4;
const PRICE_MARKER: u32 = 0x80809295;
const WEAPON_TYPE_ASSAULT_RIFLE: u32 = 0xc0af2b85;
const WEAPON_TYPE_PISTOL: u32 = 0xebbddfc6;
const WEAPON_TYPE_SUBMACHINE_GUN: u32 = 0x580d4990;
const WEAPON_TYPE_SHOTGUN: u32 = 0x49f5ae13;
const WEAPON_TYPE_SNIPER_RIFLE: u32 = 0xee34ba5d;
const WEAPON_TYPE_RAILGUN: u32 = 0x0d8e4317;
const WEAPON_TYPE_MACHINE_GUN: u32 = 0x3f45d733;
const WEAPON_TYPE_MARKSMAN_RIFLE: u32 = 0x5b53e34a;
const TRINKET_CATEGORY_HASH: u32 = 0xa34d_fefd;
const D54_BATTLE_PISTOL_HASH: u32 = 0xf3d6_6647;
const KKV_9SD_HASH: u32 = 0xc33f_db60;
const FIRESTORM_HASH: u32 = 0x01ad_1959;
const BIOTOXIC_DISINJECTOR_HASH: u32 = 0xa3df_1228;
const D54_DEFAULT_SKIN_HASH: u32 = 0xd627_3928;
const BIOTOXIC_DEFAULT_SKIN_HASH: u32 = 0xedce_641a;
const BIOTOXIC_SHADOW_INDEX_SKIN_HASH: u32 = 0xe913_05f0;
const ACID_ABYSS_SKIN_HASH: u32 = 0xcb7b_d3a7;
const CRYO_SHIFT_V11_SKIN_HASH: u32 = 0xf3a3_2d16;
const SENTINEL_CORE_HASHES: [u32; 10] = [
    0x1ae2_0d50,
    0x1ae2_0d51,
    0x1ae2_0d53,
    0x1ae2_0d54,
    0x1ae2_0d55,
    0x1ae2_0d56,
    0x1ae2_0d57,
    0x1ae2_0d5a,
    0x1ae2_0d5b,
    0x1be2_0ec5,
];
const ACHROMATIC_RUSH_MELEE_SKIN_HASH: u32 = 0xcb5b_56b8;
const VOX_NOCTURNA_MELEE_SKIN_HASH: u32 = 0x6a30_a26e;
// Authored attachment-slot identifiers embedded in investment definitions.
// They remain available when the corresponding `weapon_mods.*` path is not in
// the static wordlist, which is common for newer mods.
const MOD_CATEGORY_BARREL_MARKER: u32 = 0xac14_1b18;
const MOD_CATEGORY_CHIP_MARKER: u32 = 0x037f_053c;
const MOD_CATEGORY_FOREGRIP_MARKER: u32 = 0xbc03_4d96;
const MOD_CATEGORY_GENERATOR_MARKER: u32 = 0x325d_716f;
const MOD_CATEGORY_MAGAZINE_MARKER: u32 = 0x81c3_8342;
const MOD_CATEGORY_MUZZLE_MARKER: u32 = 0xf714_f29f;
const MOD_CATEGORY_OPTIC_MARKER: u32 = 0x5a11_6cba;
const MOD_CATEGORY_SHIELD_MARKER: u32 = 0x9a74_649f;
const MOD_CATEGORY_STOCK_MARKER: u32 = 0xddcd_e46a;
const MOD_CATEGORY_UNIQUE_MARKER: u32 = 0x037d_053c;
const PROFILE_BACKGROUND_CATEGORY: &str = "item_type.#D5006AAF";
const PROFILE_EMBLEM_CATEGORY: &str = "item_type.#5F40AEB9";
const PROFILE_TITLE_CATEGORY: &str = "item_type.#A8805E93";
// Goliath's pure-white presentation style is a default format sentinel, not
// the final UI tint. Marathon renders those formatted spans with its standard
// blue (also used by Deluxe UI surfaces). Explicit authored colors remain
// untouched.
const LOCALIZED_FORMAT_COLOR: Color32 = Color32::from_rgb(0x4a, 0xaf, 0xff);
const LOCALIZED_FORMAT_COLOR_HEX: &str = "#4aafff";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GearRarity {
    Standard,
    Enhanced,
    Deluxe,
    Superior,
    Prestige,
    Contraband,
    Dynamic,
    Quest,
    Unique,
}

impl GearRarity {
    fn from_raw_hash(hash: u32) -> Option<Self> {
        match hash {
            RARITY_STANDARD_HASH => Some(Self::Standard),
            RARITY_ENHANCED_HASH => Some(Self::Enhanced),
            RARITY_DELUXE_HASH => Some(Self::Deluxe),
            RARITY_SUPERIOR_HASH => Some(Self::Superior),
            RARITY_PRESTIGE_HASH => Some(Self::Prestige),
            RARITY_CONTRABAND_HASH => Some(Self::Contraband),
            RARITY_DYNAMIC_HASH => Some(Self::Dynamic),
            RARITY_QUEST_HASH => Some(Self::Quest),
            RARITY_UNIQUE_HASH => Some(Self::Unique),
            _ => None,
        }
    }

    fn from_tier_value(value: u32) -> Option<Self> {
        match value {
            0xffff0000 => Some(Self::Standard),
            0xffff0001 => Some(Self::Enhanced),
            0xffff0002 => Some(Self::Deluxe),
            0xffff0003 => Some(Self::Superior),
            0xffff0004 => Some(Self::Prestige),
            0xffff0005 => Some(Self::Contraband),
            0xffff0006 => Some(Self::Quest),
            // Ranked tiers and other runtime-selected tier records use table references
            // instead of the fixed FFFF000N enum.
            value if value != 0 && value != u32::MAX => Some(Self::Dynamic),
            _ => None,
        }
    }

    /// Investment definitions repeat the presentation rarity in the footer.
    /// Cosmetics and weapon mods omit the normal rarity marker, so this is
    /// their authoritative tier field.
    fn from_footer_code(code: u16) -> Option<Self> {
        match code {
            0x031c | 0x032d => Some(Self::Deluxe),
            0x031d | 0x032e => Some(Self::Contraband),
            0x031f | 0x0330 => Some(Self::Prestige),
            0x0320 | 0x0331 => Some(Self::Enhanced),
            0x0321 | 0x0332 => Some(Self::Standard),
            0x0322 | 0x0333 => Some(Self::Superior),
            0x0323 | 0x0334 => Some(Self::Quest),
            0x0324 | 0x0335 => Some(Self::Unique),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Enhanced => "Enhanced",
            Self::Deluxe => "Deluxe",
            Self::Superior => "Superior",
            Self::Prestige => "Prestige",
            Self::Contraband => "Contraband",
            Self::Dynamic => "Dynamic",
            Self::Quest => "Quest",
            Self::Unique => "Unique",
        }
    }

    fn export_code(self) -> &'static str {
        match self {
            Self::Standard => "STD",
            Self::Enhanced => "E",
            Self::Deluxe => "D",
            Self::Superior => "S",
            Self::Prestige => "P",
            Self::Contraband => "C",
            Self::Dynamic => "DYN",
            Self::Quest => "Q",
            Self::Unique => "U",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Self::Standard => Color32::from_rgb(158, 165, 174),
            Self::Enhanced => Color32::from_rgb(82, 191, 105),
            Self::Deluxe => Color32::from_rgb(72, 145, 224),
            Self::Superior => Color32::from_rgb(166, 92, 214),
            Self::Prestige => Color32::from_rgb(232, 184, 72),
            Self::Contraband => Color32::from_rgb(218, 72, 72),
            Self::Dynamic => Color32::from_rgb(72, 205, 194),
            Self::Quest => Color32::from_rgb(236, 205, 91),
            Self::Unique => Color32::from_rgb(204, 255, 0),
        }
    }
}

#[derive(Clone, Debug)]
struct GearItem {
    display_tag: TagHash,
    icon_tag: Option<TagHash>,
    detail_texture_tags: Vec<TagHash>,
    definition_tag: Option<TagHash>,
    model_tag: Option<TagHash>,
    definition_group_key: Option<u32>,
    definition_type_code: Option<u16>,
    internal_hash: Option<u32>,
    internal_name: Option<String>,
    raw_category_hash: Option<u32>,
    raw_subcategory_hash: Option<u32>,
    name: String,
    rarity: Option<GearRarity>,
    item_type: Option<String>,
    subcategory: Option<String>,
    applies_to: Option<String>,
    mod_category: Option<String>,
    mod_family: Option<String>,
    mod_is_universal: bool,
    compatible_weapons: Vec<String>,
    classification: Option<String>,
    types: Vec<String>,
    internal_categories: Vec<String>,
    price: Option<u32>,
    buying_price: Option<u32>,
    description: Option<String>,
    description_parts: Vec<LocalizedStringPart>,
}

#[derive(Serialize)]
struct GearExportRecord<'a> {
    hash: String,
    name: &'a str,
    rarity: Option<&'static str>,
    item_type: Option<&'a str>,
    subcategory: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_weapon: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_shell: Option<&'a str>,
    mod_category: Option<&'a str>,
    mod_family: Option<&'a str>,
    compatible_weapons: &'a [String],
    display_class: Option<&'a str>,
    types: &'a [String],
    price: Option<u32>,
    buying_price: Option<u32>,
    description: Option<String>,
    internal_type: Option<&'a str>,
    internal_hash: Option<String>,
    definition_tag: Option<String>,
    internal_categories: &'a [String],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    effects: Vec<GearExportEffect<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    implant_stats: Vec<GearExportStat<'a>>,
}

#[derive(Serialize)]
struct GearExportEffect<'a> {
    name: &'a str,
    description: String,
}

#[derive(Serialize)]
struct GearExportStat<'a> {
    name: &'a str,
    raw_name_hash: String,
    value: i32,
}

pub struct GearView {
    items: Vec<GearItem>,
    weapon_stats: FxHashMap<TagHash, WeaponStats>,
    weapon_mod_stats: FxHashMap<TagHash, WeaponModDetails>,
    implant_details: FxHashMap<TagHash, ImplantDetails>,
    weapon_mod_effects: FxHashMap<TagHash, Vec<ItemEffect>>,
    item_types: Vec<(String, usize)>,
    rarities: Vec<(GearRarity, usize)>,
    internal_category_counts: Vec<(String, usize)>,
    filtered_indices: Vec<usize>,
    selected: Option<usize>,
    selected_item_types: FxHashSet<String>,
    selected_rarities: FxHashSet<GearRarity>,
    selected_internal_categories: FxHashSet<String>,
    internal_category_search: String,
    search: String,
    status: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ModelWeaponCatalog {
    pub(super) weapons: Vec<ModelWeaponEntry>,
    pub(super) runner_skins: Vec<ModelRunnerSkinEntry>,
    pub(super) melee_skins: Vec<ModelMeleeSkinEntry>,
    pub(super) charms: Vec<ModelCharmEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct ModelCharmEntry {
    pub(super) name: String,
    pub(super) model_tag: TagHash,
    pub(super) color: Color32,
}

#[derive(Clone, Debug)]
pub(super) struct ModelMeleeSkinEntry {
    pub(super) name: String,
    pub(super) family_name: String,
    pub(super) model_tag: TagHash,
    pub(super) color: Color32,
}

#[derive(Clone, Debug)]
pub(super) struct ModelRunnerSkinEntry {
    pub(super) name: String,
    pub(super) shell_name: String,
    pub(super) model_tag: TagHash,
    pub(super) color: Color32,
}

#[derive(Clone, Debug)]
pub(super) struct ModelWeaponEntry {
    pub(super) name: String,
    pub(super) owner_tag: TagHash,
    pub(super) socket_owner: Option<TagHash>,
    pub(super) model_tags: Vec<TagHash>,
    pub(super) skins: Vec<ModelWeaponSkinEntry>,
    pub(super) slots: Vec<ModelModSlot>,
}

#[derive(Clone, Debug)]
pub(super) struct ModelWeaponSkinEntry {
    pub(super) name: String,
    pub(super) model_tag: TagHash,
    pub(super) rarity: Option<String>,
    pub(super) color: Color32,
}

#[derive(Clone, Debug)]
pub(super) struct ModelModSlot {
    pub(super) name: String,
    pub(super) mods: Vec<ModelModEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct ModelModEntry {
    pub(super) name: String,
    pub(super) rarity: String,
    pub(super) rarity_code: &'static str,
    pub(super) color: Color32,
    pub(super) model_tag: TagHash,
    pub(super) preview_rarity: Option<WeaponModRarity>,
}

impl GearView {
    pub fn new(strings: Arc<StringCache>) -> Self {
        Self::new_for_language(strings, LocalizedLanguage::English)
    }

    pub fn new_for_language(strings: Arc<StringCache>, language: LocalizedLanguage) -> Self {
        match load_gear_for_language(&strings, language) {
            Ok(items) => {
                let (implant_details, weapon_mod_effects) =
                    extract_item_effects(&items, &strings, language);
                let item_types = collect_item_types(&items);
                let rarities = collect_rarities(&items);
                let internal_category_counts = collect_internal_category_counts(&items);
                let mut view = Self {
                    items,
                    weapon_stats: FxHashMap::default(),
                    weapon_mod_stats: FxHashMap::default(),
                    implant_details,
                    weapon_mod_effects,
                    item_types,
                    rarities,
                    internal_category_counts,
                    filtered_indices: vec![],
                    selected: None,
                    selected_item_types: FxHashSet::default(),
                    selected_rarities: FxHashSet::default(),
                    selected_internal_categories: FxHashSet::default(),
                    internal_category_search: String::new(),
                    search: String::new(),
                    status: None,
                };
                view.update_filter();
                view
            }
            Err(error) => Self {
                items: vec![],
                weapon_stats: FxHashMap::default(),
                weapon_mod_stats: FxHashMap::default(),
                implant_details: FxHashMap::default(),
                weapon_mod_effects: FxHashMap::default(),
                item_types: vec![],
                rarities: vec![],
                internal_category_counts: vec![],
                filtered_indices: vec![],
                selected: None,
                selected_item_types: FxHashSet::default(),
                selected_rarities: FxHashSet::default(),
                selected_internal_categories: FxHashSet::default(),
                internal_category_search: String::new(),
                search: String::new(),
                status: Some(error),
            },
        }
    }

    pub(crate) fn export_implant_icons(
        output: &Path,
        render_state: &eframe::egui_wgpu::RenderState,
    ) -> Result<usize, String> {
        let strings = Arc::new(
            quicktag_strings::localized::create_stringmap_for_language(LocalizedLanguage::Korean)
                .map_err(|error| format!("Failed to load Korean localized strings: {error:#}"))?,
        );
        let view = Self::new_for_language(strings, LocalizedLanguage::Korean);
        if let Some(error) = view.load_error() {
            return Err(error.to_owned());
        }
        std::fs::create_dir_all(output)
            .map_err(|error| format!("Could not create {}: {error}", output.display()))?;

        let mut used_names = HashSet::new();
        let mut seen_icons = HashSet::new();
        let mut exported = 0;
        for item in &view.items {
            let (slot, prefix) = match implant_slot(item) {
                Some("Head") => ("Head", "머리"),
                Some("Torso") => ("Torso", "상체"),
                Some("Leg") => ("Leg", "다리"),
                _ => continue,
            };
            let Some(icon) = item.icon_tag else { continue };
            if !seen_icons.insert((slot, icon)) {
                continue;
            }
            let base_name = sanitize_icon_name(&item.name);
            let base_name = if base_name.is_empty() {
                "unnamed".to_owned()
            } else {
                base_name
            };
            let mut filename = format!("마라톤아이콘_{prefix}_{base_name}.png");
            if !used_names.insert(filename.clone()) {
                filename = format!("마라톤아이콘_{prefix}_{base_name}_{}.png", item.display_tag);
                used_names.insert(filename.clone());
            }
            let image = Texture::load(render_state, icon, false)
                .and_then(|texture| texture.to_image(render_state, 0))
                .map_err(|error| format!("Failed to decode {icon} for {}: {error:#}", item.name))?;
            image
                .save(output.join(filename))
                .map_err(|error| format!("Failed to write icon for {}: {error}", item.name))?;
            exported += 1;
        }
        Ok(exported)
    }

    /// A compact, render-ready projection of Gear's authored weapon/mod data.
    /// Keeping this derived from the same records prevents the Models and Gear
    /// panels from developing separate compatibility rules.
    pub(super) fn model_weapon_catalog(&self) -> ModelWeaponCatalog {
        let model_owners = self
            .items
            .iter()
            .filter_map(|item| {
                let owner = match item.item_type.as_deref()? {
                    "Weapon" => item.name.as_str(),
                    "Weapon Skin" => item.applies_to.as_deref()?,
                    _ => return None,
                };
                Some((item.model_tag?, owner))
            })
            .into_group_map();
        let model_is_unambiguous = |model: &TagHash| {
            model_owners
                .get(model)
                .is_some_and(|owners| owners.iter().unique().count() == 1)
        };
        let mut weapon_names = self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .map(|item| item.name.clone())
            .collect::<Vec<_>>();
        weapon_names.sort_by_key(|name| name.to_lowercase());
        weapon_names.dedup();

        let weapons = weapon_names
            .into_iter()
            .filter_map(|name| {
                let weapon = self
                    .items
                    .iter()
                    .filter(|item| item.item_type.as_deref() == Some("Weapon") && item.name == name)
                    .min_by_key(|item| {
                        (item.rarity != Some(GearRarity::Standard), item.display_tag)
                    })?;
                let mut model_tags = self
                    .items
                    .iter()
                    .filter(|item| {
                        (item.item_type.as_deref() == Some("Weapon") && item.name == name)
                            || (item.item_type.as_deref() == Some("Weapon Skin")
                                && item.applies_to.as_deref() == Some(name.as_str()))
                    })
                    .filter_map(|item| item.model_tag)
                    .filter(&model_is_unambiguous)
                    .collect::<Vec<_>>();
                model_tags.sort_unstable();
                model_tags.dedup();
                if model_tags.is_empty() {
                    return None;
                }
                // Runtime sockets/defaults belong to the non-rarity Weapon
                // record's Pattern. Cosmetic ownership ambiguity must affect
                // list identification only; substituting a skin Pattern here
                // discards the engine's attachment/default branches.
                let owner_tag = weapon.model_tag?;

                let mut skins = self
                    .items
                    .iter()
                    .filter(|item| {
                        item.item_type.as_deref() == Some("Weapon Skin")
                            && item.applies_to.as_deref() == Some(name.as_str())
                    })
                    .filter_map(|item| {
                        Some(ModelWeaponSkinEntry {
                            name: item.name.clone(),
                            model_tag: item.model_tag?,
                            rarity: item.rarity.map(|rarity| rarity.label().to_owned()),
                            color: item.rarity.map(GearRarity::color).unwrap_or(Color32::GRAY),
                        })
                    })
                    .collect::<Vec<_>>();
                skins.sort_by_cached_key(|skin| (skin.model_tag, skin.name.to_lowercase()));
                skins.dedup_by(|left, right| {
                    left.model_tag == right.model_tag && left.name == right.name
                });

                let slots = compatible_mod_sections(&self.items, weapon)
                    .into_iter()
                    .filter_map(|(slot, indices)| {
                        let mods = indices
                            .into_iter()
                            .filter_map(|index| {
                                let item = &self.items[index];
                                let rarity = item.rarity?;
                                Some(ModelModEntry {
                                    name: item.name.clone(),
                                    rarity: rarity.label().to_owned(),
                                    rarity_code: rarity.export_code(),
                                    color: rarity.color(),
                                    model_tag: item.model_tag?,
                                    preview_rarity: match rarity {
                                        GearRarity::Enhanced => Some(WeaponModRarity::Enhanced),
                                        GearRarity::Deluxe => Some(WeaponModRarity::Deluxe),
                                        GearRarity::Superior => Some(WeaponModRarity::Superior),
                                        _ => None,
                                    },
                                })
                            })
                            .collect::<Vec<_>>();
                        (!mods.is_empty()).then_some(ModelModSlot { name: slot, mods })
                    })
                    .take(4)
                    .collect::<Vec<_>>();
                Some(ModelWeaponEntry {
                    name,
                    owner_tag,
                    socket_owner: None,
                    model_tags,
                    skins,
                    slots,
                })
            })
            .collect();

        let mut runner_skins = self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Runner Skin"))
            .filter_map(|item| {
                Some(ModelRunnerSkinEntry {
                    name: item.name.clone(),
                    shell_name: item.applies_to.clone()?,
                    model_tag: item.model_tag?,
                    color: item.rarity.map(GearRarity::color).unwrap_or(Color32::GRAY),
                })
            })
            .collect::<Vec<_>>();
        runner_skins.sort_by_cached_key(|skin| (skin.model_tag, skin.name.to_lowercase()));
        runner_skins.dedup_by(|left, right| {
            left.model_tag == right.model_tag
                && left.name == right.name
                && left.shell_name == right.shell_name
        });

        let mut melee_skins = self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Melee"))
            .filter_map(|item| {
                Some(ModelMeleeSkinEntry {
                    name: item.name.clone(),
                    family_name: item
                        .subcategory
                        .clone()
                        .unwrap_or_else(|| "Melee".to_owned()),
                    model_tag: item.model_tag?,
                    color: item.rarity.map(GearRarity::color).unwrap_or(Color32::GRAY),
                })
            })
            .collect::<Vec<_>>();
        melee_skins.sort_by_cached_key(|skin| (skin.model_tag, skin.name.to_lowercase()));
        melee_skins.dedup_by(|left, right| {
            left.model_tag == right.model_tag
                && left.name == right.name
                && left.family_name == right.family_name
        });

        let mut charms = self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Charm"))
            .filter_map(|item| {
                Some(ModelCharmEntry {
                    name: item.name.clone(),
                    model_tag: item.model_tag?,
                    color: item.rarity.map(GearRarity::color).unwrap_or(Color32::GRAY),
                })
            })
            .collect::<Vec<_>>();
        charms.sort_by_cached_key(|charm| (charm.model_tag, charm.name.to_lowercase()));
        charms.dedup_by(|left, right| left.model_tag == right.model_tag && left.name == right.name);

        ModelWeaponCatalog {
            weapons,
            runner_skins,
            melee_skins,
            charms,
        }
    }

    /// Reconciles cosmetic ownership with the render Pattern templates used by
    /// concrete Weapon records. Investment cosmetic groups remain a load-time
    /// fallback, but cannot override an unambiguous authored model structure.
    pub(super) fn reconcile_weapon_skin_models(&mut self, cache: &quicktag_scanner::TagCache) {
        const GEOMETRY_FRAME_TOLERANCE: f32 = 0.025;

        if cache.hashes.is_empty() {
            return;
        }

        let canonical_subcategories = self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .filter_map(|item| Some((item.name.clone(), item.subcategory.clone()?)))
            .collect::<FxHashMap<_, _>>();
        let base_weapons = self
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon")
                    && item.rarity != Some(GearRarity::Unique)
                    && is_canonical_weapon_skin_owner(item)
            })
            .filter_map(|item| {
                Some((
                    item.model_tag?,
                    item.name.clone(),
                    canonical_subcategories
                        .get(&item.name)
                        .cloned()
                        .or_else(|| item.subcategory.clone()),
                    is_canonical_weapon_skin_owner(item),
                ))
            })
            .collect::<Vec<_>>();

        let weapon_names_by_model = base_weapons
            .iter()
            .map(|(model, name, _, _)| (*model, name.as_str()))
            .into_group_map();
        let mut pattern_anchors = base_weapons
            .iter()
            .filter(|(model, _, _, _)| {
                weapon_names_by_model
                    .get(model)
                    .is_some_and(|names| names.iter().unique().count() == 1)
            })
            .filter_map(|(model, name, subcategory, canonical_owner)| {
                Some((
                    crate::geometry::model_pattern_structure_signature(cache, *model)?,
                    name.clone(),
                    subcategory.clone(),
                    *canonical_owner,
                ))
            })
            .collect::<Vec<_>>();
        pattern_anchors.extend(
            self.items
                .iter()
                .filter(|item| item.item_type.as_deref() == Some("Weapon Skin"))
                .filter_map(|item| {
                    Some((
                        crate::geometry::model_pattern_structure_signature(cache, item.model_tag?)?,
                        item.applies_to.clone()?,
                        item.subcategory.clone(),
                        true,
                    ))
                }),
        );

        for skin in self.items.iter_mut().filter(|item| {
            item.item_type.as_deref() == Some("Weapon Skin") && item.applies_to.is_none()
        }) {
            let Some(model) = skin.model_tag else {
                continue;
            };
            let Some(signature) = crate::geometry::model_pattern_structure_signature(cache, model)
            else {
                continue;
            };
            let exact_candidates = pattern_anchors
                .iter()
                .filter(|(candidate, _, _, _)| candidate == &signature)
                .collect::<Vec<_>>();
            let exact_has_canonical_owner = exact_candidates
                .iter()
                .any(|(_, _, _, canonical_owner)| *canonical_owner);
            let exact_matches = exact_candidates
                .into_iter()
                .filter(|(_, _, _, canonical_owner)| !exact_has_canonical_owner || *canonical_owner)
                .map(|(_, name, subcategory, _)| (name, subcategory))
                .unique_by(|(name, _)| name.as_str())
                .collect::<Vec<_>>();
            let pattern_matches = if exact_matches.is_empty() {
                // Cosmetic Pattern wrappers are often independently compiled,
                // so their payload and number of geometry parts can differ
                // from the base weapon. Their authored quantization frame is
                // retained for weapon sockets and animation. Accept it only
                // when it identifies exactly one weapon name; group IDs remain
                // the fallback for genuinely ambiguous frames.
                let geometry_matches = pattern_anchors
                    .iter()
                    .filter(|(candidate, _, _, _)| {
                        signature
                            .closest_geometry_distance(candidate)
                            .is_some_and(|distance| distance <= GEOMETRY_FRAME_TOLERANCE)
                    })
                    .collect::<Vec<_>>();
                let has_canonical_owner = geometry_matches
                    .iter()
                    .any(|(_, _, _, canonical_owner)| *canonical_owner);
                geometry_matches
                    .into_iter()
                    .filter(|(_, _, _, canonical_owner)| !has_canonical_owner || *canonical_owner)
                    .map(|(_, name, subcategory, _)| (name, subcategory))
                    .unique_by(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
            } else {
                exact_matches
            };
            if let [(name, subcategory)] = pattern_matches.as_slice() {
                skin.applies_to = Some(name.to_string());
                skin.subcategory.clone_from(subcategory);
            }
        }
        loop {
            let mut changed = 0;
            changed += reconcile_skin_shared_pattern_components(&mut self.items, cache);
            changed += propagate_current_skin_groups(&mut self.items);
            if changed == 0 {
                break;
            }
        }
        reconcile_distinct_weapon_variant_skins(&mut self.items, cache, GEOMETRY_FRAME_TOLERANCE);
        assign_legacy_unresolved_skin_owners(&mut self.items);
        self.reconcile_weapon_mod_models(cache);
        self.extract_weapon_stats(cache);
        self.extract_weapon_mod_stats(cache);
        self.update_filter();
    }

    fn extract_weapon_stats(&mut self, cache: &quicktag_scanner::TagCache) {
        let resolver = WeaponStatResolver::load(cache);
        self.weapon_stats = self
            .items
            .iter()
            .filter(|item| is_authored_weapon(item))
            .filter_map(|item| Some((item.display_tag, resolver.extract(item.definition_tag?)?)))
            .collect();
    }

    fn extract_weapon_mod_stats(&mut self, cache: &quicktag_scanner::TagCache) {
        let manager = package_manager();
        let metadata = quicktag_core::implant::ImplantStatResolver::load(&manager)
            .and_then(|resolver| manager.read_tag(resolver.semantic_table))
            .inspect_err(|error| log::warn!("Failed to load weapon rating routes: {error:#}"))
            .ok();
        let resolver = WeaponStatResolver::load(cache);

        let mut candidates = self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .filter_map(|item| Some((item.name.clone(), item.rarity, item.definition_tag?)))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(name, rarity, tag)| {
            (
                name.clone(),
                usize::from(*rarity != Some(GearRarity::Standard)),
                *tag,
            )
        });
        let mut weapons = FxHashMap::default();
        for (name, _, tag) in candidates {
            if weapons.contains_key(&name) {
                continue;
            }
            match resolver.mod_weapon_context(tag) {
                Ok(context) => {
                    weapons.insert(name, context);
                }
                Err(error) => log::debug!("Skipping weapon mod context {tag}: {error:#}"),
            }
        }
        let mut weapon_names = weapons.keys().cloned().collect::<Vec<_>>();
        weapon_names.sort_unstable_by_key(|name| name.to_lowercase());

        self.weapon_mod_stats.clear();
        for item in self
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
        {
            let Some(definition) = item.definition_tag else {
                continue;
            };
            let ratings = match resolver.mod_raw_stats(definition) {
                Ok(ratings) => ratings,
                Err(error) => {
                    log::warn!("Failed to decode weapon mod {definition}: {error:#}");
                    continue;
                }
            };
            let targets = if item.mod_is_universal {
                weapon_names.as_slice()
            } else {
                item.compatible_weapons.as_slice()
            };
            let mut contextual = vec![];
            if !ratings.is_empty()
                && let Some(metadata) = metadata.as_deref()
            {
                for weapon in targets {
                    let Some(context) = weapons.get(weapon) else {
                        continue;
                    };
                    match resolver.mod_stat_changes_for_ratings(context, &ratings, metadata) {
                        Ok(evaluation) if !evaluation.curves.is_empty() => {
                            contextual.push(WeaponModWeaponStats {
                                weapon: weapon.clone(),
                                changes: evaluation.changes,
                                curves: evaluation.curves,
                            });
                        }
                        Ok(_) => {}
                        Err(error) => log::debug!(
                            "Failed contextual weapon mod stats {definition} on {weapon}: {error:#}"
                        ),
                    }
                }
            }
            self.weapon_mod_stats.insert(
                item.display_tag,
                WeaponModDetails {
                    ratings,
                    weapons: contextual,
                },
            );
        }
    }

    fn reconcile_weapon_mod_models(&mut self, cache: &quicktag_scanner::TagCache) {
        let explicitly_compatible = self
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon Mod") && !item.mod_is_universal
            })
            .flat_map(|item| item.compatible_weapons.iter().cloned())
            .collect::<FxHashSet<_>>();
        let weapons = self
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon")
                    && ((item.rarity == Some(GearRarity::Standard)
                        && (is_canonical_weapon_skin_owner(item)
                            || item.internal_hash == Some(KKV_9SD_HASH)))
                        || explicitly_compatible.contains(&item.name))
            })
            .filter_map(|item| Some((item.name.clone(), item.model_tag?)))
            .unique()
            .collect::<Vec<_>>();
        let socket_index = crate::geometry::WeaponModSocketIndex::new();
        let weapon_families = weapons
            .into_iter()
            .filter_map(|(name, model)| {
                let families = socket_index
                    .signature_for_model(cache, model)?
                    .into_iter()
                    .map(|(family, _variant)| family)
                    .collect::<FxHashSet<_>>();
                (!families.is_empty()).then_some((name, families))
            })
            .collect::<Vec<_>>();
        let mut mod_families = FxHashMap::default();

        for modification in self.items.iter_mut().filter(|item| {
            item.item_type.as_deref() == Some("Weapon Mod")
                && !item.mod_is_universal
                && item.mod_category.as_deref() != Some("Unique")
        }) {
            let Some(model) = modification.model_tag else {
                continue;
            };
            let families = mod_families
                .entry(model)
                .or_insert_with(|| crate::geometry::weapon_mod_authored_families(cache, model));
            if families.is_empty() {
                continue;
            }
            // Investment paths/categories author gameplay compatibility;
            // Pattern families author whether that mod can actually spawn on
            // the selected render rig. Both branches must agree. Never expand
            // compatibility from a shared visual family alone.
            modification.compatible_weapons.retain(|compatible| {
                weapon_families
                    .iter()
                    .find(|(name, _)| name == compatible)
                    .is_some_and(|(_, weapon)| !weapon.is_disjoint(families))
            });
        }
    }

    /// Carries filters and selection across a localized data rebuild.
    pub fn inherit_ui_state(&mut self, previous: &Self) {
        let selected_tag = previous
            .selected
            .and_then(|selected| previous.items.get(selected))
            .map(|item| item.display_tag);

        self.selected_item_types = previous.selected_item_types.clone();
        self.selected_item_types.retain(|selected| {
            self.item_types
                .iter()
                .any(|(item_type, _)| item_type == selected)
        });
        self.selected_rarities = previous.selected_rarities.clone();
        self.selected_rarities
            .retain(|selected| self.rarities.iter().any(|(rarity, _)| rarity == selected));
        self.selected_internal_categories = previous.selected_internal_categories.clone();
        self.selected_internal_categories.retain(|selected| {
            self.internal_category_counts
                .iter()
                .any(|(category, _)| category == selected)
        });
        self.internal_category_search = previous.internal_category_search.clone();
        self.search = previous.search.clone();
        self.update_filter();

        self.selected = selected_tag
            .and_then(|tag| self.items.iter().position(|item| item.display_tag == tag))
            .filter(|selected| self.filtered_indices.contains(selected))
            .or_else(|| selected_tag.and_then(|_| self.filtered_indices.first().copied()));
    }

    pub fn load_error(&self) -> Option<&str> {
        self.status.as_deref()
    }

    fn update_filter(&mut self) {
        let search = self.search.trim().to_lowercase();
        self.filtered_indices = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                let matches_chips = matches_chip_filters(
                    item,
                    &self.selected_item_types,
                    &self.selected_rarities,
                    &self.selected_internal_categories,
                );
                let matches_search = search.is_empty()
                    || item.name.to_lowercase().contains(&search)
                    || gear_item_type(item)
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || gear_item_subcategory(item)
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .applies_to
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .mod_category
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .mod_family
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || (item.mod_is_universal && "universal".contains(&search))
                    || item
                        .compatible_weapons
                        .iter()
                        .any(|value| value.to_lowercase().contains(&search))
                    || item
                        .types
                        .iter()
                        .any(|value| value.to_lowercase().contains(&search))
                    || item
                        .internal_categories
                        .iter()
                        .any(|value| value.to_lowercase().contains(&search))
                    || item
                        .internal_name
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .description
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .rarity
                        .is_some_and(|rarity| rarity.label().to_lowercase().contains(&search))
                    || format!("{:08x}", item.display_tag.0).contains(&search);
                matches_chips && matches_search
            })
            .map(|(index, _)| index)
            .collect();

        self.filtered_indices.sort_by(|left, right| {
            let left = &self.items[*left];
            let right = &self.items[*right];
            rarity_sort_key(left.rarity)
                .cmp(&rarity_sort_key(right.rarity))
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
                .then_with(|| left.name.cmp(&right.name))
        });

        if self
            .selected
            .is_some_and(|selected| !self.filtered_indices.contains(&selected))
        {
            self.selected = self.filtered_indices.first().copied();
        }
    }
}

impl GearView {
    fn filtered_export_json(&self) -> Result<String, serde_json::Error> {
        let records = self
            .filtered_indices
            .iter()
            .map(|&index| {
                let item = &self.items[index];
                let implant = self.implant_details.get(&item.display_tag);
                let effects = implant
                    .map(|details| details.effects.as_slice())
                    .or_else(|| {
                        self.weapon_mod_effects
                            .get(&item.display_tag)
                            .map(Vec::as_slice)
                    })
                    .unwrap_or_default()
                    .iter()
                    .map(|effect| GearExportEffect {
                        name: &effect.name,
                        description: export_localized_text(
                            &effect.description,
                            &effect.description_parts,
                        ),
                    })
                    .collect();
                let implant_stats = implant
                    .map(|details| details.stats.as_slice())
                    .unwrap_or_default()
                    .iter()
                    .map(|stat| GearExportStat {
                        name: &stat.name,
                        raw_name_hash: format!("{:08X}", stat.raw_name_hash),
                        value: stat.value,
                    })
                    .collect();

                GearExportRecord {
                    hash: item.display_tag.to_string(),
                    name: &item.name,
                    rarity: item.rarity.map(GearRarity::label),
                    item_type: gear_item_type(item),
                    subcategory: gear_item_subcategory(item),
                    target_weapon: (item.item_type.as_deref() == Some("Weapon Skin"))
                        .then_some(item.applies_to.as_deref())
                        .flatten(),
                    target_shell: matches!(
                        item.item_type.as_deref(),
                        Some("Runner Core" | "Runner Skin")
                    )
                    .then_some(item.applies_to.as_deref())
                    .flatten(),
                    mod_category: item.mod_category.as_deref(),
                    mod_family: item.mod_family.as_deref(),
                    compatible_weapons: &item.compatible_weapons,
                    display_class: item.classification.as_deref(),
                    types: &item.types,
                    price: item.price,
                    buying_price: item.buying_price,
                    description: item.description.as_deref().map(|description| {
                        export_localized_text(description, &item.description_parts)
                    }),
                    internal_type: item.internal_name.as_deref(),
                    internal_hash: item.internal_hash.map(|hash| format!("{hash:08X}")),
                    definition_tag: item.definition_tag.map(|tag| tag.to_string()),
                    internal_categories: &item.internal_categories,
                    effects,
                    implant_stats,
                }
            })
            .collect::<Vec<_>>();
        serde_json::to_string_pretty(&records)
    }

    fn export_filtered_json(&self) -> Result<Option<PathBuf>, String> {
        let filename = format!("quicktag-gear-{}-records.json", self.filtered_indices.len());
        let Some(mut path) = native_dialog::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_filename(&filename)
            .show_save_single_file()
            .map_err(|error| format!("Could not open export dialog: {error}"))?
        else {
            return Ok(None);
        };
        if !path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        {
            path.set_extension("json");
        }
        let json = self
            .filtered_export_json()
            .map_err(|error| format!("Could not encode Gear JSON: {error}"))?;
        std::fs::write(&path, json)
            .map_err(|error| format!("Could not write {}: {error}", path.display()))?;
        Ok(Some(path))
    }

    pub fn view(
        &mut self,
        _ctx: &egui::Context,
        ui: &mut egui::Ui,
        texture_cache: &TextureCache,
    ) -> Option<ViewAction> {
        let mut action = None;

        egui::SidePanel::left("gear_left_panel")
            .resizable(true)
            .default_width(440.0)
            .min_width(360.0)
            .max_width(680.0)
            .show_inside(ui, |ui| {
                let mut filter_changed = false;

                // Plain horizontal rows already center controls on their Y axis.
                // `horizontal_centered` reserves the entire remaining panel height
                // and would push every following filter and Gear result off-screen.
                ui.horizontal(|ui| {
                    ui.label("Search");
                    let control_height = ui.spacing().interact_size.y;
                    let search_width =
                        (ui.available_width() - control_height - ui.spacing().item_spacing.x)
                            .clamp(160.0, 460.0);
                    let response = ui.add_sized(
                        [search_width, control_height],
                        egui::TextEdit::singleline(&mut self.search)
                            .hint_text("Name, type, weapon, rarity, description, hash…"),
                    );
                    filter_changed |= response.changed();
                    let clear_size = egui::vec2(response.rect.height(), response.rect.height());
                    if ui
                        .add_enabled(
                            !self.search.is_empty(),
                            egui::Button::new("×").min_size(clear_size),
                        )
                        .on_hover_text("Clear search")
                        .clicked()
                    {
                        self.search.clear();
                        filter_changed = true;
                    }
                });

                let mut item_type_toggles = vec![];
                let mut rarity_toggles = vec![];
                let mut category_toggles = vec![];
                egui::Frame::group(ui.style())
                    .inner_margin(6)
                    .show(ui, |ui| {
                        // Keep this row content-sized for the same reason as the
                        // primary search row above.
                        ui.horizontal(|ui| {
                            let reserved_width = 102.0;
                            let search_width = (ui.available_width() - reserved_width).max(120.0);
                            let response = ui.add_sized(
                                [search_width, ui.spacing().interact_size.y],
                                egui::TextEdit::singleline(&mut self.internal_category_search)
                                    .hint_text("Find filter, category, or hash…"),
                            );
                            let clear_size =
                                egui::vec2(response.rect.height(), response.rect.height());
                            if ui
                                .add_enabled(
                                    !self.internal_category_search.is_empty(),
                                    egui::Button::new("×").min_size(clear_size),
                                )
                                .on_hover_text("Clear filter search")
                                .clicked()
                            {
                                self.internal_category_search.clear();
                                response.request_focus();
                            }
                            let active_filters = self.selected_item_types.len()
                                + self.selected_rarities.len()
                                + self.selected_internal_categories.len();
                            if ui
                                .add_enabled(active_filters > 0, egui::Button::new("Clear").small())
                                .on_hover_text("Clear all chip filters")
                                .clicked()
                            {
                                self.selected_item_types.clear();
                                self.selected_rarities.clear();
                                self.selected_internal_categories.clear();
                                filter_changed = true;
                            }
                        });

                        let chip_search = self.internal_category_search.trim().to_lowercase();
                        egui::ScrollArea::vertical()
                            .id_salt("gear_internal_category_chip_scroll")
                            .max_height(235.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                let visible_item_types = self
                                    .item_types
                                    .iter()
                                    .filter(|(item_type, _)| {
                                        chip_search.is_empty()
                                            || item_type.to_lowercase().contains(&chip_search)
                                    })
                                    .collect::<Vec<_>>();
                                if !visible_item_types.is_empty() {
                                    ui.weak("Item type");
                                    ui.horizontal_wrapped(|ui| {
                                        for (item_type, count) in visible_item_types {
                                            let selected =
                                                self.selected_item_types.contains(item_type);
                                            if filter_chip(
                                                ui,
                                                format!("{item_type}  {count}"),
                                                selected,
                                            )
                                            .on_hover_text(if selected {
                                                "Remove this item type"
                                            } else {
                                                "Add this item type"
                                            })
                                            .clicked()
                                            {
                                                item_type_toggles.push(item_type.clone());
                                            }
                                        }
                                    });
                                    ui.add_space(5.0);
                                }

                                let visible_rarities = self
                                    .rarities
                                    .iter()
                                    .filter(|(rarity, _)| {
                                        chip_search.is_empty()
                                            || rarity.label().to_lowercase().contains(&chip_search)
                                    })
                                    .collect::<Vec<_>>();
                                if !visible_rarities.is_empty() {
                                    ui.weak("Rarity");
                                    ui.horizontal_wrapped(|ui| {
                                        for (rarity, count) in visible_rarities {
                                            let selected = self.selected_rarities.contains(rarity);
                                            if filter_chip(
                                                ui,
                                                RichText::new(format!(
                                                    "{}  {count}",
                                                    rarity.label()
                                                ))
                                                .color(rarity.color()),
                                                selected,
                                            )
                                            .on_hover_text(if selected {
                                                "Remove this rarity"
                                            } else {
                                                "Add this rarity"
                                            })
                                            .clicked()
                                            {
                                                rarity_toggles.push(*rarity);
                                            }
                                        }
                                    });
                                    ui.add_space(5.0);
                                }

                                let visible_categories = self
                                    .internal_category_counts
                                    .iter()
                                    .filter(|(category, _)| {
                                        chip_search.is_empty()
                                            || category.to_lowercase().contains(&chip_search)
                                    })
                                    .collect::<Vec<_>>();
                                if !visible_categories.is_empty() {
                                    ui.weak("Internal category");
                                    ui.horizontal_wrapped(|ui| {
                                        for (category, count) in visible_categories {
                                            let selected = self
                                                .selected_internal_categories
                                                .contains(category);
                                            if filter_chip(
                                                ui,
                                                format!("{category}  {count}"),
                                                selected,
                                            )
                                            .on_hover_text(if selected {
                                                "Remove this category"
                                            } else {
                                                "Add matching category results"
                                            })
                                            .clicked()
                                            {
                                                category_toggles.push(category.clone());
                                            }
                                        }
                                    });
                                }
                            });
                    });

                for item_type in item_type_toggles {
                    if !self.selected_item_types.remove(&item_type) {
                        self.selected_item_types.insert(item_type);
                    }
                    filter_changed = true;
                }
                for rarity in rarity_toggles {
                    if !self.selected_rarities.remove(&rarity) {
                        self.selected_rarities.insert(rarity);
                    }
                    filter_changed = true;
                }
                for category in category_toggles {
                    if !self.selected_internal_categories.remove(&category) {
                        self.selected_internal_categories.insert(category);
                    }
                    filter_changed = true;
                }

                if filter_changed {
                    self.update_filter();
                }

                let mut export_requested = false;
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Showing {} of {} gear records",
                        self.filtered_indices.len(),
                        self.items.len()
                    ));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        export_requested = ui
                            .add_enabled(
                                !self.filtered_indices.is_empty(),
                                egui::Button::new("Export JSON"),
                            )
                            .on_hover_text("Export every record in the current search and filters")
                            .clicked();
                    });
                });
                if export_requested {
                    match self.export_filtered_json() {
                        Ok(Some(path)) => {
                            TOASTS
                                .lock()
                                .success(format!("Exported {}", path.display()));
                        }
                        Ok(None) => {}
                        Err(error) => {
                            TOASTS.lock().error(error);
                        }
                    }
                }
                ui.separator();

                let show_subcategories = !self.selected_item_types.is_empty();
                let mut section_map = FxHashMap::<String, Vec<usize>>::default();
                for &index in &self.filtered_indices {
                    for section in gear_sections(&self.items[index], show_subcategories) {
                        section_map.entry(section).or_default().push(index);
                    }
                }
                let mut sections = section_map.into_iter().collect::<Vec<_>>();
                sections.sort_by(|left, right| {
                    left.0
                        .to_lowercase()
                        .cmp(&right.0.to_lowercase())
                        .then(left.0.cmp(&right.0))
                });

                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                    for (section, indices) in sections {
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.heading(gear_section_label(
                                &section,
                                &self.selected_internal_categories,
                            ));
                            ui.weak(format!("{}", indices.len()));
                        });
                        ui.separator();

                        for index in indices {
                            let item = &self.items[index];
                            let selected = self.selected == Some(index);
                            if gear_item_button(ui, item, selected, texture_cache).clicked() {
                                self.selected = Some(index);
                            }
                            ui.add_space(3.0);
                        }
                    }
                });
            });

        let mut detail_category_toggle = None;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(index) = self.selected else {
                if let Some(status) = &self.status {
                    ui.label(RichText::new(status).color(Color32::YELLOW));
                } else {
                    ui.label(RichText::new("Select gear").italics());
                }
                return;
            };
            let item = &self.items[index];

            let selected_mod = egui::ScrollArea::vertical()
                .id_salt(("gear_detail", item.display_tag.0))
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if let Some(icon) = item.icon_tag {
                            let (_, texture_id) = texture_cache.get_or_default(icon);
                            ui.add(
                                egui::Image::new((texture_id, egui::vec2(64.0, 64.0)))
                                    .sense(egui::Sense::click()),
                            )
                            .tag_context_with_preview(
                                icon,
                                Some(texture_cache),
                                true,
                            );
                        }
                        ui.heading(
                            RichText::new(&item.name).color(
                                item.rarity.map(GearRarity::color).unwrap_or(Color32::WHITE),
                            ),
                        );
                        if ui.small_button(format!("{}", item.display_tag)).clicked() {
                            action = Some(ViewAction::OpenTag(item.display_tag));
                        }
                        if let Some(model) = gear_model_navigation_tag(item)
                            && ui.button("Go to Models").clicked()
                        {
                            action = Some(ViewAction::ShowModel(model));
                        }
                    });
                    ui.separator();

                    let mod_compatibility = if item.mod_is_universal {
                        Some("Universal".to_owned())
                    } else {
                        (!item.compatible_weapons.is_empty())
                            .then(|| item.compatible_weapons.join(", "))
                    };
                    let metadata = gear_metadata_schema(item);

                    egui::Grid::new("gear_metadata")
                        .num_columns(2)
                        .spacing([16.0, 6.0])
                        .show(ui, |ui| {
                            metadata_row(ui, "Rarity", item.rarity.map(GearRarity::label));
                            metadata_row(ui, "Category", gear_item_type(item));
                            metadata_row(
                                ui,
                                if item.mod_category.is_some() {
                                    "Mod category"
                                } else {
                                    "Subtype"
                                },
                                gear_item_subcategory(item),
                            );
                            if let Some(owner_label) = metadata.owner_label {
                                metadata_row(ui, owner_label, item.applies_to.as_deref());
                            }
                            if metadata.compatible_weapons {
                                metadata_row(
                                    ui,
                                    "Compatible weapons",
                                    mod_compatibility.as_deref(),
                                );
                            }
                            metadata_row(ui, "Display class", item.classification.as_deref());
                            metadata_row(
                                ui,
                                "Types",
                                (!item.types.is_empty())
                                    .then(|| item.types.join(", "))
                                    .as_deref(),
                            );
                            if metadata.price
                                && let Some(price) = item.price
                            {
                                metadata_row(ui, "Price", Some(&format_number(price)));
                            }
                            if let Some(price) = item.buying_price {
                                metadata_row(ui, "Buying price", Some(&format_number(price)));
                            }
                            metadata_row(ui, "Internal type", item.internal_name.as_deref());
                            metadata_row(
                                ui,
                                "Internal hash",
                                item.internal_hash.map(|v| format!("{v:08X}")).as_deref(),
                            );
                            ui.label("Definition tag");
                            if let Some(tag) = item.definition_tag {
                                if ui.link(tag.to_string()).tag_context(tag).clicked() {
                                    action = Some(ViewAction::OpenTag(tag));
                                }
                            } else {
                                ui.weak("—");
                            }
                            ui.end_row();
                            if metadata.icon_texture {
                                ui.label("Icon texture");
                                if let Some(tag) = item.icon_tag {
                                    if ui
                                        .link(tag.to_string())
                                        .tag_context_with_preview(tag, Some(texture_cache), true)
                                        .clicked()
                                    {
                                        action = Some(ViewAction::ShowTexture(tag));
                                    }
                                } else {
                                    ui.weak("—");
                                }
                                ui.end_row();
                            }
                        });

                    ui.add_space(12.0);
                    ui.strong("Internal categories")
                        .on_hover_text("Click a chip to add or remove its matching results");
                    if item.internal_categories.is_empty() {
                        ui.weak("—");
                    } else {
                        ui.horizontal_wrapped(|ui| {
                            for category in &item.internal_categories {
                                let selected = self.selected_internal_categories.contains(category);
                                if filter_chip(ui, category, selected)
                                    .on_hover_text(if selected {
                                        "Remove this category"
                                    } else {
                                        "Add matching category results"
                                    })
                                    .clicked()
                                {
                                    detail_category_toggle = Some(category.clone());
                                }
                            }
                        });
                    }

                    if let Some(details) = self.implant_details.get(&item.display_tag) {
                        implant_details_table(ui, details);
                    } else {
                        ui.add_space(12.0);
                        ui.strong("Description");
                        ui.separator();
                        localized_text_label(
                            ui,
                            item.description.as_deref().unwrap_or("—"),
                            &item.description_parts,
                        );
                        for &tag in &item.detail_texture_tags {
                            ui.add_space(8.0);
                            let (texture, texture_id) = texture_cache.get_or_default(tag);
                            let width = 256.0;
                            let size = egui::vec2(width, width / texture.aspect_ratio.max(0.01));
                            ui.add(
                                egui::Image::new((texture_id, size)).sense(egui::Sense::click()),
                            )
                            .tag_context_with_preview(
                                tag,
                                Some(texture_cache),
                                true,
                            );
                        }
                        if let Some(effects) = self.weapon_mod_effects.get(&item.display_tag) {
                            item_effect_panel(
                                ui,
                                if effects.len() == 1 {
                                    "Mod ability"
                                } else {
                                    "Mod abilities"
                                },
                                effects,
                            );
                        }
                        if let Some(details) = self.weapon_mod_stats.get(&item.display_tag) {
                            weapon_mod_stats_panel(ui, details);
                        }
                    }

                    ui.add_space(12.0);
                    if let Some(stats) = self.weapon_stats.get(&item.display_tag) {
                        weapon_stats_table(ui, stats);
                        ui.add_space(12.0);
                    }

                    compatible_mod_cards(ui, &self.items, item, self.selected, texture_cache)
                })
                .inner;
            if let Some(index) = selected_mod {
                self.selected = Some(index);
            }
        });

        if let Some(category) = detail_category_toggle {
            if !self.selected_internal_categories.remove(&category) {
                self.selected_internal_categories.insert(category);
            }
            self.update_filter();
        }

        action
    }
}

fn gear_item_type(item: &GearItem) -> Option<&str> {
    match item.item_type.as_deref() {
        Some("Melee") => Some("Weapon Skin"),
        item_type => item_type,
    }
}

fn gear_item_subcategory(item: &GearItem) -> Option<&str> {
    if item.item_type.as_deref() == Some("Melee") {
        Some("Melee")
    } else {
        item.subcategory.as_deref()
    }
}

fn gear_model_navigation_tag(item: &GearItem) -> Option<TagHash> {
    matches!(
        item.item_type.as_deref(),
        Some("Weapon Skin" | "Runner Skin" | "Charm" | "Melee")
    )
    .then_some(item.model_tag)
    .flatten()
}

fn is_authored_weapon(item: &GearItem) -> bool {
    item.internal_categories
        .iter()
        .any(|category| category.starts_with("item_type.weapon."))
        || (item.item_type.as_deref() == Some("Weapon")
            && item
                .internal_categories
                .iter()
                .any(|category| category == "behaviors.durable_item"))
}

fn implant_details_table(ui: &mut egui::Ui, details: &ImplantDetails) {
    if !details.effects.is_empty() {
        item_effect_panel(
            ui,
            if details.effects.len() == 1 {
                "Implant effect"
            } else {
                "Implant effects"
            },
            &details.effects,
        );
    }

    ui.add_space(12.0);
    ui.heading("Implant stats");
    ui.separator();
    if details.stats.is_empty() {
        ui.weak("—");
    } else {
        egui::Grid::new("implant_stats_grid")
            .num_columns(2)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                for stat in &details.stats {
                    ui.label(&stat.name);
                    ui.strong(format!("{:+}", stat.value));
                    ui.end_row();
                }
            });
    }
}

fn weapon_mod_stats_panel(ui: &mut egui::Ui, details: &WeaponModDetails) {
    ui.add_space(12.0);
    ui.heading("Mod stats");
    ui.separator();
    if details.ratings.is_empty() {
        ui.weak("No direct rating modifiers");
    } else {
        egui::Grid::new("weapon_mod_rating_grid")
            .num_columns(2)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                for stat in &details.ratings {
                    ui.label(&stat.name)
                        .on_hover_text(format!("Authored rating ID {}", stat.rating_id));
                    ui.strong(format!("{:+}", stat.value));
                    ui.end_row();
                }
            });
    }

    if !details.weapons.is_empty() {
        ui.add_space(8.0);
        ui.strong("Weapon-specific changes");
        for weapon in &details.weapons {
            egui::CollapsingHeader::new(&weapon.weapon)
                .default_open(details.weapons.len() == 1)
                .show(ui, |ui| {
                    egui::Grid::new(("weapon_mod_physical_grid", &weapon.weapon))
                        .num_columns(2)
                        .spacing([24.0, 6.0])
                        .show(ui, |ui| {
                            for change in &weapon.changes {
                                ui.label(change.name);
                                let source =
                                    change.derived_from.map(str::to_owned).unwrap_or_else(|| {
                                        format!(
                                            "Rating {:.0} → {:.0} (ID {})",
                                            change.base_rating,
                                            change.modified_rating,
                                            change.rating_id
                                        )
                                    });
                                ui.strong(format_weapon_mod_change(change))
                                    .on_hover_text(format!(
                                        "{} → {}\n{source}",
                                        format_display_float(change.before, false),
                                        format_display_float(change.after, false)
                                    ));
                                ui.end_row();
                            }
                        });
                    egui::CollapsingHeader::new("Package curve values").show(ui, |ui| {
                        for curve in &weapon.curves {
                            ui.label(format!(
                                "Semantic {} / {} · rating {}: {} → {}",
                                curve.semantic,
                                curve.occurrence,
                                curve.rating_id,
                                format_display_float(curve.base_rating, false),
                                format_display_float(curve.modified_rating, false)
                            ));
                            for (channel, (before, after)) in
                                curve.before.iter().zip(&curve.after).enumerate()
                            {
                                if (after - before).abs() > 0.000001 {
                                    ui.monospace(format!(
                                        "Channel {channel}: {} → {} ({})",
                                        format_display_float(*before, false),
                                        format_display_float(*after, false),
                                        format_display_float(after - before, true)
                                    ));
                                }
                            }
                        }
                    });
                });
        }
    }
}

fn format_weapon_mod_change(change: &WeaponModStatChange) -> String {
    let delta = change.delta();
    match change.unit {
        "rounds" => format!("{}", format_display_float(delta, true)),
        "m" => format!("{} m", format_display_float(delta, true)),
        "RPM" => format!("{} RPM", format_display_float(delta, true)),
        "x" => format!("{}×", format_display_float(delta, true)),
        "degrees" => format!("{}°", format_display_float(delta, true)),
        "percentage points" => format!("{}%", format_display_float(delta, true)),
        "s" => format!("{} s", format_display_float(delta, true)),
        "" => format_display_float(delta, true),
        unit => format!("{} {unit}", format_display_float(delta, true)),
    }
}

fn format_display_float(value: f32, signed: bool) -> String {
    let mut text = if signed {
        format!("{value:+.3}")
    } else {
        format!("{value:.3}")
    };
    if text.contains('.') {
        text = text.trim_end_matches('0').trim_end_matches('.').to_owned();
    }
    text
}

fn item_effect_panel(ui: &mut egui::Ui, heading: &str, effects: &[ItemEffect]) {
    ui.add_space(12.0);
    ui.heading(heading);
    ui.separator();
    for (index, effect) in effects.iter().enumerate() {
        if index > 0 {
            ui.add_space(8.0);
        }
        ui.strong(&effect.name);
        localized_text_label(ui, &effect.description, &effect.description_parts);
    }
}

fn localized_text_label(
    ui: &mut egui::Ui,
    text: &str,
    parts: &[LocalizedStringPart],
) -> egui::Response {
    if parts.is_empty()
        || parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<String>()
            != text
    {
        return ui.label(text);
    }

    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = ui.available_width();
    for part in parts {
        job.append(
            &part.text,
            0.0,
            egui::TextFormat {
                font_id: egui::TextStyle::Body.resolve(ui.style()),
                color: if part.highlighted {
                    localized_part_color(part).unwrap_or(LOCALIZED_FORMAT_COLOR)
                } else {
                    ui.visuals().text_color()
                },
                ..Default::default()
            },
        );
    }
    ui.label(job)
}

fn export_localized_text(text: &str, parts: &[LocalizedStringPart]) -> String {
    let valid_parts = !parts.is_empty()
        && parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<String>()
            == text;
    if !valid_parts {
        return export_html_text(text);
    }

    parts
        .iter()
        .map(|part| {
            let text = export_html_text(&part.text);
            if part.highlighted {
                let color = localized_part_color_hex(part)
                    .unwrap_or_else(|| LOCALIZED_FORMAT_COLOR_HEX.to_owned());
                format!(r#"<span style="color: {color};">{text}</span>"#)
            } else {
                text
            }
        })
        .collect()
}

fn localized_part_color(part: &LocalizedStringPart) -> Option<Color32> {
    let [red, green, blue, alpha] = part.authored_color?;
    if is_default_format_color([red, green, blue, alpha]) {
        return None;
    }
    Some(Color32::from_rgba_unmultiplied(
        color_channel(red),
        color_channel(green),
        color_channel(blue),
        color_channel(alpha),
    ))
}

fn localized_part_color_hex(part: &LocalizedStringPart) -> Option<String> {
    let [red, green, blue, alpha] = part.authored_color?;
    if is_default_format_color([red, green, blue, alpha]) {
        return None;
    }
    let red = color_channel(red);
    let green = color_channel(green);
    let blue = color_channel(blue);
    let alpha = color_channel(alpha);
    if alpha == u8::MAX {
        Some(format!("#{red:02x}{green:02x}{blue:02x}"))
    } else {
        Some(format!("#{red:02x}{green:02x}{blue:02x}{alpha:02x}"))
    }
}

fn is_default_format_color([red, green, blue, alpha]: [f32; 4]) -> bool {
    [red, green, blue, alpha]
        .iter()
        .all(|channel| (*channel - 1.0).abs() <= f32::EPSILON)
}

fn color_channel(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn export_html_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace("\r\n", "<br>")
        .replace(['\r', '\n'], "<br>")
}

fn weapon_stats_table(ui: &mut egui::Ui, stats: &WeaponStats) {
    ui.add_space(12.0);
    ui.heading("Weapon stats");
    ui.separator();
    ui.add_space(4.0);

    ui.horizontal(|ui| {
        ui.strong("Firepower");
        ui.label(format_optional(stats.firepower, 1, ""));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Damage");
        ui.label(format_optional(stats.damage, 1, ""));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Precision");
        ui.label(format_optional(stats.headshot_multiplier, 2, "×"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Rate of Fire");
        ui.label(format_optional(stats.rounds_per_minute, 0, " RPM"));
    });
    if stats.bullets_per_shot.is_some() {
        ui.horizontal(|ui| {
            ui.add_space(18.0);
            ui.weak("Pellets per Shot");
            ui.label(format_optional(stats.bullets_per_shot, 0, ""));
        });
    }

    ui.horizontal(|ui| {
        ui.strong("Accuracy");
        ui.label(format_optional(stats.accuracy, 1, ""));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Hipfire Spread");
        ui.label(format_optional(stats.hip_fire_spread_degrees, 2, "°"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("ADS Spread");
        ui.label(format_optional(stats.ads_spread_degrees, 2, "°"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Moving Inaccuracy");
        ui.label(format_optional(
            stats.movement_accuracy_loss.map(|value| value * 100.0),
            1,
            "%",
        ));
    });

    ui.horizontal(|ui| {
        ui.strong("Handling");
        ui.label(format_optional(stats.handling, 0, ""));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Equip Speed");
        ui.label(format_optional(stats.equip_seconds, 2, " s"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("ADS Speed");
        ui.label(format_optional(stats.aim_seconds, 2, " s"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Weight");
        ui.label(format_optional(
            stats.weight.map(|value| value * 100.0),
            1,
            "%",
        ));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Recoil");
        ui.label(format_optional(
            stats.recoil.map(|value| value * 100.0),
            1,
            "%",
        ));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Aim Assist");
        ui.label(format_optional(stats.aim_correction_degrees, 2, "°"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Reload Speed");
        ui.label(format_optional(stats.reload_seconds, 2, " s"));
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.weak("Crouch Spread Bonus");
        ui.label(format_optional(
            stats.crouch_spread_bonus.map(|value| value * 100.0),
            1,
            "%",
        ));
    });

    ui.horizontal(|ui| {
        ui.strong("Range");
        ui.label(format_optional(stats.range_metres, 0, " m"));
    });
    if stats.shotgun_spread_degrees.is_some() {
        ui.horizontal(|ui| {
            ui.strong("Spread Angle");
            ui.label(format_optional(stats.shotgun_spread_degrees, 1, "°"));
        });
    }
    if stats.volt_drain_percent.is_some() {
        ui.horizontal(|ui| {
            ui.strong("Volt Drain");
            ui.label(format_optional(stats.volt_drain_percent, 1, "%"));
        });
    } else {
        ui.horizontal(|ui| {
            ui.strong("Magazine");
            ui.label(format_optional(stats.magazine, 0, ""));
        });
    }
    ui.horizontal(|ui| {
        ui.strong("Zoom");
        ui.label(format_optional(stats.zoom, 1, "×"));
    });
}

fn format_optional(value: Option<f32>, _decimals: usize, suffix: &str) -> String {
    value
        .map(|value| format!("{}{suffix}", format_display_float(value, false)))
        .unwrap_or_else(|| "—".to_owned())
}

fn gear_item_button(
    ui: &mut egui::Ui,
    item: &GearItem,
    selected: bool,
    texture_cache: &TextureCache,
) -> egui::Response {
    let color = item.rarity.map(GearRarity::color).unwrap_or(Color32::GRAY);
    let fill_alpha = if selected { 82 } else { 34 };
    let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), fill_alpha);
    let stroke = egui::Stroke::new(if selected { 2.0 } else { 1.0 }, color);
    let label = item.name.as_str();
    let text = RichText::new(label).color(Color32::WHITE).strong();
    let button = if let Some(icon) = item.icon_tag {
        let (_, texture_id) = texture_cache.get_or_default(icon);
        egui::Button::image_and_text((texture_id, egui::vec2(34.0, 34.0)), text)
    } else {
        egui::Button::new(text)
    };
    ui.add_sized(
        [
            ui.available_width(),
            if item.icon_tag.is_some() { 48.0 } else { 38.0 },
        ],
        button.fill(fill).stroke(stroke),
    )
    .tag_context(item.display_tag)
    .on_hover_text(format!(
        "{label}\n{} · {}",
        item.rarity
            .map(GearRarity::label)
            .unwrap_or("Unknown rarity"),
        gear_item_subcategory(item)
            .or(item.mod_category.as_deref())
            .unwrap_or("Uncategorized")
    ))
}

fn gear_section_label(section: &str, category_filters: &FxHashSet<String>) -> String {
    if category_filters.is_empty() {
        return section.to_owned();
    }

    let mut categories = category_filters
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    categories.sort_unstable_by_key(|category| category.to_lowercase());
    format!("{section} ({})", categories.join(", "))
}

fn mod_slot_sort_key(category: &str) -> u8 {
    match category {
        "Chip" => 0,
        "Optic" => 1,
        "Barrel" => 2,
        "Grip" => 3,
        "Generator" => 4,
        "Magazine" => 5,
        "Shield" => 6,
        "Unique" => 7,
        _ => 8,
    }
}

fn weapon_mod_slot_name(category: &str) -> &str {
    match category {
        // Authored mod families are more granular than the four attachment
        // slots shown by the game. Normalize them to those player-facing slots.
        "Muzzle" | "Barrel" => "Barrel",
        "Foregrip" => "Grip",
        "Stock" | "Shield" => "Shield",
        category => category,
    }
}

fn compatible_mod_sections(items: &[GearItem], weapon: &GearItem) -> Vec<(String, Vec<usize>)> {
    if weapon.item_type.as_deref() != Some("Weapon") {
        return vec![];
    }

    let mut sections = FxHashMap::<String, Vec<usize>>::default();
    for (index, item) in items.iter().enumerate().filter(|(_, item)| {
        item.item_type.as_deref() == Some("Weapon Mod")
            && !item.mod_is_universal
            && item
                .compatible_weapons
                .iter()
                .any(|name| name == &weapon.name)
    }) {
        let category = item
            .mod_category
            .as_deref()
            .or(item.subcategory.as_deref())
            .unwrap_or("Other");
        sections
            .entry(weapon_mod_slot_name(category).to_owned())
            .or_default()
            .push(index);
    }

    for indices in sections.values_mut() {
        indices.sort_by(|left, right| {
            let left = &items[*left];
            let right = &items[*right];
            rarity_sort_key(left.rarity)
                .cmp(&rarity_sort_key(right.rarity))
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
                .then_with(|| left.name.cmp(&right.name))
        });
    }

    let mut sections = sections.into_iter().collect::<Vec<_>>();
    sections.sort_by(|left, right| {
        mod_slot_sort_key(&left.0)
            .cmp(&mod_slot_sort_key(&right.0))
            .then_with(|| left.0.to_lowercase().cmp(&right.0.to_lowercase()))
            .then_with(|| left.0.cmp(&right.0))
    });
    sections
}

fn compatible_mod_cards(
    ui: &mut egui::Ui,
    items: &[GearItem],
    weapon: &GearItem,
    selected: Option<usize>,
    texture_cache: &TextureCache,
) -> Option<usize> {
    let sections = compatible_mod_sections(items, weapon);
    if sections.is_empty() {
        return None;
    }

    ui.add_space(18.0);
    ui.horizontal(|ui| {
        ui.heading("Compatible mods");
        ui.weak(format!("{} slots", sections.len()));
    });
    ui.separator();
    ui.add_space(4.0);

    let columns = if ui.available_width() >= 920.0 {
        sections.len().min(4)
    } else if ui.available_width() >= 560.0 {
        sections.len().min(2)
    } else {
        1
    };
    let mut clicked = None;
    for row in sections.chunks(columns) {
        ui.columns(row.len(), |uis| {
            for (column, (slot, indices)) in uis.iter_mut().zip(row) {
                egui::Frame::group(column.style())
                    .inner_margin(8)
                    .show(column, |ui| {
                        ui.horizontal(|ui| {
                            ui.strong(slot);
                            ui.weak(format!("{} mods", indices.len()));
                        });
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .id_salt(("weapon_mod_slot", weapon.display_tag.0, slot))
                            .max_height(320.0)
                            .min_scrolled_height(320.0)
                            .show(ui, |ui| {
                                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                for &index in indices {
                                    if gear_item_button(
                                        ui,
                                        &items[index],
                                        selected == Some(index),
                                        texture_cache,
                                    )
                                    .clicked()
                                    {
                                        clicked = Some(index);
                                    }
                                    ui.add_space(3.0);
                                }
                            });
                    });
            }
        });
        ui.add_space(8.0);
    }

    clicked
}

fn metadata_row(ui: &mut egui::Ui, label: &str, value: Option<&str>) {
    ui.label(label);
    ui.label(value.unwrap_or("—"));
    ui.end_row();
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GearMetadataSchema {
    owner_label: Option<&'static str>,
    compatible_weapons: bool,
    price: bool,
    icon_texture: bool,
}

fn gear_metadata_schema(item: &GearItem) -> GearMetadataSchema {
    GearMetadataSchema {
        owner_label: match item.item_type.as_deref() {
            Some("Weapon Skin") => Some("Weapon"),
            Some("Melee") => None,
            Some("Runner Core" | "Runner Skin") => Some("Shell"),
            _ => None,
        },
        compatible_weapons: item.item_type.as_deref() == Some("Weapon Mod"),
        price: item.price.is_some(),
        icon_texture: item.item_type.as_deref() == Some("Implant"),
    }
}

fn filter_chip(
    ui: &mut egui::Ui,
    label: impl Into<egui::WidgetText>,
    selected: bool,
) -> egui::Response {
    ui.add(
        egui::Button::new(label)
            .small()
            .selected(selected)
            .corner_radius(10),
    )
}

fn collect_item_types(items: &[GearItem]) -> Vec<(String, usize)> {
    let mut counts: FxHashMap<String, usize> = FxHashMap::default();
    for item_type in items.iter().filter_map(gear_item_type) {
        *counts.entry(item_type.to_owned()).or_default() += 1;
    }
    let mut item_types: Vec<_> = counts.into_iter().collect();
    item_types.sort_by(|a, b| {
        a.0.to_lowercase()
            .cmp(&b.0.to_lowercase())
            .then(a.0.cmp(&b.0))
    });
    item_types
}

fn collect_rarities(items: &[GearItem]) -> Vec<(GearRarity, usize)> {
    [
        GearRarity::Contraband,
        GearRarity::Unique,
        GearRarity::Prestige,
        GearRarity::Superior,
        GearRarity::Deluxe,
        GearRarity::Enhanced,
        GearRarity::Standard,
        GearRarity::Quest,
        GearRarity::Dynamic,
    ]
    .into_iter()
    .filter_map(|rarity| {
        let count = items
            .iter()
            .filter(|item| item.rarity == Some(rarity))
            .count();
        (count > 0).then_some((rarity, count))
    })
    .collect()
}

fn collect_internal_category_counts(items: &[GearItem]) -> Vec<(String, usize)> {
    let mut counts = FxHashMap::<String, usize>::default();
    for category in items
        .iter()
        .flat_map(|item| item.internal_categories.iter())
    {
        *counts.entry(category.clone()).or_default() += 1;
    }
    let mut categories = counts.into_iter().collect::<Vec<_>>();
    categories.sort_by(|left, right| {
        left.0
            .to_lowercase()
            .cmp(&right.0.to_lowercase())
            .then_with(|| left.0.cmp(&right.0))
    });
    categories
}

fn matches_internal_category_filters(item: &GearItem, selected: &FxHashSet<String>) -> bool {
    selected.is_empty()
        || selected
            .iter()
            .any(|category| item.internal_categories.contains(category))
}

fn matches_chip_filters(
    item: &GearItem,
    selected_item_types: &FxHashSet<String>,
    selected_rarities: &FxHashSet<GearRarity>,
    selected_internal_categories: &FxHashSet<String>,
) -> bool {
    let matches_type = selected_item_types.is_empty()
        || gear_item_type(item).is_some_and(|value| selected_item_types.contains(value));
    let matches_rarity = selected_rarities.is_empty()
        || item
            .rarity
            .is_some_and(|rarity| selected_rarities.contains(&rarity));
    matches_type
        && matches_rarity
        && matches_internal_category_filters(item, selected_internal_categories)
}

fn rarity_sort_key(rarity: Option<GearRarity>) -> u8 {
    match rarity {
        Some(GearRarity::Contraband) => 0,
        Some(GearRarity::Unique) => 1,
        Some(GearRarity::Prestige) => 2,
        Some(GearRarity::Superior) => 3,
        Some(GearRarity::Deluxe) => 4,
        Some(GearRarity::Enhanced) => 5,
        Some(GearRarity::Standard) => 6,
        Some(GearRarity::Quest) => 7,
        Some(GearRarity::Dynamic) => 8,
        None => 9,
    }
}

fn gear_sections(item: &GearItem, show_subcategories: bool) -> Vec<String> {
    let item_type = gear_item_type(item).unwrap_or("Uncategorized");
    let section_owner = match item_type {
        "Runner Skin" => item.applies_to.as_deref().or(item.subcategory.as_deref()),
        "Weapon Skin" => item.applies_to.as_deref(),
        "Runner Core" => item.applies_to.as_deref(),
        _ => None,
    };

    if let Some(owner) = section_owner {
        if show_subcategories {
            vec![owner.to_owned()]
        } else {
            vec![format!("{item_type} · {owner}")]
        }
    } else if item_type == "Weapon Mod" {
        let category = item
            .mod_family
            .as_deref()
            .or(item.subcategory.as_deref())
            .or(item.mod_category.as_deref())
            .unwrap_or("Uncategorized");
        vec![if show_subcategories {
            category.to_owned()
        } else {
            format!("{item_type} · {category}")
        }]
    } else if show_subcategories {
        vec![gear_item_subcategory(item).unwrap_or(item_type).to_owned()]
    } else {
        vec![item_type.to_owned()]
    }
}

fn format_number(value: u32) -> String {
    let text = value.to_string();
    let mut result = String::with_capacity(text.len() + text.len() / 3);
    for (index, character) in text.chars().enumerate() {
        if index > 0 && (text.len() - index) % 3 == 0 {
            result.push(',');
        }
        result.push(character);
    }
    result
}

fn load_gear_for_language(
    strings: &StringCache,
    language: LocalizedLanguage,
) -> Result<Vec<GearItem>, String> {
    if !matches!(package_manager().version, GameVersion::Marathon(_)) {
        return Err("Gear preview currently supports Marathon packages only".to_owned());
    }

    let localized = quicktag_strings::localized::create_stringresolver_d2_for_language(language)
        .map_err(|error| format!("Failed to load scoped Marathon strings: {error:#}"))?;
    if language == LocalizedLanguage::English {
        let mut items = load_gear_resolved(strings, &localized)?;
        localize_weapon_mod_categories(&mut items, strings, strings);
        return Ok(items);
    }

    // Taxonomy must never depend on translated prose. Build it from the
    // English authoring text and stable internal records, then replace only
    // the game-authored text shown to the user with the requested language.
    let english_strings =
        quicktag_strings::localized::create_stringmap_for_language(LocalizedLanguage::English)
            .map_err(|error| format!("Failed to load English Gear metadata: {error:#}"))?;
    let english_localized = quicktag_strings::localized::create_stringresolver_d2_for_language(
        LocalizedLanguage::English,
    )
    .map_err(|error| format!("Failed to load scoped English Gear metadata: {error:#}"))?;
    let mut items = load_gear_resolved(&english_strings, &english_localized)?;
    localize_gear_display(&mut items, strings, &localized);
    localize_weapon_mod_categories(&mut items, &english_strings, strings);
    Ok(items)
}

fn load_gear_resolved(
    strings: &StringCache,
    localized: &LocalizedStringResolver,
) -> Result<Vec<GearItem>, String> {
    let display_to_hash = load_display_to_hash_map();
    let hash_to_definition = load_hash_to_definition_map();
    let mut wordlist = FxHashMap::default();
    quicktag_strings::wordlist::load_wordlist(|word, hash| {
        wordlist.entry(hash).or_insert_with(|| word.to_owned());
    });
    let internal_category_registry = load_internal_category_registry(&wordlist);
    let pattern_resolver = InvestmentPatternResolver::load();
    expand_gear_wordlist(&mut wordlist);
    let mut items = vec![];

    for (tag, entry) in package_manager().get_all_by_reference(GEAR_DISPLAY_REFERENCE) {
        if entry.file_type != 8 || entry.file_subtype != 0 {
            continue;
        }
        let Ok(data) = package_manager().read_tag(tag) else {
            continue;
        };
        let Some(name_hash) = localized_hash_at(&data, 0xc0) else {
            continue;
        };
        let mut name = localized_at(&data, 0xbc, 0xc0, strings, localized)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| format!("Unlocalized #{name_hash:08X}"));
        let description = localized_at(&data, 0xd8, 0xdc, strings, localized);
        let description_parts = localized_parts_at(&data, 0xd8, 0xdc, localized);
        if name.starts_with("Unlocalized #")
            && let Some(fallback) = description
                .as_deref()
                .filter(|value| value.len() <= 80 && !value.contains(['\n', '.']))
        {
            name = fallback.to_owned();
        }

        let internal_hash = display_to_hash.get(&tag).copied();
        let definition_tag = internal_hash.and_then(|hash| hash_to_definition.get(&hash).copied());
        let definition = definition_tag
            .and_then(|tag| package_manager().read_tag(tag).ok())
            .unwrap_or_default();
        let structurally_weapon =
            structural_item_type_at(&data, 0xc8, &wordlist).as_deref() == Some("Weapon");
        let model_tag = (!definition.is_empty())
            .then(|| pattern_resolver.resolve_definition(internal_hash, &definition))
            .flatten();
        let (
            internal_name,
            definition_group_key,
            definition_type_code,
            rarity,
            price,
            internal_categories,
        ) = definition_tag
            .and_then(|definition_tag| package_manager().read_tag(definition_tag).ok())
            .map(|definition| {
                let internal_categories =
                    extract_internal_categories(&definition, &internal_category_registry);
                let definition_internal_name = wordlist_name_at(&definition, 0xf0, &wordlist);
                let pattern_internal_name = pattern_resolver
                    .definition_internal_path_hash(&definition)
                    .and_then(|hash| wordlist.get(&hash).cloned())
                    .filter(|name| {
                        name.starts_with("weapon_mods.")
                            || (structurally_weapon && name.starts_with("weapons."))
                    });
                // The API hash is the stable investment identity. The July
                // layout moved the old 0xf0 string field; prefer its direct
                // wordlist/generated-path resolution whenever available.
                let api_internal_name = internal_hash
                    .and_then(|hash| wordlist.get(&hash))
                    .filter(|name| !name.starts_with('#'))
                    .cloned();
                let internal_name = api_internal_name
                    .or_else(|| {
                        definition_internal_name
                            .as_ref()
                            .filter(|name| !name.starts_with('#'))
                            .cloned()
                    })
                    .or(pattern_internal_name)
                    .or(definition_internal_name);
                (
                    internal_name,
                    definition_group_key(&definition),
                    definition_type_code(&definition),
                    parse_rarity_from_categories(&internal_categories)
                        .or_else(|| parse_rarity(&definition)),
                    parse_price(&definition),
                    internal_categories,
                )
            })
            .unwrap_or_default();
        let raw_category_hash = localized_hash_at(&data, 0xc8);
        let localized_category_hash = localized_hash_at(&data, 0xd4);
        let classification = wordlist
            .get(&read_u32(&data, 0x114).unwrap_or_default())
            .filter(|value| matches!(value.as_str(), "generic" | "ranked" | "weapon"))
            .cloned();
        let structural_category = structural_item_type_at(&data, 0xc8, &wordlist);
        let item_type = structural_category
            .clone()
            .or_else(|| (classification.as_deref() == Some("weapon")).then(|| "Weapon".to_owned()))
            .or_else(|| localized_at(&data, 0xcc, 0xd4, strings, localized));
        let subcategory = structural_category
            .is_some()
            .then(|| {
                (structural_category.as_deref() == Some("Weapon"))
                    .then(|| {
                        weapon_subcategory_from_hash(localized_category_hash.unwrap_or_default())
                    })
                    .flatten()
                    .or_else(|| localized_at(&data, 0xcc, 0xd4, strings, localized))
            })
            .flatten()
            .or_else(|| {
                (structural_category.as_deref() == Some("Weapon")
                    && rarity == Some(GearRarity::Contraband))
                .then(|| "Hybrid Weapon".to_owned())
            });

        items.push(GearItem {
            display_tag: tag,
            icon_tag: None,
            detail_texture_tags: vec![],
            definition_tag,
            model_tag,
            definition_group_key,
            definition_type_code,
            internal_hash,
            internal_name,
            raw_category_hash,
            raw_subcategory_hash: localized_category_hash,
            name,
            rarity,
            item_type,
            subcategory,
            applies_to: None,
            mod_category: None,
            mod_family: None,
            mod_is_universal: false,
            compatible_weapons: vec![],
            classification,
            types: extract_types(&data, &wordlist),
            internal_categories,
            price,
            buying_price: parse_buying_price(&definition),
            description,
            description_parts,
        });
    }

    let implant_icons = ImplantIconResolver::load();
    let mut facets = Vec::with_capacity(items.len());
    for item in &mut items {
        let inferred = infer_taxonomy(item);
        if item.item_type.is_none() {
            item.item_type = inferred.category;
        }
        if item.subcategory.is_none() {
            item.subcategory = inferred.subcategory;
        }
        facets.push(inferred.facets);
    }
    let grouped_taxonomy = grouped_cosmetic_taxonomy(&items);
    for item in &mut items {
        if item.item_type.is_none()
            && let Some(inferred) = item
                .cosmetic_group_key()
                .and_then(|key| grouped_taxonomy.get(&key))
        {
            item.item_type = Some(inferred.category.clone());
            item.subcategory.clone_from(&inferred.subcategory);
        }
    }
    classify_unseeded_weapon_skins(&mut items);
    correct_current_cosmetic_taxonomy(&mut items);
    resolve_internal_item_type_taxonomy(&mut items);
    correct_structural_item_taxonomy(&mut items);
    propagate_weapon_subcategories(&mut items);
    assign_weapon_mod_metadata(&mut items);
    let profile_textures = resolve_profile_textures(
        items
            .iter()
            .filter_map(|item| Some((item.definition_type_code?, item.internal_hash?))),
    );
    let shell_taxonomy = runner_shell_taxonomy(&items);
    for (item, facets) in items.iter_mut().zip(&mut facets) {
        if item.item_type.is_none()
            && let Some(model) = item.description.as_deref().and_then(runner_shell_model)
        {
            let subcategory = shell_taxonomy
                .get(model)
                .cloned()
                .unwrap_or_else(|| humanize_identifier(model));
            item.item_type = Some("Runner Skin".to_owned());
            item.subcategory = Some(subcategory.clone());
            facets.push(format!("runner_{}", identifier(&subcategory)));
        }
        if item.item_type.as_deref() == Some("Weapon") && item.subcategory.is_none() {
            item.subcategory = weapon_subcategory(item);
        }
        if item.item_type.as_deref() == Some("Implant") {
            if let Some(slot) = implant_slot(item) {
                item.subcategory = Some(slot.to_owned());
            } else if item.subcategory.is_none() {
                item.subcategory = Some("Implant".to_owned());
            }
            item.icon_tag = item
                .definition_tag
                .and_then(|tag| implant_icons.resolve_implant(tag));
        }
        if item.item_type.as_deref() == Some("Sticker") {
            item.detail_texture_tags = item
                .model_tag
                .and_then(resolve_sticker_texture)
                .into_iter()
                .collect();
        } else if item.item_type.as_deref() == Some("Profile") {
            item.detail_texture_tags = item
                .internal_hash
                .and_then(|internal_hash| profile_textures.get(&internal_hash).cloned())
                .unwrap_or_default();
        }
        append_taxonomy_tags(item, std::mem::take(facets));
        if let Some(hash) = item.name.strip_prefix("Unlocalized #") {
            item.name = format!(
                "Unnamed {} #{hash}",
                item.item_type.as_deref().unwrap_or("Gear")
            );
        }
    }
    propagate_runner_skin_taxonomy(&mut items);
    assign_runner_owners(&mut items);
    assign_skin_weapons(&mut items);
    items.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then(a.display_tag.cmp(&b.display_tag))
    });
    Ok(items)
}

fn implant_slot(item: &GearItem) -> Option<&'static str> {
    item.internal_categories
        .iter()
        .find_map(|category| match category.as_str() {
            "item_type.implant.shield" | "item_type.implant.shields" => Some("Shields"),
            "item_type.implant.head" => Some("Head"),
            "item_type.implant.upper" | "item_type.implant.torso" => Some("Torso"),
            "item_type.implant.lower" | "item_type.implant.leg" => Some("Leg"),
            _ => None,
        })
}

fn sanitize_icon_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .filter(|character| !character.is_whitespace())
        .map(|character| match character {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            character if character.is_control() => '_',
            character => character,
        })
        .collect();
    sanitized.trim().trim_matches('.').to_owned()
}

/// Correct stale localized UI labels with authored definition semantics. These
/// signals describe the record layout/behavior and remain valid across names,
/// rarities, and localization.
fn correct_structural_item_taxonomy(items: &mut [GearItem]) {
    for item in items {
        let taxonomy = if item.definition_type_code == Some(0x1aa)
            && item
                .internal_categories
                .iter()
                .any(|category| category == "item_type.#E6BE7E0C")
        {
            Some(("Implant", Some("Implant")))
        } else if item.definition_type_code == Some(0x156) {
            Some(("Key Objective", None))
        } else if item
            .internal_categories
            .iter()
            .any(|category| category.starts_with("item_type.item_status_effect."))
            || (item.item_type.as_deref() == Some("Trinket")
                && item
                    .description
                    .as_deref()
                    .is_some_and(|description| description.starts_with("When equipped,")))
        {
            Some(("Item Modifier", None))
        } else {
            None
        };

        if let Some((category, subcategory)) = taxonomy {
            item.item_type = Some(category.to_owned());
            item.subcategory = subcategory.map(str::to_owned);
        }
    }
}

/// Display-only rarity rows omit the weapon subtype. Recover it from the one
/// base weapon with the same localized identity, rejecting ambiguous names.
fn propagate_weapon_subcategories(items: &mut [GearItem]) {
    let candidates = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.name.clone(), item.subcategory.clone()?)))
        .into_group_map();
    let subcategories = candidates
        .into_iter()
        .filter_map(|(name, subcategories)| {
            let mut subcategories = subcategories.into_iter().unique();
            let subcategory = subcategories.next()?;
            subcategories
                .next()
                .is_none()
                .then_some((name, subcategory))
        })
        .collect::<FxHashMap<_, _>>();

    for item in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Weapon")
            && item.subcategory.is_none()
            && item
                .internal_categories
                .iter()
                .any(|category| category == "behaviors.display_only")
    }) {
        item.subcategory = subcategories.get(&item.name).cloned();
    }
}

fn extract_item_effects(
    items: &[GearItem],
    strings: &StringCache,
    language: LocalizedLanguage,
) -> (
    FxHashMap<TagHash, ImplantDetails>,
    FxHashMap<TagHash, Vec<ItemEffect>>,
) {
    let Ok(localized) =
        quicktag_strings::localized::create_stringresolver_d2_for_language(language)
    else {
        return (FxHashMap::default(), FxHashMap::default());
    };
    let resolver = ItemEffectResolver::load();
    let stat_resolver = ImplantStatsResolver::load(language, &localized);
    let mut implants = FxHashMap::default();
    let mut weapon_mods = FxHashMap::default();

    for item in items.iter().filter(|item| {
        implant_slot(item).is_some() || item.item_type.as_deref() == Some("Weapon Mod")
    }) {
        let Some(definition_tag) = item.definition_tag else {
            continue;
        };
        let effects = resolver.extract(definition_tag, strings, &localized);
        if implant_slot(item).is_some() {
            let stats = stat_resolver.resolve(definition_tag);
            if effects.is_empty() && stats.is_empty() {
                continue;
            }
            implants.insert(item.display_tag, ImplantDetails { effects, stats });
        } else if !effects.is_empty() {
            weapon_mods.insert(item.display_tag, effects);
        }
    }

    (implants, weapon_mods)
}

/// The July display record no longer carries a structural UI type for every
/// inventory row. Its compact internal `item_type.*` category remains intact.
/// Learn each category's presentation label from independently typed rows and
/// apply it only when that authored category has one unambiguous meaning.
fn resolve_internal_item_type_taxonomy(items: &mut [GearItem]) {
    let mut labels = FxHashMap::<String, FxHashSet<String>>::default();
    for item in items.iter().filter(|item| item.item_type.is_some()) {
        let label = item.item_type.as_ref().expect("filtered");
        for category in item
            .internal_categories
            .iter()
            .filter(|category| category.starts_with("item_type."))
        {
            labels
                .entry(category.clone())
                .or_default()
                .insert(label.clone());
        }
    }
    let labels = labels
        .into_iter()
        .filter_map(|(category, labels)| {
            let mut labels = labels.into_iter();
            let label = labels.next()?;
            labels.next().is_none().then_some((category, label))
        })
        .collect::<FxHashMap<_, _>>();

    for item in items.iter_mut().filter(|item| item.item_type.is_none()) {
        let candidates = item
            .internal_categories
            .iter()
            .filter_map(|category| labels.get(category))
            .unique()
            .collect::<Vec<_>>();
        if let [category] = candidates.as_slice() {
            item.item_type = Some((*category).clone());
        }
    }
}

fn localize_gear_display(
    items: &mut [GearItem],
    strings: &StringCache,
    localized: &LocalizedStringResolver,
) {
    let previous_weapon_names = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.internal_hash?, item.name.clone())))
        .collect::<FxHashMap<_, _>>();

    for item in items.iter_mut() {
        let Ok(data) = package_manager().read_tag(item.display_tag) else {
            continue;
        };
        if let Some(name) =
            localized_at(&data, 0xbc, 0xc0, strings, localized).filter(|name| !name.is_empty())
        {
            item.name = name;
        }
        if let Some(description) = localized_at(&data, 0xd8, 0xdc, strings, localized) {
            item.description = Some(description);
            item.description_parts = localized_parts_at(&data, 0xd8, 0xdc, localized);
        }
    }

    let localized_weapon_names = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.internal_hash?, item.name.clone())))
        .collect::<FxHashMap<_, _>>();
    let replacements = previous_weapon_names
        .into_iter()
        .filter_map(|(hash, previous)| {
            localized_weapon_names
                .get(&hash)
                .cloned()
                .map(|localized| (previous, localized))
        })
        .collect::<FxHashMap<_, _>>();

    for item in items.iter_mut() {
        if let Some(owner) = item.applies_to.as_mut()
            && let Some(localized_owner) = replacements.get(owner)
        {
            owner.clone_from(localized_owner);
        }
        for weapon in &mut item.compatible_weapons {
            if let Some(localized_weapon) = replacements.get(weapon) {
                weapon.clone_from(localized_weapon);
            }
        }
        item.compatible_weapons = sorted_unique_names(std::mem::take(&mut item.compatible_weapons));
    }

    items.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then(a.display_tag.cmp(&b.display_tag))
    });
}

fn translated_values_for_english(
    english: &StringCache,
    translated: &StringCache,
    english_value: &str,
) -> Vec<String> {
    english
        .iter()
        .filter(|(_, values)| values.iter().any(|value| value == english_value))
        .flat_map(|(hash, _)| translated.get(hash).into_iter().flatten().cloned())
        .filter(|value| !value.is_empty())
        .unique()
        .sorted()
        .collect()
}

fn localize_weapon_mod_categories(
    items: &mut [GearItem],
    english: &StringCache,
    translated: &StringCache,
) {
    for item in items
        .iter_mut()
        .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
    {
        let Some(category) = item.subcategory.as_deref() else {
            continue;
        };
        let source = format!("{category} Mod");
        let localized = translated_values_for_english(english, translated, &source)
            .into_iter()
            .exactly_one()
            .ok();
        if let Some(localized) = localized {
            item.subcategory = Some(localized);
        }
    }
}

fn structural_item_type_at(
    data: &[u8],
    offset: usize,
    wordlist: &FxHashMap<u32, String>,
) -> Option<String> {
    let value = wordlist.get(&read_u32(data, offset)?)?;
    let item_type = value.strip_prefix("item_type_")?;
    Some(humanize_identifier(item_type))
}

/// Extend the general hash wordlist with the numbered investment paths used by
/// Marathon cosmetics. Shipping builds routinely contain more numbered skins,
/// charms, and stickers than the static reverse-engineering wordlist.
fn expand_gear_wordlist(wordlist: &mut FxHashMap<u32, String>) {
    let seeds = wordlist
        .values()
        .filter(|word| {
            [
                "charms.",
                "heroes.",
                "implant_cores.",
                "stickers.",
                "weapon_mods.",
                "weapons.",
            ]
            .iter()
            .any(|prefix| word.starts_with(prefix))
        })
        .cloned()
        .collect::<Vec<_>>();
    let runner_ids = seeds
        .iter()
        .filter_map(|seed| seed.strip_prefix("heroes.")?.split('.').next())
        .map(str::to_owned)
        .collect::<FxHashSet<_>>();
    let mut stems = FxHashSet::default();

    for seed in seeds {
        let stem = seed.trim_end_matches(|character: char| character.is_ascii_digit());
        if stem.len() < seed.len() {
            stems.insert(stem.to_owned());
            if stem.contains(".skins.default.") {
                stems.insert(stem.replace(".skins.default.", ".skins.store."));
            } else if stem.contains(".skins.store.") {
                stems.insert(stem.replace(".skins.store.", ".skins.default."));
            }
        }

        let parts = seed.split('.').collect::<Vec<_>>();
        if seed.starts_with("weapons.") && !seed.contains(".skins.") && parts.len() == 4 {
            stems.insert(format!("{seed}.skins.default.v100.skin"));
            stems.insert(format!("{seed}.skins.store.v100.skin"));
        }
    }

    // Runner-core paths use the same authored runner identifiers as shell
    // skins. Several shipping core paths are absent from the static wordlist,
    // so reconstruct their numbered identifiers from that independent roster.
    for runner in runner_ids {
        stems.insert(format!("implant_cores.{runner}.v100.core"));
    }

    for stem in stems {
        for number in 0..=256 {
            for suffix in [number.to_string(), format!("{number:02}")] {
                let candidate = format!("{stem}{suffix}");
                let hash = quicktag_core::util::fnv1(candidate.as_bytes());
                let replace_collision = wordlist.get(&hash).is_some_and(|existing| {
                    candidate.starts_with("weapons.")
                        && candidate.contains(".skins.")
                        && !existing.starts_with("weapons.")
                });
                if replace_collision || !wordlist.contains_key(&hash) {
                    wordlist.insert(hash, candidate);
                }
            }
        }
    }
}

fn humanize_identifier(value: &str) -> String {
    value
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            characters
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + characters.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default)]
struct TaxonomyInference {
    category: Option<String>,
    subcategory: Option<String>,
    facets: Vec<String>,
}

fn infer_taxonomy(item: &GearItem) -> TaxonomyInference {
    let mut result = TaxonomyInference::default();

    // Stable internal item-type categories survive localization and layout
    // changes. Current Profile records use one category and definition class
    // per cosmetic kind, including hash-only records without authored paths.
    if item
        .internal_categories
        .iter()
        .any(|category| category == "item_type.#A4200795")
    {
        result.category = Some("Weapon Skin".to_owned());
    } else if let Some(subcategory) =
        item.internal_categories
            .iter()
            .find_map(|category| match category.as_str() {
                PROFILE_BACKGROUND_CATEGORY => Some("Backgrounds"),
                PROFILE_EMBLEM_CATEGORY => Some("Emblems"),
                PROFILE_TITLE_CATEGORY => Some("Title"),
                _ => None,
            })
    {
        result.category = Some("Profile".to_owned());
        result.subcategory = Some(subcategory.to_owned());
    }

    match item.classification.as_deref() {
        Some("weapon") => result.category = Some("Weapon".to_owned()),
        Some("ranked") => result.category = Some("Ranked Reward".to_owned()),
        _ => {}
    }

    let internal_name = item
        .internal_name
        .as_deref()
        .filter(|name| !name.starts_with('#'));
    let parts = internal_name
        .map(|name| name.split('.').collect::<Vec<_>>())
        .unwrap_or_default();

    if result.category.is_none() {
        match parts.as_slice() {
            ["artifacts", _, _, kind, ..] => {
                result.category = Some("Artifact".to_owned());
                result.subcategory = Some(humanize_identifier(kind));
            }
            ["charms", ..] => result.category = Some("Charm".to_owned()),
            ["display_items", "loot_types", _, kind, ..] => {
                result.category = Some("Loot Type".to_owned());
                result.subcategory = Some(humanize_identifier(kind));
            }
            ["heroes", runner, "skins", ..] => {
                let runner = humanize_identifier(runner);
                result.category = Some("Runner Skin".to_owned());
                result.subcategory = Some(runner.clone());
                result
                    .facets
                    .push(format!("runner_{}", identifier(&runner)));
            }
            ["melees", family, ..] => {
                result.category = Some("Melee".to_owned());
                result.subcategory = Some(humanize_identifier(family));
            }
            ["profiles", kind, ..] => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some(humanize_identifier(kind));
            }
            ["progression_rewards", kind, ..] => {
                result.category = Some("Progression Reward".to_owned());
                result.subcategory = Some(humanize_identifier(kind));
            }
            ["quest", "factions", faction, ..] => {
                result.category = Some("Quest Item".to_owned());
                result.subcategory = Some("Faction".to_owned());
                result.facets.push(format!("faction_{faction}"));
            }
            ["quest", "dynamic_events", zone, ..] => {
                result.category = Some("Quest Item".to_owned());
                result.subcategory = Some("Dynamic Event".to_owned());
                result.facets.push(format!("zone_{zone}"));
            }
            ["quest", "run_objectives", zone, ..] => {
                result.category = Some("Quest Item".to_owned());
                result.subcategory = Some("Run Objective".to_owned());
                result.facets.push(format!("zone_{zone}"));
            }
            ["stickers", ..] => result.category = Some("Sticker".to_owned()),
            ["weapons", ..] => {
                result.category = Some("Weapon Skin".to_owned());
                result.subcategory = weapon_subcategory_from_internal(internal_name.unwrap());
            }
            _ => {}
        }
    }

    if let ["keys", "locked_room", zone, ..] = parts.as_slice() {
        result.facets.push(format!("zone_{zone}"));
    }

    if result.category.is_none() && item.name.trim_end().ends_with("Schema") {
        result.category = Some("Schema".to_owned());
    }
    if result.category.is_none() && item.rarity == Some(GearRarity::Quest) {
        result.category = Some("Quest Item".to_owned());
    }

    if result.category.is_none() {
        match item.definition_type_code {
            Some(0x02e) => result.category = Some("Tag Chip".to_owned()),
            Some(0x02f) => result.category = Some("Reward Package".to_owned()),
            Some(0x032) => result.category = Some("Item Modifier".to_owned()),
            Some(0x127) => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some("Backgrounds".to_owned());
            }
            Some(0x128) => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some("Emblems".to_owned());
            }
            Some(0x129) => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some("Title".to_owned());
            }
            Some(0x12c) => {
                result.category = Some("Runner Skin".to_owned());
                result.subcategory = Some("Rook".to_owned());
                result.facets.push("runner_rook".to_owned());
            }
            Some(0x132) => result.category = Some("Charm".to_owned()),
            Some(0x134) => result.category = Some("Sticker".to_owned()),
            Some(0x117) => {
                result.category = Some("Progression Reward".to_owned());
                result.subcategory = Some("Cosmetics".to_owned());
            }
            Some(0x153) => result.category = Some("Challenge".to_owned()),
            Some(0x158) => result.category = Some("Item Modifier".to_owned()),
            Some(0x186..=0x190 | 0x197..=0x19b) => {
                result.category = Some("Sponsored Kit".to_owned())
            }
            Some(0x19c | 0x1a9 | 0x1aa) => {
                result.category = Some("Implant".to_owned());
                result.subcategory = Some("Implant".to_owned());
            }
            _ => {}
        }
    }
    if result.category.is_none() && item.raw_category_hash == Some(TRINKET_CATEGORY_HASH) {
        result.category = Some("Trinket".to_owned());
    }
    if result.category.is_none() && item.types.iter().any(|kind| kind == "static") {
        result.category = Some("Item Modifier".to_owned());
    }
    if result.category.is_none() && item.definition_type_code == Some(0x030) {
        // Authored loot-type/display-item rows. The contained `item_type.*`
        // category describes what the table yields, not the row's UI class.
        result.category = Some("Loot Type".to_owned());
    }
    if result.category.is_none() && item.definition_type_code == Some(0x031) {
        result.category = Some("Key".to_owned());
        result.subcategory = Some("Template".to_owned());
    }
    if result.category.is_none() && item.definition_type_code == Some(0x156) {
        result.category = Some("Key Objective".to_owned());
    }
    if result.category.is_none() && item.definition_type_code == Some(0x113) {
        result.category = Some("Progression Reward".to_owned());
        result.subcategory = Some("Cosmetics".to_owned());
    }
    if result.category.is_none() && item.definition_type_code == Some(0) {
        let description = item.description.as_deref().unwrap_or_default();
        if description.starts_with("Ranked Rewards given") {
            result.category = Some("Ranked Reward".to_owned());
        } else if item.name.contains("Rewards Package")
            || description.contains("Rank-Up Rewards")
            || description.to_ascii_lowercase().contains("reward package")
        {
            result.category = Some("Reward Package".to_owned());
        } else if description.starts_with("When equipped,") {
            result.category = Some("Item Modifier".to_owned());
        } else {
            result.category = Some("Progression Reward".to_owned());
        }
    }

    if result.category.as_deref() == Some("Weapon") {
        result.subcategory =
            weapon_subcategory(item).or_else(|| weapon_subcategory_from_name(&item.name));
    }

    result
}

#[derive(Clone)]
struct GroupedTaxonomy {
    category: String,
    subcategory: Option<String>,
}

/// Infer hash-only weapon/melee skins from a group key shared with a skin whose
/// internal path is present in the wordlist. Both the structural skin code and
/// group index must match, and conflicting seed categories are rejected.
fn grouped_cosmetic_taxonomy(items: &[GearItem]) -> FxHashMap<(u16, u16), GroupedTaxonomy> {
    let mut candidates = FxHashMap::<(u16, u16), Vec<&GearItem>>::default();
    for item in items {
        if matches!(
            item.item_type.as_deref(),
            Some("Melee") | Some("Weapon Skin")
        ) && item
            .definition_type_code
            .is_some_and(is_weapon_or_melee_skin_type)
            && item
                .internal_name
                .as_deref()
                .is_some_and(|name| !name.starts_with('#'))
            && let Some(key) = item.cosmetic_group_key()
        {
            candidates.entry(key).or_default().push(item);
        }
    }

    candidates
        .into_iter()
        .filter_map(|(key, candidates)| {
            let categories = candidates
                .iter()
                .filter_map(|item| item.item_type.as_deref())
                .collect::<FxHashSet<_>>();
            let subcategories = candidates
                .iter()
                .filter_map(|item| item.subcategory.as_deref())
                .collect::<FxHashSet<_>>();
            if categories.len() != 1 {
                return None;
            }
            let category = (*categories.iter().next()?).to_owned();
            let subcategory = (subcategories.len() == 1)
                .then(|| subcategories.iter().next().map(|value| (*value).to_owned()))
                .flatten();
            Some((
                key,
                GroupedTaxonomy {
                    category,
                    subcategory,
                },
            ))
        })
        .collect()
}

impl GearItem {
    fn definition_group_index(&self) -> Option<u16> {
        Some((self.definition_group_key? >> 16) as u16)
    }

    fn cosmetic_group_key(&self) -> Option<(u16, u16)> {
        Some((self.definition_type_code?, self.definition_group_index()?))
    }
}

fn classify_unseeded_weapon_skins(items: &mut [GearItem]) {
    let generated_weapon_owners = generated_skin_path_owners();
    let generated_melee_hashes = generated_skin_hashes(MELEE_SKIN_OWNER_PATHS);
    let melee_models = items
        .iter()
        .filter(|item| direct_melee_skin(item, &generated_melee_hashes))
        .filter_map(|item| item.model_tag)
        .collect::<FxHashSet<_>>();
    // The compact `item_type` category is shared by weapon and melee
    // cosmetics. A concrete investment path/hash is therefore authoritative
    // before group propagation; group ordinals are reused across these two
    // cosmetic domains in the updated tables.
    for item in items.iter_mut().filter(|item| {
        item.definition_type_code
            .is_some_and(is_weapon_or_melee_skin_type)
    }) {
        if direct_melee_skin(item, &generated_melee_hashes)
            || item
                .model_tag
                .is_some_and(|model| melee_models.contains(&model))
        {
            item.item_type = Some("Melee".to_owned());
            item.subcategory = item
                .internal_name
                .as_deref()
                .and_then(|name| name.strip_prefix("melees."))
                .and_then(|path| path.split('.').next())
                .map(humanize_identifier)
                .or_else(|| Some("Knives".to_owned()));
        } else if direct_skin_weapon_hash(item, &generated_weapon_owners).is_some() {
            item.item_type = Some("Weapon Skin".to_owned());
        }
    }
    let seeded_groups = items
        .iter()
        .filter(|item| {
            item.definition_type_code
                .is_some_and(is_weapon_or_melee_skin_type)
        })
        .filter_map(|item| {
            let category = if direct_melee_skin(item, &generated_melee_hashes)
                || item
                    .model_tag
                    .is_some_and(|model| melee_models.contains(&model))
            {
                "Melee"
            } else if direct_skin_weapon_hash(item, &generated_weapon_owners).is_some() {
                "Weapon Skin"
            } else {
                return None;
            };
            Some((item.cosmetic_group_key()?, category))
        })
        .into_group_map();
    let seeded_groups = seeded_groups
        .into_iter()
        .filter_map(|(group, categories)| {
            let mut unique = categories.into_iter().unique();
            let category = unique.next()?;
            unique.next().is_none().then_some((group, category))
        })
        .collect::<FxHashMap<_, _>>();
    let melee_start = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Melee")
                && item
                    .definition_type_code
                    .is_some_and(is_weapon_or_melee_skin_type)
        })
        .filter_map(GearItem::definition_group_index)
        .filter(|group| *group > 3)
        .min();
    let melee_subcategories = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Melee"))
        .filter_map(|item| item.subcategory.as_deref())
        .collect::<FxHashSet<_>>();
    let melee_subcategory = (melee_subcategories.len() == 1)
        .then(|| (*melee_subcategories.iter().next().expect("one subtype")).to_owned());

    // A localized/grouped taxonomy seed can still call an opaque updated knife
    // row a Weapon Skin. A path-backed sibling is authoritative for the whole
    // current cosmetic group.
    for item in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Weapon Skin")
            && item
                .cosmetic_group_key()
                .and_then(|key| seeded_groups.get(&key))
                == Some(&"Melee")
    }) {
        item.item_type = Some("Melee".to_owned());
        item.subcategory.clone_from(&melee_subcategory);
    }

    for item in items.iter_mut().filter(|item| {
        item.item_type.is_none()
            && item
                .definition_type_code
                .is_some_and(is_weapon_or_melee_skin_type)
    }) {
        let Some(group) = item.definition_group_index() else {
            continue;
        };
        // The cosmetic group sequence is not a strict weapon/melee boundary:
        // later content can append a weapon skin after the first melee group.
        // Prefer exact path/hash/group ownership whenever it is available.
        if direct_skin_weapon_hash(item, &generated_weapon_owners).is_some()
            || item
                .cosmetic_group_key()
                .and_then(|key| seeded_groups.get(&key))
                == Some(&"Weapon Skin")
            || (item.definition_type_code != Some(0x138) && skin_weapon_hash(item).is_some())
        {
            item.item_type = Some("Weapon Skin".to_owned());
        } else if direct_melee_skin(item, &generated_melee_hashes)
            || item
                .model_tag
                .is_some_and(|model| melee_models.contains(&model))
            || item
                .cosmetic_group_key()
                .and_then(|key| seeded_groups.get(&key))
                == Some(&"Melee")
        {
            item.item_type = Some("Melee".to_owned());
            item.subcategory.clone_from(&melee_subcategory);
        } else if melee_start.is_some_and(|start| group >= start) {
            item.item_type = Some("Melee".to_owned());
            item.subcategory.clone_from(&melee_subcategory);
        } else {
            item.item_type = Some("Weapon Skin".to_owned());
        }
    }
}

/// Current cosmetic records share a stale structural UI label across runner,
/// charm, weapon-skin, and melee-skin layouts. Prefer current definition class
/// plus runner shell header.
fn correct_current_cosmetic_taxonomy(items: &mut [GearItem]) {
    for item in items.iter_mut() {
        if let Some(shell) = item.description.as_deref().and_then(runner_shell_model) {
            item.item_type = Some("Runner Skin".to_owned());
            item.subcategory = Some(canonical_runner_shell_model(shell).to_owned());
            continue;
        }
        if item.definition_type_code == Some(0x137)
            && item
                .internal_categories
                .iter()
                .any(|category| category == "item_type.#C114DEA3")
        {
            item.item_type = Some("Charm".to_owned());
            item.subcategory = None;
        }
    }
}

fn weapon_mod_category_from_path(internal_name: &str) -> Option<&'static str> {
    let category = internal_name
        .strip_prefix("weapon_mods.")?
        .split('.')
        .next()?;
    match category {
        "barrels" => Some("Barrel"),
        "chips" => Some("Chip"),
        "foregrips" => Some("Foregrip"),
        "generators" => Some("Generator"),
        "magazines" => Some("Magazine"),
        "muzzles" => Some("Muzzle"),
        "optics" => Some("Optic"),
        "shields" => Some("Shield"),
        "stocks" => Some("Stock"),
        "unique" => Some("Unique"),
        _ => None,
    }
}

fn weapon_mod_category_from_definition(definition: &[u8]) -> Option<&'static str> {
    let values = definition
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect::<FxHashSet<_>>();
    [
        (MOD_CATEGORY_BARREL_MARKER, "Barrel"),
        (MOD_CATEGORY_CHIP_MARKER, "Chip"),
        (MOD_CATEGORY_FOREGRIP_MARKER, "Foregrip"),
        (MOD_CATEGORY_GENERATOR_MARKER, "Generator"),
        (MOD_CATEGORY_MAGAZINE_MARKER, "Magazine"),
        (MOD_CATEGORY_MUZZLE_MARKER, "Muzzle"),
        (MOD_CATEGORY_OPTIC_MARKER, "Optic"),
        (MOD_CATEGORY_SHIELD_MARKER, "Shield"),
        (MOD_CATEGORY_STOCK_MARKER, "Stock"),
        (MOD_CATEGORY_UNIQUE_MARKER, "Unique"),
    ]
    .into_iter()
    .find_map(|(marker, category)| values.contains(&marker).then_some(category))
}

fn weapon_mod_category_from_internal_categories(item: &GearItem) -> Option<&'static str> {
    if item
        .internal_categories
        .iter()
        .any(|category| category == "item_type.#51E74E6F")
    {
        // Authored singleton class for Law of Embers: intrinsic behavior,
        // not an attachment slot or weapon-exclusive component.
        return Some("Universal");
    }
    if item
        .internal_categories
        .iter()
        .any(|category| category == "item_type.#F97B117F")
    {
        // Current universal behaviour-chip rows no longer expose their old
        // `weapon_mods.chips.*` wordlist path, but retain this authored type.
        return Some("Chip");
    }
    item.internal_categories
        .iter()
        .any(|category| category == "item_type.weapon_mod.unique")
        .then_some("Unique")
}

fn weapon_mod_path_family(internal_name: &str) -> Option<(&str, &str)> {
    let mut parts = internal_name.strip_prefix("weapon_mods.")?.split('.');
    Some((parts.next()?, parts.next()?))
}

fn weapon_mod_authored_family<'a>(item: &'a GearItem, category: &str) -> Option<&'a str> {
    if let Some(internal_name) = item
        .internal_name
        .as_deref()
        .filter(|name| !name.starts_with('#'))
        && weapon_mod_category_from_path(internal_name) == Some(category)
    {
        return weapon_mod_path_family(internal_name).map(|(_, family)| family);
    }

    let prefix = match category {
        "Barrel" => "item_type.weapon_mod.barrel.",
        "Chip" => "item_type.weapon_mod.chip.",
        "Foregrip" => "item_type.weapon_mod.foregrip.",
        "Magazine" => "item_type.weapon_mod.magazine.",
        "Optic" => "item_type.weapon_mod.sight.",
        _ => return None,
    };
    item.internal_categories
        .iter()
        .find_map(|authored| authored.strip_prefix(prefix))
}

fn specific_weapon_mod_category(category: &str, family: &str) -> Option<&'static str> {
    match (category, family) {
        ("Barrel", "precision") => Some("Precision Barrel"),
        ("Magazine", "precision") => Some("Precision Magazine"),
        ("Magazine", "rifle") => Some("Assault Magazine"),
        ("Magazine", "pistol") => Some("Pistol Magazine"),
        ("Magazine", "lmg") => Some("Belt-Fed Magazine"),
        ("Magazine", "battery") => Some("Volt Array"),
        ("Magazine", "heavy_battery") => Some("Volt Cell"),
        ("Muzzle", "base") => Some("CQC Barrel"),
        ("Muzzle", "battery_front") => Some("Ion Dampener"),
        ("Muzzle", "dampener") => Some("Volt Dampener"),
        ("Muzzle", "underbarrel") => Some("Underbarrel"),
        ("Optic", "marksman") => Some("Precision Optic"),
        ("Optic", "rifle") => Some("Assault Optic"),
        ("Optic", "pistol") => Some("Pistol Optic"),
        ("Optic", "lmg") => Some("LMG Optic"),
        ("Optic", "sniper") => Some("Sniper Optic"),
        ("Foregrip", "rifle") => Some("Rifle Grip"),
        ("Foregrip", "shotgun") => Some("Shotgun Grip"),
        _ => None,
    }
}

fn specific_weapon_mod_category_from_internal_categories(item: &GearItem) -> Option<&'static str> {
    item.internal_categories
        .iter()
        .find_map(|category| match category.as_str() {
            "item_type.#21B2978C" => Some("CQC Barrel"),
            "item_type.#0AEFF616" => Some("Ion Dampener"),
            "item_type.#03B1FCA1" => Some("Volt Dampener"),
            "item_type.#6C016283" => Some("Underbarrel"),
            "item_type.#7C7BCB06" => Some("Shotgun Grip"),
            "item_type.#9F7C1E53" => Some("Volt Cell"),
            "item_type.#AC5146B3" => Some("Shield"),
            _ => None,
        })
}

fn weapon_mod_category_slug(item: &GearItem, category: &str) -> String {
    if category == "Chip" {
        return "chip".to_owned();
    }

    if let Some(family) = weapon_mod_authored_family(item, category)
        && family != "v100"
    {
        return family.to_owned();
    }

    if let Some(slug) =
        item.internal_categories
            .iter()
            .find_map(|authored| match authored.as_str() {
                "item_type.#21B2978C" => Some("base"),
                "item_type.#0AEFF616" => Some("battery_front"),
                "item_type.#03B1FCA1" => Some("dampener"),
                "item_type.#6C016283" => Some("underbarrel"),
                "item_type.#7C7BCB06" => Some("shotgun"),
                "item_type.#9F7C1E53" => Some("heavy_battery"),
                "item_type.#AC5146B3" => Some("shield"),
                _ => None,
            })
    {
        return slug.to_owned();
    }

    match category {
        "Universal" => "universal",
        "Foregrip" => "foregrip",
        "Generator" => "generator",
        "Magazine" => "magazine",
        "Muzzle" => "muzzle",
        "Optic" => "optic",
        "Shield" => "shield",
        "Stock" => "stock",
        "Unique" => "unique",
        _ => "barrel",
    }
    .to_owned()
}

#[derive(Clone)]
struct CompatibleWeapon {
    name: String,
    internal_name: Option<String>,
    internal_hash: Option<u32>,
    subcategory: Option<String>,
}

impl CompatibleWeapon {
    fn is_archetype(&self, archetype: &str) -> bool {
        self.internal_name
            .as_deref()
            .and_then(|path| path.rsplit('.').next())
            == Some(archetype)
    }

    fn is_battery(&self) -> bool {
        self.internal_name
            .as_deref()
            .is_some_and(|name| name.contains("_battery_"))
    }

    fn is_fusion(&self) -> bool {
        self.internal_name
            .as_deref()
            .is_some_and(|name| name.contains("_fusion_"))
    }

    fn is_ballistic(&self) -> bool {
        !self.is_battery() && !self.is_fusion()
    }

    fn is_subcategory(&self, subcategory: &str) -> bool {
        self.subcategory.as_deref() == Some(subcategory)
    }

    /// These authored archetypes use a fixed, internal, or per-round feed and
    /// do not expose the otherwise shared magazine family as a mod slot.
    fn supports_magazine_mods(&self) -> bool {
        ![
            "smg_light_01",
            "dmr_heavy_02",
            "shotgun_mips_01",
            "sniper_mips_02",
            "pistol_heavy_01",
        ]
        .into_iter()
        .any(|archetype| self.is_archetype(archetype))
    }
}

fn base_weapon_matches_mod_family(
    weapon: &CompatibleWeapon,
    category: &str,
    compatibility: &str,
) -> bool {
    // The path family is the authoritative broad compatibility group. Energy
    // magazine/muzzle families subdivide the normal weapon classes by their
    // authored `_battery_` and `_fusion_` archetypes.
    match (category, compatibility) {
        ("Barrel", "precision") => {
            (weapon.is_subcategory("Marksman Rifle") || weapon.is_subcategory("Sniper Rifle"))
                && weapon.is_ballistic()
        }
        ("Foregrip", "rifle") => weapon.is_subcategory("Assault Rifle") && weapon.is_ballistic(),
        ("Foregrip", "shotgun") => weapon.is_subcategory("Shotgun"),
        // The July investment update exposes the generator family on both
        // railgun feeds. Ammo type is not the slot discriminator.
        ("Generator", _) => weapon.is_subcategory("Railgun"),
        ("Magazine", "battery") => weapon.is_battery() && weapon.supports_magazine_mods(),
        ("Magazine", "heavy_battery") => weapon.is_fusion() && weapon.supports_magazine_mods(),
        ("Magazine", "lmg") => {
            weapon.is_subcategory("Machine Gun") && weapon.supports_magazine_mods()
        }
        ("Magazine", "payload") => {
            weapon.is_subcategory("Railgun")
                && weapon.is_ballistic()
                && weapon.supports_magazine_mods()
        }
        ("Magazine", "pistol") => {
            weapon.is_subcategory("Pistol")
                && weapon.is_ballistic()
                && weapon.supports_magazine_mods()
        }
        ("Magazine", "precision") => {
            (weapon.is_subcategory("Marksman Rifle") || weapon.is_subcategory("Sniper Rifle"))
                && weapon.is_ballistic()
                && weapon.supports_magazine_mods()
        }
        ("Magazine", "rifle") => {
            (weapon.is_subcategory("Assault Rifle") || weapon.is_subcategory("Submachine Gun"))
                && weapon.is_ballistic()
                && weapon.supports_magazine_mods()
        }
        ("Magazine", "shotgun") => {
            weapon.is_subcategory("Shotgun")
                && weapon.is_ballistic()
                && weapon.supports_magazine_mods()
        }
        ("Muzzle", "base") => {
            (weapon.is_subcategory("Pistol") || weapon.is_subcategory("Submachine Gun"))
                && weapon.is_ballistic()
                && weapon.internal_hash != Some(KKV_9SD_HASH)
        }
        // The authored Prestige leaves disambiguate these otherwise similar
        // battery front slots: `gold_battery_dmr` targets V66's DMR archetype,
        // while `gold_btsmg` targets V22 within the broader Dampener family.
        // V11 and V75 share that same Dampener family; display subtypes alone
        // are not sufficient to make this distinction.
        ("Muzzle", "battery_front") => {
            weapon.is_archetype("dmr_battery_01") || weapon.is_archetype("sniper_fusion_01")
        }
        ("Muzzle", "dampener") => ["pistol_battery_01", "smg_battery_01", "auto_battery_01"]
            .into_iter()
            .any(|archetype| weapon.is_archetype(archetype)),
        ("Muzzle", "underbarrel") => weapon.is_subcategory("Shotgun") && weapon.is_ballistic(),
        ("Optic", "lmg") => weapon.is_subcategory("Machine Gun"),
        ("Optic", "marksman") => weapon.is_subcategory("Marksman Rifle"),
        // The D54 ships with an integrated sight and uses its fourth authored
        // attachment slot for Stock mods, so it does not accept pistol optics.
        ("Optic", "pistol") => {
            weapon.is_subcategory("Pistol") && weapon.internal_hash != Some(D54_BATTLE_PISTOL_HASH)
        }
        ("Optic", "rifle") => {
            (weapon.is_subcategory("Assault Rifle") || weapon.is_subcategory("Submachine Gun"))
                // M77 has an authored flip sight and no optic attachment slot.
                && !weapon.is_archetype("auto_light_01")
                // V22's smart-lock sight is integrated and cannot be replaced.
                && !weapon.is_archetype("smg_battery_01")
        }
        ("Optic", "sniper") => weapon.is_subcategory("Sniper Rifle"),
        ("Shield", "lmg") => weapon.is_subcategory("Machine Gun"),
        _ => false,
    }
}

fn specific_weapon_mod_family_from_weapon(
    category: &str,
    weapon: &CompatibleWeapon,
) -> Option<&'static str> {
    const FAMILIES: &[(&str, &str)] = &[
        ("Barrel", "precision"),
        ("Foregrip", "rifle"),
        ("Foregrip", "shotgun"),
        ("Magazine", "battery"),
        ("Magazine", "heavy_battery"),
        ("Magazine", "lmg"),
        ("Magazine", "payload"),
        ("Magazine", "pistol"),
        ("Magazine", "precision"),
        ("Magazine", "rifle"),
        ("Magazine", "shotgun"),
        ("Muzzle", "base"),
        ("Muzzle", "battery_front"),
        ("Muzzle", "dampener"),
        ("Muzzle", "underbarrel"),
        ("Optic", "lmg"),
        ("Optic", "marksman"),
        ("Optic", "pistol"),
        ("Optic", "rifle"),
        ("Optic", "sniper"),
    ];

    FAMILIES
        .iter()
        .filter(|(family_category, family)| {
            *family_category == category
                && base_weapon_matches_mod_family(weapon, family_category, family)
        })
        .map(|(_, family)| *family)
        .exactly_one()
        .ok()
}

/// Resolve the concrete weapon archetype encoded in a custom mod's authored
/// path. This deliberately compares archetype identifiers rather than full
/// paths or localized presentation text. The latter changes by language and
/// caused unrelated weapons with similar descriptions to be selected.
fn exact_weapon_archetype_from_mod_path(internal_name: &str) -> Option<&'static str> {
    let leaf = internal_name.rsplit('.').next()?;

    // Most custom-mod leaves describe the weapon mechanically. A few use the
    // weapon's internal project codename (btar, btpstl, etc.); those aliases
    // are normalized here, while matching remains archetype-based below.
    match leaf {
        "gold_auto_light_02" => Some("auto_light_02"),
        value if value.contains("heavy_ar") => Some("auto_heavy_01"),
        value if value.contains("light_ar") => Some("auto_light_01"),
        value if value.contains("heavy_smg") => Some("smg_heavy_01"),
        value if value.contains("battery_dmr") => Some("dmr_battery_01"),
        value if value.contains("battery_sniper") => Some("sniper_fusion_01"),
        value if value.contains("battery_shotgun") => Some("shotgun_fusion_01"),
        value if value.contains("battery_railgun") => Some("railgun_fusion_01"),
        value if value.contains("ballistic_railgun") => Some("railgun_mips_01"),
        value if value.contains("bolt_action_sniper") => Some("sniper_mips_02"),
        value if value.contains("ballistic_sniper") => Some("sniper_mips_01"),
        value if value.contains("ballistic_burst") => Some("burst_light_01"),
        value if value.contains("burst_smg") => Some("smg_light_01"),
        value if value.contains("lever_action") => Some("dmr_heavy_02"),
        value if value.contains("double_tap") => Some("burst_heavy_01"),
        value if value.contains("copperhead") => Some("smg_light_02"),
        value if value.contains("vital_intel") => Some("dmr_light_01"),
        value if value.contains("outpost") => Some("dmr_heavy_01"),
        value if value.contains("peashooter") => Some("pistol_light_01"),
        value if value.contains("brksg") => Some("shotgun_mips_01"),
        value if value.contains("btpstl") => Some("pistol_battery_01"),
        value if value.contains("btsmg") => Some("smg_battery_01"),
        value if value.contains("btar") => Some("auto_battery_01"),
        value if value.contains("magnum") => Some("pistol_heavy_01"),
        value if value.contains("lmg2") => Some("mg_light_02"),
        value if value.contains("hmg") => Some("mg_heavy_01"),
        "gold_ballistic0" => Some("mg_light_01"),
        "machineguns_light_unique" => Some("mg_light_01_unique"),
        "smg_light_unique" => Some("smg_light_01_unique"),
        "shotgun_battery_unique" => Some("shotgun_fusion_01_unique"),
        "railgun_battery_unique" => Some("railgun_fusion_01_unique"),
        "sniper_fusion_unique" => Some("sniper_fusion_01_unique"),
        _ => None,
    }
}

fn weapon_matches_exact_mod_archetype(weapon: &CompatibleWeapon, mod_path: &str) -> bool {
    let Some(expected) = exact_weapon_archetype_from_mod_path(mod_path) else {
        return false;
    };
    weapon.is_archetype(expected)
}

/// Newer package records occasionally omit their wordlist path entirely. Keep
/// the small set of verified opaque investment IDs isolated from the general
/// resolver; this is never consulted for path-backed mods.
fn opaque_mod_target_hash(internal_hash: u32) -> Option<u32> {
    match internal_hash {
        0x03d0_7204 | 0xfd28_13bd => Some(FIRESTORM_HASH), // Eyes of Ash / Heart of Fire
        0xf81e_1c03 => Some(KKV_9SD_HASH),                 // Flechette Drum
        0x774d_a5d3 => Some(weapon_internal_hash(
            "weapons.shotguns.v100.shotgun_mips_02",
        )), // Full-Auto Selector
        0xafc2_33f4 => Some(0x033e_c2d0),                  // Gestalt Complex
        0xeb64_31d8 => Some(0xc851_6e72),                  // Pike Formation
        0xf1eb_808c => Some(0x0b15_4800),                  // Serpentine!
        0x632e_130a => Some(0x3961_29b9),                  // Short-Throw Projector
        0x94b4_9ee0 => Some(0xba19_1c63),                  // Tilt-Shift LS
        _ => None,
    }
}

fn sorted_unique_names(mut names: Vec<String>) -> Vec<String> {
    names.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()).then(a.cmp(b)));
    names.dedup();
    names
}

fn assign_weapon_mod_metadata(items: &mut [GearItem]) {
    let to_compatible_weapon = |item: &GearItem| CompatibleWeapon {
        name: item.name.clone(),
        internal_name: item.internal_name.clone(),
        internal_hash: item.internal_hash,
        subcategory: item
            .subcategory
            .clone()
            .or_else(|| weapon_subcategory(item))
            .or_else(|| weapon_subcategory_from_name(&item.name)),
    };
    // Gameplay mod compatibility uses canonical weapon ownership. Named
    // variants often reuse the base weapon's archetype path/render sockets,
    // but do not inherit its ordinary or Prestige mods.
    let base_weapons = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter(|item| is_canonical_weapon_skin_owner(item))
        .map(to_compatible_weapon)
        .collect::<Vec<_>>();
    let unique_weapons = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && item.rarity == Some(GearRarity::Unique)
                && !is_canonical_weapon_skin_owner(item)
        })
        .map(to_compatible_weapon)
        .collect::<Vec<_>>();
    let quest_weapons = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && item.rarity == Some(GearRarity::Quest)
                && item.internal_hash == Some(FIRESTORM_HASH)
        })
        .map(to_compatible_weapon)
        .collect::<Vec<_>>();
    let unique_weapon_names = sorted_unique_names(
        unique_weapons
            .iter()
            .map(|weapon| weapon.name.clone())
            .collect(),
    );
    let special_weapons = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon") && item.internal_hash == Some(KKV_9SD_HASH)
        })
        .map(to_compatible_weapon)
        .collect::<Vec<_>>();
    let weapons = base_weapons
        .iter()
        .chain(&unique_weapons)
        .chain(&special_weapons)
        .chain(&quest_weapons)
        .cloned()
        .collect::<Vec<_>>();

    let mut named_families = FxHashMap::<String, FxHashSet<(String, String)>>::default();
    for item in items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
    {
        let Some(internal_name) = item
            .internal_name
            .as_deref()
            .filter(|name| !name.starts_with('#'))
        else {
            continue;
        };
        let Some((_, compatibility)) = weapon_mod_path_family(internal_name) else {
            continue;
        };
        let Some(category) = weapon_mod_category_from_path(internal_name) else {
            continue;
        };
        named_families
            .entry(item.name.clone())
            .or_default()
            .insert((category.to_owned(), compatibility.to_owned()));
    }

    for item in items
        .iter_mut()
        .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
    {
        let category = item
            .internal_name
            .as_deref()
            .and_then(weapon_mod_category_from_path)
            .or_else(|| weapon_mod_category_from_internal_categories(item))
            .or_else(|| {
                item.definition_tag
                    .and_then(|tag| package_manager().read_tag(tag).ok())
                    .and_then(|definition| weapon_mod_category_from_definition(&definition))
            });
        let Some(category) = category else {
            continue;
        };
        let display_category = weapon_mod_authored_family(item, category)
            .and_then(|family| specific_weapon_mod_category(category, family))
            .or_else(|| specific_weapon_mod_category_from_internal_categories(item))
            .unwrap_or(category);
        item.mod_family = Some(weapon_mod_category_slug(item, category));
        item.mod_category = Some(category.to_owned());
        item.subcategory = Some(display_category.to_owned());

        if matches!(category, "Chip" | "Universal") {
            item.mod_is_universal = true;
            continue;
        }

        // Custom mods encode their concrete target as a weapon archetype in
        // the authored path (for example heavy_ar -> auto_heavy_01).
        if let Some(path) = item
            .internal_name
            .as_deref()
            .filter(|path| exact_weapon_archetype_from_mod_path(path).is_some())
        {
            let variant_slot = exact_weapon_archetype_from_mod_path(path)
                .is_some_and(|archetype| archetype.ends_with("_unique"));
            let candidates = if variant_slot {
                &unique_weapons
            } else {
                &base_weapons
            };
            item.compatible_weapons = candidates
                .iter()
                .filter(|weapon| weapon_matches_exact_mod_archetype(weapon, path))
                .map(|weapon| weapon.name.clone())
                .collect();
            if !item.compatible_weapons.is_empty() {
                continue;
            }
        }

        if item
            .internal_name
            .as_deref()
            .is_some_and(|name| name.starts_with('#'))
        {
            if let Some(target_hash) = item.internal_hash.and_then(opaque_mod_target_hash) {
                let targets = weapons
                    .iter()
                    .filter(|weapon| weapon.internal_hash == Some(target_hash))
                    .collect::<Vec<_>>();
                item.compatible_weapons =
                    targets.iter().map(|weapon| weapon.name.clone()).collect();
                if let Ok(target) = targets.into_iter().exactly_one()
                    && let Some(family) = specific_weapon_mod_family_from_weapon(category, target)
                {
                    item.mod_family = Some(family.to_owned());
                    if let Some(display_category) = specific_weapon_mod_category(category, family) {
                        item.subcategory = Some(display_category.to_owned());
                    }
                }
                if !item.compatible_weapons.is_empty() {
                    continue;
                }
            }
        }

        if category == "Stock" {
            // Stock definitions are hash-only and shared by both pistol rigs.
            item.compatible_weapons = sorted_unique_names(
                weapons
                    .iter()
                    .filter(|weapon| {
                        matches!(
                            weapon.internal_hash,
                            Some(D54_BATTLE_PISTOL_HASH | KKV_9SD_HASH)
                        )
                    })
                    .map(|weapon| weapon.name.clone())
                    .collect(),
            );
            continue;
        }

        if category == "Unique" {
            item.compatible_weapons.clone_from(&unique_weapon_names);
            continue;
        }

        let direct_family = item
            .internal_name
            .as_deref()
            .filter(|name| !name.starts_with('#'))
            .and_then(weapon_mod_path_family)
            .map(|(_, compatibility)| (category, compatibility));
        let propagated_family = named_families.get(&item.name).and_then(|families| {
            (families.len() == 1).then(|| {
                let (category, compatibility) =
                    families.iter().next().expect("one propagated family");
                (category.as_str(), compatibility.as_str())
            })
        });
        // Prestige optics such as Darksight are hash-only definitions, but
        // their display records still author an exact optic subtype. Prefer
        // that metadata over inferring compatibility from localized names.
        let authored_family = weapon_mod_authored_family(item, category)
            .map(|compatibility| (category, compatibility));
        if let Some((family_category, compatibility)) =
            direct_family.or(propagated_family).or(authored_family)
        {
            item.compatible_weapons = sorted_unique_names(
                base_weapons
                    .iter()
                    .filter(|weapon| {
                        base_weapon_matches_mod_family(weapon, family_category, compatibility)
                    })
                    .map(|weapon| weapon.name.clone())
                    .collect(),
            );
        }
    }
}

fn weapon_internal_hash(path: &str) -> u32 {
    quicktag_core::util::fnv1(path.as_bytes())
}

// Canonical base-weapon investment paths. Cosmetic group ordinals are mutable
// content indices and changed in the July update; these path hashes are the
// stable identities authored into path-backed skin API hashes.
const WEAPON_SKIN_OWNER_PATHS: &[&str] = &[
    "weapons.auto_rifles.v100.auto_light_01",
    "weapons.auto_rifles.v100.auto_light_02",
    "weapons.auto_rifles.v100.auto_battery_01",
    "weapons.auto_rifles.v100.auto_heavy_01",
    "weapons.machineguns.v100.mg_heavy_01",
    "weapons.machineguns.v100.mg_light_01",
    "weapons.machineguns.v100.mg_light_02",
    "weapons.marksman_rifles.v100.burst_light_01",
    "weapons.marksman_rifles.v100.burst_heavy_01",
    "weapons.marksman_rifles.v100.dmr_battery_01",
    "weapons.marksman_rifles.v100.dmr_heavy_01",
    "weapons.marksman_rifles.v100.dmr_heavy_02",
    "weapons.marksman_rifles.v100.dmr_light_01",
    "weapons.pistols.v100.pistol_battery_01",
    "weapons.pistols.v100.pistol_heavy_01",
    "weapons.pistols.v100.pistol_light_01",
    "weapons.railguns.v100.railgun_mips_01",
    "weapons.railguns.v100.railgun_fusion_01",
    "weapons.shotguns.v100.shotgun_fusion_01",
    "weapons.shotguns.v100.shotgun_mips_01",
    "weapons.shotguns.v100.shotgun_mips_02",
    "weapons.sniper_rifles.v100.sniper_mips_01",
    "weapons.sniper_rifles.v100.sniper_mips_02",
    "weapons.sniper_rifles.v100.sniper_fusion_01",
    "weapons.submachineguns.v100.smg_light_01",
    "weapons.submachineguns.v100.smg_light_02",
    "weapons.submachineguns.v100.smg_heavy_01",
    "weapons.submachineguns.v100.smg_battery_01",
];

const MELEE_SKIN_OWNER_PATHS: &[&str] = &["melees.knives.v100.melee_knife_01"];

fn generated_skin_path_owners_for(paths: &[&str]) -> FxHashMap<u32, u32> {
    let mut owners = FxHashMap::default();
    for owner in paths {
        let owner_hash = weapon_internal_hash(owner);
        for collection in ["default", "store"] {
            for number in 0..=256 {
                for suffix in [number.to_string(), format!("{number:02}")] {
                    let skin = format!("{owner}.skins.{collection}.v100.skin{suffix}");
                    owners.insert(quicktag_core::util::fnv1(skin.as_bytes()), owner_hash);
                }
            }
        }
    }
    owners
}

fn generated_skin_path_owners() -> FxHashMap<u32, u32> {
    generated_skin_path_owners_for(WEAPON_SKIN_OWNER_PATHS)
}

fn generated_skin_hashes(paths: &[&str]) -> FxHashSet<u32> {
    generated_skin_path_owners_for(paths).into_keys().collect()
}

fn direct_melee_skin(item: &GearItem, generated_hashes: &FxHashSet<u32>) -> bool {
    item.internal_name
        .as_deref()
        .is_some_and(|name| name.starts_with("melees.") && name.contains(".skins."))
        || item
            .internal_hash
            .is_some_and(|hash| generated_hashes.contains(&hash))
        || matches!(
            item.internal_hash,
            Some(ACHROMATIC_RUSH_MELEE_SKIN_HASH) | Some(VOX_NOCTURNA_MELEE_SKIN_HASH)
        )
}

fn direct_skin_weapon_hash(item: &GearItem, generated_owners: &FxHashMap<u32, u32>) -> Option<u32> {
    if let Some(owner) = item
        .internal_name
        .as_deref()
        .and_then(|name| name.split_once(".skins."))
        .map(|(owner, _)| owner)
        .filter(|owner| owner.starts_with("weapons."))
    {
        return Some(weapon_internal_hash(owner));
    }

    match item.internal_hash? {
        D54_DEFAULT_SKIN_HASH => Some(D54_BATTLE_PISTOL_HASH),
        BIOTOXIC_DEFAULT_SKIN_HASH | BIOTOXIC_SHADOW_INDEX_SKIN_HASH => {
            Some(BIOTOXIC_DISINJECTOR_HASH)
        }
        ACID_ABYSS_SKIN_HASH => Some(weapon_internal_hash(
            "weapons.marksman_rifles.v100.dmr_light_01",
        )),
        CRYO_SHIFT_V11_SKIN_HASH => Some(weapon_internal_hash(
            "weapons.pistols.v100.pistol_battery_01",
        )),
        internal_hash => generated_owners.get(&internal_hash).copied(),
    }
}

/// The high half of the cosmetic group key is the concrete weapon archetype
/// index used by store skins. Map that authored index back to the base weapon
/// investment identifier; names are then taken from the weapon's own scoped
/// display record rather than being duplicated here.
fn skin_group_weapon_hash(group: u16) -> Option<u32> {
    let path = match group {
        18 => "weapons.auto_rifles.v100.auto_light_01",
        19 => "weapons.auto_rifles.v100.auto_light_02",
        20 => "weapons.auto_rifles.v100.auto_battery_01",
        21 => "weapons.auto_rifles.v100.auto_heavy_01",
        23 => "weapons.machineguns.v100.mg_heavy_01",
        24 => "weapons.machineguns.v100.mg_light_01",
        25 => "weapons.machineguns.v100.mg_light_02",
        27 => "weapons.marksman_rifles.v100.burst_light_01",
        28 => "weapons.marksman_rifles.v100.burst_heavy_01",
        29 => "weapons.marksman_rifles.v100.dmr_battery_01",
        30 => "weapons.marksman_rifles.v100.dmr_heavy_01",
        31 => "weapons.marksman_rifles.v100.dmr_heavy_02",
        32 => "weapons.marksman_rifles.v100.dmr_light_01",
        33 => "weapons.pistols.v100.pistol_battery_01",
        34 => "weapons.pistols.v100.pistol_heavy_01",
        35 => "weapons.pistols.v100.pistol_light_01",
        37 => "weapons.railguns.v100.railgun_mips_01",
        38 => "weapons.railguns.v100.railgun_fusion_01",
        // Warmonger is the named mips variant; its cosmetics still belong to
        // the base ARES archetype in Gear/Models integration.
        39 => "weapons.railguns.v100.railgun_mips_01",
        41 => "weapons.shotguns.v100.shotgun_fusion_01",
        42 => "weapons.shotguns.v100.shotgun_mips_01",
        43 => "weapons.shotguns.v100.shotgun_mips_02",
        44 => "weapons.sniper_rifles.v100.sniper_mips_01",
        45 => "weapons.sniper_rifles.v100.sniper_mips_02",
        46 => "weapons.sniper_rifles.v100.sniper_fusion_01",
        47 => "weapons.submachineguns.v100.smg_light_01",
        48 | 49 => "weapons.submachineguns.v100.smg_light_02",
        50 | 51 => "weapons.submachineguns.v100.smg_heavy_01",
        36 => return Some(D54_BATTLE_PISTOL_HASH),
        // The extra groups select named-variant art (KKV, Sidewinder, and
        // Cascade), but the game exposes those variants under their canonical
        // base weapon families.
        52 | 53 => "weapons.submachineguns.v100.smg_battery_01",
        _ => return None,
    };
    Some(weapon_internal_hash(path))
}

/// Whether this inventory Weapon record is the archetype referenced by the
/// game's cosmetic-group table. Named weapon variants can reuse the same
/// render coordinate frame, but they do not own that archetype's skins.
fn is_canonical_weapon_skin_owner(item: &GearItem) -> bool {
    item.internal_hash.is_some_and(|internal_hash| {
        matches!(
            internal_hash,
            D54_BATTLE_PISTOL_HASH | BIOTOXIC_DISINJECTOR_HASH
        ) || WEAPON_SKIN_OWNER_PATHS
            .iter()
            .any(|path| weapon_internal_hash(path) == internal_hash)
    })
}

fn skin_weapon_hash(item: &GearItem) -> Option<u32> {
    if let Some(owner) = item
        .internal_name
        .as_deref()
        .and_then(|name| name.split_once(".skins."))
        .map(|(owner, _)| owner)
        .filter(|owner| owner.starts_with("weapons."))
    {
        return Some(weapon_internal_hash(owner));
    }

    match item.internal_hash? {
        D54_DEFAULT_SKIN_HASH => Some(D54_BATTLE_PISTOL_HASH),
        BIOTOXIC_DEFAULT_SKIN_HASH => Some(BIOTOXIC_DISINJECTOR_HASH),
        BIOTOXIC_SHADOW_INDEX_SKIN_HASH => Some(BIOTOXIC_DISINJECTOR_HASH),
        ACID_ABYSS_SKIN_HASH => skin_group_weapon_hash(32),
        CRYO_SHIFT_V11_SKIN_HASH => skin_group_weapon_hash(33),
        _ => item
            .definition_group_index()
            .and_then(skin_group_weapon_hash),
    }
}

fn assign_skin_weapons(items: &mut [GearItem]) {
    let weapon_names = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.internal_hash?, item.name.clone())))
        .collect::<FxHashMap<_, _>>();
    let weapon_names_by_model = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.model_tag?, item.name.clone())))
        .into_group_map();
    let generated_owners = generated_skin_path_owners();

    for item in items
        .iter_mut()
        .filter(|item| item.item_type.as_deref() == Some("Weapon Skin"))
    {
        item.applies_to = if item
            .description
            .as_deref()
            .is_some_and(|description| description.contains("cert validation"))
        {
            Some("Certification test (no weapon)".to_owned())
        } else {
            direct_skin_weapon_hash(item, &generated_owners)
                .or_else(|| {
                    (item.definition_type_code != Some(0x138))
                        .then(|| skin_weapon_hash(item))
                        .flatten()
                })
                .and_then(|hash| weapon_names.get(&hash).cloned())
                .or_else(|| {
                    if item.rarity != Some(GearRarity::Standard) {
                        return None;
                    }
                    let names = weapon_names_by_model.get(&item.model_tag?)?;
                    let mut unique_names = names.iter().unique();
                    let owner = unique_names.next()?;
                    unique_names.next().is_none().then(|| owner.clone())
                })
        };
    }
}

/// Resolve opaque updated cosmetics through exact Pattern-component reuse.
/// A cosmetic may have a new wrapper and no usable investment path/group, but
/// it still instantiates the same weapon-authored component as a proven skin.
/// Shared/global components naturally map to several owners and are ignored.
fn reconcile_skin_shared_pattern_components(
    items: &mut [GearItem],
    cache: &quicktag_scanner::TagCache,
) -> usize {
    let canonical_names = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon") && item.rarity != Some(GearRarity::Unique)
        })
        .map(|item| item.name.clone())
        .collect::<FxHashSet<_>>();
    let subcategories = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.name.clone(), item.subcategory.clone()?)))
        .collect::<FxHashMap<_, _>>();
    let mut component_owners = FxHashMap::<TagHash, FxHashSet<String>>::default();
    for skin in items.iter().filter(|item| {
        item.item_type.as_deref() == Some("Weapon Skin")
            && item
                .applies_to
                .as_deref()
                .is_some_and(|owner| canonical_names.contains(owner))
    }) {
        let Some((model, owner)) = skin.model_tag.zip(skin.applies_to.as_ref()) else {
            continue;
        };
        for component in model_pattern_components(cache, model) {
            component_owners
                .entry(component)
                .or_default()
                .insert(owner.clone());
        }
    }

    let mut resolved = 0;
    for skin in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Weapon Skin") && item.applies_to.is_none()
    }) {
        let Some(model) = skin.model_tag else {
            continue;
        };
        let owners = model_pattern_components(cache, model)
            .into_iter()
            .filter_map(|component| component_owners.get(&component))
            .filter(|owners| owners.len() == 1)
            .flatten()
            .unique()
            .collect::<Vec<_>>();
        let [owner] = owners.as_slice() else {
            continue;
        };
        skin.applies_to = Some((*owner).clone());
        skin.subcategory = subcategories.get(*owner).cloned();
        resolved += 1;
    }
    resolved
}

/// Correct component-only cosmetic matches when the base weapon has a
/// separately authored variant. Shared Pattern components identify the weapon
/// family, but cannot distinguish a substantially different variant mesh.
/// Exclusive mod compatibility and matching socket families provide that
/// missing authored relationship. Only a lone geometry outlier in an
/// otherwise tight base-skin cluster is moved.
fn reconcile_distinct_weapon_variant_skins(
    items: &mut [GearItem],
    cache: &quicktag_scanner::TagCache,
    geometry_tolerance: f32,
) -> usize {
    let explicit_weapon_names = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon Mod") && !item.mod_is_universal)
        .flat_map(|item| item.compatible_weapons.iter().cloned())
        .collect::<FxHashSet<_>>();
    if explicit_weapon_names.is_empty() {
        return 0;
    }

    let socket_index = crate::geometry::WeaponModSocketIndex::new();
    let socket_families = |model| {
        socket_index
            .signature_for_model(cache, model)
            .map(|signature| {
                signature
                    .into_iter()
                    .map(|(family, _)| family)
                    .sorted()
                    .dedup()
                    .collect::<Vec<_>>()
            })
    };
    let base_weapons = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && item.rarity == Some(GearRarity::Standard)
                && is_canonical_weapon_skin_owner(item)
        })
        .filter_map(|item| {
            Some((
                item.name.clone(),
                (
                    item.model_tag?,
                    item.subcategory.clone(),
                    socket_families(item.model_tag?)?,
                ),
            ))
        })
        .collect::<FxHashMap<_, _>>();
    let variants = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && item.rarity == Some(GearRarity::Quest)
                && explicit_weapon_names.contains(&item.name)
                && !base_weapons.contains_key(&item.name)
        })
        .filter_map(|item| {
            Some((
                item.name.clone(),
                item.subcategory.clone(),
                socket_families(item.model_tag?)?,
            ))
        })
        .unique_by(|(name, _, _)| name.clone())
        .collect::<Vec<_>>();
    let model_signatures = items
        .iter()
        .filter_map(|item| {
            let model = item.model_tag?;
            Some((
                model,
                crate::geometry::model_pattern_structure_signature(cache, model)?,
            ))
        })
        .collect::<FxHashMap<_, _>>();

    let mut corrections = vec![];
    for (index, skin) in items.iter().enumerate().filter(|(_, item)| {
        item.item_type.as_deref() == Some("Weapon Skin") && item.applies_to.is_some()
    }) {
        let Some((base_model, base_subcategory, base_families)) = skin
            .applies_to
            .as_ref()
            .and_then(|owner| base_weapons.get(owner))
        else {
            continue;
        };
        let Some(skin_signature) = skin
            .model_tag
            .and_then(|model| model_signatures.get(&model))
        else {
            continue;
        };
        let Some(base_signature) = model_signatures.get(base_model) else {
            continue;
        };
        if skin_signature
            .closest_geometry_distance(base_signature)
            .is_none_or(|distance| distance <= geometry_tolerance)
        {
            continue;
        }

        let clustered_siblings = items
            .iter()
            .filter(|candidate| {
                candidate.item_type.as_deref() == Some("Weapon Skin")
                    && candidate.applies_to == skin.applies_to
                    && candidate.display_tag != skin.display_tag
            })
            .filter_map(|candidate| {
                model_signatures
                    .get(&candidate.model_tag?)?
                    .closest_geometry_distance(base_signature)
            })
            .filter(|distance| *distance <= geometry_tolerance)
            .count();
        if clustered_siblings < 2 {
            continue;
        }

        let candidates = variants
            .iter()
            .filter(|(_, subcategory, families)| {
                subcategory == base_subcategory && families == base_families
            })
            .collect::<Vec<_>>();
        let [(owner, subcategory, _)] = candidates.as_slice() else {
            continue;
        };
        corrections.push((index, owner.clone(), subcategory.clone()));
    }

    let count = corrections.len();
    for (index, owner, subcategory) in corrections {
        items[index].applies_to = Some(owner);
        items[index].subcategory = subcategory;
    }
    count
}

fn model_pattern_components(
    cache: &quicktag_scanner::TagCache,
    model: TagHash,
) -> FxHashSet<TagHash> {
    let mut components = FxHashSet::default();
    let mut seen = FxHashSet::default();
    let mut queue = std::collections::VecDeque::from([(model, 0usize)]);
    seen.insert(model);
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= 12 {
            continue;
        }
        let Some(scan) = cache.hashes.get(&node) else {
            continue;
        };
        for child in scan.file_hashes.iter().map(|reference| reference.hash) {
            let Some(entry) = package_manager().get_entry(child) else {
                continue;
            };
            if entry.reference == MODEL_PATTERN_COMPONENT_REFERENCE {
                components.insert(child);
            }
            if matches!(
                entry.reference,
                MODEL_PATTERN_REFERENCE | MODEL_PATTERN_COMPONENT_REFERENCE
            ) && seen.insert(child)
            {
                queue.push_back((child, depth + 1));
            }
        }
    }
    components
}

/// Some current rows omit a usable path and have no resolvable Pattern frame.
/// A current cosmetic group may fill those holes only after the independent
/// path/geometry/reference passes prove either one owner or a 2x-supported
/// dominant owner. This preserves explicit special-family collisions.
fn propagate_current_skin_groups(items: &mut [GearItem]) -> usize {
    let weapon_names = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.internal_hash?, item.name.clone())))
        .collect::<FxHashMap<_, _>>();
    let generated_owners = generated_skin_path_owners();
    let exact_candidates = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon Skin")
                && item.definition_type_code == Some(0x138)
                // 0x811c is the high half of the FNV empty sentinel, used by
                // ungrouped/special cosmetics rather than a real group row.
                && item
                    .definition_group_key
                    .is_some_and(|key| key as u16 != 0x811c)
        })
        .filter_map(|item| {
            Some((
                item.cosmetic_group_key()?,
                direct_skin_weapon_hash(item, &generated_owners)
                    .and_then(|hash| weapon_names.get(&hash))?
                    .clone(),
            ))
        })
        .into_group_map();
    let exact_owners = exact_candidates
        .into_iter()
        .filter_map(|(group, owners)| {
            let owners = owners.into_iter().unique().collect::<Vec<_>>();
            (owners.len() == 1).then(|| (group, owners.into_iter().next().unwrap()))
        })
        .collect::<FxHashMap<_, _>>();
    let subcategories = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.name.clone(), item.subcategory.clone()?)))
        .collect::<FxHashMap<_, _>>();
    let mut resolved = 0;
    for skin in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Weapon Skin")
            && item.definition_type_code == Some(0x138)
            && item
                .definition_group_key
                .is_some_and(|key| key as u16 != 0x811c)
    }) {
        let Some(owner) = skin
            .cosmetic_group_key()
            .and_then(|group| exact_owners.get(&group))
        else {
            continue;
        };
        if skin.applies_to.as_ref() == Some(owner) {
            continue;
        }
        skin.applies_to = Some(owner.clone());
        skin.subcategory = subcategories.get(owner).cloned();
        resolved += 1;
    }

    let candidates = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon Skin")
                && item.definition_type_code == Some(0x138)
        })
        .filter_map(|item| {
            let owner = item.applies_to.as_deref()?;
            if owner == "Certification test (no weapon)" {
                return None;
            }
            Some((item.cosmetic_group_key()?, owner.to_owned()))
        })
        .into_group_map();
    let owners = candidates
        .into_iter()
        .filter_map(|(group, owners)| {
            let mut ranked = owners.into_iter().counts().into_iter().collect::<Vec<_>>();
            ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
            let (owner, support) = ranked.first()?;
            let runner_up = ranked.get(1).map_or(0, |(_, count)| *count);
            (ranked.len() == 1 || (*support >= 2 && *support >= runner_up.saturating_mul(2)))
                .then(|| (group, owner.clone()))
        })
        .collect::<FxHashMap<_, _>>();
    for skin in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Weapon Skin")
            && item.definition_type_code == Some(0x138)
            && item.applies_to.is_none()
    }) {
        let Some(owner) = skin
            .cosmetic_group_key()
            .and_then(|group| owners.get(&group))
        else {
            continue;
        };
        skin.applies_to = Some(owner.clone());
        skin.subcategory = subcategories.get(owner).cloned();
        resolved += 1;
    }
    resolved
}

fn assign_legacy_unresolved_skin_owners(items: &mut [GearItem]) {
    let weapon_names = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.internal_hash?, item.name.clone())))
        .collect::<FxHashMap<_, _>>();
    let subcategories = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter_map(|item| Some((item.name.clone(), item.subcategory.clone()?)))
        .collect::<FxHashMap<_, _>>();

    for skin in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Weapon Skin")
            && item.definition_type_code != Some(0x138)
            && item.applies_to.is_none()
    }) {
        let Some(owner) = skin
            .definition_group_index()
            .and_then(skin_group_weapon_hash)
            .and_then(|hash| weapon_names.get(&hash))
        else {
            continue;
        };
        skin.applies_to = Some(owner.clone());
        skin.subcategory = subcategories.get(owner).cloned();
    }
}

fn runner_shell_model(description: &str) -> Option<&str> {
    let header = description.lines().next()?;
    header
        .strip_prefix("SHELL MODEL:")
        .or_else(|| header.strip_prefix("FRAME MODEL:"))
        .or_else(|| header.strip_prefix("SHELL TYPE:"))
        .map(str::trim)
        .filter(|model| !model.is_empty())
}

fn runner_core_archetype(
    internal_name: Option<&str>,
    internal_hash: Option<u32>,
) -> Option<String> {
    if let Some(owner) = internal_name
        .and_then(|name| name.strip_prefix("implant_cores."))
        .and_then(|path| path.split('.').next())
    {
        return Some(if owner == "shared" {
            "All Shells".to_owned()
        } else {
            humanize_identifier(owner)
        });
    }

    internal_hash
        .is_some_and(|hash| SENTINEL_CORE_HASHES.contains(&hash))
        .then(|| "Sentinel".to_owned())
}

fn canonical_runner_shell_model(model: &str) -> &str {
    let model = model.strip_suffix(" [basic]").unwrap_or(model);
    if model.eq_ignore_ascii_case("rook") {
        "Rook"
    } else {
        model
    }
}

/// Current runner cosmetics retain their authored definition group even when
/// their wordlist path and `SHELL/FRAME MODEL` debug header are absent. Learn a
/// group only from path/header-backed runner rows and reject conflicting groups.
fn propagate_runner_skin_taxonomy(items: &mut [GearItem]) {
    let groups = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Runner Skin"))
        .filter_map(|item| Some((item.cosmetic_group_key()?, item.subcategory.clone()?)))
        .into_group_map();
    let groups = groups
        .into_iter()
        .filter_map(|(group, subcategories)| {
            let mut subcategories = subcategories.into_iter().unique();
            let subcategory = subcategories.next()?;
            subcategories
                .next()
                .is_none()
                .then_some((group, subcategory))
        })
        .collect::<FxHashMap<_, _>>();

    for item in items.iter_mut().filter(|item| {
        item.item_type.as_deref() == Some("Runner Skin") && item.subcategory.is_none()
    }) {
        if let Some(subcategory) = item
            .cosmetic_group_key()
            .and_then(|group| groups.get(&group))
        {
            item.subcategory = Some(subcategory.clone());
        }
    }
}

fn runner_shell_names(items: &[GearItem]) -> FxHashMap<String, String> {
    let mut candidates = FxHashMap::<String, FxHashSet<String>>::default();
    for item in items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Runner Skin"))
    {
        let Some(model) = item.description.as_deref().and_then(runner_shell_model) else {
            continue;
        };
        let model = canonical_runner_shell_model(model).to_owned();
        let internal_archetype = item
            .internal_name
            .as_deref()
            .and_then(|name| name.strip_prefix("heroes."))
            .and_then(|path| path.split('.').next())
            .map(humanize_identifier);
        for archetype in internal_archetype
            .into_iter()
            .chain(item.subcategory.iter().cloned())
        {
            candidates
                .entry(archetype)
                .or_default()
                .insert(model.clone());
        }
    }
    candidates
        .into_iter()
        .filter_map(|(archetype, models)| {
            (models.len() == 1).then(|| {
                (
                    archetype,
                    models.into_iter().next().expect("one shell model"),
                )
            })
        })
        .collect()
}

fn assign_runner_owners(items: &mut [GearItem]) {
    let shell_names = runner_shell_names(items);
    for item in items.iter_mut() {
        let Some(archetype) = (match item.item_type.as_deref() {
            Some("Runner Skin") => item.subcategory.clone(),
            Some("Runner Core") => {
                runner_core_archetype(item.internal_name.as_deref(), item.internal_hash)
            }
            _ => None,
        }) else {
            continue;
        };
        item.applies_to = Some(shell_names.get(&archetype).cloned().unwrap_or(archetype));
    }
}

fn runner_shell_taxonomy(items: &[GearItem]) -> FxHashMap<String, String> {
    let mut candidates = FxHashMap::<String, FxHashSet<&str>>::default();
    for item in items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Runner Skin"))
    {
        let Some(model) = item.description.as_deref().and_then(runner_shell_model) else {
            continue;
        };
        let Some(subcategory) = item.subcategory.as_deref() else {
            continue;
        };
        candidates
            .entry(model.to_owned())
            .or_default()
            .insert(subcategory);
    }
    candidates
        .into_iter()
        .filter_map(|(model, subcategories)| {
            (subcategories.len() == 1).then(|| {
                (
                    model,
                    (*subcategories.iter().next().expect("one subtype")).to_owned(),
                )
            })
        })
        .collect()
}

fn append_taxonomy_tags(item: &mut GearItem, facets: Vec<String>) {
    item.types
        .retain(|kind| !kind.starts_with("item_type_") && !kind.starts_with("subtype_"));
    item.types.extend(facets);
    if let Some(category) = item.item_type.as_deref() {
        item.types
            .push(format!("item_type_{}", identifier(category)));
    }
    if let Some(subcategory) = item.subcategory.as_deref() {
        item.types
            .push(format!("subtype_{}", identifier(subcategory)));
    }
    if let Some(hash) = item.raw_category_hash
        && !matches!(hash, EMPTY_HASH)
    {
        item.types.push(format!("category_#{hash:08X}"));
    }
    if let Some(hash) = item.raw_subcategory_hash
        && !matches!(hash, EMPTY_HASH)
    {
        item.types.push(format!("subtype_#{hash:08X}"));
    }
    item.types.sort();
    item.types.dedup();
}

fn identifier(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn weapon_subcategory_from_internal(internal_name: &str) -> Option<String> {
    let category = if internal_name.contains(".auto_rifles.")
        || internal_name.starts_with("weapons.auto_rifles.")
    {
        "Assault Rifle"
    } else if internal_name.contains(".pistols.") || internal_name.starts_with("weapons.pistols.") {
        "Pistol"
    } else if internal_name.contains(".submachineguns.")
        || internal_name.starts_with("weapons.submachineguns.")
    {
        "Submachine Gun"
    } else if internal_name.contains(".shotguns.") || internal_name.starts_with("weapons.shotguns.")
    {
        "Shotgun"
    } else if internal_name.contains(".sniper_rifles.")
        || internal_name.starts_with("weapons.sniper_rifles.")
    {
        "Sniper Rifle"
    } else if internal_name.contains(".railguns.") || internal_name.starts_with("weapons.railguns.")
    {
        "Railgun"
    } else if internal_name.contains(".machineguns.")
        || internal_name.starts_with("weapons.machineguns.")
    {
        "Machine Gun"
    } else if internal_name.contains(".marksman_rifles.")
        || internal_name.starts_with("weapons.marksman_rifles.")
    {
        "Marksman Rifle"
    } else {
        return None;
    };
    Some(category.to_owned())
}

fn weapon_subcategory_from_name(name: &str) -> Option<String> {
    let normalized = format!(" {} ", name.to_ascii_uppercase().replace('-', " "));
    let category = if normalized.contains(" ASSAULT RIFLE ") || normalized.contains(" AR ") {
        "Assault Rifle"
    } else if normalized.contains(" PISTOL ") {
        "Pistol"
    } else if normalized.contains(" SMG ") {
        "Submachine Gun"
    } else if normalized.contains(" SHOTGUN ") {
        "Shotgun"
    } else if normalized.contains(" SNIPER ") {
        "Sniper Rifle"
    } else if normalized.contains(" RAILGUN ") || normalized.contains(" RG ") {
        "Railgun"
    } else if normalized.contains(" LMG ") || normalized.contains(" HMG ") {
        "Machine Gun"
    } else {
        return None;
    };
    Some(category.to_owned())
}

fn weapon_subcategory(item: &GearItem) -> Option<String> {
    if item.rarity == Some(GearRarity::Contraband) {
        return Some("Hybrid Weapon".to_owned());
    }

    weapon_subcategory_from_internal(item.internal_name.as_deref()?)
}

fn weapon_subcategory_from_hash(hash: u32) -> Option<String> {
    let value = match hash {
        WEAPON_TYPE_ASSAULT_RIFLE => "Assault Rifle",
        WEAPON_TYPE_PISTOL => "Pistol",
        WEAPON_TYPE_SUBMACHINE_GUN => "Submachine Gun",
        WEAPON_TYPE_SHOTGUN => "Shotgun",
        WEAPON_TYPE_SNIPER_RIFLE => "Sniper Rifle",
        WEAPON_TYPE_RAILGUN => "Railgun",
        WEAPON_TYPE_MACHINE_GUN => "Machine Gun",
        WEAPON_TYPE_MARKSMAN_RIFLE => "Marksman Rifle",
        _ => return None,
    };
    Some(value.to_owned())
}

fn load_display_to_hash_map() -> FxHashMap<TagHash, u32> {
    let mut result = FxHashMap::default();
    for (tag, _) in package_manager().get_all_by_reference(DISPLAY_TO_HASH_REFERENCE) {
        let Ok(data) = package_manager().read_tag(tag) else {
            continue;
        };
        for (internal_hash, display_tag) in parse_hash_tag_pairs(&data, |display_tag| {
            package_manager()
                .get_entry(display_tag)
                .is_some_and(|entry| entry.reference == GEAR_DISPLAY_REFERENCE)
        }) {
            if internal_hash != EMPTY_HASH && internal_hash != 0 {
                result.insert(display_tag, internal_hash);
            }
        }
    }
    result
}

fn load_hash_to_definition_map() -> FxHashMap<u32, TagHash> {
    let mut result = FxHashMap::default();
    for (tag, _) in package_manager().get_all_by_reference(HASH_TO_DEFINITION_REFERENCE) {
        let Ok(data) = package_manager().read_tag(tag) else {
            continue;
        };
        for (internal_hash, definition_tag) in parse_hash_tag_pairs(&data, |definition_tag| {
            package_manager()
                .get_entry(definition_tag)
                .is_some_and(|entry| entry.reference == GEAR_DEFINITION_REFERENCE)
        }) {
            result.insert(internal_hash, definition_tag);
        }
    }
    result
}

/// Resolves an inventory definition to the exact render Pattern selected by
/// the game's investment data. This is the same two-table join used at
/// runtime: inventory pattern index -> pattern-global id -> Tag64 Pattern.
struct InvestmentPatternResolver {
    pattern_internal_hashes: Vec<u32>,
    pattern_globals: Vec<u32>,
    pattern_skin_ids: Vec<u16>,
    assignments: FxHashMap<u32, TagHash>,
    cosmetics: Vec<Option<TagHash>>,
    cosmetics_by_api_hash: FxHashMap<u32, TagHash>,
}

impl InvestmentPatternResolver {
    fn load() -> Self {
        let mut assignments = FxHashMap::default();
        let mut assignment_tables =
            package_manager().get_all_by_reference(PATTERN_ASSIGNMENT_TABLE_REFERENCE);
        assignment_tables.sort_by_key(|(tag, _)| *tag);
        for (tag, _) in assignment_tables {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            let Some(range) = table_range(&data, 0x8, 0x18) else {
                continue;
            };
            for record in data[range].chunks_exact(0x18) {
                let Some(global_id) = read_u32(record, 0) else {
                    continue;
                };
                let Some(tag) = resolve_tag64_union(record, 0x8) else {
                    continue;
                };
                if package_manager()
                    .get_entry(tag)
                    .is_some_and(|entry| entry.reference == MODEL_PATTERN_REFERENCE)
                {
                    assignments.entry(global_id).or_insert(tag);
                }
            }
        }

        let mut pattern_globals = vec![];
        let mut pattern_internal_hashes = vec![];
        let mut pattern_skin_ids = vec![];
        let mut best_global_table_score = (0usize, 0usize);
        let mut global_tables =
            package_manager().get_all_by_reference(PATTERN_GLOBAL_TABLE_REFERENCE);
        global_tables.sort_by_key(|(tag, _)| *tag);
        for (tag, _) in global_tables {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            // Goliath expanded pattern-global rows from 0x38 to 0x48 bytes.
            // Both layouts have the same header and field offsets, so blindly
            // accepting the shorter stride still yields an in-bounds table but
            // drifts every definition index onto unrelated bytes. Select the
            // layout whose global IDs resolve through the authored assignment
            // table; this also retains support for pre-Goliath packages.
            for stride in [0x38, 0x48] {
                let Some(range) = table_range(&data, 0x8, stride) else {
                    continue;
                };
                let records = data[range].chunks_exact(stride).collect::<Vec<_>>();
                let globals = records
                    .iter()
                    .filter_map(|record| read_u32(record, 0x8))
                    .collect::<Vec<_>>();
                let assignment_hits = globals
                    .iter()
                    .filter(|global| assignments.contains_key(global))
                    .count();
                let score = (assignment_hits, globals.len());
                if score > best_global_table_score {
                    best_global_table_score = score;
                    pattern_internal_hashes = records
                        .iter()
                        .filter_map(|record| read_u32(record, 0))
                        .collect();
                    pattern_globals = globals;
                    pattern_skin_ids = records
                        .iter()
                        .filter_map(|record| read_u16(record, 0xc))
                        .collect();
                }
            }
        }

        let mut cosmetics = vec![];
        let mut cosmetics_by_api_hash = FxHashMap::default();
        let mut best_cosmetic_map_score = (0usize, 0usize);
        let mut cosmetic_maps =
            package_manager().get_all_by_reference(INVESTMENT_COSMETIC_MAP_REFERENCE);
        cosmetic_maps.sort_by_key(|(tag, _)| *tag);
        for (tag, _) in cosmetic_maps {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            for stride in [0x20, 0x8] {
                let Some(range) = table_range(&data, 0x8, stride) else {
                    continue;
                };
                let records = data[range].chunks_exact(stride);
                let candidate = records
                    .map(|record| {
                        let api_hash = read_u32(record, 0);
                        let pattern = if stride == 0x20 {
                            resolve_direct_tag64(record, 0x8)
                                .or_else(|| resolve_tag64_union(record, 0x8))
                                .and_then(resolve_cosmetic_pattern)
                        } else {
                            read_u32(record, 4)
                                .map(TagHash)
                                .and_then(resolve_cosmetic_pattern)
                        };
                        (api_hash, pattern)
                    })
                    .collect::<Vec<_>>();
                let score = (
                    candidate
                        .iter()
                        .filter(|(_, pattern)| pattern.is_some())
                        .count(),
                    candidate.len(),
                );
                if score > best_cosmetic_map_score {
                    best_cosmetic_map_score = score;
                    cosmetics = candidate.iter().map(|(_, pattern)| *pattern).collect();
                    cosmetics_by_api_hash = candidate
                        .iter()
                        .filter_map(|(api_hash, pattern)| Some(((*api_hash)?, (*pattern)?)))
                        .collect();
                }
            }
        }

        Self {
            pattern_internal_hashes,
            pattern_globals,
            pattern_skin_ids,
            assignments,
            cosmetics,
            cosmetics_by_api_hash,
        }
    }

    fn resolve_definition(&self, internal_hash: Option<u32>, definition: &[u8]) -> Option<TagHash> {
        let (translation, pattern_index) = definition_pattern_translation(definition)?;

        // Cosmetic definitions use the same serialized field for an art
        // arrangement index. In current packages that integer can also index
        // a valid weapon Pattern row, so resolve the authored cosmetic API map
        // before interpreting it as WeaponPatternIndex.
        if let Some(cosmetic) = internal_hash
            .and_then(|hash| self.cosmetics_by_api_hash.get(&hash))
            .copied()
        {
            return Some(cosmetic);
        }

        if pattern_index != u16::MAX {
            if let Some(pattern) = self
                .pattern_globals
                .get(usize::from(pattern_index))
                .and_then(|global_id| self.assignments.get(global_id))
                .copied()
            {
                return Some(pattern);
            }
        }

        // Cosmetics use an art-arrangement/global-row index in place of the
        // normal weapon Pattern index. Goliath stores it directly immediately
        // before WeaponPatternIndex; older layouts use the arrangement array.
        let arrangement_index = (pattern_index != u16::MAX)
            .then_some(u32::from(pattern_index))
            .or_else(|| {
                let arrangement_header = translation.checked_add(0x2c)?;
                let range = table_range(definition, arrangement_header, 4)?;
                read_u32(definition, range.start).filter(|index| *index != u32::MAX)
            })?;
        let skin_id = *self
            .pattern_skin_ids
            .get(usize::try_from(arrangement_index).ok()?)?;
        self.cosmetics.get(usize::from(skin_id)).copied().flatten()
    }

    fn definition_internal_path_hash(&self, definition: &[u8]) -> Option<u32> {
        let (_, pattern_index) = definition_pattern_translation(definition)?;
        self.pattern_internal_hashes
            .get(usize::from(pattern_index))
            .copied()
            .filter(|hash| !matches!(*hash, 0 | EMPTY_HASH))
    }
}

fn definition_pattern_translation(definition: &[u8]) -> Option<(usize, u16)> {
    let translation = definition
        .chunks_exact(4)
        .position(|bytes| {
            u32::from_le_bytes(bytes.try_into().unwrap()) == PATTERN_TRANSLATION_BLOCK_MARKER
        })?
        .checked_mul(4)?;
    let pattern_index = translation
        // ResourcePointer serialization starts the translation payload four
        // bytes after its class marker; the global row is at +0x68.
        .checked_add(0x6c)
        .and_then(|offset| read_u16(definition, offset))?;
    Some((translation, pattern_index))
}

fn resolve_direct_tag64(data: &[u8], offset: usize) -> Option<TagHash> {
    let wide = read_u64(data, offset)?;
    package_manager()
        .lookup
        .tag64_entries
        .get(&wide)
        .map(|entry| entry.hash32)
}

fn resolve_cosmetic_pattern(tag: TagHash) -> Option<TagHash> {
    if package_manager()
        .get_entry(tag)
        .is_some_and(|entry| entry.reference == MODEL_PATTERN_REFERENCE)
    {
        return Some(tag);
    }
    let data = package_manager().read_tag(tag).ok()?;
    let mut candidates = (0..data.len().saturating_sub(7))
        .step_by(8)
        .filter_map(|offset| resolve_direct_tag64(&data, offset))
        .chain(resolve_tag64_union(&data, 8))
        .filter(|candidate| {
            package_manager()
                .get_entry(*candidate)
                .is_some_and(|entry| entry.reference == MODEL_PATTERN_REFERENCE)
        })
        .collect::<FxHashSet<_>>();
    (candidates.len() == 1)
        .then(|| candidates.drain().next())
        .flatten()
}

fn resolve_tag64_union(data: &[u8], offset: usize) -> Option<TagHash> {
    if read_u32(data, offset + 4)? != 0 {
        let tag = TagHash(read_u32(data, offset)?);
        package_manager().get_entry(tag).map(|_| tag)
    } else {
        let wide = read_u64(data, offset + 8)?;
        package_manager()
            .lookup
            .tag64_entries
            .get(&wide)
            .map(|entry| entry.hash32)
    }
}

/// Resolve the compact category IDs used by investment definitions through
/// the game's authored namespace and category registries. Category hashes that
/// have no known preimage remain lossless and searchable as `namespace.#HASH`.
fn load_internal_category_registry(wordlist: &FxHashMap<u32, String>) -> FxHashMap<u16, String> {
    let mut namespace_hashes = FxHashMap::default();
    let mut namespace_tags =
        package_manager().get_all_by_reference(INTERNAL_CATEGORY_NAMESPACE_REGISTRY_REFERENCE);
    namespace_tags.sort_by_key(|(tag, _)| *tag);
    for (tag, _) in namespace_tags {
        let Ok(data) = package_manager().read_tag(tag) else {
            continue;
        };
        namespace_hashes.extend(parse_internal_category_namespace_hashes(&data));
    }

    let mut registry = FxHashMap::default();
    let mut tags = package_manager().get_all_by_reference(INTERNAL_CATEGORY_REGISTRY_REFERENCE);
    tags.sort_by_key(|(tag, _)| *tag);

    for (tag, _) in tags {
        let Ok(data) = package_manager().read_tag(tag) else {
            continue;
        };
        registry.extend(parse_internal_category_registry(
            &data,
            &namespace_hashes,
            wordlist,
        ));
    }

    registry
}

fn parse_internal_category_namespace_hashes(data: &[u8]) -> FxHashMap<u16, u32> {
    let Some(range) = table_range(data, INTERNAL_CATEGORY_NAMESPACE_TABLE_OFFSET, 4) else {
        return FxHashMap::default();
    };
    data[range]
        .chunks_exact(4)
        .enumerate()
        .filter_map(|(index, bytes)| {
            Some((
                u16::try_from(index).ok()?,
                u32::from_le_bytes(bytes.try_into().unwrap()),
            ))
        })
        .collect()
}

fn parse_internal_category_registry(
    data: &[u8],
    namespace_hashes: &FxHashMap<u16, u32>,
    wordlist: &FxHashMap<u32, String>,
) -> FxHashMap<u16, String> {
    let Some(index_range) = table_range(data, INTERNAL_CATEGORY_INDEX_TABLE_OFFSET, 8) else {
        return FxHashMap::default();
    };
    let Some(lookup_range) = table_range(data, INTERNAL_CATEGORY_LOOKUP_TABLE_OFFSET, 8) else {
        return FxHashMap::default();
    };

    let index_records = data[index_range]
        .chunks_exact(8)
        .enumerate()
        .filter_map(|(index, record)| {
            let hash = u32::from_le_bytes(record[0..4].try_into().unwrap());
            let packed = u32::from_le_bytes(record[4..8].try_into().unwrap());
            Some((u16::try_from(index).ok()?, (hash, (packed >> 16) as u16)))
        })
        .collect::<FxHashMap<_, _>>();
    let lookup_hashes = data[lookup_range]
        .chunks_exact(8)
        .filter_map(|record| {
            let hash = u32::from_le_bytes(record[0..4].try_into().unwrap());
            let index = u32::from_le_bytes(record[4..8].try_into().unwrap());
            Some((u16::try_from(index).ok()?, hash))
        })
        .collect::<FxHashMap<_, _>>();

    let mut namespace_names = namespace_hashes
        .iter()
        .filter_map(|(index, hash)| Some((*index, wordlist.get(hash)?.clone())))
        .collect::<FxHashMap<_, _>>();
    for (hash, namespace) in index_records.values() {
        let Some((prefix, _)) = wordlist.get(hash).and_then(|name| name.split_once('.')) else {
            continue;
        };
        namespace_names
            .entry(*namespace)
            .or_insert_with(|| prefix.to_owned());
    }

    let mut registry = FxHashMap::default();
    for (index, (indexed_hash, namespace)) in index_records {
        let hash = lookup_hashes.get(&index).copied().unwrap_or(indexed_hash);
        let label = wordlist.get(&hash).cloned().unwrap_or_else(|| {
            if let Some(namespace) = namespace_names.get(&namespace) {
                format!("{namespace}.#{hash:08X}")
            } else if let Some(namespace_hash) = namespace_hashes.get(&namespace) {
                format!("namespace_#{namespace_hash:08X}.#{hash:08X}")
            } else {
                format!("category_hash_#{hash:08X}")
            }
        });
        registry.insert(index, label);
    }

    // Retain a lossless entry even if a future package adds a lookup record
    // before its parallel namespace/index record is understood.
    for (index, hash) in lookup_hashes {
        registry.entry(index).or_insert_with(|| {
            wordlist
                .get(&hash)
                .cloned()
                .unwrap_or_else(|| format!("category_hash_#{hash:08X}"))
        });
    }
    registry
}

fn extract_internal_categories(
    definition: &[u8],
    registry: &FxHashMap<u16, String>,
) -> Vec<String> {
    let Some(range) = table_range(definition, INTERNAL_CATEGORY_LIST_OFFSET, 2) else {
        return vec![];
    };
    let mut seen = FxHashSet::default();
    definition[range]
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .filter(|index| seen.insert(*index))
        .map(|index| {
            registry
                .get(&index)
                .cloned()
                .unwrap_or_else(|| format!("category_index_#{index:04X}"))
        })
        .collect()
}

fn table_range(data: &[u8], header: usize, record_size: usize) -> Option<std::ops::Range<usize>> {
    let count = usize::try_from(read_u64(data, header)?).ok()?;
    let relative = read_i64(data, header + 8)?;
    // Tiger table offsets are based at the relative-offset field and point to
    // a standard 0x10-byte table prefix before the first record.
    let start: usize = i64::try_from(header)
        .ok()?
        .checked_add(8)?
        .checked_add(relative)?
        .checked_add(0x10)?
        .try_into()
        .ok()?;
    let end = start.checked_add(count.checked_mul(record_size)?)?;
    (end <= data.len()).then_some(start..end)
}

fn parse_hash_tag_pairs(
    data: &[u8],
    mut is_expected_tag: impl FnMut(TagHash) -> bool,
) -> Vec<(u32, TagHash)> {
    let mut result = vec![];
    for offset in (0..data.len().saturating_sub(0x14)).step_by(4) {
        let internal_hash = read_u32(data, offset).unwrap_or_default();
        let tag = TagHash(read_u32(data, offset + 0x10).unwrap_or_default());
        if is_expected_tag(tag) {
            result.push((internal_hash, tag));
        }
    }
    result
}

fn localized_at(
    data: &[u8],
    scope_offset: usize,
    hash_offset: usize,
    strings: &StringCache,
    localized: &LocalizedStringResolver,
) -> Option<String> {
    let hash = localized_hash_at(data, hash_offset)?;
    let scoped = read_u32(data, scope_offset)
        .and_then(|scope| localized.get(scope, hash))
        .cloned();
    scoped.or_else(|| {
        let values = strings.get(&hash)?;
        (values.len() == 1).then(|| values[0].clone())
    })
}

fn localized_parts_at(
    data: &[u8],
    scope_offset: usize,
    hash_offset: usize,
    localized: &LocalizedStringResolver,
) -> Vec<LocalizedStringPart> {
    let Some(scope) = read_u32(data, scope_offset) else {
        return vec![];
    };
    let Some(hash) = localized_hash_at(data, hash_offset) else {
        return vec![];
    };
    localized.parts(scope, hash).unwrap_or_default().to_vec()
}

fn localized_hash_at(data: &[u8], offset: usize) -> Option<u32> {
    let hash = read_u32(data, offset)?;
    (!matches!(hash, 0 | EMPTY_HASH)).then_some(hash)
}

fn wordlist_name_at(
    data: &[u8],
    offset: usize,
    wordlist: &FxHashMap<u32, String>,
) -> Option<String> {
    let hash = read_u32(data, offset)?;
    if matches!(hash, 0 | EMPTY_HASH) {
        return None;
    }
    wordlist
        .get(&hash)
        .cloned()
        .or_else(|| Some(format!("#{hash:08X}")))
}

fn extract_types(data: &[u8], wordlist: &FxHashMap<u32, String>) -> Vec<String> {
    let mut types = vec![];
    for bytes in data.chunks_exact(4) {
        let hash = u32::from_le_bytes(bytes.try_into().unwrap());
        let Some(value) = wordlist.get(&hash) else {
            continue;
        };
        if value.starts_with("type_")
            || value.starts_with("item_type")
            || value.starts_with("faction_")
            || value.starts_with("zone_")
            || matches!(value.as_str(), "generic" | "ranked" | "static" | "weapon")
        {
            types.push(value.clone());
        }
    }
    types.sort();
    types.dedup();
    types
}

fn parse_rarity(data: &[u8]) -> Option<GearRarity> {
    // Normal item tiers are stored as an enum immediately after this field marker.
    let tier = data
        .chunks_exact(4)
        .position(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) == RARITY_MARKER)
        .and_then(|index| read_u32(data, index * 4 + 4))
        .and_then(GearRarity::from_tier_value);
    if tier.is_some() {
        return tier;
    }

    // Special tiers may be stored directly as raw hashes.
    data.chunks_exact(4)
        .find_map(|bytes| GearRarity::from_raw_hash(u32::from_le_bytes(bytes.try_into().unwrap())))
        .or_else(|| {
            read_u32(data, data.len().checked_sub(4)?)
                .and_then(|footer| GearRarity::from_footer_code((footer >> 16) as u16))
        })
}

/// The updated investment definition explicitly authors rarity as an internal
/// category.  It is more specific than incidental packed words elsewhere in
/// the (now much larger) definition. `dynamic` is the ordinary baseline weapon
/// tier shown as gray by the game; runtime rolls live in separate display rows.
fn parse_rarity_from_categories(categories: &[String]) -> Option<GearRarity> {
    categories.iter().find_map(|category| {
        Some(match category.as_str() {
            "rarity_tier.grey" | "rarity_tier.dynamic" => GearRarity::Standard,
            "rarity_tier.green" => GearRarity::Enhanced,
            "rarity_tier.blue" => GearRarity::Deluxe,
            "rarity_tier.purple" => GearRarity::Superior,
            "rarity_tier.gold" => GearRarity::Prestige,
            "rarity_tier.contraband" => GearRarity::Contraband,
            "rarity_tier.quest" => GearRarity::Quest,
            "rarity_tier.unique" => GearRarity::Unique,
            _ => return None,
        })
    })
}

fn parse_buying_price(data: &[u8]) -> Option<u32> {
    let marker = data
        .chunks_exact(4)
        .position(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) == PRICE_MARKER)?
        * 4;
    // Implant offset 0x52C is this shared component's base price at +8.
    let price = f32::from_bits(read_u32(data, marker + 8)?);
    (price.is_finite() && price >= 0.0 && f64::from(price) <= f64::from(u32::MAX))
        .then_some(price as u32)
}

fn parse_price(data: &[u8]) -> Option<u32> {
    data.chunks_exact(4)
        .position(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) == PRICE_MARKER)
        .and_then(|index| {
            let offset = index * 4;
            let base = f32::from_bits(read_u32(data, offset + 8)?);
            let multiplier = f32::from_bits(read_u32(data, offset + 20)?);
            let price = base * multiplier;
            (base.is_finite()
                && multiplier.is_finite()
                && base > 0.0
                && multiplier > 0.0
                && price.is_finite()
                && price <= u32::MAX as f32)
                .then_some(price as u32)
        })
}

fn is_weapon_or_melee_skin_type(code: u16) -> bool {
    code == 0x138
}

fn definition_group_key(data: &[u8]) -> Option<u32> {
    // Marathon's July 2026 definition record appended fields to cosmetic
    // definitions. The type code identifies the record layout reliably.
    let footer_offset = if definition_type_code(data) == Some(0x138) {
        0x74
    } else {
        0x4c
    };
    read_u32(data, data.len().checked_sub(footer_offset)?)
}

fn definition_type_code(data: &[u8]) -> Option<u16> {
    let value = read_u32(data, data.len().checked_sub(8)?)?;
    Some(if data.len() % 4 == 2 {
        (value >> 16) as u16
    } else {
        value as u16
    })
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u16(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        data.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn read_i64(data: &[u8], offset: usize) -> Option<i64> {
    Some(i64::from_le_bytes(
        data.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

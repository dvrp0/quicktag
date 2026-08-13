use std::sync::Arc;

use eframe::egui::{self, Color32, RichText};
use itertools::Itertools;
use quicktag_strings::localized::{LocalizedLanguage, LocalizedStringResolver, StringCache};
use rustc_hash::{FxHashMap, FxHashSet};
use tiger_pkg::{GameVersion, TagHash, package_manager};

use crate::geometry::WeaponModRarity;

use super::{View, ViewAction, common::ResponseExt};

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
const RARITY_UNIQUE_VALUE: u32 = 0x00f60000;
const RARITY_CONTRABAND_VALUE: u32 = 0x00f70000;
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
    mod_is_universal: bool,
    compatible_weapons: Vec<String>,
    classification: Option<String>,
    types: Vec<String>,
    internal_categories: Vec<String>,
    price: Option<u32>,
    description: Option<String>,
}

pub struct GearView {
    items: Vec<GearItem>,
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
                let item_types = collect_item_types(&items);
                let rarities = collect_rarities(&items);
                let internal_category_counts = collect_internal_category_counts(&items);
                let mut view = Self {
                    items,
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

        ModelWeaponCatalog {
            weapons,
            runner_skins,
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
        assign_legacy_unresolved_skin_owners(&mut self.items);
        self.reconcile_weapon_mod_models(cache);
        self.update_filter();
    }

    fn reconcile_weapon_mod_models(&mut self, cache: &quicktag_scanner::TagCache) {
        let weapons = self
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon")
                    && item.rarity == Some(GearRarity::Standard)
                    && (is_canonical_weapon_skin_owner(item)
                        || item.internal_hash == Some(KKV_9SD_HASH))
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
                    || item
                        .item_type
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .subcategory
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .applies_to
                        .as_deref()
                        .is_some_and(|value| value.to_lowercase().contains(&search))
                    || item
                        .mod_category
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

impl View for GearView {
    fn view(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) -> Option<ViewAction> {
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

                ui.label(format!(
                    "Showing {} of {} gear records",
                    self.filtered_indices.len(),
                    self.items.len()
                ));
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
                            if gear_item_button(ui, item, selected).clicked() {
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

            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new(&item.name)
                        .color(item.rarity.map(GearRarity::color).unwrap_or(Color32::WHITE)),
                );
                if ui.small_button(format!("{}", item.display_tag)).clicked() {
                    action = Some(ViewAction::OpenTag(item.display_tag));
                }
                if item.item_type.as_deref() == Some("Weapon Skin")
                    && let Some(model) = item.model_tag
                    && ui.button("Go to Models").clicked()
                {
                    action = Some(ViewAction::ShowModel(model));
                }
            });
            ui.separator();

            let mod_compatibility = if item.mod_is_universal {
                Some("Universal".to_owned())
            } else {
                (!item.compatible_weapons.is_empty()).then(|| item.compatible_weapons.join(", "))
            };

            egui::Grid::new("gear_metadata")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .show(ui, |ui| {
                    metadata_row(ui, "Rarity", item.rarity.map(GearRarity::label));
                    metadata_row(ui, "Category", item.item_type.as_deref());
                    metadata_row(
                        ui,
                        if item.mod_category.is_some() {
                            "Mod category"
                        } else {
                            "Subtype"
                        },
                        item.subcategory.as_deref(),
                    );
                    metadata_row(
                        ui,
                        if matches!(
                            item.item_type.as_deref(),
                            Some("Runner Core" | "Runner Skin")
                        ) {
                            "Shell"
                        } else {
                            "Weapon"
                        },
                        item.applies_to.as_deref(),
                    );
                    metadata_row(ui, "Compatible weapons", mod_compatibility.as_deref());
                    metadata_row(ui, "Display class", item.classification.as_deref());
                    metadata_row(
                        ui,
                        "Types",
                        (!item.types.is_empty())
                            .then(|| item.types.join(", "))
                            .as_deref(),
                    );
                    metadata_row(ui, "Price", item.price.map(|v| format_number(v)).as_deref());
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

            ui.add_space(12.0);
            ui.strong("Description");
            ui.separator();
            let selected_mod = egui::ScrollArea::vertical()
                .id_salt(("gear_detail", item.display_tag.0))
                .show(ui, |ui| {
                    ui.label(item.description.as_deref().unwrap_or("—"));
                    compatible_mod_cards(ui, &self.items, item, self.selected)
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

fn gear_item_button(ui: &mut egui::Ui, item: &GearItem, selected: bool) -> egui::Response {
    let color = item.rarity.map(GearRarity::color).unwrap_or(Color32::GRAY);
    let fill_alpha = if selected { 82 } else { 34 };
    let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), fill_alpha);
    let stroke = egui::Stroke::new(if selected { 2.0 } else { 1.0 }, color);
    let label = item.name.as_str();
    ui.add_sized(
        [ui.available_width(), 38.0],
        egui::Button::new(RichText::new(label).color(Color32::WHITE).strong())
            .fill(fill)
            .stroke(stroke),
    )
    .tag_context(item.display_tag)
    .on_hover_text(format!(
        "{label}\n{} · {}",
        item.rarity
            .map(GearRarity::label)
            .unwrap_or("Unknown rarity"),
        item.mod_category
            .as_deref()
            .or(item.subcategory.as_deref())
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
                                    if gear_item_button(ui, &items[index], selected == Some(index))
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
    for item_type in items.iter().filter_map(|item| item.item_type.as_deref()) {
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
        || item
            .item_type
            .as_ref()
            .is_some_and(|value| selected_item_types.contains(value));
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
    let item_type = item.item_type.as_deref().unwrap_or("Uncategorized");
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
        let owners = if item.mod_is_universal {
            vec!["Universal".to_owned()]
        } else if !item.compatible_weapons.is_empty() {
            item.compatible_weapons.clone()
        } else {
            vec![
                item.mod_category
                    .as_deref()
                    .or(item.subcategory.as_deref())
                    .unwrap_or("Uncategorized")
                    .to_owned(),
            ]
        };
        owners
            .into_iter()
            .map(|owner| {
                if show_subcategories {
                    owner
                } else {
                    format!("{item_type} · {owner}")
                }
            })
            .collect()
    } else if show_subcategories {
        vec![
            item.subcategory
                .as_deref()
                .unwrap_or("Uncategorized")
                .to_owned(),
        ]
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

fn load_gear(strings: &StringCache) -> Result<Vec<GearItem>, String> {
    load_gear_for_language(strings, LocalizedLanguage::English)
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
        return load_gear_resolved(strings, &localized);
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
            mod_is_universal: false,
            compatible_weapons: vec![],
            classification,
            types: extract_types(&data, &wordlist),
            internal_categories,
            price,
            description,
        });
    }

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
    correct_special_cosmetic_taxonomy(&mut items);
    resolve_internal_item_type_taxonomy(&mut items);
    assign_weapon_mod_metadata(&mut items);
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
        if matches!(item.definition_type_code, Some(0x19c | 0x1a9)) {
            item.types.retain(|kind| {
                !matches!(kind.as_str(), "item_type_trinket" | "category_#A34DFEFD")
            });
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
    // changes. These two opaque registry names are decoded by their unanimous
    // live record sets: weapon cosmetics and profile titles respectively.
    if item
        .internal_categories
        .iter()
        .any(|category| category == "item_type.#A4200795")
    {
        result.category = Some("Weapon Skin".to_owned());
    } else if item
        .internal_categories
        .iter()
        .any(|category| category == "item_type.#A8805E93")
    {
        result.category = Some("Profile".to_owned());
        result.subcategory = Some("Title".to_owned());
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
                result.category = Some("Runner Skin".to_owned());
                result.subcategory = Some(humanize_identifier(runner));
                result.facets.push(format!("runner_{runner}"));
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
            Some(0x122) => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some("Background".to_owned());
            }
            Some(0x123) => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some("Emblem".to_owned());
            }
            Some(0x124) => {
                result.category = Some("Profile".to_owned());
                result.subcategory = Some("Title".to_owned());
            }
            Some(0x128) => {
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
            Some(0x158) => result.category = Some("Trinket".to_owned()),
            Some(0x186..=0x190 | 0x197..=0x19b) => {
                result.category = Some("Sponsored Kit".to_owned())
            }
            Some(0x19c | 0x1a9) => {
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
        result.category = Some("Trinket".to_owned());
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
            result.category = Some("Trinket".to_owned());
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
    // The compact `item_type` category is shared by weapon and melee
    // cosmetics. A concrete investment path/hash is therefore authoritative
    // before group propagation; group ordinals are reused across these two
    // cosmetic domains in the updated tables.
    for item in items.iter_mut().filter(|item| {
        item.definition_type_code
            .is_some_and(is_weapon_or_melee_skin_type)
    }) {
        if direct_melee_skin(item, &generated_melee_hashes) {
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
            let category = if direct_skin_weapon_hash(item, &generated_weapon_owners).is_some() {
                "Weapon Skin"
            } else if direct_melee_skin(item, &generated_melee_hashes) {
                "Melee"
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
            || (item.definition_type_code != Some(0x137) && skin_weapon_hash(item).is_some())
        {
            item.item_type = Some("Weapon Skin".to_owned());
        } else if direct_melee_skin(item, &generated_melee_hashes)
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

/// A small set of shipping cosmetic definitions use a generic reward group
/// instead of their owning cosmetic group. Their internal hashes are stable
/// investment identifiers, so correct the two knife skins before assigning
/// weapon ownership.
fn correct_special_cosmetic_taxonomy(items: &mut [GearItem]) {
    for item in items.iter_mut().filter(|item| {
        matches!(
            item.internal_hash,
            Some(ACHROMATIC_RUSH_MELEE_SKIN_HASH) | Some(VOX_NOCTURNA_MELEE_SKIN_HASH)
        )
    }) {
        item.item_type = Some("Melee".to_owned());
        item.subcategory = Some("Knives".to_owned());
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
        0xf81e_1c03 => Some(KKV_9SD_HASH), // Flechette Drum
        0x774d_a5d3 => Some(weapon_internal_hash(
            "weapons.shotguns.v100.shotgun_mips_02",
        )), // Full-Auto Selector
        0xafc2_33f4 => Some(0x033e_c2d0),  // Gestalt Complex
        0xeb64_31d8 => Some(0xc851_6e72),  // Pike Formation
        0xf1eb_808c => Some(0x0b15_4800),  // Serpentine!
        0x632e_130a => Some(0x3961_29b9),  // Short-Throw Projector
        0x94b4_9ee0 => Some(0xba19_1c63),  // Tilt-Shift LS
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
    // Base-weapon identity is authored by the inventory/cosmetic owner hash,
    // not presentation rarity. Updated definitions reuse packed values that
    // look like Unique/Contraband tiers on several ordinary base weapons.
    let base_weapons = items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        .filter(|item| {
            is_canonical_weapon_skin_owner(item) || item.internal_hash == Some(KKV_9SD_HASH)
        })
        .map(to_compatible_weapon)
        .collect::<Vec<_>>();
    let unique_weapons = items
        .iter()
        .filter(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && item.rarity == Some(GearRarity::Unique)
                && !is_canonical_weapon_skin_owner(item)
                && item.internal_hash != Some(KKV_9SD_HASH)
        })
        .map(to_compatible_weapon)
        .collect::<Vec<_>>();
    let unique_weapon_names = sorted_unique_names(
        unique_weapons
            .iter()
            .map(|weapon| weapon.name.clone())
            .collect(),
    );
    let weapons = base_weapons
        .iter()
        .chain(&unique_weapons)
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
        item.mod_category = Some(category.to_owned());
        item.subcategory = Some(category.to_owned());

        if category == "Chip" {
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
            item.compatible_weapons = weapons
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
                item.compatible_weapons = weapons
                    .iter()
                    .filter(|weapon| weapon.internal_hash == Some(target_hash))
                    .map(|weapon| weapon.name.clone())
                    .collect();
                if !item.compatible_weapons.is_empty() {
                    continue;
                }
            }
        }

        if category == "Stock" {
            item.compatible_weapons = base_weapons
                .iter()
                .filter(|weapon| {
                    matches!(
                        weapon.internal_hash,
                        Some(D54_BATTLE_PISTOL_HASH | KKV_9SD_HASH)
                    )
                })
                .map(|weapon| weapon.name.clone())
                .collect();
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
        // Darksight is the five-family Prestige optic set. These definitions
        // are hash-only, but their authored form names are stable and map one
        // to one to the normal optic families.
        let darksight_family = match item.name.as_str() {
            "Darksight Holo" => Some(("Optic", "pistol")),
            "Darksight Lens" => Some(("Optic", "lmg")),
            "Darksight Optic" => Some(("Optic", "rifle")),
            "Darksight Scope" => Some(("Optic", "sniper")),
            "Darksight Surveyor" => Some(("Optic", "marksman")),
            _ => None,
        };
        if let Some((family_category, compatibility)) =
            direct_family.or(propagated_family).or(darksight_family)
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
                    (item.definition_type_code != Some(0x137))
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
                && item.definition_type_code == Some(0x137)
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
            && item.definition_type_code == Some(0x137)
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
                && item.definition_type_code == Some(0x137)
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
            && item.definition_type_code == Some(0x137)
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
            && item.definition_type_code != Some(0x137)
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
        let Some(archetype) = item.subcategory.as_deref() else {
            continue;
        };
        let Some(model) = item.description.as_deref().and_then(runner_shell_model) else {
            continue;
        };
        candidates
            .entry(archetype.to_owned())
            .or_default()
            .insert(canonical_runner_shell_model(model).to_owned());
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
    matches!(code, 0x133 | 0x137)
}

fn definition_group_key(data: &[u8]) -> Option<u32> {
    // Marathon's July 2026 definition record appended fields to cosmetic
    // definitions. The type code identifies the record layout reliably.
    let footer_offset = if definition_type_code(data) == Some(0x137) {
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

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use itertools::Itertools;

    use super::*;

    #[test]
    fn uses_requested_unique_color() {
        assert_eq!(GearRarity::Unique.color(), Color32::from_rgb(204, 255, 0));
    }

    #[test]
    fn sorts_gear_rarities_highest_first() {
        let mut rarities = [
            None,
            Some(GearRarity::Enhanced),
            Some(GearRarity::Prestige),
            Some(GearRarity::Superior),
            Some(GearRarity::Contraband),
        ];
        rarities.sort_by_key(|rarity| rarity_sort_key(*rarity));
        assert_eq!(
            rarities,
            [
                Some(GearRarity::Contraband),
                Some(GearRarity::Prestige),
                Some(GearRarity::Superior),
                Some(GearRarity::Enhanced),
                None,
            ]
        );
    }

    #[test]
    fn groups_compatible_weapon_mods_by_slot_and_rarity() {
        assert_eq!(weapon_mod_slot_name("Muzzle"), "Barrel");
        assert_eq!(weapon_mod_slot_name("Foregrip"), "Grip");
        assert_eq!(weapon_mod_slot_name("Stock"), "Shield");

        let item = |name: &str, item_type: &str| GearItem {
            display_tag: TagHash(0),
            definition_tag: None,
            model_tag: None,
            definition_group_key: None,
            definition_type_code: None,
            internal_hash: None,
            internal_name: None,
            raw_category_hash: None,
            raw_subcategory_hash: None,
            name: name.to_owned(),
            rarity: None,
            item_type: Some(item_type.to_owned()),
            subcategory: None,
            applies_to: None,
            mod_category: None,
            mod_is_universal: false,
            compatible_weapons: vec![],
            classification: None,
            types: vec![],
            internal_categories: vec![],
            price: None,
            description: None,
        };

        let weapon = item("BRRT SMG", "Weapon");
        let mut standard_optic = item("Accu-Sight", "Weapon Mod");
        standard_optic.rarity = Some(GearRarity::Standard);
        standard_optic.mod_category = Some("Optic".to_owned());
        standard_optic.compatible_weapons = vec![weapon.name.clone()];
        let mut prestige_optic = item("Darksight", "Weapon Mod");
        prestige_optic.rarity = Some(GearRarity::Prestige);
        prestige_optic.mod_category = Some("Optic".to_owned());
        prestige_optic.compatible_weapons = vec![weapon.name.clone()];
        let mut chip = item("Overclock", "Weapon Mod");
        chip.rarity = Some(GearRarity::Enhanced);
        chip.mod_category = Some("Chip".to_owned());
        chip.mod_is_universal = true;
        let mut incompatible = item("Shotgun Drum", "Weapon Mod");
        incompatible.mod_category = Some("Magazine".to_owned());
        incompatible.compatible_weapons = vec!["Misriah 2442".to_owned()];

        let items = [weapon, standard_optic, prestige_optic, chip, incompatible];
        let sections = compatible_mod_sections(&items, &items[0]);
        assert_eq!(sections, vec![("Optic".to_owned(), vec![2, 1])]);
    }

    #[test]
    #[ignore = "requires a local Marathon package installation"]
    fn resolves_marathon_inventory_visual_patterns() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        for (model, expected_owner) in [
            (TagHash(0x80a9b203), "V22 Volt Thrower"),
            (TagHash(0x80aa0c0f), "V22 Volt Thrower"),
            (TagHash(0x80aa0c49), "V22 Volt Thrower"),
            (TagHash(0x80b6db07), "V22 Volt Thrower"),
            (TagHash(0x80b6e64a), "KKV-9SD"),
            (TagHash(0x80b6e68c), "KKV-9SD"),
            (TagHash(0x80b6e6d2), "KKV-9SD"),
            (TagHash(0x80b6e714), "KKV-9SD"),
            (TagHash(0x80b6e7d4), "V22 Volt Thrower"),
        ] {
            assert!(
                view.items.iter().any(|item| {
                    item.item_type.as_deref() == Some("Weapon Skin")
                        && item.model_tag == Some(model)
                        && item.applies_to.as_deref() == Some(expected_owner)
                }),
                "load-time Gear owner is wrong for {model}"
            );
        }
        for (name, expected) in [
            ("Precision Choke", TagHash(0x80a60313)),
            ("Slick Mag III", TagHash(0x80a6068a)),
            ("Snapshot Grip", TagHash(0x80a600ac)),
            ("Flechette Split Action", TagHash(0x80a61398)),
            ("Darksight Optic", TagHash(0x80a61cf0)),
        ] {
            assert!(
                view.items
                    .iter()
                    .any(|item| item.name == name && item.model_tag == Some(expected)),
                "{name} did not resolve to {expected}"
            );
        }

        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let v11_probe = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "V11 Punch")
            .expect("V11 probe catalog");
        let v11_probe_mods = v11_probe
            .slots
            .iter()
            .flat_map(|slot| slot.mods.iter())
            .filter(|modification| {
                matches!(
                    modification.name.as_str(),
                    "Rangefinder Lens" | "Suppression Dampener"
                )
            })
            .map(|modification| modification.model_tag)
            .unique()
            .collect::<Vec<_>>();
        let v11_probe_owner = crate::geometry::WeaponModSocketIndex::new()
            .owner_for(&cache, v11_probe.owner_tag, &v11_probe_mods)
            .expect("V11 probe socket owner");
        assert_eq!(v11_probe_owner, TagHash(0x80A7C7B5));
        for tag in [
            TagHash(0x80b6e64a),
            TagHash(0x80b6e68c),
            TagHash(0x80b6e6d2),
            TagHash(0x80b6e714),
        ] {
            let owners = catalog
                .weapons
                .iter()
                .filter(|weapon| weapon.model_tags.contains(&tag))
                .map(|weapon| weapon.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(owners, vec!["KKV-9SD"], "wrong owner for {tag}");
        }
        for tag in [
            TagHash(0x80a9b203),
            TagHash(0x80aa0c0f),
            TagHash(0x80aa0c49),
            TagHash(0x80b6db07),
            TagHash(0x80b6e7d4),
        ] {
            let owners = catalog
                .weapons
                .iter()
                .filter(|weapon| weapon.model_tags.contains(&tag))
                .map(|weapon| weapon.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(owners, vec!["V22 Volt Thrower"], "wrong owner for {tag}");
        }
        for skin in view.items.iter().filter(|item| {
            item.item_type.as_deref() == Some("Weapon Skin")
                && matches!(
                    item.applies_to.as_deref(),
                    Some("KKV-9SD" | "V22 Volt Thrower")
                )
        }) {
            let model = skin.model_tag.expect("KKV/V22 skin model");
            let owners = catalog
                .weapons
                .iter()
                .filter(|weapon| weapon.model_tags.contains(&model))
                .map(|weapon| weapon.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                owners,
                vec![skin.applies_to.as_deref().expect("KKV/V22 skin owner")],
                "Gear and Models disagree for {} ({model})",
                skin.name
            );
        }
        let tac_standard = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "KKV-9SD")
            .and_then(|weapon| {
                weapon
                    .skins
                    .iter()
                    .find(|skin| skin.model_tag == TagHash(0x80b6e64a))
            })
            .expect("KKV-9SD TAC Standard model identity");
        assert_eq!(tac_standard.name, "TAC Standard");
        assert_eq!(tac_standard.color, GearRarity::Standard.color());
        let screenshot_skin = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "V75 SCAR")
            .and_then(|weapon| {
                weapon
                    .skins
                    .iter()
                    .find(|skin| skin.model_tag == TagHash(0x80b6c555))
            })
            .expect("screenshot model identity");
        assert_eq!(screenshot_skin.name, "Aqua Stellar");
        let skin_models = view
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Skin"))
            .filter_map(|item| item.model_tag)
            .collect::<FxHashSet<_>>();
        let mut catalog_owner_by_model = FxHashMap::default();
        for weapon in &catalog.weapons {
            for model in weapon
                .model_tags
                .iter()
                .filter(|model| skin_models.contains(model))
            {
                if let Some(previous) = catalog_owner_by_model.insert(*model, weapon.name.as_str())
                {
                    assert_eq!(
                        previous, weapon.name,
                        "model {model} appears under multiple weapons"
                    );
                }
            }
        }
        let kkv_subcategory = view
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon") && item.name == "KKV-9SD")
            .find_map(|item| item.subcategory.as_deref());
        for (name, tag) in [
            ("Anti-Caution", TagHash(0x80b6e68c)),
            ("Vox Nocturna", TagHash(0x80b6e6d2)),
            ("Shadow Index", TagHash(0x80b6e714)),
        ] {
            assert!(view.items.iter().any(|item| {
                item.item_type.as_deref() == Some("Weapon Skin")
                    && item.name == name
                    && item.model_tag == Some(tag)
                    && item.applies_to.as_deref() == Some("KKV-9SD")
                    && item.subcategory.as_deref() == kkv_subcategory
            }));
        }
        for (name, tag) in [
            ("Finish Line", TagHash(0x80aa0c0f)),
            ("Pleasure Mint", TagHash(0x80aa0c49)),
        ] {
            assert!(view.items.iter().any(|item| {
                item.item_type.as_deref() == Some("Weapon Skin")
                    && item.name == name
                    && item.model_tag == Some(tag)
                    && item.applies_to.as_deref() == Some("V22 Volt Thrower")
            }));
        }
        assert!(view.items.iter().any(|item| {
            item.name == "Retro_Remix"
                && item.model_tag == Some(TagHash(0x80b6db07))
                && item.applies_to.as_deref() == Some("V22 Volt Thrower")
        }));
        assert!(view.items.iter().any(|item| {
            item.internal_hash == Some(BIOTOXIC_SHADOW_INDEX_SKIN_HASH)
                && item.name == "Shadow Index"
                && item.item_type.as_deref() == Some("Weapon Skin")
                && item.applies_to.as_deref() == Some("Biotoxic Disinjector")
                && item.model_tag.is_some()
        }));
        let weapon_skins = view
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Skin"))
            .collect::<Vec<_>>();
        let unresolved_skins = weapon_skins
            .iter()
            .filter(|item| item.model_tag.is_none())
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>();
        // This installed build authors Crash Casket with a null FileHash; do
        // not silently substitute V75 SCAR's base Pattern for a missing skin.
        assert_eq!(unresolved_skins, vec!["Crash Casket"]);
        let unmatched_skins = weapon_skins
            .iter()
            .filter_map(|skin| {
                let owner = skin.applies_to.as_deref()?;
                if owner == "Certification test (no weapon)" {
                    return None;
                }
                let pattern = skin.model_tag?;
                (!catalog
                    .weapons
                    .iter()
                    .find(|weapon| weapon.name == owner)
                    .is_some_and(|weapon| weapon.model_tags.contains(&pattern))
                    || !package_manager()
                        .get_entry(pattern)
                        .is_some_and(|entry| entry.reference == MODEL_PATTERN_REFERENCE))
                .then_some((skin.name.as_str(), owner, pattern))
            })
            .collect::<Vec<_>>();
        assert!(
            unmatched_skins.is_empty(),
            "resolved skins missing from model catalog: {unmatched_skins:?}"
        );
        assert!(view.items.iter().any(|item| {
            item.name == "Yōkai's Claw"
                && item.applies_to.as_deref() == Some("Misriah 2442")
                && item.model_tag == Some(TagHash(0x80b6cc5d))
        }));
        let misriah = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "Misriah 2442")
            .expect("Misriah model catalog");
        assert!(!misriah.slots.iter().any(|slot| slot.name == "Chip"));
        let misriah_mods = [
            TagHash(0x80a60313),
            TagHash(0x80a6068a),
            TagHash(0x80a600ac),
        ];
        let socket_index = crate::geometry::WeaponModSocketIndex::new();
        for (target, expected_slots) in [
            ("V99 Channel Rifle", &["Optic", "Barrel", "Magazine"][..]),
            ("V00 ZEUS RG", &["Generator", "Magazine"]),
            ("V85 Circuit Breaker", &["Grip", "Magazine"]),
        ] {
            let weapon = catalog
                .weapons
                .iter()
                .find(|weapon| weapon.name == target)
                .expect("Volt Cell weapon catalog");
            assert_eq!(
                weapon
                    .slots
                    .iter()
                    .map(|slot| slot.name.as_str())
                    .collect::<Vec<_>>(),
                expected_slots
            );
            let representative_mods = weapon
                .slots
                .iter()
                .filter_map(|slot| slot.mods.first().map(|item| item.model_tag))
                .collect::<Vec<_>>();
            let owner = socket_index
                .owner_for(&cache, weapon.owner_tag, &representative_mods)
                .expect("Volt Cell socket owner");
            let distinct_families = representative_mods
                .iter()
                .filter_map(|modification| {
                    crate::geometry::weapon_mod_attachment_pose(&cache, owner, *modification)
                        .map(|pose| pose.family_id)
                })
                .collect::<FxHashSet<_>>();
            assert_eq!(
                distinct_families.len(),
                expected_slots.len(),
                "{target} mod slots must map to distinct authored attachment families"
            );
        }
        let misriah_owner = socket_index
            .owner_for(&cache, misriah.owner_tag, &misriah_mods)
            .expect("Misriah socket owner");
        for (modification, expected_translation) in misriah_mods.into_iter().zip([
            [0.52950084, 0.0, 0.09353443],
            [0.20186345, -1.2300719e-5, 0.1453178],
            [0.33317506, -1.0606527e-8, 0.1091585],
        ]) {
            let pose =
                crate::geometry::weapon_mod_attachment_pose(&cache, misriah_owner, modification)
                    .unwrap_or_else(|| {
                        panic!("Misriah owner {misriah_owner} cannot place {modification}")
                    });
            for axis in 0..3 {
                assert!(
                    (pose.translation[axis] - expected_translation[axis]).abs() < 0.000001,
                    "Misriah {modification} axis {axis}: {:?}",
                    pose.translation
                );
            }
        }

        let brrt = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "BRRT SMG")
            .expect("BRRT model catalog");
        let brrt_mods = [TagHash(0x80a61398), TagHash(0x80a61cf0)];
        let brrt_owner = socket_index
            .owner_for(&cache, brrt.owner_tag, &brrt_mods)
            .expect("BRRT socket owner");
        assert_eq!(brrt_owner, TagHash(0x80a7d43d));
        for modification in brrt_mods {
            assert!(
                crate::geometry::weapon_mod_attachment_pose(&cache, brrt_owner, modification)
                    .is_some(),
                "BRRT owner cannot place {modification}"
            );
        }

        let simulation_failures = catalog
            .weapons
            .iter()
            .filter(|weapon| !weapon.slots.is_empty())
            .filter_map(|weapon| {
                let modifications = weapon
                    .slots
                    .iter()
                    .flat_map(|slot| {
                        slot.mods
                            .iter()
                            .map(|modification| (modification.model_tag, modification.name.clone()))
                    })
                    .unique_by(|(tag, _name)| *tag)
                    .collect::<Vec<_>>();
                let modification_tags = modifications
                    .iter()
                    .map(|(tag, _name)| *tag)
                    .collect::<Vec<_>>();
                let Some(owner) =
                    socket_index.owner_for(&cache, weapon.owner_tag, &modification_tags)
                else {
                    return Some((weapon.name.clone(), None, modifications));
                };
                let unplaced = modifications
                    .into_iter()
                    .filter(|(modification, _name)| {
                        crate::geometry::weapon_mod_attachment_pose(&cache, owner, *modification)
                            .is_none()
                    })
                    .collect::<Vec<_>>();
                (!unplaced.is_empty()).then_some((weapon.name.clone(), Some(owner), unplaced))
            })
            .collect::<Vec<_>>();
        assert!(
            simulation_failures.is_empty(),
            "weapon/mod simulation coverage failures: {simulation_failures:?}"
        );

        let visual_mods = catalog
            .weapons
            .iter()
            .flat_map(|weapon| weapon.slots.iter())
            .flat_map(|slot| slot.mods.iter())
            .map(|modification| (modification.model_tag, modification.name.as_str()))
            .unique_by(|(tag, _name)| *tag)
            .collect::<Vec<_>>();
        let visual_mod_count = visual_mods.len();
        let mut authored_condition_models = 0usize;
        let condition_failures = visual_mods
            .into_iter()
            .filter_map(|(model, name)| {
                let Some(entry) = package_manager().get_entry(model) else {
                    return Some((name.to_owned(), model, "missing model entry".to_owned()));
                };
                let Ok(data) = package_manager().read_tag(model) else {
                    return Some((name.to_owned(), model, "model read failed".to_owned()));
                };
                let tag_type = quicktag_core::tagtypes::TagType::from_type_subtype(
                    entry.file_type,
                    entry.file_subtype,
                );
                let Some(preview) = crate::geometry::GeometryTagPreview::load(
                    cache.clone(),
                    model,
                    &entry,
                    tag_type,
                    &data,
                ) else {
                    return Some((name.to_owned(), model, "model parse failed".to_owned()));
                };
                let crate::geometry::GeometryPreviewKind::Model(model_preview) = preview.kind
                else {
                    return Some((name.to_owned(), model, "not a model".to_owned()));
                };
                let Some(mesh) = model_preview.wireframe.as_ref() else {
                    // A small set of catalog entries (for example Alert
                    // Extender) resolve through non-renderable placeholders.
                    return None;
                };
                let age_ranges = mesh
                    .material_ranges
                    .iter()
                    .filter(|range| {
                        let Some(technique) = range.technique else {
                            return false;
                        };
                        let Some(entry) = package_manager().get_entry(technique) else {
                            return false;
                        };
                        let Ok(data) = package_manager().read_tag(technique) else {
                            return false;
                        };
                        let Some(preview) =
                            crate::material::MaterialTagPreview::load(&entry, &data)
                        else {
                            return false;
                        };
                        let crate::material::MaterialPreviewKind::Technique(preview) = preview.kind;
                        preview.stages.iter().any(|stage| {
                            stage.stage == "PS"
                                && stage.bytecode.expressions.iter().any(|expression| {
                                    expression.expression.contains("object_channel(0x138DE801)")
                                })
                        })
                    })
                    .collect::<Vec<_>>();
                if age_ranges.is_empty() {
                    // The engine does not route every electronic/display
                    // material through the common aged-surface shader.
                    return None;
                }
                authored_condition_models += 1;
                let malformed = age_ranges
                    .into_iter()
                    .filter_map(|range| {
                        let valid = range.textures.mod_wear.is_some_and(|wear| {
                            wear.condition_controls
                                == [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
                                && wear.condition_blend == 0.5
                        });
                        (!valid).then_some(range.technique)
                    })
                    .collect::<Vec<_>>();
                (!malformed.is_empty()).then_some((
                    name.to_owned(),
                    model,
                    format!("malformed age techniques {malformed:?}"),
                ))
            })
            .collect::<Vec<_>>();
        assert!(
            condition_failures.is_empty(),
            "visual weapon mods missing authored three-tier condition data: {condition_failures:?}"
        );
        assert!(
            authored_condition_models > 0,
            "installed catalog exposed no authored weapon-mod condition materials"
        );
        eprintln!(
            "validated {authored_condition_models}/{visual_mod_count} visual mod models that use the engine's common aged-surface shader"
        );
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_updated_pattern_global_layout() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        let mut wordlist = FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            wordlist.entry(hash).or_insert_with(|| word.to_owned());
        });
        let resolver = InvestmentPatternResolver::load();
        for (tag, _) in package_manager().get_all_by_reference(PATTERN_GLOBAL_TABLE_REFERENCE) {
            let data = package_manager().read_tag(tag).expect("pattern globals");
            for stride in [0x38, 0x48] {
                let range = table_range(&data, 0x8, stride).expect("known layout");
                let records = data[range].chunks_exact(stride).collect_vec();
                let assignment_hits = records
                    .iter()
                    .filter_map(|record| read_u32(record, 0x8))
                    .filter(|global| resolver.assignments.contains_key(global))
                    .count();
                let path_hits = records
                    .iter()
                    .filter_map(|record| read_u32(record, 0))
                    .filter_map(|hash| wordlist.get(&hash))
                    .filter(|word| {
                        word.starts_with("weapons.")
                            || word.starts_with("weapon_mods.")
                            || word.starts_with("patterns.")
                    })
                    .count();
                eprintln!(
                    "PATTERN_GLOBAL_SCORE tag={tag} stride={stride:X} assignments={assignment_hits} paths={path_hits} records={}",
                    records.len(),
                );
            }
            for stride in (0x30..=0x90).step_by(4) {
                let Some(range) = table_range(&data, 0x8, stride) else {
                    continue;
                };
                let records = data[range].chunks_exact(stride).collect_vec();
                for offset in (0..stride).step_by(4) {
                    let matches = records
                        .iter()
                        .filter_map(|record| read_u32(record, offset))
                        .filter_map(|hash| wordlist.get(&hash))
                        .filter(|word| {
                            word.starts_with("weapons.")
                                || word.starts_with("weapon_mods.")
                                || word.starts_with("patterns.")
                        })
                        .count();
                    if matches >= 10 {
                        let examples = records
                            .iter()
                            .filter_map(|record| read_u32(record, offset))
                            .filter_map(|hash| wordlist.get(&hash))
                            .filter(|word| {
                                word.starts_with("weapons.")
                                    || word.starts_with("weapon_mods.")
                                    || word.starts_with("patterns.")
                            })
                            .take(4)
                            .cloned()
                            .collect_vec();
                        eprintln!(
                            "PATTERN_GLOBAL_LAYOUT tag={tag} stride={stride:X} offset={offset:X} matches={matches} records={} examples={examples:?}",
                            records.len(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_mod_visual_binding_records() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        for item in view.items.iter().filter(|item| {
            item.item_type.as_deref() == Some("Weapon Mod")
                && [
                    "Precision Choke",
                    "Darksight Optic",
                    "Full-Auto Selector",
                    "Slick Mag II",
                    "Extra Mag III",
                    "Suppression Dampener",
                ]
                .contains(&item.name.as_str())
        }) {
            let Some(model) = item.model_tag else {
                continue;
            };
            eprintln!(
                "MOD_BINDING name={} rarity={:?} category={:?} model={} definition={:?} internal={:?} group={:?} type={:?} categories={:?} types={:?} words={:X?}",
                item.name,
                item.rarity,
                item.mod_category,
                model,
                item.definition_tag,
                item.internal_name,
                item.definition_group_key,
                item.definition_type_code,
                item.internal_categories,
                item.types,
                crate::geometry::debug_weapon_mod_visual_binding_words(&cache, model),
            );
            if let Some(definition) = item.definition_tag
                && let Some(scan) = cache.hashes.get(&definition)
            {
                let children = scan
                    .file_hashes
                    .iter()
                    .map(|reference| reference.hash)
                    .chain(scan.references.iter().copied())
                    .unique()
                    .sorted()
                    .filter_map(|tag| Some((tag, package_manager().get_entry(tag)?.reference)))
                    .collect_vec();
                eprintln!(
                    "MOD_DEFINITION_GRAPH name={} children={children:X?}",
                    item.name
                );
            }
        }
        for name in [
            "BRRT SMG",
            "Bully SMG",
            "Demolition HMG",
            "Longshot",
            "M77 Assault Rifle",
            "Stryder M1T",
            "V00 ZEUS RG",
            "WSTR Combat Shotgun",
        ] {
            let compatible = view
                .items
                .iter()
                .filter(|item| {
                    item.item_type.as_deref() == Some("Weapon Mod")
                        && item.compatible_weapons.iter().any(|weapon| weapon == name)
                })
                .map(|item| {
                    (
                        item.mod_category.as_deref().unwrap_or("?"),
                        item.name.as_str(),
                        item.internal_name.as_deref().unwrap_or("?"),
                    )
                })
                .collect_vec();
            eprintln!("WEAPON_COMPATIBILITY name={name} mods={compatible:?}");
        }
        for item in view.items.iter().filter(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && [
                    "Misriah 2442",
                    "BRRT SMG",
                    "Bully SMG",
                    "Longshot",
                    "V11 Punch",
                    "V22 Volt Thrower",
                ]
                .contains(&item.name.as_str())
        }) {
            eprintln!(
                "WEAPON_BINDING name={} model={:?} definition={:?} internal={:?} group={:?} type={:?} categories={:?} types={:?}",
                item.name,
                item.model_tag,
                item.definition_tag,
                item.internal_name,
                item.definition_group_key,
                item.definition_type_code,
                item.internal_categories,
                item.types,
            );
            let compatibility_probe = CompatibleWeapon {
                name: item.name.clone(),
                internal_name: item.internal_name.clone(),
                internal_hash: item.internal_hash,
                subcategory: item
                    .subcategory
                    .clone()
                    .or_else(|| weapon_subcategory(item))
                    .or_else(|| weapon_subcategory_from_name(&item.name)),
            };
            eprintln!(
                "WEAPON_MATCH_PROBE name={} rarity={:?} subcategory={:?} optic_rifle={} muzzle_base={} magazine_rifle={}",
                item.name,
                item.rarity,
                compatibility_probe.subcategory,
                base_weapon_matches_mod_family(&compatibility_probe, "Optic", "rifle"),
                base_weapon_matches_mod_family(&compatibility_probe, "Muzzle", "base"),
                base_weapon_matches_mod_family(&compatibility_probe, "Magazine", "rifle"),
            );
            if let Some(definition) = item.definition_tag
                && let Some(scan) = cache.hashes.get(&definition)
            {
                let children = scan
                    .file_hashes
                    .iter()
                    .map(|reference| reference.hash)
                    .chain(scan.references.iter().copied())
                    .unique()
                    .sorted()
                    .filter_map(|tag| Some((tag, package_manager().get_entry(tag)?.reference)))
                    .collect_vec();
                eprintln!(
                    "WEAPON_DEFINITION_GRAPH name={} children={children:X?}",
                    item.name
                );
            }
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_engine_authored_weapon_mod_joins() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);

        let visual_mods = view
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon Mod")
                    && item.mod_category.as_deref() != Some("Chip")
                    && item.model_tag.is_some()
            })
            .collect_vec();
        for weapon in view
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .filter(|item| item.model_tag.is_some())
            .sorted_by_key(|item| item.name.to_lowercase())
        {
            let owner = weapon.model_tag.unwrap();
            let matches = visual_mods
                .iter()
                .filter(|item| {
                    crate::geometry::weapon_mod_attachment_pose(
                        &cache,
                        owner,
                        item.model_tag.unwrap(),
                    )
                    .is_some()
                })
                .map(|item| {
                    (
                        weapon_mod_slot_name(item.mod_category.as_deref().unwrap_or("Other")),
                        item.model_tag.unwrap(),
                    )
                })
                .unique()
                .collect_vec();
            let slots = matches
                .iter()
                .map(|(slot, _tag)| *slot)
                .unique()
                .sorted_by_key(|slot| mod_slot_sort_key(slot))
                .collect_vec();
            let defaults = crate::geometry::weapon_default_mod_patterns(&cache, owner, owner);
            eprintln!(
                "ENGINE_MOD_JOIN name={} owner={} slots={slots:?} mods={} defaults={}",
                weapon.name,
                owner,
                matches.len(),
                defaults.len(),
            );
        }
    }

    #[test]
    #[ignore = "probe: requires installed Marathon packages"]
    fn probes_weapon_default_visual_owners() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let socket_index = crate::geometry::WeaponModSocketIndex::new();
        let candidates = [
            TagHash(0x80A61D82),
            TagHash(0x80A9A582),
            TagHash(0x80A61E8F),
        ];
        for name in [
            "Bully SMG",
            "Misriah 2442",
            "Longshot",
            "Outland",
            "V99 Channel Rifle",
        ] {
            let weapon = catalog
                .weapons
                .iter()
                .find(|weapon| weapon.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let mods = weapon
                .slots
                .iter()
                .flat_map(|slot| slot.mods.iter().map(|item| item.model_tag))
                .unique()
                .collect::<Vec<_>>();
            let owner = socket_index.owner_for(&cache, weapon.owner_tag, &mods);
            eprintln!(
                "DEFAULT_WEAPON name={name} owner={} socket={owner:?} models={:?} slots={:?}",
                weapon.owner_tag,
                weapon.model_tags,
                weapon
                    .slots
                    .iter()
                    .map(|slot| {
                        (
                            &slot.name,
                            slot.mods.first().map(|item| item.model_tag.to_string()),
                        )
                    })
                    .collect::<Vec<_>>()
            );
            if let Some(owner) = owner {
                eprintln!(
                    "DEFAULT_CATALOG name={name} owner={} socket={} defaults={:?}",
                    weapon.owner_tag,
                    owner,
                    crate::geometry::weapon_default_mod_patterns(&cache, weapon.owner_tag, owner,)
                );
                for candidate in candidates {
                    if let Some(pose) =
                        crate::geometry::weapon_mod_attachment_pose(&cache, owner, candidate)
                    {
                        eprintln!("  DEFAULT_CANDIDATE candidate={candidate} pose={pose:?}");
                    }
                }
            }
        }
        for weapon in &catalog.weapons {
            let mods = weapon
                .slots
                .iter()
                .flat_map(|slot| slot.mods.iter().map(|item| item.model_tag))
                .unique()
                .collect::<Vec<_>>();
            let owner = socket_index.owner_for(&cache, weapon.owner_tag, &mods);
            eprintln!(
                "MOD_RUNTIME_CATALOG name={} base={} models={} slots={} mods={} socket={owner:?}",
                weapon.name,
                weapon.owner_tag,
                weapon.model_tags.len(),
                weapon.slots.len(),
                mods.len(),
            );
            let Some(owner) = owner else {
                continue;
            };
            let defaults =
                crate::geometry::weapon_default_mod_patterns(&cache, weapon.owner_tag, owner);
            if !defaults.is_empty() {
                eprintln!(
                    "DEFAULT_ALL name={} owner={} socket={} defaults={:?}",
                    weapon.name, weapon.owner_tag, owner, defaults
                );
            }
        }
    }

    #[test]
    fn collects_sorted_item_type_counts() {
        let item = |name: &str, item_type: Option<&str>| GearItem {
            display_tag: TagHash(0),
            definition_tag: None,
            model_tag: None,
            definition_group_key: None,
            definition_type_code: None,
            internal_hash: None,
            internal_name: None,
            raw_category_hash: None,
            raw_subcategory_hash: None,
            name: name.to_owned(),
            rarity: None,
            item_type: item_type.map(str::to_owned),
            subcategory: None,
            applies_to: None,
            mod_category: None,
            mod_is_universal: false,
            compatible_weapons: vec![],
            classification: None,
            types: vec![],
            internal_categories: vec![],
            price: None,
            description: None,
        };
        let mut items = [
            item("First", Some("Weapon")),
            item("Second", Some("Key")),
            item("Third", Some("Weapon")),
            item("Fourth", None),
        ];
        items[0].internal_categories = vec![
            "behaviors.durable_item".to_owned(),
            "loot_table_behaviors.always_excluded".to_owned(),
        ];
        items[1].internal_categories = vec!["behaviors.durable_item".to_owned()];
        items[2].internal_categories = vec!["loot_table_behaviors.always_excluded".to_owned()];

        assert_eq!(
            collect_item_types(&items),
            vec![("Key".to_owned(), 1), ("Weapon".to_owned(), 2)]
        );
        assert_eq!(
            collect_internal_category_counts(&items),
            vec![
                ("behaviors.durable_item".to_owned(), 2),
                ("loot_table_behaviors.always_excluded".to_owned(), 2),
            ]
        );
        let selected_categories = FxHashSet::from_iter([
            "behaviors.durable_item".to_owned(),
            "loot_table_behaviors.always_excluded".to_owned(),
        ]);
        assert!(matches_internal_category_filters(
            &items[0],
            &selected_categories
        ));
        assert!(matches_internal_category_filters(
            &items[1],
            &selected_categories
        ));
        assert!(!matches_internal_category_filters(
            &items[3],
            &selected_categories
        ));
        assert!(matches_internal_category_filters(
            &items[3],
            &FxHashSet::default()
        ));
        items[0].rarity = Some(GearRarity::Standard);
        items[1].rarity = Some(GearRarity::Superior);
        items[2].rarity = Some(GearRarity::Enhanced);
        let selected_item_types = FxHashSet::from_iter(["Weapon".to_owned(), "Key".to_owned()]);
        let selected_rarities = FxHashSet::from_iter([GearRarity::Standard, GearRarity::Superior]);
        assert!(matches_chip_filters(
            &items[0],
            &selected_item_types,
            &selected_rarities,
            &selected_categories,
        ));
        assert!(matches_chip_filters(
            &items[1],
            &selected_item_types,
            &selected_rarities,
            &selected_categories,
        ));
        assert!(!matches_chip_filters(
            &items[2],
            &selected_item_types,
            &selected_rarities,
            &selected_categories,
        ));
        assert!(!matches_chip_filters(
            &items[3],
            &selected_item_types,
            &selected_rarities,
            &selected_categories,
        ));
        assert_eq!(
            gear_section_label("Weapon", &selected_categories),
            "Weapon (behaviors.durable_item, loot_table_behaviors.always_excluded)"
        );
        assert_eq!(
            gear_section_label("Weapon", &FxHashSet::default()),
            "Weapon"
        );

        let mut weapon_mod = item("Test Mod", Some("Weapon Mod"));
        weapon_mod.compatible_weapons =
            vec!["Impact H-AR".to_owned(), "M77 Assault Rifle".to_owned()];
        assert_eq!(
            gear_sections(&weapon_mod, true),
            vec!["Impact H-AR".to_owned(), "M77 Assault Rifle".to_owned()]
        );
        assert_eq!(
            gear_sections(&weapon_mod, false),
            vec![
                "Weapon Mod · Impact H-AR".to_owned(),
                "Weapon Mod · M77 Assault Rifle".to_owned(),
            ]
        );

        weapon_mod.mod_is_universal = true;
        assert_eq!(gear_sections(&weapon_mod, true), vec!["Universal"]);

        let mut grouped_skin = item("Opaque skin", Some("Weapon Skin"));
        grouped_skin.internal_hash = Some(0x1234_5678);
        for (group, expected_owner) in [
            (
                49_u32,
                weapon_internal_hash("weapons.submachineguns.v100.smg_light_02"),
            ),
            (
                51,
                weapon_internal_hash("weapons.submachineguns.v100.smg_heavy_01"),
            ),
        ] {
            grouped_skin.definition_group_key = Some((group << 16) | 6);
            assert_eq!(
                skin_weapon_hash(&grouped_skin),
                Some(expected_owner),
                "cosmetic group {group} maps to the wrong weapon"
            );
        }

        let mut biotoxic_shadow_index = item("Shadow Index", None);
        biotoxic_shadow_index.internal_hash = Some(BIOTOXIC_SHADOW_INDEX_SKIN_HASH);
        biotoxic_shadow_index.definition_type_code = Some(0x133);
        biotoxic_shadow_index.definition_group_key = Some(53_u32 << 16);
        classify_unseeded_weapon_skins(std::slice::from_mut(&mut biotoxic_shadow_index));
        assert_eq!(
            biotoxic_shadow_index.item_type.as_deref(),
            Some("Weapon Skin")
        );
        assert_eq!(
            skin_weapon_hash(&biotoxic_shadow_index),
            Some(BIOTOXIC_DISINJECTOR_HASH)
        );

        let mut runner_core = item("Adrenal Core", Some("Runner Core"));
        runner_core.internal_name = Some("implant_cores.agile.v100.core02".to_owned());
        runner_core.applies_to = runner_core_archetype(
            runner_core.internal_name.as_deref(),
            runner_core.internal_hash,
        );
        assert_eq!(runner_core.applies_to.as_deref(), Some("Agile"));
        assert_eq!(gear_sections(&runner_core, true), vec!["Agile"]);

        runner_core.internal_name = Some("#1AE20D50".to_owned());
        runner_core.internal_hash = Some(SENTINEL_CORE_HASHES[0]);
        runner_core.applies_to = runner_core_archetype(
            runner_core.internal_name.as_deref(),
            runner_core.internal_hash,
        );
        assert_eq!(runner_core.applies_to.as_deref(), Some("Sentinel"));
    }

    #[test]
    fn formats_price() {
        assert_eq!(format_number(3_200), "3,200");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }

    #[test]
    fn parses_fixed_rarity_tiers() {
        for (value, rarity) in [
            (0xffff0000_u32, GearRarity::Standard),
            (0xffff0001, GearRarity::Enhanced),
            (0xffff0002, GearRarity::Deluxe),
            (0xffff0003, GearRarity::Superior),
            (0xffff0004, GearRarity::Prestige),
            (0xffff0005, GearRarity::Contraband),
            (0xffff0006, GearRarity::Quest),
        ] {
            let mut data = vec![0; 12];
            data[4..8].copy_from_slice(&RARITY_MARKER.to_le_bytes());
            data[8..12].copy_from_slice(&value.to_le_bytes());
            assert_eq!(parse_rarity(&data), Some(rarity));
        }
    }

    #[test]
    fn parses_dynamic_and_unique_rarities() {
        let mut dynamic = vec![0; 12];
        dynamic[4..8].copy_from_slice(&RARITY_MARKER.to_le_bytes());
        dynamic[8..12].copy_from_slice(&0x005d0004_u32.to_le_bytes());
        assert_eq!(parse_rarity(&dynamic), Some(GearRarity::Dynamic));
        assert_eq!(
            parse_rarity(&RARITY_UNIQUE_HASH.to_le_bytes()),
            Some(GearRarity::Unique)
        );
        assert_eq!(parse_rarity(&RARITY_UNIQUE_VALUE.to_le_bytes()), None);
        assert_eq!(parse_rarity(&RARITY_CONTRABAND_VALUE.to_le_bytes()), None);
        for (category, rarity) in [
            ("rarity_tier.dynamic", GearRarity::Standard),
            ("rarity_tier.grey", GearRarity::Standard),
            ("rarity_tier.green", GearRarity::Enhanced),
            ("rarity_tier.blue", GearRarity::Deluxe),
            ("rarity_tier.purple", GearRarity::Superior),
            ("rarity_tier.gold", GearRarity::Prestige),
            ("rarity_tier.contraband", GearRarity::Contraband),
            ("rarity_tier.quest", GearRarity::Quest),
            ("rarity_tier.unique", GearRarity::Unique),
        ] {
            assert_eq!(
                parse_rarity_from_categories(&[category.to_owned()]),
                Some(rarity)
            );
        }
    }

    #[test]
    fn parses_cosmetic_footer_rarities() {
        for (code, rarity) in [
            (0x031c_u16, GearRarity::Deluxe),
            (0x031d, GearRarity::Contraband),
            (0x031f, GearRarity::Prestige),
            (0x0320, GearRarity::Enhanced),
            (0x0321, GearRarity::Standard),
            (0x0322, GearRarity::Superior),
            (0x0323, GearRarity::Quest),
            (0x0324, GearRarity::Unique),
            (0x032d, GearRarity::Deluxe),
            (0x032e, GearRarity::Contraband),
            (0x0330, GearRarity::Prestige),
            (0x0331, GearRarity::Enhanced),
            (0x0332, GearRarity::Standard),
            (0x0333, GearRarity::Superior),
            (0x0334, GearRarity::Quest),
            (0x0335, GearRarity::Unique),
        ] {
            let footer = ((code as u32) << 16) | 0x02b3;
            assert_eq!(parse_rarity(&footer.to_le_bytes()), Some(rarity));
        }
    }

    #[test]
    fn parses_price_formula() {
        let mut data = vec![0; 32];
        data[4..8].copy_from_slice(&PRICE_MARKER.to_le_bytes());
        data[12..16].copy_from_slice(&40_000.0_f32.to_bits().to_le_bytes());
        data[24..28].copy_from_slice(&0.08_f32.to_bits().to_le_bytes());
        assert_eq!(parse_price(&data), Some(3_200));

        data[12..16].copy_from_slice(&77_777.0_f32.to_bits().to_le_bytes());
        data[24..28].copy_from_slice(&0.1_f32.to_bits().to_le_bytes());
        assert_eq!(parse_price(&data), Some(7_777));
    }

    #[test]
    fn humanizes_structural_item_types() {
        assert_eq!(humanize_identifier("weapon"), "Weapon");
        assert_eq!(humanize_identifier("weapon_mod"), "Weapon Mod");
    }

    #[test]
    fn classifies_weapon_mod_categories_from_paths_and_markers() {
        assert_eq!(
            weapon_mod_category_from_path("weapon_mods.optics.rifle.v100.short0"),
            Some("Optic")
        );
        assert_eq!(
            weapon_mod_category_from_path("weapon_mods.magazines.payload.v100.capacity0"),
            Some("Magazine")
        );
        assert_eq!(
            exact_weapon_archetype_from_mod_path(
                "weapon_mods.magazines.rifle.v100.gold_heavy_ar_stopping"
            ),
            Some("auto_heavy_01")
        );

        for (marker, expected) in [
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
        ] {
            assert_eq!(
                weapon_mod_category_from_definition(&marker.to_le_bytes()),
                Some(expected)
            );
        }
    }

    #[test]
    fn matches_authored_weapon_mod_families() {
        let weapon = |subcategory: &str, internal_name: &str| CompatibleWeapon {
            name: "Test".to_owned(),
            internal_name: Some(internal_name.to_owned()),
            internal_hash: None,
            subcategory: Some(subcategory.to_owned()),
        };
        let v75 = weapon("Assault Rifle", "weapons.auto_rifles.v100.auto_battery_01");
        let v22 = weapon(
            "Submachine Gun",
            "weapons.submachineguns.v100.smg_battery_01",
        );
        let v11 = weapon("Pistol", "weapons.pistols.v100.pistol_battery_01");
        let v66 = weapon(
            "Marksman Rifle",
            "weapons.marksman_rifles.v100.dmr_battery_01",
        );
        let longshot = weapon("Sniper Rifle", "weapons.sniper_rifles.v100.sniper_mips_01");
        let impact = weapon("Assault Rifle", "weapons.auto_rifles.v100.auto_heavy_01");
        let m77 = weapon("Assault Rifle", "weapons.auto_rifles.v100.auto_light_01");
        let outland = weapon("Sniper Rifle", "weapons.sniper_rifles.v100.sniper_mips_02");
        let brrt = weapon("Submachine Gun", "weapons.submachineguns.v100.smg_light_01");
        let bully = weapon("Submachine Gun", "weapons.submachineguns.v100.smg_heavy_01");
        let repeater = weapon("Marksman Rifle", "weapons.dmr_rifles.v100.dmr_heavy_02");
        let wstr = weapon("Shotgun", "weapons.shotguns.v100.shotgun_mips_01");
        let magnum = weapon("Pistol", "weapons.pistols.v100.pistol_heavy_01");
        let ares = weapon("Railgun", "weapons.railguns.v100.railgun_mips_01");
        let zeus = weapon("Railgun", "weapons.railguns.v100.railgun_fusion_01");
        let circuit_breaker = weapon("Shotgun", "weapons.shotguns.v100.shotgun_fusion_01");
        let channel_rifle = weapon(
            "Sniper Rifle",
            "weapons.sniper_rifles.v100.sniper_fusion_01",
        );

        assert!(base_weapon_matches_mod_family(&v75, "Magazine", "battery"));
        assert!(!base_weapon_matches_mod_family(
            &v75,
            "Muzzle",
            "battery_front"
        ));
        assert!(base_weapon_matches_mod_family(&v75, "Muzzle", "dampener"));
        assert!(!base_weapon_matches_mod_family(&v75, "Foregrip", "rifle"));
        assert!(!base_weapon_matches_mod_family(
            &v11,
            "Muzzle",
            "battery_front"
        ));
        assert!(base_weapon_matches_mod_family(&v11, "Muzzle", "dampener"));
        assert!(!base_weapon_matches_mod_family(
            &v22,
            "Muzzle",
            "battery_front"
        ));
        assert!(base_weapon_matches_mod_family(&v22, "Muzzle", "dampener"));
        assert!(base_weapon_matches_mod_family(
            &v66,
            "Muzzle",
            "battery_front"
        ));
        assert!(!base_weapon_matches_mod_family(&v66, "Barrel", "precision"));
        assert!(!base_weapon_matches_mod_family(&v22, "Optic", "rifle"));
        assert!(!base_weapon_matches_mod_family(&m77, "Optic", "rifle"));
        assert!(base_weapon_matches_mod_family(
            &longshot,
            "Magazine",
            "precision"
        ));
        assert!(!base_weapon_matches_mod_family(
            &longshot,
            "Magazine",
            "heavy_battery"
        ));
        let stopping_mag = "weapon_mods.magazines.rifle.v100.gold_heavy_ar_stopping";
        assert!(weapon_matches_exact_mod_archetype(&impact, stopping_mag));
        assert!(!weapon_matches_exact_mod_archetype(&m77, stopping_mag));
        for fixed_feed in [&brrt, &repeater, &wstr, &outland, &magnum] {
            assert!(!fixed_feed.supports_magazine_mods());
        }
        assert!(bully.supports_magazine_mods());
        assert!(base_weapon_matches_mod_family(
            &ares,
            "Generator",
            "payload"
        ));
        assert!(base_weapon_matches_mod_family(
            &zeus,
            "Generator",
            "heavy_battery"
        ));
        for volt_cell_weapon in [&zeus, &circuit_breaker, &channel_rifle] {
            assert!(base_weapon_matches_mod_family(
                volt_cell_weapon,
                "Magazine",
                "heavy_battery"
            ));
        }
        assert!(!base_weapon_matches_mod_family(
            &circuit_breaker,
            "Generator",
            "heavy_battery"
        ));
        assert!(base_weapon_matches_mod_family(
            &circuit_breaker,
            "Foregrip",
            "shotgun"
        ));
        assert!(!base_weapon_matches_mod_family(
            &circuit_breaker,
            "Muzzle",
            "underbarrel"
        ));
        assert!(!base_weapon_matches_mod_family(
            &channel_rifle,
            "Generator",
            "heavy_battery"
        ));
        assert!(!base_weapon_matches_mod_family(
            &channel_rifle,
            "Barrel",
            "precision"
        ));
        assert!(base_weapon_matches_mod_family(
            &channel_rifle,
            "Muzzle",
            "battery_front"
        ));
        assert!(base_weapon_matches_mod_family(
            &channel_rifle,
            "Optic",
            "sniper"
        ));
    }

    #[test]
    fn extracts_structural_types() {
        let mut wordlist = FxHashMap::default();
        wordlist.insert(1, "type_backpack".to_owned());
        wordlist.insert(2, "generic".to_owned());
        wordlist.insert(3, "unrelated".to_owned());
        let data = [1_u32, 2, 3]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            extract_types(&data, &wordlist),
            vec!["generic".to_owned(), "type_backpack".to_owned()]
        );
    }

    #[test]
    fn resolves_authored_internal_categories_and_preserves_unknown_hashes() {
        fn write_table(data: &mut [u8], header: usize, target: usize, count: u64) {
            let relative = i64::try_from(target).unwrap() - i64::try_from(header).unwrap() - 0x18;
            data[header..header + 8].copy_from_slice(&count.to_le_bytes());
            data[header + 8..header + 16].copy_from_slice(&relative.to_le_bytes());
        }

        let namespace_hash = 0xc130_a42a_u32;
        let known_hash = 0xe6f4_930d_u32;
        let unknown_hash = 0x2222_2222_u32;
        let mut wordlist = FxHashMap::default();
        wordlist.insert(namespace_hash, "behaviors".to_owned());
        wordlist.insert(known_hash, "behaviors.durable_item".to_owned());

        let mut namespaces = vec![0_u8; 0x34];
        write_table(&mut namespaces, 0x8, 0x30, 1);
        namespaces[0x30..0x34].copy_from_slice(&namespace_hash.to_le_bytes());
        let namespace_hashes = parse_internal_category_namespace_hashes(&namespaces);
        assert_eq!(namespace_hashes.get(&0), Some(&namespace_hash));

        let mut registry_data = vec![0_u8; 0x70];
        write_table(&mut registry_data, 0x8, 0x40, 2);
        write_table(&mut registry_data, 0x18, 0x60, 2);
        for (offset, hash, packed) in [
            (0x40, known_hash, 0_u32),
            (0x48, unknown_hash, 0_u32),
            (0x60, known_hash, 0_u32),
            (0x68, unknown_hash, 1_u32),
        ] {
            registry_data[offset..offset + 4].copy_from_slice(&hash.to_le_bytes());
            registry_data[offset + 4..offset + 8].copy_from_slice(&packed.to_le_bytes());
        }
        let registry =
            parse_internal_category_registry(&registry_data, &namespace_hashes, &wordlist);
        assert_eq!(
            registry.get(&0).map(String::as_str),
            Some("behaviors.durable_item")
        );
        assert_eq!(
            registry.get(&1).map(String::as_str),
            Some("behaviors.#22222222")
        );

        let mut definition = vec![0_u8; 0x196];
        write_table(&mut definition, INTERNAL_CATEGORY_LIST_OFFSET, 0x190, 3);
        definition[0x190..0x192].copy_from_slice(&0_u16.to_le_bytes());
        definition[0x192..0x194].copy_from_slice(&1_u16.to_le_bytes());
        definition[0x194..0x196].copy_from_slice(&0_u16.to_le_bytes());
        assert_eq!(
            extract_internal_categories(&definition, &registry),
            vec![
                "behaviors.durable_item".to_owned(),
                "behaviors.#22222222".to_owned(),
            ]
        );
    }

    #[test]
    fn lookup_pairs_are_hash_then_tag() {
        let internal_hash = 0x67cc7158_u32;
        let display_tag = TagHash(0x80b2a4da);
        let mut data = vec![0; 0x20];
        data[0..4].copy_from_slice(&internal_hash.to_le_bytes());
        data[0x10..0x14].copy_from_slice(&display_tag.0.to_le_bytes());

        assert_eq!(
            parse_hash_tag_pairs(&data, |tag| tag == display_tag),
            vec![(internal_hash, display_tag)]
        );
    }

    #[test]
    fn reads_definition_type_code_at_both_record_alignments() {
        let mut aligned = vec![0_u8; 0x408];
        let aligned_offset = aligned.len() - 8;
        aligned[aligned_offset..aligned_offset + 4].copy_from_slice(&0x01e5_0133_u32.to_le_bytes());
        assert_eq!(definition_type_code(&aligned), Some(0x133));

        let mut shifted = vec![0_u8; 0x406];
        let shifted_offset = shifted.len() - 8;
        shifted[shifted_offset..shifted_offset + 4].copy_from_slice(&0x0133_0000_u32.to_le_bytes());
        assert_eq!(definition_type_code(&shifted), Some(0x133));
    }

    #[test]
    fn reads_cosmetic_group_keys_from_legacy_and_updated_records() {
        let legacy_key = (20_u32 << 16) | 6;
        let mut legacy = vec![0_u8; 0x408];
        let legacy_group_offset = legacy.len() - 0x4c;
        let legacy_type_offset = legacy.len() - 8;
        legacy[legacy_group_offset..legacy_group_offset + 4]
            .copy_from_slice(&legacy_key.to_le_bytes());
        legacy[legacy_type_offset..legacy_type_offset + 4]
            .copy_from_slice(&0x01e5_0133_u32.to_le_bytes());
        assert_eq!(definition_group_key(&legacy), Some(legacy_key));

        let updated_key = (51_u32 << 16) | 6;
        let mut updated = vec![0_u8; 0x438];
        let updated_group_offset = updated.len() - 0x74;
        let updated_type_offset = updated.len() - 8;
        updated[updated_group_offset..updated_group_offset + 4]
            .copy_from_slice(&updated_key.to_le_bytes());
        updated[updated_type_offset..updated_type_offset + 4]
            .copy_from_slice(&0x01ef_0137_u32.to_le_bytes());
        assert_eq!(definition_group_key(&updated), Some(updated_key));
        assert!(is_weapon_or_melee_skin_type(0x133));
        assert!(is_weapon_or_melee_skin_type(0x137));
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn resolves_revamp_br33_visual_fixture() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let weapon = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "BR33 Volley Rifle")
            .expect("BR33 Volley Rifle catalog entry");
        let skin = weapon
            .skins
            .iter()
            .find(|skin| skin.model_tag == TagHash(0x80A9FF17))
            .expect("Vibrant Sport skin");
        assert_eq!(skin.name, "Vibrant Sport");

        let fixture_mods = weapon
            .slots
            .iter()
            .flat_map(|slot| {
                slot.mods
                    .iter()
                    .map(move |modification| (slot.name.as_str(), modification))
            })
            .filter(|(_slot, modification)| {
                matches!(
                    modification.name.as_str(),
                    "Cold Vigilance Scope" | "Impulse Brake"
                ) && modification.preview_rarity == Some(WeaponModRarity::Deluxe)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            fixture_mods.len(),
            2,
            "BR33 Deluxe fixture mods: {:?}",
            weapon
                .slots
                .iter()
                .flat_map(|slot| slot.mods.iter().map(move |item| (
                    slot.name.as_str(),
                    item.name.as_str(),
                    item.rarity.as_str(),
                    item.model_tag,
                )))
                .collect::<Vec<_>>()
        );

        let mod_tags = fixture_mods
            .iter()
            .map(|(_, item)| item.model_tag)
            .collect::<Vec<_>>();
        let socket = crate::geometry::WeaponModSocketIndex::new()
            .owner_for(&cache, weapon.owner_tag, &mod_tags)
            .expect("BR33 socket owner");
        let entry = tiger_pkg::package_manager()
            .get_entry(skin.model_tag)
            .expect("Vibrant Sport Pattern");
        let attachments = fixture_mods
            .iter()
            .map(|(_, item)| crate::geometry::WeaponModPreviewAttachment {
                model_tag: item.model_tag,
                rarity: item.preview_rarity,
                unique_id: 0.5,
            })
            .collect::<Vec<_>>();
        let preview = crate::geometry::GeometryTagPreview::load_model_with_weapon_mod_attachments(
            cache.clone(),
            skin.model_tag,
            &entry,
            weapon.owner_tag,
            socket,
            &attachments,
        )
        .expect("assembled BR33 fixture");
        let crate::geometry::GeometryPreviewKind::Model(model) = preview.kind else {
            panic!("BR33 fixture must be a model")
        };
        let wireframe = model.wireframe.expect("BR33 fixture wireframe");
        let expected_palette =
            crate::geometry::weapon_skin_gear_dye_palette(&cache, skin.model_tag)
                .expect("Vibrant Sport GearDye palette");
        let dyed_ranges = wireframe
            .material_ranges
            .iter()
            .filter(|range| range.textures.gear_dye_palette.is_some())
            .collect::<Vec<_>>();
        assert!(!dyed_ranges.is_empty(), "fixture mods must consume GearDye");
        assert!(dyed_ranges.iter().all(|range| {
            let index = usize::from(range.gear_dye_change_color_index.expect("GearDye channel"));
            range.textures.gear_dye_palette == Some(expected_palette)
                && range.textures.gear_dye == expected_palette.get(index).copied()
        }));
        eprintln!(
            "REVAMP_BR33_FIXTURE skin={} owner={} mods={:?}",
            skin.model_tag,
            weapon.owner_tag,
            fixture_mods
                .iter()
                .map(|(slot, item)| (
                    *slot,
                    item.name.as_str(),
                    item.model_tag,
                    item.preview_rarity
                ))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn applies_every_br33_skin_palette_to_every_br33_mod() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let weapon = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "BR33 Volley Rifle")
            .expect("BR33 Volley Rifle catalog entry");
        let mods = weapon
            .slots
            .iter()
            .flat_map(|slot| slot.mods.iter())
            .unique_by(|modification| modification.model_tag)
            .collect::<Vec<_>>();
        assert!(!weapon.skins.is_empty(), "BR33 skins");
        assert!(!mods.is_empty(), "BR33 mods");

        for skin in &weapon.skins {
            let expected_palette =
                crate::geometry::weapon_skin_gear_dye_palette(&cache, skin.model_tag)
                    .unwrap_or_else(|| {
                        panic!("{} ({}) has no GearDye palette", skin.name, skin.model_tag)
                    });
            let entry = tiger_pkg::package_manager()
                .get_entry(skin.model_tag)
                .unwrap_or_else(|| panic!("missing skin Pattern {}", skin.model_tag));

            for modification in &mods {
                let socket = crate::geometry::WeaponModSocketIndex::new()
                    .owner_for(&cache, weapon.owner_tag, &[modification.model_tag])
                    .unwrap_or_else(|| {
                        panic!(
                            "{} ({}) has no BR33 socket",
                            modification.name, modification.model_tag
                        )
                    });
                let attachment = crate::geometry::WeaponModPreviewAttachment {
                    model_tag: modification.model_tag,
                    rarity: modification.preview_rarity,
                    unique_id: 0.5,
                };
                let preview =
                    crate::geometry::GeometryTagPreview::load_model_with_weapon_mod_attachments(
                        cache.clone(),
                        skin.model_tag,
                        &entry,
                        weapon.owner_tag,
                        socket,
                        &[attachment],
                    )
                    .unwrap_or_else(|| {
                        panic!(
                            "could not assemble skin {} with mod {}",
                            skin.model_tag, modification.model_tag
                        )
                    });
                let crate::geometry::GeometryPreviewKind::Model(model) = preview.kind else {
                    panic!("BR33 skin/mod preview must be a model")
                };
                let wireframe = model.wireframe.expect("BR33 skin/mod wireframe");
                let dyed_ranges = wireframe
                    .material_ranges
                    .iter()
                    .filter(|range| range.textures.gear_dye_palette.is_some())
                    .collect::<Vec<_>>();
                assert!(
                    !dyed_ranges.is_empty(),
                    "{} ({}) on {} ({}) has no dyed material ranges",
                    modification.name,
                    modification.model_tag,
                    skin.name,
                    skin.model_tag
                );
                assert!(dyed_ranges.iter().all(|range| {
                    let Some(index) = range.gear_dye_change_color_index.map(usize::from) else {
                        return false;
                    };
                    range.textures.gear_dye_palette == Some(expected_palette)
                        && range.textures.gear_dye == expected_palette.get(index).copied()
                }));
            }
        }

        eprintln!(
            "verified {} BR33 skins x {} BR33 mods = {} GearDye assemblies",
            weapon.skins.len(),
            mods.len(),
            weapon.skins.len() * mods.len()
        );
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn applies_crash_casket_palette_to_every_overrun_mod() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let weapon = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "Overrun AR")
            .expect("Overrun AR catalog entry");
        let skin = weapon
            .skins
            .iter()
            .find(|skin| skin.name == "Crash Casket")
            .expect("Crash Casket skin");
        assert_eq!(skin.model_tag, TagHash(0x80B7B76C));
        let expected_palette =
            crate::geometry::weapon_skin_gear_dye_palette(&cache, skin.model_tag)
                .expect("Crash Casket GearDye palette");
        assert_eq!(
            expected_palette[4].color,
            [0.021219, 0.107023, 0.491021, 1.0]
        );
        assert_eq!(
            expected_palette[5].color,
            [0.637597, 0.042311, 0.042311, 1.0]
        );
        let mods = weapon
            .slots
            .iter()
            .flat_map(|slot| slot.mods.iter())
            .unique_by(|modification| modification.model_tag)
            .collect::<Vec<_>>();
        assert_eq!(mods.len(), 15, "Overrun AR mod models");
        let entry = tiger_pkg::package_manager()
            .get_entry(skin.model_tag)
            .expect("Crash Casket Pattern");

        for modification in mods {
            let socket = crate::geometry::WeaponModSocketIndex::new()
                .owner_for(&cache, weapon.owner_tag, &[modification.model_tag])
                .unwrap_or_else(|| panic!("{} has no Overrun socket", modification.name));
            let attachment = crate::geometry::WeaponModPreviewAttachment {
                model_tag: modification.model_tag,
                rarity: modification.preview_rarity,
                unique_id: 0.5,
            };
            let preview =
                crate::geometry::GeometryTagPreview::load_model_with_weapon_mod_attachments(
                    cache.clone(),
                    skin.model_tag,
                    &entry,
                    weapon.owner_tag,
                    socket,
                    &[attachment],
                )
                .unwrap_or_else(|| panic!("could not assemble {}", modification.name));
            let crate::geometry::GeometryPreviewKind::Model(model) = preview.kind else {
                panic!("Crash Casket mod preview must be a model")
            };
            let dyed_ranges = model
                .wireframe
                .expect("Crash Casket mod wireframe")
                .material_ranges
                .into_iter()
                .filter(|range| range.textures.gear_dye_palette.is_some())
                .collect::<Vec<_>>();
            assert!(
                !dyed_ranges.is_empty(),
                "{} ({}) has no dyed ranges",
                modification.name,
                modification.model_tag
            );
            assert!(dyed_ranges.iter().all(|range| {
                let Some(index) = range.gear_dye_change_color_index.map(usize::from) else {
                    return false;
                };
                range.textures.gear_dye_palette == Some(expected_palette)
                    && range.textures.gear_dye == expected_palette.get(index).copied()
            }));
        }
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn audits_every_catalog_weapon_skin_gear_dye_palette() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let mut checked = 0usize;
        let mut object_channels = 0usize;
        let mut legacy_singletons = 0usize;
        let mut missing = vec![];
        for weapon in &catalog.weapons {
            for skin in &weapon.skins {
                checked += 1;
                match crate::geometry::weapon_skin_gear_dye_uses_object_channels(
                    &cache,
                    skin.model_tag,
                ) {
                    Some(true) => object_channels += 1,
                    Some(false) => legacy_singletons += 1,
                    None => {
                        missing.push((weapon.name.as_str(), skin.name.as_str(), skin.model_tag))
                    }
                }
            }
        }
        eprintln!(
            "catalog GearDye palettes: checked={checked} object_channels={object_channels} legacy_singletons={legacy_singletons} missing={}",
            missing.len()
        );
        assert!(
            missing.is_empty(),
            "catalog weapon skins missing GearDye palettes: {missing:#?}"
        );
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn audits_every_catalog_weapon_skin_mod_gear_dye_contract() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let socket_index = crate::geometry::WeaponModSocketIndex::new();
        let mut associations = 0usize;
        let mut combinations = 0usize;
        let mut failures = vec![];

        for weapon in &catalog.weapons {
            let mods = weapon
                .slots
                .iter()
                .flat_map(|slot| slot.mods.iter())
                .unique_by(|modification| modification.model_tag)
                .collect::<Vec<_>>();
            combinations += weapon.skins.len() * mods.len();
            for modification in mods {
                associations += 1;
                if socket_index
                    .owner_for(&cache, weapon.owner_tag, &[modification.model_tag])
                    .is_none()
                {
                    failures.push((
                        weapon.name.as_str(),
                        modification.name.as_str(),
                        modification.model_tag,
                        "missing socket",
                    ));
                    continue;
                }
                let geometry =
                    crate::geometry::weapon_mod_gear_dye_channels(&cache, modification.model_tag);
                if geometry.is_empty() {
                    failures.push((
                        weapon.name.as_str(),
                        modification.name.as_str(),
                        modification.model_tag,
                        "missing geometry",
                    ));
                } else if geometry
                    .iter()
                    .all(|(_geometry, channels)| channels.is_empty())
                {
                    failures.push((
                        weapon.name.as_str(),
                        modification.name.as_str(),
                        modification.model_tag,
                        "missing GearDye channels",
                    ));
                } else if geometry
                    .iter()
                    .flat_map(|(_geometry, channels)| channels)
                    .any(|channel| *channel >= 6)
                {
                    failures.push((
                        weapon.name.as_str(),
                        modification.name.as_str(),
                        modification.model_tag,
                        "invalid GearDye channel",
                    ));
                }
            }
        }

        eprintln!(
            "catalog mod GearDye: associations={associations} skin_mod_combinations={combinations} failures={}",
            failures.len()
        );
        assert!(
            failures.is_empty(),
            "catalog weapon mod GearDye contract failures: {failures:#?}"
        );
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn verifies_shared_grips_use_authored_weapon_socket_transforms() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let weapon = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "M77 Assault Rifle")
            .expect("M77 catalog entry");
        let grips = weapon
            .slots
            .iter()
            .find(|slot| slot.name == "Grip")
            .expect("M77 grip slot")
            .mods
            .iter()
            .unique_by(|modification| modification.model_tag)
            .collect::<Vec<_>>();
        assert_eq!(grips.len(), 4, "M77 grip model count");
        let tags = grips
            .iter()
            .map(|modification| modification.model_tag)
            .collect::<Vec<_>>();
        let socket_index = crate::geometry::WeaponModSocketIndex::new();
        let socket = socket_index
            .owner_for(&cache, weapon.owner_tag, &tags)
            .expect("M77 socket owner");
        let impact = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "Impact H-AR")
            .expect("Impact catalog entry");
        let impact_socket = socket_index
            .owner_for(&cache, impact.owner_tag, &[TagHash(0x80A60163)])
            .expect("Impact socket owner");
        assert_eq!(socket, TagHash(0x80A7C09E));
        assert_eq!(impact_socket, TagHash(0x80A7C165));
        let impact_pose =
            crate::geometry::weapon_mod_attachment_pose(&cache, impact_socket, TagHash(0x80A60163))
                .expect("Impact Sturdy pose");
        assert!((impact_pose.translation[0] - 0.24762633).abs() < 0.000_001);
        assert!((impact_pose.translation[2] - 0.0621138).abs() < 0.000_001);
        assert!((impact_pose.scale - 1.0).abs() < 0.000_001);
        let reference_pose = crate::geometry::weapon_mod_attachment_pose(&cache, socket, tags[0])
            .expect("M77 reference grip pose");
        for grip in grips {
            let pose = crate::geometry::weapon_mod_attachment_pose(&cache, socket, grip.model_tag)
                .expect("M77 grip pose");
            assert_eq!(pose.family_id, 0x88116137, "{} family", grip.name);
            assert_eq!(pose.bone_index, 0, "{} bone", grip.name);
            assert!(pose.rotation.iter().all(|value| value.is_finite()));
            assert!(pose.translation.iter().all(|value| value.is_finite()));
            assert!(pose.scale.is_finite() && pose.scale > 0.0);
            if grip.model_tag == TagHash(0x80A60163) {
                assert!((pose.translation[0] - 0.24254519).abs() < 0.000_001);
                assert!((pose.translation[2] - 0.0811943).abs() < 0.000_001);
            }

            let bounds =
                crate::geometry::debug_weapon_mod_geometry_bounds(&cache, grip.model_tag, pose);
            assert!(!bounds.is_empty(), "{} geometry", grip.name);
            for (_geometry, _raw_min, _raw_max, placed_min, placed_max) in bounds {
                for axis in 0..3 {
                    assert!(placed_min[axis].is_finite());
                    assert!(placed_max[axis].is_finite());
                    assert!(placed_min[axis] <= placed_max[axis]);
                }
            }
        }

        // Cosmetic M77 Patterns that carry their own socket signature resolve
        // the same transform. Carbon Bloom and Crash Casket omit that duplicate
        // table and intentionally fall back to the owning weapon above.
        for skin in &weapon.skins {
            let Some(skin_socket) = socket_index.owner_for(&cache, skin.model_tag, &tags) else {
                continue;
            };
            let pose = crate::geometry::weapon_mod_attachment_pose(&cache, skin_socket, tags[0])
                .unwrap_or_else(|| panic!("{} grip pose", skin.name));
            assert_eq!(
                pose.rotation, reference_pose.rotation,
                "{} rotation",
                skin.name
            );
            assert_eq!(pose.scale, reference_pose.scale, "{} scale", skin.name);
            assert_eq!(
                pose.translation, reference_pose.translation,
                "{} translation",
                skin.name
            );
        }
    }

    #[test]
    #[ignore = "requires a local updated Marathon package installation"]
    fn loads_updated_marathon_weapon_skin_catalog() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let updated_skins = view
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon Skin")
                    && item.definition_type_code == Some(0x137)
            })
            .collect::<Vec<_>>();
        let knife_tac = view
            .items
            .iter()
            .find(|item| item.internal_hash == Some(0xac3f_9841))
            .expect("updated knife TAC Standard");
        assert_eq!(knife_tac.item_type.as_deref(), Some("Melee"));
        let kkv_tac = view
            .items
            .iter()
            .find(|item| item.internal_hash == Some(0x3811_c817))
            .expect("updated KKV TAC Standard");
        assert_eq!(kkv_tac.item_type.as_deref(), Some("Weapon Skin"));
        assert_eq!(kkv_tac.applies_to.as_deref(), Some("KKV-9SD"));
        assert!(view.items.iter().any(|item| {
            item.item_type.as_deref() == Some("Weapon")
                && item.name == "KKV-9SD"
                && item.model_tag == kkv_tac.model_tag
        }));
        for (model, owner) in [
            (0x80B7_D13F, "Misriah 2442"),
            (0x80B7_CB76, "Magnum MC"),
            (0x80A9_F38C, "Outland"),
            (0x80AA_0CA3, "BRRT SMG"),
            (0x80B1_4992, "WSTR Combat Shotgun"),
        ] {
            let skin = updated_skins
                .iter()
                .find(|skin| skin.model_tag == Some(TagHash(model)))
                .expect("reported updated skin model");
            assert_eq!(
                skin.applies_to.as_deref(),
                Some(owner),
                "wrong owner for {model:08X}"
            );
        }
        assert!(updated_skins.len() > 220, "updated skins disappeared");
        assert!(
            updated_skins
                .iter()
                .all(|skin| skin.definition_group_index().is_some_and(|group| group > 0))
        );
        let unresolved_skins = updated_skins
            .iter()
            .filter(|skin| skin.applies_to.is_none())
            .map(|skin| {
                (
                    skin.model_tag,
                    skin.name.as_str(),
                    skin.definition_group_index(),
                    skin.internal_hash,
                    skin.internal_name.as_deref(),
                    skin.description.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        assert!(
            unresolved_skins.is_empty(),
            "updated weapon skins lost their weapon ownership: {unresolved_skins:?}"
        );
        let reused_group_owners = updated_skins
            .iter()
            .filter(|skin| skin.definition_group_index() == Some(4))
            .filter_map(|skin| skin.applies_to.as_deref())
            .collect::<FxHashSet<_>>();
        assert!(reused_group_owners.contains("KKV-9SD"));
        assert!(reused_group_owners.contains("Bully SMG"));
        assert!(
            updated_skins.iter().all(|skin| {
                skin.model_tag.is_some()
                    || skin.applies_to.as_deref() == Some("Certification test (no weapon)")
            }),
            "renderable updated weapon skin lost its model tag"
        );
        assert!(
            updated_skins.iter().all(|skin| skin.rarity.is_some()),
            "updated skin rarity footer IDs were not decoded"
        );
        let standard_model_owners = updated_skins
            .iter()
            .filter(|skin| skin.rarity == Some(GearRarity::Standard))
            .filter_map(|skin| {
                let model = skin.model_tag?;
                let mut owners = view
                    .items
                    .iter()
                    .filter(|item| {
                        item.item_type.as_deref() == Some("Weapon") && item.model_tag == Some(model)
                    })
                    .map(|item| item.name.as_str())
                    .unique();
                let owner = owners.next()?;
                owners.next().is_none().then_some(owner)
            })
            .collect::<FxHashSet<_>>();
        let canonical_weapon_names = view
            .items
            .iter()
            .filter(|item| {
                item.item_type.as_deref() == Some("Weapon")
                    && (is_canonical_weapon_skin_owner(item)
                        || item.internal_hash == Some(BIOTOXIC_DISINJECTOR_HASH))
            })
            .map(|item| item.name.as_str())
            .chain(standard_model_owners)
            .collect::<FxHashSet<_>>();
        let noncanonical_owned_skins = updated_skins
            .iter()
            .filter(|skin| {
                skin.applies_to.as_deref().is_some_and(|owner| {
                    owner != "Certification test (no weapon)"
                        && !canonical_weapon_names.contains(owner)
                })
            })
            .map(|skin| (skin.name.as_str(), skin.applies_to.as_deref()))
            .collect::<Vec<_>>();
        assert!(
            noncanonical_owned_skins.is_empty(),
            "skins assigned outside canonical base weapons: {noncanonical_owned_skins:?}"
        );
        for (model, name, owner, rarity) in [
            (
                0x80b7db07,
                "Vox Nocturna",
                "Copperhead RF",
                GearRarity::Superior,
            ),
            (
                0x80b7d9e8,
                "Arata Vectus",
                "Bully SMG",
                GearRarity::Superior,
            ),
        ] {
            let skin = updated_skins
                .iter()
                .find(|skin| skin.model_tag == Some(TagHash(model)))
                .expect("reported updated skin model");
            assert_eq!(skin.name, name);
            assert_eq!(skin.applies_to.as_deref(), Some(owner));
            assert_eq!(skin.rarity, Some(rarity));
        }

        let aqua = updated_skins
            .iter()
            .find(|skin| skin.name == "Aqua Stellar")
            .expect("Aqua Stellar skin");
        assert_eq!(aqua.applies_to.as_deref(), Some("V75 SCAR"));
        let aqua_model = aqua.model_tag.expect("Aqua Stellar model");
        let catalog = view.model_weapon_catalog();
        for (model, owner) in [
            (TagHash(0x80b7db07), "Copperhead RF"),
            (TagHash(0x80b7d9e8), "Bully SMG"),
        ] {
            let weapon = catalog
                .weapons
                .iter()
                .find(|weapon| weapon.name == owner)
                .expect("reported skin base weapon in Models catalog");
            let skin = weapon
                .skins
                .iter()
                .find(|skin| skin.model_tag == model)
                .expect("reported skin in Models catalog");
            assert_eq!(skin.rarity.as_deref(), Some("Superior"));
            assert_eq!(skin.color, GearRarity::Superior.color());
        }
        let ambiguous_base_models = view
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .filter_map(|item| Some((item.model_tag?, item.name.as_str())))
            .into_group_map()
            .into_iter()
            .filter(|(_, names)| names.iter().unique().count() > 1)
            .map(|(model, _)| model)
            .collect::<FxHashSet<_>>();
        assert!(catalog.weapons.iter().all(|weapon| {
            weapon
                .model_tags
                .iter()
                .all(|model| !ambiguous_base_models.contains(model))
        }));
        assert!(catalog.weapons.iter().any(|weapon| {
            weapon.name == "V75 SCAR"
                && weapon
                    .skins
                    .iter()
                    .any(|skin| skin.name == "Aqua Stellar" && skin.model_tag == aqua_model)
        }));

        let textured_grip_path = "weapon_mods.foregrips.rifle.v100.textured0";
        assert!(view.items.iter().any(|item| {
            item.item_type.as_deref() == Some("Weapon Mod")
                && item.internal_name.as_deref() == Some(textured_grip_path)
                && item.model_tag == Some(TagHash(0x80A6_05EF))
        }));
        let visual_mod_models = view
            .items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .filter_map(|item| item.model_tag)
            .collect::<FxHashSet<_>>();
        assert!(
            visual_mod_models.len() > 100,
            "updated pattern-global rows collapsed weapon mods to {} placeholder models",
            visual_mod_models.len()
        );

        let bully = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "Bully SMG")
            .expect("Bully Models catalog");
        assert_eq!(
            bully
                .slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            ["Optic", "Barrel", "Magazine"]
        );
        let bully_mods = bully
            .slots
            .iter()
            .filter_map(|slot| slot.mods.first().map(|item| item.model_tag))
            .collect::<Vec<_>>();
        let socket_index = crate::geometry::WeaponModSocketIndex::new();
        let bully_socket = socket_index
            .owner_for(&cache, bully.owner_tag, &bully_mods)
            .expect("Bully authored socket owner");
        assert!(bully_mods.iter().all(|modification| {
            crate::geometry::weapon_mod_attachment_pose(&cache, bully_socket, *modification)
                .is_some()
        }));
        let bully_defaults =
            crate::geometry::weapon_default_mod_patterns(&cache, bully.owner_tag, bully_socket);
        assert!(
            !bully_defaults.is_empty(),
            "updated Bully Pattern lost authored default mods"
        );
        assert!(bully_defaults.iter().all(|default| {
            crate::geometry::weapon_mod_attachment_pose(&cache, bully_socket, *default).is_some()
        }));

        let kkv_model = kkv_tac.model_tag.expect("KKV TAC Pattern");
        let kkv_index = crate::gui::modellist::weapon_index_for_model(&catalog, kkv_model)
            .expect("KKV TAC identifies exactly one Models weapon");
        assert_eq!(catalog.weapons[kkv_index].name, "KKV-9SD");
        assert!(
            !catalog
                .weapons
                .iter()
                .find(|weapon| weapon.name == "Bully SMG")
                .expect("Bully Models catalog")
                .model_tags
                .contains(&kkv_model)
        );

        let expected_modifiable_weapons = [
            "ARES RG",
            "BR33 Volley Rifle",
            "BRRT SMG",
            "Bully SMG",
            "CE Tactical Sidearm",
            "Conquest LMG",
            "Copperhead RF",
            "D54 Battle Pistol",
            "Demolition HMG",
            "Hardline PR",
            "Impact H-AR",
            "KKV-9SD",
            "Longshot",
            "M77 Assault Rifle",
            "Magnum MC",
            "Misriah 2442",
            "Outland",
            "Overrun AR",
            "Repeater HPR",
            "Retaliator LMG",
            "Stryder M1T",
            "Twin Tap HBR",
            "V00 ZEUS RG",
            "V11 Punch",
            "V22 Volt Thrower",
            "V66 Lookout",
            "V75 SCAR",
            "V85 Circuit Breaker",
            "V99 Channel Rifle",
            "WSTR Combat Shotgun",
        ];
        for name in expected_modifiable_weapons {
            let weapon = catalog
                .weapons
                .iter()
                .find(|weapon| weapon.name == name)
                .unwrap_or_else(|| panic!("missing Models weapon {name}"));
            assert!(
                !weapon.slots.is_empty(),
                "{name} lost its authored selectable mod slots"
            );
        }

        let mut runtime_failures = vec![];
        for weapon in catalog
            .weapons
            .iter()
            .filter(|weapon| !weapon.slots.is_empty())
        {
            let modifications = weapon
                .slots
                .iter()
                .flat_map(|slot| slot.mods.iter().map(|item| item.model_tag))
                .unique()
                .collect::<Vec<_>>();
            let Some(socket) = socket_index.owner_for(&cache, weapon.owner_tag, &modifications)
            else {
                runtime_failures.push(format!(
                    "{} {}: no socket owner for {} selectable mods",
                    weapon.name,
                    weapon.owner_tag,
                    modifications.len()
                ));
                continue;
            };

            for modification in &modifications {
                let Some(pose) =
                    crate::geometry::weapon_mod_attachment_pose(&cache, socket, *modification)
                else {
                    runtime_failures.push(format!(
                        "{}: {modification} has no authored socket pose under {socket}",
                        weapon.name
                    ));
                    continue;
                };
                let authored_families =
                    crate::geometry::weapon_mod_authored_families(&cache, *modification);
                if !authored_families.contains(&pose.family_id) {
                    runtime_failures.push(format!(
                        "{}: {modification} resolved through fallback family {:08X}",
                        weapon.name, pose.family_id
                    ));
                }
                if crate::geometry::weapon_mod_geometry_tags(&cache, *modification).is_empty() {
                    runtime_failures.push(format!(
                        "{}: {modification} has no visual geometry",
                        weapon.name
                    ));
                }
            }

            let defaults =
                crate::geometry::weapon_default_mod_patterns(&cache, weapon.owner_tag, socket);
            if defaults.is_empty()
                && matches!(
                    weapon.name.as_str(),
                    "Bully SMG" | "Misriah 2442" | "Longshot" | "Outland" | "V99 Channel Rifle"
                )
            {
                runtime_failures.push(format!(
                    "{} {}: no authored default visual mods",
                    weapon.name, weapon.owner_tag
                ));
            }
            for default in &defaults {
                if crate::geometry::weapon_mod_attachment_pose(&cache, socket, *default).is_none() {
                    runtime_failures.push(format!(
                        "{}: default {default} has no authored socket pose under {socket}",
                        weapon.name
                    ));
                }
                if crate::geometry::weapon_mod_geometry_tags(&cache, *default).is_empty() {
                    runtime_failures.push(format!(
                        "{}: default {default} has no visual geometry",
                        weapon.name
                    ));
                }
            }

            let selected = weapon
                .slots
                .iter()
                .filter_map(|slot| slot.mods.first())
                .unique_by(|item| item.model_tag)
                .collect::<Vec<_>>();
            let selected_tags = selected
                .iter()
                .map(|item| item.model_tag)
                .collect::<Vec<_>>();
            let unoccupied_defaults = crate::geometry::weapon_unoccupied_default_mod_patterns(
                &cache,
                weapon.owner_tag,
                socket,
                &selected_tags,
            );
            let attachments = selected
                .iter()
                .map(|item| crate::geometry::WeaponModPreviewAttachment {
                    model_tag: item.model_tag,
                    rarity: item.preview_rarity,
                    unique_id: 0.5,
                })
                .chain(unoccupied_defaults.iter().map(|default| {
                    crate::geometry::WeaponModPreviewAttachment {
                        model_tag: *default,
                        rarity: None,
                        unique_id: 0.5,
                    }
                }))
                .collect::<Vec<_>>();
            let expected_geometry = attachments
                .iter()
                .flat_map(|attachment| {
                    crate::geometry::weapon_mod_geometry_tags(&cache, attachment.model_tag)
                })
                .collect::<FxHashSet<_>>();
            let render_tag = (weapon.name == "KKV-9SD")
                .then_some(kkv_model)
                .unwrap_or(weapon.owner_tag);
            let Some(entry) = package_manager().get_entry(render_tag) else {
                runtime_failures.push(format!(
                    "{}: missing render Pattern {render_tag}",
                    weapon.name
                ));
                continue;
            };
            let Some(preview) =
                crate::geometry::GeometryTagPreview::load_model_with_weapon_mod_attachments(
                    cache.clone(),
                    render_tag,
                    &entry,
                    weapon.owner_tag,
                    socket,
                    &attachments,
                )
            else {
                runtime_failures.push(format!("{}: preview assembly failed", weapon.name));
                continue;
            };
            let crate::geometry::GeometryPreviewKind::Model(preview) = preview.kind else {
                runtime_failures.push(format!("{}: preview was not a model", weapon.name));
                continue;
            };
            let assembled = preview
                .geometry_parts
                .iter()
                .copied()
                .collect::<FxHashSet<_>>();
            let missing_geometry = expected_geometry
                .difference(&assembled)
                .copied()
                .collect::<Vec<_>>();
            if !missing_geometry.is_empty() {
                runtime_failures.push(format!(
                    "{}: assembled preview omitted attachment geometry {missing_geometry:?}",
                    weapon.name
                ));
            }
        }
        assert!(
            runtime_failures.is_empty(),
            "Models weapon runtime verification failed:\n{}",
            runtime_failures.join("\n")
        );
    }

    #[test]
    fn recognizes_runner_shell_description_headers() {
        assert_eq!(
            runner_shell_model("SHELL MODEL: Vandal\nLore"),
            Some("Vandal")
        );
        assert_eq!(runner_shell_model("Unrelated"), None);
    }

    #[test]
    #[ignore = "requires a local Marathon package installation"]
    fn audits_marathon_internal_gear_categories() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("Gear");
        let categories = items
            .iter()
            .flat_map(|item| item.internal_categories.iter().cloned())
            .collect::<FxHashSet<_>>();
        let missing_registry_entries = categories
            .iter()
            .filter(|category| category.starts_with("category_index_#"))
            .collect::<Vec<_>>();
        let missing_namespace_records = categories
            .iter()
            .filter(|category| category.starts_with("category_hash_#"))
            .collect::<Vec<_>>();

        assert!(
            missing_registry_entries.is_empty(),
            "Gear category IDs missing from the game registry: {missing_registry_entries:#?}"
        );
        assert!(
            missing_namespace_records.is_empty(),
            "Gear categories missing a namespace registry record: {missing_namespace_records:#?}"
        );
        for expected in [
            "ammo_compatibility.ammo_heavy_battery",
            "behaviors.durable_item",
            "loot_table_behaviors.always_excluded",
        ] {
            assert!(
                categories.contains(expected),
                "Gear categories did not contain {expected}"
            );
        }

        let categorized_items = items
            .iter()
            .filter(|item| !item.internal_categories.is_empty())
            .count();
        let hash_only_names = categories
            .iter()
            .filter(|category| category.contains(".#"))
            .count();
        let hash_only_namespaces = categories
            .iter()
            .filter(|category| category.starts_with("namespace_#"))
            .count();
        eprintln!(
            "resolved {} distinct internal categories across {categorized_items}/{} Gear items ({hash_only_names} leaf-hash and {hash_only_namespaces} namespace-hash fallbacks)",
            categories.len(),
            items.len()
        );
    }

    #[test]
    #[ignore = "requires a local Marathon package installation"]
    fn loads_every_marathon_language_without_changing_gear_taxonomy() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));

        let english_strings =
            quicktag_strings::localized::create_stringmap_for_language(LocalizedLanguage::English)
                .expect("English strings");
        let english_items = load_gear_for_language(&english_strings, LocalizedLanguage::English)
            .expect("English Gear");
        let english_by_tag = english_items
            .iter()
            .map(|item| (item.display_tag, item))
            .collect::<FxHashMap<_, _>>();

        for language in LocalizedLanguage::ALL {
            let strings = quicktag_strings::localized::create_stringmap_for_language(language)
                .unwrap_or_else(|error| panic!("{} strings: {error:#}", language.label()));
            assert!(
                !strings.is_empty(),
                "{} string cache is empty",
                language.label()
            );

            let items = load_gear_for_language(&strings, language)
                .unwrap_or_else(|error| panic!("{} Gear: {error}", language.label()));
            assert_eq!(
                items.len(),
                english_items.len(),
                "{} changed the Gear record count",
                language.label()
            );

            let mut changed_names = 0;
            for item in &items {
                let english = english_by_tag[&item.display_tag];
                assert_eq!(
                    item.item_type,
                    english.item_type,
                    "{} changed the category of {}",
                    language.label(),
                    item.display_tag
                );
                assert_eq!(
                    item.subcategory,
                    english.subcategory,
                    "{} changed the subtype of {}",
                    language.label(),
                    item.display_tag
                );
                changed_names += usize::from(item.name != english.name);
            }

            if language != LocalizedLanguage::English {
                assert!(
                    changed_names > 100,
                    "{} only changed {changed_names} Gear names",
                    language.label()
                );
            }
            eprintln!(
                "{}: {} strings, {} Gear records, {changed_names} translated names",
                language.label(),
                strings.len(),
                items.len()
            );
        }
    }

    #[test]
    #[ignore = "requires the current local Marathon package installation"]
    fn audits_current_marathon_gear_contract() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = Arc::new(quicktag_scanner::load_tag_cache());
        view.reconcile_weapon_skin_models(&cache);
        let catalog = view.model_weapon_catalog();
        let items = &view.items;

        assert_eq!(items.len(), 2_415, "current investment row count changed");
        assert!(
            items.iter().all(|item| item.item_type.is_some()),
            "unresolved Gear UI taxonomy: {:#?}",
            items
                .iter()
                .filter(|item| item.item_type.is_none())
                .collect::<Vec<_>>()
        );
        assert!(
            items
                .iter()
                .all(|item| !item.internal_categories.is_empty()),
            "some Gear definitions lost their authored internal-category table"
        );
        let internal_categories = items
            .iter()
            .flat_map(|item| item.internal_categories.iter().map(String::as_str))
            .collect::<FxHashSet<_>>();
        for category in [
            "ammo_compatibility.ammo_heavy_battery",
            "behaviors.durable_item",
            "loot_table_behaviors.always_excluded",
        ] {
            assert!(internal_categories.contains(category), "missing {category}");
        }

        let weapons = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .collect::<Vec<_>>();
        assert_eq!(weapons.len(), 83);
        for name in [
            "Demolition HMG",
            "M77 Assault Rifle",
            "Overrun AR",
            "Stryder M1T",
            "Twin Tap HBR",
        ] {
            let matching = weapons
                .iter()
                .filter(|weapon| weapon.name == name)
                .copied()
                .collect::<Vec<_>>();
            assert!(
                matching
                    .iter()
                    .any(|weapon| weapon.rarity == Some(GearRarity::Standard)),
                "missing baseline weapon {name}"
            );
            assert!(
                matching.iter().all(|weapon| !matches!(
                    weapon.rarity,
                    Some(GearRarity::Contraband | GearRarity::Unique)
                )),
                "baseline weapon {name} has false special rarity: {matching:#?}"
            );
        }

        for (model, expected) in [
            (TagHash(0x80B7C92A), "Hardline PR"),
            (TagHash(0x80B7B8D9), "Overrun AR"),
        ] {
            let owners = catalog
                .weapons
                .iter()
                .filter(|weapon| weapon.model_tags.contains(&model))
                .map(|weapon| weapon.name.as_str())
                .unique()
                .collect::<Vec<_>>();
            assert_eq!(owners, [expected], "wrong Models owner for {model}");
        }

        let runner_cores = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Runner Core"))
            .collect::<Vec<_>>();
        assert_eq!(runner_cores.len(), 80);
        let mut core_owner_counts = runner_cores
            .iter()
            .map(|core| core.applies_to.as_deref().unwrap_or("<missing>"))
            .counts()
            .into_iter()
            .collect::<Vec<_>>();
        core_owner_counts.sort_unstable();
        assert_eq!(
            core_owner_counts,
            vec![
                ("All Shells", 10),
                ("Assassin", 10),
                ("Destroyer", 10),
                ("Recon", 10),
                ("Sentinel", 10),
                ("Thief", 10),
                ("Triage", 10),
                ("Vandal", 10),
            ]
        );
        assert_eq!(
            runner_cores
                .iter()
                .map(|core| core.name.as_str())
                .unique()
                .count(),
            runner_cores.len(),
            "Runner Core localization collapsed"
        );

        let runner_skins = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Runner Skin"))
            .collect::<Vec<_>>();
        for (tag, name, shell, rarity) in [
            (
                TagHash(0x80B140CE),
                "Arata Vectus",
                "Assassin",
                GearRarity::Prestige,
            ),
            (
                TagHash(0x80AA053A),
                "SHADOW INDEX",
                "Destroyer",
                GearRarity::Deluxe,
            ),
        ] {
            let skin = catalog
                .runner_skins
                .iter()
                .find(|skin| skin.model_tag == tag)
                .expect("runner skin Models identity");
            assert_eq!(skin.name, name);
            assert_eq!(skin.shell_name, shell);
            assert_eq!(skin.color, rarity.color());
        }
        let runner_owner_counts = runner_skins
            .iter()
            .map(|skin| skin.applies_to.as_deref().unwrap_or("<missing>"))
            .counts();
        let mut shell_combinations_by_package = FxHashMap::default();
        for package_id in catalog
            .runner_skins
            .iter()
            .map(|skin| skin.model_tag.pkg_id())
            .unique()
        {
            let containers = package_manager().lookup.tag32_entries_by_pkg[&package_id]
                .iter()
                .enumerate()
                .filter(|(_, entry)| matches!(entry.reference, 0x8080BADB | 0x8080BAAD))
                .map(|(index, _)| TagHash::new(package_id, index as u16))
                .collect_vec();
            shell_combinations_by_package.insert(
                package_id,
                crate::geometry::runner_shell_combinations(&cache, &containers),
            );
        }
        for skin in &catalog.runner_skins {
            // Rook cosmetics use a separate non-runner-shell assembly family.
            if skin.shell_name == "Rook" {
                continue;
            }
            let combination = shell_combinations_by_package[&skin.model_tag.pkg_id()]
                .iter()
                .find(|combination| combination.contains(skin.model_tag));
            if skin.shell_name == "Vandal" {
                let combination = combination.unwrap_or_else(|| {
                    panic!("unassembled Vandal skin {} ({})", skin.name, skin.model_tag)
                });
                assert_eq!(
                    combination.additional_parts.len(),
                    1,
                    "Vandal must assemble body + face + hair: {} ({}) {combination:?}",
                    skin.name,
                    skin.model_tag
                );
            }
        }
        for combinations in shell_combinations_by_package.values() {
            assert!(
                combinations
                    .iter()
                    .all(|combination| combination.additional_parts.len() <= 1),
                "runner combination absorbed another skin"
            );
        }
        assert!(
            runner_skins
                .iter()
                .any(|skin| skin.applies_to.as_deref() == Some("Recon")),
            "runner owners: {runner_owner_counts:?}"
        );
        assert!(
            runner_skins
                .iter()
                .any(|skin| skin.applies_to.as_deref() == Some("Rook")),
            "runner owners: {runner_owner_counts:?}"
        );
        assert!(
            runner_skins
                .iter()
                .all(|skin| skin.subcategory.is_some() && skin.applies_to.is_some()),
            "unowned runner skins: {:#?}",
            runner_skins
                .iter()
                .filter(|skin| skin.subcategory.is_none() || skin.applies_to.is_none())
                .collect::<Vec<_>>()
        );
        assert!(items.iter().all(|item| {
            item.definition_type_code != Some(0x128)
                || (item.item_type.as_deref() == Some("Profile")
                    && item.subcategory.as_deref() == Some("Title"))
        }));

        assert!(items.iter().all(|item| {
            !matches!(item.definition_type_code, Some(0x19c | 0x1a9))
                || (item.item_type.as_deref() == Some("Implant")
                    && !item.types.iter().any(|kind| kind.contains("trinket")))
        }));
        assert!(items.iter().all(|item| {
            item.item_type.as_deref() != Some("Trinket")
                || !matches!(item.definition_type_code, Some(0x19c | 0x1a9))
        }));

        let mods = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .collect::<Vec<_>>();
        assert_eq!(mods.len(), 512);
        assert!(mods.iter().all(|item| item.rarity.is_some()));
        let unmatched_mods = mods
            .iter()
            .filter(|item| {
                item.mod_category.as_deref() != Some("Chip")
                    && (item.compatible_weapons.is_empty() || item.mod_is_universal)
            })
            .collect::<Vec<_>>();
        // Firestorm's three Contraband affixes are private quest-weapon
        // behavior rows, not selectable attachment-pool mods. Their authored
        // definitions intentionally omit public compatibility. Every ordinary
        // attachment must still resolve to a concrete, non-universal pool.
        assert_eq!(unmatched_mods.len(), 3);
        assert!(
            unmatched_mods.iter().all(|item| {
                item.rarity == Some(GearRarity::Contraband)
                    && item
                        .internal_categories
                        .iter()
                        .any(|category| category == "loot_table_behaviors.always_excluded")
            }),
            "ordinary weapon mods lost compatibility: {:#?}",
            unmatched_mods
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>()
        );
        assert!(
            mods.iter()
                .filter(|item| item.mod_category.as_deref() == Some("Chip"))
                .all(|item| item.mod_is_universal && item.compatible_weapons.is_empty())
        );

        let compatible = |name: &str, weapon: &str| {
            mods.iter().any(|item| {
                item.name == name
                    && item
                        .compatible_weapons
                        .iter()
                        .any(|candidate| candidate == weapon)
            })
        };
        assert!(compatible("Daredevil Stock", "KKV-9SD"));
        assert!(compatible("Daredevil Stock", "D54 Battle Pistol"));
        assert!(compatible("Null-Grav Generator", "ARES RG"));
        assert!(compatible("Null-Grav Generator", "V00 ZEUS RG"));
        assert!(!mods.iter().any(|item| {
            item.mod_category.as_deref() == Some("Muzzle")
                && item
                    .compatible_weapons
                    .iter()
                    .any(|weapon| weapon == "V85 Circuit Breaker")
        }));
        assert!(compatible("Accu-Point Barrel", "V66 Lookout"));
        assert!(compatible("Accu-Point Barrel", "V99 Channel Rifle"));
        assert!(!compatible("Precision Barrel", "V99 Channel Rifle"));

        let slots = |weapon_name: &str| {
            let weapon = items
                .iter()
                .find(|item| {
                    item.item_type.as_deref() == Some("Weapon") && item.name == weapon_name
                })
                .unwrap_or_else(|| panic!("missing weapon {weapon_name}"));
            compatible_mod_sections(items, weapon)
                .into_iter()
                .map(|(slot, _)| slot)
                .collect::<Vec<_>>()
        };
        assert_eq!(slots("ARES RG"), ["Generator", "Magazine"]);
        assert_eq!(slots("V85 Circuit Breaker"), ["Grip", "Magazine"]);
        assert_eq!(slots("V99 Channel Rifle"), ["Optic", "Barrel", "Magazine"]);

        let weapon_skins = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Skin"))
            .collect::<Vec<_>>();
        assert_eq!(weapon_skins.len(), 225);
        assert!(
            weapon_skins.iter().all(|skin| skin.applies_to.is_some()),
            "unowned weapon skins: {:#?}",
            weapon_skins
                .iter()
                .filter(|skin| skin.applies_to.is_none())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires a local Marathon package installation"]
    fn audits_marathon_localization_collisions() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));

        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut view = GearView::new(strings);
        let cache = quicktag_scanner::load_tag_cache();
        view.reconcile_weapon_skin_models(&cache);
        let items = view.items;
        let mut category_counts = FxHashMap::<&str, usize>::default();
        for item in &items {
            *category_counts
                .entry(item.item_type.as_deref().unwrap_or("<unresolved>"))
                .or_default() += 1;
        }
        let mut category_counts = category_counts.into_iter().collect::<Vec<_>>();
        category_counts.sort_by_key(|(category, _)| *category);
        eprintln!(
            "gear records: {}; categories: {category_counts:?}",
            items.len()
        );
        assert_eq!(items.len(), 2_415);
        let runner_cores = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Runner Core"))
            .collect::<Vec<_>>();
        assert_eq!(runner_cores.len(), 80);
        let mut core_owner_counts = FxHashMap::<&str, usize>::default();
        for core in &runner_cores {
            *core_owner_counts
                .entry(core.applies_to.as_deref().unwrap_or("<missing>"))
                .or_default() += 1;
        }
        let mut core_owner_counts = core_owner_counts.into_iter().collect::<Vec<_>>();
        core_owner_counts.sort_by_key(|(owner, _)| *owner);
        assert_eq!(
            core_owner_counts,
            vec![
                ("All Shells", 10),
                ("Assassin", 10),
                ("Destroyer", 10),
                ("Recon", 10),
                ("Sentinel", 10),
                ("Thief", 10),
                ("Triage", 10),
                ("Vandal", 10),
            ],
            "runner cores must resolve to one complete ten-core shell set"
        );
        let pathless_core_hashes = runner_cores
            .iter()
            .filter(|core| {
                core.internal_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with('#'))
            })
            .filter_map(|core| core.internal_hash)
            .collect::<FxHashSet<_>>();
        assert_eq!(
            pathless_core_hashes,
            SENTINEL_CORE_HASHES.into_iter().collect(),
            "only the verified Sentinel family may require hash-only ownership"
        );
        let shell_names = runner_shell_names(&items);
        for core in runner_cores.iter().filter(|core| {
            core.internal_name
                .as_deref()
                .is_some_and(|name| name.starts_with("implant_cores."))
        }) {
            let archetype =
                runner_core_archetype(core.internal_name.as_deref(), core.internal_hash)
                    .expect("path-backed core archetype");
            let expected = shell_names
                .get(&archetype)
                .map_or(archetype.as_str(), String::as_str);
            assert_eq!(
                Some(expected),
                core.applies_to.as_deref(),
                "runner core owner must agree with its authored path: {core:?}"
            );
        }
        assert_eq!(
            runner_cores
                .iter()
                .map(|item| item.name.as_str())
                .collect::<FxHashSet<_>>()
                .len(),
            runner_cores.len(),
            "runner core names must be unique"
        );
        let deconstruction = items
            .iter()
            .filter(|item| item.name.eq_ignore_ascii_case("Deconstruction"))
            .collect::<Vec<_>>();
        assert_eq!(
            deconstruction.len(),
            1,
            "only the real Deconstruction cosmetic should retain that name: {deconstruction:?}"
        );

        let by_internal = items
            .iter()
            .filter_map(|item| Some((item.internal_name.as_deref()?, item)))
            .collect::<FxHashMap<_, _>>();

        for (internal, expected_name, expected_description) in [
            (
                "materials.core.drive.green",
                "Storage Drive",
                "potential treasure trove",
            ),
            (
                "materials.core.lens.green",
                "Surveillance Lens",
                "high-risk military recon",
            ),
            (
                "materials.core.drive.blue",
                "Amygdala Drive",
                "biological memory response",
            ),
            (
                "materials.core.lens.blue",
                "Thoughtwave Lens",
                "impressions of subject's thoughts",
            ),
        ] {
            let item = by_internal[internal];
            assert_eq!(item.name, expected_name);
            assert!(
                item.description
                    .as_deref()
                    .is_some_and(|value| value.contains(expected_description)),
                "{internal} has wrong description: {:?}",
                item.description
            );
        }

        let biotoxic = by_internal["#A3DF1228"];
        assert_eq!(biotoxic.name, "Biotoxic Disinjector");
        assert_eq!(biotoxic.rarity, Some(GearRarity::Contraband));
        assert_eq!(biotoxic.price, Some(7_777));
        assert_eq!(biotoxic.item_type.as_deref(), Some("Weapon"));
        assert_eq!(biotoxic.subcategory.as_deref(), Some("Hybrid Weapon"));

        let weapons = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .collect::<Vec<_>>();
        assert_eq!(weapons.len(), 78);
        assert!(
            weapons
                .iter()
                .filter(|item| item.subcategory.is_some())
                .count()
                >= 63,
            "too few weapon subtypes resolved"
        );
        for expected in [
            "Assault Rifle",
            "Pistol",
            "Submachine Gun",
            "Shotgun",
            "Sniper Rifle",
            "Railgun",
            "Hybrid Weapon",
        ] {
            assert!(
                weapons
                    .iter()
                    .any(|item| item.subcategory.as_deref() == Some(expected)),
                "missing weapon subcategory {expected}"
            );
        }

        for (internal, category, subtype, facet) in [
            (
                "quest.factions.mida.v100.contracts.uesc_credentials",
                "Quest Item",
                "Faction",
                "faction_mida",
            ),
            (
                "heroes.agile.skins.default.v100.skin01",
                "Runner Skin",
                "Agile",
                "runner_agile",
            ),
            (
                "weapons.auto_rifles.v100.auto_battery_01.skins.default.v100.skin01",
                "Weapon Skin",
                "Assault Rifle",
                "item_type_weapon_skin",
            ),
        ] {
            let item = by_internal[internal];
            assert_eq!(item.item_type.as_deref(), Some(category), "{internal}");
            assert_eq!(item.subcategory.as_deref(), Some(subtype), "{internal}");
            assert!(item.types.iter().any(|value| value == facet), "{internal}");
        }

        let categories = items
            .iter()
            .filter_map(|item| item.item_type.as_deref())
            .collect::<FxHashSet<_>>();
        for expected in [
            "Ammo",
            "Artifact",
            "Backpack",
            "Challenge",
            "Charm",
            "Consumable",
            "Currency",
            "Data Card",
            "Equipment",
            "Finisher",
            "Implant",
            "Item Modifier",
            "Key",
            "Loot Type",
            "Melee",
            "Profile",
            "Progression Reward",
            "Quest Item",
            "Ranked Reward",
            "Reward Package",
            "Runner Core",
            "Runner Skin",
            "Salvage",
            "Schema",
            "Sponsored Kit",
            "Sticker",
            "Tag Chip",
            "Trinket",
            "Valuable",
            "Weapon",
            "Weapon Mod",
            "Weapon Skin",
        ] {
            assert!(categories.contains(expected), "missing category {expected}");
        }

        let unresolved = items
            .iter()
            .filter(|item| item.item_type.is_none())
            .collect::<Vec<_>>();
        assert!(
            unresolved.is_empty(),
            "all gear records should have a category: {unresolved:?}"
        );

        for (category, expected_count) in [
            ("Charm", 52),
            ("Melee", 15),
            ("Profile", 208),
            ("Runner Core", 80),
            ("Runner Skin", 110),
            ("Sticker", 70),
            ("Weapon Skin", 224),
        ] {
            assert_eq!(
                items
                    .iter()
                    .filter(|item| item.item_type.as_deref() == Some(category))
                    .count(),
                expected_count,
                "unexpected {category} count"
            );
        }

        for category in ["Melee", "Runner Skin", "Weapon Skin"] {
            let missing = items
                .iter()
                .filter(|item| item.item_type.as_deref() == Some(category) && item.rarity.is_none())
                .collect::<Vec<_>>();
            assert!(
                missing.is_empty(),
                "{category} records without rarity: {missing:?}"
            );
        }

        let weapon_mods = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .collect::<Vec<_>>();
        assert_eq!(weapon_mods.len(), 509);
        let standard_volt_cell_chambers = weapon_mods
            .iter()
            .filter(|item| {
                item.internal_name.as_deref().is_some_and(|path| {
                    path.starts_with("weapon_mods.magazines.heavy_battery.")
                        && !path.contains(".gold_")
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(standard_volt_cell_chambers.len(), 12);
        assert!(standard_volt_cell_chambers.iter().all(|item| {
            item.compatible_weapons == ["V00 ZEUS RG", "V85 Circuit Breaker", "V99 Channel Rifle"]
        }));
        let volt_cell_generators = weapon_mods
            .iter()
            .filter(|item| {
                item.internal_name
                    .as_deref()
                    .is_some_and(|path| path.starts_with("weapon_mods.generators."))
            })
            .collect::<Vec<_>>();
        assert_eq!(volt_cell_generators.len(), 13);
        assert!(
            volt_cell_generators
                .iter()
                .all(|item| item.compatible_weapons == ["V00 ZEUS RG"])
        );
        let standard_dampeners = weapon_mods
            .iter()
            .filter(|item| {
                item.internal_name.as_deref().is_some_and(|path| {
                    path.starts_with("weapon_mods.muzzles.dampener.") && !path.contains(".gold_")
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(standard_dampeners.len(), 12);
        assert!(standard_dampeners.iter().all(|item| {
            item.compatible_weapons == ["V11 Punch", "V22 Volt Thrower", "V75 SCAR"]
        }));
        let battery_front_mods = weapon_mods
            .iter()
            .filter(|item| {
                item.internal_name
                    .as_deref()
                    .is_some_and(|path| path.starts_with("weapon_mods.muzzles.battery_front."))
            })
            .collect::<Vec<_>>();
        assert_eq!(battery_front_mods.len(), 13);
        assert!(
            battery_front_mods
                .iter()
                .all(|item| item.compatible_weapons == ["V66 Lookout"])
        );
        let mut mod_category_counts = FxHashMap::<&str, usize>::default();
        for item in &weapon_mods {
            *mod_category_counts
                .entry(item.mod_category.as_deref().unwrap_or("<missing>"))
                .or_default() += 1;
        }
        let mut mod_category_counts = mod_category_counts.into_iter().collect::<Vec<_>>();
        mod_category_counts.sort_by_key(|(category, _)| *category);
        assert_eq!(
            mod_category_counts,
            vec![
                ("Barrel", 14),
                ("Chip", 180),
                ("Foregrip", 25),
                ("Generator", 13),
                ("Magazine", 108),
                ("Muzzle", 54),
                ("Optic", 75),
                ("Shield", 14),
                ("Stock", 10),
                ("Unique", 16),
            ]
        );

        assert!(
            weapon_mods
                .iter()
                .filter(|item| { item.mod_category.as_deref() == Some("Chip") })
                .all(|item| item.mod_is_universal && item.compatible_weapons.is_empty())
        );
        let non_universal_without_weapons = weapon_mods
            .iter()
            .filter(|item| item.mod_category.as_deref() != Some("Chip"))
            .filter(|item| item.mod_is_universal || item.compatible_weapons.is_empty())
            .collect::<Vec<_>>();
        assert!(
            non_universal_without_weapons.is_empty(),
            "non-universal mods without weapon compatibility: {non_universal_without_weapons:?}"
        );

        let expected_weapon_slots = [
            ("Impact H-AR", &["Optic", "Grip", "Magazine"][..]),
            ("M77 Assault Rifle", &["Grip", "Magazine"]),
            ("Overrun AR", &["Optic", "Grip", "Magazine"]),
            ("V75 SCAR", &["Optic", "Barrel", "Magazine"]),
            ("Conquest LMG", &["Optic", "Magazine", "Shield"]),
            ("Demolition HMG", &["Optic", "Magazine", "Shield"]),
            ("Retaliator LMG", &["Optic", "Magazine", "Shield"]),
            ("BR33 Volley Rifle", &["Optic", "Barrel", "Magazine"]),
            ("Hardline PR", &["Optic", "Barrel", "Magazine"]),
            ("Repeater HPR", &["Optic", "Barrel"]),
            ("Stryder M1T", &["Optic", "Barrel", "Magazine"]),
            ("Twin Tap HBR", &["Optic", "Barrel", "Magazine"]),
            ("V66 Lookout", &["Optic", "Barrel", "Magazine"]),
            ("ARES RG", &["Magazine"]),
            ("V00 ZEUS RG", &["Generator", "Magazine"]),
            ("Misriah 2442", &["Barrel", "Grip", "Magazine"]),
            ("V85 Circuit Breaker", &["Grip", "Magazine"]),
            ("WSTR Combat Shotgun", &["Barrel", "Grip"]),
            ("Longshot", &["Optic", "Barrel", "Magazine"]),
            ("Outland", &["Optic", "Barrel"]),
            ("V99 Channel Rifle", &["Optic", "Barrel", "Magazine"]),
            ("BRRT SMG", &["Optic", "Barrel"]),
            ("Bully SMG", &["Optic", "Barrel", "Magazine"]),
            ("Copperhead RF", &["Optic", "Barrel", "Magazine"]),
            ("V22 Volt Thrower", &["Barrel", "Magazine"]),
            ("CE Tactical Sidearm", &["Optic", "Barrel", "Magazine"]),
            ("Magnum MC", &["Optic", "Barrel"]),
            ("V11 Punch", &["Optic", "Barrel", "Magazine"]),
            ("D54 Battle Pistol", &["Barrel", "Magazine", "Shield"]),
            ("KKV-9SD", &["Optic", "Magazine", "Shield"]),
            ("V99 Watchtower", &["Unique"]),
        ]
        .into_iter()
        .collect::<FxHashMap<_, _>>();
        let mut checked_weapon_slots = FxHashSet::default();
        let mut overflowing_weapon_slots = vec![];
        for weapon in items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
        {
            let sections = compatible_mod_sections(&items, weapon);
            assert!(
                sections.iter().all(|(_, indices)| !indices.is_empty()),
                "{} has an empty mod slot",
                weapon.name
            );
            let categories = sections
                .iter()
                .map(|(category, _)| category.clone())
                .collect::<Vec<_>>();
            if sections.len() > 4 {
                overflowing_weapon_slots.push((weapon.name.as_str(), categories.clone()));
            }
            if let Some(expected) = expected_weapon_slots.get(weapon.name.as_str()) {
                assert_eq!(
                    categories.iter().map(String::as_str).collect::<Vec<_>>(),
                    *expected,
                    "incorrect authored mod slots for {}",
                    weapon.name
                );
                checked_weapon_slots.insert(weapon.name.as_str());
            }
        }
        assert!(
            overflowing_weapon_slots.is_empty(),
            "weapons with more than four mod slots: {overflowing_weapon_slots:#?}"
        );
        let missing_slot_audits = expected_weapon_slots
            .keys()
            .filter(|weapon| !checked_weapon_slots.contains(**weapon))
            .copied()
            .sorted()
            .collect::<Vec<_>>();
        assert!(
            missing_slot_audits.is_empty(),
            "expected weapons missing from slot audit: {missing_slot_audits:?}"
        );

        let weapon_names = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .map(|item| item.name.as_str())
            .collect::<FxHashSet<_>>();
        for item in &weapon_mods {
            assert_eq!(item.subcategory, item.mod_category);
            assert!(
                item.compatible_weapons
                    .iter()
                    .all(|weapon| weapon_names.contains(weapon.as_str())),
                "unknown compatible weapon on {item:?}"
            );
            assert!(
                item.compatible_weapons.windows(2).all(|pair| {
                    pair[0].to_lowercase() < pair[1].to_lowercase()
                        || (pair[0].eq_ignore_ascii_case(&pair[1]) && pair[0] < pair[1])
                }),
                "compatible weapons must be sorted and unique: {item:?}"
            );
        }

        for (name, expected_weapons) in [
            (
                "Daredevil Stock",
                vec!["D54 Battle Pistol".to_owned(), "KKV-9SD".to_owned()],
            ),
            ("Flechette Drum", vec!["KKV-9SD".to_owned()]),
            ("Full-Auto Selector", vec!["Misriah 2442".to_owned()]),
            ("Adrenal Feedback Rounds", vec!["Hardline PR".to_owned()]),
            ("Stopping Mag", vec!["Impact H-AR".to_owned()]),
            (
                "Overclocked Delimiter",
                vec!["V85 Circuit Breaker".to_owned()],
            ),
            ("Overclocked Generator", vec!["V00 ZEUS RG".to_owned()]),
            ("Charge-Coupled Optic", vec!["V99 Channel Rifle".to_owned()]),
            ("Pressurized Coolant", vec!["V99 Watchtower".to_owned()]),
        ] {
            let matching = weapon_mods
                .iter()
                .filter(|item| item.name == name)
                .collect::<Vec<_>>();
            assert!(!matching.is_empty(), "missing weapon mod {name}");
            assert!(
                matching
                    .iter()
                    .all(|item| item.compatible_weapons == expected_weapons),
                "incorrect exact compatibility for {name}: {matching:?}"
            );
        }
        assert!(
            weapon_mods.iter().all(|item| item.rarity.is_some()),
            "all weapon mods must decode their footer rarity"
        );
        for (price, rarity) in [
            (69, GearRarity::Enhanced),
            (207, GearRarity::Deluxe),
            (621, GearRarity::Superior),
        ] {
            assert!(weapon_mods.iter().any(|item| {
                item.name == "Accu-Sight Optic"
                    && item.price == Some(price)
                    && item.rarity == Some(rarity)
            }));
        }

        let weapon_skins = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Skin"))
            .collect::<Vec<_>>();
        let missing_weapon = weapon_skins
            .iter()
            .filter(|item| item.applies_to.is_none())
            .collect::<Vec<_>>();
        assert!(
            missing_weapon.is_empty(),
            "weapon skins without an exact owner: {missing_weapon:?}"
        );
        let weapon_subcategories = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon"))
            .filter_map(|item| Some((item.name.as_str(), item.subcategory.as_deref()?)))
            .collect::<FxHashMap<_, _>>();
        for skin in &weapon_skins {
            let Some((skin_subcategory, weapon)) =
                skin.subcategory.as_deref().zip(skin.applies_to.as_deref())
            else {
                continue;
            };
            let Some(weapon_subcategory) = weapon_subcategories.get(weapon) else {
                continue;
            };
            assert!(
                skin_subcategory.eq_ignore_ascii_case(weapon_subcategory),
                "skin subtype disagrees with resolved weapon subtype: skin={skin:?}, weapon subtype={weapon_subcategory}"
            );
        }
        for skin in &weapon_skins {
            let Some((owner, _)) = skin
                .internal_name
                .as_deref()
                .and_then(|name| name.split_once(".skins."))
            else {
                continue;
            };
            let Some(group) = skin.definition_group_index().filter(|group| *group >= 18) else {
                continue;
            };
            assert_eq!(
                skin_group_weapon_hash(group),
                Some(weapon_internal_hash(owner)),
                "skin group {group} is mapped to the wrong weapon for {owner}"
            );
        }

        for (skin_hash, weapon) in [
            (D54_DEFAULT_SKIN_HASH, "D54 Battle Pistol"),
            (BIOTOXIC_DEFAULT_SKIN_HASH, "Biotoxic Disinjector"),
            (BIOTOXIC_SHADOW_INDEX_SKIN_HASH, "Biotoxic Disinjector"),
            (ACID_ABYSS_SKIN_HASH, "Stryder M1T"),
            (CRYO_SHIFT_V11_SKIN_HASH, "V11 Punch"),
        ] {
            let skin = items
                .iter()
                .find(|item| item.internal_hash == Some(skin_hash))
                .expect("known hash-only skin");
            assert_eq!(skin.applies_to.as_deref(), Some(weapon));
        }
        let corrected_implants = items
            .iter()
            .filter(|item| item.definition_type_code == Some(0x19c))
            .collect::<Vec<_>>();
        assert_eq!(corrected_implants.len(), 8);
        assert!(corrected_implants.iter().all(|item| {
            item.item_type.as_deref() == Some("Implant")
                && !item.types.iter().any(|kind| kind.contains("trinket"))
        }));

        let unnamed = items
            .iter()
            .filter(|item| item.name.starts_with("Unnamed "))
            .collect::<Vec<_>>();
        assert_eq!(
            unnamed.len(),
            15,
            "only records with no source string should use an unnamed placeholder: {unnamed:?}"
        );

        let salvage_names = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Salvage"))
            .map(|item| item.name.as_str())
            .collect::<FxHashSet<_>>();
        assert!(
            salvage_names.len() >= 40,
            "salvage localization still collapsed: {} names",
            salvage_names.len()
        );
        for collapsed_name in ["Dermachem Pack", "Thoughtwave Lens"] {
            let collapsed = items
                .iter()
                .filter(|item| {
                    item.item_type.as_deref() == Some("Salvage") && item.name == collapsed_name
                })
                .map(|item| item.internal_name.as_deref())
                .collect::<Vec<_>>();
            assert_eq!(
                collapsed.len(),
                1,
                "{collapsed_name} still appears for {collapsed:?} after resolution"
            );
        }
    }
    #[test]
    #[ignore = "probe: requires a local Marathon package installation"]
    fn probes_atrax_sting_inventory_dye_links() {
        use quicktag_core::classes::get_class_by_id;

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("gear");
        let cache = quicktag_scanner::load_tag_cache();
        let item = items
            .iter()
            .find(|item| item.model_tag == Some(TagHash(0x80B6D750)))
            .expect("Atrax Sting inventory item");
        eprintln!(
            "ATRAX_ITEM name={} display={} definition={:?} internal_hash={:?} internal={:?} group={:?} type={:?}",
            item.name,
            item.display_tag,
            item.definition_tag,
            item.internal_hash,
            item.internal_name,
            item.definition_group_key,
            item.definition_type_code
        );
        let internal_hash = item.internal_hash.expect("Atrax API hash");
        for tag in [
            TagHash(0x80A66100),
            TagHash(0x80A66102),
            TagHash(0x80A66138),
            TagHash(0x80A66199),
            TagHash(0x80A661BF),
            TagHash(0x80A661E6),
            TagHash(0x80A66229),
        ] {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            let offsets = data
                .chunks_exact(4)
                .enumerate()
                .filter_map(|(index, bytes)| {
                    (u32::from_le_bytes(bytes.try_into().ok()?) == internal_hash)
                        .then_some(index * 4)
                })
                .collect_vec();
            if offsets.is_empty() {
                continue;
            }
            let reference = package_manager()
                .get_entry(tag)
                .map(|entry| entry.reference)
                .unwrap_or_default();
            let class = get_class_by_id(reference)
                .map(|class| class.name.into_owned())
                .unwrap_or_default();
            for offset in offsets {
                let start = offset.saturating_sub(0x40);
                let end = (offset + 0x80).min(data.len());
                eprintln!(
                    "ATRAX_HASH_LINK tag={tag} reference={reference:08X} class={class} offset=0x{offset:X} context={:?}",
                    data[start..end]
                        .chunks_exact(4)
                        .enumerate()
                        .map(|(index, bytes)| format!(
                            "+{:X}={:08X}",
                            start + index * 4,
                            u32::from_le_bytes(bytes.try_into().unwrap())
                        ))
                        .collect_vec()
                );
            }
        }
        let mut wrapper_frontier = vec![TagHash(0x80AA2336)];
        let mut wrapper_seen = FxHashSet::default();
        for depth in 0..10 {
            let mut next = vec![];
            for tag in wrapper_frontier.drain(..) {
                if !wrapper_seen.insert(tag) {
                    continue;
                }
                let reference = package_manager()
                    .get_entry(tag)
                    .map(|entry| entry.reference)
                    .unwrap_or_default();
                let refs = cache
                    .hashes
                    .get(&tag)
                    .into_iter()
                    .flat_map(|scan| scan.file_hashes.iter().map(|reference| reference.hash))
                    .unique()
                    .collect_vec();
                let data = package_manager().read_tag(tag).unwrap_or_default();
                let wide_refs = (0..data.len().saturating_sub(7))
                    .step_by(8)
                    .filter_map(|offset| {
                        resolve_direct_tag64(&data, offset).map(|tag| (offset, tag))
                    })
                    .unique_by(|(_offset, tag)| *tag)
                    .collect_vec();
                eprintln!(
                    "ATRAX_WRAPPER depth={depth} tag={tag} reference={reference:08X} refs={:?} wide_refs={wide_refs:?} words={:?}",
                    refs.iter()
                        .map(|child| (
                            *child,
                            package_manager()
                                .get_entry(*child)
                                .map(|entry| entry.reference)
                                .unwrap_or_default()
                        ))
                        .collect_vec(),
                    data.chunks_exact(4)
                        .take(64)
                        .map(|bytes| format!(
                            "{:08X}",
                            u32::from_le_bytes(bytes.try_into().unwrap())
                        ))
                        .collect_vec()
                );
                next.extend(refs);
                next.extend(wide_refs.into_iter().map(|(_offset, tag)| tag));
            }
            wrapper_frontier = next;
        }
        let mut frontier = item.definition_tag.into_iter().collect_vec();
        let mut seen = FxHashSet::default();
        for depth in 0..12 {
            let mut next = vec![];
            for tag in frontier.drain(..) {
                if !seen.insert(tag) {
                    continue;
                }
                let Some(entry) = package_manager().get_entry(tag) else {
                    continue;
                };
                let class = get_class_by_id(entry.reference)
                    .map(|class| class.name.into_owned())
                    .unwrap_or_else(|| format!("{:08X}", entry.reference));
                let scan = cache.hashes.get(&tag);
                let data = package_manager().read_tag(tag).unwrap_or_default();
                let wanted = [
                    0xC8939EBF_u32,
                    0xC8939EBA,
                    0xC8939EBC,
                    0xC8939EBD,
                    0xC8939EBB,
                    0xC8939EB8,
                ];
                let words = data
                    .chunks_exact(4)
                    .enumerate()
                    .filter_map(|(index, bytes)| {
                        let value = u32::from_le_bytes(bytes.try_into().ok()?);
                        wanted
                            .contains(&value)
                            .then(|| format!("{value:08X}@{:X}", index * 4))
                    })
                    .collect_vec();
                eprintln!("ATRAX_LINK depth={depth} tag={tag} class={class} words={words:?}");
                next.extend(
                    scan.into_iter()
                        .flat_map(|scan| scan.file_hashes.iter().map(|reference| reference.hash)),
                );
            }
            frontier = next;
        }
    }

    #[test]
    #[ignore = "probe: requires a local Marathon package installation"]
    fn probes_marathon_weapon_mod_model_links() {
        use itertools::Itertools;
        use quicktag_core::classes::get_class_by_id;

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("gear");
        let cache = quicktag_scanner::load_tag_cache();
        let mods = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .collect_vec();
        eprintln!("weapon mods: {}", mods.len());

        for item in mods {
            let mut reachable = vec![];
            let mut frontier = item.definition_tag.into_iter().collect_vec();
            let mut seen = FxHashSet::default();
            for _depth in 0..8 {
                let mut next = vec![];
                for parent in frontier.drain(..) {
                    if !seen.insert(parent) {
                        continue;
                    }
                    let Some(entry) = package_manager().get_entry(parent) else {
                        continue;
                    };
                    let class = get_class_by_id(entry.reference)
                        .map(|class| class.name.into_owned())
                        .unwrap_or_else(|| format!("{:08X}", entry.reference));
                    if matches!(
                        class.as_str(),
                        "s_pattern" | "s_pattern_component" | "s_geometry_resource"
                    ) {
                        reachable.push((parent, class.clone()));
                    }
                    if let Some(scan) = cache.hashes.get(&parent) {
                        next.extend(scan.file_hashes.iter().map(|reference| reference.hash));
                    }
                }
                frontier = next;
            }

            let direct = item
                .definition_tag
                .and_then(|tag| cache.hashes.get(&tag))
                .into_iter()
                .flat_map(|scan| scan.file_hashes.iter())
                .map(|reference| {
                    let class = package_manager()
                        .get_entry(reference.hash)
                        .map(|entry| entry.reference)
                        .unwrap_or_default();
                    format!("{}@0x{:X}:{class:08X}", reference.hash, reference.offset)
                })
                .collect_vec();
            eprintln!(
                "{} display={} def={:?} internal={:?} model={} types={:?}\n  direct={direct:?}\n  models={reachable:?}",
                item.name,
                item.display_tag,
                item.definition_tag,
                item.internal_name,
                item.model_tag
                    .map_or_else(|| "—".to_owned(), |tag| tag.to_string()),
                item.types,
            );
        }
    }

    #[test]
    #[ignore = "probe: requires a local Marathon package installation"]
    fn probes_marathon_weapon_mod_hash_bridges() {
        use quicktag_core::{classes::get_class_by_id, util::fnv1};

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("gear");
        let cache = quicktag_scanner::load_tag_cache();
        let wanted = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .filter_map(|item| {
                let name = item.internal_name.as_deref()?;
                (!name.starts_with('#')).then_some((fnv1(name.as_bytes()), item))
            })
            .collect::<FxHashMap<_, _>>();
        let mut hits = 0usize;
        let mut hit_tags: FxHashMap<(TagHash, u32), usize> = FxHashMap::default();
        for (tag, scan) in &cache.hashes {
            for occurrence in &scan.wordlist_hashes {
                let Some(item) = wanted.get(&occurrence.hash) else {
                    continue;
                };
                if Some(*tag) == item.definition_tag || *tag == item.display_tag {
                    continue;
                }
                let reference = package_manager()
                    .get_entry(*tag)
                    .map(|entry| entry.reference)
                    .unwrap_or_default();
                let class = get_class_by_id(reference)
                    .map(|class| class.name.into_owned())
                    .unwrap_or_else(|| format!("{reference:08X}"));
                *hit_tags.entry((*tag, reference)).or_default() += 1;
                if *tag != TagHash(0x80A66199) {
                    eprintln!(
                        "{} hash={:08X} occurrence={}@0x{:X} class={class}",
                        item.internal_name.as_deref().unwrap_or_default(),
                        occurrence.hash,
                        tag,
                        occurrence.offset,
                    );
                }
                hits += 1;
            }
        }
        let mut summaries = hit_tags.into_iter().collect::<Vec<_>>();
        summaries.sort_by_key(|((tag, _class), count)| (std::cmp::Reverse(*count), *tag));
        eprintln!("non-inventory mod hash bridge hits: {hits}; tags={summaries:?}");

        let bridge = package_manager()
            .read_tag(TagHash(0x80A66199))
            .expect("mod bridge table");
        for name in [
            "weapon_mods.foregrips.rifle.v100.textured0",
            "weapon_mods.optics.rifle.v100.toggle0",
            "weapon_mods.muzzles.base.v100.extender0",
            "weapon_mods.magazines.rifle.v100.drum0",
        ] {
            let hash = fnv1(name.as_bytes());
            let offsets = bridge
                .windows(4)
                .enumerate()
                .filter(|(offset, bytes)| {
                    offset % 4 == 0 && u32::from_le_bytes((*bytes).try_into().unwrap()) == hash
                })
                .map(|(offset, _bytes)| offset)
                .collect::<Vec<_>>();
            for offset in offsets {
                let before = offset.saturating_sub(16);
                let after = (offset + 24).min(bridge.len());
                eprintln!(
                    "bridge {name} {hash:08X}@0x{offset:X}: {:02X?}",
                    &bridge[before..after]
                );
            }
        }

        let records = package_manager()
            .read_tag(TagHash(0x80A66102))
            .expect("mod records");
        let mut words = FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            words.entry(hash).or_insert_with(|| word.to_owned());
        });
        for name in [
            "weapon_mods.foregrips.rifle.v100.textured0",
            "weapon_mods.foregrips.rifle.v100.textured1",
            "weapon_mods.optics.rifle.v100.toggle0",
            "weapon_mods.muzzles.base.v100.extender0",
            "weapon_mods.magazines.rifle.v100.drum0",
        ] {
            let hash = fnv1(name.as_bytes());
            for (offset, bytes) in records.windows(4).enumerate().filter(|(offset, bytes)| {
                offset % 4 == 0 && u32::from_le_bytes((*bytes).try_into().unwrap()) == hash
            }) {
                let _ = bytes;
                let start = offset;
                let end = (start + 0x38).min(records.len());
                let fields = records[start..end]
                    .chunks_exact(4)
                    .enumerate()
                    .map(|(index, field)| {
                        let value = u32::from_le_bytes(field.try_into().unwrap());
                        let meaning = words
                            .get(&value)
                            .cloned()
                            .or_else(|| {
                                package_manager().get_entry(TagHash(value)).map(|entry| {
                                    let class = get_class_by_id(entry.reference)
                                        .map(|class| class.name.into_owned())
                                        .unwrap_or_else(|| format!("{:08X}", entry.reference));
                                    format!("tag:{class}")
                                })
                            })
                            .unwrap_or_default();
                        format!("+{:02X}={value:08X}:{meaning}", index * 4)
                    })
                    .collect::<Vec<_>>();
                eprintln!("record {name} hash@0x{offset:X} start=0x{start:X}: {fields:?}");
            }
        }
    }

    #[test]
    #[ignore = "probe: requires a local Marathon package installation"]
    fn probes_marathon_weapon_mod_visual_family_hashes() {
        use quicktag_core::util::fnv1;

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("gear");
        let mut candidates: FxHashMap<u32, FxHashSet<String>> = FxHashMap::default();
        for name in items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .filter_map(|item| item.internal_name.as_deref())
            .filter(|name| !name.starts_with('#'))
        {
            let mut variants = vec![name.to_owned()];
            if name.ends_with(['0', '1', '2']) {
                variants.push(name[..name.len() - 1].to_owned());
            }
            let parts = name.split('.').collect::<Vec<_>>();
            for end in 2..parts.len() {
                variants.push(parts[..end].join("."));
            }
            if let Some(leaf) = parts.last() {
                variants.push(leaf.trim_end_matches(['0', '1', '2']).to_owned());
            }
            for candidate in variants {
                candidates
                    .entry(fnv1(candidate.as_bytes()))
                    .or_default()
                    .insert(candidate);
            }
        }
        for item in items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
        {
            if let Some(hash) = item.internal_hash {
                candidates
                    .entry(hash)
                    .or_default()
                    .insert(format!("internal:{:08X}", item.display_tag.0));
            }
        }

        let mut hits = vec![];
        for class in [0x8080BAAD_u32, 0x8080BADB, 0x8080BA53] {
            for (tag, _entry) in package_manager().get_all_by_reference(class) {
                let Ok(data) = package_manager().read_tag(tag) else {
                    continue;
                };
                for offset in (0..data.len().saturating_sub(3)).step_by(4) {
                    let value = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    if let Some(names) = candidates.get(&value) {
                        hits.push((tag, class, offset, value, names.clone()));
                    }
                }
            }
        }
        hits.sort_by_key(|(tag, class, offset, _hash, _names)| (*class, *tag, *offset));
        eprintln!("visual family hits: {}", hits.len());
        for hit in hits {
            eprintln!("  {hit:?}");
        }
    }

    #[test]
    #[ignore = "probe: requires a local Marathon package installation"]
    fn probes_marathon_weapon_mod_internal_hash_bridges_all_tags() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("gear");
        let wanted = items
            .iter()
            .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
            .filter_map(|item| Some((item.internal_hash?, item.display_tag)))
            .collect::<FxHashMap<_, _>>();
        let cache = quicktag_scanner::load_tag_cache();
        let mut summaries: FxHashMap<(TagHash, u32), (usize, Vec<(usize, u32, TagHash)>)> =
            FxHashMap::default();
        for tag in cache.hashes.keys().copied() {
            let Some(entry) = package_manager().get_entry(tag) else {
                continue;
            };
            if entry.file_type != 8 || entry.file_subtype != 0 || entry.file_size > 1024 * 1024 {
                continue;
            }
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            for offset in (0..data.len().saturating_sub(3)).step_by(4) {
                let value = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let Some(display) = wanted.get(&value).copied() else {
                    continue;
                };
                let summary = summaries.entry((tag, entry.reference)).or_default();
                summary.0 += 1;
                if summary.1.len() < 8 {
                    summary.1.push((offset, value, display));
                }
            }
        }
        let mut summaries = summaries.into_iter().collect::<Vec<_>>();
        summaries.sort_by_key(|((tag, class), (count, _examples))| {
            (std::cmp::Reverse(*count), *class, *tag)
        });
        eprintln!("internal bridge tags: {}", summaries.len());
        for summary in summaries {
            eprintln!("  {summary:?}");
        }
    }

    #[test]
    #[ignore = "probe: requires a local Marathon package installation"]
    fn probes_quickdraw_grip_rarity_visual_parameters() {
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = tiger_pkg::PackageManager::new(
            packages,
            tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();

        let strings = quicktag_strings::localized::create_stringmap().expect("localized strings");
        let items = load_gear(&strings).expect("gear");
        for item in items.iter().filter(|item| {
            matches!(
                item.name.as_str(),
                "Retro_Remix"
                    | "Vox Nocturna"
                    | "Precision Choke"
                    | "Full-Auto Selector"
                    | "Extra Mag III"
            )
        }) {
            eprintln!(
                "Reference item {:?} owner={:?} rarity={:?} display={} definition={:?} model={:?}",
                item.name,
                item.applies_to,
                item.rarity,
                item.display_tag,
                item.definition_tag,
                item.model_tag
            );
        }
        let mut quickdraw = items
            .iter()
            .filter(|item| item.name == "Quickdraw Grip")
            .collect::<Vec<_>>();
        quickdraw.sort_by_key(|item| (rarity_sort_key(item.rarity), item.definition_tag));
        eprintln!("Quickdraw Grip records: {}", quickdraw.len());
        for item in &quickdraw {
            eprintln!(
                "  rarity={:?} price={:?} display={} definition={:?} internal={:?}/{:?} model={:?} categories={:?}",
                item.rarity,
                item.price,
                item.display_tag,
                item.definition_tag,
                item.internal_hash.map(|hash| format!("{hash:08X}")),
                item.internal_name,
                item.model_tag,
                item.internal_categories,
            );
        }

        let definitions = quickdraw
            .iter()
            .filter_map(|item| {
                Some((
                    item.rarity?,
                    item.definition_tag?,
                    package_manager().read_tag(item.definition_tag?).ok()?,
                ))
            })
            .collect::<Vec<_>>();

        let pattern_resolver = InvestmentPatternResolver::load();
        for (rarity, definition_tag, definition) in &definitions {
            let translation = definition
                .chunks_exact(4)
                .position(|bytes| {
                    u32::from_le_bytes(bytes.try_into().unwrap())
                        == PATTERN_TRANSLATION_BLOCK_MARKER
                })
                .map(|index| index * 4)
                .expect("pattern translation block");
            let pattern_index =
                read_u16(definition, translation + 0x6c).expect("pattern-global table index");
            let global_id = pattern_resolver.pattern_globals[usize::from(pattern_index)];
            let pattern = pattern_resolver.assignments.get(&global_id).copied();
            eprintln!(
                "  pattern route rarity={rarity:?} definition={definition_tag} index={pattern_index} global={global_id:08X} pattern={pattern:?}"
            );
            if let Some(pattern) = pattern {
                let entry = package_manager().get_entry(pattern).expect("pattern entry");
                let data = package_manager().read_tag(pattern).expect("pattern data");
                eprintln!(
                    "    pattern class={:08X} len=0x{:X} words={:?}",
                    entry.reference,
                    data.len(),
                    data.chunks_exact(4)
                        .take(48)
                        .map(|bytes| format!(
                            "{:08X}",
                            u32::from_le_bytes(bytes.try_into().unwrap())
                        ))
                        .collect::<Vec<_>>()
                );
            }
        }
        for pair in definitions.windows(2) {
            let (left_rarity, left_tag, left) = &pair[0];
            let (right_rarity, right_tag, right) = &pair[1];
            let differences = left
                .iter()
                .zip(right)
                .enumerate()
                .filter(|(_offset, (left, right))| left != right)
                .map(|(offset, (left, right))| (offset, *left, *right))
                .collect::<Vec<_>>();
            eprintln!(
                "  definition diff {left_rarity:?} {left_tag} -> {right_rarity:?} {right_tag}: lengths {} / {}, bytes {}",
                left.len(),
                right.len(),
                differences.len(),
            );
            for group in differences.chunk_by(|left, right| left.0 + 1 == right.0) {
                let start = group[0].0;
                let end = group.last().unwrap().0 + 1;
                eprintln!(
                    "    0x{start:04X}..0x{end:04X}: {:02X?} -> {:02X?}",
                    &left[start..end],
                    &right[start..end],
                );
            }
        }

        use quicktag_core::util::fnv1;
        let authored_hashes = quickdraw
            .iter()
            .filter_map(|item| {
                let name = item.internal_name.as_deref()?;
                Some((fnv1(name.as_bytes()), name.to_owned()))
            })
            .collect::<FxHashMap<_, _>>();
        let cache = quicktag_scanner::load_tag_cache();
        for (tag, scan) in &cache.hashes {
            for occurrence in &scan.wordlist_hashes {
                if let Some(name) = authored_hashes.get(&occurrence.hash) {
                    eprintln!(
                        "  authored name ref {name}={:08X} tag={tag} class={:08X} offset=0x{:X}",
                        occurrence.hash,
                        package_manager()
                            .get_entry(*tag)
                            .map(|entry| entry.reference)
                            .unwrap_or_default(),
                        occurrence.offset,
                    );
                }
            }
        }

        let inventory_hashes = quickdraw
            .iter()
            .filter_map(|item| {
                item.internal_hash
                    .map(|hash| (hash, item.internal_name.clone()))
            })
            .collect::<FxHashMap<_, _>>();
        for class in [0x8080BAAD_u32, 0x8080BADB, 0x8080BA53] {
            for (tag, _entry) in package_manager().get_all_by_reference(class) {
                let Ok(data) = package_manager().read_tag(tag) else {
                    continue;
                };
                for offset in (0..data.len().saturating_sub(3)).step_by(4) {
                    let value = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    if let Some(name) = inventory_hashes.get(&value) {
                        eprintln!(
                            "  inventory hash ref {:?}={value:08X} tag={tag} class={class:08X} offset=0x{offset:X}",
                            name
                        );
                    }
                }
            }
        }

        for tag in [TagHash(0x80A66199), TagHash(0x80A66102)] {
            let data = package_manager()
                .read_tag(tag)
                .expect("weapon mod visual table");
            for (hash, name) in &authored_hashes {
                for (offset, _) in data.windows(4).enumerate().filter(|(offset, bytes)| {
                    offset % 4 == 0 && u32::from_le_bytes((*bytes).try_into().unwrap()) == *hash
                }) {
                    let start = offset.saturating_sub(0x10) & !3;
                    let end = (offset + 0x50).min(data.len()) & !3;
                    eprintln!(
                        "  visual table {tag} {name}={hash:08X}@0x{offset:X}: {:?}",
                        data[start..end]
                            .chunks_exact(4)
                            .map(|bytes| format!(
                                "{:08X}",
                                u32::from_le_bytes(bytes.try_into().unwrap())
                            ))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }

        let mut words = FxHashMap::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            words.entry(hash).or_insert_with(|| word.to_owned());
        });
        for tag in [
            TagHash(0x80AA2213),
            TagHash(0x80A66138),
            TagHash(0x80A66229),
        ] {
            let data = package_manager()
                .read_tag(tag)
                .expect("weapon mod rarity table");
            for (hash, name) in &authored_hashes {
                for (offset, _) in data.windows(4).enumerate().filter(|(offset, bytes)| {
                    offset % 4 == 0 && u32::from_le_bytes((*bytes).try_into().unwrap()) == *hash
                }) {
                    let start = offset.saturating_sub(0x20) & !3;
                    let end = (offset + 0x80).min(data.len()) & !3;
                    let fields = data[start..end]
                        .chunks_exact(4)
                        .enumerate()
                        .map(|(index, bytes)| {
                            let value = u32::from_le_bytes(bytes.try_into().unwrap());
                            let meaning = words
                                .get(&value)
                                .cloned()
                                .or_else(|| {
                                    package_manager()
                                        .get_entry(TagHash(value))
                                        .map(|entry| format!("tag-class:{:08X}", entry.reference))
                                })
                                .unwrap_or_default();
                            format!(
                                "{:+04X}={value:08X}/{}:{meaning}",
                                index as isize * 4 + start as isize - offset as isize,
                                f32::from_bits(value),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!("  rarity table {tag} {name}={hash:08X}@0x{offset:X}: {fields:?}");
                }
            }
        }
    }
}

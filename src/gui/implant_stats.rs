use rustc_hash::FxHashMap;
use tiger_pkg::{TagHash, package_manager};

use quicktag_core::implant::ImplantStatResolver;
use quicktag_strings::localized::{LocalizedLanguage, LocalizedStringResolver};

const ICON_DEFINITION_REFERENCE: u32 = 0x8080_5335;
const ICON_CONTAINER_REFERENCE: u32 = 0x8080_5350;
const PATTERN_GLOBAL_REFERENCE: u32 = 0x8080_6CAC;
const PATTERN_TRANSLATION_CLASS: u32 = 0x8080_91D3;
const ICON_PROPERTY: u32 = 0x4D77_84B8;
const PATTERN_GLOBAL_STRIDE: usize = 0x48;

#[derive(Clone, Debug, Default)]
pub(super) struct ImplantDetails {
    pub(super) effects: Vec<super::item_effect::ItemEffect>,
    pub(super) stats: Vec<ImplantStat>,
}

#[derive(Clone, Debug)]
pub(super) struct ImplantStat {
    pub(super) name: String,
    pub(super) raw_name_hash: u32,
    pub(super) value: i32,
}

pub(super) struct ImplantStatsResolver {
    resolver: Option<ImplantStatResolver>,
    names: FxHashMap<u32, String>,
}

impl ImplantStatsResolver {
    pub(super) fn load(language: LocalizedLanguage, localized: &LocalizedStringResolver) -> Self {
        let manager = package_manager();
        let resolver = ImplantStatResolver::load(&manager)
            .inspect_err(|error| log::warn!("Failed to load implant stat tables: {error:#}"))
            .ok();
        let english = (language != LocalizedLanguage::English)
            .then(|| {
                quicktag_strings::localized::create_stringresolver_d2_for_language(
                    LocalizedLanguage::English,
                )
            })
            .transpose()
            .inspect_err(|error| log::warn!("Failed to load English stat labels: {error:#}"))
            .ok()
            .flatten();
        let labels = STAT_LABELS
            .iter()
            .map(|(_, label)| *label)
            .collect::<Vec<_>>();
        let translated = english
            .as_ref()
            .and_then(|source| localized.translate_group_from(source, &labels));
        let names = STAT_LABELS
            .iter()
            .enumerate()
            .map(|(index, &(hash, label))| {
                let name = translated
                    .as_ref()
                    .map(|names| names[index].clone())
                    .unwrap_or_else(|| label.to_owned());
                (hash, name)
            })
            .collect();
        Self { resolver, names }
    }

    pub(super) fn resolve(&self, definition: TagHash) -> Vec<ImplantStat> {
        let Some(resolver) = &self.resolver else {
            return vec![];
        };
        let decoded = package_manager()
            .read_tag(definition)
            .and_then(|data| resolver.resolve(&data));
        match decoded {
            Ok(decoded) => decoded
                .stats
                .into_iter()
                .map(|stat| ImplantStat {
                    name: self
                        .names
                        .get(&stat.raw_name_hash)
                        .cloned()
                        .unwrap_or_else(|| stat_display_name(stat.raw_name_hash)),
                    raw_name_hash: stat.raw_name_hash,
                    value: stat.value,
                })
                .collect(),
            Err(error) => {
                log::warn!("Failed to decode implant stats for {definition}: {error:#}");
                vec![]
            }
        }
    }
}

/// Presentation vocabulary is global per stat semantic, never implant-specific.
/// Unknown future semantics remain lossless through their package hash.
const STAT_LABELS: &[(u32, &str)] = &[
    (0x9B56_2AEF, "Heat Capacity"),
    (0x168C_E416, "Agility"),
    (0xF7BE_E2FB, "Loot Speed"),
    (0x661E_F6CF, "Melee Damage"),
    (0xA70A_0BB4, "Prime Recovery"),
    (0xE384_9468, "Tactical Recovery"),
    (0x0CD9_B0AC, "Self-Repair Speed"),
    (0x3EF0_19B9, "Finisher Siphon"),
    (0x11F3_089C, "Revive Speed"),
    (0xF5CD_B4B5, "Hardware"),
    (0xB91C_4CE4, "Firewall"),
    (0x299A_0E03, "Fall Resistance"),
    (0xC0D5_35D4, "Ping Duration"),
];

fn stat_display_name(raw_name_hash: u32) -> String {
    STAT_LABELS
        .iter()
        .find(|(hash, _)| *hash == raw_name_hash)
        .map(|(_, label)| (*label).to_owned())
        .unwrap_or_else(|| {
            super::get_string_for_hash(raw_name_hash)
                .map(|name| humanize_stat_name(&name))
                .unwrap_or_else(|| format!("Stat #{raw_name_hash:08X}"))
        })
}

fn humanize_stat_name(name: &str) -> String {
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) struct ImplantIconResolver {
    textures: FxHashMap<u32, TagHash>,
    presentation_icons: Vec<Option<u32>>,
}

impl ImplantIconResolver {
    pub(super) fn load() -> Self {
        let mut definitions = package_manager().get_all_by_reference(ICON_DEFINITION_REFERENCE);
        definitions.sort_unstable_by_key(|(tag, _)| *tag);
        let textures = definitions
            .into_iter()
            .filter_map(|(definition, _)| {
                let data = package_manager().read_tag(definition).ok()?;
                let identifier = read_u32(&data, 0x10)?;
                let container = TagHash(read_u32(&data, 0x18)?);
                (package_manager().get_entry(container)?.reference == ICON_CONTAINER_REFERENCE)
                    .then_some(())?;
                let data = package_manager().read_tag(container).ok()?;
                let texture = TagHash(read_u32(&data, 0x90)?);
                package_manager().get_entry(texture)?;
                Some((identifier, texture))
            })
            .collect();
        let presentation_icons = package_manager()
            .get_all_by_reference(PATTERN_GLOBAL_REFERENCE)
            .into_iter()
            .filter_map(|(tag, _)| {
                let data = package_manager().read_tag(tag).ok()?;
                parse_presentation_icons(&data)
            })
            .max_by_key(Vec::len)
            .unwrap_or_default();
        Self {
            textures,
            presentation_icons,
        }
    }

    /// The translation carries two adjacent u16 selectors: model at +0x6C,
    /// presentation at +0x6E (relative to the class marker). The latter indexes
    /// PatternGlobal; its property array supplies the authored icon identifier.
    pub(super) fn resolve_implant(&self, definition: TagHash) -> Option<TagHash> {
        let data = package_manager().read_tag(definition).ok()?;
        let index = presentation_index(&data)?;
        let identifier = self.presentation_icons.get(index).copied().flatten()?;
        self.textures.get(&identifier).copied()
    }
}

fn presentation_index(data: &[u8]) -> Option<usize> {
    let marker = data.chunks_exact(4).position(|bytes| {
        u32::from_le_bytes(bytes.try_into().unwrap()) == PATTERN_TRANSLATION_CLASS
    })? * 4;
    let bytes = data.get(marker + 0x6E..marker + 0x70)?;
    let index = u16::from_le_bytes(bytes.try_into().ok()?);
    (index != u16::MAX).then_some(usize::from(index))
}

fn parse_presentation_icons(data: &[u8]) -> Option<Vec<Option<u32>>> {
    let rows = table_range(data, 8, PATTERN_GLOBAL_STRIDE)?;
    Some(
        (rows.start..rows.end)
            .step_by(PATTERN_GLOBAL_STRIDE)
            .map(|row| {
                let properties = table_range(data, row + 0x28, 8)?;
                data[properties].chunks_exact(8).find_map(|property| {
                    (read_u32(property, 0)? == ICON_PROPERTY)
                        .then(|| read_u32(property, 4))
                        .flatten()
                })
            })
            .collect(),
    )
}

fn table_range(data: &[u8], header: usize, stride: usize) -> Option<std::ops::Range<usize>> {
    let count = usize::try_from(u64::from_le_bytes(
        data.get(header..header + 8)?.try_into().ok()?,
    ))
    .ok()?;
    let pointer = header.checked_add(8)?;
    let relative = i64::from_le_bytes(data.get(pointer..pointer + 8)?.try_into().ok()?);
    let start = usize::try_from(
        i64::try_from(pointer)
            .ok()?
            .checked_add(relative)?
            .checked_add(0x10)?,
    )
    .ok()?;
    let end = start.checked_add(count.checked_mul(stride)?)?;
    (end <= data.len()).then_some(start..end)
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

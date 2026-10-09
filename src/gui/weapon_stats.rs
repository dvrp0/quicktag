use rustc_hash::FxHashMap;
use tiger_pkg::{TagHash, package_manager};

#[path = "weapon_mod_stats.rs"]
mod weapon_mod_stats;
#[path = "weapon_stat_expressions.rs"]
mod weapon_stat_expressions;
pub(super) use weapon_mod_stats::{
    WeaponBaseRating, WeaponModDetails, WeaponModRawStat, WeaponModStatChange,
    WeaponModWeaponStats,
};
use weapon_stat_expressions::PropertyProgram;
#[path = "weapon_stat_display.rs"]
mod weapon_stat_display;
use weapon_stat_display::StatDisplayPrograms;

const ARRAY_MARKER: u32 = 0x8080_bfcd;
const STAT_RATING_ARRAY: u32 = 0x8080_924b;
const CURVE_DESCRIPTOR_ARRAY: u32 = 0x8080_bc4d;
const CURVE_VALUE_ARRAY: u32 = 0x8080_000f;
const CURVE_SEMANTIC_ARRAY: u32 = 0x8080_3f8e;
const SOCKET_ARRAY: u32 = 0x8080_6919;
const GAMEPLAY_COMPONENT_REFERENCE: u32 = 0x8080_badb;
const PATTERN_REFERENCE: u32 = 0x8080_baad;
const PATTERN_ASSIGNMENT_TABLE_REFERENCE: u32 = 0x8080_b61c;
const PATTERN_GLOBAL_TABLE_REFERENCE: u32 = 0x8080_6cac;
const PATTERN_TRANSLATION_BLOCK_MARKER: u32 = 0x8080_91d3;

#[derive(Clone, Copy, Debug)]
struct Array {
    class: u32,
    count: usize,
    start: usize,
}

#[derive(Clone, Debug)]
struct CurveGroup {
    semantic: u32,
    domain: [f32; 2],
    values: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct CurveSet(Vec<CurveGroup>);

#[derive(Clone, Copy, Debug)]
pub(super) struct WeaponStats {
    pub(super) definition_tag: TagHash,
    pub(super) firepower: Option<f32>,
    pub(super) damage: Option<f32>,
    pub(super) headshot_multiplier: Option<f32>,
    pub(super) bullets_per_shot: Option<f32>,
    pub(super) accuracy: Option<f32>,
    pub(super) handling: Option<f32>,
    pub(super) rounds_per_minute: Option<f32>,
    pub(super) magazine: Option<f32>,
    pub(super) volt_drain_percent: Option<f32>,
    pub(super) range_metres: Option<f32>,
    pub(super) zoom: Option<f32>,
    pub(super) equip_seconds: Option<f32>,
    pub(super) aim_seconds: Option<f32>,
    pub(super) reload_seconds: Option<f32>,
    pub(super) charge_seconds: Option<f32>,
    pub(super) weight: Option<f32>,
    pub(super) hip_fire_spread_degrees: Option<f32>,
    pub(super) ads_spread_degrees: Option<f32>,
    pub(super) crouch_spread_bonus: Option<f32>,
    pub(super) movement_accuracy_loss: Option<f32>,
    pub(super) recoil: Option<f32>,
    pub(super) aim_correction_degrees: Option<f32>,
    pub(super) shotgun_spread_degrees: Option<f32>,
}

pub(super) struct WeaponStatResolver<'a> {
    cache: &'a quicktag_scanner::TagCache,
    pattern_globals: Vec<u32>,
    assignments: FxHashMap<u32, Vec<TagHash>>,
    rating_metadata: Option<Vec<u8>>,
    display_programs: Option<StatDisplayPrograms>,
}

impl<'a> WeaponStatResolver<'a> {
    pub(super) fn load(cache: &'a quicktag_scanner::TagCache) -> Self {
        let mut assignments = FxHashMap::<u32, Vec<TagHash>>::default();
        for (tag, _) in package_manager().get_all_by_reference(PATTERN_ASSIGNMENT_TABLE_REFERENCE) {
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
                let Some(pattern) = resolve_tag64_union(record, 0x8) else {
                    continue;
                };
                if package_manager()
                    .get_entry(pattern)
                    .is_some_and(|entry| entry.reference == PATTERN_REFERENCE)
                {
                    assignments.entry(global_id).or_default().push(pattern);
                }
            }
        }
        for patterns in assignments.values_mut() {
            patterns.sort_unstable();
            patterns.dedup();
        }

        let mut pattern_globals = vec![];
        let mut best_score = (0usize, 0usize);
        for (tag, _) in package_manager().get_all_by_reference(PATTERN_GLOBAL_TABLE_REFERENCE) {
            let Ok(data) = package_manager().read_tag(tag) else {
                continue;
            };
            for stride in [0x38, 0x48] {
                let Some(range) = table_range(&data, 0x8, stride) else {
                    continue;
                };
                let globals = data[range]
                    .chunks_exact(stride)
                    .filter_map(|record| read_u32(record, 0x8))
                    .collect::<Vec<_>>();
                let score = (
                    globals
                        .iter()
                        .filter(|global| assignments.contains_key(global))
                        .count(),
                    globals.len(),
                );
                if score > best_score {
                    best_score = score;
                    pattern_globals = globals;
                }
            }
        }
        Self {
            cache,
            pattern_globals,
            assignments,
            display_programs: StatDisplayPrograms::load(),
            rating_metadata: quicktag_core::implant::ImplantStatResolver::load(&package_manager())
                .ok()
                .and_then(|registry| package_manager().read_tag(registry.semantic_table).ok()),
        }
    }

    pub(super) fn debug_dump(&self, weapon: TagHash, mods: &[TagHash]) -> String {
        let mut out = String::new();
        let mut ratings = vec![];
        for modification in mods {
            match self.mod_raw_stats(*modification) {
                Ok(rows) => {
                    out += &format!("  MOD {modification} {rows:?}\n");
                    ratings.extend(rows);
                }
                Err(error) => out += &format!("  MOD {modification} failed: {error:#}\n"),
            }
        }
        if let Ok(definition) = package_manager().read_tag(weapon) {
            out += &format!("  DEFINITION {weapon} {} bytes\n", definition.len());
            for array in arrays(&definition) {
                out += &format!(
                    "  ARRAY class={:08X} count={} start={:#x}\n",
                    array.class, array.count, array.start
                );
                if array.class == STAT_RATING_ARRAY {
                    for index in 0..array.count {
                        let start = array.start + index * 0x28;
                        out += &format!(
                            "    {:?}\n",
                            (0..10).filter_map(|word| read_u32(&definition, start + word * 4)).collect::<Vec<_>>()
                        );
                    }
                }
            }
            for offset in (0..definition.len().saturating_sub(3)).step_by(4) {
                let Some(value) = read_u32(&definition, offset) else { continue };
                if value >> 16 == 0x8080 && value != ARRAY_MARKER {
                    out += &format!("  CLASS {offset:#06x} {value:08X}\n");
                } else if let Some(entry) = package_manager().get_entry(TagHash(value)) {
                    out += &format!("  TAGREF {offset:#06x} {} ref={:08X}\n", TagHash(value), entry.reference);
                }
            }
        }
        let Some(metadata) = self.rating_metadata.as_deref() else {
            return out + "  no rating metadata\n";
        };
        match self.mod_weapon_context(weapon) {
            Ok(context) => out += &context.debug_dump(&ratings, metadata),
            Err(error) => out += &format!("  context failed: {error:#}\n"),
        }
        out += &self.debug_components(weapon);
        out += &format!("  STATS {:?}\n", self.extract(weapon, mods));
        out
    }

    pub(super) fn debug_display_programs(&self) -> String {
        self.display_programs
            .as_ref()
            .map_or_else(|| "  no display programs\n".to_owned(), StatDisplayPrograms::debug_dump)
    }

    /// Stats as the game shows them: the frame's ratings plus those of the
    /// plugs preset in its sockets (`preset_plugs`), which unique weapons carry.
    pub(super) fn extract(&self, definition_tag: TagHash, presets: &[TagHash]) -> Option<WeaponStats> {
        self.extract_with_ratings(definition_tag, presets)
            .map(|(stats, _)| stats)
    }

    /// `extract`, with the frame's base ratings the stats were sampled from.
    pub(super) fn extract_with_ratings(
        &self,
        definition_tag: TagHash,
        presets: &[TagHash],
    ) -> Option<(WeaponStats, Vec<WeaponBaseRating>)> {
        let context = self.mod_weapon_context(definition_tag).ok()?;
        let mut ratings = vec![];
        for plug in presets {
            match self.mod_raw_stats(*plug) {
                Ok(rows) => ratings.extend(rows),
                Err(error) => {
                    log::warn!("Preset plug {plug} of weapon {definition_tag} has no ratings: {error:#}")
                }
            }
        }
        let stats = context.stats(
            definition_tag,
            &weapon_mod_stats::rating_deltas(&ratings),
            self.rating_metadata.as_deref()?,
            self.display_programs.as_ref(),
        );
        Some((stats, context.base_ratings()))
    }
}

/// Definitions of the plugs preset in a weapon definition's sockets. Each
/// socket row names its plug by position in the hash-to-definition table
/// (`definitions`); an empty socket holds 0xFFFF.
pub(super) fn preset_plugs(definition: &[u8], definitions: &[TagHash]) -> Vec<TagHash> {
    arrays(definition)
        .into_iter()
        .filter(|array| array.class == SOCKET_ARRAY)
        .flat_map(|array| (0..array.count).map(move |row| array.start + row * 0x50 + 0x28))
        .filter_map(|offset| definitions.get(read_u32(definition, offset)? as usize).copied())
        .collect()
}

impl CurveSet {
    fn parse(data: &[u8]) -> Option<Self> {
        let arrays = arrays(data);
        let semantic_array = arrays
            .iter()
            .find(|array| array.class == CURVE_SEMANTIC_ARRAY)?;
        let semantics = (0..semantic_array.count)
            .map(|index| {
                let start = semantic_array.start.checked_add(index.checked_mul(0x38)?)?;
                Some((
                    read_u32(data, start.checked_add(0x20)?)?,
                    [
                        f32::from_bits(read_u32(data, start + 0x2c)?),
                        f32::from_bits(read_u32(data, start + 0x30)?),
                    ],
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let descriptors = arrays
            .iter()
            .enumerate()
            .filter_map(|(index, array)| (array.class == CURVE_DESCRIPTOR_ARRAY).then_some(index))
            .collect::<Vec<_>>();
        if descriptors.len() != semantics.len() {
            return None;
        }
        let mut groups = vec![];
        for ((semantic, domain), descriptor) in semantics.into_iter().zip(descriptors) {
            let Some(values) = arrays[descriptor + 1..]
                .iter()
                .take_while(|array| array.class == CURVE_VALUE_ARRAY)
                .take(arrays[descriptor].count)
                .map(|array| floats(data, *array))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if values.len() != arrays[descriptor].count
                || values.is_empty()
                || !values.iter().map(Vec::len).all_equal()
            {
                return None;
            }
            groups.push(CurveGroup {
                semantic,
                domain,
                values,
            });
        }
        (!groups.is_empty()).then_some(Self(groups))
    }

    fn layout_score(&self) -> usize {
        const WEAPON_SEMANTICS: [u32; 21] = [
            0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
            0x10, 0x11, 0x12, 0x13, 0x16, 0x20, 0x21,
        ];
        self.0
            .iter()
            .filter(|group| WEAPON_SEMANTICS.contains(&group.semantic))
            .count()
            * 10
            + self.0.len()
    }
}

fn parse_ratings(data: &[u8]) -> Option<FxHashMap<u32, f32>> {
    let array = arrays(data)
        .into_iter()
        .find(|array| array.class == STAT_RATING_ARRAY)?;
    let mut ratings = FxHashMap::default();
    for index in 0..array.count {
        let start = array.start.checked_add(index.checked_mul(0x28)?)?;
        let id = read_u32(data, start)?;
        let rating = read_u32(data, start + 4)?;
        if rating <= 100 {
            ratings.insert(id, rating as f32);
        }
    }
    (!ratings.is_empty()).then_some(ratings)
}

fn model_gameplay_components(cache: &quicktag_scanner::TagCache, model: TagHash) -> Vec<TagHash> {
    let mut found = vec![];
    let mut seen = rustc_hash::FxHashSet::default();
    let mut queue = std::collections::VecDeque::from([(model, 0usize)]);
    seen.insert(model);
    while let Some((tag, depth)) = queue.pop_front() {
        if depth >= 12 {
            continue;
        }
        let Some(scan) = cache.hashes.get(&tag) else {
            continue;
        };
        for child in scan.file_hashes.iter().map(|item| item.hash) {
            let Some(entry) = package_manager().get_entry(child) else {
                continue;
            };
            if entry.reference == GAMEPLAY_COMPONENT_REFERENCE {
                found.push(child);
            }
            if entry.reference == GAMEPLAY_COMPONENT_REFERENCE && seen.insert(child) {
                queue.push_back((child, depth + 1));
            }
        }
    }
    let has_curves = found.iter().any(|tag| {
        package_manager()
            .read_tag(*tag)
            .ok()
            .and_then(|data| CurveSet::parse(&data))
            .is_some()
    });
    if !has_curves {
        for scan in cache.hashes.values().filter(|scan| {
            scan.file_hashes
                .iter()
                .any(|item| seen.contains(&item.hash))
        }) {
            found.extend(scan.file_hashes.iter().filter_map(|item| {
                package_manager()
                    .get_entry(item.hash)
                    .is_some_and(|entry| entry.reference == GAMEPLAY_COMPONENT_REFERENCE)
                    .then_some(item.hash)
            }));
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

fn definition_pattern_index(definition: &[u8]) -> Option<u16> {
    let translation = definition
        .chunks_exact(4)
        .position(|bytes| {
            u32::from_le_bytes(bytes.try_into().expect("four bytes"))
                == PATTERN_TRANSLATION_BLOCK_MARKER
        })?
        .checked_mul(4)?;
    read_u16(definition, translation.checked_add(0x6c)?)
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

fn table_range(data: &[u8], header: usize, stride: usize) -> Option<std::ops::Range<usize>> {
    let count = usize::try_from(read_u64(data, header)?).ok()?;
    let relative = read_i64(data, header + 8)?;
    let start: usize = i64::try_from(header)
        .ok()?
        .checked_add(8)?
        .checked_add(relative)?
        .checked_add(0x10)?
        .try_into()
        .ok()?;
    let end = start.checked_add(count.checked_mul(stride)?)?;
    (end <= data.len()).then_some(start..end)
}

fn arrays(data: &[u8]) -> Vec<Array> {
    if data.len() < 20 {
        return vec![];
    }
    (0..=data.len() - 20)
        .step_by(4)
        .filter_map(|offset| {
            (read_u32(data, offset)? == ARRAY_MARKER).then(|| Array {
                class: read_u32(data, offset + 12).unwrap_or_default(),
                count: read_u64(data, offset + 4)
                    .and_then(|count| usize::try_from(count).ok())
                    .unwrap_or_default(),
                start: offset + 20,
            })
        })
        .collect()
}

fn floats(data: &[u8], array: Array) -> Option<Vec<f32>> {
    let byte_len = array.count.checked_mul(4)?;
    let end = array.start.checked_add(byte_len)?;
    let bytes = data.get(array.start..end)?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect(),
    )
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

trait AllEqual: Iterator {
    fn all_equal(mut self) -> bool
    where
        Self: Sized,
        Self::Item: PartialEq,
    {
        let Some(first) = self.next() else {
            return true;
        };
        self.all(|item| item == first)
    }
}

impl<T: Iterator> AllEqual for T {}

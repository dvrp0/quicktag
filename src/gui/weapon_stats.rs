use rustc_hash::FxHashMap;
use tiger_pkg::{TagHash, package_manager};

const ARRAY_MARKER: u32 = 0x8080_bfcd;
const STAT_RATING_ARRAY: u32 = 0x8080_924b;
const CURVE_DESCRIPTOR_ARRAY: u32 = 0x8080_bc4d;
const CURVE_VALUE_ARRAY: u32 = 0x8080_000f;
const CURVE_SEMANTIC_ARRAY: u32 = 0x8080_3f8e;
const GAMEPLAY_COMPONENT_REFERENCE: u32 = 0x8080_badb;
const PATTERN_REFERENCE: u32 = 0x8080_baad;
const PATTERN_ASSIGNMENT_TABLE_REFERENCE: u32 = 0x8080_b61c;
const PATTERN_GLOBAL_TABLE_REFERENCE: u32 = 0x8080_6cac;
const PATTERN_TRANSLATION_BLOCK_MARKER: u32 = 0x8080_91d3;

const RATE_RATING: u32 = 0x0b;
const DAMAGE_RATING: u32 = 0x0c;
const ZOOM_RATING: u32 = 0x0d;
const PRECISION_RATING: u32 = 0x0e;
const MAGAZINE_RATING: u32 = 0x0f;
const RANGE_RATING: u32 = 0x10;
const EQUIP_RATING: u32 = 0x17;
const ADS_SPEED_RATING: u32 = 0x15;
const RELOAD_RATING: u32 = 0x13;
const WEIGHT_RATING: u32 = 0x15;
const RECOIL_RATING: u32 = 0x17;
const SPREAD_ANGLE_RATING: u32 = 0x18;
const AIM_CORRECTION_RATING: u32 = 0x19;
const CHARGE_TIME_RATING: u32 = 0x1c;

#[derive(Clone, Copy, Debug)]
struct Array {
    class: u32,
    count: usize,
    start: usize,
}

#[derive(Clone, Debug)]
struct CurveGroup {
    semantic: u32,
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
        }
    }

    pub(super) fn extract(&self, definition_tag: TagHash) -> Option<WeaponStats> {
        let definition = package_manager().read_tag(definition_tag).ok()?;
        let ratings = parse_ratings(&definition)?;
        let pattern_index = definition_pattern_index(&definition)?;
        let global_id = *self.pattern_globals.get(usize::from(pattern_index))?;
        let patterns = self.assignments.get(&global_id)?;
        WeaponStats::from_patterns(self.cache, definition_tag, &ratings, patterns)
    }
}

impl WeaponStats {
    fn from_patterns(
        cache: &quicktag_scanner::TagCache,
        definition_tag: TagHash,
        ratings: &FxHashMap<u32, f32>,
        patterns: &[TagHash],
    ) -> Option<Self> {
        let mut candidates = patterns
            .iter()
            .flat_map(|pattern| model_gameplay_components(cache, *pattern))
            .into_iter()
            .filter_map(|component_tag| {
                let data = package_manager().read_tag(component_tag).ok()?;
                let curves = CurveSet::parse(&data)?;
                let score = curves.layout_score();
                Some((score, component_tag, curves))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(score, tag, _)| (std::cmp::Reverse(*score), *tag));
        let (_, _, curves) = candidates.into_iter().next()?;

        let value = |semantic, occurrence, row, rating| {
            curves.sample(semantic, occurrence, row, ratings.get(&rating).copied()?)
        };
        let finite = |value: Option<f32>| value.filter(|value| value.is_finite());
        let rows = |semantic| {
            curves
                .group(semantic, 0)
                .and_then(|group| group.values.first())
                .map_or(0, Vec::len)
        };
        let volley_damage = finite(value(0x0c, 0, 0, DAMAGE_RATING));
        let headshot_multiplier = finite(value(0x0d, 0, 0, PRECISION_RATING));
        let single_spread_layout = rows(0x04) == 1;
        let volt_cell_layout = rows(0x01) >= 3
            && curves.group(0x01, 0).is_some_and(|group| {
                group.values.iter().all(|point| {
                    point
                        .first()
                        .is_some_and(|value| (*value - 1_000.0).abs() < 0.01)
                })
            });
        let bullets_per_shot = (single_spread_layout && !volt_cell_layout)
            .then(|| curves.sample(0x00, 0, 2, 10.0))
            .flatten()
            .filter(|value| value.is_finite() && *value > 1.0 && *value <= 64.0);
        let pellet_shotgun = bullets_per_shot.is_some();
        let damage =
            volley_damage.map(|damage| bullets_per_shot.map_or(damage, |pellets| damage / pellets));
        let charge_weapon = volt_cell_layout && single_spread_layout;
        let firepower = if pellet_shotgun {
            volley_damage
        } else if charge_weapon {
            damage
                .zip(finite(value(0x07, 0, 0, CHARGE_TIME_RATING)))
                .and_then(|(damage, charge_seconds)| {
                    (charge_seconds > 0.0).then_some(damage / charge_seconds)
                })
        } else {
            damage
                .zip(headshot_multiplier)
                .map(|(damage, headshot)| damage * headshot)
        };
        let burst_layout = rows(0x00) == 2
            && curves
                .sample(0x00, 0, 1, 10.0)
                .is_some_and(|burst_size| burst_size > 1.0);
        let rounds_per_minute = if burst_layout {
            finite(value(0x11, 0, 0, DAMAGE_RATING))
                .and_then(|shot_interval| (shot_interval > 0.0).then_some(60.0 / shot_interval))
        } else {
            finite(value(0x00, 0, 0, RATE_RATING).map(|value| value * 60.0))
        };
        let range_metres = finite(value(0x08, 1, 0, RANGE_RATING));
        let zoom = finite(value(0x03, 0, 0, ZOOM_RATING));
        let equip_seconds = finite(value(0x0f, 0, 0, EQUIP_RATING));
        let aim_seconds = finite(value(0x10, 0, 0, ADS_SPEED_RATING));
        let reload_seconds = finite(value(0x0a, 0, 0, RELOAD_RATING));
        let weight = finite(value(0x11, 0, 1, WEIGHT_RATING));
        let average = |group, rating| {
            let first = value(group, 0, 0, rating)?;
            let second = value(group, 0, 1, rating)?;
            finite(Some((first + second) * 0.5))
        };
        let hip_fire_spread_degrees = average(0x04, MAGAZINE_RATING).map(f32::to_degrees);
        let ads_spread_degrees = average(0x05, RANGE_RATING).map(f32::to_degrees);
        let crouch_spread_bonus = finite(value(0x20, 0, 0, 0x13));
        let movement_accuracy_loss = finite(value(0x21, 0, 0, 0x10).map(|value| value / 1.1));
        let recoil = finite(value(0x06, 0, 1, RECOIL_RATING));
        let aim_correction_degrees =
            finite(value(0x12, 0, 0, AIM_CORRECTION_RATING).map(|value| {
                let degrees = value.to_degrees();
                if pellet_shotgun {
                    degrees
                } else {
                    degrees / 1.2
                }
            }));
        let shotgun_spread_degrees = single_spread_layout
            .then(|| value(0x04, 0, 0, SPREAD_ANGLE_RATING))
            .flatten()
            .and_then(|value| finite(Some(value)));
        let magazine = (!volt_cell_layout)
            .then(|| value(0x01, 0, 0, MAGAZINE_RATING))
            .flatten()
            .filter(|value| value.is_finite() && *value > 0.0 && *value < 500.0);
        let volt_drain_percent = volt_cell_layout
            .then(|| {
                (1..rows(0x01)).find_map(|row| {
                    finite(value(0x01, 0, row, RATE_RATING)).filter(|value| *value > 0.0)
                })
            })
            .flatten()
            .map(|value| if value > 20.0 { value / 10.0 } else { value });
        let accuracy = accuracy_score(
            hip_fire_spread_degrees,
            ads_spread_degrees,
            movement_accuracy_loss,
        );
        let handling_adjustment = if single_spread_layout { 2.0 } else { 0.0 };
        let handling = handling_score(equip_seconds, recoil, reload_seconds, handling_adjustment);

        [firepower, damage, headshot_multiplier, rounds_per_minute]
            .into_iter()
            .any(|value| value.is_some())
            .then_some(Self {
                definition_tag,
                firepower,
                damage,
                headshot_multiplier,
                bullets_per_shot,
                accuracy,
                handling,
                rounds_per_minute,
                magazine,
                volt_drain_percent,
                range_metres,
                zoom,
                equip_seconds,
                aim_seconds,
                reload_seconds,
                weight,
                hip_fire_spread_degrees,
                ads_spread_degrees,
                crouch_spread_bonus,
                movement_accuracy_loss,
                recoil,
                aim_correction_degrees,
                shotgun_spread_degrees,
            })
    }
}

fn accuracy_score(
    hip_spread: Option<f32>,
    ads_spread: Option<f32>,
    movement: Option<f32>,
) -> Option<f32> {
    let score = 87.75 - 7.0 * hip_spread? - 9.0 * ads_spread? - 14.0 * movement?;
    score.is_finite().then(|| score.clamp(0.0, 100.0))
}

fn handling_score(
    equip: Option<f32>,
    recoil: Option<f32>,
    reload: Option<f32>,
    layout_adjustment: f32,
) -> Option<f32> {
    let score = 95.4 - 28.0 * equip? - 19.0 * recoil? - 3.64 * reload? - layout_adjustment;
    score.is_finite().then(|| score.clamp(0.0, 100.0))
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
                read_u32(data, start.checked_add(0x20)?)
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
        for (semantic, descriptor) in semantics.into_iter().zip(descriptors) {
            let Some(values) = arrays[descriptor + 1..]
                .iter()
                .take_while(|array| array.class == CURVE_VALUE_ARRAY)
                .take(11)
                .map(|array| floats(data, *array))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if values.len() == 11 && values.iter().map(Vec::len).all_equal() {
                groups.push(CurveGroup { semantic, values });
            }
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

    fn group(&self, semantic: u32, occurrence: usize) -> Option<&CurveGroup> {
        self.0
            .iter()
            .filter(|group| group.semantic == semantic)
            .nth(occurrence)
    }

    fn sample(&self, semantic: u32, occurrence: usize, row: usize, rating: f32) -> Option<f32> {
        let curve = &self.group(semantic, occurrence)?.values;
        let position = (rating / 10.0).clamp(0.0, 10.0);
        let low = position.floor() as usize;
        let high = position.ceil() as usize;
        let fraction = position - low as f32;
        let low = *curve.get(low)?.get(row)?;
        let high = *curve.get(high)?.get(row)?;
        Some(low * (1.0 - fraction) + high * fraction)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_curve_ratings() {
        let set = CurveSet(vec![CurveGroup {
            semantic: 7,
            values: (0..=10).map(|rating| vec![rating as f32 * 10.0]).collect(),
        }]);
        assert_eq!(set.sample(7, 0, 0, 25.0), Some(25.0));
    }

    #[test]
    fn computes_client_summary_scores() {
        for (hip, ads, movement, expected) in [
            (2.15, 0.98, 0.327, 59.3),
            (2.32, 0.94, 0.909, 50.3),
            (1.27, 1.13, 0.191, 66.0),
            (1.52, 1.16, 0.205, 63.8),
        ] {
            let actual = accuracy_score(Some(hip), Some(ads), Some(movement)).unwrap();
            assert_eq!(format!("{actual:.1}"), format!("{expected:.1}"));
        }

        for (equip, recoil, reload, expected) in [
            (0.94, 1.140, 2.60, 38.0),
            (0.94, 0.657, 2.37, 48.0),
            (0.94, 0.496, 3.76, 46.0),
            (1.20, 0.577, 5.46, 31.0),
        ] {
            let actual = handling_score(Some(equip), Some(recoil), Some(reload), 0.0).unwrap();
            assert_eq!(format!("{actual:.0}"), format!("{expected:.0}"));
        }

        for (equip, recoil, reload, adjustment, expected) in [
            (0.76, 0.800, 4.10, 2.0, 42.0),
            (0.90, 0.505, 2.645, 2.0, 49.0),
        ] {
            let actual =
                handling_score(Some(equip), Some(recoil), Some(reload), adjustment).unwrap();
            assert_eq!(format!("{actual:.0}"), format!("{expected:.0}"));
        }
    }
}

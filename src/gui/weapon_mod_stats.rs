//! Weapon-context mod deltas. Rating routing comes from investment metadata.
use super::*;
use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug)]
pub(in crate::gui) struct WeaponModStatChange {
    pub name: &'static str,
    pub unit: &'static str,
    pub rating_id: u32,
    pub base_rating: f32,
    pub modified_rating: f32,
    pub before: f32,
    pub after: f32,
    pub derived_from: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub(in crate::gui) struct WeaponModRawStat {
    pub name: String,
    pub rating_id: u32,
    pub value: i32,
}

#[derive(Clone, Debug)]
pub(in crate::gui) struct WeaponModWeaponStats {
    pub weapon: String,
    pub changes: Vec<WeaponModStatChange>,
    pub curves: Vec<WeaponModCurveChange>,
}

/// Lossless evaluation layer: every channel of every routed curve is sampled.
/// Presentation rules never decide which package data gets evaluated.
#[derive(Clone, Debug)]
pub(in crate::gui) struct WeaponModCurveChange {
    pub semantic: u32,
    pub occurrence: usize,
    pub rating_id: u32,
    pub base_rating: f32,
    pub modified_rating: f32,
    pub before: Vec<f32>,
    pub after: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub(in crate::gui) struct WeaponModEvaluation {
    pub changes: Vec<WeaponModStatChange>,
    pub curves: Vec<WeaponModCurveChange>,
}

#[derive(Clone, Debug, Default)]
pub(in crate::gui) struct WeaponModDetails {
    pub ratings: Vec<WeaponModRawStat>,
    pub weapons: Vec<WeaponModWeaponStats>,
}

#[derive(Clone, Debug)]
pub(in crate::gui) struct WeaponModWeaponContext {
    base: FxHashMap<u32, f32>,
    curves: CurveSet,
    properties: PropertyProgram,
}

impl WeaponModStatChange {
    pub fn delta(&self) -> f32 {
        self.after - self.before
    }
}

/// The investment semantic definition stores 35 byte-sized weapon rating routes
/// at 0x3E2. Range has separate four-channel gameplay and two-channel display
/// slots, so semantic IDs after range are shifted by one in this routing table.
fn rating_route(metadata: &[u8], semantic: u32, width: usize) -> Result<u32> {
    let slot = match (semantic, width) {
        (8, 4) => 8,
        (8, 2) => 9,
        (8, _) => anyhow::bail!("unsupported range curve width {width}"),
        (id, _) if id > 8 => usize::try_from(id)? + 1,
        (id, _) => usize::try_from(id)?,
    };
    ensure!(slot < 35, "unknown weapon curve semantic {semantic}");
    let rating = *metadata
        .get(0x3E2 + slot)
        .context("truncated weapon rating routes")?;
    ensure!(
        rating != 255,
        "curve {semantic} has no investment rating route"
    );
    Ok(u32::from(rating))
}

fn sample(curve: &CurveGroup, column: usize, rating: f32) -> Result<f32> {
    ensure!(
        !curve.values.is_empty() && curve.domain[1] > curve.domain[0],
        "invalid authored curve domain"
    );
    let position = ((rating - curve.domain[0]) / (curve.domain[1] - curve.domain[0]))
        .clamp(0.0, 1.0)
        * (curve.values.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let a = *curve.values[low]
        .get(column)
        .context("missing curve channel")?;
    let b = *curve.values[high]
        .get(column)
        .context("missing curve channel")?;
    let value = a + (b - a) * (position - low as f32);
    ensure!(value.is_finite(), "non-finite curve sample");
    Ok(value)
}

impl WeaponStatResolver<'_> {
    pub(in crate::gui) fn mod_raw_stats(
        &self,
        modification: TagHash,
    ) -> Result<Vec<WeaponModRawStat>> {
        let rows = quicktag_core::weapon_mod::decode_weapon_mod_ratings(
            &package_manager().read_tag(modification)?,
        )?;
        ensure!(
            rows.iter().all(|row| row.extra == [0; 32]),
            "unsupported additional mod rating fields"
        );
        Ok(rows
            .into_iter()
            .filter(|row| row.delta != 0)
            .map(|row| WeaponModRawStat {
                name: raw_rating_name(row.stat_id),
                rating_id: row.stat_id,
                value: row.delta,
            })
            .collect())
    }

    /// Cacheable weapon half of mod evaluation. Curves and base ratings are
    /// independent of the selected mod and expensive to rediscover.
    pub(in crate::gui) fn mod_weapon_context(
        &self,
        weapon: TagHash,
    ) -> Result<WeaponModWeaponContext> {
        let pm = package_manager();
        let definition = pm.read_tag(weapon)?;
        let base = parse_ratings(&definition).context("weapon has no base ratings")?;
        let index = definition_pattern_index(&definition).context("weapon has no Pattern index")?;
        let global = self
            .pattern_globals
            .get(index as usize)
            .context("invalid weapon Pattern index")?;
        let patterns = self
            .assignments
            .get(global)
            .context("weapon has no Pattern assignment")?;
        let mut components = patterns
            .iter()
            .flat_map(|pattern| model_gameplay_components(self.cache, *pattern))
            .collect::<Vec<_>>();
        components.sort_unstable();
        components.dedup();
        let mut candidates = components
            .into_iter()
            .filter_map(|tag| {
                let bytes = pm.read_tag(tag).ok()?;
                let curves = CurveSet::parse(&bytes)?;
                let properties = PropertyProgram::parse(&bytes)?;
                Some((curves.layout_score(), tag, curves, properties))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(score, tag, _, _)| (std::cmp::Reverse(*score), *tag));
        let (score, _, curves, properties) =
            candidates.first().context("weapon has no stat curves")?;
        ensure!(
            candidates
                .iter()
                .filter(|(other, _, _, _)| other == score)
                .count()
                == 1,
            "ambiguous weapon stat curve owners"
        );
        Ok(WeaponModWeaponContext {
            base,
            curves: curves.clone(),
            properties: properties.clone(),
        })
    }

    /// `metadata` is the registry-selected 808092D6 definition.
    /// This handles direct investment ratings, not conditional effect execution.


    pub(in crate::gui) fn mod_stat_changes_for_ratings(
        &self,
        context: &WeaponModWeaponContext,
        ratings: &[WeaponModRawStat],
        metadata: &[u8],
    ) -> Result<WeaponModEvaluation> {
        context.evaluate(ratings, metadata)
    }
}

impl WeaponModWeaponContext {
    fn sample_curves(
        &self,
        ratings: &[WeaponModRawStat],
        metadata: &[u8],
    ) -> Result<Vec<WeaponModCurveChange>> {
        let mut deltas = FxHashMap::default();
        for rating in ratings {
            ensure!(
                deltas.insert(rating.rating_id, rating.value).is_none(),
                "duplicate modifier rating {}",
                rating.rating_id
            );
        }
        let mut occurrences = FxHashMap::<u32, usize>::default();
        let mut sampled = vec![];
        for curve in &self.curves.0 {
            let occurrence = occurrences.entry(curve.semantic).or_default();
            let ordinal = *occurrence;
            *occurrence += 1;
            let width = curve.values.first().context("empty stat curve")?.len();
            // Unrouted semantics have no investment input. Keep the authored
            // rating visible, but do not invent a route or base value.
            let Ok(id) = rating_route(metadata, curve.semantic, width) else {
                continue;
            };
            let Some(&base) = self.base.get(&id) else {
                continue;
            };
            let modified =
                (base + deltas.get(&id).copied().unwrap_or_default() as f32).clamp(0.0, 100.0);
            sampled.push(WeaponModCurveChange {
                semantic: curve.semantic,
                occurrence: ordinal,
                rating_id: id,
                base_rating: base,
                modified_rating: modified,
                before: (0..width)
                    .map(|column| sample(curve, column, base))
                    .collect::<Result<_>>()?,
                after: (0..width)
                    .map(|column| sample(curve, column, modified))
                    .collect::<Result<_>>()?,
            });
        }
        Ok(sampled)
    }

    fn is_volt(&self) -> bool {
        self.curves.0.iter().any(|curve| {
            curve.semantic == 1
                && curve
                    .values
                    .first()
                    .is_some_and(|point| point.first() == Some(&1000.0))
        })
    }

    fn evaluate(
        &self,
        ratings: &[WeaponModRawStat],
        metadata: &[u8],
    ) -> Result<WeaponModEvaluation> {
        let sampled = self.sample_curves(ratings, metadata)?;
        let changes = present_curves(&sampled, self.is_volt())
            .into_iter()
            .filter(|change| change.delta().abs() > 0.000001)
            .collect();
        Ok(WeaponModEvaluation {
            changes,
            curves: sampled
                .into_iter()
                .filter(|curve| {
                    ratings
                        .iter()
                        .any(|rating| rating.rating_id == curve.rating_id)
                })
                .collect(),
        })
    }

    pub(super) fn baseline(&self, definition_tag: TagHash, metadata: &[u8]) -> Result<WeaponStats> {
        let sampled = self
            .curves
            .0
            .iter()
            .map(|curve| {
                let width = curve.values.first()?.len();
                let id = rating_route(metadata, curve.semantic, width).ok()?;
                let rating = self.base.get(&id).copied().unwrap_or(0.0);
                (0..width)
                    .map(|column| sample(curve, column, rating).ok())
                    .collect()
            })
            .collect::<Vec<Option<Vec<f32>>>>();
        let properties = self.properties.evaluate(&sampled);
        let value = |group, field| properties.get(&(group, field)).copied();
        let average = |group, first, second| {
            value(group, first)
                .zip(value(group, second))
                .map(|(a, b)| (a + b) * 0.5)
        };
        let curves = self.sample_curves(&[], metadata)?;
        let displayed = present_curves(&curves, self.is_volt());
        let raw_value = |name| {
            displayed
                .iter()
                .find(|stat| stat.name == name)
                .map(|stat| stat.before)
        };
        let pellet = self
            .curves
            .0
            .iter()
            .any(|curve| curve.semantic == 4 && curve.values[0].len() == 1);
        let bullets_per_shot = if pellet && !self.is_volt() {
            self.curves
                .0
                .iter()
                .position(|curve| curve.semantic == 0)
                .and_then(|index| sampled[index].as_ref()?.get(2).copied())
                .filter(|value| *value > 1.0)
        } else {
            None
        };
        let volley_damage = value(2, 0x25);
        let damage = volley_damage.map(|damage| damage / bullets_per_shot.unwrap_or(1.0));
        let precision = value(2, 0x29).map(|bonus| bonus + 1.0);
        Ok(WeaponStats {
            definition_tag,
            firepower: if bullets_per_shot.is_some() {
                volley_damage
            } else {
                damage
                    .zip(precision)
                    .map(|(damage, precision)| damage * precision)
            },
            damage,
            headshot_multiplier: precision,
            bullets_per_shot,
            // No authored formula has been established for these composite scores.
            accuracy: None,
            handling: None,
            rounds_per_minute: value(2, 1).map(|rate| rate * 60.0),
            magazine: if self.is_volt() { None } else { value(1, 0) },
            volt_drain_percent: raw_value("Volt Drain"),
            range_metres: value(0, 9),
            zoom: value(0, 3),
            equip_seconds: value(0, 0),
            aim_seconds: value(0, 4),
            reload_seconds: value(3, 1),
            weight: value(0, 5),
            hip_fire_spread_degrees: if pellet {
                None
            } else {
                average(2, 0x0e, 0x0f).map(f32::to_degrees)
            },
            ads_spread_degrees: average(2, 0x10, 0x11).map(f32::to_degrees),
            crouch_spread_bonus: value(2, 0x13),
            movement_accuracy_loss: value(2, 0x14),
            recoil: value(2, 0x46),
            aim_correction_degrees: value(0, 0x0b).map(f32::to_degrees),
            shotgun_spread_degrees: if pellet {
                raw_value("Spread Angle")
            } else {
                None
            },
        })
    }
}

/// Human display interpretation is separate from the exhaustive curve evaluator.
/// Unknown semantics and additional channels remain available in `curves`.
fn present_curves(curves: &[WeaponModCurveChange], volt: bool) -> Vec<WeaponModStatChange> {
    let magazine = curves.iter().find(|curve| curve.semantic == 1);
    let volt_cell = volt
        && magazine.is_some_and(|curve| {
            curve.before.len() >= 3 && curve.before[0] == 1000.0 && curve.after[0] == 1000.0
        });
    let pellet = curves
        .iter()
        .any(|curve| curve.semantic == 4 && curve.before.len() == 1);
    let mut changes = vec![];
    let mut push = |curve: &WeaponModCurveChange, name, unit, before: f32, after: f32| {
        if before.is_finite() && after.is_finite() {
            changes.push(WeaponModStatChange {
                name,
                unit,
                rating_id: curve.rating_id,
                base_rating: curve.base_rating,
                modified_rating: curve.modified_rating,
                before,
                after,
                derived_from: None,
            });
        }
    };
    for curve in curves {
        let a = &curve.before;
        let b = &curve.after;
        if a.is_empty() {
            continue;
        }
        match (curve.semantic, a.len(), curve.occurrence) {
            (0, width, _) => {
                let column = if !pellet && width >= 3 && (volt || a[2] == 0.0) {
                    1
                } else if !pellet && width >= 3 && a[2] > 0.0 {
                    2
                } else {
                    0
                };
                push(
                    curve,
                    "Rate of Fire",
                    "RPM",
                    a[column] * 60.0,
                    b[column] * 60.0,
                );
            }
            (1, 4, _) if volt_cell => push(curve, "Volt Drain", "percentage points", a[1], b[1]),
            (1, _, _) if volt_cell => push(
                curve,
                "Volt Drain",
                "percentage points",
                a[2] / a[0] * 100.0,
                b[2] / b[0] * 100.0,
            ),
            (1, _, _) if volt => push(
                curve,
                "Volt Drain",
                "percentage points",
                a[0] / 10.0,
                b[0] / 10.0,
            ),
            (1, _, _) => push(curve, "Magazine", "rounds", a[0], b[0]),
            (3, _, _) => push(curve, "Zoom", "x", a[0], b[0]),
            (4, 1, _) => push(curve, "Spread Angle", "degrees", a[0], b[0]),
            (4, width, _) if width >= 2 => push(
                curve,
                "Hipfire Spread",
                "degrees",
                ((a[0] + a[1]) * 0.5).to_degrees(),
                ((b[0] + b[1]) * 0.5).to_degrees(),
            ),
            (5, width, _) if width >= 2 => push(
                curve,
                "ADS spread",
                "degrees",
                ((a[0] + a[1]) * 0.5).to_degrees(),
                ((b[0] + b[1]) * 0.5).to_degrees(),
            ),
            (6, width, _) if width >= 2 => push(
                curve,
                "Recoil",
                "percentage points",
                a[1] * 100.0,
                b[1] * 100.0,
            ),
            (8, 2, _) => push(curve, "Range", "m", a[0], b[0]),
            (10, _, _) => push(curve, "Reload time", "s", a[0], b[0]),
            (11, _, _) => push(curve, "Charge Time", "s", a[0], b[0]),
            (12, _, _) => push(curve, "Damage", "", a[0], b[0]),
            (13, _, _) => push(curve, "Precision", "x", a[0], b[0]),
            (15, _, _) => push(curve, "Equip time", "s", a[0], b[0]),
            (16, _, _) => push(curve, "ADS time", "s", a[0], b[0]),
            (17, width, _) if width >= 2 => push(
                curve,
                "Weight",
                "percentage points",
                a[1] * 100.0,
                b[1] * 100.0,
            ),
            (18, _, 0) => {
                let scale = if pellet { 1.0 } else { 1.2 };
                push(
                    curve,
                    "Aim Assist",
                    "",
                    a[0].to_degrees() / scale,
                    b[0].to_degrees() / scale,
                );
            }
            (32, _, _) => push(
                curve,
                "Crouch Spread Bonus",
                "percentage points",
                a[0] * 100.0,
                b[0] * 100.0,
            ),
            (33, _, _) => push(
                curve,
                "Moving Inaccuracy",
                "percentage points",
                a[0] * 100.0,
                b[0] * 100.0,
            ),
            _ => {}
        }
    }
    if let (Some(damage), Some(precision)) = (
        curves.iter().find(|curve| curve.semantic == 12),
        curves.iter().find(|curve| curve.semantic == 13),
    ) {
        // Recompute the composite at both endpoints; multiplying individual
        // deltas would omit cross terms when multiple inputs change together.
        let source = if damage.base_rating != damage.modified_rating {
            damage
        } else {
            precision
        };
        push(
            source,
            "Firepower",
            "",
            damage.before[0] * precision.before[0],
            damage.after[0] * precision.after[0],
        );
    }
    if let Some(firepower) = changes.iter_mut().find(|change| change.name == "Firepower") {
        firepower.derived_from = Some("Damage × Precision (recomputed before and after)");
    }
    changes
}

fn raw_rating_name(id: u32) -> String {
    let name = match id {
        11 => "Rate of Fire",
        12 => "Damage",
        13 => "Zoom",
        14 => "Precision",
        15 => "Stability",
        16 => "Range",
        17 => "Accuracy",
        18 => "Ammo capacity / consumption",
        19 => "Reload speed",
        21 => "Weight",
        22 => "ADS speed",
        23 => "Equip speed",
        25 => "ADS assist",
        26 => "Charge time",
        66 => "ADS accuracy",
        _ => return format!("Rating {id}"),
    };
    name.to_owned()
}

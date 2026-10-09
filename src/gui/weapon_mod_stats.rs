//! Weapon-context mod deltas. Rating routing comes from investment metadata.
use super::*;
use anyhow::{Context, Result, ensure};

/// Radians-to-degrees factor of the authored display programs (808028FB).
const DISPLAY_DEGREES: f32 = 57.3;

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
pub(in crate::gui) struct WeaponBaseRating {
    pub name: String,
    pub rating_id: u32,
    pub value: f32,
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

/// Several installed mods may move the same rating; their points add.
pub(super) fn rating_deltas(ratings: &[WeaponModRawStat]) -> FxHashMap<u32, i32> {
    let mut deltas = FxHashMap::default();
    for rating in ratings {
        *deltas.entry(rating.rating_id).or_default() += rating.value;
    }
    deltas
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

    /// Diagnostic: every gameplay component reachable from the weapon's
    /// Patterns, with the curves each one carries.
    pub(in crate::gui) fn debug_components(&self, weapon: TagHash) -> String {
        let pm = package_manager();
        let Some(patterns) = pm
            .read_tag(weapon)
            .ok()
            .and_then(|definition| definition_pattern_index(&definition))
            .and_then(|index| self.pattern_globals.get(index as usize))
            .and_then(|global| self.assignments.get(global))
        else {
            return "  no patterns\n".to_owned();
        };
        let mut out = String::new();
        // Every (0xC, pattern-global index) pair the definition holds, not
        // only the render Pattern.
        let definition = pm.read_tag(weapon).unwrap_or_default();
        let mut linked = vec![];
        for offset in (0..definition.len().saturating_sub(7)).step_by(4) {
            if read_u32(&definition, offset) != Some(0xc) {
                continue;
            }
            let Some(index) = read_u32(&definition, offset + 4) else { continue };
            let Some(global) = self.pattern_globals.get(index as usize) else { continue };
            let resolved = self.assignments.get(global).cloned().unwrap_or_default();
            out += &format!("  LINK {offset:#x} index={index:#x} global={global:08X} patterns={resolved:?}\n");
            linked.extend(resolved);
        }
        linked.sort_unstable();
        linked.dedup();
        for pattern in patterns.iter().chain(linked.iter().filter(|pattern| !patterns.contains(pattern))) {
            out += &format!("  PATTERN {pattern}\n");
            for tag in model_gameplay_components(self.cache, *pattern) {
                let Ok(bytes) = pm.read_tag(tag) else { continue };
                let curves = CurveSet::parse(&bytes);
                out += &format!(
                    "    COMPONENT {tag} {} bytes properties={} score={:?} semantics={:?}\n",
                    bytes.len(),
                    PropertyProgram::parse(&bytes).map_or(0, |program| program.rows().len()),
                    curves.as_ref().map(CurveSet::layout_score),
                    curves.map(|curves| {
                        curves
                            .0
                            .iter()
                            .map(|curve| (curve.semantic, curve.values[0].len()))
                            .collect::<Vec<_>>()
                    }),
                );
            }
        }
        out
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

    /// This handles direct investment ratings, not conditional effect execution.
    pub(in crate::gui) fn mod_stat_changes_for_ratings(
        &self,
        context: &WeaponModWeaponContext,
        ratings: &[WeaponModRawStat],
    ) -> Result<WeaponModEvaluation> {
        let metadata = self
            .rating_metadata
            .as_deref()
            .context("weapon rating routes are unavailable")?;
        context.evaluate(ratings, metadata, self.display_programs.as_ref())
    }
}

impl WeaponModWeaponContext {
    /// The frame's own investment ratings, before any plug moves them.
    pub(in crate::gui) fn base_ratings(&self) -> Vec<WeaponBaseRating> {
        let mut ratings = self
            .base
            .iter()
            .map(|(id, value)| WeaponBaseRating {
                name: raw_rating_name(*id),
                rating_id: *id,
                value: *value,
            })
            .collect::<Vec<_>>();
        ratings.sort_by_key(|rating| rating.rating_id);
        ratings
    }

    /// Diagnostic text for `--dump-weapon-stats`: every authored input and
    /// every evaluated property, before any presentation rule is applied.
    pub(in crate::gui) fn debug_dump(&self, ratings: &[WeaponModRawStat], metadata: &[u8]) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let deltas = rating_deltas(ratings);
        let delta = |id: u32| deltas.get(&id).copied().unwrap_or_default();
        let mut base = self.base.iter().collect::<Vec<_>>();
        base.sort_by_key(|(id, _)| **id);
        for (id, rating) in base {
            let _ = writeln!(out, "  RATING {id:3} = {rating} ({:+}) {}", delta(*id), raw_rating_name(*id));
        }
        let sampled = self.sample_properties(&deltas, metadata);
        for (index, (curve, values)) in self.curves.0.iter().zip(&sampled).enumerate() {
            let width = curve.values.first().map_or(0, Vec::len);
            let route = rating_route(metadata, curve.semantic, width).ok();
            let rating = route.and_then(|id| self.base.get(&id).copied());
            let _ = writeln!(
                out,
                "  CURVE {index:2} semantic={:2} width={width} route={route:?} rating={rating:?} ({:+})\n      all    ={:?}\n      sampled={values:?}",
                curve.semantic,
                route.map_or(0, delta),
                curve.values,
            );
        }
        for row in self.properties.rows() {
            let _ = writeln!(
                out,
                "  ROW dst=({},{:#x}) op={:#x} a={:x?} b={:x?} order={}",
                row[0], row[1], row[4], &row[5..11], &row[11..17], row[23]
            );
        }
        let mut properties = self.properties.evaluate(&sampled).into_iter().collect::<Vec<_>>();
        properties.sort_by_key(|(key, _)| *key);
        for ((group, field), value) in properties {
            let _ = writeln!(out, "  PROP ({group},{field:#04x}) = {value}");
        }
        out
    }

    /// The modified rating of a routed curve; `None` when the weapon does not
    /// carry the rating, so no mod can move it.
    fn curve_rating(
        &self,
        curve: &CurveGroup,
        deltas: &FxHashMap<u32, i32>,
        metadata: &[u8],
    ) -> Option<(u32, Option<(f32, f32)>)> {
        let width = curve.values.first()?.len();
        let id = rating_route(metadata, curve.semantic, width).ok()?;
        let ratings = self.base.get(&id).map(|base| {
            let delta = deltas.get(&id).copied().unwrap_or_default();
            (*base, (base + delta as f32).clamp(0.0, 100.0))
        });
        Some((id, ratings))
    }

    fn sample_curves(
        &self,
        deltas: &FxHashMap<u32, i32>,
        metadata: &[u8],
    ) -> Result<Vec<WeaponModCurveChange>> {
        let mut occurrences = FxHashMap::<u32, usize>::default();
        let mut sampled = vec![];
        for curve in &self.curves.0 {
            let occurrence = occurrences.entry(curve.semantic).or_default();
            let ordinal = *occurrence;
            *occurrence += 1;
            let width = curve.values.first().context("empty stat curve")?.len();
            // Unrouted semantics have no investment input. Keep the authored
            // rating visible, but do not invent a route or base value.
            let Some((id, Some((base, modified)))) = self.curve_rating(curve, deltas, metadata)
            else {
                continue;
            };
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

    /// Curve inputs of the property program, one entry per authored curve.
    /// A routed curve whose rating the weapon lacks rests at rating zero.
    fn sample_properties(
        &self,
        deltas: &FxHashMap<u32, i32>,
        metadata: &[u8],
    ) -> Vec<Option<Vec<f32>>> {
        self.curves
            .0
            .iter()
            .map(|curve| {
                let (_, ratings) = self.curve_rating(curve, deltas, metadata)?;
                let rating = ratings.map_or(0.0, |(_, modified)| modified);
                (0..curve.values.first()?.len())
                    .map(|column| sample(curve, column, rating).ok())
                    .collect()
            })
            .collect()
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
        display: Option<&StatDisplayPrograms>,
    ) -> Result<WeaponModEvaluation> {
        let deltas = rating_deltas(ratings);
        let sampled = self.sample_curves(&deltas, metadata)?;
        // Both endpoints run the authored property program, the same path as
        // the weapon panel, so a delta is always the difference of two
        // displayed values.
        let before = self.stats(TagHash(0), &FxHashMap::default(), metadata, display);
        let after = self.stats(TagHash(0), &deltas, metadata, display);
        Ok(WeaponModEvaluation {
            changes: stat_changes(&before, &after, &sampled),
            curves: sampled
                .into_iter()
                .filter(|curve| deltas.contains_key(&curve.rating_id))
                .collect(),
        })
    }

    /// Displayed weapon stats with `deltas` rating points installed.
    pub(super) fn stats(
        &self,
        definition_tag: TagHash,
        deltas: &FxHashMap<u32, i32>,
        metadata: &[u8],
        display: Option<&StatDisplayPrograms>,
    ) -> WeaponStats {
        let sampled = self.sample_properties(deltas, metadata);
        let properties = self.properties.evaluate(&sampled);
        let value = |group, field| properties.get(&(group, field)).copied();
        let average = |group, first, second| {
            value(group, first)
                .zip(value(group, second))
                .map(|(a, b)| (a + b) * 0.5 * DISPLAY_DEGREES)
        };
        let curve = |semantic| {
            self.curves
                .0
                .iter()
                .position(|curve| curve.semantic == semantic)
                .and_then(|index| sampled[index].as_deref())
        };
        let volt = self.is_volt();
        let pellet = self
            .curves
            .0
            .iter()
            .any(|curve| curve.semantic == 4 && curve.values[0].len() == 1);
        let bullets_per_shot = (pellet && !volt)
            .then(|| curve(0)?.get(2).copied())
            .flatten()
            .filter(|value| *value > 1.0);
        // (2,0x25) is damage per projectile; the authored row already divides
        // a shotgun's volley by its pellet count.
        let damage = value(2, 0x25);
        let precision_bonus = value(2, 0x29);
        let firepower = damage.zip(precision_bonus).and_then(|(damage, bonus)| {
            display
                .and_then(|display| display.firepower(damage, bonus, bullets_per_shot))
                .or(Some(damage * (bonus + 1.0) * bullets_per_shot.unwrap_or(1.0)))
        });
        let volt_drain_percent = curve(1).filter(|_| volt).and_then(|cell| {
            let volt_cell = cell.len() >= 3 && cell[0] == 1000.0;
            match (volt_cell, cell.len()) {
                (true, 4) => Some(cell[1]),
                (true, _) => Some(cell[2] / cell[0] * 100.0),
                (false, _) => cell.first().map(|charge| charge / 10.0),
            }
        });
        WeaponStats {
            definition_tag,
            firepower,
            damage,
            headshot_multiplier: precision_bonus.map(|bonus| bonus + 1.0),
            bullets_per_shot,
            // No authored formula has been established for these composite scores.
            accuracy: None,
            handling: None,
            rounds_per_minute: value(2, 1).map(|rate| rate * 60.0),
            magazine: if volt { None } else { value(1, 0) },
            volt_drain_percent,
            range_metres: value(0, 9),
            zoom: value(0, 3),
            equip_seconds: value(0, 0),
            aim_seconds: value(0, 4),
            reload_seconds: value(3, 1),
            charge_seconds: value(4, 0),
            weight: value(0, 5),
            hip_fire_spread_degrees: if pellet {
                None
            } else {
                average(2, 0x0e, 0x0f)
            },
            ads_spread_degrees: average(2, 0x10, 0x11),
            crouch_spread_bonus: value(2, 0x13),
            // WeaponStats stores a fraction; the authored UI program returns percent.
            movement_accuracy_loss: value(2, 0x14).and_then(|loss| {
                display?
                    .single_property((2, 0x14), loss)
                    .map(|percent| percent / 100.0)
            }),
            recoil: value(2, 0x46),
            aim_correction_degrees: value(0, 0x0b).map(|angle| angle * DISPLAY_DEGREES),
            shotgun_spread_degrees: value(2, 0x2c).filter(|_| pellet),
        }
    }
}

type StatRow = (
    &'static str,
    &'static str,
    Option<u32>,
    fn(&WeaponStats) -> Option<f32>,
);

/// Mod-panel rows: name, unit, the curve semantic whose rating normally drives
/// the stat (for the hover text only), and the displayed value.
const STAT_ROWS: [StatRow; 20] = [
    ("Rate of Fire", "RPM", Some(0), |stats| stats.rounds_per_minute),
    ("Magazine", "rounds", Some(1), |stats| stats.magazine),
    ("Volt Drain", "percentage points", Some(1), |stats| stats.volt_drain_percent),
    ("Zoom", "x", Some(3), |stats| stats.zoom),
    ("Spread Angle", "degrees", Some(4), |stats| stats.shotgun_spread_degrees),
    ("Hipfire Spread", "degrees", Some(4), |stats| stats.hip_fire_spread_degrees),
    ("ADS spread", "degrees", Some(5), |stats| stats.ads_spread_degrees),
    ("Recoil", "percentage points", Some(6), |stats| stats.recoil.map(|value| value * 100.0)),
    ("Range", "m", Some(8), |stats| stats.range_metres),
    ("Reload time", "s", Some(10), |stats| stats.reload_seconds),
    ("Charge Time", "s", Some(11), |stats| stats.charge_seconds),
    ("Damage", "", Some(12), |stats| stats.damage),
    ("Precision", "x", Some(13), |stats| stats.headshot_multiplier),
    ("Equip time", "s", Some(15), |stats| stats.equip_seconds),
    ("ADS time", "s", Some(16), |stats| stats.aim_seconds),
    ("Weight", "percentage points", Some(17), |stats| stats.weight.map(|value| value * 100.0)),
    ("Aim Assist", "", Some(18), |stats| stats.aim_correction_degrees),
    ("Crouch Spread Bonus", "percentage points", Some(32), |stats| {
        stats.crouch_spread_bonus.map(|value| value * 100.0)
    }),
    ("Moving Inaccuracy", "percentage points", Some(33), |stats| {
        stats.movement_accuracy_loss.map(|value| value * 100.0)
    }),
    ("Firepower", "", None, |stats| stats.firepower),
];

fn stat_changes(
    before: &WeaponStats,
    after: &WeaponStats,
    curves: &[WeaponModCurveChange],
) -> Vec<WeaponModStatChange> {
    STAT_ROWS
        .iter()
        .filter_map(|(name, unit, semantic, value)| {
            let (before, after) = value(before).zip(value(after))?;
            if !before.is_finite() || !after.is_finite() || (after - before).abs() <= 0.000001 {
                return None;
            }
            // Range has a four-channel gameplay curve ahead of its display curve.
            let source = semantic.and_then(|semantic| {
                curves
                    .iter()
                    .filter(|curve| curve.semantic == semantic)
                    .find(|curve| semantic != 8 || curve.before.len() == 2)
            });
            // A property may combine curves, so the stat's own rating can be at rest.
            let source = source.filter(|curve| curve.base_rating != curve.modified_rating);
            Some(WeaponModStatChange {
                name: *name,
                unit: *unit,
                rating_id: source.map_or(0, |curve| curve.rating_id),
                base_rating: source.map_or(0.0, |curve| curve.base_rating),
                modified_rating: source.map_or(0.0, |curve| curve.modified_rating),
                before,
                after,
                derived_from: match (source, *name) {
                    (Some(_), _) => None,
                    (None, "Firepower") => {
                        Some("Damage × Precision (recomputed before and after)")
                    }
                    (None, _) => Some("Authored property program (recomputed before and after)"),
                },
            })
        })
        .collect()
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
        64 => "Crouch accuracy",
        65 => "Moving accuracy",
        66 => "ADS accuracy",
        _ => return format!("Rating {id}"),
    };
    name.to_owned()
}

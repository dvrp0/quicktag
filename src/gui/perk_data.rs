//! Raw gameplay data behind an item's effects: the perk tag each effect row
//! resolves to, its inline constants, and the stat modifiers of the hop-on
//! Patterns it applies. Nothing here is interpreted beyond naming indices.
use std::sync::{Arc, LazyLock, Mutex};

use eframe::egui::{self, Color32, RichText};
use rustc_hash::FxHashMap;
use serde::Serialize;
use tiger_pkg::{TagHash, package_manager};

const ARRAY_MARKER: u32 = 0x8080_bfcd;
const ITEM_EFFECT_ARRAY: u32 = 0x8080_924c;
const EFFECT_DEFINITION_REFERENCE: u32 = 0x8080_6b54;
const PATTERN_ASSIGNMENT_TABLE_REFERENCE: u32 = 0x8080_b61c;
const PERK_REFERENCE: u32 = 0x8080_4d45;
const PATTERN_REFERENCE: u32 = 0x8080_baad;
const COMPONENT_REFERENCE: u32 = 0x8080_badb;
const SCALAR_ARRAY: u32 = 0x8080_4553;
const LABEL_ARRAY: u32 = 0x8080_b723;
const STAT_MODIFIER: u32 = 0x8080_4113;

#[derive(Clone, Debug)]
pub(super) struct PerkData {
    pub effect_index: u32,
    pub tag: TagHash,
    /// Authoring paths embedded in the perk tag.
    pub sources: Vec<String>,
    /// Scalars of the perk's own modifier rows (class 80804553).
    pub scalars: Vec<f32>,
    pub labels: Vec<PerkLabel>,
    pub constants: Vec<PerkConstant>,
    pub hop_ons: Vec<HopOn>,
}

#[derive(Clone, Debug)]
pub(super) struct PerkLabel {
    pub hash: u32,
    pub name: Option<&'static str>,
}

/// A float found in the perk tag, with the structure class that precedes it.
#[derive(Clone, Debug)]
pub(super) struct PerkConstant {
    pub offset: usize,
    pub class: u32,
    pub class_offset: usize,
    pub value: f32,
}

#[derive(Clone, Debug)]
pub(super) struct HopOn {
    pub pattern: TagHash,
    pub modifiers: Vec<StatModifier>,
}

#[derive(Clone, Debug)]
pub(super) struct StatModifier {
    pub component: TagHash,
    pub value: f32,
    pub index: u16,
    pub mode: u32,
}

impl PerkData {
    /// The authoring folder of the perk (`content\sandbox\<kind>\<group>\<name>\...`),
    /// or the perk tag when the tag embeds no content path of its own.
    pub(super) fn title(&self) -> String {
        self.sources
            .iter()
            .filter(|path| !path.contains("label_globals") && !path.contains("_feedback"))
            .find_map(|path| path.split(['\\', '/']).nth(4))
            .map_or_else(|| self.tag.to_string(), str::to_owned)
    }
}

impl StatModifier {
    /// Mode 14 adds rating points to a weapon curve semantic; mode 2 scales
    /// gameplay property (2, index). Both readings are inferred from how the
    /// authored perks line up with their descriptions.
    pub(super) fn describe(&self) -> (String, String) {
        match self.mode {
            14 => (
                curve_semantic_name(self.index)
                    .map_or_else(|| format!("curve semantic {}", self.index), str::to_owned),
                format!("{:+} rating", self.value),
            ),
            2 => (
                property_name(self.index).map_or_else(
                    || format!("property (2, {:#04x})", self.index),
                    |name| format!("{name} (2, {:#04x})", self.index),
                ),
                format!("×{}", self.value),
            ),
            mode => (
                format!("index {} · mode {mode}", self.index),
                format!("{}", self.value),
            ),
        }
    }
}

fn curve_semantic_name(index: u16) -> Option<&'static str> {
    Some(match index {
        0 => "Rate of fire",
        1 => "Magazine",
        3 => "Zoom",
        4 => "Hipfire accuracy",
        5 => "ADS accuracy",
        6 => "Stability",
        8 => "Range",
        10 => "Reload speed",
        11 => "Charge time",
        12 => "Damage",
        13 => "Precision",
        15 => "Equip speed",
        16 => "ADS speed",
        17 => "Weight",
        18 => "Aim assist",
        26 => "Stock state",
        32 => "Crouch accuracy",
        33 => "Moving accuracy",
        _ => return None,
    })
}

fn property_name(index: u16) -> Option<&'static str> {
    Some(match index {
        0x00..=0x03 => "Rate of fire",
        0x0e | 0x0f => "Hipfire spread",
        0x10 | 0x11 => "ADS spread",
        0x13 => "Crouch spread",
        0x14 => "Moving inaccuracy",
        0x25 => "Damage",
        0x29 => "Precision bonus",
        0x2c => "Spread angle",
        0x46 => "Recoil",
        _ => return None,
    })
}

/// Gear JSON form of one perk. Tags and offsets are kept for reference but
/// renumber between builds; `name`, stat names and indices are the stable part.
#[derive(Serialize)]
pub(super) struct PerkExport {
    name: String,
    effect_index: u32,
    perk_tag: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    scalars: Vec<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    labels: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hop_ons: Vec<HopOnExport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    constants: Vec<ConstantExport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    sources: Vec<String>,
}

#[derive(Serialize)]
struct HopOnExport {
    pattern: String,
    modifiers: Vec<ModifierExport>,
}

#[derive(Serialize)]
struct ModifierExport {
    stat: String,
    /// `rating` (points on a curve semantic), `multiplier` (on a gameplay
    /// property) or `raw` for a mode that has no reading yet.
    kind: &'static str,
    index: u16,
    mode: u32,
    value: f32,
    component: String,
}

#[derive(Serialize)]
struct ConstantExport {
    /// Structure class the value follows, and its distance from it.
    after: String,
    rel: usize,
    offset: String,
    value: f32,
}

pub(super) fn export(perks: &[PerkData]) -> Vec<PerkExport> {
    perks
        .iter()
        .map(|perk| PerkExport {
            name: perk.title(),
            effect_index: perk.effect_index,
            perk_tag: perk.tag.to_string(),
            scalars: perk.scalars.clone(),
            labels: perk
                .labels
                .iter()
                .map(|label| {
                    label
                        .name
                        .map_or_else(|| format!("#{:08X}", label.hash), str::to_owned)
                })
                .collect(),
            hop_ons: perk
                .hop_ons
                .iter()
                .map(|hop_on| HopOnExport {
                    pattern: hop_on.pattern.to_string(),
                    modifiers: hop_on
                        .modifiers
                        .iter()
                        .map(|modifier| ModifierExport {
                            stat: modifier.describe().0,
                            kind: match modifier.mode {
                                14 => "rating",
                                2 => "multiplier",
                                _ => "raw",
                            },
                            index: modifier.index,
                            mode: modifier.mode,
                            value: modifier.value,
                            component: modifier.component.to_string(),
                        })
                        .collect(),
                })
                .collect(),
            constants: perk
                .constants
                .iter()
                .map(|constant| ConstantExport {
                    after: format!("{:08X}", constant.class),
                    rel: constant.offset - constant.class_offset,
                    offset: format!("{:#06x}", constant.offset),
                    value: constant.value,
                })
                .collect(),
            sources: perk.sources.clone(),
        })
        .collect()
}

struct PerkStore {
    /// Effect-table rows: the key each effect is assigned by.
    effect_keys: Vec<u32>,
    /// Assignment key to perk tag.
    perks: FxHashMap<u32, TagHash>,
    labels: FxHashMap<u32, &'static str>,
    cache: FxHashMap<TagHash, Arc<Vec<PerkData>>>,
}

static STORE: LazyLock<Mutex<PerkStore>> = LazyLock::new(|| Mutex::new(PerkStore::load()));

/// Perk data of every effect row in an item definition. Decoded once per item.
pub(super) fn perk_data_for(definition: TagHash) -> Arc<Vec<PerkData>> {
    let mut store = STORE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cached) = store.cache.get(&definition) {
        return cached.clone();
    }
    let decoded = Arc::new(store.decode(definition));
    store.cache.insert(definition, decoded.clone());
    decoded
}

impl PerkStore {
    fn load() -> Self {
        let pm = package_manager();
        let effect_keys = pm
            .get_all_by_reference(EFFECT_DEFINITION_REFERENCE)
            .into_iter()
            .filter_map(|(tag, _)| pm.read_tag(tag).ok())
            .filter_map(|data| {
                let range = table_range(&data, 0x8, 0x28)?;
                Some(
                    data[range]
                        .chunks_exact(0x28)
                        .map(|row| read_u32(row, 4).unwrap_or_default())
                        .collect::<Vec<_>>(),
                )
            })
            .max_by_key(Vec::len)
            .unwrap_or_default();

        let mut perks = FxHashMap::default();
        for (tag, _) in pm.get_all_by_reference(PATTERN_ASSIGNMENT_TABLE_REFERENCE) {
            let Ok(data) = pm.read_tag(tag) else { continue };
            let Some(range) = table_range(&data, 0x8, 0x18) else { continue };
            for record in data[range].chunks_exact(0x18) {
                let Some(key) = read_u32(record, 0) else { continue };
                let Some(target) = resolve_tag64_union(record, 0x8) else { continue };
                if has_reference(target, PERK_REFERENCE) {
                    perks.insert(key, target);
                }
            }
        }

        Self {
            effect_keys,
            perks,
            labels: LABEL_WORDS.iter().map(|word| (fnv1(word), *word)).collect(),
            cache: FxHashMap::default(),
        }
    }

    fn decode(&self, definition: TagHash) -> Vec<PerkData> {
        let pm = package_manager();
        let Ok(data) = pm.read_tag(definition) else {
            return vec![];
        };
        arrays(&data, ITEM_EFFECT_ARRAY)
            .flat_map(|(start, count)| (0..count).map(move |row| start + row * 0x18))
            .filter_map(|row| read_u32(&data, row))
            .filter_map(|effect_index| {
                let key = self.effect_keys.get(effect_index as usize)?;
                let tag = *self.perks.get(key)?;
                Some(self.decode_perk(effect_index, tag, &pm.read_tag(tag).ok()?))
            })
            .collect()
    }

    fn decode_perk(&self, effect_index: u32, tag: TagHash, data: &[u8]) -> PerkData {
        let scalars = arrays(data, SCALAR_ARRAY)
            .filter_map(|(start, _)| read_f32(data, start + 4))
            .collect();
        let labels = arrays(data, LABEL_ARRAY)
            .flat_map(|(start, count)| (0..count).map(move |row| start + row * 0x20))
            .filter_map(|row| read_u32(data, row))
            .map(|hash| PerkLabel {
                hash,
                name: self.labels.get(&hash).copied(),
            })
            .collect();

        let sources = ascii_runs(data, 12);
        let text_start = sources
            .iter()
            .map(|(offset, _)| *offset)
            .min()
            .unwrap_or(data.len());
        let mut constants = vec![];
        let mut class = (0u32, 0usize);
        for offset in (0..text_start.saturating_sub(3)).step_by(4) {
            let Some(word) = read_u32(data, offset) else { break };
            if word >> 16 == 0x8080 {
                if word != ARRAY_MARKER {
                    class = (word, offset);
                }
                continue;
            }
            let value = f32::from_bits(word);
            // Authored constants are short decimals; anything else is an
            // offset, hash or packed integer.
            let authored = word != 0
                && (0.001..=100_000.0).contains(&value.abs())
                && (value - (value * 10_000.0).round() / 10_000.0).abs()
                    < 1e-6 * value.abs().max(1.0)
                && value.abs() != 1.0;
            if authored && class.0 != SCALAR_ARRAY {
                constants.push(PerkConstant {
                    offset,
                    class: class.0,
                    class_offset: class.1,
                    value,
                });
            }
        }

        let mut hop_ons = Vec::<HopOn>::new();
        for offset in (0..data.len().saturating_sub(7)).step_by(4) {
            let wide = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            let Some(pattern) = pm_tag64(wide) else { continue };
            if !has_reference(pattern, PATTERN_REFERENCE)
                || hop_ons.iter().any(|hop_on| hop_on.pattern == pattern)
            {
                continue;
            }
            hop_ons.push(HopOn {
                pattern,
                modifiers: pattern_modifiers(pattern),
            });
        }

        PerkData {
            effect_index,
            tag,
            sources: sources.into_iter().map(|(_, text)| text).collect(),
            scalars,
            labels,
            constants,
            hop_ons,
        }
    }
}

/// Stat modifier rows of the components a hop-on Pattern owns directly.
fn pattern_modifiers(pattern: TagHash) -> Vec<StatModifier> {
    let pm = package_manager();
    let Ok(data) = pm.read_tag(pattern) else {
        return vec![];
    };
    let mut components = vec![];
    for offset in (0..data.len().saturating_sub(3)).step_by(4) {
        let Some(tag) = read_u32(&data, offset).map(TagHash) else { break };
        if has_reference(tag, COMPONENT_REFERENCE) && !components.contains(&tag) {
            components.push(tag);
        }
    }
    let mut modifiers = vec![];
    for component in components {
        let Ok(data) = pm.read_tag(component) else { continue };
        for offset in (0..data.len().saturating_sub(0x6f)).step_by(4) {
            // Each row opens with a self reference and its class.
            if read_u32(&data, offset) != Some(component.0)
                || read_u32(&data, offset + 4) != Some(STAT_MODIFIER)
            {
                continue;
            }
            let (Some(value), Some(index), Some(mode)) = (
                read_f32(&data, offset + 0x10),
                read_u16(&data, offset + 0x5a),
                read_u32(&data, offset + 0x60),
            ) else {
                continue;
            };
            modifiers.push(StatModifier {
                component,
                value,
                index,
                mode,
            });
        }
    }
    modifiers
}

const ACCENT: Color32 = Color32::from_rgb(120, 200, 255);
const POSITIVE: Color32 = Color32::from_rgb(110, 220, 150);
const NEGATIVE: Color32 = Color32::from_rgb(240, 120, 110);

fn chip(ui: &mut egui::Ui, text: impl Into<String>, color: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.16))
        .stroke(egui::Stroke::new(1.0, color.gamma_multiply(0.6)))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(7, 2))
        .show(ui, |ui| ui.label(RichText::new(text).monospace().color(color)))
        .response
}

/// The Gear detail section: every perk of the item, unprocessed.
pub(super) fn perk_data_panel(ui: &mut egui::Ui, perks: &[PerkData]) {
    if perks.is_empty() {
        return;
    }
    ui.add_space(12.0);
    ui.heading("Perk data");
    ui.weak("Raw values from the perk tags and their hop-on Patterns.");
    ui.separator();
    for perk in perks {
        egui::CollapsingHeader::new(RichText::new(perk.title()).strong())
            .id_salt(("perk_data", perk.tag, perk.effect_index))
            .default_open(true)
            .show(ui, |ui| perk_body(ui, perk));
    }
}

fn perk_body(ui: &mut egui::Ui, perk: &PerkData) {
    ui.weak(format!("effect {:#x} · perk tag {}", perk.effect_index, perk.tag));

    if !perk.scalars.is_empty() {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Scalars");
            for scalar in &perk.scalars {
                chip(ui, format!("×{scalar}"), ACCENT);
            }
        });
    }
    if !perk.labels.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.label("Labels");
            for label in &perk.labels {
                match label.name {
                    Some(name) => chip(ui, name, Color32::from_rgb(230, 200, 120))
                        .on_hover_text(format!("{:08X}", label.hash)),
                    None => chip(ui, format!("#{:08X}", label.hash), Color32::GRAY),
                };
            }
        });
    }

    for hop_on in &perk.hop_ons {
        ui.add_space(6.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Hop-on");
                ui.monospace(hop_on.pattern.to_string());
                if hop_on.modifiers.is_empty() {
                    ui.weak("no stat modifiers");
                }
            });
            if hop_on.modifiers.is_empty() {
                return;
            }
            egui::Grid::new(("perk_hop_on", perk.tag, perk.effect_index, hop_on.pattern))
                .num_columns(3)
                .striped(true)
                .spacing([18.0, 4.0])
                .show(ui, |ui| {
                    for modifier in &hop_on.modifiers {
                        let (stat, amount) = modifier.describe();
                        let weaker = match modifier.mode {
                            2 => modifier.value < 1.0,
                            _ => modifier.value < 0.0,
                        };
                        ui.label(stat);
                        ui.label(
                            RichText::new(amount)
                                .monospace()
                                .strong()
                                .color(if weaker { NEGATIVE } else { POSITIVE }),
                        );
                        ui.weak(format!(
                            "index {} · mode {} · {}",
                            modifier.index, modifier.mode, modifier.component
                        ));
                        ui.end_row();
                    }
                });
        });
    }

    if !perk.constants.is_empty() {
        egui::CollapsingHeader::new(format!("Other constants ({})", perk.constants.len()))
            .id_salt(("perk_constants", perk.tag, perk.effect_index))
            .show(ui, |ui| {
                egui::Grid::new(("perk_constant_grid", perk.tag, perk.effect_index))
                    .num_columns(3)
                    .striped(true)
                    .spacing([18.0, 2.0])
                    .show(ui, |ui| {
                        ui.weak("offset");
                        ui.weak("value");
                        ui.weak("after structure");
                        ui.end_row();
                        for constant in &perk.constants {
                            ui.monospace(format!("{:#06x}", constant.offset));
                            ui.label(
                                RichText::new(format!("{}", constant.value))
                                    .monospace()
                                    .color(ACCENT),
                            );
                            ui.monospace(format!(
                                "{:08X} +{:#x}",
                                constant.class,
                                constant.offset - constant.class_offset
                            ));
                            ui.end_row();
                        }
                    });
            });
    }
    if !perk.sources.is_empty() {
        egui::CollapsingHeader::new(format!("Source paths ({})", perk.sources.len()))
            .id_salt(("perk_sources", perk.tag, perk.effect_index))
            .show(ui, |ui| {
                for source in &perk.sources {
                    ui.add(egui::Label::new(RichText::new(source).monospace().weak()).wrap());
                }
            });
    }
}

/// Plain names tried against label hashes (FNV-1 of the authored name).
const LABEL_WORDS: &[&str] = &[
    "precision", "shotgun", "ammo_light", "ammo_heavy", "ammo_mips", "ammo_volt", "ammo_special",
    "pistol", "sidearm", "smg", "rifle", "sniper", "lmg", "railgun", "melee", "grenade",
    "ability", "explosive", "weapon", "player", "enemy", "hostile", "ally", "uesc", "runner",
    "shield", "health", "headshot", "body", "volt", "ballistic", "light", "heavy",
];

fn fnv1(text: &str) -> u32 {
    text.bytes().fold(0x811c_9dc5u32, |hash, byte| {
        hash.wrapping_mul(0x0100_0193) ^ u32::from(byte)
    })
}

fn has_reference(tag: TagHash, reference: u32) -> bool {
    package_manager()
        .get_entry(tag)
        .is_some_and(|entry| entry.reference == reference)
}

fn pm_tag64(wide: u64) -> Option<TagHash> {
    package_manager()
        .lookup
        .tag64_entries
        .get(&wide)
        .map(|entry| entry.hash32)
}

fn resolve_tag64_union(data: &[u8], offset: usize) -> Option<TagHash> {
    if read_u32(data, offset + 4)? != 0 {
        let tag = TagHash(read_u32(data, offset)?);
        package_manager().get_entry(tag).map(|_| tag)
    } else {
        pm_tag64(read_u64(data, offset + 8)?)
    }
}

/// `(first row offset, row count)` of every array of `class`.
fn arrays(data: &[u8], class: u32) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..data.len().saturating_sub(19))
        .step_by(4)
        .filter_map(move |offset| {
            if read_u32(data, offset)? != ARRAY_MARKER || read_u32(data, offset + 12)? != class {
                return None;
            }
            let count = usize::try_from(read_u64(data, offset + 4)?).ok()?;
            // A corrupt count must not walk far outside the tag.
            (count <= data.len() / 4).then_some((offset + 20, count))
        })
}

fn ascii_runs(data: &[u8], minimum: usize) -> Vec<(usize, String)> {
    let mut runs = vec![];
    let mut start = None;
    for (offset, byte) in data.iter().chain(std::iter::once(&0)).enumerate() {
        let printable = (0x20..0x7f).contains(byte);
        match (start, printable) {
            (None, true) => start = Some(offset),
            (Some(begin), false) => {
                if offset - begin >= minimum {
                    runs.push((begin, String::from_utf8_lossy(&data[begin..offset]).into_owned()));
                }
                start = None;
            }
            _ => {}
        }
    }
    runs
}

fn table_range(data: &[u8], header: usize, stride: usize) -> Option<std::ops::Range<usize>> {
    let count = usize::try_from(read_u64(data, header)?).ok()?;
    let relative = i64::from_le_bytes(data.get(header + 8..header + 0x10)?.try_into().ok()?);
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

fn read_u16(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        data.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn read_f32(data: &[u8], offset: usize) -> Option<f32> {
    read_u32(data, offset).map(f32::from_bits)
}

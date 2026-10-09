use std::collections::{BTreeMap, BTreeSet, HashSet};

use tiger_pkg::TagHash;

use crate::{
    geometry::{geometry_compute_technique, WireframePreview},
    render::{
        technique::TechniqueDescriptor,
        tfx::{marathon_global_channel_table, TfxRuntimeInputs, TfxValue},
    },
};

/// The two numeric TFX channel namespaces exposed by the model preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ChannelDomain {
    Object,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ChannelKey {
    pub(crate) domain: ChannelDomain,
    pub(crate) id: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelRow {
    pub(crate) domain: ChannelDomain,
    pub(crate) id: u32,
    /// Authored/package hash. Global `id` is a runtime table index, so it is
    /// deliberately kept separate from this optional package identifier.
    pub(crate) hash: Option<u32>,
    /// Resolved source/default value. `None` means unknown or conflicting.
    pub(crate) default_value: Option<[f32; 4]>,
    /// Last committed editor value. `None` means the unresolved source value.
    pub(crate) current_value: Option<[f32; 4]>,
    pub(crate) buffers: [String; 4],
    pub(crate) errors: [Option<String>; 4],
}

impl ChannelRow {
    pub(crate) fn key(&self) -> ChannelKey {
        ChannelKey {
            domain: self.domain,
            id: self.id,
        }
    }
}

/// Pure channel discovery/editor state. GPU code can consume `apply` without
/// making this module depend on egui or a render frame.
#[derive(Debug, Clone)]
pub(crate) struct ModelChannels {
    rows: Vec<ChannelRow>,
    object_overrides: BTreeMap<u32, [f32; 4]>,
    global_overrides: BTreeMap<u32, [f32; 4]>,
    original_inputs: TfxRuntimeInputs,
    revision: u64,
}

impl ModelChannels {
    pub(crate) fn discover(
        wireframe: &WireframePreview,
        inputs: &TfxRuntimeInputs,
    ) -> Self {
        let (object_ids, global_ids) = collect_used_channels(wireframe);
        let global_table = marathon_global_channel_table();

        let mut rows = Vec::with_capacity(object_ids.len() + global_ids.len());
        for id in object_ids {
            let value = resolved_object_value(inputs, id);
            rows.push(make_row(
                ChannelDomain::Object,
                id,
                Some(id),
                value,
            ));
        }
        for id in global_ids {
            let value = resolved_global_value(inputs, id, global_table.as_ref());
            let hash = global_table
                .as_ref()
                .and_then(|table| table.channel_ids.get(id as usize).copied());
            rows.push(make_row(ChannelDomain::Global, id, hash, value));
        }

        Self {
            rows,
            object_overrides: BTreeMap::new(),
            global_overrides: BTreeMap::new(),
            original_inputs: inputs.clone(),
            revision: 0,
        }
    }

    pub(crate) fn rows(&self) -> &[ChannelRow] {
        &self.rows
    }

    pub(crate) fn rows_mut(&mut self) -> &mut [ChannelRow] {
        &mut self.rows
    }

    /// Restore discovered channel keys from the captured runtime input, then
    /// overlay committed edits. This keeps `for_geometry` from losing an
    /// object edit when it replaces the root object map.
    pub(crate) fn apply(&self, inputs: &mut TfxRuntimeInputs) {
        for row in &self.rows {
            match row.domain {
                ChannelDomain::Object => {
                    if !self.original_inputs.driven_channels.contains(&row.id) {
                        inputs.driven_channels.remove(&row.id);
                    }
                    restore_value(
                        &mut inputs.object_channels,
                        &self.original_inputs.object_channels,
                        row.id,
                    );
                    let geometries = inputs
                        .geometry_object_channels
                        .keys()
                        .chain(self.original_inputs.geometry_object_channels.keys())
                        .copied()
                        .collect::<HashSet<_>>();
                    for geometry in geometries {
                        let baseline = self
                            .original_inputs
                            .geometry_object_channels
                            .get(&geometry);
                        if let Some(channels) = inputs.geometry_object_channels.get_mut(&geometry) {
                            if let Some(baseline) = baseline {
                                restore_value(channels, baseline, row.id);
                            } else {
                                channels.remove(&row.id);
                            }
                        } else if let Some(baseline) = baseline {
                            let mut channels = BTreeMap::new();
                            if let Some(value) = baseline.get(&row.id) {
                                channels.insert(row.id, value.clone());
                            }
                            inputs.geometry_object_channels.insert(geometry, channels);
                        }
                    }
                }
                ChannelDomain::Global => {
                    restore_value(
                        &mut inputs.global_channels,
                        &self.original_inputs.global_channels,
                        row.id,
                    );
                }
            }
        }
        self.apply_overrides(inputs);
    }

    /// Overlay committed values without restoring a newly rebuilt runtime
    /// input. Use this after geometry/material rebuilds.
    pub(crate) fn apply_overrides(&self, inputs: &mut TfxRuntimeInputs) {
        for (&id, value) in &self.object_overrides {
            let value = TfxValue::Vector(*value);
            inputs.driven_channels.insert(id);
            inputs.object_channels.insert(id, value.clone());
            for channels in inputs.geometry_object_channels.values_mut() {
                channels.insert(id, value.clone());
            }
        }
        for (&id, value) in &self.global_overrides {
            inputs
                .global_channels
                .insert(id, TfxValue::Vector(*value));
        }
    }

    /// Parse all four text fields as finite f32 values. Blank all-four means
    /// restore the captured/default row value; a partially blank row is rejected.
    pub(crate) fn commit(&mut self) -> Result<bool, String> {
        let mut parsed = Vec::with_capacity(self.rows.len());
        let mut fields = Vec::<(ChannelKey, usize, String)>::new();

        for row in &mut self.rows {
            match parse_buffers(&row.buffers, row.default_value) {
                Ok(value) => {
                    row.errors = empty_errors();
                    parsed.push((row.key(), value));
                }
                Err(errors) => {
                    row.errors = errors.clone();
                    for (lane, message) in errors.into_iter().enumerate() {
                        if let Some(message) = message {
                            fields.push((row.key(), lane, message));
                        }
                    }
                }
            }
        }

        if !fields.is_empty() {
            return Err(fields
                .into_iter()
                .map(|(key, lane, message)| {
                    format!("{:?} 0x{:08X} lane {}: {}", key.domain, key.id, lane, message)
                })
                .collect::<Vec<_>>()
                .join("; "));
        }

        let mut changed = false;
        for (key, value) in parsed {
            let Some(row) = self.rows.iter_mut().find(|row| row.key() == key) else {
                continue;
            };
            if !same_value(row.current_value, value) {
                changed = true;
            }
            row.current_value = value;
            match key.domain {
                ChannelDomain::Object => {
                    update_override(&mut self.object_overrides, key.id, value, row.default_value)
                }
                ChannelDomain::Global => {
                    update_override(&mut self.global_overrides, key.id, value, row.default_value)
                }
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
        Ok(changed)
    }

    pub(crate) fn reset(&mut self) -> bool {
        let was_dirty = self.is_dirty();
        self.object_overrides.clear();
        self.global_overrides.clear();
        for row in &mut self.rows {
            row.current_value = row.default_value;
            row.buffers = buffers_for(row.default_value);
            row.errors = empty_errors();
        }
        if was_dirty {
            self.revision = self.revision.wrapping_add(1);
        }
        was_dirty
    }

    pub(crate) fn is_dirty(&self) -> bool {
        if !self.object_overrides.is_empty() || !self.global_overrides.is_empty() {
            return true;
        }
        self.rows.iter().any(|row| {
            row.buffers != buffers_for(row.current_value)
                || row.errors.iter().any(Option::is_some)
        })
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Carry edits across a rediscovery when the preview's geometry/material
    /// set changes. Keys that disappeared are dropped by the new row set.
    pub(crate) fn carry_overrides_from(&mut self, previous: &Self) -> bool {
        let old_object = previous.object_overrides.clone();
        let old_global = previous.global_overrides.clone();
        self.object_overrides = old_object
            .into_iter()
            .filter(|(id, _)| {
                self.rows
                    .iter()
                    .any(|row| row.domain == ChannelDomain::Object && row.id == *id)
            })
            .collect();
        self.global_overrides = old_global
            .into_iter()
            .filter(|(id, _)| {
                self.rows
                    .iter()
                    .any(|row| row.domain == ChannelDomain::Global && row.id == *id)
            })
            .collect();
        for row in &mut self.rows {
            let value = match row.domain {
                ChannelDomain::Object => self.object_overrides.get(&row.id).copied(),
                ChannelDomain::Global => self.global_overrides.get(&row.id).copied(),
            };
            if let Some(value) = value {
                row.current_value = Some(value);
                row.buffers = buffers_for(Some(value));
            }
        }
        let changed = !self.object_overrides.is_empty() || !self.global_overrides.is_empty();
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
        changed
    }
}

fn collect_used_channels(wireframe: &WireframePreview) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut techniques = HashSet::<TagHash>::new();
    let mut add_range = |technique: Option<TagHash>, source: Option<usize>, part: usize,
                         start: u32, count: u32| {
        if let Some(technique) = technique {
            techniques.insert(technique);
        }
        let Some(source) = source else { return };
        let Some(geometry) = wireframe.authored_inputs.get(source).map(|input| input.geometry)
        else {
            return;
        };
        let Some(end) = start.checked_add(count) else { return };
        if let Some(technique) = geometry_compute_technique(geometry, part, start..end) {
            techniques.insert(technique);
        }
    };

    for range in &wireframe.material_ranges {
        let end = range.index_start.saturating_add(range.index_count).min(wireframe.indices.len());
        if range.index_start.min(wireframe.indices.len()) >= end { continue; }
        if let Some(draw) = range.authored_draw {
            add_range(
                range.technique,
                range.authored_source,
                draw.part_index,
                draw.source_index_start,
                draw.source_index_count,
            );
        } else if range.technique.is_some() {
            add_range(range.technique, None, 0, 0, 0);
        }
    }
    for range in &wireframe.authored_shadow_ranges {
        if range.source_index_count == 0 { continue; }
        add_range(
            range.technique,
            range.authored_source,
            range.part_index,
            range.source_index_start,
            range.source_index_count,
        );
    }

    let mut object_ids = BTreeSet::new();
    let mut global_ids = BTreeSet::new();
    for technique in techniques {
        let Some(descriptor) = TechniqueDescriptor::load(technique) else {
            continue;
        };
        for stage in descriptor.stages {
            for op in stage.tfx.ops {
                match op.name {
                    "push_object_channel" => {
                        if let Some(id) = parse_hex_id(&op.detail) {
                            object_ids.insert(id);
                        }
                    }
                    "push_global_channel" => {
                        if let Some(id) = parse_index(&op.detail) {
                            global_ids.insert(id);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    (object_ids, global_ids)
}

fn make_row(
    domain: ChannelDomain,
    id: u32,
    hash: Option<u32>,
    value: Option<[f32; 4]>,
) -> ChannelRow {
    ChannelRow {
        domain,
        id,
        hash,
        default_value: value,
        current_value: value,
        buffers: buffers_for(value),
        errors: empty_errors(),
    }
}

fn resolved_object_value(
    inputs: &TfxRuntimeInputs,
    id: u32,
) -> Option<[f32; 4]> {
    let mut values = Vec::new();
    if let Some(value) = inputs.object_channels.get(&id) {
        values.push(numeric_value(value)?);
    }
    for channels in inputs.geometry_object_channels.values() {
        if let Some(value) = channels.get(&id) {
            values.push(numeric_value(value)?);
        }
    }
    if values.is_empty() {
        return None;
    }
    let first = values[0];
    values.iter().all(|value| same_value(Some(*value), Some(first))).then_some(first)
}

fn resolved_global_value(
    inputs: &TfxRuntimeInputs,
    id: u32,
    table: Option<&crate::render::tfx::MarathonGlobalChannelTable>,
) -> Option<[f32; 4]> {
    if let Some(value) = inputs.global_channels.get(&id) {
        return numeric_value(value);
    }
    table
        .and_then(|table| table.default_values.get(id as usize).copied())
        .filter(|value| value.iter().all(|value| value.is_finite()))
}

fn numeric_value(value: &TfxValue) -> Option<[f32; 4]> {
    let value = match value {
        TfxValue::Scalar(value) => [*value; 4],
        TfxValue::Vector(value) => *value,
        TfxValue::TextureBinding { .. } | TfxValue::Unknown(_) => return None,
    };
    value.iter().all(|value| value.is_finite()).then_some(value)
}

fn restore_value(
    destination: &mut BTreeMap<u32, TfxValue>,
    baseline: &BTreeMap<u32, TfxValue>,
    id: u32,
) {
    if let Some(value) = baseline.get(&id) {
        destination.insert(id, value.clone());
    } else {
        destination.remove(&id);
    }
}

fn update_override(
    overrides: &mut BTreeMap<u32, [f32; 4]>,
    id: u32,
    value: Option<[f32; 4]>,
    default: Option<[f32; 4]>,
) {
    if same_value(value, default) {
        overrides.remove(&id);
    } else if let Some(value) = value {
        overrides.insert(id, value);
    } else {
        overrides.remove(&id);
    }
}

pub(crate) fn parse_hex_id(detail: &str) -> Option<u32> {
    let start = detail
        .find("0x")
        .map(|start| start + 2)
        .or_else(|| detail.find("0X").map(|start| start + 2))?;
    let digits = detail[start..]
        .chars()
        .take_while(|character| character.is_ascii_hexdigit())
        .collect::<String>();
    (!digits.is_empty())
        .then(|| u32::from_str_radix(&digits, 16).ok())
        .flatten()
}

fn parse_index(detail: &str) -> Option<u32> {
    let start = detail.find("index=")? + "index=".len();
    let digits = detail[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn empty_errors() -> [Option<String>; 4] {
    std::array::from_fn(|_| None)
}

fn buffers_for(value: Option<[f32; 4]>) -> [String; 4] {
    match value {
        Some(value) => value.map(|value| value.to_string()),
        None => std::array::from_fn(|_| String::new()),
    }
}

fn parse_buffers(
    buffers: &[String; 4],
    default: Option<[f32; 4]>,
) -> Result<Option<[f32; 4]>, [Option<String>; 4]> {
    let trimmed = buffers.each_ref().map(|buffer| buffer.trim().to_owned());
    if trimmed.iter().all(String::is_empty) {
        return Ok(default);
    }

    let mut errors = empty_errors();
    let mut values = [0.0; 4];
    for (lane, value) in trimmed.iter().enumerate() {
        if value.is_empty() {
            errors[lane] = Some("enter a finite f32".into());
            continue;
        }
        match value.parse::<f32>() {
            Ok(value) if value.is_finite() => values[lane] = value,
            Ok(_) => errors[lane] = Some("value must be finite".into()),
            Err(_) => errors[lane] = Some("enter a finite f32".into()),
        }
    }
    errors.iter().all(Option::is_none).then_some(Ok(Some(values))).unwrap_or(Err(errors))
}

fn same_value(left: Option<[f32; 4]>, right: Option<[f32; 4]>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.map(f32::to_bits) == right.map(f32::to_bits),
        (None, None) => true,
        _ => false,
    }
}

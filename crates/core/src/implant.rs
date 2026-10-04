//! Authored Marathon implant stat groups. See docs/research/marathon-implant-stat-link.md.

use std::{collections::BTreeMap, ops::Range};

use anyhow::{Context, Result, ensure};
use tiger_pkg::{PackageManager, TagHash};

const REGISTRY: u32 = 0x8080_97B6;
const GROUP_TABLE: u32 = 0x8080_914C;
const SEMANTIC_TABLE: u32 = 0x8080_92D6;
const ARRAY: u32 = 0x8080_BFCD;
const BINDING_COMPONENT: u32 = 0x8080_92B2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplantStat {
    pub semantic_id: u32,
    pub raw_name_hash: u32,
    pub value: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplantStatBinding {
    pub group_index: u16,
    pub group_hash: u32,
    pub level: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImplantStats {
    pub bindings: Vec<ImplantStatBinding>,
    pub stats: Vec<ImplantStat>,
}

struct StatGroup {
    hash: u32,
    levels: Vec<Vec<(u32, i32)>>,
}

pub struct ImplantStatResolver {
    pub group_table: TagHash,
    pub semantic_table: TagHash,
    groups: Vec<StatGroup>,
    semantics: BTreeMap<u32, u32>,
}

impl ImplantStatResolver {
    pub fn load(pm: &PackageManager) -> Result<Self> {
        // The registry chooses active tables across packages. Scanning one package
        // or choosing a table by tag order can silently select obsolete content.
        let registries = pm.get_all_by_reference(REGISTRY);
        ensure!(
            registries.len() == 1,
            "expected one active investment registry"
        );
        let registry = pm.read_tag(registries[0].0)?;
        let group_table = registered_table(pm, &registry, GROUP_TABLE)?;
        let semantic_table = registered_table(pm, &registry, SEMANTIC_TABLE)?;
        let groups = parse_groups(&pm.read_tag(group_table)?)?;
        let semantics = parse_semantics(&pm.read_tag(semantic_table)?)?;
        Ok(Self {
            group_table,
            semantic_table,
            groups,
            semantics,
        })
    }

    pub fn resolve(&self, definition: &[u8]) -> Result<DecodedImplantStats> {
        let mut components = (0..definition.len().saturating_sub(3))
            .step_by(4)
            .filter(|&o| read_u32(definition, o).ok() == Some(BINDING_COMPONENT));
        let component = components
            .next()
            .context("definition has no stat-group component")?;
        ensure!(
            components.next().is_none(),
            "ambiguous stat-group components"
        );
        let bindings = table_range(definition, component + 0xC, 8, 0x8080_92B4)?;
        let mut resolved_bindings = vec![];
        let mut values = vec![];
        for offset in bindings.step_by(8) {
            let group_index = read_u16(definition, offset)?;
            let level = read_u16(definition, offset + 2)?;
            let group = self
                .groups
                .get(usize::from(group_index))
                .with_context(|| format!("unknown stat group {group_index}"))?;
            // Binding levels are one-based positions in the group's level array.
            // Row +4 is not a unique key: shipped groups contain repeated values.
            let index = level
                .checked_sub(1)
                .context("stat-group level must be positive")?;
            let stats = group
                .levels
                .get(usize::from(index))
                .with_context(|| format!("group {group_index} has no level position {level}"))?;
            resolved_bindings.push(ImplantStatBinding {
                group_index,
                group_hash: group.hash,
                level,
            });
            // Preserve authored records; stacking behavior belongs to game logic.
            values.extend_from_slice(stats);
        }
        let stats = values
            .into_iter()
            .map(|(semantic_id, value)| {
                let raw_name_hash = *self
                    .semantics
                    .get(&semantic_id)
                    .with_context(|| format!("semantic {semantic_id} has no raw-name mapping"))?;
                Ok(ImplantStat {
                    semantic_id,
                    raw_name_hash,
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodedImplantStats {
            bindings: resolved_bindings,
            stats,
        })
    }
}

fn registered_table(pm: &PackageManager, registry: &[u8], reference: u32) -> Result<TagHash> {
    let mut matches = vec![];
    for offset in (0..registry.len().saturating_sub(7)).step_by(4) {
        if read_u32(registry, offset + 4)? != 1 {
            continue;
        }
        let tag = TagHash(read_u32(registry, offset)?);
        if pm
            .get_entry(tag)
            .is_some_and(|entry| entry.reference == reference)
        {
            matches.push(tag);
        }
    }
    matches.sort_unstable();
    matches.dedup();
    ensure!(
        matches.len() == 1,
        "registry has {} tables for class {reference:08X}",
        matches.len()
    );
    Ok(matches[0])
}

fn parse_groups(data: &[u8]) -> Result<Vec<StatGroup>> {
    table_range(data, 8, 0x58, 0x8080_9150)?
        .step_by(0x58)
        .map(|row| {
            let mut levels = vec![];
            for level_row in table_range(data, row + 8, 0x70, 0x8080_9159)?.step_by(0x70) {
                let stats = table_range(data, level_row + 0x50, 8, 0x8080_679D)?
                    .step_by(8)
                    .map(|offset| Ok((read_u32(data, offset)?, read_u32(data, offset + 4)? as i32)))
                    .collect::<Result<Vec<_>>>()?;
                levels.push(stats);
            }
            Ok(StatGroup {
                hash: read_u32(data, row)?,
                levels,
            })
        })
        .collect()
}

fn parse_semantics(data: &[u8]) -> Result<BTreeMap<u32, u32>> {
    // Inline counted dictionary: semantic u32, raw FNV-1 hash u32, flags u32.
    // The ID PRECEDES its hash; reading the following ID shifts every mapping.
    let count = usize::try_from(read_u32(data, 0x25C)?)?;
    let end = 0x260usize
        .checked_add(count.checked_mul(12).context("semantic count overflow")?)
        .context("semantic range overflow")?;
    let records = data
        .get(0x260..end)
        .context("truncated semantic dictionary")?;
    let mut result = BTreeMap::new();
    for record in records.chunks_exact(12) {
        ensure!(
            result
                .insert(read_u32(record, 0)?, read_u32(record, 4)?)
                .is_none(),
            "duplicate semantic"
        );
    }
    Ok(result)
}

fn table_range(data: &[u8], header: usize, stride: usize, class: u32) -> Result<Range<usize>> {
    let count = usize::try_from(read_u64(data, header)?)?;
    if count == 0 {
        return Ok(0..0);
    }
    let pointer = header.checked_add(8).context("pointer overflow")?;
    let relative = read_u64(data, pointer)? as i64;
    let start = i64::try_from(pointer)?
        .checked_add(relative)
        .and_then(|v| v.checked_add(16))
        .context("array pointer overflow")?;
    let start = usize::try_from(start)?;
    let marker = start
        .checked_sub(20)
        .context("invalid array marker location")?;
    ensure!(
        read_u32(data, marker)? == ARRAY && read_u32(data, marker + 12)? == class,
        "expected array class {class:08X} at {marker:X}"
    );
    ensure!(
        read_u64(data, marker + 4)? == count as u64,
        "array counts disagree"
    );
    let end = start
        .checked_add(count.checked_mul(stride).context("array count overflow")?)
        .context("array range overflow")?;
    ensure!(end <= data.len(), "array extends beyond tag");
    Ok(start..end)
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16> {
    let end = offset.checked_add(2).context("offset overflow")?;
    Ok(u16::from_le_bytes(
        data.get(offset..end).context("truncated u16")?.try_into()?,
    ))
}
fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    let end = offset.checked_add(4).context("offset overflow")?;
    Ok(u32::from_le_bytes(
        data.get(offset..end).context("truncated u32")?.try_into()?,
    ))
}
fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
    let end = offset.checked_add(8).context("offset overflow")?;
    Ok(u64::from_le_bytes(
        data.get(offset..end).context("truncated u64")?.try_into()?,
    ))
}

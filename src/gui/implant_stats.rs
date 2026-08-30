use quicktag_strings::localized::{LocalizedStringResolver, StringCache};
use tiger_pkg::{TagHash, package_manager};

const ARRAY_MARKER: u32 = 0x8080_BFCD;
const IMPLANT_PERK_ARRAY: u32 = 0x8080_924C;
const PERK_DEFINITION_REFERENCE: u32 = 0x8080_6B54;

#[derive(Clone, Debug, Default)]
pub(super) struct ImplantDetails {
    pub(super) effect: Option<String>,
    pub(super) stats: Vec<ImplantStat>,
}

#[derive(Clone, Debug)]
pub(super) struct ImplantStat {
    pub(super) name: String,
    pub(super) value: f32,
}

pub(super) struct ImplantResolver {
    perks: Vec<Vec<u8>>,
}

impl ImplantResolver {
    pub(super) fn load() -> Self {
        let perks = package_manager()
            .get_all_by_reference(PERK_DEFINITION_REFERENCE)
            .into_iter()
            .filter_map(|(tag, _)| package_manager().read_tag(tag).ok())
            .filter_map(|data| {
                let range = table_range(&data, 0x8, 0x28)?;
                Some(data[range].chunks_exact(0x28).map(Vec::from).collect())
            })
            .max_by_key(Vec::len)
            .unwrap_or_default();
        Self { perks }
    }

    pub(super) fn extract(
        &self,
        definition_tag: TagHash,
        strings: &StringCache,
        localized: &LocalizedStringResolver,
    ) -> Option<ImplantDetails> {
        let definition = package_manager().read_tag(definition_tag).ok()?;
        let perk_index = find_array_index(&definition, IMPLANT_PERK_ARRAY)?;
        let perk = self.perks.get(perk_index)?;
        let effect = localized_at(perk, 0x10, 0x14, strings, localized);
        effect.map(|effect| ImplantDetails {
            effect: Some(effect),
            stats: vec![],
        })
    }
}

fn find_array_index(data: &[u8], class: u32) -> Option<usize> {
    (0..data.len().saturating_sub(0x18))
        .step_by(4)
        .find_map(|offset| {
            (read_u32(data, offset) == Some(ARRAY_MARKER)
                && read_u64(data, offset + 4) == Some(1)
                && read_u32(data, offset + 0x0C) == Some(class))
            .then(|| read_u32(data, offset + 0x14))
            .flatten()
            .and_then(|index| usize::try_from(index).ok())
        })
}

fn localized_at(
    data: &[u8],
    scope_offset: usize,
    hash_offset: usize,
    strings: &StringCache,
    localized: &LocalizedStringResolver,
) -> Option<String> {
    let scope = read_u32(data, scope_offset)?;
    let hash = read_u32(data, hash_offset)?;
    localized.get(scope, hash).cloned().or_else(|| {
        let values = strings.get(&hash)?;
        (values.len() == 1).then(|| values[0].clone())
    })
}

fn table_range(data: &[u8], header: usize, record_size: usize) -> Option<std::ops::Range<usize>> {
    let count = usize::try_from(read_u64(data, header)?).ok()?;
    let relative = i64::from_le_bytes(data.get(header + 8..header + 0x10)?.try_into().ok()?);
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

use quicktag_strings::localized::{LocalizedStringPart, LocalizedStringResolver, StringCache};
use tiger_pkg::{TagHash, package_manager};

const ARRAY_MARKER: u32 = 0x8080_BFCD;
const ITEM_EFFECT_ARRAY: u32 = 0x8080_924C;
const EFFECT_DEFINITION_REFERENCE: u32 = 0x8080_6B54;

#[derive(Clone, Debug)]
pub(super) struct ItemEffect {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) description_parts: Vec<LocalizedStringPart>,
}

pub(super) struct ItemEffectResolver {
    effects: Vec<Vec<u8>>,
}

impl ItemEffectResolver {
    pub(super) fn load() -> Self {
        let effects = package_manager()
            .get_all_by_reference(EFFECT_DEFINITION_REFERENCE)
            .into_iter()
            .filter_map(|(tag, _)| package_manager().read_tag(tag).ok())
            .filter_map(|data| {
                let range = table_range(&data, 0x8, 0x28)?;
                Some(data[range].chunks_exact(0x28).map(Vec::from).collect())
            })
            .max_by_key(Vec::len)
            .unwrap_or_default();
        Self { effects }
    }

    pub(super) fn extract(
        &self,
        definition_tag: TagHash,
        strings: &StringCache,
        localized: &LocalizedStringResolver,
    ) -> Vec<ItemEffect> {
        self.extract_inner(definition_tag, strings, localized)
            .unwrap_or_default()
    }

    fn extract_inner(
        &self,
        definition_tag: TagHash,
        strings: &StringCache,
        localized: &LocalizedStringResolver,
    ) -> Option<Vec<ItemEffect>> {
        let definition = package_manager().read_tag(definition_tag).ok()?;
        let range = find_array_range(&definition, ITEM_EFFECT_ARRAY)?;
        Some(
            self.effects
                .get(range)?
                .iter()
                .filter_map(|effect| {
                    Some(ItemEffect {
                        name: localized_at(effect, 0x08, 0x0C, strings, localized)?,
                        description: localized_at(effect, 0x10, 0x14, strings, localized)?,
                        description_parts: localized_parts_at(effect, 0x10, 0x14, localized),
                    })
                })
                .collect(),
        )
    }
}

fn localized_parts_at(
    data: &[u8],
    scope_offset: usize,
    hash_offset: usize,
    localized: &LocalizedStringResolver,
) -> Vec<LocalizedStringPart> {
    let Some(scope) = read_u32(data, scope_offset) else {
        return vec![];
    };
    let Some(hash) = read_u32(data, hash_offset) else {
        return vec![];
    };
    localized.parts(scope, hash).unwrap_or_default().to_vec()
}

fn find_array_range(data: &[u8], class: u32) -> Option<std::ops::Range<usize>> {
    (0..data.len().saturating_sub(0x18))
        .step_by(4)
        .find_map(|offset| {
            if read_u32(data, offset) != Some(ARRAY_MARKER)
                || read_u32(data, offset + 0x0C) != Some(class)
            {
                return None;
            }
            let count = usize::try_from(read_u64(data, offset + 4)?).ok()?;
            let start = usize::try_from(read_u32(data, offset + 0x14)?).ok()?;
            let end = start.checked_add(count)?;
            (count > 0).then_some(start..end)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_multi_effect_array_range() {
        let mut data = vec![0; 0x20];
        data[0..4].copy_from_slice(&ARRAY_MARKER.to_le_bytes());
        data[4..12].copy_from_slice(&2_u64.to_le_bytes());
        data[0x0C..0x10].copy_from_slice(&ITEM_EFFECT_ARRAY.to_le_bytes());
        data[0x14..0x18].copy_from_slice(&0x1F2_u32.to_le_bytes());
        assert_eq!(
            find_array_range(&data, ITEM_EFFECT_ARRAY),
            Some(0x1F2..0x1F4)
        );
    }
}

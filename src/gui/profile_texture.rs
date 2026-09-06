use std::collections::HashSet;

use rustc_hash::FxHashMap;
use tiger_pkg::{TagHash, package_manager};

use crate::texture::Texture;

const PROFILE_REGISTRY_REFERENCE: u32 = 0x8080_71C0;
const PROFILE_REGISTRY_HEADER_SIZE: usize = 0x10;
const PROFILE_REGISTRY_RECORD_SIZE: usize = 0x20;
const PROFILE_REGISTRY_ICON_OFFSET: usize = 0x10;
const ICON_DEFINITION_REFERENCE: u32 = 0x8080_5335;
const ICON_DEFINITION_CONTAINER_OFFSET: usize = 0x18;
const ICON_CONTAINER_REFERENCE: u32 = 0x8080_5350;

pub(super) const PROFILE_BACKGROUND_TYPE: u16 = 0x127;
pub(super) const PROFILE_EMBLEM_TYPE: u16 = 0x128;

/// Resolves investment identity -> authored Profile art through the UI registry.
/// Background containers own portrait, large-banner, and small-banner textures.
pub(super) fn resolve_profile_textures(
    items: impl IntoIterator<Item = (u16, u32)>,
) -> FxHashMap<u32, Vec<TagHash>> {
    let requested = items
        .into_iter()
        .filter(|(type_code, _)| {
            matches!(*type_code, PROFILE_BACKGROUND_TYPE | PROFILE_EMBLEM_TYPE)
        })
        .map(|(type_code, internal_hash)| (internal_hash, type_code))
        .collect::<FxHashMap<_, _>>();
    let mut result = FxHashMap::default();

    let mut registries = package_manager().get_all_by_reference(PROFILE_REGISTRY_REFERENCE);
    registries.sort_unstable_by_key(|(registry, _)| *registry);
    for (registry, _) in registries {
        let Ok(data) = package_manager().read_tag(registry) else {
            continue;
        };
        let Some(records) = data.get(PROFILE_REGISTRY_HEADER_SIZE..) else {
            continue;
        };
        for record in records.chunks_exact(PROFILE_REGISTRY_RECORD_SIZE) {
            let Some(internal_hash) = read_u32(record, 0) else {
                continue;
            };
            let Some(&type_code) = requested.get(&internal_hash) else {
                continue;
            };
            let Some(icon_definition) = read_tag(record, PROFILE_REGISTRY_ICON_OFFSET) else {
                continue;
            };
            if package_manager()
                .get_entry(icon_definition)
                .is_none_or(|entry| entry.reference != ICON_DEFINITION_REFERENCE)
            {
                continue;
            }
            if let Some(textures) = profile_textures_from_definition(icon_definition, type_code) {
                result.insert(internal_hash, textures);
            }
        }
    }
    result
}

fn profile_textures_from_definition(definition: TagHash, type_code: u16) -> Option<Vec<TagHash>> {
    let definition = package_manager().read_tag(definition).ok()?;
    let container = read_tag(&definition, ICON_DEFINITION_CONTAINER_OFFSET)?;
    (package_manager().get_entry(container)?.reference == ICON_CONTAINER_REFERENCE).then_some(())?;
    let container = package_manager().read_tag(container).ok()?;
    let mut seen = HashSet::new();
    let mut textures = container
        .chunks_exact(4)
        .filter_map(|bytes| {
            let texture = TagHash(u32::from_le_bytes(bytes.try_into().ok()?));
            Texture::load_desc(texture).ok()?;
            let expected = match type_code {
                PROFILE_BACKGROUND_TYPE | PROFILE_EMBLEM_TYPE => true,
                _ => false,
            };
            (expected && seen.insert(texture)).then_some(texture)
        })
        .collect::<Vec<_>>();
    textures.sort_unstable();
    (!textures.is_empty()).then_some(textures)
}

fn read_tag(data: &[u8], offset: usize) -> Option<TagHash> {
    read_u32(data, offset).map(TagHash)
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

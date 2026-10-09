use std::collections::HashSet;

use tiger_pkg::{TagHash, TagHash64, package_manager};

use crate::texture::Texture;

const PATTERN_REFERENCE: u32 = 0x8080_BAAD;
const PATTERN_COMPONENT_REFERENCE: u32 = 0x8080_BADB;
const RENDER_CONTAINER_REFERENCE: u32 = 0x8080_85DA;
const GEOMETRY_REFERENCE: u32 = 0x8080_31D8;

/// Resolves the color texture authored by a sticker's investment Pattern.
///
/// Sticker display names are localization, not asset identifiers. The durable
/// relation is Pattern -> component -> render container -> geometry -> texture.
pub(super) fn resolve_sticker_texture(pattern: TagHash) -> Option<TagHash> {
    (package_manager().get_entry(pattern)?.reference == PATTERN_REFERENCE).then_some(())?;
    let pattern = package_manager().read_tag(pattern).ok()?;
    let components = direct_tags_with_reference(&pattern, PATTERN_COMPONENT_REFERENCE);

    let mut candidates = vec![];
    for component in components {
        let Ok(component) = package_manager().read_tag(component) else {
            continue;
        };
        for container in direct_tags_with_reference(&component, RENDER_CONTAINER_REFERENCE) {
            let Ok(container) = package_manager().read_tag(container) else {
                continue;
            };
            for geometry in direct_tags_with_reference(&container, GEOMETRY_REFERENCE) {
                let Ok(geometry) = package_manager().read_tag(geometry) else {
                    continue;
                };
                candidates.extend(wide_texture_tags(&geometry));
            }
        }
    }

    let candidates = candidates
        .into_iter()
        .filter_map(|tag| {
            let desc = Texture::load_desc(tag).ok()?;
            desc.format
                .is_srgb()
                .then_some((u64::from(desc.width) * u64::from(desc.height), tag))
        })
        .collect::<Vec<_>>();
    let largest_area = candidates.iter().map(|(area, _)| *area).max()?;
    let mut largest = candidates
        .into_iter()
        .filter_map(|(area, tag)| (area == largest_area).then_some(tag))
        .collect::<HashSet<_>>();
    (largest.len() == 1)
        .then(|| largest.drain().next())
        .flatten()
}

fn direct_tags_with_reference(data: &[u8], reference: u32) -> Vec<TagHash> {
    data.chunks_exact(4)
        .filter_map(|bytes| {
            let tag = TagHash(u32::from_le_bytes(bytes.try_into().ok()?));
            (package_manager().get_entry(tag)?.reference == reference).then_some(tag)
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn wide_texture_tags(data: &[u8]) -> Vec<TagHash> {
    (0..data.len().saturating_sub(7))
        .filter_map(|offset| {
            let wide = u64::from_le_bytes(data.get(offset..offset + 8)?.try_into().ok()?);
            package_manager()
                .lookup
                .tag64_entries
                .get(&TagHash64(wide).0)
                .map(|entry| entry.hash32)
        })
        .filter(|tag| Texture::load_desc(*tag).is_ok())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

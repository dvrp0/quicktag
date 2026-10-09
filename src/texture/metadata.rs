//! Metadata queries retain only the validated descriptor, never texture pixels.

use std::sync::LazyLock;

use parking_lot::Mutex;

use tiger_pkg::{PackageManager, TagHash, package_manager};

use super::{Texture, TextureHeaderGeneric};
use crate::asset_cache::AssetCache;

/// Serialized PC atlas metadata, not dimensions-derived padding or defaults.
#[derive(Debug, Clone, Copy)]
pub struct TextureExpressionMetadata {
    pub tiling_params: [f32; 4],
    pub tile_count: u16,
}

impl TextureExpressionMetadata {
    pub fn evaluate(&self, operation: &str, fields: u8) -> Option<[f32; 4]> {
        let mut result = [0.0; 4];
        for (component, value) in result.iter_mut().enumerate() {
            let selected = usize::from((fields >> (6 - component * 2)) & 3);
            *value = match operation {
                "push_tex_tiling_params" => self.tiling_params[selected],
                // The observed layer-count operation selects x four times.
                // Other components have no recovered source contract.
                "push_tex_tile_layer_count" if selected == 0 => f32::from(self.tile_count),
                _ => return None,
            };
        }
        result.iter().all(|value| value.is_finite()).then_some(result)
    }
}

static DESCRIPTORS: LazyLock<Mutex<AssetCache<PackageManager, TagHash, TextureHeaderGeneric>>> =
    LazyLock::new(|| Mutex::new(AssetCache::new(1024 * 1024)));

pub(super) fn descriptor(hash: TagHash) -> anyhow::Result<TextureHeaderGeneric> {
    let manager = package_manager();
    if let Some(value) = DESCRIPTORS.lock().get(&manager, &hash) {
        return Ok(value.clone());
    }
    // Keep the original read/validation path on a miss. A header alone does
    // not prove the payload exists or (on consoles) can be deswizzled.
    // Failed reads are deliberately retried rather than retained.
    let (descriptor, _, _) = Texture::load_data_d2(hash, false)?;
    DESCRIPTORS
        .lock()
        .insert(&manager, hash, descriptor.clone(), 0);
    Ok(descriptor)
}

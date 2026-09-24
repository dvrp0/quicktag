//! Metadata queries retain only the validated descriptor, never texture pixels.

use std::sync::LazyLock;

use parking_lot::Mutex;

use tiger_pkg::{PackageManager, TagHash, package_manager};

use super::{Texture, TextureHeaderGeneric};
use crate::asset_cache::AssetCache;

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

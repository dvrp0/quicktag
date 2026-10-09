//! Bounded raw mesh and composed model caches. Raw meshes are cloned before
//! mutation; composed keys include every attachment transform and instance value.

use std::sync::{LazyLock, Weak};

use parking_lot::Mutex;
use tiger_pkg::PackageManager;

use super::*;
use crate::asset_cache::AssetCache;

const PREVIEW_BUDGET: usize = 256 * 1024 * 1024;
const MESH_BUDGET: usize = 64 * 1024 * 1024;

#[derive(Hash, PartialEq, Eq)]
struct AttachmentKey {
    pattern: TagHash,
    geometry: TagHash,
    family: u32,
    variant: u32,
    bone: u32,
    rotation: [u32; 4],
    translation: [u32; 3],
    scale: u32,
    rarity: Option<WeaponModRarity>,
    unique_id: u32,
}

impl From<&ResolvedWeaponModAttachment> for AttachmentKey {
    fn from(value: &ResolvedWeaponModAttachment) -> Self {
        Self {
            pattern: value.pattern,
            geometry: value.geometry,
            family: value.pose.family_id,
            variant: value.pose.variant_id,
            bone: value.pose.bone_index,
            rotation: value.pose.rotation.map(f32::to_bits),
            translation: value.pose.translation.map(f32::to_bits),
            scale: value.pose.scale.to_bits(),
            rarity: value.rarity,
            unique_id: value.unique_id.to_bits(),
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
struct PreviewKey {
    tag: TagHash,
    reference: u32,
    label: &'static str,
    models: Vec<TagHash>,
    attachments: Vec<AttachmentKey>,
}

struct PreviewCache {
    graph: Weak<TagCache>,
    values: AssetCache<PackageManager, PreviewKey, ModelPreview>,
}

impl PreviewCache {
    fn select_graph(&mut self, graph: &Arc<TagCache>) {
        if !Weak::ptr_eq(&self.graph, &Arc::downgrade(graph)) {
            self.graph = Arc::downgrade(graph);
            self.values = AssetCache::new(PREVIEW_BUDGET);
        }
    }
}

static PREVIEWS: LazyLock<Mutex<PreviewCache>> = LazyLock::new(|| {
    Mutex::new(PreviewCache {
        graph: Weak::new(),
        values: AssetCache::new(PREVIEW_BUDGET),
    })
});

type Mesh = (MeshSourcePreview, WireframePreview);
static MESHES: LazyLock<Mutex<AssetCache<PackageManager, (TagHash, u32), Mesh>>> =
    LazyLock::new(|| Mutex::new(AssetCache::new(MESH_BUDGET)));

pub(super) fn invalidate() {
    PREVIEWS.lock().values.clear();
    MESHES.lock().clear();
}

pub(super) fn model(
    graph: &Arc<TagCache>,
    tag: TagHash,
    reference: u32,
    label: &'static str,
    models: &[TagHash],
    attachments: &[ResolvedWeaponModAttachment],
    load: impl FnOnce() -> ModelPreview,
) -> ModelPreview {
    let manager = package_manager();
    let key = PreviewKey {
        tag,
        reference,
        label,
        models: models.to_vec(),
        attachments: attachments.iter().map(AttachmentKey::from).collect(),
    };
    {
        let mut cache = PREVIEWS.lock();
        cache.select_graph(graph);
        if let Some(preview) = cache.values.get(&manager, &key) {
            return preview.clone();
        }
    }
    // Never hold a cache lock during package reads or recursive decoding.
    let preview = load();
    if preview.wireframe.is_some() {
        let bytes =
            model_heap_bytes(&preview) + vec_bytes(&key.models) + vec_bytes(&key.attachments);
        let mut cache = PREVIEWS.lock();
        cache.select_graph(graph);
        cache.values.insert(&manager, key, preview.clone(), bytes);
    }
    preview
}

pub(super) fn mesh(
    tag: TagHash,
    entry: &UEntryHeader,
    load: impl FnOnce() -> Option<Mesh>,
) -> Option<Mesh> {
    let manager = package_manager();
    let key = (tag, entry.reference);
    if let Some(mesh) = MESHES.lock().get(&manager, &key) {
        return Some(mesh.clone());
    }
    let mesh = load()?;
    let bytes = vec_bytes(&mesh.0.shader_constants) + wireframe_heap_bytes(&mesh.1);
    MESHES.lock().insert(&manager, key, mesh.clone(), bytes);
    Some(mesh)
}

fn vec_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * std::mem::size_of::<T>()
}

fn optional_vec_bytes<T>(values: &Option<Vec<T>>) -> usize {
    values.as_ref().map_or(0, vec_bytes)
}

fn wireframe_heap_bytes(value: &WireframePreview) -> usize {
    value.source.capacity()
        + [
            &value.uv_format,
            &value.normal_format,
            &value.tangent_format,
        ]
        .into_iter()
        .map(|s| s.as_ref().map_or(0, String::capacity))
        .sum::<usize>()
        + vec_bytes(&value.vertices)
        + optional_vec_bytes(&value.rigid_indices)
        + optional_vec_bytes(&value.normals)
        + optional_vec_bytes(&value.procedural_positions)
        + optional_vec_bytes(&value.procedural_normals)
        + optional_vec_bytes(&value.tangents)
        + optional_vec_bytes(&value.uvs)
        + vec_bytes(&value.indices)
        + vec_bytes(&value.material_ranges)
        + value
            .material_ranges
            .iter()
            .map(|range| vec_bytes(&range.textures.aux) + vec_bytes(&range.textures.layers))
            .sum::<usize>()
}

fn model_heap_bytes(value: &ModelPreview) -> usize {
    value.class_name.as_ref().map_or(0, String::capacity)
        + value
            .mesh_source
            .as_ref()
            .map_or(0, |source| vec_bytes(&source.shader_constants))
        + vec_bytes(&value.vertex_buffers)
        + vec_bytes(&value.index_buffers)
        + vec_bytes(&value.techniques)
        + vec_bytes(&value.textures)
        + vec_bytes(&value.shaders)
        + vec_bytes(&value.geometry_parts)
        + value.wireframe.as_ref().map_or(0, wireframe_heap_bytes)
}

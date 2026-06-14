use anyhow::{Context, bail};
use binrw::Endian;
use itertools::Itertools;
use quicktag_core::classes::get_class_by_id;
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::TagCache;
use std::cmp::Reverse;
use std::sync::Arc;
use tiger_pkg::{TagHash, TagHash64, Version, package::UEntryHeader, package_manager};
use wgpu::util::DeviceExt;

use crate::material::{
    TechniqueTextureBinding, is_technique_entry, texture_bindings_for_technique,
};
use crate::texture::Texture;

const MAX_PREVIEW_VERTICES: usize = 200_000;
const MAX_PREVIEW_INDICES: usize = 900_000;
const TECHNIQUE_SCAN_CHILD_LIMIT: usize = 128;
const CLASS_GEOMETRY_RESOURCE: u32 = 0x8080881C;
const CLASS_GEOMETRY_BUFFER_SET: u32 = 0x808087CB;
const CLASS_VERTEX_INPUT_LAYOUT_MAPPING: u32 = 0x80808664;
const CLASS_VERTEX_INPUT_ELEMENT_SETS: u32 = 0x80808668;
const CLASS_VERTEX_LAYOUT_ARRAY: u32 = 0x80808667;
const CLASS_VERTEX_INPUT_ELEMENT_ARRAY: u32 = 0x8080866D;
const CLASS_ENTITY_RESOURCE: u32 = 0x80809B06;
const SEMANTIC_TEXCOORD: u8 = 0x05;

#[derive(Debug, Clone)]
pub struct GeometryTagPreview {
    pub kind: GeometryPreviewKind,
}

#[derive(Debug, Clone)]
pub enum GeometryPreviewKind {
    VertexBuffer(VertexBufferPreview),
    IndexBuffer(IndexBufferPreview),
    Model(ModelPreview),
}

#[derive(Debug, Clone)]
pub struct VertexBufferPreview {
    pub header: VertexBufferHeader,
    pub data_tag: TagHash,
    pub data_len: usize,
    pub element_count: u32,
    pub candidates: Vec<VertexPositionCandidate>,
    pub uv_candidates: Vec<VertexUvCandidate>,
    pub wireframe: Option<WireframePreview>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IndexBufferPreview {
    pub header: IndexBufferHeader,
    pub data_tag: TagHash,
    pub data_len: usize,
    pub index_count: usize,
    pub min_index: Option<u32>,
    pub max_index: Option<u32>,
    pub first_indices: Vec<u32>,
    pub indices: Vec<u32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ModelPreview {
    pub label: &'static str,
    pub class_name: Option<String>,
    pub mesh_source: Option<MeshSourcePreview>,
    pub vertex_buffers: Vec<(TagHash, UEntryHeader)>,
    pub index_buffers: Vec<(TagHash, UEntryHeader)>,
    pub techniques: Vec<(TagHash, UEntryHeader)>,
    pub textures: Vec<(TagHash, UEntryHeader)>,
    pub shaders: Vec<(TagHash, UEntryHeader)>,
    pub wireframe: Option<WireframePreview>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTagRole {
    Mesh,
    MeshData,
    Geometry,
    Dynamic,
    Container,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelTagInfo {
    pub role: ModelTagRole,
    pub label: &'static str,
}

#[derive(Debug, Clone)]
pub struct VertexBufferHeader {
    pub data_size: u32,
    pub stride: u16,
    pub vtype: u16,
    pub deadbeef: u32,
}

#[derive(Debug, Clone)]
pub struct IndexBufferHeader {
    pub unk0: i8,
    pub is_32bit: bool,
    pub unk1: u16,
    pub zero: u32,
    pub data_size: u64,
    pub deadbeef: u32,
    pub zero1: u32,
}

#[derive(Debug, Clone)]
pub struct VertexPositionCandidate {
    pub format: PositionFormat,
    pub label: &'static str,
    pub offset: usize,
    pub valid_vertices: usize,
    pub sampled_vertices: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct VertexUvCandidate {
    pub format: UvFormat,
    pub label: String,
    pub offset: usize,
    pub valid_vertices: usize,
    pub sampled_vertices: usize,
    pub min: [f32; 2],
    pub max: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionFormat {
    F32x3,
    I16x4,
    I16x3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvFormat {
    F32x2,
    F16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputLayoutFormat {
    R32G32Float,
    R32G32B32Float,
    R32G32B32A32Float,
    R16G16Snorm,
}

#[derive(Debug, Clone, Copy)]
struct InputLayoutTexcoord {
    buffer_index: usize,
    offset: usize,
    format: InputLayoutFormat,
}

#[derive(Clone, Copy)]
struct TagArray {
    class: u32,
    count: usize,
    data_offset: usize,
    end_offset: usize,
}

#[derive(Clone, Copy)]
struct VertexInputElement {
    semantic: u8,
    semantic_index: u8,
    format: u8,
}

impl std::fmt::Display for InputLayoutFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            InputLayoutFormat::R32G32Float => "R32G32_FLOAT TEXCOORD",
            InputLayoutFormat::R32G32B32Float => "R32G32B32_FLOAT TEXCOORD.xy",
            InputLayoutFormat::R32G32B32A32Float => "R32G32B32A32_FLOAT TEXCOORD.xy",
            InputLayoutFormat::R16G16Snorm => "R16G16_SNORM TEXCOORD",
        };
        f.write_str(label)
    }
}

impl UvFormat {
    fn label(self) -> &'static str {
        match self {
            UvFormat::F32x2 => "f32x2 uv",
            UvFormat::F16x2 => "f16x2 uv",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WireframePreview {
    pub source: String,
    pub position_format: &'static str,
    pub uv_format: Option<String>,
    pub vertices: Vec<[f32; 3]>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub indices: Vec<u32>,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub vertex_count_total: usize,
    pub index_count_total: usize,
}

#[derive(Debug, Clone)]
pub struct MeshSourcePreview {
    pub kind: &'static str,
    pub buffer_index: usize,
    pub technique: Option<TagHash>,
    pub index_start: u32,
    pub index_count: u32,
    pub primitive_type: u8,
    pub lod_category: u8,
    pub input_layout_index: Option<u8>,
    pub index_buffer: TagHash,
    pub vertex0_buffer: TagHash,
    pub vertex1_buffer: TagHash,
    pub color_buffer: TagHash,
    pub uv_transform: Option<UvTransformPreview>,
    pub shader_constants: Vec<ShaderConstantPreview>,
}

#[derive(Debug, Clone, Copy)]
pub struct UvTransformPreview {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct ShaderConstantPreview {
    pub name: &'static str,
    pub value: [f32; 4],
    pub source: &'static str,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPreviewVertex {
    pub position: [f32; 3],
}

pub struct GpuWireframePreview {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub line_count: u32,
}

impl GpuWireframePreview {
    pub fn create(device: &wgpu::Device, wireframe: &WireframePreview) -> Option<Self> {
        if wireframe.vertices.is_empty() {
            return None;
        }

        let vertices = wireframe
            .vertices
            .iter()
            .map(|position| GpuPreviewVertex {
                position: *position,
            })
            .collect_vec();
        let line_indices = triangle_indices_to_line_indices(&wireframe.indices);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quicktag_geometry_preview_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = (!line_indices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quicktag_geometry_preview_indices"),
                contents: bytemuck::cast_slice(&line_indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        });

        Some(Self {
            vertex_buffer,
            index_buffer,
            vertex_count: vertices.len() as u32,
            index_count: line_indices.len() as u32,
            line_count: line_indices.len() as u32 / 2,
        })
    }

    pub fn has_vertex_buffer(&self) -> bool {
        let _ = &self.vertex_buffer;
        true
    }

    pub fn has_index_buffer(&self) -> bool {
        self.index_buffer.is_some()
    }
}

impl GeometryTagPreview {
    pub fn load(
        cache: Arc<TagCache>,
        tag: TagHash,
        entry: &UEntryHeader,
        tag_type: TagType,
        tag_data: &[u8],
    ) -> Option<Self> {
        if matches!(tag_type, TagType::VertexBuffer { is_header: true }) {
            return Some(Self {
                kind: GeometryPreviewKind::VertexBuffer(
                    load_vertex_buffer_preview_for_tag(tag, entry, tag_data).ok()?,
                ),
            });
        }

        if matches!(tag_type, TagType::IndexBuffer { is_header: true }) {
            return Some(Self {
                kind: GeometryPreviewKind::IndexBuffer(
                    load_index_buffer_preview_for_tag(tag, entry, tag_data).ok()?,
                ),
            });
        }

        model_label_for_reference(entry.reference).map(|label| Self {
            kind: GeometryPreviewKind::Model(load_model_preview(cache, tag, entry, label)),
        })
    }

    pub fn wireframe(&self) -> Option<&WireframePreview> {
        match &self.kind {
            GeometryPreviewKind::VertexBuffer(buffer) => buffer.wireframe.as_ref(),
            GeometryPreviewKind::IndexBuffer(_) => None,
            GeometryPreviewKind::Model(model) => model.wireframe.as_ref(),
        }
    }
}

impl VertexBufferPreview {
    pub fn summary(&self) -> String {
        format!(
            "{} bytes, stride {}, {} vertices",
            self.header.data_size, self.header.stride, self.element_count
        )
    }
}

impl IndexBufferPreview {
    pub fn summary(&self) -> String {
        let width = if self.header.is_32bit { 32 } else { 16 };
        format!(
            "{} bytes, u{}, {} indices",
            self.header.data_size, width, self.index_count
        )
    }
}

pub fn model_info_for_reference(reference: u32) -> Option<ModelTagInfo> {
    match reference {
        0x80806D44 | 0x80808635 => Some(ModelTagInfo {
            role: ModelTagRole::Mesh,
            label: "Mesh",
        }),
        0x80806D30 | 0x80808620 => Some(ModelTagInfo {
            role: ModelTagRole::MeshData,
            label: "Mesh data",
        }),
        0x80808567 => Some(ModelTagInfo {
            role: ModelTagRole::Mesh,
            label: "Terrain mesh",
        }),
        0x8080881C => Some(ModelTagInfo {
            role: ModelTagRole::Geometry,
            label: "Geometry",
        }),
        0x80806F07 => Some(ModelTagInfo {
            role: ModelTagRole::Dynamic,
            label: "Dynamic model",
        }),
        0x80806EC5 => Some(ModelTagInfo {
            role: ModelTagRole::Dynamic,
            label: "Dynamic mesh",
        }),
        0x8080BADB | 0x8080BAAD => Some(ModelTagInfo {
            role: ModelTagRole::Container,
            label: "Model container",
        }),
        _ => None,
    }
}

pub fn model_label_for_reference(reference: u32) -> Option<&'static str> {
    model_info_for_reference(reference).map(|info| info.label)
}

fn load_vertex_buffer_preview_for_tag(
    tag: TagHash,
    entry: &UEntryHeader,
    tag_data: &[u8],
) -> anyhow::Result<VertexBufferPreview> {
    let endian = package_manager().version.endian();
    let header = VertexBufferHeader::parse(tag_data, endian)?;
    let data_tag = TagHash(entry.reference);
    let data = package_manager()
        .read_tag(data_tag)
        .with_context(|| format!("Failed to read vertex buffer data tag {data_tag}"))?;

    let element_count = if header.stride == 0 {
        0
    } else {
        header.data_size / header.stride as u32
    };

    let mut warnings = vec![];
    if header.stride == 0 {
        warnings.push("Stride is zero".to_string());
    }
    if header.data_size as usize != data.len() {
        warnings.push(format!(
            "Header data_size {} differs from referenced data length {}",
            header.data_size,
            data.len()
        ));
    }
    if header.deadbeef != 0xDEADBEEF {
        warnings.push(format!(
            "Unexpected header marker 0x{:08X}",
            header.deadbeef
        ));
    }

    let candidates = find_position_candidates(&data, header.stride as usize, endian);
    let uv_candidates = find_uv_candidates(&data, header.stride as usize, endian);
    let wireframe = build_vertex_wireframe(
        tag,
        &data,
        header.stride as usize,
        endian,
        &candidates,
        element_count as usize,
    );

    Ok(VertexBufferPreview {
        candidates,
        uv_candidates,
        wireframe,
        header,
        data_tag,
        data_len: data.len(),
        element_count,
        warnings,
    })
}

fn load_index_buffer_preview_for_tag(
    _tag: TagHash,
    entry: &UEntryHeader,
    tag_data: &[u8],
) -> anyhow::Result<IndexBufferPreview> {
    let endian = package_manager().version.endian();
    let header = IndexBufferHeader::parse(tag_data, endian)?;
    let data_tag = TagHash(entry.reference);
    let data = package_manager()
        .read_tag(data_tag)
        .with_context(|| format!("Failed to read index buffer data tag {data_tag}"))?;

    let index_size = if header.is_32bit { 4 } else { 2 };
    let index_count = header.data_size as usize / index_size;
    let mut indices = Vec::with_capacity(index_count.min(64));
    let mut preview_indices = Vec::with_capacity(index_count.min(MAX_PREVIEW_INDICES));
    let mut min_index = None::<u32>;
    let mut max_index = None::<u32>;

    for chunk in data.chunks_exact(index_size) {
        let value = if header.is_32bit {
            read_u32(chunk, endian)
        } else {
            read_u16(chunk, endian) as u32
        };
        min_index = Some(min_index.map(|v| v.min(value)).unwrap_or(value));
        max_index = Some(max_index.map(|v| v.max(value)).unwrap_or(value));
        if indices.len() < 64 {
            indices.push(value);
        }
        if preview_indices.len() < MAX_PREVIEW_INDICES {
            preview_indices.push(value);
        }
    }

    let mut warnings = vec![];
    if header.data_size as usize != data.len() {
        warnings.push(format!(
            "Header data_size {} differs from referenced data length {}",
            header.data_size,
            data.len()
        ));
    }
    if header.deadbeef != 0xDEADBEEF {
        warnings.push(format!(
            "Unexpected header marker 0x{:08X}",
            header.deadbeef
        ));
    }
    if data.len() % index_size != 0 {
        warnings.push(format!(
            "Referenced data length is not divisible by {}",
            index_size
        ));
    }
    if index_count > MAX_PREVIEW_INDICES {
        warnings.push(format!(
            "Preview indices truncated to {} of {}",
            MAX_PREVIEW_INDICES, index_count
        ));
    }

    Ok(IndexBufferPreview {
        header,
        data_tag,
        data_len: data.len(),
        index_count,
        min_index,
        max_index,
        first_indices: indices,
        indices: preview_indices,
        warnings,
    })
}

fn load_model_preview(
    cache: Arc<TagCache>,
    tag: TagHash,
    entry: &UEntryHeader,
    label: &'static str,
) -> ModelPreview {
    let class_name = get_class_by_id(entry.reference).map(|c| c.name.to_string());
    let vertex_buffers = find_related_tags(&cache, tag, TagSearchKind::VertexBuffer, 8);
    let index_buffers = find_related_tags(&cache, tag, TagSearchKind::IndexBuffer, 8);
    let techniques = find_model_technique_entries(&cache, tag, entry);
    let textures = find_model_textures(&cache, tag, &techniques);
    let shaders = find_related_tags(&cache, tag, TagSearchKind::Shader, 8);
    let (mesh_source, wireframe) = parse_model_wireframe(tag, entry)
        .map(|parsed| (Some(parsed.0), Some(parsed.1)))
        .unwrap_or_else(|| (None, build_model_wireframe(&vertex_buffers, &index_buffers)));

    ModelPreview {
        label,
        class_name,
        mesh_source,
        vertex_buffers,
        index_buffers,
        techniques,
        textures,
        shaders,
        wireframe,
    }
}

fn build_model_wireframe(
    vertex_buffers: &[(TagHash, UEntryHeader)],
    index_buffers: &[(TagHash, UEntryHeader)],
) -> Option<WireframePreview> {
    let (vertex_tag, vertex_entry) = vertex_buffers.first()?;
    let vertex_header_data = package_manager().read_tag(*vertex_tag).ok()?;
    let vertex_preview =
        load_vertex_buffer_preview_for_tag(*vertex_tag, vertex_entry, &vertex_header_data).ok()?;

    let mut wireframe = vertex_preview.wireframe?;
    if let Some((index_tag, index_entry)) = index_buffers.first() {
        let index_header_data = package_manager().read_tag(*index_tag).ok()?;
        if let Ok(index_preview) =
            load_index_buffer_preview_for_tag(*index_tag, index_entry, &index_header_data)
        {
            wireframe.index_count_total = index_preview.index_count;
            wireframe.indices = index_preview.indices;
            wireframe.source = format!("{vertex_tag} + {index_tag}");
        }
    }

    Some(wireframe)
}

fn find_model_textures(
    cache: &TagCache,
    tag: TagHash,
    techniques: &[(TagHash, UEntryHeader)],
) -> Vec<(TagHash, UEntryHeader)> {
    let technique_order = techniques
        .iter()
        .enumerate()
        .map(|(index, (tag, _entry))| (*tag, index))
        .collect::<rustc_hash::FxHashMap<_, _>>();
    let mut textures = techniques
        .iter()
        .map(|(tag, _entry)| *tag)
        .into_iter()
        .enumerate()
        .flat_map(texture_bindings_from_technique)
        .filter_map(|texture_tag| {
            package_manager()
                .get_entry(texture_tag.binding.tag)
                .map(|entry| {
                    (
                        texture_tag.binding.tag,
                        entry,
                        texture_tag.technique_order,
                        texture_binding_rank(texture_tag.binding),
                        (texture_tag.binding.tag.pkg_id() != tag.pkg_id()) as u8,
                        0usize,
                        texture_tag.technique_order,
                        texture_preview_rank(texture_tag.binding.tag),
                    )
                })
        })
        .chain(
            find_related_tags(cache, tag, TagSearchKind::Texture, 8)
                .into_iter()
                .map(|(texture_tag, entry)| {
                    let (parent_count, parent_order) =
                        texture_parent_technique_rank(cache, texture_tag, &technique_order);
                    (
                        texture_tag,
                        entry,
                        usize::MAX,
                        200,
                        (texture_tag.pkg_id() != tag.pkg_id()) as u8,
                        parent_count,
                        parent_order,
                        texture_preview_rank(texture_tag),
                    )
                }),
        )
        .collect_vec();

    textures.sort_by_key(
        |(
            tag,
            entry,
            technique_order,
            rank,
            locality,
            parent_count,
            parent_order,
            preview_rank,
        )| {
            (
                *technique_order,
                *rank,
                *locality,
                *preview_rank,
                Reverse(*parent_count),
                *parent_order,
                Reverse(entry.file_size),
                tag.0,
            )
        },
    );
    textures
        .into_iter()
        .unique_by(
            |(
                tag,
                _entry,
                _technique_order,
                _rank,
                _locality,
                _parent_count,
                _parent_order,
                _preview_rank,
            )| { *tag },
        )
        .map(
            |(
                tag,
                entry,
                _technique_order,
                _rank,
                _locality,
                _parent_count,
                _parent_order,
                _preview_rank,
            )| { (tag, entry) },
        )
        .take(128)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TexturePreviewRank {
    shape: u8,
    format: u8,
    area: Reverse<u64>,
    width: Reverse<u32>,
}

fn texture_preview_rank(tag: TagHash) -> TexturePreviewRank {
    let Ok((desc, _data, _comment)) = Texture::load_data_d2(tag, false) else {
        return TexturePreviewRank {
            shape: 100,
            format: 100,
            area: Reverse(0),
            width: Reverse(0),
        };
    };

    let format = format!("{:?}", desc.format);
    let format_rank = if format.contains("Srgb") {
        0
    } else if format.contains("Bc7") || format.contains("Rgba") {
        20
    } else if format.contains("Bc4") {
        80
    } else {
        50
    };
    let shape_rank = if desc.depth == 1 && desc.array_size == 1 && desc.width > 1 && desc.height > 1
    {
        0
    } else {
        50
    };

    TexturePreviewRank {
        shape: shape_rank,
        format: format_rank,
        area: Reverse(desc.width as u64 * desc.height as u64),
        width: Reverse(desc.width as u32),
    }
}

fn texture_parent_technique_rank(
    cache: &TagCache,
    texture_tag: TagHash,
    technique_order: &rustc_hash::FxHashMap<TagHash, usize>,
) -> (usize, usize) {
    let Some(scan) = cache.hashes.get(&texture_tag) else {
        return (0, usize::MAX);
    };

    let parents = scan
        .references
        .iter()
        .filter_map(|parent| technique_order.get(parent).copied())
        .collect_vec();
    (
        parents.len(),
        parents.into_iter().min().unwrap_or(usize::MAX),
    )
}

fn find_model_technique_entries(
    cache: &TagCache,
    tag: TagHash,
    entry: &UEntryHeader,
) -> Vec<(TagHash, UEntryHeader)> {
    find_model_technique_tags(cache, tag, entry)
        .into_iter()
        .filter_map(|tag| package_manager().get_entry(tag).map(|entry| (tag, entry)))
        .unique_by(|(tag, _entry)| *tag)
        .take(128)
        .collect()
}

struct TechniqueOrderedTextureBinding {
    technique_order: usize,
    binding: TechniqueTextureBinding,
}

fn texture_bindings_from_technique(
    (technique_order, tag): (usize, TagHash),
) -> Vec<TechniqueOrderedTextureBinding> {
    let Some(entry) = package_manager().get_entry(tag) else {
        return vec![];
    };
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };

    texture_bindings_for_technique(&entry, &data)
        .into_iter()
        .map(|binding| TechniqueOrderedTextureBinding {
            technique_order,
            binding,
        })
        .collect()
}

fn texture_binding_rank(binding: TechniqueTextureBinding) -> u8 {
    let slot_rank = match binding.slot {
        0 => 0,
        2 => 20,
        5 => 50,
        10 => 60,
        _ => 100,
    };
    let stage_rank = match binding.stage {
        "PS" => 0,
        "VS" => 5,
        "GS" => 10,
        "CS" => 15,
        _ => 20,
    };
    slot_rank + stage_rank
}

fn find_model_technique_tags(cache: &TagCache, tag: TagHash, entry: &UEntryHeader) -> Vec<TagHash> {
    let Ok(data) = package_manager().read_tag(tag) else {
        let recursive = find_related_tags(cache, tag, TagSearchKind::Technique, 8)
            .into_iter()
            .map(|(tag, _entry)| tag)
            .collect_vec();
        return recursive
            .into_iter()
            .chain(find_child_technique_tags(
                cache,
                tag,
                package_manager().version.endian(),
            ))
            .unique()
            .collect();
    };
    let endian = package_manager().version.endian();

    let direct = match entry.reference {
        0x80806D44 | 0x80808635 => static_mesh_technique_tags(&data, endian),
        0x80806D30 | 0x80808620 => find_static_meshdata_owner_techniques(cache, tag),
        0x80806F07 => dynamic_model_technique_tags(&data, endian),
        0x80806EC5 => dynamic_mesh_technique_tags(&data, endian),
        CLASS_GEOMETRY_RESOURCE => geometry_resource_technique_tags(&data, endian),
        _ => vec![],
    };
    let parent_entity_resource_materials =
        find_parent_entity_resource_material_techniques(cache, tag, endian);
    let raw_current = technique_tags_in_blob(&data, endian);
    let raw_children = find_child_technique_tags(cache, tag, endian);
    let recursive = find_related_tags(cache, tag, TagSearchKind::Technique, 8)
        .into_iter()
        .map(|(tag, _entry)| tag);

    direct
        .into_iter()
        .chain(parent_entity_resource_materials)
        .chain(raw_current)
        .chain(raw_children)
        .chain(recursive)
        .unique()
        .collect()
}

fn find_parent_entity_resource_material_techniques(
    cache: &TagCache,
    tag: TagHash,
    endian: Endian,
) -> Vec<TagHash> {
    cache
        .hashes
        .get(&tag)
        .into_iter()
        .flat_map(|scan| scan.references.iter().copied())
        .filter_map(|parent| {
            let entry = package_manager().get_entry(parent)?;
            (entry.reference == CLASS_ENTITY_RESOURCE).then_some(parent)
        })
        .filter_map(|parent| package_manager().read_tag(parent).ok())
        .flat_map(|data| entity_resource_material_technique_tags(&data, tag, endian))
        .unique()
        .collect()
}

fn entity_resource_material_technique_tags(
    data: &[u8],
    referenced_model: TagHash,
    endian: Endian,
) -> Vec<TagHash> {
    // Alkahest reads SEntityResource.unk18.offset + 0x224 for model,
    // +0x3c0 for material variant map, +0x400 for material techniques.
    let Some(resource_offset) = read_u32_at(data, 0x18, endian).map(|offset| offset as usize)
    else {
        return vec![];
    };
    if resource_offset + 0x410 > data.len() {
        return vec![];
    }

    let model_matches = read_tag_at(data, resource_offset + 0x224, endian)
        .is_some_and(|model| model == referenced_model);
    let parent_references_model = candidate_tag_hashes_in_blob(data, endian)
        .into_iter()
        .any(|tag| tag == referenced_model);
    if !model_matches && !parent_references_model {
        return vec![];
    }

    read_tag_array(data, resource_offset + 0x400, endian)
        .into_iter()
        .filter(|tag| is_technique_tag(*tag))
        .unique()
        .collect()
}

fn find_static_meshdata_owner_techniques(cache: &TagCache, mesh_data_tag: TagHash) -> Vec<TagHash> {
    cache
        .hashes
        .get(&mesh_data_tag)
        .into_iter()
        .flat_map(|scan| scan.references.iter().copied())
        .filter_map(|parent| {
            let entry = package_manager().get_entry(parent)?;
            matches!(entry.reference, 0x80806D44 | 0x80808635).then_some((parent, entry))
        })
        .flat_map(|(parent, _entry)| {
            let data = package_manager().read_tag(parent).ok()?;
            Some(static_mesh_technique_tags(
                &data,
                package_manager().version.endian(),
            ))
        })
        .flatten()
        .unique()
        .collect()
}

fn find_child_technique_tags(cache: &TagCache, tag: TagHash, endian: Endian) -> Vec<TagHash> {
    let Some(scan) = cache.hashes.get(&tag) else {
        return vec![];
    };

    scan.file_hashes
        .iter()
        .map(|scanned| scanned.hash)
        .chain(
            scan.file_hashes64
                .iter()
                .filter_map(|scanned| tag64_to_hash32(scanned.hash)),
        )
        .filter(|child| *child != tag)
        .unique()
        .take(TECHNIQUE_SCAN_CHILD_LIMIT)
        .flat_map(|child| {
            let Some(entry) = package_manager().get_entry(child) else {
                return vec![];
            };
            if is_technique_tag(child) {
                return vec![child];
            }

            let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
            if !tag_type.is_tag() {
                return vec![];
            }

            package_manager()
                .read_tag(child)
                .map(|data| technique_tags_in_blob(&data, endian))
                .unwrap_or_default()
        })
        .unique()
        .collect()
}

fn triangle_indices_to_line_indices(indices: &[u32]) -> Vec<u32> {
    let mut lines = Vec::with_capacity(indices.len().saturating_mul(2));
    for tri in indices.chunks_exact(3) {
        lines.extend_from_slice(&[tri[0], tri[1], tri[1], tri[2], tri[2], tri[0]]);
    }

    lines
}

fn parse_model_wireframe(
    tag: TagHash,
    entry: &UEntryHeader,
) -> Option<(MeshSourcePreview, WireframePreview)> {
    match entry.reference {
        0x80806D44 => parse_static_mesh_wireframe(tag),
        0x80806D30 => {
            let data = package_manager().read_tag(tag).ok()?;
            parse_static_mesh_data_wireframe(&data)
        }
        0x80806F07 => parse_dynamic_model_wireframe(tag),
        0x80806EC5 => {
            let data = package_manager().read_tag(tag).ok()?;
            parse_dynamic_mesh_wireframe(&data)
        }
        CLASS_GEOMETRY_RESOURCE => parse_geometry_resource_wireframe(tag),
        _ => None,
    }
}

fn parse_static_mesh_wireframe(tag: TagHash) -> Option<(MeshSourcePreview, WireframePreview)> {
    let data = package_manager().read_tag(tag).ok()?;
    let endian = package_manager().version.endian();
    let mesh_data_tag = TagHash(read_u32(data.get(0x8..0xc)?, endian));
    let mesh_data = package_manager().read_tag(mesh_data_tag).ok()?;
    parse_static_mesh_data_wireframe(&mesh_data)
}

fn parse_static_mesh_data_wireframe(data: &[u8]) -> Option<(MeshSourcePreview, WireframePreview)> {
    let endian = package_manager().version.endian();
    let parts = read_static_mesh_parts(data, 0x18, endian);
    let groups = read_static_mesh_groups(data, 0x8, endian);
    let buffers = read_static_buffer_tuples(data, 0x28, endian);
    let group = groups
        .iter()
        .filter_map(|group| {
            parts
                .get(group.part_index as usize)
                .map(|part| (group, part))
        })
        .min_by_key(|(_group, part)| (lod_selection_rank(part.lod_category), part.index_start))
        .map(|(group, _part)| group)
        .or_else(|| groups.first())?;
    let part = parts.get(group.part_index as usize)?;
    let buffers = buffers.get(part.buffer_index as usize)?;

    let uv_transform = read_static_uv_transform(data, endian);
    let source = MeshSourcePreview {
        kind: "static mesh data",
        buffer_index: part.buffer_index as usize,
        technique: None,
        index_start: part.index_start,
        index_count: part.index_count,
        primitive_type: part.primitive_type,
        lod_category: lod_preview_value(part.lod_category),
        input_layout_index: Some(group.input_layout_index),
        index_buffer: buffers.index_buffer,
        vertex0_buffer: buffers.vertex0_buffer,
        vertex1_buffer: buffers.vertex1_buffer,
        color_buffer: buffers.color_buffer,
        uv_transform,
        shader_constants: shader_constants_from_uv_transform(uv_transform),
    };
    let wireframe = build_wireframe_from_refs(
        &[source.vertex0_buffer, source.vertex1_buffer],
        source.index_buffer,
        &[PreviewIndexRange {
            range: source.index_start as usize..(source.index_start + source.index_count) as usize,
            primitive_type: source.primitive_type,
        }],
        source.input_layout_index,
    )?;

    Some((source, wireframe))
}

fn parse_dynamic_model_wireframe(tag: TagHash) -> Option<(MeshSourcePreview, WireframePreview)> {
    let data = package_manager().read_tag(tag).ok()?;
    let endian = package_manager().version.endian();
    let mesh_array = read_array(data.get(..)?, 0x10, 0x80, endian)?;
    let (mut source, wireframe) =
        parse_dynamic_mesh_wireframe(mesh_array.chunks_exact(0x80).next()?)?;
    source.uv_transform = read_dynamic_model_uv_transform(&data, endian);
    source.shader_constants = shader_constants_from_uv_transform(source.uv_transform);
    Some((source, wireframe))
}

fn parse_dynamic_mesh_wireframe(data: &[u8]) -> Option<(MeshSourcePreview, WireframePreview)> {
    let endian = package_manager().version.endian();
    let parts = read_dynamic_mesh_parts(data, 0x20, endian);
    let part = parts
        .iter()
        .min_by_key(|part| (lod_selection_rank(part.lod_category), part.index_start))
        .or_else(|| parts.first())?;
    let source = MeshSourcePreview {
        kind: "dynamic mesh",
        buffer_index: 0,
        technique: Some(part.technique),
        index_start: part.index_start,
        index_count: part.index_count,
        primitive_type: part.primitive_type,
        lod_category: lod_preview_value(part.lod_category),
        input_layout_index: first_dynamic_input_layout(data),
        index_buffer: TagHash(read_u32(data.get(0x10..0x14)?, endian)),
        vertex0_buffer: TagHash(read_u32(data.get(0x0..0x4)?, endian)),
        vertex1_buffer: TagHash(read_u32(data.get(0x4..0x8)?, endian)),
        color_buffer: TagHash(read_u32(data.get(0x14..0x18)?, endian)),
        uv_transform: None,
        shader_constants: vec![],
    };
    let wireframe = build_wireframe_from_refs(
        &[source.vertex0_buffer, source.vertex1_buffer],
        source.index_buffer,
        &[PreviewIndexRange {
            range: source.index_start as usize..(source.index_start + source.index_count) as usize,
            primitive_type: source.primitive_type,
        }],
        source.input_layout_index,
    )?;

    Some((source, wireframe))
}

fn parse_geometry_resource_wireframe(
    tag: TagHash,
) -> Option<(MeshSourcePreview, WireframePreview)> {
    let data = package_manager().read_tag(tag).ok()?;
    let endian = package_manager().version.endian();
    let uv_transform = read_geometry_uv_transform(&data, endian);
    let ranges = geometry_primary_index_ranges(&data, endian);
    let range = ranges.first();
    let index_count = ranges
        .iter()
        .map(|range| range.index_count)
        .fold(0u32, u32::saturating_add);

    let mesh_records = scan_arrays(&data, endian)
        .into_iter()
        .filter(|array| array.class == CLASS_GEOMETRY_BUFFER_SET)
        .flat_map(|array| array_records(&data, array, 0x80))
        .collect_vec();

    if let Some(mesh) = mesh_records.first() {
        let vertex0_buffer = read_tag_at(mesh, 0x0, endian)?;
        let vertex1_buffer = read_tag_at(mesh, 0x4, endian).unwrap_or(TagHash(0));
        let index_buffer = read_tag_at(mesh, 0x10, endian)?;
        let input_layout_index = geometry_buffer_set_input_layout_id(mesh);
        let index_ranges = index_ranges_from_geometry_ranges(&ranges);
        let source = MeshSourcePreview {
            kind: "geometry resource",
            buffer_index: 0,
            technique: range.map(|range| range.technique),
            index_start: range.map(|range| range.index_start).unwrap_or(0),
            index_count: if index_count == 0 {
                u32::MAX
            } else {
                index_count
            },
            primitive_type: range.map(|range| range.primitive_type).unwrap_or(0),
            lod_category: range
                .map(|range| lod_preview_value(range.lod_category))
                .unwrap_or(0),
            input_layout_index,
            index_buffer,
            vertex0_buffer,
            vertex1_buffer,
            color_buffer: TagHash(0),
            uv_transform,
            shader_constants: shader_constants_from_uv_transform(uv_transform),
        };
        let wireframe = build_wireframe_from_refs(
            &[source.vertex0_buffer, source.vertex1_buffer],
            source.index_buffer,
            index_ranges.as_slice(),
            source.input_layout_index,
        )?;

        return Some((source, wireframe));
    }

    let index_buffer = read_tag_at(&data, 0x140, endian)?;
    let vertex0_buffer = read_tag_at(&data, 0x130, endian)?;
    let vertex1_buffer = read_tag_at(&data, 0x134, endian).unwrap_or(TagHash(0));
    let input_layout_index = geometry_primary_layout_id(&data, endian);
    let index_ranges = index_ranges_from_geometry_ranges(&ranges);
    let source = MeshSourcePreview {
        kind: "geometry resource",
        buffer_index: 0,
        technique: range.map(|range| range.technique),
        index_start: range.map(|range| range.index_start).unwrap_or(0),
        index_count: if index_count == 0 {
            u32::MAX
        } else {
            index_count
        },
        primitive_type: range.map(|range| range.primitive_type).unwrap_or(0),
        lod_category: range
            .map(|range| lod_preview_value(range.lod_category))
            .unwrap_or(0),
        input_layout_index,
        index_buffer,
        vertex0_buffer,
        vertex1_buffer,
        color_buffer: TagHash(0),
        uv_transform,
        shader_constants: shader_constants_from_uv_transform(uv_transform),
    };
    let wireframe = build_wireframe_from_refs(
        &[source.vertex0_buffer, source.vertex1_buffer],
        source.index_buffer,
        index_ranges.as_slice(),
        source.input_layout_index,
    )?;

    Some((source, wireframe))
}

fn build_wireframe_from_refs(
    vertex_tags: &[TagHash],
    index_tag: TagHash,
    index_ranges: &[PreviewIndexRange],
    input_layout_index: Option<u8>,
) -> Option<WireframePreview> {
    let mut vertex_previews = vertex_tags
        .iter()
        .copied()
        .filter_map(|vertex_tag| {
            let vertex_entry = package_manager().get_entry(vertex_tag)?;
            let vertex_header = package_manager().read_tag(vertex_tag).ok()?;
            let vertex_preview =
                load_vertex_buffer_preview_for_tag(vertex_tag, &vertex_entry, &vertex_header)
                    .ok()?;
            Some((vertex_tag, vertex_preview))
        })
        .collect_vec();

    let (vertex_tag, mut wireframe) = vertex_previews
        .iter_mut()
        .find_map(|(vertex_tag, preview)| Some((*vertex_tag, preview.wireframe.take()?)))?;

    if let Some((uv_tag, uv_format, uvs)) = input_layout_uvs(
        &vertex_previews,
        input_layout_index,
        wireframe.vertices.len(),
    ) {
        wireframe.uv_format = Some(format!(
            "{uv_format} from {uv_tag} layout {input_layout_index:?}"
        ));
        wireframe.uvs = Some(uvs);
    } else if let Some((uv_tag, uv_candidate, uvs)) =
        best_model_uvs(&vertex_previews, wireframe.vertices.len())
    {
        wireframe.uv_format = Some(format!("{} from {}", uv_candidate.label, uv_tag));
        wireframe.uvs = Some(uvs);
    }

    let index_entry = package_manager().get_entry(index_tag)?;
    let index_header = package_manager().read_tag(index_tag).ok()?;
    let index_preview =
        load_index_buffer_preview_for_tag(index_tag, &index_entry, &index_header).ok()?;
    wireframe.index_count_total = index_preview.index_count;
    wireframe.indices = if index_ranges.is_empty() {
        index_preview
            .indices
            .into_iter()
            .take(MAX_PREVIEW_INDICES)
            .collect()
    } else {
        index_ranges
            .iter()
            .flat_map(|range| {
                let source = index_preview
                    .indices
                    .get(
                        range.range.start.min(index_preview.indices.len())
                            ..range.range.end.min(index_preview.indices.len()),
                    )
                    .unwrap_or_default();
                preview_triangles_from_indices(source, range.primitive_type)
            })
            .take(MAX_PREVIEW_INDICES)
            .collect()
    };
    wireframe.source = format!("{vertex_tag} + {index_tag}");

    Some(wireframe)
}

fn best_model_uvs(
    vertex_previews: &[(TagHash, VertexBufferPreview)],
    vertex_count: usize,
) -> Option<(TagHash, VertexUvCandidate, Vec<[f32; 2]>)> {
    vertex_previews
        .iter()
        .flat_map(|(tag, preview)| {
            preview
                .uv_candidates
                .iter()
                .map(move |candidate| (*tag, preview, candidate))
        })
        .max_by_key(|(_tag, _preview, candidate)| uv_candidate_score(candidate))
        .and_then(|(tag, preview, candidate)| {
            let uvs = load_uvs_for_vertex_buffer(tag, preview, candidate, vertex_count)?;
            Some((tag, candidate.clone(), uvs))
        })
}

fn input_layout_uvs(
    vertex_previews: &[(TagHash, VertexBufferPreview)],
    input_layout_index: Option<u8>,
    vertex_count: usize,
) -> Option<(TagHash, InputLayoutFormat, Vec<[f32; 2]>)> {
    let layout_index = input_layout_index?;
    let layout = resolved_input_layout_texcoord0(layout_index)
        .or_else(|| alkahest_input_layout_texcoord0(layout_index))?;
    let (tag, preview) = vertex_previews.get(layout.buffer_index)?;
    let entry = package_manager().get_entry(*tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let endian = package_manager().version.endian();
    let uvs = decode_input_layout_uvs(
        &data,
        preview.header.stride as usize,
        endian,
        layout,
        vertex_count,
    );

    (!uvs.is_empty()).then_some((*tag, layout.format, uvs))
}

fn resolved_input_layout_texcoord0(layout_id: u8) -> Option<InputLayoutTexcoord> {
    let stream_sets = vertex_layout_stream_sets_for_any_mapping(layout_id)?;

    for element_tag in tags_by_class(CLASS_VERTEX_INPUT_ELEMENT_SETS) {
        let sets = vertex_input_element_sets(element_tag);
        for (buffer_index, set_index) in stream_sets.iter().copied().enumerate() {
            let Some(elements) = sets.get(set_index) else {
                continue;
            };
            let mut offset = 0usize;
            for element in elements {
                let size = vertex_input_element_size(*element)?;
                if element.semantic == SEMANTIC_TEXCOORD && element.semantic_index == 0 {
                    return Some(InputLayoutTexcoord {
                        buffer_index,
                        offset,
                        format: input_layout_format_from_vertex_format(element.format)?,
                    });
                }
                offset = offset.checked_add(size)?;
            }
        }
    }

    None
}

fn alkahest_input_layout_texcoord0(layout_id: u8) -> Option<InputLayoutTexcoord> {
    match layout_id {
        2 | 5 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R32G32Float,
        }),
        3 | 12 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 12,
            format: InputLayoutFormat::R32G32Float,
        }),
        6 | 75 | 76 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 40,
            format: InputLayoutFormat::R32G32Float,
        }),
        7 | 8 | 13 => Some(InputLayoutTexcoord {
            buffer_index: 1,
            offset: 0,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        9 | 18 | 26 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        11 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 8,
            format: InputLayoutFormat::R32G32B32Float,
        }),
        14 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 40,
            format: InputLayoutFormat::R32G32B32A32Float,
        }),
        15 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 16,
            format: InputLayoutFormat::R32G32Float,
        }),
        17 | 21 => Some(InputLayoutTexcoord {
            buffer_index: 1,
            offset: 0,
            format: InputLayoutFormat::R32G32B32Float,
        }),
        19 | 20 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 16,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        24 | 25 => Some(InputLayoutTexcoord {
            buffer_index: 0,
            offset: 12,
            format: InputLayoutFormat::R16G16Snorm,
        }),
        27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44
        | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61
        | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 => {
            Some(InputLayoutTexcoord {
                buffer_index: 0,
                offset: 0,
                format: InputLayoutFormat::R32G32B32A32Float,
            })
        }
        _ => None,
    }
}

fn decode_input_layout_uvs(
    data: &[u8],
    stride: usize,
    endian: Endian,
    layout: InputLayoutTexcoord,
    vertex_count: usize,
) -> Vec<[f32; 2]> {
    if stride == 0 || layout.offset >= stride {
        return vec![];
    }

    data.chunks_exact(stride)
        .take(vertex_count.min(MAX_PREVIEW_VERTICES))
        .filter_map(|vertex| {
            decode_input_layout_uv(vertex.get(layout.offset..)?, endian, layout.format)
        })
        .collect()
}

fn decode_input_layout_uv(
    bytes: &[u8],
    endian: Endian,
    format: InputLayoutFormat,
) -> Option<[f32; 2]> {
    let uv = match format {
        InputLayoutFormat::R32G32Float
        | InputLayoutFormat::R32G32B32Float
        | InputLayoutFormat::R32G32B32A32Float => [
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
        ],
        InputLayoutFormat::R16G16Snorm => [
            read_snorm16(bytes.get(0..2)?, endian),
            read_snorm16(bytes.get(2..4)?, endian),
        ],
    };

    uv.iter()
        .all(|v| v.is_finite() && v.abs() <= 1024.0)
        .then_some(uv)
}

fn load_uvs_for_vertex_buffer(
    tag: TagHash,
    preview: &VertexBufferPreview,
    candidate: &VertexUvCandidate,
    vertex_count: usize,
) -> Option<Vec<[f32; 2]>> {
    let entry = package_manager().get_entry(tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let endian = package_manager().version.endian();
    Some(decode_uvs(
        &data,
        preview.header.stride as usize,
        endian,
        candidate.format,
        candidate.offset,
        vertex_count,
    ))
    .filter(|uvs| !uvs.is_empty())
}

fn uv_candidate_score(candidate: &VertexUvCandidate) -> usize {
    let span_u = (candidate.max[0] - candidate.min[0]).abs();
    let span_v = (candidate.max[1] - candidate.min[1]).abs();
    let format_bonus = match candidate.format {
        UvFormat::F16x2 => 10_000,
        UvFormat::F32x2 => 0,
    };
    let compact_bonus = if candidate
        .min
        .into_iter()
        .chain(candidate.max)
        .all(|v| v.abs() <= 16.0)
    {
        20_000
    } else {
        0
    };
    let span_penalty = ((span_u + span_v) * 10.0).max(0.0) as usize;
    (candidate.valid_vertices + format_bonus + compact_bonus)
        .saturating_sub(span_penalty.min(10_000))
}

#[derive(Debug, Clone)]
struct StaticMeshPartPreview {
    index_start: u32,
    index_count: u32,
    buffer_index: u8,
    lod_category: u8,
    primitive_type: u8,
}

#[derive(Debug, Clone)]
struct StaticMeshGroupPreview {
    part_index: u16,
    input_layout_index: u8,
}

#[derive(Debug, Clone)]
struct DynamicMeshPartPreview {
    technique: TagHash,
    index_start: u32,
    index_count: u32,
    primitive_type: u8,
    lod_category: u8,
}

#[derive(Debug, Clone)]
struct GeometryIndexRangePreview {
    technique: TagHash,
    index_start: u32,
    index_count: u32,
    primitive_type: u8,
    lod_category: u8,
}

#[derive(Debug, Clone)]
struct PreviewIndexRange {
    range: std::ops::Range<usize>,
    primitive_type: u8,
}

#[derive(Debug, Clone)]
struct StaticBufferTuple {
    index_buffer: TagHash,
    vertex0_buffer: TagHash,
    vertex1_buffer: TagHash,
    color_buffer: TagHash,
}

fn read_static_mesh_parts(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<StaticMeshPartPreview> {
    read_array(data, vec_offset, 12, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(12))
        .filter_map(|part| {
            Some(StaticMeshPartPreview {
                index_start: read_u32(part.get(0x0..0x4)?, endian),
                index_count: read_u32(part.get(0x4..0x8)?, endian),
                buffer_index: *part.get(0x8)?,
                lod_category: *part.get(0xa)?,
                primitive_type: *part.get(0xb)?,
            })
        })
        .collect()
}

fn read_static_mesh_groups(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<StaticMeshGroupPreview> {
    read_array(data, vec_offset, 6, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(6))
        .filter_map(|group| {
            Some(StaticMeshGroupPreview {
                part_index: read_u16(group.get(0x0..0x2)?, endian),
                input_layout_index: *group.get(0x3)?,
            })
        })
        .collect()
}

fn read_static_buffer_tuples(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<StaticBufferTuple> {
    read_array(data, vec_offset, 16, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(16))
        .filter_map(|tuple| {
            Some(StaticBufferTuple {
                index_buffer: TagHash(read_u32(tuple.get(0x0..0x4)?, endian)),
                vertex0_buffer: TagHash(read_u32(tuple.get(0x4..0x8)?, endian)),
                vertex1_buffer: TagHash(read_u32(tuple.get(0x8..0xc)?, endian)),
                color_buffer: TagHash(read_u32(tuple.get(0xc..0x10)?, endian)),
            })
        })
        .collect()
}

fn static_mesh_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    let opaque = read_tag_array(data, 0x10, endian);
    let special = read_array(data, 0x20, 0x24, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x24))
        .filter_map(|mesh| Some(TagHash(read_u32(mesh.get(0x20..0x24)?, endian))));

    opaque
        .into_iter()
        .chain(special)
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn dynamic_model_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    read_array(data, 0x10, 0x80, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x80))
        .flat_map(|mesh| dynamic_mesh_technique_tags(mesh, endian))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn dynamic_mesh_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    read_array(data, 0x20, 0x24, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x24))
        .filter_map(|part| Some(TagHash(read_u32(part.get(0x0..0x4)?, endian))))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn geometry_resource_technique_tags(data: &[u8], endian: Endian) -> Vec<TagHash> {
    let ranges = geometry_index_range_candidates(data, endian);
    if ranges.is_empty() {
        return vec![];
    }

    let primary_lod = ranges.iter().map(|(_, lod, _)| *lod).min().unwrap_or(0);
    let primary_pass = ranges
        .iter()
        .filter(|(_, lod, pass)| *lod == primary_lod && *pass > 0)
        .map(|(_, _, pass)| *pass)
        .min()
        .unwrap_or(0);

    ranges
        .into_iter()
        .sorted_by_key(|(range, lod, pass)| {
            (
                *lod != primary_lod,
                primary_pass != 0 && *pass != primary_pass,
                range.index_start,
            )
        })
        .map(|(range, _, _)| range.technique)
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn technique_tags_in_blob(data: &[u8], endian: Endian) -> Vec<TagHash> {
    let alternate = match endian {
        Endian::Little => Endian::Big,
        Endian::Big => Endian::Little,
    };

    candidate_tag_hashes_in_blob(data, endian)
        .into_iter()
        .chain(candidate_tag_hashes_in_blob(data, alternate))
        .chain(candidate_tag64_hashes_in_blob(data, endian))
        .chain(candidate_tag64_hashes_in_blob(data, alternate))
        .filter(|tag| is_technique_tag(*tag))
        .unique()
        .collect()
}

fn candidate_tag_hashes_in_blob(data: &[u8], endian: Endian) -> Vec<TagHash> {
    data.chunks_exact(4)
        .map(|bytes| TagHash(read_u32(bytes, endian)))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn candidate_tag64_hashes_in_blob(data: &[u8], endian: Endian) -> Vec<TagHash> {
    data.chunks_exact(8)
        .filter_map(|bytes| tag64_to_hash32(TagHash64(read_u64(bytes, endian))))
        .filter(|tag| tag.is_some())
        .unique()
        .collect()
}

fn tag64_to_hash32(tag: TagHash64) -> Option<TagHash> {
    package_manager()
        .lookup
        .tag64_entries
        .get(&tag.0)
        .map(|entry| entry.hash32)
}

fn is_technique_tag(tag: TagHash) -> bool {
    package_manager()
        .get_entry(tag)
        .is_some_and(|entry| is_technique_entry(&entry))
}

fn read_tag_array(data: &[u8], vec_offset: usize, endian: Endian) -> Vec<TagHash> {
    read_array(data, vec_offset, 4, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(4))
        .filter_map(|tag| Some(TagHash(read_u32(tag, endian))))
        .collect()
}

fn read_dynamic_mesh_parts(
    data: &[u8],
    vec_offset: usize,
    endian: Endian,
) -> Vec<DynamicMeshPartPreview> {
    read_array(data, vec_offset, 0x24, endian)
        .into_iter()
        .flat_map(|array| array.chunks_exact(0x24))
        .filter_map(|part| {
            Some(DynamicMeshPartPreview {
                technique: TagHash(read_u32(part.get(0x0..0x4)?, endian)),
                primitive_type: *part.get(0x6)?,
                index_start: read_u32(part.get(0x8..0xc)?, endian),
                index_count: read_u32(part.get(0xc..0x10)?, endian),
                lod_category: *part.get(0x1d)?,
            })
        })
        .collect()
}

fn geometry_buffer_set_input_layout_id(mesh: &[u8]) -> Option<u8> {
    mesh.get(0x64)
        .copied()
        .filter(|layout| *layout != 0)
        .or_else(|| mesh.get(0x62).copied().filter(|layout| *layout != 0))
}

fn geometry_primary_index_ranges(data: &[u8], endian: Endian) -> Vec<GeometryIndexRangePreview> {
    let candidates = geometry_index_range_candidates(data, endian);

    let Some(primary_lod) = candidates.iter().map(|(_, lod, _)| *lod).min() else {
        return vec![];
    };
    let primary_pass = candidates
        .iter()
        .filter(|(_, lod, pass)| *lod == primary_lod && *pass > 0)
        .map(|(_, _, pass)| *pass)
        .min()
        .unwrap_or(0);

    candidates
        .into_iter()
        .filter(|(_, lod, pass)| {
            *lod == primary_lod && (primary_pass == 0 || *pass == primary_pass)
        })
        .map(|(range, _, _)| range)
        .sorted_by_key(|range| range.index_start)
        .collect()
}

fn index_ranges_from_geometry_ranges(
    ranges: &[GeometryIndexRangePreview],
) -> Vec<PreviewIndexRange> {
    ranges
        .iter()
        .map(|range| PreviewIndexRange {
            range: range.index_start as usize
                ..range.index_start.saturating_add(range.index_count) as usize,
            primitive_type: range.primitive_type,
        })
        .collect()
}

fn preview_triangles_from_indices(indices: &[u32], primitive_type: u8) -> Vec<u32> {
    if primitive_type != 5 {
        return indices.to_vec();
    }

    let mut out = Vec::with_capacity(indices.len().saturating_sub(2).saturating_mul(3));
    for (i, window) in indices.windows(3).enumerate() {
        let a = window[0];
        let b = window[1];
        let c = window[2];
        if a == b || b == c || a == c {
            continue;
        }
        if i % 2 == 0 {
            out.extend_from_slice(&[a, b, c]);
        } else {
            out.extend_from_slice(&[b, a, c]);
        }
    }
    out
}

fn geometry_index_range_candidates(
    data: &[u8],
    endian: Endian,
) -> Vec<(GeometryIndexRangePreview, u8, u8)> {
    let mut candidates = Vec::<(GeometryIndexRangePreview, u8, u8)>::new();

    for offset in (0..data.len().saturating_sub(0x28)).step_by(4) {
        let Some(material) = read_tag_at(data, offset, endian) else {
            continue;
        };
        if !package_manager()
            .get_entry(material)
            .is_some_and(|entry| is_technique_entry(&entry))
        {
            continue;
        }

        let Some(index_start) = read_u32_at(data, offset + 0x8, endian) else {
            continue;
        };
        let Some(index_count) = read_u32_at(data, offset + 0xc, endian) else {
            continue;
        };
        let Some(triangle_count) = read_u32_at(data, offset + 0x10, endian) else {
            continue;
        };
        let Some(flags) = read_u32_at(data, offset + 0x20, endian) else {
            continue;
        };
        let lod = ((flags >> 8) & 0xff) as u8;
        let pass = ((flags >> 16) & 0xff) as u8;
        if index_count < 3 || triangle_count == 0 {
            continue;
        }

        candidates.push((
            GeometryIndexRangePreview {
                technique: material,
                index_start,
                index_count,
                primitive_type: if index_count == triangle_count.saturating_mul(3) {
                    3
                } else {
                    5
                },
                lod_category: lod,
            },
            lod,
            pass,
        ));
    }

    candidates
}

fn geometry_primary_layout_id(data: &[u8], endian: Endian) -> Option<u8> {
    let mut best = None::<(u8, usize)>;

    for array in scan_arrays(data, endian) {
        if array.class != CLASS_GEOMETRY_BUFFER_SET {
            continue;
        }

        let record =
            data.get(array.data_offset.min(data.len())..array.end_offset.min(data.len()))?;
        let mut counts = [0usize; 0x40];
        for &byte in record.iter().skip(record.len() / 2) {
            if byte > 0 && (byte as usize) < counts.len() {
                counts[byte as usize] += 1;
            }
        }

        if let Some((layout_id, count)) = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .filter(|(_, count)| **count >= 4)
        {
            let replace = best
                .map(|(_, best_count)| *count > best_count)
                .unwrap_or(true);
            if replace {
                best = Some((layout_id as u8, *count));
            }
        }
    }

    best.map(|(layout_id, _)| layout_id)
}

fn read_geometry_uv_transform(data: &[u8], endian: Endian) -> Option<UvTransformPreview> {
    let scale = [
        read_f32(data.get(0xc0..0xc4)?, endian),
        read_f32(data.get(0xc4..0xc8)?, endian),
    ];
    let offset = [
        read_f32(data.get(0xc8..0xcc)?, endian),
        read_f32(data.get(0xcc..0xd0)?, endian),
    ];
    (scale
        .into_iter()
        .chain(offset)
        .all(|value| value.is_finite())
        && scale.iter().any(|value| value.abs() > 0.000001))
    .then_some(UvTransformPreview { scale, offset })
}

fn scan_arrays(data: &[u8], endian: Endian) -> Vec<TagArray> {
    let marker_offsets = (0..data.len().saturating_sub(4))
        .step_by(4)
        .filter(|offset| read_u32_at(data, *offset, endian) == Some(0x8080BFCD))
        .collect_vec();

    marker_offsets
        .iter()
        .enumerate()
        .filter_map(|(i, marker_offset)| {
            Some(TagArray {
                class: read_u32_at(data, marker_offset + 12, endian)?,
                count: read_u64_at(data, marker_offset + 4, endian)? as usize,
                data_offset: marker_offset + 20,
                end_offset: marker_offsets
                    .get(i + 1)
                    .copied()
                    .unwrap_or(data.len())
                    .min(data.len()),
            })
        })
        .collect()
}

fn array_records<'a>(data: &'a [u8], array: TagArray, stride: usize) -> Vec<&'a [u8]> {
    let start = array.data_offset.min(data.len());
    let end = array.end_offset.min(data.len());
    data[start..end]
        .chunks_exact(stride)
        .take(array.count)
        .collect()
}

fn vertex_layout_stream_sets_for_any_mapping(layout_id: u8) -> Option<Vec<usize>> {
    for layout_tag in tags_by_class(CLASS_VERTEX_INPUT_LAYOUT_MAPPING) {
        if let Some(stream_sets) = vertex_layout_stream_sets(layout_tag, layout_id) {
            return Some(stream_sets);
        }
    }

    None
}

fn vertex_layout_stream_sets(layout_tag: TagHash, layout_id: u8) -> Option<Vec<usize>> {
    let endian = package_manager().version.endian();
    let data = package_manager().read_tag(layout_tag).ok()?;
    for array in scan_arrays(&data, endian) {
        if array.class != CLASS_VERTEX_LAYOUT_ARRAY {
            continue;
        }

        for record in array_records(&data, array, 0x1c) {
            let record_layout_id = (read_u32_at(record, 0, endian)? & 0xffff) as u8;
            if record_layout_id != layout_id {
                continue;
            }

            let stream_sets = [0x8, 0xc, 0x10, 0x14]
                .into_iter()
                .filter_map(|offset| read_u32_at(record, offset, endian))
                .filter(|set_index| *set_index != u32::MAX)
                .map(|set_index| set_index as usize)
                .collect_vec();
            return (!stream_sets.is_empty()).then_some(stream_sets);
        }
    }

    None
}

fn vertex_input_element_sets(tag: TagHash) -> Vec<Vec<VertexInputElement>> {
    let endian = package_manager().version.endian();
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };

    scan_arrays(&data, endian)
        .into_iter()
        .filter(|array| array.class == CLASS_VERTEX_INPUT_ELEMENT_ARRAY)
        .map(|array| {
            data[array.data_offset.min(data.len())..array.end_offset.min(data.len())]
                .chunks_exact(3)
                .take(array.count)
                .map(|record| VertexInputElement {
                    semantic: record[0],
                    semantic_index: record[1],
                    format: record[2],
                })
                .collect()
        })
        .collect()
}

fn vertex_input_element_size(element: VertexInputElement) -> Option<usize> {
    match element.format {
        0x00 => Some(0),
        0x01 => Some(4),
        0x02 => Some(8),
        0x03 => Some(12),
        0x04 => Some(16),
        0x05 | 0x06 | 0x07 => Some(4),
        0x08 | 0x09 => Some(8),
        0x0A => Some(4),
        0x0B => Some(8),
        0x0C => Some(4),
        0x0D => Some(8),
        0x0E | 0x0F | 0x10 | 0x11 | 0x12 => Some(4),
        0x13 => Some(8),
        0x14 => Some(16),
        0x15 => Some(4),
        0x16 => Some(8),
        0x17 => Some(16),
        0x18 => Some(2),
        0x19 => Some(1),
        0x1F | 0x20 => Some(4),
        0x21 => Some(8),
        _ => None,
    }
}

fn input_layout_format_from_vertex_format(format: u8) -> Option<InputLayoutFormat> {
    match format {
        0x02 => Some(InputLayoutFormat::R32G32Float),
        0x03 => Some(InputLayoutFormat::R32G32B32Float),
        0x04 => Some(InputLayoutFormat::R32G32B32A32Float),
        0x0A => Some(InputLayoutFormat::R16G16Snorm),
        _ => None,
    }
}

fn tags_by_class(class: u32) -> Vec<TagHash> {
    let pm = package_manager();
    let mut tags = Vec::new();
    for (pkg_id, entries) in &pm.lookup.tag32_entries_by_pkg {
        for (index, entry) in entries.iter().enumerate() {
            if entry.reference == class {
                tags.push(TagHash::new(*pkg_id, index as u16));
            }
        }
    }
    tags.sort_by_key(|tag| (tag.pkg_id(), tag.entry_index()));
    tags
}

fn read_tag_at(data: &[u8], offset: usize, endian: Endian) -> Option<TagHash> {
    Some(TagHash(read_u32_at(data, offset, endian)?))
        .filter(|tag| package_manager().get_entry(*tag).is_some())
}

fn read_u32_at(data: &[u8], offset: usize, endian: Endian) -> Option<u32> {
    data.get(offset..offset + 4)
        .map(|bytes| read_u32(bytes, endian))
}

fn read_u64_at(data: &[u8], offset: usize, endian: Endian) -> Option<u64> {
    data.get(offset..offset + 8)
        .map(|bytes| read_u64(bytes, endian))
}

fn read_array(data: &[u8], vec_offset: usize, elem_size: usize, endian: Endian) -> Option<&[u8]> {
    let count = read_u64(data.get(vec_offset..vec_offset + 8)?, endian) as usize;
    if count == 0 || elem_size == 0 {
        return Some(&[]);
    }

    let rel = read_i64(data.get(vec_offset + 8..vec_offset + 16)?, endian);
    let header_offset = (vec_offset as i64 + 8).checked_add(rel)? as usize;
    let header_count = read_u64(data.get(header_offset..header_offset + 8)?, endian) as usize;
    if header_count != count {
        return None;
    }

    let data_start = header_offset + 16;
    let data_end = data_start.checked_add(count.checked_mul(elem_size)?)?;
    data.get(data_start..data_end)
}

fn first_dynamic_input_layout(data: &[u8]) -> Option<u8> {
    data.get(0x62..0x62 + 24)?.iter().copied().find(|v| *v != 0)
}

fn read_static_uv_transform(data: &[u8], endian: Endian) -> Option<UvTransformPreview> {
    let scale = read_f32(data.get(0x54..0x58)?, endian);
    let offset = [
        read_f32(data.get(0x58..0x5c)?, endian),
        read_f32(data.get(0x5c..0x60)?, endian),
    ];
    Some(UvTransformPreview {
        scale: [scale, scale],
        offset,
    })
}

fn read_dynamic_model_uv_transform(data: &[u8], endian: Endian) -> Option<UvTransformPreview> {
    Some(UvTransformPreview {
        scale: [
            read_f32(data.get(0x60..0x64)?, endian),
            read_f32(data.get(0x64..0x68)?, endian),
        ],
        offset: [
            read_f32(data.get(0x68..0x6c)?, endian),
            read_f32(data.get(0x6c..0x70)?, endian),
        ],
    })
}

fn shader_constants_from_uv_transform(
    uv_transform: Option<UvTransformPreview>,
) -> Vec<ShaderConstantPreview> {
    uv_transform
        .map(|uv| {
            vec![ShaderConstantPreview {
                name: "uv_scale_offset",
                value: [uv.scale[0], uv.scale[1], uv.offset[0], uv.offset[1]],
                source: "mesh instance UV transform",
            }]
        })
        .unwrap_or_default()
}

fn lod_selection_rank(lod: u8) -> u8 {
    match lod {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        10 => 4,
        4 => 10,
        7 => 20,
        8 => 21,
        9 => 30,
        _ => 40,
    }
}

fn lod_preview_value(lod: u8) -> u8 {
    match lod {
        0 | 1 | 2 | 3 | 10 => 0,
        4 => 1,
        7 | 8 => 2,
        9 => 3,
        _ => lod,
    }
}

#[derive(Clone, Copy)]
enum TagSearchKind {
    VertexBuffer,
    IndexBuffer,
    Texture,
    Technique,
    Shader,
}

fn find_related_tags(
    cache: &TagCache,
    start: TagHash,
    kind: TagSearchKind,
    max_depth: usize,
) -> Vec<(TagHash, UEntryHeader)> {
    let mut out = vec![];
    let mut seen = rustc_hash::FxHashSet::default();
    find_related_tags_recursive(cache, start, kind, 0, max_depth, &mut seen, &mut out);
    out.into_iter()
        .unique_by(|(tag, _)| *tag)
        .take(128)
        .collect()
}

fn find_related_tags_recursive(
    cache: &TagCache,
    tag: TagHash,
    kind: TagSearchKind,
    depth: usize,
    max_depth: usize,
    seen: &mut rustc_hash::FxHashSet<TagHash>,
    out: &mut Vec<(TagHash, UEntryHeader)>,
) {
    if depth > max_depth || !seen.insert(tag) {
        return;
    }

    let Some(scan) = cache.hashes.get(&tag) else {
        return;
    };

    let children = scan
        .file_hashes
        .iter()
        .map(|scanned| scanned.hash)
        .chain(
            scan.file_hashes64
                .iter()
                .filter_map(|scanned| tag64_to_hash32(scanned.hash)),
        )
        .unique()
        .collect_vec();

    for child in children {
        let Some(entry) = package_manager().get_entry(child) else {
            continue;
        };
        let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
        if tag_matches_kind(&entry, tag_type, kind) {
            out.push((child, entry));
        } else if tag_type.is_tag() {
            find_related_tags_recursive(cache, child, kind, depth + 1, max_depth, seen, out);
        }
    }
}

fn tag_matches_kind(entry: &UEntryHeader, tag_type: TagType, kind: TagSearchKind) -> bool {
    match kind {
        TagSearchKind::VertexBuffer => {
            matches!(tag_type, TagType::VertexBuffer { is_header: true })
        }
        TagSearchKind::IndexBuffer => matches!(tag_type, TagType::IndexBuffer { is_header: true }),
        TagSearchKind::Texture => tag_type.is_texture() && tag_type.is_header(),
        TagSearchKind::Technique => is_technique_entry(entry),
        TagSearchKind::Shader => tag_type.is_shader() && tag_type.is_header(),
    }
}

impl VertexBufferHeader {
    fn parse(data: &[u8], endian: Endian) -> anyhow::Result<Self> {
        if data.len() < 12 {
            bail!("Vertex buffer header needs 12 bytes, got {}", data.len());
        }
        Ok(Self {
            data_size: read_u32(&data[0..4], endian),
            stride: read_u16(&data[4..6], endian),
            vtype: read_u16(&data[6..8], endian),
            deadbeef: read_u32(&data[8..12], endian),
        })
    }
}

impl IndexBufferHeader {
    fn parse(data: &[u8], endian: Endian) -> anyhow::Result<Self> {
        if data.len() < 24 {
            bail!("Index buffer header needs 24 bytes, got {}", data.len());
        }
        Ok(Self {
            unk0: data[0] as i8,
            is_32bit: data[1] != 0,
            unk1: read_u16(&data[2..4], endian),
            zero: read_u32(&data[4..8], endian),
            data_size: read_u64(&data[8..16], endian),
            deadbeef: read_u32(&data[16..20], endian),
            zero1: read_u32(&data[20..24], endian),
        })
    }
}

fn find_position_candidates(
    data: &[u8],
    stride: usize,
    endian: Endian,
) -> Vec<VertexPositionCandidate> {
    if stride == 0 || data.len() < stride {
        return vec![];
    }

    let mut candidates = vec![];
    if stride >= 12 {
        if let Some(candidate) = candidate_f32x3(data, stride, 0, endian) {
            candidates.push(candidate);
        }
    }
    if stride >= 8 {
        if let Some(candidate) = candidate_i16x4(data, stride, 0, endian) {
            candidates.push(candidate);
        }
    }
    if stride >= 6 {
        if let Some(candidate) = candidate_i16x3(data, stride, 0, endian) {
            candidates.push(candidate);
        }
    }

    candidates
}

fn find_uv_candidates(data: &[u8], stride: usize, endian: Endian) -> Vec<VertexUvCandidate> {
    if stride == 0 || data.len() < stride {
        return vec![];
    }

    let mut candidates = vec![];
    for offset in (0..stride).step_by(2) {
        if offset + 4 <= stride
            && let Some(candidate) = candidate_f16x2_uv(data, stride, offset, endian)
        {
            candidates.push(candidate);
        }
    }

    candidates
        .into_iter()
        .sorted_by_key(|candidate| std::cmp::Reverse(uv_candidate_score(candidate)))
        .take(8)
        .collect()
}

fn candidate_f16x2_uv(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexUvCandidate> {
    build_uv_candidate(data, stride, offset, UvFormat::F16x2, |bytes| {
        Some([
            read_f16(bytes.get(0..2)?, endian),
            read_f16(bytes.get(2..4)?, endian),
        ])
    })
}

fn candidate_f32x2_uv(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexUvCandidate> {
    build_uv_candidate(data, stride, offset, UvFormat::F32x2, |bytes| {
        Some([
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
        ])
    })
}

fn build_uv_candidate(
    data: &[u8],
    stride: usize,
    offset: usize,
    format: UvFormat,
    mut decode: impl FnMut(&[u8]) -> Option<[f32; 2]>,
) -> Option<VertexUvCandidate> {
    let vertex_count = data.len() / stride;
    let sample_count = vertex_count.min(2048);
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    let mut valid = 0usize;

    for i in 0..sample_count {
        let start = i * stride + offset;
        let vertex_end = (i + 1) * stride;
        let uv = decode(data.get(start..vertex_end)?)?;
        if !uv.iter().all(|v| v.is_finite() && v.abs() <= 1024.0) {
            continue;
        }
        for axis in 0..2 {
            min[axis] = min[axis].min(uv[axis]);
            max[axis] = max[axis].max(uv[axis]);
        }
        valid += 1;
    }

    if valid < sample_count.saturating_div(2).max(1) {
        return None;
    }
    let span_u = (max[0] - min[0]).abs();
    let span_v = (max[1] - min[1]).abs();
    if span_u < 0.0001 && span_v < 0.0001 {
        return None;
    }

    Some(VertexUvCandidate {
        format,
        label: format!("{} @ +{}", format.label(), offset),
        offset,
        valid_vertices: valid,
        sampled_vertices: sample_count,
        min,
        max,
    })
}

fn candidate_f32x3(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(
        data,
        stride,
        offset,
        PositionFormat::F32x3,
        "f32x3 @ +0",
        |bytes| {
            Some([
                read_f32(bytes.get(0..4)?, endian),
                read_f32(bytes.get(4..8)?, endian),
                read_f32(bytes.get(8..12)?, endian),
            ])
        },
    )
}

fn candidate_i16x4(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(
        data,
        stride,
        offset,
        PositionFormat::I16x4,
        "i16x4.xyz @ +0",
        |bytes| {
            Some([
                read_i16(bytes.get(0..2)?, endian) as f32,
                read_i16(bytes.get(2..4)?, endian) as f32,
                read_i16(bytes.get(4..6)?, endian) as f32,
            ])
        },
    )
}

fn candidate_i16x3(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(
        data,
        stride,
        offset,
        PositionFormat::I16x3,
        "i16x3 @ +0",
        |bytes| {
            Some([
                read_i16(bytes.get(0..2)?, endian) as f32,
                read_i16(bytes.get(2..4)?, endian) as f32,
                read_i16(bytes.get(4..6)?, endian) as f32,
            ])
        },
    )
}

fn build_candidate(
    data: &[u8],
    stride: usize,
    offset: usize,
    format: PositionFormat,
    label: &'static str,
    mut decode: impl FnMut(&[u8]) -> Option<[f32; 3]>,
) -> Option<VertexPositionCandidate> {
    let vertex_count = data.len() / stride;
    let sample_count = vertex_count.min(2048);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut valid = 0usize;

    for i in 0..sample_count {
        let start = i * stride + offset;
        let position = decode(data.get(start..start + stride)?)?;
        if !position.iter().all(|v| v.is_finite() && v.abs() < 1.0e8) {
            continue;
        }
        if position == [0.0, 0.0, 0.0] {
            continue;
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
        valid += 1;
    }

    if valid == 0 {
        return None;
    }

    Some(VertexPositionCandidate {
        format,
        label,
        offset,
        valid_vertices: valid,
        sampled_vertices: sample_count,
        min,
        max,
    })
}

fn build_vertex_wireframe(
    tag: TagHash,
    data: &[u8],
    stride: usize,
    endian: Endian,
    candidates: &[VertexPositionCandidate],
    vertex_count_total: usize,
) -> Option<WireframePreview> {
    let candidate = candidates
        .iter()
        .filter(|candidate| candidate.valid_vertices > 0)
        .max_by_key(|candidate| candidate.valid_vertices)?;
    let vertices = decode_positions(data, stride, endian, candidate.format, candidate.offset);
    let (min, max) = bounds(&vertices)?;

    Some(WireframePreview {
        source: tag.to_string(),
        position_format: candidate.label,
        uv_format: None,
        index_count_total: 0,
        vertex_count_total,
        vertices,
        uvs: None,
        indices: vec![],
        min,
        max,
    })
}

fn decode_positions(
    data: &[u8],
    stride: usize,
    endian: Endian,
    format: PositionFormat,
    offset: usize,
) -> Vec<[f32; 3]> {
    if stride == 0 {
        return vec![];
    }

    data.chunks_exact(stride)
        .take(MAX_PREVIEW_VERTICES)
        .filter_map(|vertex| {
            let bytes = vertex.get(offset..)?;
            match format {
                PositionFormat::F32x3 => Some([
                    read_f32(bytes.get(0..4)?, endian),
                    read_f32(bytes.get(4..8)?, endian),
                    read_f32(bytes.get(8..12)?, endian),
                ]),
                PositionFormat::I16x4 | PositionFormat::I16x3 => Some([
                    read_i16(bytes.get(0..2)?, endian) as f32,
                    read_i16(bytes.get(2..4)?, endian) as f32,
                    read_i16(bytes.get(4..6)?, endian) as f32,
                ]),
            }
        })
        .filter(|position| position.iter().all(|v| v.is_finite() && v.abs() < 1.0e8))
        .collect()
}

fn decode_uvs(
    data: &[u8],
    stride: usize,
    endian: Endian,
    format: UvFormat,
    offset: usize,
    vertex_count: usize,
) -> Vec<[f32; 2]> {
    if stride == 0 {
        return vec![];
    }

    data.chunks_exact(stride)
        .take(vertex_count.min(MAX_PREVIEW_VERTICES))
        .filter_map(|vertex| {
            let bytes = vertex.get(offset..)?;
            let uv = match format {
                UvFormat::F32x2 => [
                    read_f32(bytes.get(0..4)?, endian),
                    read_f32(bytes.get(4..8)?, endian),
                ],
                UvFormat::F16x2 => [
                    read_f16(bytes.get(0..2)?, endian),
                    read_f16(bytes.get(2..4)?, endian),
                ],
            };
            uv.iter()
                .all(|v| v.is_finite() && v.abs() <= 1024.0)
                .then_some(uv)
        })
        .collect()
}

fn bounds(vertices: &[[f32; 3]]) -> Option<([f32; 3], [f32; 3])> {
    if vertices.is_empty() {
        return None;
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }

    Some((min, max))
}

fn read_u16(data: &[u8], endian: Endian) -> u16 {
    let bytes = data[0..2].try_into().expect("u16 slice length checked");
    match endian {
        Endian::Big => u16::from_be_bytes(bytes),
        Endian::Little => u16::from_le_bytes(bytes),
    }
}

fn read_i16(data: &[u8], endian: Endian) -> i16 {
    let bytes = data[0..2].try_into().expect("i16 slice length checked");
    match endian {
        Endian::Big => i16::from_be_bytes(bytes),
        Endian::Little => i16::from_le_bytes(bytes),
    }
}

fn read_snorm16(data: &[u8], endian: Endian) -> f32 {
    (read_i16(data, endian) as f32 / 32767.0).clamp(-1.0, 1.0)
}

fn read_unorm16(data: &[u8], endian: Endian) -> f32 {
    read_u16(data, endian) as f32 / 65535.0
}

fn read_f32(data: &[u8], endian: Endian) -> f32 {
    let bytes = data[0..4].try_into().expect("f32 slice length checked");
    match endian {
        Endian::Big => f32::from_be_bytes(bytes),
        Endian::Little => f32::from_le_bytes(bytes),
    }
}

fn read_f16(data: &[u8], endian: Endian) -> f32 {
    let bits = read_u16(data, endian);
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x03ff) as u32;

    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut mantissa = mant;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13),
    };

    f32::from_bits(f32_bits)
}

fn read_u32(data: &[u8], endian: Endian) -> u32 {
    let bytes = data[0..4].try_into().expect("u32 slice length checked");
    match endian {
        Endian::Big => u32::from_be_bytes(bytes),
        Endian::Little => u32::from_le_bytes(bytes),
    }
}

fn read_u64(data: &[u8], endian: Endian) -> u64 {
    let bytes = data[0..8].try_into().expect("u64 slice length checked");
    match endian {
        Endian::Big => u64::from_be_bytes(bytes),
        Endian::Little => u64::from_le_bytes(bytes),
    }
}

fn read_i64(data: &[u8], endian: Endian) -> i64 {
    let bytes = data[0..8].try_into().expect("i64 slice length checked");
    match endian {
        Endian::Big => i64::from_be_bytes(bytes),
        Endian::Little => i64::from_le_bytes(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_sample_model_uvs_and_textures() {
        init_goliath_test_package_manager();

        let tag = TagHash(0x80B14039);
        let entry = package_manager().get_entry(tag).expect("sample tag entry");
        assert_eq!(entry.reference, CLASS_GEOMETRY_RESOURCE);

        let (source, wireframe) = parse_model_wireframe(tag, &entry).expect("sample wireframe");
        eprintln!(
            "mesh kind={} technique={:?} layout={:?} ib={} vb0={} vb1={} index_start={} index_count={} lod={} wireframe verts={} indices={} source={} uv={:?}",
            source.kind,
            source.technique,
            source.input_layout_index,
            source.index_buffer,
            source.vertex0_buffer,
            source.vertex1_buffer,
            source.index_start,
            source.index_count,
            source.lod_category,
            wireframe.vertices.len(),
            wireframe.indices.len(),
            wireframe.source,
            wireframe.uv_format
        );
        assert_eq!(source.lod_category, 0, "sample preview should expose lod0");
        assert_eq!(source.input_layout_index, Some(7));
        assert_eq!(source.technique, Some(TagHash(0x80B1247F)));
        assert!(
            wireframe.uvs.as_ref().is_some_and(|uvs| !uvs.is_empty()),
            "sample must decode real UVs"
        );
        assert!(
            wireframe
                .uv_format
                .as_deref()
                .is_some_and(|uv| uv.contains("R16G16_SNORM") && uv.contains("layout Some(7)")),
            "sample must use declared geometry layout UVs, got {:?}",
            wireframe.uv_format
        );

        let cache = quicktag_scanner::load_tag_cache();
        let techniques = find_model_technique_entries(&cache, tag, &entry);
        eprintln!(
            "techniques={:?}",
            techniques
                .iter()
                .take(16)
                .map(|(tag, _)| format!("{tag}"))
                .collect_vec()
        );
        assert!(!techniques.is_empty(), "sample must resolve techniques");
        assert_eq!(
            techniques.first().map(|(tag, _)| *tag),
            source.technique,
            "selected geometry technique should rank first"
        );

        let textures = find_model_textures(&cache, tag, &techniques);
        eprintln!(
            "textures={:?}",
            textures
                .iter()
                .take(32)
                .map(|(tag, entry)| {
                    let desc = crate::texture::Texture::load_data_d2(*tag, false)
                        .map(|(desc, _, _)| {
                            format!(
                                "{}x{}x{} {:?}",
                                desc.width, desc.height, desc.depth, desc.format
                            )
                        })
                        .unwrap_or_else(|err| format!("load_err={err}"));
                    format!("{tag}:{}:{desc}", entry.file_size)
                })
                .collect_vec()
        );
        assert!(!textures.is_empty(), "sample must resolve textures");
        assert_eq!(
            textures.first().map(|(tag, _)| *tag),
            Some(TagHash(0x80B14064))
        );

        let complete_tag = TagHash(0x80B140B7);
        let complete_entry = package_manager()
            .get_entry(complete_tag)
            .expect("sample complete geometry entry");
        let Some((complete_source, complete_wireframe)) =
            parse_model_wireframe(complete_tag, &complete_entry)
        else {
            panic!("sample complete geometry should parse");
        };
        eprintln!(
            "complete mesh kind={} technique={:?} index_start={} index_count={} lod={} wireframe indices={}",
            complete_source.kind,
            complete_source.technique,
            complete_source.index_start,
            complete_source.index_count,
            complete_source.lod_category,
            complete_wireframe.indices.len()
        );
        assert_eq!(complete_source.lod_category, 0);
        assert!(
            complete_source.index_count > 3849,
            "geometry resources should merge all primary ranges, not just the first range"
        );
        assert!(
            complete_wireframe.indices.len() > 3849,
            "wireframe should include all primary ranges for 80B140B7"
        );
    }

    #[test]
    #[ignore = "requires installed Marathon packages; set QUICKTAG_MARATHON_PACKAGES"]
    fn probes_goliath_model_uv_coverage() {
        init_goliath_test_package_manager();

        let max_models = std::env::var("QUICKTAG_UV_PROBE_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(400);
        let package_filter = std::env::var("QUICKTAG_UV_PROBE_PACKAGE_FILTER").ok();
        let tags = package_manager()
            .get_all_by_reference(CLASS_GEOMETRY_RESOURCE)
            .into_iter()
            .filter(|(tag, _entry)| {
                package_filter.as_ref().is_none_or(|filter| {
                    package_manager()
                        .package_paths
                        .get(&tag.pkg_id())
                        .is_some_and(|path| path.name.contains(filter))
                })
            })
            .take(max_models)
            .collect_vec();

        let mut parsed = 0usize;
        let mut declared = 0usize;
        let mut heuristic = Vec::new();
        let mut missing = Vec::new();
        let mut no_techniques = Vec::new();
        let mut no_textures = Vec::new();
        let cache = if std::env::var("QUICKTAG_UV_PROBE_FULL_CACHE").is_ok() {
            quicktag_scanner::load_tag_cache()
        } else {
            TagCache::default()
        };

        for (tag, entry) in tags {
            let Some((source, wireframe)) = parse_model_wireframe(tag, &entry) else {
                continue;
            };
            parsed += 1;
            match wireframe.uv_format.as_deref() {
                Some(uv) if uv.contains(" layout ") => declared += 1,
                Some(uv) => heuristic.push((tag, source.input_layout_index, uv.to_string())),
                None => missing.push((tag, source.input_layout_index)),
            }

            let techniques = find_model_technique_entries(&cache, tag, &entry);
            if techniques.is_empty() {
                no_techniques.push(tag);
                continue;
            }
            if find_model_textures(&cache, tag, &techniques).is_empty() {
                no_textures.push((tag, techniques.iter().map(|(tag, _)| *tag).collect_vec()));
            }
        }

        eprintln!(
            "goliath uv coverage parsed={parsed} declared={declared} heuristic={} missing={} no_techniques={} no_textures={}",
            heuristic.len(),
            missing.len(),
            no_techniques.len(),
            no_textures.len()
        );
        eprintln!(
            "heuristic first={:?}",
            heuristic
                .iter()
                .take(16)
                .map(|(tag, layout, uv)| format!("{tag}:{layout:?}:{uv}"))
                .collect_vec()
        );
        eprintln!(
            "missing first={:?}",
            missing
                .iter()
                .take(16)
                .map(|(tag, layout)| format!("{tag}:{layout:?}"))
                .collect_vec()
        );
        eprintln!(
            "no_techniques first={:?}",
            no_techniques
                .iter()
                .take(16)
                .map(|tag| format!("{tag}"))
                .collect_vec()
        );
        eprintln!(
            "no_textures first={:?}",
            no_textures
                .iter()
                .take(16)
                .map(|(tag, techniques)| format!("{tag}:{}", techniques.len()))
                .collect_vec()
        );
        for (tag, techniques) in no_textures.iter().take(16) {
            eprintln!(
                "no_texture tag={tag} techniques={:?}",
                techniques
                    .iter()
                    .take(16)
                    .map(|technique| format!("{technique}"))
                    .collect_vec()
            );
            for technique in techniques.iter().take(4) {
                let entry = package_manager()
                    .get_entry(*technique)
                    .expect("technique entry");
                let data = package_manager()
                    .read_tag(*technique)
                    .expect("technique data");
                eprintln!(
                    "  technique {technique} texture_bindings={:?}",
                    texture_bindings_for_technique(&entry, &data)
                );
            }
            if let Some(scan) = cache.hashes.get(tag) {
                eprintln!(
                    "  parents={:?}",
                    scan.references
                        .iter()
                        .take(16)
                        .map(|parent| {
                            let class_name = package_manager()
                                .get_entry(*parent)
                                .and_then(|entry| get_class_by_id(entry.reference))
                                .map(|class| class.name.to_string())
                                .unwrap_or_else(|| "unknown".to_string());
                            format!("{parent}:{class_name}")
                        })
                        .collect_vec()
                );
                for parent in scan.references.iter().take(8) {
                    let parent_textures =
                        find_related_tags(&cache, *parent, TagSearchKind::Texture, 4);
                    let parent_techniques =
                        find_related_tags(&cache, *parent, TagSearchKind::Technique, 4);
                    eprintln!(
                        "  parent {parent}: textures={:?} techniques={:?}",
                        parent_textures
                            .iter()
                            .take(8)
                            .map(|(tag, _)| format!("{tag}"))
                            .collect_vec(),
                        parent_techniques
                            .iter()
                            .take(8)
                            .map(|(tag, _)| format!("{tag}"))
                            .collect_vec()
                    );
                }
            }
        }

        assert!(parsed > 0, "probe found no parseable geometry resources");
        assert!(
            heuristic.is_empty(),
            "some parsed models still use heuristic UVs"
        );
    }

    fn init_goliath_test_package_manager() {
        use std::{path::PathBuf, sync::Arc};
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        assert!(
            packages.exists(),
            "packages path missing: {}",
            packages.display()
        );

        let pm = PackageManager::new(
            packages.to_string_lossy().to_string(),
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
    }

    #[test]
    fn parses_vertex_buffer_header_little_endian() {
        let data = [
            0x80, 0x00, 0x00, 0x00, // data_size
            0x20, 0x00, // stride
            0x03, 0x00, // vtype
            0xEF, 0xBE, 0xAD, 0xDE, // marker
        ];

        let header = VertexBufferHeader::parse(&data, Endian::Little).unwrap();
        assert_eq!(header.data_size, 0x80);
        assert_eq!(header.stride, 0x20);
        assert_eq!(header.vtype, 3);
        assert_eq!(header.deadbeef, 0xDEADBEEF);
    }

    #[test]
    fn parses_index_buffer_header_little_endian() {
        let data = [
            0x7F, 0x01, // unk0, is_32bit
            0x34, 0x12, // unk1
            0x00, 0x00, 0x00, 0x00, // zero
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // data_size
            0xEF, 0xBE, 0xAD, 0xDE, // marker
            0x00, 0x00, 0x00, 0x00, // zero1
        ];

        let header = IndexBufferHeader::parse(&data, Endian::Little).unwrap();
        assert_eq!(header.unk0, 0x7F);
        assert!(header.is_32bit);
        assert_eq!(header.unk1, 0x1234);
        assert_eq!(header.data_size, 0x40);
        assert_eq!(header.deadbeef, 0xDEADBEEF);
    }

    #[test]
    fn reads_relative_array_payload() {
        let mut data = vec![0u8; 0x40];
        data[0x08..0x10].copy_from_slice(&2u64.to_le_bytes());
        data[0x10..0x18].copy_from_slice(&0x10i64.to_le_bytes());
        data[0x20..0x28].copy_from_slice(&2u64.to_le_bytes());
        data[0x28..0x2c].copy_from_slice(&0x80806D37u32.to_le_bytes());
        data[0x30..0x34].copy_from_slice(&0x11223344u32.to_le_bytes());
        data[0x34..0x38].copy_from_slice(&0x55667788u32.to_le_bytes());

        let array = read_array(&data, 0x08, 4, Endian::Little).unwrap();
        assert_eq!(array, &data[0x30..0x38]);
    }

    #[test]
    fn reads_static_mesh_technique_tags() {
        let mut data = vec![0u8; 0xa0];
        let technique0 = TagHash::new(0x0102, 0x0304);
        let technique1 = TagHash::new(0x0506, 0x0708);
        let technique2 = TagHash::new(0x0a0b, 0x0c0d);
        data[0x10..0x18].copy_from_slice(&2u64.to_le_bytes());
        data[0x18..0x20].copy_from_slice(&0x28i64.to_le_bytes());
        data[0x40..0x48].copy_from_slice(&2u64.to_le_bytes());
        data[0x50..0x54].copy_from_slice(&technique0.0.to_le_bytes());
        data[0x54..0x58].copy_from_slice(&technique1.0.to_le_bytes());

        data[0x20..0x28].copy_from_slice(&1u64.to_le_bytes());
        data[0x28..0x30].copy_from_slice(&0x38i64.to_le_bytes());
        data[0x60..0x68].copy_from_slice(&1u64.to_le_bytes());
        data[0x90..0x94].copy_from_slice(&technique2.0.to_le_bytes());

        let tags = static_mesh_technique_tags(&data, Endian::Little);

        assert_eq!(tags, vec![technique0, technique1, technique2]);
    }

    #[test]
    fn reads_dynamic_mesh_part_technique_tags() {
        let mut data = vec![0u8; 0xa0];
        let technique0 = TagHash::new(0x0102, 0x0304);
        let technique1 = TagHash::new(0x0506, 0x0708);
        data[0x20..0x28].copy_from_slice(&2u64.to_le_bytes());
        data[0x28..0x30].copy_from_slice(&0x18i64.to_le_bytes());
        data[0x40..0x48].copy_from_slice(&2u64.to_le_bytes());
        data[0x50..0x54].copy_from_slice(&technique0.0.to_le_bytes());
        data[0x74..0x78].copy_from_slice(&technique1.0.to_le_bytes());

        let tags = dynamic_mesh_technique_tags(&data, Endian::Little);

        assert_eq!(tags, vec![technique0, technique1]);
    }

    #[test]
    fn scans_candidate_tag_hashes_in_blob() {
        let little = TagHash::new(0x0102, 0x0304);
        let big = TagHash::new(0x0506, 0x0708);
        let mut data = vec![0u8; 0x10];
        data[0x04..0x08].copy_from_slice(&little.0.to_le_bytes());
        data[0x0c..0x10].copy_from_slice(&big.0.to_be_bytes());

        let little_tags = candidate_tag_hashes_in_blob(&data, Endian::Little);
        let big_tags = candidate_tag_hashes_in_blob(&data, Endian::Big);

        assert!(little_tags.contains(&little));
        assert!(big_tags.contains(&big));
    }

    #[test]
    fn reads_half_float_values() {
        assert_eq!(read_f16(&0x3c00u16.to_le_bytes(), Endian::Little), 1.0);
        assert_eq!(read_f16(&0xc000u16.to_le_bytes(), Endian::Little), -2.0);
        assert_eq!(read_f16(&0x3800u16.to_le_bytes(), Endian::Little), 0.5);
    }

    #[test]
    fn finds_half_uv_candidates() {
        let mut data = vec![0u8; 0x18];
        data[0x04..0x06].copy_from_slice(&0x0000u16.to_le_bytes());
        data[0x06..0x08].copy_from_slice(&0x0000u16.to_le_bytes());
        data[0x0c..0x0e].copy_from_slice(&0x3c00u16.to_le_bytes());
        data[0x0e..0x10].copy_from_slice(&0x3800u16.to_le_bytes());
        data[0x14..0x16].copy_from_slice(&0x4000u16.to_le_bytes());
        data[0x16..0x18].copy_from_slice(&0x3c00u16.to_le_bytes());

        let candidate = candidate_f16x2_uv(&data, 0x08, 0x04, Endian::Little).unwrap();

        assert_eq!(candidate.format, UvFormat::F16x2);
        assert_eq!(candidate.offset, 0x04);
        assert_eq!(candidate.valid_vertices, 3);
        assert_eq!(candidate.min, [0.0, 0.0]);
        assert_eq!(candidate.max, [2.0, 1.0]);
    }

    #[test]
    fn decodes_input_layout_snorm_uvs() {
        let mut data = vec![0u8; 0x10];
        data[0x00..0x02].copy_from_slice(&0x4000i16.to_le_bytes());
        data[0x02..0x04].copy_from_slice(&(-0x4000i16).to_le_bytes());
        data[0x08..0x0a].copy_from_slice(&0x7fffi16.to_le_bytes());
        data[0x0a..0x0c].copy_from_slice(&0i16.to_le_bytes());

        let uvs = decode_input_layout_uvs(
            &data,
            0x08,
            Endian::Little,
            InputLayoutTexcoord {
                buffer_index: 0,
                offset: 0,
                format: InputLayoutFormat::R16G16Snorm,
            },
            2,
        );

        assert_eq!(uvs.len(), 2);
        assert!((uvs[0][0] - 0.50001526).abs() < 0.00001);
        assert!((uvs[0][1] + 0.50001526).abs() < 0.00001);
        assert_eq!(uvs[1], [1.0, 0.0]);
    }

    #[test]
    fn propagates_uv_transform_to_shader_constant() {
        let constants = shader_constants_from_uv_transform(Some(UvTransformPreview {
            scale: [2.0, 3.0],
            offset: [0.25, 0.5],
        }));

        assert_eq!(constants.len(), 1);
        assert_eq!(constants[0].name, "uv_scale_offset");
        assert_eq!(constants[0].value, [2.0, 3.0, 0.25, 0.5]);
        assert_eq!(constants[0].source, "mesh instance UV transform");
    }

    #[test]
    fn reads_geometry_buffer_set_generate_gbuffer_layout() {
        let mut mesh = vec![0u8; 0x80];
        mesh[0x62] = 0x57;
        mesh[0x64] = 0x07;

        assert_eq!(geometry_buffer_set_input_layout_id(&mesh), Some(7));
    }
}

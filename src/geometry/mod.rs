use anyhow::{Context, bail};
use binrw::Endian;
use itertools::Itertools;
use quicktag_core::classes::get_class_by_id;
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::TagCache;
use std::sync::Arc;
use tiger_pkg::{TagHash, TagHash64, Version, package::UEntryHeader, package_manager};
use wgpu::util::DeviceExt;

use crate::material::{is_technique_entry, texture_tags_for_technique};

const MAX_PREVIEW_VERTICES: usize = 200_000;
const MAX_PREVIEW_INDICES: usize = 300_000;
const TECHNIQUE_SCAN_CHILD_LIMIT: usize = 128;

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

const INPUT_LAYOUT_TEXCOORDS: [Option<InputLayoutTexcoord>; 27] = [
    None,
    None,
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 8,
        format: InputLayoutFormat::R32G32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 12,
        format: InputLayoutFormat::R32G32Float,
    }),
    None,
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 8,
        format: InputLayoutFormat::R32G32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 40,
        format: InputLayoutFormat::R32G32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 1,
        offset: 0,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 1,
        offset: 0,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 8,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    None,
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 8,
        format: InputLayoutFormat::R32G32B32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 12,
        format: InputLayoutFormat::R32G32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 1,
        offset: 0,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 40,
        format: InputLayoutFormat::R32G32B32A32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 16,
        format: InputLayoutFormat::R32G32Float,
    }),
    None,
    Some(InputLayoutTexcoord {
        buffer_index: 1,
        offset: 0,
        format: InputLayoutFormat::R32G32B32Float,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 8,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 16,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 16,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 1,
        offset: 0,
        format: InputLayoutFormat::R32G32B32Float,
    }),
    None,
    None,
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 12,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 12,
        format: InputLayoutFormat::R16G16Snorm,
    }),
    Some(InputLayoutTexcoord {
        buffer_index: 0,
        offset: 8,
        format: InputLayoutFormat::R16G16Snorm,
    }),
];

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
    let technique_textures = techniques
        .iter()
        .map(|(tag, _entry)| *tag)
        .into_iter()
        .flat_map(texture_tags_from_technique)
        .filter_map(|texture_tag| {
            package_manager()
                .get_entry(texture_tag)
                .map(|entry| (texture_tag, entry))
        });
    let recursive_textures = find_related_tags(cache, tag, TagSearchKind::Texture, 8);

    technique_textures
        .chain(recursive_textures)
        .unique_by(|(tag, _entry)| *tag)
        .take(128)
        .collect()
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

fn texture_tags_from_technique(tag: TagHash) -> Vec<TagHash> {
    let Some(entry) = package_manager().get_entry(tag) else {
        return vec![];
    };
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };

    texture_tags_for_technique(&entry, &data)
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
        _ => vec![],
    };
    let raw_current = technique_tags_in_blob(&data, endian);
    let raw_children = find_child_technique_tags(cache, tag, endian);
    let recursive = find_related_tags(cache, tag, TagSearchKind::Technique, 8)
        .into_iter()
        .map(|(tag, _entry)| tag);

    direct
        .into_iter()
        .chain(raw_current)
        .chain(raw_children)
        .chain(recursive)
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
    for tri in indices.chunks_exact(3).take(20_000) {
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
        .find(|group| {
            parts
                .get(group.part_index as usize)
                .is_some_and(|part| is_highest_detail_lod(part.lod_category))
        })
        .or_else(|| groups.first())?;
    let part = parts.get(group.part_index as usize)?;
    let buffers = buffers.get(part.buffer_index as usize)?;

    let uv_transform = read_static_uv_transform(data, endian);
    let source = MeshSourcePreview {
        kind: "static mesh data",
        buffer_index: part.buffer_index as usize,
        index_start: part.index_start,
        index_count: part.index_count,
        primitive_type: part.primitive_type,
        lod_category: part.lod_category,
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
        Some(source.index_start as usize..(source.index_start + source.index_count) as usize),
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
        .find(|part| is_highest_detail_lod(part.lod_category))
        .or_else(|| parts.first())?;
    let source = MeshSourcePreview {
        kind: "dynamic mesh",
        buffer_index: 0,
        index_start: part.index_start,
        index_count: part.index_count,
        primitive_type: part.primitive_type,
        lod_category: part.lod_category,
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
        Some(source.index_start as usize..(source.index_start + source.index_count) as usize),
        source.input_layout_index,
    )?;

    Some((source, wireframe))
}

fn build_wireframe_from_refs(
    vertex_tags: &[TagHash],
    index_tag: TagHash,
    index_range: Option<std::ops::Range<usize>>,
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
    wireframe.indices = if let Some(range) = index_range {
        index_preview
            .indices
            .get(
                range.start.min(index_preview.indices.len())
                    ..range.end.min(index_preview.indices.len()),
            )
            .unwrap_or_default()
            .to_vec()
    } else {
        index_preview.indices
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
    let layout = INPUT_LAYOUT_TEXCOORDS
        .get(input_layout_index? as usize)?
        .as_ref()?;
    let (tag, preview) = vertex_previews.get(layout.buffer_index)?;
    let entry = package_manager().get_entry(*tag)?;
    let data = package_manager().read_tag(TagHash(entry.reference)).ok()?;
    let endian = package_manager().version.endian();
    let uvs = decode_input_layout_uvs(
        &data,
        preview.header.stride as usize,
        endian,
        *layout,
        vertex_count,
    );

    (!uvs.is_empty()).then_some((*tag, layout.format, uvs))
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
    index_start: u32,
    index_count: u32,
    primitive_type: u8,
    lod_category: u8,
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
                primitive_type: *part.get(0x6)?,
                index_start: read_u32(part.get(0x8..0xc)?, endian),
                index_count: read_u32(part.get(0xc..0x10)?, endian),
                lod_category: *part.get(0x1d)?,
            })
        })
        .collect()
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

fn is_highest_detail_lod(lod: u8) -> bool {
    matches!(lod, 0 | 1 | 2 | 3 | 10)
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
}

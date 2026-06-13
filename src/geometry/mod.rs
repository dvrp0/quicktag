use anyhow::{Context, bail};
use binrw::Endian;
use itertools::Itertools;
use quicktag_core::classes::get_class_by_id;
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::TagCache;
use std::sync::Arc;
use tiger_pkg::{TagHash, Version, package::UEntryHeader, package_manager};

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
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ModelPreview {
    pub label: &'static str,
    pub class_name: Option<String>,
    pub vertex_buffers: Vec<(TagHash, UEntryHeader)>,
    pub index_buffers: Vec<(TagHash, UEntryHeader)>,
    pub textures: Vec<(TagHash, UEntryHeader)>,
    pub shaders: Vec<(TagHash, UEntryHeader)>,
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
    pub label: &'static str,
    pub offset: usize,
    pub valid_vertices: usize,
    pub sampled_vertices: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
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
                    load_vertex_buffer_preview(entry, tag_data).ok()?,
                ),
            });
        }

        if matches!(tag_type, TagType::IndexBuffer { is_header: true }) {
            return Some(Self {
                kind: GeometryPreviewKind::IndexBuffer(
                    load_index_buffer_preview(entry, tag_data).ok()?,
                ),
            });
        }

        model_label_for_reference(entry.reference).map(|label| Self {
            kind: GeometryPreviewKind::Model(load_model_preview(cache, tag, entry, label)),
        })
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

pub fn model_label_for_reference(reference: u32) -> Option<&'static str> {
    match reference {
        0x80806D44 => Some("Static mesh"),
        0x80806D30 => Some("Static mesh data"),
        0x80806F07 => Some("Dynamic model"),
        0x80806EC5 => Some("Dynamic mesh"),
        _ => None,
    }
}

fn load_vertex_buffer_preview(
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

    Ok(VertexBufferPreview {
        candidates: find_position_candidates(&data, header.stride as usize, endian),
        header,
        data_tag,
        data_len: data.len(),
        element_count,
        warnings,
    })
}

fn load_index_buffer_preview(
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

    Ok(IndexBufferPreview {
        header,
        data_tag,
        data_len: data.len(),
        index_count,
        min_index,
        max_index,
        first_indices: indices,
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
    ModelPreview {
        label,
        class_name,
        vertex_buffers: find_related_tags(&cache, tag, TagSearchKind::VertexBuffer, 8),
        index_buffers: find_related_tags(&cache, tag, TagSearchKind::IndexBuffer, 8),
        textures: find_related_tags(&cache, tag, TagSearchKind::Texture, 8),
        shaders: find_related_tags(&cache, tag, TagSearchKind::Shader, 8),
    }
}

#[derive(Clone, Copy)]
enum TagSearchKind {
    VertexBuffer,
    IndexBuffer,
    Texture,
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

    for scanned in &scan.file_hashes {
        let child = scanned.hash;
        let Some(entry) = package_manager().get_entry(child) else {
            continue;
        };
        let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
        if tag_matches_kind(tag_type, kind) {
            out.push((child, entry));
        } else if tag_type.is_tag() {
            find_related_tags_recursive(cache, child, kind, depth + 1, max_depth, seen, out);
        }
    }
}

fn tag_matches_kind(tag_type: TagType, kind: TagSearchKind) -> bool {
    match kind {
        TagSearchKind::VertexBuffer => {
            matches!(tag_type, TagType::VertexBuffer { is_header: true })
        }
        TagSearchKind::IndexBuffer => matches!(tag_type, TagType::IndexBuffer { is_header: true }),
        TagSearchKind::Texture => tag_type.is_texture() && tag_type.is_header(),
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

fn candidate_f32x3(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(data, stride, offset, "f32x3 @ +0", |bytes| {
        Some([
            read_f32(bytes.get(0..4)?, endian),
            read_f32(bytes.get(4..8)?, endian),
            read_f32(bytes.get(8..12)?, endian),
        ])
    })
}

fn candidate_i16x4(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(data, stride, offset, "i16x4.xyz @ +0", |bytes| {
        Some([
            read_i16(bytes.get(0..2)?, endian) as f32,
            read_i16(bytes.get(2..4)?, endian) as f32,
            read_i16(bytes.get(4..6)?, endian) as f32,
        ])
    })
}

fn candidate_i16x3(
    data: &[u8],
    stride: usize,
    offset: usize,
    endian: Endian,
) -> Option<VertexPositionCandidate> {
    build_candidate(data, stride, offset, "i16x3 @ +0", |bytes| {
        Some([
            read_i16(bytes.get(0..2)?, endian) as f32,
            read_i16(bytes.get(2..4)?, endian) as f32,
            read_i16(bytes.get(4..6)?, endian) as f32,
        ])
    })
}

fn build_candidate(
    data: &[u8],
    stride: usize,
    offset: usize,
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
        label,
        offset,
        valid_vertices: valid,
        sampled_vertices: sample_count,
        min,
        max,
    })
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

fn read_f32(data: &[u8], endian: Endian) -> f32 {
    let bytes = data[0..4].try_into().expect("f32 slice length checked");
    match endian {
        Endian::Big => f32::from_be_bytes(bytes),
        Endian::Little => f32::from_le_bytes(bytes),
    }
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
}

use binrw::Endian;
use quicktag_core::classes::get_class_by_id;
use tiger_pkg::{TagHash, TagHash64, Version, package::UEntryHeader, package_manager};

#[derive(Debug, Clone)]
pub struct MaterialTagPreview {
    pub class_name: Option<String>,
    pub kind: MaterialPreviewKind,
}

#[derive(Debug, Clone)]
pub enum MaterialPreviewKind {
    Technique(TechniquePreview),
}

#[derive(Debug, Clone)]
pub struct TechniquePreview {
    pub bind_mode: u32,
    pub state_selection: u32,
    pub used_scopes: u64,
    pub compatible_scopes: u64,
    pub stages: Vec<TechniqueStagePreview>,
}

#[derive(Debug, Clone)]
pub struct TechniqueStagePreview {
    pub stage: &'static str,
    pub shader: Option<TagHash>,
    pub textures: Vec<TextureSlotBindingPreview>,
    pub sampler_count: usize,
    pub constant_count: usize,
    pub bytecode_len: usize,
    pub constant_buffer_slot: Option<i32>,
    pub constant_buffer: Option<TagHash>,
}

#[derive(Debug, Clone)]
pub struct TextureSlotBindingPreview {
    pub slot: u32,
    pub texture: WideHashPreview,
}

#[derive(Debug, Clone, Copy)]
pub struct WideHashPreview {
    pub raw32: TagHash,
    pub is_hash32: bool,
    pub raw64: TagHash64,
    pub resolved: Option<TagHash>,
}

impl MaterialTagPreview {
    pub fn load(entry: &UEntryHeader, data: &[u8]) -> Option<Self> {
        let class_name = get_class_by_id(entry.reference).map(|c| c.name.to_string());
        match entry.reference {
            0x80806DAA => Some(Self {
                class_name,
                kind: MaterialPreviewKind::Technique(parse_technique(data)?),
            }),
            _ => None,
        }
    }
}

fn parse_technique(data: &[u8]) -> Option<TechniquePreview> {
    let endian = package_manager().version.endian();
    let bind_mode = read_u32(data.get(0x8..0xc)?, endian);
    let used_scopes = read_u64(data.get(0x20..0x28)?, endian);
    let compatible_scopes = read_u64(data.get(0x28..0x30)?, endian);
    let state_selection = read_u32(data.get(0x30..0x34)?, endian);
    let stages = [
        ("VS", 0usize),
        ("GS", 3usize),
        ("PS", 4usize),
        ("CS", 5usize),
    ]
    .into_iter()
    .filter_map(|(name, index)| parse_technique_stage(data, name, 0x70 + index * 0x90, endian))
    .collect();

    Some(TechniquePreview {
        bind_mode,
        state_selection,
        used_scopes,
        compatible_scopes,
        stages,
    })
}

fn parse_technique_stage(
    data: &[u8],
    stage: &'static str,
    offset: usize,
    endian: Endian,
) -> Option<TechniqueStagePreview> {
    let shader = read_tag(data.get(offset..offset + 4)?, endian);
    let constants_offset = offset + 0x20;
    let textures: Vec<TextureSlotBindingPreview> = read_array(data, offset + 0x8, 0x18, endian)
        .unwrap_or_default()
        .chunks_exact(0x18)
        .filter_map(|texture| {
            Some(TextureSlotBindingPreview {
                slot: read_u32(texture.get(0x0..0x4)?, endian),
                texture: read_wide_hash(texture.get(0x8..0x18)?, endian)?,
            })
        })
        .collect();
    let bytecode_len = read_array(data, constants_offset, 1, endian)
        .map(|bytecode| bytecode.len())
        .unwrap_or_default();
    let constant_count = read_array(data, constants_offset + 0x10, 0x10, endian)
        .map(|constants| constants.len() / 0x10)
        .unwrap_or_default();
    let sampler_count = read_array(data, constants_offset + 0x20, 0x10, endian)
        .map(|samplers| samplers.len() / 0x10)
        .unwrap_or_default();
    let constant_buffer_slot = data
        .get(constants_offset + 0x50..constants_offset + 0x54)
        .map(|bytes| read_i32(bytes, endian))
        .filter(|slot| *slot >= 0);
    let constant_buffer = data
        .get(constants_offset + 0x54..constants_offset + 0x58)
        .map(|bytes| read_tag(bytes, endian))
        .filter(|tag| tag.is_some());

    if shader.is_none()
        && textures.is_empty()
        && sampler_count == 0
        && constant_count == 0
        && bytecode_len == 0
        && constant_buffer_slot.is_none()
    {
        return None;
    }

    Some(TechniqueStagePreview {
        stage,
        shader: shader.is_some().then_some(shader),
        textures,
        sampler_count,
        constant_count,
        bytecode_len,
        constant_buffer_slot,
        constant_buffer,
    })
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

fn read_wide_hash(data: &[u8], endian: Endian) -> Option<WideHashPreview> {
    let raw32 = read_tag(data.get(0x0..0x4)?, endian);
    let is_hash32 = read_u32(data.get(0x4..0x8)?, endian) != 0;
    let raw64 = TagHash64(read_u64(data.get(0x8..0x10)?, endian));
    let resolved = if is_hash32 {
        raw32.is_some().then_some(raw32)
    } else {
        package_manager()
            .lookup
            .tag64_entries
            .get(&raw64.0)
            .map(|entry| entry.hash32)
    };

    Some(WideHashPreview {
        raw32,
        is_hash32,
        raw64,
        resolved,
    })
}

fn read_tag(data: &[u8], endian: Endian) -> TagHash {
    TagHash(read_u32(data, endian))
}

fn read_u32(data: &[u8], endian: Endian) -> u32 {
    match endian {
        Endian::Little => u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        Endian::Big => u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
    }
}

fn read_i32(data: &[u8], endian: Endian) -> i32 {
    read_u32(data, endian) as i32
}

fn read_u64(data: &[u8], endian: Endian) -> u64 {
    match endian {
        Endian::Little => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
        Endian::Big => u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
    }
}

fn read_i64(data: &[u8], endian: Endian) -> i64 {
    read_u64(data, endian) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_wide_hash32() {
        let mut data = vec![0; 0x10];
        data[0x0..0x4].copy_from_slice(&0x80806DCF_u32.to_le_bytes());
        data[0x4..0x8].copy_from_slice(&1_u32.to_le_bytes());

        let hash = read_wide_hash(&data, Endian::Little).unwrap();

        assert!(hash.is_hash32);
        assert_eq!(hash.raw32.0, 0x80806DCF);
        assert_eq!(hash.resolved.unwrap().0, 0x80806DCF);
    }

    #[test]
    fn reads_relative_array_payload() {
        let mut data = vec![0; 0x50];
        data[0x10..0x18].copy_from_slice(&2_u64.to_le_bytes());
        data[0x18..0x20].copy_from_slice(&0x18_i64.to_le_bytes());
        data[0x30..0x38].copy_from_slice(&2_u64.to_le_bytes());
        data[0x40..0x42].copy_from_slice(&0x1122_u16.to_le_bytes());
        data[0x42..0x44].copy_from_slice(&0x3344_u16.to_le_bytes());

        let array = read_array(&data, 0x10, 2, Endian::Little).unwrap();

        assert_eq!(u16::from_le_bytes([array[0], array[1]]), 0x1122);
        assert_eq!(u16::from_le_bytes([array[2], array[3]]), 0x3344);
    }
}

use binrw::Endian;
use quicktag_core::classes::get_class_by_id;
use quicktag_core::tagtypes::TagType;
use tiger_pkg::{GameVersion, TagHash, TagHash64, Version, package::UEntryHeader, package_manager};

use crate::texture::Texture;

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TechniqueRenderState {
    pub blend: Option<u8>,
    pub depth_stencil: Option<u8>,
    pub rasterizer: Option<u8>,
    pub depth_bias: Option<u8>,
}

impl TechniqueRenderState {
    pub fn from_raw(raw: u32) -> Self {
        let decode = |shift: u32| {
            let value = ((raw >> shift) & 0xff) as u8;
            (value & 0x80 != 0).then_some(value & 0x7f)
        };
        Self {
            blend: decode(0),
            depth_stencil: decode(8),
            rasterizer: decode(16),
            depth_bias: decode(24),
        }
    }
}

impl TechniquePreview {
    pub fn used_scope_names(&self) -> Vec<&'static str> {
        tfx_scope_names(self.used_scopes)
    }

    pub fn compatible_scope_names(&self) -> Vec<&'static str> {
        tfx_scope_names(self.compatible_scopes)
    }
}

#[derive(Debug, Clone)]
pub struct TechniqueStagePreview {
    pub stage: &'static str,
    pub shader: Option<TagHash>,
    pub textures: Vec<TextureSlotBindingPreview>,
    pub constants: Vec<[f32; 4]>,
    pub indexed_resources: Vec<WideHashPreview>,
    pub inline_constants: Vec<[f32; 4]>,
    pub bytecode_len: usize,
    pub constant_buffer_slot: Option<i32>,
    pub constant_buffer: Option<TagHash>,
    pub constant_buffer_preview: Option<ConstantBufferPreview>,
    pub bytecode: TfxBytecodePreview,
}

#[derive(Debug, Clone)]
pub struct ConstantBufferPreview {
    pub header_tag: TagHash,
    pub data_tag: TagHash,
    pub header_len: usize,
    pub data_len: usize,
    pub values: Vec<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TextureSlotBindingPreview {
    pub slot: u32,
    pub texture: WideHashPreview,
}

#[derive(Debug, Clone, Copy)]
pub struct TechniqueTextureBinding {
    pub stage: &'static str,
    pub slot: u32,
    pub tag: TagHash,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TechniqueMaterialConstants {
    pub color_tint: Option<[f32; 4]>,
    pub mask_palette: Option<[[f32; 4]; 2]>,
    pub emissive_strength: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct TfxBytecodePreview {
    pub total_bytes: usize,
    pub ops: Vec<TfxBytecodeOpPreview>,
    pub bindings: Vec<TfxBindingPreview>,
    pub expressions: Vec<TfxExpressionPreview>,
    pub externs: Vec<TfxExternRefPreview>,
    pub constant_refs: Vec<TfxConstantRefPreview>,
    pub decoded_ops: usize,
    pub unknown_ops: usize,
    pub status: TfxDecodeStatus,
    pub undecoded_offset: Option<usize>,
    pub undecoded_bytes: Vec<u8>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TfxDecodeStatus {
    #[default]
    Complete,
    Partial,
    StoppedAtUnknown,
    Invalid,
}

#[derive(Debug, Clone)]
pub struct TfxBindingPreview {
    pub kind: &'static str,
    pub stage: &'static str,
    pub slot: u8,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct TfxExpressionPreview {
    pub op_offset: usize,
    pub target: String,
    pub expression: String,
    pub value: Option<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TfxExternRefPreview {
    pub op_offset: usize,
    pub scope_id: u8,
    pub value_type: &'static str,
    pub scope: String,
    pub byte_offset: usize,
    pub hint: &'static str,
}

#[derive(Debug, Clone)]
pub struct TfxConstantRefPreview {
    pub op_offset: usize,
    pub op_name: &'static str,
    pub start: usize,
    pub count: usize,
    pub values: Vec<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TfxBytecodeOpPreview {
    pub offset: usize,
    pub opcode: u8,
    pub name: &'static str,
    pub detail: String,
    pub extern_scope_id: Option<u8>,
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
        if is_technique_entry(entry) {
            Some(Self {
                class_name,
                kind: MaterialPreviewKind::Technique(parse_technique(data)?),
            })
        } else {
            None
        }
    }
}

pub fn is_technique_entry(entry: &UEntryHeader) -> bool {
    get_class_by_id(entry.reference).is_some_and(|class| class.name.as_ref() == "s_technique")
}

pub fn is_sticker_proxy_technique(tag: TagHash) -> bool {
    let Some(entry) = package_manager().get_entry(tag) else {
        return false;
    };
    let Ok(data) = package_manager().read_tag(tag) else {
        return false;
    };
    let Some(preview) = MaterialTagPreview::load(&entry, &data) else {
        return false;
    };
    let MaterialPreviewKind::Technique(technique) = preview.kind;
    let no_pixel_textures = texture_bindings_for_technique(&entry, &data)
        .into_iter()
        .all(|binding| binding.stage != "PS");
    no_pixel_textures
        && technique
            .stages
            .iter()
            .filter(|stage| stage.stage == "PS")
            .any(|stage| {
                stage.textures.is_empty()
                    && stage.constants.is_empty()
                    && stage.inline_constants.len() <= 4
                    && stage.inline_constants.first().is_some_and(|color| {
                        color
                            .iter()
                            .zip([1.0_f32; 4])
                            .all(|(value, expected)| (value - expected).abs() < 0.0001)
                    })
            })
}

pub fn render_state_for_technique(tag: TagHash) -> TechniqueRenderState {
    package_manager()
        .get_entry(tag)
        .filter(is_technique_entry)
        .and_then(|_entry| package_manager().read_tag(tag).ok())
        .and_then(|data| {
            data.get(0x30..0x34)
                .map(|bytes| read_u32(bytes, package_manager().version.endian()))
        })
        .map(TechniqueRenderState::from_raw)
        .unwrap_or_default()
}

pub fn texture_bindings_for_technique(
    entry: &UEntryHeader,
    data: &[u8],
) -> Vec<TechniqueTextureBinding> {
    if !is_technique_entry(entry) {
        return vec![];
    }

    parse_technique(data)
        .map(|technique| {
            technique
                .stages
                .into_iter()
                .flat_map(|stage| {
                    stage.textures.into_iter().filter_map(move |binding| {
                        Some(TechniqueTextureBinding {
                            stage: stage.stage,
                            slot: binding.slot,
                            tag: binding
                                .texture
                                .resolved
                                .and_then(texture_header_tag)
                                .or_else(|| texture_header_tag(binding.texture.raw32))?,
                        })
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn primary_sampler_for_technique(entry: &UEntryHeader, data: &[u8]) -> Option<TagHash> {
    if !is_technique_entry(entry) {
        return None;
    }

    let technique = parse_technique(data)?;
    let stage = technique
        .stages
        .into_iter()
        .find(|stage| stage.stage == "PS")?;
    let sampler_index = stage
        .bytecode
        .bindings
        .iter()
        .filter(|binding| binding.kind == "sampler" && binding.stage == "PS")
        .min_by_key(|binding| binding.slot)
        .and_then(|binding| {
            binding
                .source
                .strip_prefix("sampler[")?
                .strip_suffix(']')?
                .parse::<usize>()
                .ok()
        })
        .or_else(|| (!stage.indexed_resources.is_empty()).then_some(0))?;
    let sampler = stage.indexed_resources.get(sampler_index)?;
    sampler
        .resolved
        .and_then(sampler_header_tag)
        .or_else(|| sampler_header_tag(sampler.raw32))
}

pub fn sampler_for_technique_slot(
    entry: &UEntryHeader,
    data: &[u8],
    shader_stage: &str,
    slot: u8,
) -> Option<TagHash> {
    if !is_technique_entry(entry) {
        return None;
    }

    let technique = parse_technique(data)?;
    let stage = technique
        .stages
        .into_iter()
        .find(|stage| stage.stage == shader_stage)?;
    let sampler_index = stage
        .bytecode
        .bindings
        .iter()
        .find(|binding| {
            binding.kind == "sampler" && binding.stage == shader_stage && binding.slot == slot
        })?
        .source
        .strip_prefix("sampler[")?
        .strip_suffix(']')?
        .parse::<usize>()
        .ok()?;
    let sampler = stage.indexed_resources.get(sampler_index)?;
    sampler
        .resolved
        .and_then(sampler_header_tag)
        .or_else(|| sampler_header_tag(sampler.raw32))
}

pub fn material_constants_for_technique(
    entry: &UEntryHeader,
    data: &[u8],
) -> TechniqueMaterialConstants {
    if !is_technique_entry(entry) {
        return TechniqueMaterialConstants::default();
    }

    let bindings = texture_bindings_for_technique(entry, data);
    let primary_is_bc4 = bindings
        .iter()
        .find(|binding| binding.stage == "PS" && binding.slot == 0)
        .is_some_and(|binding| texture_is_bc4(binding.tag));
    let pixel_binding_count = bindings
        .iter()
        .filter(|binding| binding.stage == "PS")
        .count();
    let pixel_stage = parse_technique(data).and_then(|technique| {
        technique
            .stages
            .into_iter()
            .find(|stage| stage.stage == "PS")
    });
    let color_tint = primary_is_bc4
        .then(|| {
            pixel_stage
                .as_ref()
                .and_then(|stage| select_confident_mask_tint(&stage.inline_constants))
        })
        .flatten();
    let mask_palette = (!primary_is_bc4 && pixel_binding_count == 5)
        .then(|| {
            pixel_stage
                .as_ref()
                .and_then(|stage| compact_mask_palette(&stage.inline_constants))
        })
        .flatten();

    TechniqueMaterialConstants {
        color_tint,
        mask_palette,
        emissive_strength: None,
    }
}

fn compact_mask_palette(values: &[[f32; 4]]) -> Option<[[f32; 4]; 2]> {
    let base = *values.get(53)?;
    let delta = *values.get(54)?;
    let valid = [base, delta].into_iter().all(|value| {
        value
            .iter()
            .all(|component| component.is_finite() && (-1.0..=1.0).contains(component))
    });
    let chroma = base[..3]
        .iter()
        .chain(&delta[..3])
        .any(|component| component.abs() > 0.02);
    (valid && chroma).then_some([base, delta])
}

fn texture_is_bc4(tag: TagHash) -> bool {
    Texture::validated_descriptor_d2(tag)
        .map(|desc| format!("{:?}", desc.format).contains("Bc4"))
        .unwrap_or(false)
}

fn select_confident_mask_tint(values: &[[f32; 4]]) -> Option<[f32; 4]> {
    let mut candidates = values
        .iter()
        .copied()
        .filter(|value| {
            value[..3]
                .iter()
                .all(|component| component.is_finite() && (0.0..=1.0).contains(component))
                && (value[3].abs() < 0.0001 || (value[3] - 1.0).abs() < 0.0001)
                && !is_luminance_weights(*value)
        })
        .map(|value| {
            let min = value[..3].iter().copied().fold(f32::INFINITY, f32::min);
            let max = value[..3].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min, max - min, value)
        })
        .filter(|(min, saturation, _value)| *min > 0.01 && *saturation > 0.15)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    let (_best_min, best_saturation, mut best) = *candidates.first()?;
    let runner_up = candidates
        .get(1)
        .map(|candidate| candidate.1)
        .unwrap_or(0.0);
    if best_saturation < 0.25 || best_saturation - runner_up < 0.1 {
        return None;
    }
    best[3] = 1.0;
    Some(best)
}

fn is_luminance_weights(value: [f32; 4]) -> bool {
    [0.3, 0.59, 0.11]
        .into_iter()
        .zip(value)
        .take(3)
        .all(|(expected, actual)| (expected - actual).abs() < 0.03)
}

fn texture_header_tag(tag: TagHash) -> Option<TagHash> {
    let entry = package_manager().get_entry(tag)?;
    let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
    (tag_type.is_texture() && tag_type.is_header()).then_some(tag)
}

fn sampler_header_tag(tag: TagHash) -> Option<TagHash> {
    let entry = package_manager().get_entry(tag)?;
    (entry.file_type == 34 && entry.file_subtype == 1).then_some(tag)
}

fn parse_technique(data: &[u8]) -> Option<TechniquePreview> {
    let endian = package_manager().version.endian();
    let bind_mode = read_u32(data.get(0x8..0xc)?, endian);
    let used_scopes = read_u64(data.get(0x20..0x28)?, endian);
    let compatible_scopes = read_u64(data.get(0x28..0x30)?, endian);
    let state_selection = read_u32(data.get(0x30..0x34)?, endian);
    let (stage_base, stage_stride, marathon_tfx) = match package_manager().version {
        GameVersion::Marathon(_) => (0x58usize, 0x88usize, true),
        GameVersion::Destiny(_) => (0x70usize, 0x90usize, false),
    };
    let stages = [
        ("VS", 0usize),
        ("GS", 3usize),
        ("PS", 4usize),
        ("CS", 5usize),
    ]
    .into_iter()
    .filter_map(|(name, index)| {
        parse_technique_stage(
            data,
            name,
            stage_base + index * stage_stride,
            endian,
            marathon_tfx,
        )
    })
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
    marathon_tfx: bool,
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
    let constants = read_array(data, constants_offset + 0x10, 0x10, endian)
        .map(|constants| parse_vec4_array(constants, endian))
        .unwrap_or_default();
    let indexed_resources = read_array(data, constants_offset + 0x20, 0x10, endian)
        .map(|samplers| {
            samplers
                .chunks_exact(0x10)
                .filter_map(|sampler| read_wide_hash(sampler, endian))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let inline_constants = read_array(data, constants_offset + 0x30, 0x10, endian)
        .map(|constants| parse_vec4_array(constants, endian))
        .unwrap_or_default();
    let (constant_buffer_slot, constant_buffer) =
        constant_buffer_binding(data, offset, endian, marathon_tfx);
    let constant_buffer_preview = constant_buffer.and_then(load_constant_buffer_preview);
    let bytecode = read_array(data, constants_offset, 1, endian)
        .map(|bytecode| {
            parse_tfx_bytecode_with_constants_dialect(bytecode, &constants, marathon_tfx)
        })
        .unwrap_or_default();
    let bytecode_len = bytecode.total_bytes;

    if shader.is_none()
        && textures.is_empty()
        && indexed_resources.is_empty()
        && constants.is_empty()
        && inline_constants.is_empty()
        && bytecode_len == 0
        && constant_buffer_slot.is_none()
    {
        return None;
    }

    Some(TechniqueStagePreview {
        stage,
        shader: shader.is_some().then_some(shader),
        textures,
        constants,
        indexed_resources,
        inline_constants,
        bytecode_len,
        constant_buffer_slot,
        constant_buffer,
        constant_buffer_preview,
        bytecode,
    })
}

fn constant_buffer_binding(
    data: &[u8],
    stage_offset: usize,
    endian: Endian,
    marathon: bool,
) -> (Option<i32>, Option<TagHash>) {
    // Marathon's stage has an eight-byte reserved block after inline data;
    // Destiny has sixteen. The C827 payload proves slot/tag at +0x68/+0x6c.
    let binding_offset = stage_offset + if marathon { 0x68 } else { 0x70 };
    let slot = data
        .get(binding_offset..binding_offset + 4)
        .map(|bytes| read_i32(bytes, endian))
        .filter(|slot| *slot >= 0);
    let buffer = data
        .get(binding_offset + 4..binding_offset + 8)
        .map(|bytes| read_tag(bytes, endian))
        .filter(|tag| tag.is_some());
    (slot, buffer)
}

fn load_constant_buffer_preview(tag: TagHash) -> Option<ConstantBufferPreview> {
    let entry = package_manager().get_entry(tag)?;
    let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
    let header = package_manager().read_tag(tag).ok()?;
    let data_tag = if matches!(tag_type, TagType::ConstantBuffer { is_header: true }) {
        TagHash(entry.reference)
    } else {
        tag
    };
    let data = if data_tag == tag {
        header.clone()
    } else {
        package_manager().read_tag(data_tag).ok()?
    };
    let endian = package_manager().version.endian();
    let values = parse_constant_buffer_values(&data, endian)?;

    Some(ConstantBufferPreview {
        header_tag: tag,
        data_tag,
        header_len: header.len(),
        data_len: data.len(),
        values,
    })
}

fn parse_constant_buffer_values(data: &[u8], endian: Endian) -> Option<Vec<[f32; 4]>> {
    if !data.len().is_multiple_of(4) {
        return None;
    }
    // DXIL buffers may end partway through a vec4 (e.g. 136 bytes).
    // Preserve the final scalars; only absent components receive zero padding.
    Some(
        data.chunks(16)
            .map(|chunk| {
                std::array::from_fn(|component| {
                    let offset = component * 4;
                    chunk
                        .get(offset..offset + 4)
                        .map(|bytes| read_f32(bytes, endian))
                        .unwrap_or(0.0)
                })
            })
            .collect(),
    )
}

fn parse_vec4_array(data: &[u8], endian: Endian) -> Vec<[f32; 4]> {
    data.chunks_exact(0x10)
        .map(|chunk| {
            [
                read_f32(&chunk[0x0..0x4], endian),
                read_f32(&chunk[0x4..0x8], endian),
                read_f32(&chunk[0x8..0xc], endian),
                read_f32(&chunk[0xc..0x10], endian),
            ]
        })
        .collect()
}

fn parse_tfx_bytecode_with_constants_dialect(
    data: &[u8],
    constants: &[[f32; 4]],
    marathon: bool,
) -> TfxBytecodePreview {
    const MAX_UI_OPS: usize = 160;

    let mut cursor = 0usize;
    let mut all_ops = Vec::with_capacity(data.len());
    let mut decoded_ops = 0usize;
    let mut unknown_ops = 0usize;
    let mut undecoded_offset = None;
    let mut undecoded_bytes = vec![];
    let extern_name = if marathon {
        tfx_marathon_extern_name as fn(u8) -> &'static str
    } else {
        tfx_extern_name
    };

    while cursor < data.len() {
        let offset = cursor;
        let opcode = data[cursor];
        cursor += 1;

        let parsed = if marathon {
            parse_marathon_tfx_bytecode_op(data, &mut cursor, offset, opcode, extern_name)
        } else {
            parse_tfx_bytecode_op(data, &mut cursor, offset, opcode, extern_name)
        };
        let Some(mut op) = parsed else {
            unknown_ops += 1;
            undecoded_offset = Some(offset);
            undecoded_bytes.extend_from_slice(&data[offset..]);
            all_ops.push(TfxBytecodeOpPreview {
                offset,
                opcode,
                name: "unknown",
                detail: String::new(),
                extern_scope_id: None,
            });
            break;
        };
        if op.name.contains("unknown") {
            unknown_ops += 1;
        }
        op.opcode = opcode;
        if op.name.starts_with("push_extern_") {
            op.extern_scope_id = data.get(offset + 1).copied();
        }

        decoded_ops += 1;
        all_ops.push(op);
    }

    let (bindings, expressions) = interpret_tfx_stack(&all_ops, constants);
    let externs = summarize_tfx_externs(&all_ops);
    let constant_refs = summarize_tfx_constant_refs(&all_ops, constants);
    let truncated = all_ops.len() > MAX_UI_OPS;
    let status = if undecoded_offset.is_some() {
        TfxDecodeStatus::StoppedAtUnknown
    } else if unknown_ops > 0 {
        TfxDecodeStatus::Partial
    } else {
        TfxDecodeStatus::Complete
    };
    TfxBytecodePreview {
        total_bytes: data.len(),
        bindings,
        expressions,
        externs,
        constant_refs,
        decoded_ops,
        unknown_ops,
        status,
        undecoded_offset,
        undecoded_bytes,
        truncated,
        // Keep the complete program for render-time interpretation. The tag
        // inspector applies its own display cap; truncating here silently
        // discarded material outputs after the first 160 instructions.
        ops: all_ops,
    }
}

fn parse_marathon_tfx_bytecode_op(
    data: &[u8],
    cursor: &mut usize,
    offset: usize,
    opcode: u8,
    extern_name: fn(u8) -> &'static str,
) -> Option<TfxBytecodeOpPreview> {
    let read_u8 = |cursor: &mut usize| {
        let value = *data.get(*cursor)?;
        *cursor += 1;
        Some(value)
    };
    let parsed = match opcode {
        0x3b => Some(("compare_less_than", String::new())),
        0x3c => Some(("compare_less_equal", String::new())),
        0x3d => Some(("compare_greater_than", String::new())),
        0x3e => Some(("compare_greater_equal", String::new())),
        0x3f => Some(("compare_equal", String::new())),
        0x40 => Some(("compare_not_equal", String::new())),
        0x41 => Some(("compare_not_zero_ternary", String::new())),
        0x51 => Some(("push_from_output", format!("element={}", read_u8(cursor)?))),
        0x57 => Some(("push_temp", format!("slot={}", read_u8(cursor)?))),
        0x58 => Some(("pop_temp", format!("slot={}", read_u8(cursor)?))),
        0x5b => {
            let value = read_u8(cursor)?;
            Some((
                "set_shader_resource",
                format!("{} slot={}", tfx_shader_stage_name(value), value & 0x1f),
            ))
        }
        // Tiger's second indexed global push; it reads the same global table.
        0x64 => Some(("push_global_channel", format!("index={}", read_u8(cursor)?))),
        0x67..=0x69 => {
            let index = read_u8(cursor)?;
            let fields = read_u8(cursor)?;
            Some((
                match opcode {
                    0x67 => "push_tex_dimensions",
                    0x68 => "push_tex_tiling_params",
                    0x69 => "push_tex_tile_layer_count",
                    _ => unreachable!(),
                },
                format!("index={index} fields=0x{fields:02X}"),
            ))
        }
        0x6d => Some(("marathon_unknown_no_args", "opcode=0x6D".to_string())),
        _ => None,
    };
    if let Some((name, detail)) = parsed {
        return Some(TfxBytecodeOpPreview {
            offset,
            opcode,
            name,
            detail,
            extern_scope_id: None,
        });
    }

    if matches!(opcode, 0x10..=0x12 | 0x24..=0x27 | 0x36..=0x3a) {
        return Some(TfxBytecodeOpPreview {
            offset,
            opcode,
            name: "marathon_unknown_no_args",
            detail: format!("opcode=0x{opcode:02X}"),
            extern_scope_id: None,
        });
    }

    let legacy_opcode = marathon_tfx_legacy_opcode(opcode)?;
    parse_tfx_bytecode_op(data, cursor, offset, legacy_opcode, extern_name).map(|mut op| {
        op.opcode = opcode;
        op
    })
}

fn marathon_tfx_legacy_opcode(opcode: u8) -> Option<u8> {
    match opcode {
        0x01..=0x0f => Some(opcode),
        // Marathon keeps three operations at 0x10..0x12 that the older table
        // lacks, so lerp..vector_rotations_sin_cos sit three higher.
        0x13..=0x23 => Some(opcode - 3),
        0x28..=0x35 => Some(opcode - 7),
        0x42..=0x49 => Some(opcode - 0x0e),
        0x4a => Some(0x3c),
        0x4b => Some(0x3d),
        0x4c => Some(0x3e),
        0x4d => Some(0x3f),
        0x4e => Some(0x40),
        0x4f => Some(0x41),
        0x50 => Some(0x42),
        0x52 => Some(0x43),
        0x53 => Some(0x44),
        // Packaged Marathon View/RigidModel scopes store four matrix rows
        // with 0x55. 0x54/0x56 have no verified meaning; do not reuse legacy mappings.
        0x55 => Some(0x45),
        0x59 => Some(0x48),
        0x5d => Some(0x4a),
        0x5e => Some(0x4b),
        0x61 => Some(0x4d),
        0x62 => Some(0x4e),
        0x63 => Some(0x4f),
        _ => None,
    }
}

fn summarize_tfx_constant_refs(
    ops: &[TfxBytecodeOpPreview],
    constants: &[[f32; 4]],
) -> Vec<TfxConstantRefPreview> {
    ops.iter()
        .filter_map(|op| {
            let (start, count) = tfx_constant_range(op)?;
            Some(TfxConstantRefPreview {
                op_offset: op.offset,
                op_name: op.name,
                start,
                count,
                values: constants
                    .get(start..start.saturating_add(count).min(constants.len()))
                    .unwrap_or_default()
                    .to_vec(),
            })
        })
        .collect()
}

fn tfx_constant_range(op: &TfxBytecodeOpPreview) -> Option<(usize, usize)> {
    let start = if let Some(index) = op.detail.strip_prefix("constant=") {
        index.parse().ok()?
    } else if let Some(start) = op.detail.strip_prefix("start=") {
        start.parse().ok()?
    } else {
        return None;
    };

    let count = match op.name {
        "push_const_vec4" => 1,
        "lerp_constant" | "lerp_constant_saturated" => 2,
        "spline4_const" => 5,
        "spline8_const" | "spline8_chain_const" => 10,
        "gradient4_const" => 6,
        "unk3b" => 11,
        _ => return None,
    };

    Some((start, count))
}

fn summarize_tfx_externs(ops: &[TfxBytecodeOpPreview]) -> Vec<TfxExternRefPreview> {
    ops.iter()
        .filter_map(|op| {
            let value_type = op.name.strip_prefix("push_extern_")?;
            let (scope, offset_hex) = op.detail.split_once("+0x")?;
            let byte_offset = usize::from_str_radix(offset_hex, 16).ok()?;
            Some(TfxExternRefPreview {
                op_offset: op.offset,
                scope_id: op.extern_scope_id?,
                value_type: match value_type {
                    "float" => "float",
                    "vec4" => "vec4",
                    "mat4" => "mat4",
                    "texture" => "texture",
                    "u32" => "u32",
                    "uav" => "uav",
                    _ => "value",
                },
                scope: scope.to_string(),
                byte_offset,
                hint: tfx_extern_hint(scope, byte_offset),
            })
        })
        .collect()
}

fn tfx_extern_hint(scope: &str, byte_offset: usize) -> &'static str {
    match scope {
        "Frame" => "frame-global runtime constant",
        "View" => "camera/view runtime constant",
        "RigidModel" | "SimpleGeometry" | "EditorMesh" => "model or instance runtime constant",
        "TextureSet" => "material texture-set runtime binding",
        "Decal" => "decal runtime constant",
        "Water" => "water runtime constant",
        "Generic" if byte_offset < 0x40 => "material parameter block",
        "Generic" => "generic runtime constant",
        _ => "runtime extern",
    }
}

pub(crate) fn tfx_has_marathon_decal_abi(program: &TfxBytecodePreview) -> bool {
    let has_normals_read = program.externs.iter().any(|external| {
        external.scope_id == 45 && external.value_type == "texture" && external.byte_offset == 0x08
    });
    let has_resolution_offset = program.externs.iter().any(|external| {
        external.scope_id == 45 && external.value_type == "vec4" && external.byte_offset == 0x30
    });
    has_normals_read && has_resolution_offset
}

fn interpret_tfx_stack(
    ops: &[TfxBytecodeOpPreview],
    constants: &[[f32; 4]],
) -> (Vec<TfxBindingPreview>, Vec<TfxExpressionPreview>) {
    interpret_tfx_stack_with_object_channels(ops, constants, &std::collections::HashMap::new())
}

pub(crate) fn interpret_tfx_stack_with_object_channels(
    ops: &[TfxBytecodeOpPreview],
    constants: &[[f32; 4]],
    object_channels: &std::collections::HashMap<u32, [f32; 4]>,
) -> (Vec<TfxBindingPreview>, Vec<TfxExpressionPreview>) {
    interpret_tfx_stack_with_runtime_values(
        ops,
        constants,
        object_channels,
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        &std::collections::BTreeMap::new(),
    )
}

pub(crate) fn interpret_tfx_stack_with_runtime_values(
    ops: &[TfxBytecodeOpPreview],
    constants: &[[f32; 4]],
    object_channels: &std::collections::HashMap<u32, [f32; 4]>,
    extern_values: &std::collections::HashMap<(String, u32), [f32; 4]>,
    global_channels: &std::collections::HashMap<u32, [f32; 4]>,
    texture_metadata: &std::collections::BTreeMap<u8, crate::texture::TextureExpressionMetadata>,
) -> (Vec<TfxBindingPreview>, Vec<TfxExpressionPreview>) {
    let mut stack = Vec::<TfxStackValue>::new();
    let mut temps = std::collections::BTreeMap::<u8, TfxStackValue>::new();
    let mut outputs = std::collections::BTreeMap::<u8, TfxStackValue>::new();
    let mut bindings = Vec::new();
    let mut expressions = Vec::new();

    for op in ops {
        match op.name {
            "push_const_vec4"
            | "push_sampler"
            | "push_extern_float"
            | "push_extern_vec4"
            | "push_extern_texture"
            | "push_extern_u32"
            | "push_extern_uav"
            | "push_object_channel"
            | "push_global_channel"
            | "push_tex_dimensions" => stack.push(format_tfx_runtime_value(
                op,
                constants,
                object_channels,
                extern_values,
                global_channels,
            )),
            "push_tex_tiling_params" | "push_tex_tile_layer_count" => {
                let value = op.detail.split_once(" fields=0x").and_then(|(index, fields)| {
                    let index = index.strip_prefix("index=")?.parse::<u8>().ok()?;
                    let fields = u8::from_str_radix(fields,16).ok()?;
                    texture_metadata.get(&index)?.evaluate(op.name,fields)
                });
                let mut source = format_tfx_value(op,constants);
                source.value = value;
                stack.push(source);
            }
            "push_extern_mat4" => {
                let source = op.detail.split_once("+0x").and_then(|(scope, offset)| {
                    Some((scope, u32::from_str_radix(offset, 16).ok()?))
                });
                // Tiger matrices push four column vectors in source order.
                for row in 0..4 {
                    stack.push(TfxStackValue {
                        expression: format!("extern_mat4({})[{row}]", op.detail),
                        value: source.and_then(|(scope, offset)| {
                            extern_values
                                .get(&(scope.to_string(), offset + row * 16))
                                .copied()
                        }),
                    });
                }
            }
            "push_from_output" => {
                let element = op
                    .detail
                    .strip_prefix("element=")
                    .and_then(|element| element.parse::<u8>().ok());
                if let Some(element) = element
                    && let Some(value) = outputs.get(&element)
                {
                    stack.push(value.clone());
                } else {
                    stack.push(format_tfx_value(op, constants));
                }
            }
            "lerp_constant" | "lerp_constant_saturated" => {
                collapse_lerp_constant(&mut stack, op, constants);
            }
            "spline4_const"
            | "spline8_const"
            | "spline8_chain_const"
            | "gradient4_const"
            | "unk3b" => {
                collapse_constant_range_op(&mut stack, op, constants);
            }
            "push_temp" => {
                let slot = op
                    .detail
                    .strip_prefix("slot=")
                    .and_then(|slot| slot.parse::<u8>().ok());
                if let Some(slot) = slot
                    && let Some(value) = temps.get(&slot)
                {
                    stack.push(value.clone());
                } else {
                    stack.push(format_tfx_value(op, constants));
                }
            }
            "set_shader_texture"
            | "set_shader_sampler"
            | "set_shader_uav"
            | "set_shader_resource" => {
                if let Some((stage, slot)) = parse_stage_slot(&op.detail) {
                    bindings.push(TfxBindingPreview {
                        kind: match op.name {
                            "set_shader_texture" => "texture",
                            "set_shader_sampler" => "sampler",
                            "set_shader_uav" => "uav",
                            "set_shader_resource" => "resource",
                            _ => "binding",
                        },
                        stage,
                        slot,
                        source: stack
                            .pop()
                            .map(|value| value.expression)
                            .unwrap_or_else(|| "<empty stack>".to_string()),
                    });
                }
            }
            "pop_output" | "pop_output_mat4" => {
                let first = op
                    .detail
                    .strip_prefix("element=")
                    .and_then(|element| element.parse::<u8>().ok());
                let count = if op.name == "pop_output_mat4" { 4 } else { 1 };
                for row in (0..count).rev() {
                    let value = stack.pop().unwrap_or_else(|| TfxStackValue {
                        expression: "<empty stack>".to_string(),
                        value: None,
                    });
                    let element = first.and_then(|first| first.checked_add(row));
                    if let Some(element) = element {
                        outputs.insert(element, value.clone());
                    }
                    expressions.push(TfxExpressionPreview {
                        op_offset: op.offset,
                        target: element
                            .map(|element| format!("output[{element}]"))
                            .unwrap_or_else(|| "output[?]".to_string()),
                        expression: value.expression,
                        value: value.value,
                    });
                }
            }
            "pop_temp" => {
                let value = stack.pop().unwrap_or_else(|| TfxStackValue {
                    expression: "<empty stack>".to_string(),
                    value: None,
                });
                let target = op
                    .detail
                    .strip_prefix("slot=")
                    .map(|slot| format!("temp[{slot}]"))
                    .unwrap_or_else(|| "temp[?]".to_string());
                if let Some(slot) = op
                    .detail
                    .strip_prefix("slot=")
                    .and_then(|slot| slot.parse::<u8>().ok())
                {
                    temps.insert(slot, value.clone());
                }
                expressions.push(TfxExpressionPreview {
                    op_offset: op.offset,
                    target,
                    expression: value.expression,
                    value: value.value,
                });
            }
            "add" | "subtract" | "multiply" | "divide" | "min" | "max" | "less_than" | "dot" => {
                collapse_stack(&mut stack, op.name, 2);
            }
            "compare_less_than"
            | "compare_less_equal"
            | "compare_greater_than"
            | "compare_greater_equal"
            | "compare_equal"
            | "compare_not_equal" => collapse_stack(&mut stack, op.name, 2),
            "compare_not_zero_ternary" => collapse_stack(&mut stack, op.name, 3),
            "cubic" => collapse_stack(&mut stack, op.name, 2),
            "lerp" | "lerp_saturated" | "multiply_add" | "clamp" => {
                collapse_stack(&mut stack, op.name, 3)
            }
            "transform_vec4" => collapse_stack(&mut stack, op.name, 5),
            "merge_1_3" | "merge_2_2" | "merge_3_1" => {
                collapse_stack(&mut stack, op.name, 2);
            }
            "permute" => collapse_permute(&mut stack, op),
            "permute_extend_x"
            | "is_zero"
            | "abs"
            | "signum"
            | "floor"
            | "ceil"
            | "round"
            | "frac"
            | "negate"
            | "saturate"
            | "vector_rotations_sin"
            | "vector_rotations_cos"
            | "vector_rotations_sin_cos"
            | "triangle"
            | "jitter"
            | "wander"
            | "rand"
            | "rand_smooth" => collapse_stack(&mut stack, op.name, 1),
            "normalize3" => collapse_stack(&mut stack, op.name, 1),
            _ => {}
        }
    }

    (bindings, expressions)
}

#[derive(Debug, Clone)]
struct TfxStackValue {
    expression: String,
    value: Option<[f32; 4]>,
}

fn collapse_stack(stack: &mut Vec<TfxStackValue>, name: &str, inputs: usize) {
    if stack.len() < inputs {
        stack.push(TfxStackValue {
            expression: format!("{name}(?)"),
            value: None,
        });
        return;
    }

    let args = stack.split_off(stack.len() - inputs);
    let expression = format!(
        "{name}({})",
        args.iter()
            .map(|value| value.expression.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let value = evaluate_stack_op(name, &args);
    stack.push(TfxStackValue { expression, value });
}

fn collapse_permute(stack: &mut Vec<TfxStackValue>, op: &TfxBytecodeOpPreview) {
    let input = stack.pop().unwrap_or_else(|| TfxStackValue {
        expression: "?".to_string(),
        value: None,
    });
    let fields = op
        .detail
        .strip_prefix("fields=0x")
        .and_then(|fields| u8::from_str_radix(fields, 16).ok());
    let Some(fields) = fields else {
        stack.push(TfxStackValue {
            expression: format!("permute({})", input.expression),
            value: None,
        });
        return;
    };
    // Selectors are read high-to-low: the top two bits choose output x.
    let lanes = [
        ((fields >> 6) & 0b0000_0011) as usize,
        ((fields >> 4) & 0b0000_0011) as usize,
        ((fields >> 2) & 0b0000_0011) as usize,
        (fields & 0b0000_0011) as usize,
    ];
    let suffix = lanes
        .iter()
        .map(|lane| ["x", "y", "z", "w"][*lane])
        .collect::<String>();
    let value = input.value.map(|value| {
        [
            value[lanes[0]],
            value[lanes[1]],
            value[lanes[2]],
            value[lanes[3]],
        ]
    });
    stack.push(TfxStackValue {
        expression: format!("{}.{}", input.expression, suffix),
        value,
    });
}

fn collapse_lerp_constant(
    stack: &mut Vec<TfxStackValue>,
    op: &TfxBytecodeOpPreview,
    constants: &[[f32; 4]],
) {
    let input = stack.pop().unwrap_or_else(|| TfxStackValue {
        expression: "?".to_string(),
        value: None,
    });
    let Some((start, _count)) = tfx_constant_range(op) else {
        stack.push(TfxStackValue {
            expression: format!("{}({})", op.name, input.expression),
            value: None,
        });
        return;
    };
    let value = input.value.and_then(|input| {
        let a = constants.get(start).copied()?;
        let b = constants.get(start + 1).copied()?;
        Some(std::array::from_fn(|lane| {
            let t = if op.name == "lerp_constant_saturated" {
                input[lane].clamp(0.0, 1.0)
            } else {
                input[lane]
            };
            a[lane] + (b[lane] - a[lane]) * t
        }))
    });
    stack.push(TfxStackValue {
        expression: format!(
            "{}(constant[{}..{}], {})",
            op.name,
            start,
            start + 2,
            input.expression
        ),
        value,
    });
}

fn collapse_constant_range_op(
    stack: &mut Vec<TfxStackValue>,
    op: &TfxBytecodeOpPreview,
    constants: &[[f32; 4]],
) {
    let input = stack.pop().unwrap_or_else(|| TfxStackValue {
        expression: "?".to_string(),
        value: None,
    });
    let recursion = (op.name == "spline8_chain_const").then(|| {
        stack.pop().unwrap_or_else(|| TfxStackValue {
            expression: "?".to_string(),
            value: None,
        })
    });
    let Some((start, count)) = tfx_constant_range(op) else {
        stack.push(TfxStackValue {
            expression: format!("{}({})", op.name, input.expression),
            value: None,
        });
        return;
    };
    let available = constants
        .get(start..start.saturating_add(count).min(constants.len()))
        .map(|values| values.len())
        .unwrap_or_default();
    let values = constants.get(start..start.saturating_add(count));
    let value = input.value.and_then(|input| match (op.name, values) {
        ("spline4_const", Some(values)) => eval_spline4(input, values),
        ("spline8_const", Some(values)) => eval_spline8(input, values),
        ("spline8_chain_const", Some(values)) => {
            eval_spline8_chain(input, recursion.as_ref()?.value?, values)
        }
        ("gradient4_const", Some(values)) => eval_gradient4(input, values),
        ("unk3b", Some(values)) => eval_piecewise_vector8(input, values),
        _ => None,
    });
    stack.push(TfxStackValue {
        expression: format!(
            "{}(constant[{}..{}; {} available], {}{})",
            op.name,
            start,
            start + count,
            available,
            recursion
                .as_ref()
                .map(|value| format!("{}, ", value.expression))
                .unwrap_or_default(),
            input.expression,
        ),
        value,
    });
}

/// Exact scalar form of Alkahest's bytecode_op_unk3b_const SSE helper.
/// Marathon encodes this operation as 0x49. Two groups of four clamped ramps
/// add authored vectors to a base value; preserve the reference reduction order.
fn eval_piecewise_vector8(input: [f32; 4], constants: &[[f32; 4]]) -> Option<[f32; 4]> {
    let constants: &[[f32; 4]; 11] = constants.try_into().ok()?;
    let ramps = |thresholds: [f32; 4], last: f32| {
        std::array::from_fn::<_, 4, _>(|lane| {
            let next = if lane < 3 { thresholds[lane + 1] } else { last };
            let width = next - thresholds[lane];
            let offset = input[lane] - thresholds[lane];
            let value = if width.abs() > 0.0001 {
                offset / width
            } else {
                if offset >= 0.0 { 1.0 } else { 0.0 }
            };
            // SSE max/min select their second operand on a tie or unordered
            // comparison, including NaN and signed zero.
            let value = if value > 0.0 { value } else { 0.0 };
            if value < 1.0 { value } else { 1.0 }
        })
    };
    let first = ramps(constants[9], constants[10][0]);
    let second = ramps(constants[10], 1.0);
    let mut result = constants[0];
    for component in 0..4 {
        let weighted = std::array::from_fn::<_, 4, _>(|lane| {
            constants[component + 1][lane] * first[lane]
                + constants[component + 5][lane] * second[lane]
        });
        let pairs = [weighted[2] + weighted[0], weighted[3] + weighted[1],
                     weighted[0] + weighted[2], weighted[1] + weighted[3]];
        let sum = pairs[(component + 3) % 4] + pairs[component];
        for lane in 0..4 {
            result[lane] = (if lane == component { sum } else { 0.0 }) + result[lane];
        }
    }
    Some(result)
}

fn spline_channel_mask(x: [f32; 4], thresholds: [f32; 4]) -> [f32; 4] {
    let mask = std::array::from_fn::<_, 4, _>(|lane| (x[lane] >= thresholds[lane]) as u8 as f32);
    [
        (mask[0] - mask[1]).abs(),
        (mask[1] - mask[2]).abs(),
        (mask[2] - mask[3]).abs(),
        mask[3],
    ]
}

fn eval_spline_polynomial(
    x: [f32; 4],
    c3: [f32; 4],
    c2: [f32; 4],
    c1: [f32; 4],
    c0: [f32; 4],
) -> [f32; 4] {
    std::array::from_fn(|lane| {
        (c3[lane] * x[lane] + c2[lane]) * x[lane] * x[lane] + c1[lane] * x[lane] + c0[lane]
    })
}

fn masked_spline_sum(values: [f32; 4], mask: [f32; 4]) -> f32 {
    values
        .iter()
        .zip(mask)
        .map(|(value, mask)| value * mask)
        .sum()
}

fn eval_spline4(x: [f32; 4], constants: &[[f32; 4]]) -> Option<[f32; 4]> {
    // Coefficients are stored cubic first, like spline8.
    let [c3, c2, c1, c0, thresholds] = constants.try_into().ok()?;
    let result = masked_spline_sum(
        eval_spline_polynomial(x, c3, c2, c1, c0),
        spline_channel_mask(x, thresholds),
    );
    Some([result; 4])
}

fn eval_spline8(x: [f32; 4], constants: &[[f32; 4]]) -> Option<[f32; 4]> {
    let [c3, c2, c1, c0, d3, d2, d1, d0, ct, dt] = constants.try_into().ok()?;
    let c = masked_spline_sum(
        eval_spline_polynomial(x, c3, c2, c1, c0),
        spline_channel_mask(x, ct),
    );
    let d_mask = spline_channel_mask(x, dt);
    let d = masked_spline_sum(eval_spline_polynomial(x, d3, d2, d1, d0), d_mask);
    Some([if x[0] >= dt[0] { d } else { c }; 4])
}

fn eval_spline8_chain(
    x: [f32; 4],
    recursion: [f32; 4],
    constants: &[[f32; 4]],
) -> Option<[f32; 4]> {
    let [c3, c2, c1, c0, d3, d2, d1, d0, ct, dt] = constants.try_into().ok()?;
    let c = masked_spline_sum(
        eval_spline_polynomial(x, c3, c2, c1, c0),
        spline_channel_mask(x, ct),
    );
    let d = masked_spline_sum(
        eval_spline_polynomial(x, d3, d2, d1, d0),
        spline_channel_mask(x, dt),
    );
    let intermediate = if x[0] >= ct[0] { c } else { recursion[0] };
    Some([if x[0] >= dt[0] { d } else { intermediate }; 4])
}

fn eval_gradient4(x: [f32; 4], constants: &[[f32; 4]]) -> Option<[f32; 4]> {
    let [base, red, green, blue, alpha, thresholds] = constants.try_into().ok()?;
    let percentages = std::array::from_fn::<_, 4, _>(|lane| {
        let end = if lane == 3 { 1.0 } else { thresholds[lane + 1] };
        let interval = end - thresholds[lane];
        if interval.abs() < 1e-19 {
            (x[lane] > thresholds[lane]) as u8 as f32
        } else {
            ((x[lane] - thresholds[lane]) / interval).clamp(0.0, 1.0)
        }
    });
    Some([
        base[0] + red.iter().zip(percentages).map(|(v, p)| v * p).sum::<f32>(),
        base[1]
            + green
                .iter()
                .zip(percentages)
                .map(|(v, p)| v * p)
                .sum::<f32>(),
        base[2]
            + blue
                .iter()
                .zip(percentages)
                .map(|(v, p)| v * p)
                .sum::<f32>(),
        base[3]
            + alpha
                .iter()
                .zip(percentages)
                .map(|(v, p)| v * p)
                .sum::<f32>(),
    ])
}

fn format_tfx_value(op: &TfxBytecodeOpPreview, constants: &[[f32; 4]]) -> TfxStackValue {
    format_tfx_value_with_object_channels(op, constants, &std::collections::HashMap::new())
}

fn format_tfx_value_with_object_channels(
    op: &TfxBytecodeOpPreview,
    constants: &[[f32; 4]],
    object_channels: &std::collections::HashMap<u32, [f32; 4]>,
) -> TfxStackValue {
    let (expression, value) = match op.name {
        "push_const_vec4" => op
            .detail
            .strip_prefix("constant=")
            .and_then(|index| {
                let index = index.parse::<usize>().ok()?;
                Some((format!("constant[{index}]"), constants.get(index).copied()))
            })
            .unwrap_or_else(|| (op.detail.clone(), None)),
        "push_sampler" => op
            .detail
            .strip_prefix("index=")
            .map(|index| (format!("sampler[{index}]"), None))
            .unwrap_or_else(|| (op.detail.clone(), None)),
        "push_temp" => op
            .detail
            .strip_prefix("slot=")
            .map(|slot| (format!("temp[{slot}]"), None))
            .unwrap_or_else(|| (op.detail.clone(), None)),
        "push_object_channel" => {
            let value = op
                .detail
                .strip_prefix("0x")
                .and_then(|hash| u32::from_str_radix(hash, 16).ok())
                .and_then(|hash| object_channels.get(&hash).copied());
            (format!("object_channel({})", op.detail), value)
        }
        _ => {
            if op.detail.is_empty() {
                (op.name.to_string(), None)
            } else {
                (
                    format!("{}({})", op.name.trim_start_matches("push_"), op.detail),
                    None,
                )
            }
        }
    };

    TfxStackValue { expression, value }
}

fn format_tfx_runtime_value(
    op: &TfxBytecodeOpPreview,
    constants: &[[f32; 4]],
    object_channels: &std::collections::HashMap<u32, [f32; 4]>,
    extern_values: &std::collections::HashMap<(String, u32), [f32; 4]>,
    global_channels: &std::collections::HashMap<u32, [f32; 4]>,
) -> TfxStackValue {
    let runtime_value = if op.name.starts_with("push_extern_") {
        op.detail.split_once("+0x").and_then(|(scope, offset)| {
            let offset = u32::from_str_radix(offset, 16).ok()?;
            extern_values.get(&(scope.to_string(), offset)).copied()
        })
    } else if op.name == "push_global_channel" {
        op.detail
            .strip_prefix("index=")
            .and_then(|index| index.parse::<u32>().ok())
            .and_then(|index| global_channels.get(&index).copied())
    } else {
        None
    };
    let mut value = format_tfx_value_with_object_channels(op, constants, object_channels);
    if runtime_value.is_some() {
        value.value = runtime_value;
    }
    value
}

fn evaluate_stack_op(name: &str, args: &[TfxStackValue]) -> Option<[f32; 4]> {
    let values = args
        .iter()
        .map(|arg| arg.value)
        .collect::<Option<Vec<_>>>()?;

    match (name, values.as_slice()) {
        ("add", [a, b]) | ("add2", [a, b]) => Some(vec4_zip(*a, *b, |a, b| a + b)),
        ("subtract", [a, b]) => Some(vec4_zip(*a, *b, |a, b| a - b)),
        ("multiply", [a, b]) | ("multiply2", [a, b]) => Some(vec4_zip(*a, *b, |a, b| a * b)),
        ("divide", [a, b]) => Some(vec4_zip(*a, *b, |a, b| {
            if b.abs() > 1e-19 {
                a / b
            } else {
                a.signum() * f32::INFINITY
            }
        })),
        ("min", [a, b]) => Some(vec4_zip(*a, *b, f32::min)),
        ("max", [a, b]) => Some(vec4_zip(*a, *b, f32::max)),
        ("less_than", [a, b]) => Some(vec4_zip(*a, *b, |a, b| if a < b { 1.0 } else { 0.0 })),
        ("compare_less_than", [a, b]) => Some(vec4_zip(*a, *b, |a, b| (a < b) as u8 as f32)),
        ("compare_less_equal", [a, b]) => Some(vec4_zip(*a, *b, |a, b| (a <= b) as u8 as f32)),
        ("compare_greater_than", [a, b]) => Some(vec4_zip(*a, *b, |a, b| (a > b) as u8 as f32)),
        ("compare_greater_equal", [a, b]) => Some(vec4_zip(*a, *b, |a, b| (a >= b) as u8 as f32)),
        ("compare_equal", [a, b]) => Some(vec4_zip(*a, *b, |a, b| (a == b) as u8 as f32)),
        ("compare_not_equal", [a, b]) => Some(vec4_zip(*a, *b, |a, b| (a != b) as u8 as f32)),
        ("compare_not_zero_ternary", [condition, if_true, if_false]) => {
            Some(std::array::from_fn(|lane| {
                if condition[lane] != 0.0 {
                    if_true[lane]
                } else {
                    if_false[lane]
                }
            }))
        }
        ("dot", [a, b]) => {
            let dot = a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum();
            Some([dot; 4])
        }
        ("cubic", [coefficients, x]) => Some(std::array::from_fn(|lane| {
            let high = coefficients[0] * x[lane] + coefficients[1];
            let low = coefficients[2] * x[lane] + coefficients[3];
            high * x[lane] * x[lane] + low
        })),
        ("lerp", [a, b, s]) => Some(std::array::from_fn(|lane| {
            a[lane] + (b[lane] - a[lane]) * s[lane]
        })),
        ("lerp_saturated", [a, b, s]) => Some(std::array::from_fn(|lane| {
            (a[lane] + (b[lane] - a[lane]) * s[lane]).clamp(0.0, f32::MAX)
        })),
        ("multiply_add", [a, b, c]) => {
            Some(vec4_zip(vec4_zip(*a, *b, |a, b| a * b), *c, |a, c| a + c))
        }
        ("clamp", [value, max, min]) => {
            Some(vec4_zip(vec4_zip(*value, *min, f32::max), *max, f32::min))
        }
        ("merge_1_3", [a, b]) => Some([a[0], b[0], b[1], b[2]]),
        ("merge_2_2", [a, b]) => Some([a[0], a[1], b[0], b[1]]),
        ("merge_3_1", [a, b]) => Some([a[0], a[1], a[2], b[0]]),
        ("transform_vec4", [x_axis, y_axis, z_axis, w_axis, value]) => {
            Some(std::array::from_fn(|lane| {
                x_axis[lane] * value[0]
                    + y_axis[lane] * value[1]
                    + z_axis[lane] * value[2]
                    + w_axis[lane] * value[3]
            }))
        }
        ("permute_extend_x", [a]) => Some([a[0]; 4]),
        ("is_zero", [a]) => Some(a.map(|v| if v == 0.0 { 1.0 } else { 0.0 })),
        ("abs", [a]) => Some(a.map(f32::abs)),
        ("signum", [a]) => Some(a.map(f32::signum)),
        ("floor", [a]) => Some(a.map(f32::floor)),
        ("ceil", [a]) => Some(a.map(f32::ceil)),
        ("round", [a]) => Some(a.map(f32::round)),
        ("frac", [a]) => Some(a.map(f32::fract)),
        ("negate", [a]) => Some(a.map(|v| -v)),
        ("saturate", [a]) => Some(a.map(|v| v.clamp(0.0, 1.0))),
        ("vector_rotations_sin", [a]) => Some(tfx_sin_rotations(*a)),
        ("vector_rotations_cos", [a]) => Some(tfx_sin_rotations(a.map(|value| value + 0.25))),
        ("vector_rotations_sin_cos", [a]) => {
            Some(tfx_sin_rotations([a[0], a[1] + 0.25, a[2], a[3] + 0.25]))
        }
        ("triangle", [a]) => Some(a.map(|value| (value - value.round()).abs() * 2.0)),
        ("jitter", [a]) => {
            let rotations = [4.67, 2.99, 1.08, 1.35]
                .iter()
                .zip([0.52, 0.37, 0.16, 0.79])
                .map(|(scale, offset)| a[0] * scale + offset);
            // Scaled sum of parabolic sines, then a Hermite smooth step.
            let v = rotations
                .map(|rotation| {
                    let wrapped = rotation - rotation.round();
                    wrapped * 0.25 * (wrapped.abs() * -16.0 + 8.0)
                })
                .sum::<f32>()
                + 0.5;
            Some([(-2.0 * v + 3.0) * v * v; 4])
        }
        ("wander", [a]) => {
            let pseudo_sin = |rotation: f32| {
                let wrapped = rotation - rotation.round();
                wrapped * (wrapped.abs() * -16.0 + 8.0)
            };
            let first = [(4.08, 0.92), (1.02, 0.33), (3.0 / 5.37, 0.26), (3.0 / 9.67, 0.54)];
            let second = [(1.83, 0.12, 0.02), (3.09, 0.37, 0.02), (0.39, 0.16, 0.28), (0.87, 0.79, 0.28)];
            let sum = first
                .iter()
                .zip(second)
                .map(|((scale0, offset0), (scale1, offset1, weight))| {
                    pseudo_sin(a[0] * scale0 + offset0) * pseudo_sin(a[0] * scale1 + offset1) * weight
                })
                .sum::<f32>();
            Some([0.5 + sum; 4])
        }
        ("rand", [a]) => Some([tfx_rand(a[0].floor()); 4]),
        ("rand_smooth", [a]) => {
            let base = a[0].round();
            let f = a[0] - base;
            let smooth = (-2.0 * f + 3.0) * f * f;
            let (from, to) = (tfx_rand(base), tfx_rand(base + 1.0));
            Some([from + (to - from) * smooth; 4])
        }
        // All four lanes are scaled by the reciprocal xyz length; a zero
        // vector stays zero.
        ("normalize3", [a]) => {
            let scale = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().recip();
            Some(if scale.is_finite() { a.map(|value| value * scale) } else { [0.0; 4] })
        }
        _ => None,
    }
}

/// Tiger's hash: the seed dotted with reciprocal primes, then one
/// Blum-Blum-Shub style squaring step.
fn tfx_rand(seed: f32) -> f32 {
    let dot = [1.0 / 1.043501, 1.0 / 0.794471, 1.0 / 0.113777, 1.0 / 0.015101]
        .iter()
        .map(|scale: &f32| seed * scale)
        .sum::<f32>();
    let value = dot.fract();
    (value * value * 251.0).fract()
}

fn tfx_sin_rotations(value: [f32; 4]) -> [f32; 4] {
    value.map(|value| {
        let wrapped = value - value.round();
        let estimate = wrapped * (-16.0 * wrapped.abs() + 8.0);
        estimate * (0.225 * estimate.abs() + 0.775)
    })
}

fn vec4_zip(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> f32) -> [f32; 4] {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}

fn parse_stage_slot(detail: &str) -> Option<(&'static str, u8)> {
    let (stage_text, slot_text) = detail.split_once(" slot=")?;
    let slot = slot_text.parse().ok()?;
    let stage = match stage_text {
        "PS" => "PS",
        "VS" => "VS",
        "GS" => "GS",
        "HS" => "HS",
        "CS" => "CS",
        "DS" => "DS",
        _ => "??",
    };
    Some((stage, slot))
}

fn tfx_extern_label(value: u8, name: fn(u8) -> &'static str) -> String {
    match name(value) {
        "Extern" => format!("Extern[{value}]"),
        known => known.to_owned(),
    }
}

fn parse_tfx_bytecode_op(
    data: &[u8],
    cursor: &mut usize,
    offset: usize,
    opcode: u8,
    extern_name: fn(u8) -> &'static str,
) -> Option<TfxBytecodeOpPreview> {
    let mut read_u8 = || {
        let value = *data.get(*cursor)?;
        *cursor += 1;
        Some(value)
    };

    let (name, detail) = match opcode {
        0x01 => ("add", String::new()),
        0x02 => ("subtract", String::new()),
        0x03 => ("multiply", String::new()),
        0x04 => ("divide", String::new()),
        0x05 => ("multiply2", String::new()),
        0x06 => ("add2", String::new()),
        0x07 => ("is_zero", String::new()),
        0x08 => ("min", String::new()),
        0x09 => ("max", String::new()),
        0x0a => ("less_than", String::new()),
        0x0b => ("dot", String::new()),
        0x0c => ("merge_1_3", String::new()),
        0x0d => ("merge_2_2", String::new()),
        0x0e => ("merge_3_1", String::new()),
        0x0f => ("cubic", String::new()),
        0x10 => ("lerp", String::new()),
        0x11 => ("lerp_saturated", String::new()),
        0x12 => ("multiply_add", String::new()),
        0x13 => ("clamp", String::new()),
        0x14 => ("unk14", String::new()),
        0x15 => ("abs", String::new()),
        0x16 => ("signum", String::new()),
        0x17 => ("floor", String::new()),
        0x18 => ("ceil", String::new()),
        0x19 => ("round", String::new()),
        0x1a => ("frac", String::new()),
        0x1b => ("unk1b", String::new()),
        0x1c => ("normalize3", String::new()),
        0x1d => ("negate", String::new()),
        0x1e => ("vector_rotations_sin", String::new()),
        0x1f => ("vector_rotations_cos", String::new()),
        0x20 => ("vector_rotations_sin_cos", String::new()),
        0x21 => ("permute_extend_x", ".xxxx".to_string()),
        0x22 => ("permute", format!("fields=0x{:02X}", read_u8()?)),
        0x23 => ("saturate", String::new()),
        0x24 => ("unk24", String::new()),
        0x25 => ("unk25", String::new()),
        0x26 => ("unk26", String::new()),
        0x27 => ("triangle", String::new()),
        0x28 => ("jitter", String::new()),
        0x29 => ("wander", String::new()),
        0x2a => ("rand", String::new()),
        0x2b => ("rand_smooth", String::new()),
        0x2c => ("unk2c", String::new()),
        0x2d => ("unk2d", String::new()),
        0x2e => ("transform_vec4", String::new()),
        0x34 => ("push_const_vec4", format!("constant={}", read_u8()?)),
        0x35 => ("lerp_constant", format!("start={}", read_u8()?)),
        0x36 => ("lerp_constant_saturated", format!("start={}", read_u8()?)),
        0x37 => ("spline4_const", format!("start={}", read_u8()?)),
        0x38 => ("spline8_const", format!("start={}", read_u8()?)),
        0x39 => ("spline8_chain_const", format!("start={}", read_u8()?)),
        0x3a => ("gradient4_const", format!("start={}", read_u8()?)),
        0x3b => ("unk3b", format!("start={}", read_u8()?)),
        0x3c => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_float",
                format!(
                    "{}+0x{:X}",
                    tfx_extern_label(extern_, extern_name),
                    offset as usize * 4
                ),
            )
        }
        0x3d => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_vec4",
                format!(
                    "{}+0x{:X}",
                    tfx_extern_label(extern_, extern_name),
                    offset as usize * 16
                ),
            )
        }
        0x3e => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_mat4",
                format!(
                    "{}+0x{:X}",
                    tfx_extern_label(extern_, extern_name),
                    offset as usize * 16
                ),
            )
        }
        0x3f => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_texture",
                format!(
                    "{}+0x{:X}",
                    tfx_extern_label(extern_, extern_name),
                    offset as usize * 8
                ),
            )
        }
        0x40 => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_u32",
                format!(
                    "{}+0x{:X}",
                    tfx_extern_label(extern_, extern_name),
                    offset as usize * 4
                ),
            )
        }
        0x41 => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_uav",
                format!(
                    "{}+0x{:X}",
                    tfx_extern_label(extern_, extern_name),
                    offset as usize * 8
                ),
            )
        }
        0x42 => ("unk42", String::new()),
        0x43 => ("push_from_output", format!("element={}", read_u8()?)),
        0x44 => ("pop_output", format!("element={}", read_u8()?)),
        0x45 => ("pop_output_mat4", format!("element={}", read_u8()?)),
        0x46 => ("push_temp", format!("slot={}", read_u8()?)),
        0x47 => ("pop_temp", format!("slot={}", read_u8()?)),
        0x48 => {
            let value = read_u8()?;
            (
                "set_shader_texture",
                format!("{} slot={}", tfx_shader_stage_name(value), value & 0x1f),
            )
        }
        0x49 => ("unk49", format!("value={}", read_u8()?)),
        0x4a => {
            let value = read_u8()?;
            (
                "set_shader_sampler",
                format!("{} slot={}", tfx_shader_stage_name(value), value & 0x1f),
            )
        }
        0x4b => {
            let value = read_u8()?;
            (
                "set_shader_uav",
                format!("{} slot={}", tfx_shader_stage_name(value), value & 0x1f),
            )
        }
        0x4c => ("unk4c", format!("value={}", read_u8()?)),
        0x4d => ("push_sampler", format!("index={}", read_u8()?)),
        0x4e => {
            let hash = read_be_u32(data.get(*cursor..*cursor + 4)?)?;
            *cursor += 4;
            ("push_object_channel", format!("0x{hash:08X}"))
        }
        0x4f => ("push_global_channel", format!("index={}", read_u8()?)),
        0x50 => ("unk50", format!("value={}", read_u8()?)),
        0x51 => ("unk51", String::new()),
        0x52 => {
            let index = read_u8()?;
            let fields = read_u8()?;
            (
                "push_tex_dimensions",
                format!("index={index} fields=0x{fields:02X}"),
            )
        }
        0x53 => {
            let index = read_u8()?;
            let fields = read_u8()?;
            (
                "push_tex_tiling_params",
                format!("index={index} fields=0x{fields:02X}"),
            )
        }
        0x54 => {
            let index = read_u8()?;
            let fields = read_u8()?;
            (
                "push_tex_tile_layer_count",
                format!("index={index} fields=0x{fields:02X}"),
            )
        }
        0x55 => ("unk55", String::new()),
        0x56 => ("unk56", String::new()),
        0x57 => ("unk57", String::new()),
        0x58 => ("unk58", String::new()),
        _ => return None,
    };

    Some(TfxBytecodeOpPreview {
        offset,
        opcode,
        name,
        detail,
        extern_scope_id: None,
    })
}

fn read_be_u32(data: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes([data[0], data[1], data[2], data[3]]))
}

fn tfx_shader_stage_name(value: u8) -> &'static str {
    match value >> 5 {
        1 => "PS",
        2 => "VS",
        3 => "GS",
        4 => "HS",
        5 => "CS",
        6 => "DS",
        _ => "??",
    }
}

fn tfx_extern_name(value: u8) -> &'static str {
    match value {
        0 => "None",
        1 => "Frame",
        2 => "View",
        3 => "Deferred",
        4 => "DeferredLight",
        5 => "DeferredUberLight",
        6 => "DeferredShadow",
        7 => "Atmosphere",
        8 => "RigidModel",
        9 => "EditorMesh",
        10 => "EditorMeshMaterial",
        11 => "EditorDecal",
        12 => "EditorTerrain",
        13 => "EditorTerrainPatch",
        14 => "EditorTerrainDebug",
        15 => "SimpleGeometry",
        16 => "UiFont",
        17 => "CuiView",
        18 => "CuiObject",
        19 => "CuiBitmap",
        20 => "CuiVideo",
        21 => "CuiStandard",
        22 => "CuiHud",
        23 => "CuiScreenspaceBoxes",
        24 => "TextureVisualizer",
        25 => "Generic",
        26 => "Particle",
        27 => "ParticleDebug",
        28 => "GearDyeVisualizationMode",
        29 => "ScreenArea",
        30 => "Mlaa",
        31 => "Msaa",
        32 => "Hdao",
        33 => "DownsampleTextureGeneric",
        34 => "DownsampleDepth",
        35 => "Ssao",
        36 => "VolumetricObscurance",
        37 => "Postprocess",
        38 => "TextureSet",
        39 => "Transparent",
        40 => "Vignette",
        41 => "GlobalLighting",
        42 => "ShadowMask",
        43 => "ObjectEffect",
        44 => "Decal",
        45 => "DecalSetTransform",
        46 => "DynamicDecal",
        47 => "DecoratorWind",
        48 => "TextureCameraLighting",
        49 => "VolumeFog",
        50 => "Fxaa",
        51 => "Smaa",
        52 => "Letterbox",
        53 => "DepthOfField",
        54 => "PostprocessInitialDownsample",
        55 => "CopyDepth",
        56 => "DisplacementMotionBlur",
        57 => "DebugShader",
        58 => "MinmaxDepth",
        59 => "SdsmBiasAndScale",
        60 => "SdsmBiasAndScaleTextures",
        61 => "ComputeShadowMapData",
        62 => "ComputeLocalLightShadowMapData",
        63 => "BilateralUpsample",
        64 => "HealthOverlay",
        65 => "LightProbeDominantLight",
        66 => "LightProbeLightInstance",
        67 => "Water",
        68 => "LensFlare",
        69 => "ScreenShader",
        70 => "Scaler",
        71 => "GammaControl",
        72 => "SpeedtreePlacements",
        73 => "Reticle",
        74 => "Distortion",
        75 => "WaterDebug",
        76 => "ScreenAreaInput",
        77 => "WaterDepthPrepass",
        78 => "OverheadVisibilityMap",
        79 => "ParticleCompute",
        80 => "CubemapFiltering",
        81 => "ParticleFastpath",
        82 => "VolumetricsPass",
        83 => "TemporalReprojection",
        84 => "FxaaCompute",
        85 => "VbCopyCompute",
        86 => "UberDepth",
        87 => "GearDye",
        88 => "Cubemaps",
        89 => "ShadowBlendWithPrevious",
        90 => "DebugShadingOutput",
        91 => "Ssao3d",
        92 => "WaterDisplacement",
        93 => "PatternBlending",
        94 => "UiHdrTransform",
        95 => "PlayerCenteredCascadedGrid",
        96 => "SoftDeform",
        _ => "Extern",
    }
}

/// Marathon inserts `CuiDrawingShader` at extern index 24 and appends
/// `ParticleMeshEmissionCompute` after the legacy table. Keep Destiny's table
/// above unchanged; its bytecode uses the legacy numbering.
fn tfx_marathon_extern_name(value: u8) -> &'static str {
    match value {
        0..=23 => tfx_extern_name(value),
        24 => "CuiDrawingShader",
        25..=97 => tfx_extern_name(value - 1),
        98 => "ParticleMeshEmissionCompute",
        _ => "Extern",
    }
}

fn tfx_scope_names(mask: u64) -> Vec<&'static str> {
    const SCOPES: &[(usize, &str)] = &[
        (0, "Frame"),
        (1, "View"),
        (2, "RigidModel"),
        (3, "EditorMesh"),
        (4, "EditorTerrain"),
        (5, "CuiView"),
        (6, "CuiObject"),
        (7, "Skinning"),
        (8, "SpeedTree"),
        (9, "ChunkModel"),
        (10, "Decal"),
        (11, "Instances"),
        (12, "SpeedTreeLodDrawcallData"),
        (13, "Transparent"),
        (14, "TransparentAdvanced"),
        (15, "SdsmBiasAndScaleTextures"),
        (16, "Terrain"),
        (17, "Postprocess"),
        (18, "CuiBitmap"),
        (19, "CuiStandard"),
        (20, "UiFont"),
        (21, "CuiHud"),
        (22, "ParticleTransforms"),
        (23, "ParticleLocationMetadata"),
        (24, "CubemapVolume"),
        (25, "GearPlatedTextures"),
        (26, "GearDye0"),
        (27, "GearDye1"),
        (28, "GearDye2"),
        (29, "GearDyeDecal"),
        (30, "GenericArray"),
        (31, "GearDyeSkin"),
        (32, "GearDyeLips"),
        (33, "GearDyeHair"),
        (34, "GearDyeFacialLayer0Mask"),
        (35, "GearDyeFacialLayer0Material"),
        (36, "GearDyeFacialLayer1Mask"),
        (37, "GearDyeFacialLayer1Material"),
        (38, "PlayerCenteredCascadedGrid"),
        (39, "GearDye012"),
        (40, "ColorGradingUbershader"),
    ];

    let mut names = SCOPES
        .iter()
        .filter_map(|(bit, name)| ((mask & (1u64 << bit)) != 0).then_some(*name))
        .collect::<Vec<_>>();
    if mask != 0 {
        let known_mask = SCOPES
            .iter()
            .fold(0u64, |acc, (bit, _)| acc | (1u64 << bit));
        let unknown = mask & !known_mask;
        if unknown != 0 {
            names.push("UnknownScopeBits");
        }
    }
    names
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

fn read_f32(data: &[u8], endian: Endian) -> f32 {
    f32::from_bits(read_u32(data, endian))
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

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "material_piecewise_reference_tests.rs"]
mod piecewise_reference_tests;

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
    pub samplers: Vec<WideHashPreview>,
    pub inline_constants: Vec<[f32; 4]>,
    pub bytecode_len: usize,
    pub constant_buffer_slot: Option<i32>,
    pub constant_buffer: Option<TagHash>,
    pub bytecode: TfxBytecodePreview,
}

#[derive(Debug, Clone)]
pub struct TextureSlotBindingPreview {
    pub slot: u32,
    pub texture: WideHashPreview,
}

#[derive(Debug, Clone, Default)]
pub struct TfxBytecodePreview {
    pub total_bytes: usize,
    pub ops: Vec<TfxBytecodeOpPreview>,
    pub bindings: Vec<TfxBindingPreview>,
    pub externs: Vec<TfxExternRefPreview>,
    pub decoded_ops: usize,
    pub unknown_ops: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct TfxBindingPreview {
    pub kind: &'static str,
    pub stage: &'static str,
    pub slot: u8,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct TfxExternRefPreview {
    pub op_offset: usize,
    pub value_type: &'static str,
    pub scope: String,
    pub byte_offset: usize,
}

#[derive(Debug, Clone)]
pub struct TfxBytecodeOpPreview {
    pub offset: usize,
    pub opcode: u8,
    pub name: &'static str,
    pub detail: String,
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
    let bytecode = read_array(data, constants_offset, 1, endian)
        .map(parse_tfx_bytecode)
        .unwrap_or_default();
    let bytecode_len = bytecode.total_bytes;
    let constants = read_array(data, constants_offset + 0x10, 0x10, endian)
        .map(|constants| parse_vec4_array(constants, endian))
        .unwrap_or_default();
    let samplers = read_array(data, constants_offset + 0x20, 0x10, endian)
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
        && samplers.is_empty()
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
        samplers,
        inline_constants,
        bytecode_len,
        constant_buffer_slot,
        constant_buffer,
        bytecode,
    })
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

fn parse_tfx_bytecode(data: &[u8]) -> TfxBytecodePreview {
    const MAX_UI_OPS: usize = 160;

    let mut cursor = 0usize;
    let mut ops = Vec::with_capacity(data.len().min(MAX_UI_OPS));
    let mut decoded_ops = 0usize;
    let mut unknown_ops = 0usize;

    while cursor < data.len() {
        let offset = cursor;
        let opcode = data[cursor];
        cursor += 1;

        let Some(op) = parse_tfx_bytecode_op(data, &mut cursor, offset, opcode) else {
            unknown_ops += 1;
            if ops.len() < MAX_UI_OPS {
                ops.push(TfxBytecodeOpPreview {
                    offset,
                    opcode,
                    name: "unknown",
                    detail: String::new(),
                });
            }
            break;
        };

        decoded_ops += 1;
        if ops.len() < MAX_UI_OPS {
            ops.push(op);
        }
    }

    let bindings = summarize_tfx_bindings(&ops);
    let externs = summarize_tfx_externs(&ops);

    TfxBytecodePreview {
        total_bytes: data.len(),
        bindings,
        externs,
        decoded_ops,
        unknown_ops,
        truncated: decoded_ops > ops.len(),
        ops,
    }
}

fn summarize_tfx_externs(ops: &[TfxBytecodeOpPreview]) -> Vec<TfxExternRefPreview> {
    ops.iter()
        .filter_map(|op| {
            let value_type = op.name.strip_prefix("push_extern_")?;
            let (scope, offset_hex) = op.detail.split_once("+0x")?;
            let byte_offset = usize::from_str_radix(offset_hex, 16).ok()?;
            Some(TfxExternRefPreview {
                op_offset: op.offset,
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
            })
        })
        .collect()
}

fn summarize_tfx_bindings(ops: &[TfxBytecodeOpPreview]) -> Vec<TfxBindingPreview> {
    let mut stack = Vec::<String>::new();
    let mut bindings = Vec::new();

    for op in ops {
        match op.name {
            "push_const_vec4"
            | "push_sampler"
            | "push_extern_float"
            | "push_extern_vec4"
            | "push_extern_mat4"
            | "push_extern_texture"
            | "push_extern_u32"
            | "push_extern_uav"
            | "push_from_output"
            | "push_object_channel"
            | "push_global_channel"
            | "push_tex_dimensions"
            | "push_tex_tiling_params"
            | "push_tex_tile_layer_count"
            | "push_temp" => stack.push(format_tfx_value(op)),
            "set_shader_texture" | "set_shader_sampler" | "set_shader_uav" => {
                if let Some((stage, slot)) = parse_stage_slot(&op.detail) {
                    bindings.push(TfxBindingPreview {
                        kind: match op.name {
                            "set_shader_texture" => "texture",
                            "set_shader_sampler" => "sampler",
                            "set_shader_uav" => "uav",
                            _ => "binding",
                        },
                        stage,
                        slot,
                        source: stack.pop().unwrap_or_else(|| "<empty stack>".to_string()),
                    });
                }
            }
            "pop_output" | "pop_output_mat4" | "pop_temp" => {
                let _ = stack.pop();
            }
            "add" | "subtract" | "multiply" | "divide" | "min" | "max" | "less_than" | "dot"
            | "lerp" | "lerp_saturated" | "multiply_add" | "clamp" => {
                collapse_stack(&mut stack, op.name, 2);
            }
            "abs" | "signum" | "floor" | "ceil" | "round" | "frac" | "negate" | "saturate" => {
                collapse_stack(&mut stack, op.name, 1);
            }
            _ => {}
        }
    }

    bindings
}

fn collapse_stack(stack: &mut Vec<String>, name: &str, inputs: usize) {
    if stack.len() < inputs {
        stack.push(format!("{name}(?)"));
        return;
    }

    let args = stack.split_off(stack.len() - inputs).join(", ");
    stack.push(format!("{name}({args})"));
}

fn format_tfx_value(op: &TfxBytecodeOpPreview) -> String {
    match op.name {
        "push_const_vec4" => op
            .detail
            .strip_prefix("constant=")
            .map(|index| format!("constant[{index}]"))
            .unwrap_or_else(|| op.detail.clone()),
        "push_sampler" => op
            .detail
            .strip_prefix("index=")
            .map(|index| format!("sampler[{index}]"))
            .unwrap_or_else(|| op.detail.clone()),
        "push_temp" => op
            .detail
            .strip_prefix("slot=")
            .map(|slot| format!("temp[{slot}]"))
            .unwrap_or_else(|| op.detail.clone()),
        _ => {
            if op.detail.is_empty() {
                op.name.to_string()
            } else {
                format!("{}({})", op.name.trim_start_matches("push_"), op.detail)
            }
        }
    }
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

fn parse_tfx_bytecode_op(
    data: &[u8],
    cursor: &mut usize,
    offset: usize,
    opcode: u8,
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
        0x1c => ("unk1c", String::new()),
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
                format!("{}+0x{:X}", tfx_extern_name(extern_), offset as usize * 4),
            )
        }
        0x3d => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_vec4",
                format!("{}+0x{:X}", tfx_extern_name(extern_), offset as usize * 16),
            )
        }
        0x3e => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_mat4",
                format!("{}+0x{:X}", tfx_extern_name(extern_), offset as usize * 16),
            )
        }
        0x3f => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_texture",
                format!("{}+0x{:X}", tfx_extern_name(extern_), offset as usize * 8),
            )
        }
        0x40 => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_u32",
                format!("{}+0x{:X}", tfx_extern_name(extern_), offset as usize * 4),
            )
        }
        0x41 => {
            let extern_ = read_u8()?;
            let offset = read_u8()?;
            (
                "push_extern_uav",
                format!("{}+0x{:X}", tfx_extern_name(extern_), offset as usize * 8),
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
        15 => "SimpleGeometry",
        25 => "Generic",
        38 => "TextureSet",
        39 => "Transparent",
        41 => "GlobalLighting",
        44 => "Decal",
        67 => "Water",
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

    #[test]
    fn decodes_tfx_binding_opcodes() {
        let bytecode = [
            0x34, 0x02, // push_const_vec4 constant 2
            0x4d, 0x01, // push_sampler index 1
            0x4a, 0x20, // set_shader_sampler PS slot 0
            0x48, 0x40, // set_shader_texture VS slot 0
            0x4e, 0x12, 0x34, 0x56, 0x78, // push_object_channel
        ];

        let decoded = parse_tfx_bytecode(&bytecode);

        assert_eq!(decoded.decoded_ops, 5);
        assert_eq!(decoded.unknown_ops, 0);
        assert_eq!(decoded.bindings.len(), 2);
        assert_eq!(decoded.bindings[0].kind, "sampler");
        assert_eq!(decoded.bindings[0].stage, "PS");
        assert_eq!(decoded.bindings[0].slot, 0);
        assert_eq!(decoded.bindings[0].source, "sampler[1]");
        assert_eq!(decoded.bindings[1].kind, "texture");
        assert_eq!(decoded.bindings[1].stage, "VS");
        assert!(decoded.externs.is_empty());
        assert_eq!(decoded.ops[0].name, "push_const_vec4");
        assert_eq!(decoded.ops[2].detail, "PS slot=0");
        assert_eq!(decoded.ops[3].detail, "VS slot=0");
        assert_eq!(decoded.ops[4].detail, "0x12345678");
    }

    #[test]
    fn parses_vec4_constants() {
        let mut data = vec![];
        for value in [1.0_f32, -2.5, 3.25, 4.5, 5.0, 6.0, 7.0, 8.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }

        let constants = parse_vec4_array(&data, Endian::Little);

        assert_eq!(constants.len(), 2);
        assert_eq!(constants[0], [1.0, -2.5, 3.25, 4.5]);
        assert_eq!(constants[1], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn decodes_extern_refs() {
        let decoded = parse_tfx_bytecode(&[
            0x3d, 0x02, 0x03, // push_extern_vec4 View+0x30
            0x3f, 0x26, 0x04, // push_extern_texture TextureSet+0x20
        ]);

        assert_eq!(decoded.externs.len(), 2);
        assert_eq!(decoded.externs[0].value_type, "vec4");
        assert_eq!(decoded.externs[0].scope, "View");
        assert_eq!(decoded.externs[0].byte_offset, 0x30);
        assert_eq!(decoded.externs[1].value_type, "texture");
        assert_eq!(decoded.externs[1].scope, "TextureSet");
        assert_eq!(decoded.externs[1].byte_offset, 0x20);
    }

    #[test]
    fn decodes_scope_bits() {
        let names = tfx_scope_names((1 << 0) | (1 << 2) | (1 << 40));

        assert_eq!(names, vec!["Frame", "RigidModel", "ColorGradingUbershader"]);
    }
}

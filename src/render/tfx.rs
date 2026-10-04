use std::collections::{BTreeMap, HashMap};

use binrw::Endian;
use serde::Serialize;
use tiger_pkg::{TagHash, Version, package_manager};

use crate::material::{
    TfxBytecodePreview, TfxDecodeStatus, interpret_tfx_stack_with_runtime_values,
};

#[derive(Debug, Clone, PartialEq)]
pub enum TfxValue {
    Scalar(f32),
    Vector([f32; 4]),
    TextureBinding { stage: &'static str, slot: u8 },
    Unknown(String),
}

#[derive(Debug, Clone, Default)]
pub struct TfxRuntimeInputs {
    pub time_seconds: f32,
    pub frame: BTreeMap<u32, TfxValue>,
    pub view: BTreeMap<u32, TfxValue>,
    pub object_channels: BTreeMap<u32, TfxValue>,
    /// Branch-owned catalog values for merged shell geometry. Shared ambiguous
    /// owners get an empty map; no sibling object's channels are imported.
    pub geometry_object_channels: BTreeMap<TagHash, BTreeMap<u32, TfxValue>>,
    pub global_channels: BTreeMap<u32, TfxValue>,
    pub gear_channels: BTreeMap<u32, TfxValue>,
    /// Numeric runtime externs keyed by their authored scope ID and byte offset.
    /// Required for producers such as Marathon compute skinning (scope108),
    /// whose identity must survive even before its semantic name is known.
    pub scoped_externs: BTreeMap<(u8, u32), TfxValue>,
    /// Typed resource externs keyed by Tiger scope name + byte offset.
    /// Numeric externs stay in the scope maps above; texture/UAV/resource
    /// externs live here so renderer binding state does not collapse into vec4s.
    pub extern_resources: BTreeMap<(String, u32), TagHash>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MarathonGlobalChannelTable {
    pub render_globals: TagHash,
    pub channel_table: TagHash,
    pub channel_ids: Vec<u32>,
    pub default_values: Vec<[f32; 4]>,
}

impl TfxRuntimeInputs {
    /// Static catalog policy retains each geometry's authored Pattern ancestry.
    /// Child scope values override parents only along that geometry's path;
    /// sibling and ambiguous owners are never imported.
    pub(crate) fn for_model_preview(cache: &quicktag_scanner::TagCache, pattern: TagHash) -> Self {
        let mut inputs=Self::for_pattern_preview(cache,pattern);
        if matches!(package_manager().version,tiger_pkg::GameVersion::Marathon(tiger_pkg::MarathonVersion::Marathon)) {
            inputs.apply_static_preview_coverage();
            inputs.apply_static_preview_highlight_planes();
        }
        if let Some(shell)=crate::geometry::RunnerShellAssembly::resolve(cache,pattern) {
            let mut scopes=BTreeMap::new();
            for part in shell.parts {
                let channels=if part.pattern_path.is_empty() { BTreeMap::new() } else {
                    let mut channels=BTreeMap::new();
                    for owner in &part.pattern_path {
                        let scope=scopes.entry(*owner).or_insert_with(||Self::for_pattern_preview(cache,*owner).object_channels);
                        channels.extend(scope.clone());
                    }
                    channels
                };
                for geometry in part.geometry { inputs.geometry_object_channels.insert(geometry,channels.clone()); }
            }
        }
        inputs
    }

    /// Viewer policy: select source shaders' analytic alpha cutoff instead of
    /// stochastic coverage. This is not a captured Marathon Frame value.
    /// Explicit frame values (including Unknown) and scoped overrides win.
    pub(crate) fn apply_static_preview_coverage(&mut self) {
        self.frame.entry(0x1e0).or_insert(TfxValue::Vector([0.0;4]));
    }

    /// Viewer policy: View+0x470..0x4E0 hold eight planes that the engine
    /// writes at runtime; no package carries them. Materials test the scaled
    /// reflection vector against them to gate an additive highlight lookup.
    /// Zero planes fail every test, so the highlight is absent while the rest
    /// of the material stays exact. This is not a captured Marathon View value.
    pub(crate) fn apply_static_preview_highlight_planes(&mut self) {
        for offset in (0x470..=0x4e0).step_by(16) {
            self.view.entry(offset).or_insert(TfxValue::Vector([0.0;4]));
        }
    }

    pub(crate) fn for_geometry(&self, geometry: TagHash) -> Self {
        let mut scoped=self.clone();
        if let Some(channels)=self.geometry_object_channels.get(&geometry) {
            scoped.object_channels=channels.clone();
        }
        scoped.geometry_object_channels.clear();
        scoped
    }

    fn for_pattern_preview(cache: &quicktag_scanner::TagCache, pattern: TagHash) -> Self {
        let mut inputs=Self::default();
        inputs.apply_marathon_global_defaults();
        let evidence=crate::geometry::pattern_object_channel_evidence(cache,pattern);
        let nearest=evidence.iter().map(|scope|scope.depth).min();
        let owners=evidence.iter().filter(|scope|Some(scope.depth)==nearest).collect::<Vec<_>>();
        if let [scope]=owners.as_slice() {
            for channel in &scope.channels {
                inputs.object_channels.insert(channel.hash,TfxValue::Vector([1.0;4]));
            }
            for binding in scope.bindings.iter().filter(|b|b.scope==0x811C9DC5) {
                if let Some(value)=binding.value {
                    inputs.object_channels.insert(binding.parameter,TfxValue::Vector(value));
                }
            }
        }
        // Catalog models have no local-player ownership. This is viewer policy,
        // independent of the serialized Pattern vector bindings.
        inputs.object_channels.insert(0x8A4DE2D7,TfxValue::Vector([0.0;4]));
        inputs
    }

    /// Load raw indexed defaults from Marathon's authored render_globals table.
    /// TFX's push_global_channel operand addresses these slots directly.
    pub fn apply_marathon_global_defaults(&mut self) -> Option<MarathonGlobalChannelTable> {
        let table = marathon_global_channel_table()?;
        // Alkahest initializes all 256 indexed globals to ONE before applying
        // the serialized prefix. Seed that complete table here, not in the
        // interpreter: explicit values and Unknown overrides must still win.
        for index in 0..256 {
            let value=table.default_values.get(index).copied().unwrap_or([1.0;4]);
            self.global_channels.entry(index as u32).or_insert(TfxValue::Vector(value));
        }
        Some(table)
    }
}

pub fn marathon_global_channel_table() -> Option<MarathonGlobalChannelTable> {
    const RENDER_GLOBALS_CLASS: u32 = 0x80808070;
    const CHANNEL_TABLE_CLASS: u32 = 0x8080A014;
    const CHANNEL_ID_CLASS: u32 = 0x80800070;
    const VECTOR4_CLASS: u32 = 0x80800090;
    const ARRAY_MARKER: u32 = 0x8080BFCD;

    let manager = package_manager();
    let endian = manager.version.endian();
    let render_globals = manager
        .get_all_by_reference(RENDER_GLOBALS_CLASS)
        .into_iter()
        .map(|(tag, _)| tag)
        .max_by_key(|tag| tag.0)?;
    let globals = manager.read_tag(render_globals).ok()?;
    let channel_table = TagHash(read_u32_tfx(globals.get(0x34..0x38)?, endian));
    if manager.get_entry(channel_table)?.reference != CHANNEL_TABLE_CLASS {
        return None;
    }
    let data = manager.read_tag(channel_table).ok()?;
    let markers = (0..data.len().saturating_sub(4))
        .step_by(4)
        .filter(|offset| read_u32_tfx(&data[*offset..*offset + 4], endian) == ARRAY_MARKER)
        .collect::<Vec<_>>();
    let mut channel_ids: Option<Vec<u32>> = None;
    let mut default_values: Option<Vec<[f32; 4]>> = None;
    for (marker_index, marker) in markers.iter().copied().enumerate() {
        let count = read_u64_tfx(data.get(marker + 4..marker + 12)?, endian) as usize;
        let class = read_u32_tfx(data.get(marker + 12..marker + 16)?, endian);
        let start = marker + 20;
        let end = markers
            .get(marker_index + 1)
            .copied()
            .unwrap_or(data.len())
            .min(data.len());
        match class {
            CHANNEL_ID_CLASS if start.checked_add(count.checked_mul(4)?)? <= end => {
                channel_ids = Some(
                    data[start..start + count * 4]
                        .chunks_exact(4)
                        .map(|bytes| read_u32_tfx(bytes, endian))
                        .collect(),
                );
            }
            VECTOR4_CLASS if start.checked_add(count.checked_mul(16)?)? <= end => {
                default_values = Some(
                    data[start..start + count * 16]
                        .chunks_exact(16)
                        .map(|bytes| {
                            [
                                f32::from_bits(read_u32_tfx(&bytes[0..4], endian)),
                                f32::from_bits(read_u32_tfx(&bytes[4..8], endian)),
                                f32::from_bits(read_u32_tfx(&bytes[8..12], endian)),
                                f32::from_bits(read_u32_tfx(&bytes[12..16], endian)),
                            ]
                        })
                        .collect(),
                );
            }
            _ => {}
        }
    }
    let channel_ids = channel_ids?;
    let default_values = default_values?;
    (channel_ids.len() == default_values.len()).then_some(MarathonGlobalChannelTable {
        render_globals,
        channel_table,
        channel_ids,
        default_values,
    })
}

fn read_u32_tfx(bytes: &[u8], endian: Endian) -> u32 {
    let bytes: [u8; 4] = bytes.try_into().expect("validated u32 slice");
    match endian {
        Endian::Little => u32::from_le_bytes(bytes),
        Endian::Big => u32::from_be_bytes(bytes),
    }
}

fn read_u64_tfx(bytes: &[u8], endian: Endian) -> u64 {
    let bytes: [u8; 8] = bytes.try_into().expect("validated u64 slice");
    match endian {
        Endian::Little => u64::from_le_bytes(bytes),
        Endian::Big => u64::from_be_bytes(bytes),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TfxDependency {
    pub scope_id: u8,
    pub scope: String,
    pub byte_offset: usize,
    pub resolved: bool,
}

#[derive(Debug, Clone)]
pub struct TfxTraceStep {
    pub byte_offset: usize,
    pub opcode: u8,
    pub operation: &'static str,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TfxRuntimeBinding {
    pub kind: &'static str,
    pub stage: &'static str,
    pub slot: u8,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct TfxExecutionResult {
    pub status: TfxDecodeStatus,
    pub outputs: BTreeMap<String, TfxValue>,
    pub bindings: Vec<TfxRuntimeBinding>,
    pub dependencies: Vec<TfxDependency>,
    pub trace: Vec<TfxTraceStep>,
    pub undecoded_offset: Option<usize>,
    pub undecoded_bytes: Vec<u8>,
}

pub fn execute_preview(
    program: &TfxBytecodePreview,
    constants: &[[f32; 4]],
    inputs: &TfxRuntimeInputs,
    texture_metadata: &BTreeMap<u8, crate::texture::TextureExpressionMetadata>,
) -> TfxExecutionResult {
    let dependencies = program
        .externs
        .iter()
        .map(|external| TfxDependency {
            scope_id: external.scope_id,
            scope: external.scope.clone(),
            byte_offset: external.byte_offset,
            resolved: (0..if external.value_type == "mat4" { 4 } else { 1 }).all(|row| {
                resolve_external(
                    inputs,
                    external.scope_id,
                    &external.scope,
                    external.byte_offset as u32 + row * 16,
                )
                .and_then(|value| value.as_vector())
                .is_some()
            }) || matches!(external.value_type, "texture" | "uav" | "resource")
                && inputs
                    .extern_resources
                    .contains_key(&(external.scope.clone(), external.byte_offset as u32)),
        })
        .collect();
    let vectors = |values: &BTreeMap<u32, TfxValue>| {
        values
            .iter()
            .filter_map(|(key, value)| value.as_vector().map(|value| (*key, value)))
            .collect::<HashMap<_, _>>()
    };
    let object_channels = vectors(&inputs.object_channels);
    let global_channels = vectors(&inputs.global_channels);
    let mut extern_values = HashMap::new();
    for (scope, values) in [
        ("Frame", &inputs.frame),
        ("View", &inputs.view),
        ("RigidModel", &inputs.object_channels),
        ("EditorMesh", &inputs.object_channels),
        ("GlobalChannel", &inputs.global_channels),
        ("Gear", &inputs.gear_channels),
        ("TextureSet", &inputs.gear_channels),
    ] {
        for (offset, value) in values {
            if let Some(value) = value.as_vector() {
                extern_values.insert((scope.to_string(), *offset), value);
            }
        }
    }
    // The static preview owns both clocks. Frame+0 is game time and Frame+4
    // is render time; Alkahest's Frame writer supplies the same packet time
    // to both. Explicit frame/scoped values still override this policy.
    for offset in [0, 4] {
        if !inputs.frame.contains_key(&offset) {
            extern_values.insert(("Frame".to_string(), offset), [inputs.time_seconds; 4]);
        }
    }
    for external in &program.externs {
        for row in 0..if external.value_type == "mat4" { 4 } else { 1 } {
            let offset = external.byte_offset as u32 + row * 16;
            if let Some(value) = inputs.scoped_externs.get(&(external.scope_id, offset)) {
                let key = (external.scope.clone(), offset);
                if let Some(vector) = value.as_vector() {
                    extern_values.insert(key, vector);
                } else {
                    extern_values.remove(&key);
                }
            }
        }
    }
    let (bindings, expressions) = interpret_tfx_stack_with_runtime_values(
        &program.ops,
        constants,
        &object_channels,
        &extern_values,
        &global_channels,
        texture_metadata,
    );
    let mut outputs = BTreeMap::new();
    for expression in &expressions {
        outputs.insert(
            expression.target.clone(),
            expression
                .value
                .map(TfxValue::Vector)
                .unwrap_or_else(|| TfxValue::Unknown(expression.expression.clone())),
        );
    }
    let runtime_bindings = bindings
        .iter()
        .map(|binding| TfxRuntimeBinding {
            kind: binding.kind,
            stage: binding.stage,
            slot: binding.slot,
            source: binding.source.clone(),
        })
        .collect();
    for binding in &bindings {
        outputs.insert(
            format!("{} {}", binding.kind, binding.slot),
            TfxValue::TextureBinding {
                stage: binding.stage,
                slot: binding.slot,
            },
        );
    }
    TfxExecutionResult {
        status: program.status,
        outputs,
        bindings: runtime_bindings,
        dependencies,
        trace: program
            .ops
            .iter()
            .map(|op| TfxTraceStep {
                byte_offset: op.offset,
                opcode: op.opcode,
                operation: op.name,
                detail: op.detail.clone(),
            })
            .collect(),
        undecoded_offset: program.undecoded_offset,
        undecoded_bytes: program.undecoded_bytes.clone(),
    }
}

impl TfxValue {
    fn as_vector(&self) -> Option<[f32; 4]> {
        match self {
            Self::Scalar(value) => Some([*value; 4]),
            Self::Vector(value) => Some(*value),
            Self::TextureBinding { .. } | Self::Unknown(_) => None,
        }
    }
}

fn resolve_external(
    inputs: &TfxRuntimeInputs,
    scope_id: u8,
    scope: &str,
    offset: u32,
) -> Option<TfxValue> {
    if let Some(value) = inputs.scoped_externs.get(&(scope_id, offset)) {
        return Some(value.clone());
    }
    match scope {
        "Frame" => inputs
            .frame
            .get(&offset)
            .cloned()
            .or_else(|| matches!(offset, 0 | 4).then_some(TfxValue::Scalar(inputs.time_seconds))),
        "View" => inputs.view.get(&offset).cloned(),
        "RigidModel" | "EditorMesh" => inputs.object_channels.get(&offset).cloned(),
        "GlobalChannel" => inputs.global_channels.get(&offset).cloned(),
        "Gear" | "TextureSet" => inputs.gear_channels.get(&offset).cloned(),
        _ => None,
    }
}

use std::collections::{BTreeMap, HashMap};

use tiger_pkg::TagHash;

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
    pub global_channels: BTreeMap<u32, TfxValue>,
    pub gear_channels: BTreeMap<u32, TfxValue>,
    pub context_values: BTreeMap<u32, TfxValue>,
    /// Typed resource externs keyed by Tiger scope name + byte offset.
    /// Numeric externs stay in the scope maps above; texture/UAV/resource
    /// externs live here so renderer binding state does not collapse into vec4s.
    pub extern_resources: BTreeMap<(String, u32), TagHash>,
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
) -> TfxExecutionResult {
    let dependencies = program
        .externs
        .iter()
        .map(|external| TfxDependency {
            scope_id: external.scope_id,
            scope: external.scope.clone(),
            byte_offset: external.byte_offset,
            resolved: resolve_external(inputs, &external.scope, external.byte_offset as u32)
                .is_some()
                || matches!(external.value_type, "texture" | "uav" | "resource")
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
    let mut context_values = vectors(&inputs.context_values);
    context_values.entry(0).or_insert([inputs.time_seconds; 4]);
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
    extern_values
        .entry(("Frame".to_string(), 0))
        .or_insert([inputs.time_seconds; 4]);
    let (bindings, expressions) = interpret_tfx_stack_with_runtime_values(
        &program.ops,
        constants,
        &object_channels,
        &extern_values,
        &global_channels,
        &context_values,
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

fn resolve_external<'a>(
    inputs: &'a TfxRuntimeInputs,
    scope: &str,
    offset: u32,
) -> Option<&'a TfxValue> {
    match scope {
        "Frame" => inputs.frame.get(&offset),
        "View" => inputs.view.get(&offset),
        "RigidModel" | "EditorMesh" => inputs.object_channels.get(&offset),
        "GlobalChannel" => inputs.global_channels.get(&offset),
        "Gear" | "TextureSet" => inputs.gear_channels.get(&offset),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::TfxBytecodeOpPreview;

    #[test]
    fn runtime_preserves_partial_state_and_trace() {
        let program = TfxBytecodePreview {
            status: TfxDecodeStatus::StoppedAtUnknown,
            undecoded_offset: Some(3),
            undecoded_bytes: vec![0xff],
            ..Default::default()
        };
        let result = execute_preview(&program, &[], &TfxRuntimeInputs::default());
        assert_eq!(result.status, TfxDecodeStatus::StoppedAtUnknown);
        assert_eq!(result.undecoded_bytes, [0xff]);
    }

    #[test]
    fn runtime_re_evaluates_live_frame_and_time_inputs() {
        let program = TfxBytecodePreview {
            ops: vec![
                TfxBytecodeOpPreview {
                    offset: 0,
                    opcode: 0x4a,
                    name: "push_extern_float",
                    detail: "Frame+0x0".into(),
                    extern_scope_id: Some(0),
                },
                TfxBytecodeOpPreview {
                    offset: 3,
                    opcode: 0x53,
                    name: "pop_output",
                    detail: "element=7".into(),
                    extern_scope_id: None,
                },
            ],
            ..Default::default()
        };
        let result = execute_preview(
            &program,
            &[],
            &TfxRuntimeInputs {
                time_seconds: 2.5,
                ..Default::default()
            },
        );
        assert_eq!(
            result.outputs.get("output[7]"),
            Some(&TfxValue::Vector([2.5; 4]))
        );
    }
}

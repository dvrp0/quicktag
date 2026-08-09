use std::collections::BTreeMap;

use crate::material::{TfxBytecodePreview, TfxDecodeStatus};

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

#[derive(Debug, Clone)]
pub struct TfxExecutionResult {
    pub status: TfxDecodeStatus,
    pub outputs: BTreeMap<String, TfxValue>,
    pub dependencies: Vec<TfxDependency>,
    pub trace: Vec<TfxTraceStep>,
    pub undecoded_offset: Option<usize>,
    pub undecoded_bytes: Vec<u8>,
}

pub fn execute_preview(
    program: &TfxBytecodePreview,
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
                .is_some(),
        })
        .collect();
    let mut outputs = BTreeMap::new();
    for expression in &program.expressions {
        outputs.insert(
            expression.target.clone(),
            expression
                .value
                .map(TfxValue::Vector)
                .unwrap_or_else(|| TfxValue::Unknown(expression.expression.clone())),
        );
    }
    for binding in &program.bindings {
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

    #[test]
    fn runtime_preserves_partial_state_and_trace() {
        let program = TfxBytecodePreview {
            status: TfxDecodeStatus::StoppedAtUnknown,
            undecoded_offset: Some(3),
            undecoded_bytes: vec![0xff],
            ..Default::default()
        };
        let result = execute_preview(&program, &TfxRuntimeInputs::default());
        assert_eq!(result.status, TfxDecodeStatus::StoppedAtUnknown);
        assert_eq!(result.undecoded_bytes, [0xff]);
    }
}

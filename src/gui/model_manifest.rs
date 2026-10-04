//! Versioned, CPU-side draw manifest for authored-model investigations.
//!
//! The manifest deliberately records three separate layers:
//! authored package input, Quicktag's resolved renderer plan, and the
//! unresolved GPU-consumption question.  It is diagnostic evidence; it is
//! not a capture of GPU commands or shader reads.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

use serde_json::{Value, json};
use tiger_pkg::version::Version;
use tiger_pkg::{TagHash, package_manager};

use crate::{
    geometry::{
        AuthoredGeometryInput, AuthoredIndexBufferRef, AuthoredInputLayoutDescriptor,
        AuthoredStageInputLayout, AuthoredVertexElementDescriptor,
        AuthoredVertexStreamLayoutDescriptor, AuthoredVertexStreamRef, GeometryPositionTransform,
        UvTransformPreview, WeaponModAttachmentPose, WireframeAuthoredDrawMetadata,
        WireframeMaterialRange, WireframeMaterialTextures, WireframePreview,
    },
    material::TfxDecodeStatus,
    render::{pass_plan::RenderPassKind, technique::ShaderStage, tfx::TfxValue},
};

use super::{GpuModelPreview, LoadedMaterial, ModelPaintCallback, PreparedDraw};

const MANIFEST_SCHEMA: u32 = 1;

/// Journal for a diagnostic capture.  Each state is replaced through a
/// temporary file, so a panic before readback still leaves an explicit
/// failure record instead of stale success metadata.
pub(super) struct CaptureJournal {
    path: PathBuf,
    state: Value,
    phase: String,
    readback_started: bool,
    closed: bool,
}

impl CaptureJournal {
    pub(super) fn begin(path: impl Into<PathBuf>, manifest: &Value) -> io::Result<Self> {
        let path = path.into();
        let manifest_bytes = serde_json::to_vec(manifest).map_err(json_error)?;
        let manifest_attached = manifest["kind"] == "quicktag_draw_manifest";
        let phase = if manifest_attached {
            "manifest_ready"
        } else {
            "case_started"
        };
        let state = json!({
            "schema": MANIFEST_SCHEMA,
            "kind": "quicktag_capture_journal",
            "status": "started",
            "phase": phase,
            "readback_started": false,
            "failure_before_readback": false,
            "manifest_attached": manifest_attached,
            "manifest_sha256": sha256(&manifest_bytes),
            "package": manifest.get("package"),
            "case": manifest.get("case"),
        });
        let journal = Self {
            path,
            state,
            phase: phase.into(),
            readback_started: false,
            closed: false,
        };
        journal.write_state()?;
        Ok(journal)
    }

    /// Attach the complete draw manifest after case setup and material loading.
    /// The journal already exists before those operations, so a panic in setup
    /// remains a pre-readback failure with the case identity intact.
    pub(super) fn attach_manifest(&mut self, manifest: &Value) -> io::Result<()> {
        let manifest_bytes = serde_json::to_vec(manifest).map_err(json_error)?;
        self.phase = "manifest_ready".into();
        self.state["status"] = Value::String("running".into());
        self.state["phase"] = Value::String(self.phase.clone());
        self.state["manifest_attached"] = Value::Bool(true);
        self.state["manifest_sha256"] = Value::String(sha256(&manifest_bytes));
        self.state["package"] = manifest.get("package").cloned().unwrap_or(Value::Null);
        self.write_state()
    }

    pub(super) fn mark_phase(&mut self, phase: &str) -> io::Result<()> {
        self.phase = phase.to_string();
        self.readback_started |= matches!(
            phase,
            "readback_started" | "readback_complete" | "readback_failed"
        );
        self.state["status"] = Value::String("running".into());
        self.state["phase"] = Value::String(self.phase.clone());
        self.state["readback_started"] = Value::Bool(self.readback_started);
        self.state["failure_before_readback"] = Value::Bool(false);
        self.write_state()
    }

    pub(super) fn succeed(&mut self, details: Value) -> io::Result<()> {
        self.closed = true;
        self.phase = "complete".into();
        self.state["status"] = Value::String("success".into());
        self.state["phase"] = Value::String(self.phase.clone());
        self.state["readback_started"] = Value::Bool(self.readback_started);
        self.state["failure_before_readback"] = Value::Bool(false);
        self.state["details"] = details;
        self.write_state()
    }

    pub(super) fn fail(&mut self, phase: &str, error: impl std::fmt::Display) -> io::Result<()> {
        self.closed = true;
        self.phase = phase.to_string();
        self.state["status"] = Value::String("failure".into());
        self.state["phase"] = Value::String(self.phase.clone());
        self.state["readback_started"] = Value::Bool(self.readback_started);
        self.state["failure_before_readback"] = Value::Bool(!self.readback_started);
        self.state["error"] = Value::String(error.to_string());
        self.write_state()
    }

    fn write_state(&self) -> io::Result<()> {
        atomic_write_json(&self.path, &self.state)
    }
}

impl Drop for CaptureJournal {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.fail(
                &self.phase.clone(),
                "capture journal dropped before success/failure was recorded",
            );
        }
    }
}

/// Build a deterministic, authored-vs-resolved manifest for one model
/// callback.  `consumed` stays explicitly unknown: this function runs before
/// command encoding/readback and cannot prove what a GPU shader read.
pub(super) fn build_draw_manifest(
    callback: &ModelPaintCallback,
    wireframe: &WireframePreview,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Value {
    let preview = &callback.preview;
    let visible = callback
        .draws
        .iter()
        .map(|draw| prepared_draw_json(preview, wireframe, draw, false, callback, runtime_inputs))
        .collect::<Vec<_>>();
    let shadows = callback
        .shadow_draws
        .iter()
        .map(|draw| prepared_draw_json(preview, wireframe, draw, true, callback, runtime_inputs))
        .collect::<Vec<_>>();

    json!({
        "schema": MANIFEST_SCHEMA,
        "kind": "quicktag_draw_manifest",
        "package_version": format!("{:?}", package_manager().version),
        "package_build_hashes": "not captured; shader payloads hashed separately",
        "gpu": {
            "bound_by_renderer": "not captured",
            "shader_consumed": "unknown",
            "command_capture": "absent",
        },
        "geometry": {
            "source": wireframe.source,
            "position_format": wireframe.position_format,
            "uv_format": wireframe.uv_format,
            "vertex_count": wireframe.vertex_count_total,
            "index_count": wireframe.index_count_total,
            "bounds_min": wireframe.min,
            "bounds_max": wireframe.max,
            "material_range_count": wireframe.material_ranges.len(),
            "authored_input_count": wireframe.authored_inputs.len(),
            "authored_shadow_range_count": wireframe.authored_shadow_ranges.len(),
        },
        "renderer": renderer_plan(),
        "draws": visible,
        "shadow_draws": shadows,
    })
}

fn prepared_draw_json(
    preview: &GpuModelPreview,
    wireframe: &WireframePreview,
    prepared: &PreparedDraw,
    shadow: bool,
    callback: &ModelPaintCallback,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Value {
    let visible_count = preview.draws.len();
    let (kind, source_draw) = if shadow && prepared.stable_index >= visible_count {
        (
            "authored_shadow",
            preview
                .authored_shadow_draws
                .get(prepared.stable_index - visible_count),
        )
    } else if shadow {
        // No authored shadow ranges means strict shadow selection can reuse a
        // visible draw. Keep that distinction explicit in the manifest.
        (
            "visible_shadow_fallback",
            preview.draws.get(prepared.stable_index),
        )
    } else {
        ("visible", preview.draws.get(prepared.stable_index))
    };
    let authored_input = prepared
        .authored_source
        .and_then(|index| wireframe.authored_inputs.get(index));
    let authored_layout = prepared.authored_stage.and_then(|stage| {
        authored_input?
            .stage_layouts
            .iter()
            .find(|layout| layout.raw_stage == stage.raw_stage)
    });
    let material = callback.materials.get(prepared.material_index);
    let technique = source_draw.and_then(|draw| draw.packet.technique.as_ref());
    let range = source_draw.and_then(|draw| matching_material_range(wireframe, draw));
    let authored_stage = prepared.authored_stage.map(authored_stage_json);
    let resolved = json!({
        "stable_index": prepared.stable_index,
        "material_index": (!prepared.native_c827 && !prepared.native_surface).then_some(prepared.material_index),
        "indices": [prepared.indices.start, prepared.indices.end],
        "passes": prepared.passes.iter().copied().map(render_pass_name).collect::<Vec<_>>(),
        "pipeline": pipeline_json(prepared.pipeline),
        "vertex_input_eligibility": if prepared.native_c827 {
            "exact C827 paired programs; original IA and viewer bind-pose producer; actual encodings in GPU audit"
        } else if prepared.native_surface {
            "exact authored body-VS material; original strip/raw IA/direct GPU producer; native four-MRT source-color consumer"
        } else if prepared.authored_source.is_some()
            && prepared.authored_native_vertex_supported
        {
            "authored_stream_compatibility_eligible; actual binding not captured"
        } else {
            "reconstructed_model_vertex required"
        },
        "authored_native_vertex_supported": prepared.authored_native_vertex_supported,
        "native_surface":prepared.native_surface,
        "material":if prepared.native_surface {
            let spec=source_draw.unwrap().native_surface.as_ref().unwrap();
            let loaded=callback.native_surfaces.iter().find(|s|s.stable_index==prepared.stable_index);
            json!({"material_present":loaded.is_some(),"program":format!("{:?}",spec.program),"group":0,
                "slots":spec.textures.iter().enumerate().map(|(slot,tag)| {
                    let texture=loaded.and_then(|s|s.textures.get(slot));
                    json!({"group":0,"slot":slot+3,"native_slot":slot,"role":"authored PS resource",
                        "selected_tag":tag_string(*tag),"loaded":texture.is_some(),
                        "gpu_texture_identity":texture.map(|t|super::model_binding_audit::identity(&t.handle)),
                        "storage":texture.map(|t|json!({"format":format!("{:?}",t.desc.format),"width":t.desc.width,"height":t.desc.height,"depth":t.desc.depth,"array_size":t.desc.array_size}))})
                }).collect::<Vec<_>>(),"consumer":"existing Base Color inspection; default lit graph remains compatibility"})
        } else {material_plan(material)},
    });

    json!({
        "id": format!("{}-{}", kind, prepared.stable_index),
        "kind": kind,
        "authored": {
            "stable_index": prepared.stable_index,
            "raw_index_range": range.map(material_range_json),
            "raw_lod_category": source_draw.and_then(|draw| draw.packet.raw_lod_category),
            "render_stage": source_draw.and_then(|draw| draw.packet.raw_render_stage),
            "technique": source_draw.and_then(|draw| draw.packet.technique_hash).map(tag_string),
            "authored_source_index": prepared.authored_source,
            "stage": authored_stage,
            "geometry_input": authored_input.map(authored_geometry_json),
            "layout": authored_layout.map(stage_layout_json),
            "procedural_scale": source_draw.map(|draw| draw.procedural_scale),
        },
        "resolved": {
            "technique": technique.map(|technique| technique_json(technique, runtime_inputs)),
            "renderer": resolved,
        },
        "evidence": {
            "provenance": source_draw
                .and_then(|draw| preview.provenance.get(draw.packet.source))
                .map(|record| format!("{:?}", record.evidence)),
            "authored_geometry_owner": authored_input.map(|input| tag_string(input.geometry)),
            "shader_consumption": "unknown",
        },
    })
}

fn matching_material_range<'a>(
    wireframe: &'a WireframePreview,
    draw: &super::ModelDraw,
) -> Option<(usize, &'a WireframeMaterialRange)> {
    wireframe
        .material_ranges
        .iter()
        .enumerate()
        .find(|(_, range)| {
            range.index_start as u32 == draw.indices.start
                && range.index_start.saturating_add(range.index_count) as u32 == draw.indices.end
        })
}

fn material_range_json((index, range): (usize, &WireframeMaterialRange)) -> Value {
    json!({
        "range_index": index,
        "index_start": range.index_start,
        "index_count": range.index_count,
        "raw_lod_category": range.raw_lod_category,
        "render_stage": range.render_stage,
        "technique": range.technique.map(tag_string),
        "authored_source": range.authored_source,
        "authored_draw": range.authored_draw.map(authored_draw_json),
        "procedural_scale": range.procedural_scale,
        "texture": range.texture.map(tag_string),
    })
}

fn authored_draw_json(draw: WireframeAuthoredDrawMetadata) -> Value {
    json!({
        "part_index": draw.part_index,
        "source_index_start": draw.source_index_start,
        "source_index_count": draw.source_index_count,
        "primitive_type": draw.primitive_type,
        "variant_shader_index": draw.variant_shader_index,
        "flags": draw.flags,
        "lod_run": draw.lod_run,
    })
}

fn authored_stage_json(stage: super::AuthoredStageMetadata) -> Value {
    json!({
        "raw_stage": stage.raw_stage,
        "input_layout_id": stage.input_layout_id,
        "part_index": stage.part_index,
        "source_index_start": stage.source_index_start,
        "source_index_count": stage.source_index_count,
        "primitive_type": stage.primitive_type,
        "variant_shader_index": stage.variant_shader_index,
        "flags": stage.flags,
        "lod_run": stage.lod_run,
    })
}

fn authored_geometry_json(input: &AuthoredGeometryInput) -> Value {
    json!({
        "geometry": tag_string(input.geometry),
        "vertex_streams": input.vertex_streams.iter().map(vertex_stream_json).collect::<Vec<_>>(),
        "color_buffer": input.color_buffer.as_ref().map(vertex_stream_json),
        "skinning_buffer": input.skinning_buffer.as_ref().map(vertex_stream_json),
        "index_buffer": input.index_buffer.as_ref().map(index_buffer_json),
        "stage_layouts": input.stage_layouts.iter().map(stage_layout_json).collect::<Vec<_>>(),
        "position_transform": input.position_transform.map(position_transform_json),
        "uv_transform": input.uv_transform.map(uv_transform_json),
        "attachment_pose": input.attachment_pose.map(attachment_pose_json),
    })
}

fn vertex_stream_json(stream: &AuthoredVertexStreamRef) -> Value {
    json!({
        "stream_index": stream.stream_index,
        "header_tag": tag_string(stream.header_tag),
        "data_tag": tag_string(stream.data_tag),
        "stride": stream.stride,
        "vertex_type": stream.vertex_type,
        "data_size": stream.data_size,
        "element_count": stream.element_count,
    })
}

fn index_buffer_json(index: &AuthoredIndexBufferRef) -> Value {
    json!({
        "header_tag": tag_string(index.header_tag),
        "data_tag": tag_string(index.data_tag),
        "is_32bit": index.is_32bit,
        "data_size": index.data_size,
        "index_count": index.index_count,
    })
}

fn stage_layout_json(layout: &AuthoredStageInputLayout) -> Value {
    json!({
        "raw_stage": layout.raw_stage,
        "layout_id": layout.layout_id,
        "descriptor": layout.descriptor.as_ref().map(input_layout_json),
    })
}

fn input_layout_json(layout: &AuthoredInputLayoutDescriptor) -> Value {
    json!({
        "layout_id": layout.layout_id,
        "streams": layout.streams.iter().map(stream_layout_json).collect::<Vec<_>>(),
    })
}

fn stream_layout_json(layout: &AuthoredVertexStreamLayoutDescriptor) -> Value {
    json!({
        "stream_index": layout.stream_index,
        "element_set_index": layout.element_set_index,
        "instanced": layout.instanced,
        "elements": layout.elements.iter().map(element_json).collect::<Vec<_>>(),
    })
}

fn element_json(element: &AuthoredVertexElementDescriptor) -> Value {
    json!({
        "semantic": element.semantic,
        "semantic_index": element.semantic_index,
        "format": element.format,
        "offset": element.offset,
    })
}

fn position_transform_json(transform: GeometryPositionTransform) -> Value {
    json!({
        "scale": transform.scale,
        "offset": transform.offset,
        "procedural_scale": transform.procedural_scale,
    })
}

fn uv_transform_json(transform: UvTransformPreview) -> Value {
    json!({"scale": transform.scale, "offset": transform.offset})
}

fn attachment_pose_json(pose: WeaponModAttachmentPose) -> Value {
    json!({
        "family_id": pose.family_id,
        "variant_id": pose.variant_id,
        "bone_index": pose.bone_index,
        "rotation": pose.rotation,
        "translation": pose.translation,
        "scale": pose.scale,
    })
}

fn technique_json(
    technique: &crate::render::technique::TechniqueDescriptor,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Value {
    json!({
        "technique": tag_string(technique.technique_hash),
        "bind_mode": technique.bind_mode,
        "render_state": {
            "blend": technique.render_state.blend,
            "depth_stencil": technique.render_state.depth_stencil,
            "rasterizer": technique.render_state.rasterizer,
            "depth_bias": technique.render_state.depth_bias,
        },
        "used_scopes": technique.raw_used_scopes,
        "compatible_scopes": technique.raw_compatible_scopes,
        "stages": technique.stages.iter().map(|stage| stage_json(stage, runtime_inputs)).collect::<Vec<_>>(),
    })
}

fn stage_json(
    stage: &crate::render::technique::TechniqueStageDescriptor,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Value {
    let runtime = stage.runtime_state(runtime_inputs);
    json!({
        "stage": stage.raw_stage_label,
        "shader_present": stage.shader.is_some(),
        "coverage_eligible": stage.shader.is_some(),
        "contract_status": {
            "bytecode_decode": format_tfx_status(runtime.status),
            "input_resolution": if stage.shader.is_none() { "not applicable" }
                else if runtime.unresolved_dependencies.is_empty() { "model-preview policy dependencies resolved; required shader inputs not established" }
                else { "unresolved model-preview policy dependencies" },
            "implemented_semantics": if stage.shader.is_none() { "not applicable" } else { "not established" },
            "gpu_consumption": if stage.shader.is_none() { "not applicable" } else { "unknown" },
        },
        "runtime_evaluation": {
            "inputs": "explicit model-preview policy inputs; not live GPU state",
            "constant_registers": runtime.constant_registers,
            "unresolved_dependencies": runtime.unresolved_dependencies,
            "decode_status": format_tfx_status(runtime.status),
            "implemented_shader_semantics": "not established by this manifest",
        },
        "shader": stage
            .shader
            .map(|shader| shader_payload_json(shader, stage.stage)),
        "signature": {
            "texture_count": stage.signature.texture_count,
            "sampler_count": stage.signature.sampler_count,
            "constant_count": stage.signature.constant_count,
            "inline_constant_count": stage.signature.inline_constant_count,
            "tfx_byte_count": stage.signature.tfx_byte_count,
        },
        "resources": stage.resources.iter().map(|resource| json!({
            "slot": resource.slot,
            "raw32": tag_string(resource.raw.raw32),
            "is_hash32": resource.raw.is_hash32,
            "raw64": format!("{:?}", resource.raw.raw64),
            "resolved": resource.resolved.map(tag_string),
            "texture_abi": resource.texture_abi.as_ref().map(|abi| json!({
                "dimension": format!("{:?}", abi.dimension),
                "format": abi.format,
                "color_space": format!("{:?}", abi.color_space),
                "width": abi.width,
                "height": abi.height,
                "depth": abi.depth,
                "array_size": abi.array_size,
            })),
            "required": resource.required,
        })).collect::<Vec<_>>(),
        "indexed_resources": stage.indexed_resources.iter().map(|sampler| json!({
            "ordinal": sampler.ordinal,
            "raw32": tag_string(sampler.raw.raw32),
            "is_hash32": sampler.raw.is_hash32,
            "raw64": format!("{:?}", sampler.raw.raw64),
            "resolved": sampler.resolved.map(tag_string),
        })).collect::<Vec<_>>(),
        "constants": stage.constants,
        "inline_constants": stage.inline_constants,
        "external_constants": stage.external_constants,
        "constant_buffer_slot": stage.constant_buffer_slot,
        "constant_buffer": stage.constant_buffer.map(tag_string),
        "tfx": tfx_json(stage),
        "execution": tfx_execution_json(stage, runtime_inputs),
    })
}

fn shader_payload_json(shader: TagHash, stage: ShaderStage) -> Value {
    let Some(entry) = package_manager().get_entry(shader) else {
        return json!({"tag": tag_string(shader), "payload": "missing"});
    };
    let Ok(data) = package_manager().read_tag(TagHash(entry.reference)) else {
        return json!({"tag": tag_string(shader), "payload": "unreadable"});
    };
    let authored_runtime =
        match crate::render::authored_program::resolve_package_program(shader, stage) {
            Ok(Some(program)) => json!({
                "status": "identity_verified",
                "descriptor_abi": format!("{:?}", program.descriptor_abi),
                "spirv_sha256": hex_sha256(program.spirv_sha256),
                "embedded_bytes": program.spirv.len(),
            }),
            Ok(None) => json!({"status": "unsupported_program_identity"}),
            Err(error) => json!({"status": "payload_error", "error": error}),
        };
    json!({
        "tag": tag_string(shader),
        "byte_len": data.len(),
        "sha256": sha256(&data),
        "authored_runtime": authored_runtime,
    })
}

fn hex_sha256(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Return package identity for all package IDs touched by the selected model.
/// Package files are streamed through SHA-256; no package is loaded wholly
/// into memory.  Caller owns when to place this object into the manifest.
pub(super) fn package_identity_for_manifest(
    callback: &ModelPaintCallback,
    wireframe: &WireframePreview,
) -> Value {
    let mut tags = manifest_tags(callback, wireframe);
    tags.sort_by_key(|tag| tag.0);
    tags.dedup_by_key(|tag| tag.0);
    let package_ids = tags.iter().map(TagHash::pkg_id).collect::<BTreeSet<_>>();
    let manager = package_manager();
    let packages = package_ids
        .iter()
        .map(|package_id| {
            let Some(path) = manager.package_paths.get(package_id) else {
                return json!({
                    "package_id": format!("{package_id:04X}"),
                    "status": "missing_from_lookup",
                });
            };
            match sha256_file(Path::new(&path.path)) {
                Ok((byte_len, digest)) => json!({
                    "package_id": format!("{package_id:04X}"),
                    "status": "hashed",
                    "filename": path.filename,
                    "path": path.path,
                    "byte_len": byte_len,
                    "sha256": digest,
                    "patch_chain": package_patch_identity(path),
                }),
                Err(error) => json!({
                    "package_id": format!("{package_id:04X}"),
                    "status": "hash_failed",
                    "filename": path.filename,
                    "path": path.path,
                    "error": error,
                }),
            }
        })
        .collect::<Vec<_>>();
    json!({
        "version_id": manager.version.id(),
        "version_name": manager.version.name(),
        "engine": format!("{:?}", manager.version.engine_version()),
        "platform": format!("{:?}", manager.platform),
        "package_dir": manager.package_dir,
        "tag_inputs": tags.into_iter().map(tag_string).collect::<Vec<_>>(),
        "packages": packages,
    })
}

fn package_patch_identity(selected: &tiger_pkg::manager::PackagePath) -> Value {
    let Some(directory) = Path::new(&selected.path).parent() else {
        return json!({"status": "directory_unavailable"});
    };
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => return json!({"status": "enumeration_failed", "error": error.to_string()}),
    };
    let mut patches = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                return json!({"status": "enumeration_failed", "error": error.to_string()});
            }
        };
        let entry_path = entry.path();
        if entry_path.extension() != Some(std::ffi::OsStr::new("pkg")) {
            continue;
        }
        let Some(path) = entry_path
            .to_str()
            .and_then(tiger_pkg::manager::PackagePath::parse)
        else {
            continue;
        };
        if path.platform == selected.platform
            && path.name == selected.name
            && path.id == selected.id
            && path.language == selected.language
            && path.patch <= selected.patch
        {
            patches.push(path);
        }
    }
    patches.sort_by_key(|path| path.patch);
    let files = patches.iter().map(|path| match sha256_file(Path::new(&path.path)) {
        Ok((byte_len, digest)) => json!({"patch": path.patch, "path": path.path,
            "byte_len": byte_len, "sha256": digest, "status": "hashed"}),
        Err(error) => json!({"patch": path.patch, "path": path.path, "status": "hash_failed", "error": error}),
    }).collect::<Vec<_>>();
    json!({"status": "enumerated", "selection": "same package family; all present patches through selected patch", "files": files})
}

fn sha256_file(path: &Path) -> Result<(u64, String), String> {
    use sha2::Digest;

    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut byte_len = 0_u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        byte_len += read as u64;
    }
    Ok((byte_len, format!("{:x}", hasher.finalize())))
}

fn manifest_tags(callback: &ModelPaintCallback, wireframe: &WireframePreview) -> Vec<TagHash> {
    let mut tags = Vec::new();
    let preview = &callback.preview;
    for draw in preview
        .draws
        .iter()
        .chain(preview.authored_shadow_draws.iter())
    {
        push_tag(&mut tags, draw.packet.technique_hash);
        if let Some(technique) = draw.packet.technique.as_ref() {
            push_tag(&mut tags, Some(technique.technique_hash));
            for stage in &technique.stages {
                push_tag(&mut tags, stage.shader);
                push_tag(&mut tags, stage.constant_buffer);
                for resource in &stage.resources {
                    push_tag(&mut tags, Some(resource.raw.raw32));
                    push_tag(&mut tags, resource.resolved);
                }
                for sampler in &stage.indexed_resources {
                    push_tag(&mut tags, Some(sampler.raw.raw32));
                    push_tag(&mut tags, sampler.resolved);
                }
            }
        }
    }
    for input in &wireframe.authored_inputs {
        push_tag(&mut tags, Some(input.geometry));
        for stream in input
            .vertex_streams
            .iter()
            .chain(input.color_buffer.iter())
            .chain(input.skinning_buffer.iter())
        {
            push_tag(&mut tags, Some(stream.header_tag));
            push_tag(&mut tags, Some(stream.data_tag));
        }
        if let Some(index) = input.index_buffer.as_ref() {
            push_tag(&mut tags, Some(index.header_tag));
            push_tag(&mut tags, Some(index.data_tag));
        }
    }
    for range in &wireframe.material_ranges {
        push_tag(&mut tags, range.technique);
        push_tag(&mut tags, range.texture);
        collect_material_tags(&range.textures, &mut tags);
    }
    for range in &wireframe.authored_shadow_ranges {
        push_tag(&mut tags, range.technique);
        push_tag(&mut tags, range.texture);
        collect_material_tags(&range.textures, &mut tags);
    }
    tags
}

fn collect_material_tags(textures: &WireframeMaterialTextures, tags: &mut Vec<TagHash>) {
    for tag in [
        textures.color,
        textures.normal,
        textures.emissive,
        textures.control,
        textures.sampler,
    ] {
        push_tag(tags, tag);
    }
    tags.extend(textures.aux.iter().copied());
    for layer in &textures.layers {
        for tag in [layer.color, layer.normal, layer.emissive] {
            push_tag(tags, tag);
        }
    }
    if let Some(character) = textures.character_surface {
        for tag in [
            Some(character.surface),
            Some(character.selector),
            Some(character.detail_color),
            Some(character.detail_normal),
            character.procedural,
        ] {
            push_tag(tags, tag);
        }
    }
    if let Some(runner) = textures.runner_layered_surface {
        for tag in [
            Some(runner.surface),
            runner.material_response,
            Some(runner.detail_normal_a),
            Some(runner.detail_normal_b),
            runner.detail_normal_c,
            runner.detail_normal_d,
            runner.procedural,
            runner.color_overlay,
        ] {
            push_tag(tags, tag);
        }
        if let Some(wear) = runner.procedural_wear {
            tags.extend(wear);
        }
    }
    push_tag(
        tags,
        textures.runner_occlusion.map(|occlusion| occlusion.texture),
    );
    push_tag(tags, textures.alpha_mask.map(|mask| mask.texture));
    push_tag(
        tags,
        textures.shared_atlas_detail.map(|detail| detail.detail),
    );
    if let Some(wear) = textures.mod_wear {
        tags.extend([wear.scratches, wear.grime, wear.damage]);
    }
    if let Some(condition) = textures.surface_condition {
        tags.extend([condition.response, condition.detail, condition.breakup]);
    }
    push_tag(tags, textures.gear_pattern.map(|pattern| pattern.field));
    if let Some(decal) = textures.investment_decal {
        tags.extend([decal.color, decal.mask]);
        push_tag(tags, decal.detail);
    }
    if let Some(coating) = textures.forward_coating {
        tags.extend([
            coating.detail,
            coating.environment,
            coating.environment_sampler,
        ]);
    }
}

fn push_tag(tags: &mut Vec<TagHash>, tag: Option<TagHash>) {
    if let Some(tag) = tag {
        tags.push(tag);
    }
}

fn tfx_json(stage: &crate::render::technique::TechniqueStageDescriptor) -> Value {
    let tfx = &stage.tfx;
    json!({
        "total_bytes": tfx.total_bytes,
        "decoded_ops": tfx.decoded_ops,
        "unknown_ops": tfx.unknown_ops,
        "status": format_tfx_status(tfx.status),
        "truncated": tfx.truncated,
        "undecoded_offset": tfx.undecoded_offset,
        "undecoded_bytes": tfx.undecoded_bytes.len(),
        "ops": tfx.ops.iter().map(|op| json!({
            "offset": op.offset,
            "opcode": op.opcode,
            "name": op.name,
            "detail": op.detail,
            "extern_scope_id": op.extern_scope_id,
        })).collect::<Vec<_>>(),
        "bindings": tfx.bindings.iter().map(|binding| json!({
            "kind": binding.kind,
            "stage": binding.stage,
            "slot": binding.slot,
            "source": binding.source,
        })).collect::<Vec<_>>(),
        "expressions": tfx.expressions.iter().map(|expression| json!({
            "op_offset": expression.op_offset,
            "target": expression.target,
            "expression": expression.expression,
            "value": expression.value,
        })).collect::<Vec<_>>(),
        "externs": tfx.externs.iter().map(|external| json!({
            "op_offset": external.op_offset,
            "scope_id": external.scope_id,
            "value_type": external.value_type,
            "scope": external.scope,
            "byte_offset": external.byte_offset,
            "hint": external.hint,
        })).collect::<Vec<_>>(),
        "constant_refs": tfx.constant_refs.iter().map(|reference| json!({
            "op_offset": reference.op_offset,
            "op_name": reference.op_name,
            "start": reference.start,
            "count": reference.count,
            "values": reference.values,
        })).collect::<Vec<_>>(),
    })
}

fn tfx_execution_json(
    stage: &crate::render::technique::TechniqueStageDescriptor,
    runtime_inputs: &crate::render::tfx::TfxRuntimeInputs,
) -> Value {
    let execution = stage.execute(runtime_inputs);
    json!({
        "status": format_tfx_status(execution.status),
        "inputs": "explicit model-preview policy inputs; not live GPU state",
        "outputs": execution.outputs.iter().map(|(target, value)| {
            (target.clone(), tfx_value_json(value))
        }).collect::<serde_json::Map<_, _>>(),
        "bindings": execution.bindings.iter().map(|binding| json!({
            "kind": binding.kind,
            "stage": binding.stage,
            "slot": binding.slot,
            "source": binding.source,
        })).collect::<Vec<_>>(),
        "dependencies": execution.dependencies.iter().map(|dependency| json!({
            "scope_id": dependency.scope_id,
            "scope": dependency.scope,
            "byte_offset": dependency.byte_offset,
            "resolved": dependency.resolved,
        })).collect::<Vec<_>>(),
        "undecoded_offset": execution.undecoded_offset,
        "undecoded_bytes": execution.undecoded_bytes.len(),
    })
}

fn tfx_value_json(value: &TfxValue) -> Value {
    match value {
        TfxValue::Scalar(value) => json!({"kind": "scalar", "value": value}),
        TfxValue::Vector(value) => json!({"kind": "vector", "value": value}),
        TfxValue::TextureBinding { stage, slot } => {
            json!({"kind": "texture_binding", "stage": stage, "slot": slot})
        }
        TfxValue::Unknown(expression) => json!({"kind": "unknown", "expression": expression}),
    }
}

fn format_tfx_status(status: TfxDecodeStatus) -> &'static str {
    match status {
        TfxDecodeStatus::Complete => "Complete",
        TfxDecodeStatus::Partial => "Partial",
        TfxDecodeStatus::StoppedAtUnknown => "StoppedAtUnknown",
        TfxDecodeStatus::Invalid => "Invalid",
    }
}

fn pipeline_json(pipeline: super::ModelPipelineKey) -> Value {
    json!({
        "blend": pipeline.blend,
        "depth_stencil": pipeline.depth_stencil,
        "rasterizer": pipeline.rasterizer,
        "depth_bias": pipeline.depth_bias,
        "family": format!("{:?}", pipeline.family),
        "pass": render_pass_name(pipeline.pass),
        "technique": pipeline.technique.map(tag_string),
        "vertex_layout_bytes": pipeline.vertex_layout_bytes,
        "target_signature": pipeline.target_signature,
    })
}

fn material_plan(material: Option<&LoadedMaterial>) -> Value {
    let slots = [
        (
            0,
            "color",
            material.and_then(|material| material.key.and_then(|key| key.color)),
            material.and_then(|material| material.color.as_ref()),
        ),
        (
            1,
            "normal",
            material.and_then(|material| material.key.and_then(|key| key.normal)),
            material.and_then(|material| material.normal.as_ref()),
        ),
        (
            2,
            "emissive",
            material.and_then(|material| material.key.and_then(|key| key.emissive)),
            material.and_then(|material| material.emissive.as_ref()),
        ),
        (
            3,
            "control",
            material.and_then(|material| material.control_tag),
            material.and_then(|material| material.control.as_ref()),
        ),
        (
            6,
            "wear_scratches",
            None,
            material.and_then(|material| material.wear_scratches.as_ref()),
        ),
        (
            7,
            "wear_grime",
            None,
            material.and_then(|material| material.wear_grime.as_ref()),
        ),
        (
            8,
            "wear_damage",
            None,
            material.and_then(|material| material.wear_damage.as_ref()),
        ),
        (
            9,
            "pattern_or_runner_procedural",
            None,
            material.and_then(|material| {
                material
                    .pattern_field
                    .as_ref()
                    .or(material.runner_procedural_map.as_ref())
            }),
        ),
        (
            10,
            "character_surface",
            None,
            material.and_then(|material| material.character_surface_map.as_ref()),
        ),
        (
            11,
            "character_detail_color",
            None,
            material.and_then(|material| {
                material
                    .character_detail_color
                    .as_ref()
                    .or(material.runner_color_overlay_map.as_ref())
            }),
        ),
        (
            12,
            "procedural_or_response",
            None,
            material.and_then(|material| {
                material
                    .character_procedural_map
                    .as_ref()
                    .or(material.runner_material_response_map.as_ref())
                    .or(material.runner_occlusion_map.as_ref())
            }),
        ),
        (
            13,
            "runner_surface",
            None,
            material.and_then(|material| material.runner_surface_map.as_ref()),
        ),
        (
            14,
            "runner_detail_normal_a",
            None,
            material.and_then(|material| material.runner_detail_normal_a.as_ref()),
        ),
        (
            15,
            "runner_detail_normal_b",
            None,
            material.and_then(|material| material.runner_detail_normal_b.as_ref()),
        ),
        (
            16,
            "runner_detail_normal_c",
            None,
            material.and_then(|material| material.runner_detail_normal_c.as_ref()),
        ),
        (
            17,
            "runner_detail_normal_d",
            None,
            material.and_then(|material| material.runner_detail_normal_d.as_ref()),
        ),
        (
            18,
            "coating_environment",
            material
                .and_then(|material| material.forward_coating.map(|coating| coating.environment)),
            material.and_then(|material| material.coating_environment_map.as_ref()),
        ),
    ];
    json!({
        "material_present": material.is_some(),
        "slots": slots.into_iter().map(|(slot, role, tag, loaded)| {
            json!({
                "group": 1,
                "slot": slot,
                "role": role,
                "selected_tag": tag.map(tag_string),
                "loaded": loaded.is_some(),
                "gpu_texture_identity": loaded.map(|texture| super::model_binding_audit::identity(&texture.handle)),
                "storage": loaded.map(|texture| json!({
                    "format": format!("{:?}", texture.desc.format),
                    "width": texture.desc.width,
                    "height": texture.desc.height,
                    "depth": texture.desc.depth,
                    "array_size": texture.desc.array_size,
                })),
            })
        }).collect::<Vec<_>>(),
    })
}

fn renderer_plan() -> Value {
    let group0 = [
        (0, "scene_uniform"),
        (1, "shadow_depth"),
        (2, "shadow_comparison_sampler"),
        (3, "environment_cube"),
        (4, "environment_sampler"),
    ];
    let group1 = [
        (0, "color"),
        (1, "normal"),
        (2, "emissive"),
        (3, "control"),
        (4, "material_sampler"),
        (5, "material_uniform"),
        (6, "wear_scratches"),
        (7, "wear_grime"),
        (8, "wear_damage"),
        (9, "pattern_or_runner_procedural"),
        (10, "character_surface"),
        (11, "character_detail_color"),
        (12, "procedural_or_response"),
        (13, "runner_surface"),
        (14, "runner_detail_normal_a"),
        (15, "runner_detail_normal_b"),
        (16, "runner_detail_normal_c"),
        (17, "runner_detail_normal_d"),
        (18, "coating_environment"),
        (19, "coating_sampler"),
    ];
    json!({
        "vertex": {
            "reconstructed": [
                [0, "POSITION0"],
                [1, "NORMAL0"],
                [2, "TEXCOORD0"],
                [3, "TANGENT0"],
                [4, "AO0"],
                [5, "PROCEDURAL_POSITION0"],
                [6, "PROCEDURAL_NORMAL0"],
            ],
            "authored_stream_read": "stage/layout dependent",
        },
        "groups": [
            {"group": 0, "bindings": group0.into_iter().map(|(slot, role)| json!({"slot": slot, "role": role})).collect::<Vec<_>>()},
            {"group": 1, "bindings": group1.into_iter().map(|(slot, role)| json!({"slot": slot, "role": role})).collect::<Vec<_>>()},
            {"group": 2, "bindings": [{"slot": 0, "role": "coating_or_authored_transform", "pass_dependent": true}]},
        ],
    })
}

fn render_pass_name(pass: RenderPassKind) -> String {
    format!("{pass:?}")
}

fn tag_string(tag: TagHash) -> String {
    tag.to_string()
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn json_error(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn atomic_write_json(path: &Path, value: &Value) -> io::Result<()> {
    let data = serde_json::to_vec_pretty(value).map_err(json_error)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&temporary, data)?;
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "quicktag-model-{label}-{}-{}.json",
            std::process::id(),
            sha256(label.as_bytes())
        ))
    }

    #[test]
    fn package_identity_includes_older_payload_patches() {
        let directory = test_path("patch-chain").with_extension("directory");
        fs::create_dir_all(&directory).unwrap();
        let filenames = [
            "w64_sr_gear_014f_0.pkg",
            "w64_sr_gear_014f_5.pkg",
            "w64_sr_gear_014f_6.pkg",
            "w64_sr_gear_0150_0.pkg",
            "w64_other_014f_0.pkg",
        ];
        for filename in filenames {
            fs::write(directory.join(filename), filename).unwrap();
        }
        let selected =
            tiger_pkg::manager::PackagePath::parse(directory.join(filenames[1]).to_str().unwrap())
                .unwrap();
        let before = package_patch_identity(&selected);
        let files = before["files"].as_array().unwrap();
        assert_eq!(
            files.len(),
            2,
            "exclude future patches and other package families"
        );
        assert_eq!(files[0]["patch"], 0);
        assert_eq!(files[1]["patch"], 5);
        fs::write(directory.join(filenames[0]), b"changed base payload").unwrap();
        let after = package_patch_identity(&selected);
        assert_ne!(before["files"][0]["sha256"], after["files"][0]["sha256"]);
        assert_eq!(before["files"][1]["sha256"], after["files"][1]["sha256"]);
        for filename in filenames {
            fs::remove_file(directory.join(filename)).unwrap();
        }
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn capture_journal_records_failure_before_readback() {
        let path = test_path("failure-before-readback");
        let _ = fs::remove_file(&path);
        let manifest = json!({"package": {"version_id": "marathon"}});
        let mut journal = CaptureJournal::begin(&path, &manifest).expect("journal start");
        journal.mark_phase("bind_groups").expect("journal phase");
        journal
            .fail("bind_groups", "bind group descriptor rejected")
            .expect("journal failure");

        let recorded: Value =
            serde_json::from_slice(&fs::read(&path).expect("journal file")).expect("journal JSON");
        assert_eq!(recorded["status"], "failure");
        assert_eq!(recorded["phase"], "bind_groups");
        assert_eq!(recorded["failure_before_readback"], true);
        assert_eq!(recorded["readback_started"], false);
        assert!(recorded["error"].as_str().unwrap().contains("rejected"));
        fs::remove_file(path).expect("cleanup journal");
    }

    #[test]
    fn capture_journal_attaches_manifest_after_case_start() {
        let path = test_path("attach-manifest");
        let _ = fs::remove_file(&path);
        let case = json!({
            "kind": "quicktag_capture_case",
            "case": {"name": "runner-thief-cryo-shift-combined"},
        });
        let mut journal = CaptureJournal::begin(&path, &case).expect("journal start");
        let started: Value = serde_json::from_slice(&fs::read(&path).expect("initial journal"))
            .expect("journal JSON");
        assert_eq!(started["phase"], "case_started");
        assert_eq!(started["manifest_attached"], false);

        let manifest = json!({
            "kind": "quicktag_draw_manifest",
            "package": {"version_id": "marathon", "build": "fixture"},
        });
        journal
            .attach_manifest(&manifest)
            .expect("attach draw manifest");
        let attached: Value = serde_json::from_slice(&fs::read(&path).expect("attached journal"))
            .expect("journal JSON");
        assert_eq!(attached["phase"], "manifest_ready");
        assert_eq!(attached["manifest_attached"], true);
        assert_eq!(attached["package"]["build"], "fixture");
        journal
            .fail("test", "close attach test")
            .expect("journal close");
        drop(journal);
        let mut replacement =
            CaptureJournal::begin(&path, &case).expect("replace existing journal atomically");
        let replaced: Value =
            serde_json::from_slice(&fs::read(&path).expect("replacement journal"))
                .expect("journal JSON");
        assert_eq!(replaced["status"], "started");
        replacement
            .fail("test", "close replacement test")
            .expect("replacement close");
        fs::remove_file(path).expect("cleanup journal");
    }

    #[test]
    fn capture_journal_records_success_after_readback() {
        let path = test_path("success-after-readback");
        let _ = fs::remove_file(&path);
        let manifest = json!({"package": {"version_id": "marathon"}});
        let mut journal = CaptureJournal::begin(&path, &manifest).expect("journal start");
        journal
            .mark_phase("readback_started")
            .expect("readback start");
        journal
            .mark_phase("readback_complete")
            .expect("readback complete");
        journal
            .succeed(json!({"image": "thief-final.png"}))
            .expect("journal success");

        let recorded: Value =
            serde_json::from_slice(&fs::read(&path).expect("journal file")).expect("journal JSON");
        assert_eq!(recorded["status"], "success");
        assert_eq!(recorded["phase"], "complete");
        assert_eq!(recorded["failure_before_readback"], false);
        assert_eq!(recorded["readback_started"], true);
        assert_eq!(recorded["details"]["image"], "thief-final.png");
        fs::remove_file(path).expect("cleanup journal");
    }

    #[test]
    fn sha256_is_stable_for_shader_payload_fixture() {
        assert_eq!(
            sha256(b"DXIL"),
            "ed6fae985241493bc3b6860d6614ba5bbb3e719305c40f8189fca57896fa7e65"
        );
    }
}

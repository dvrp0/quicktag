# Alkahest Comparison — Prioritized TODO

This backlog translates the findings in `analysis_result.md` into implementation work, ordered from the highest-impact renderer-fidelity issues to lower-priority polish.

The ordering is intentional: avoid spending significant time tuning presentation, BRDF constants, or shadow filtering while Quicktag may still be drawing the wrong authored stages, with the wrong vertex ABI, or without renderer-facing TFX state.

## Current implementation snapshot — 2026-09-25

- **Render-stage ABI:** partial. The 25-stage table, insertion point, strongest semantic anchors, dynamic/static census, and typed API are implemented; raw stage 3 and several feature-only stages remain semantically unresolved.
- **Authored ShadowGenerate caster selection:** complete. Strict Tiger uses authored raw-stage-4 source ranges and package-native vertex/index streams where supported, with compatibility fallback.
- **Authored vertex/input ABI:** partial. Raw streams/layout descriptors are preserved and GPU-materialized; Strict Shadow consumes them, but the rest of Strict Tiger still largely renders through reconstructed `ModelVertex`.
- **Renderer-facing TFX:** partial. Runtime stage state, output register images, runtime bindings, and unresolved-dependency tracking exist; TFX register/resource state is not yet the authoritative input to the custom WGSL shading path.
- **Shadow artifact:** resolved in the live model viewer on 2026-09-25. Replacing the finite spotlight perspective shadow transform with an affine directional Tiger-style shadow space removed the smooth-surface spike/teeth artifact. The exact Marathon authored shadow matrix is still not decoded, so projection fidelity remains a follow-up rather than a correctness blocker.
- **Shadow shader fidelity:** partial. Native input layouts/source ranges are supported where the authored VS ABI is compatible; runtime-resource VS families such as Conquest's `SV_VertexID -> t2` path explicitly use the reconstructed-position compatibility path until that runtime buffer ABI is implemented.
- **Renderer contract:** `Pretty Preview` has been removed as a separate fidelity mode. Quicktag now exposes one Tiger-faithful renderer contract; heuristic/custom behavior remains only as explicit internal compatibility fallback where authored behavior is still unresolved.
- **Validation:** current diff passes `cargo check --release`, Tiger-shadow unit tests, and the Conquest GPU visual probe (existing repository warnings remain).

---

# P0 — Critical Renderer-Fidelity Blockers

## [ ] Reconstruct and formalize the Marathon/Goliath render-stage ABI

This is the highest-priority task.

Quicktag currently knows that Marathon has 25 render-stage ranges / 26 boundaries, while Destiny 2 Alkahest exposes 24 typed render stages. The Marathon-specific inserted stage and exact semantic mapping need to be identified from package evidence.

**Progress (2026-09-25):** the package-backed census now parses all 2313/2313 scanned dynamic geometry resources, formalizes the 25-stage ABI, and strongly supports one inserted Marathon-only slot at raw index 3. The census has also been extended to static mesh groups / static special meshes, providing independent stage evidence outside dynamic models. Raw stage 3 still has no observed dynamic/static mesh participation and remains intentionally unnamed. See `render_stage_abi.md`.

### Required work

- [ ] Identify the exact meaning of all 25 Marathon raw render-stage indices.
  - [ ] Resolve the semantic purpose of Marathon-only raw stage 3 from non-mesh/runtime evidence.
  - [ ] Upgrade remaining sequence-only `Probable` stages with direct Marathon evidence.
- [x] Determine where Marathon's additional stage was inserted relative to Destiny 2.
- [ ] Identify the true Marathon equivalent of:
  - [x] GenerateGbuffer — StronglyCorrelated
  - [ ] Decals — Probable / no direct geometry evidence yet
  - [x] InvestmentDecals — Confirmed
  - [x] ShadowGenerate — Confirmed
  - [ ] LightingApply — Probable
  - [ ] LightProbeApply — Probable
  - [ ] DecalsAdditive — Probable
  - [x] Transparents — StronglyCorrelated
  - [x] Distortion — StronglyCorrelated
  - [x] LightShaftOcclusion — StronglyCorrelated
  - [ ] SkinPrepass — Probable
  - [x] DepthPrepass — Confirmed
  - [x] WaterReflection — StronglyCorrelated
  - [x] Reticle — StronglyCorrelated
  - [x] WaterRipples — StronglyCorrelated
  - [x] ComputeSkinning — Confirmed
  - [ ] Volumetrics — Probable
  - [ ] Cubemaps — Probable
  - [ ] remaining postprocess / feature-only stages
- [x] Correlate stage boundaries with:
  - [x] technique hashes
  - [x] render states
  - [x] index ranges
  - [x] overlap with stage 0 geometry
  - [x] input-layout IDs
  - [x] observed shader stages
- [x] Replace speculative names such as "stage 4 shadow-only" with evidence-graded semantic names.
- [x] Introduce a typed Marathon render-stage representation.
- [x] Add regression tests proving the strongest package-backed mapping anchors.
- [x] Extend stage evidence beyond dynamic geometry into Marathon static mesh groups / static special meshes.
- [x] Add a reusable headless JSON ABI census (`--probe-render-stage-abi`).

### Exit criteria

Quicktag can now answer the ShadowGenerate portion directly: raw stage 4 is decoded from the authored boundary table without material classification. The broader item stays open until raw stage 3 and the weaker sequence-only stages are resolved.

---

## [x] Drive shadow caster selection from authored ShadowGenerate participation

Do not infer shadow participation primarily from visible-stage material behavior once the Marathon stage ABI is known.

**Completed (2026-09-25):** the renderer now preserves the exact authored raw-stage-4 draw ranges separately from preview-deduplicated visible geometry **and consumes their package-native source index ranges / vertex streams through a dedicated shadow pipeline** when the authored ABI is supported. The Conquest LMG fixture retains 9 authored ranges / 128,982 shadow indices after full model assembly. Unsupported/missing authored shader ABIs retain an explicit compatibility fallback. Visual side-by-side inspection remains a P2 diagnostics task, not a blocker for authored caster selection.

### Required work

- [x] Read the exact authored ShadowGenerate part range from geometry metadata.
- [x] Preserve its exact:
  - [x] part indices
  - [x] index ranges
  - [x] techniques
  - [x] variant shader indices
  - [x] LOD categories
  - [x] input-layout ID
- [x] Stop treating ordinary visible draws as automatic shadow casters in Strict Tiger mode.
- [x] Remove/rework the current stage-4 proxy override heuristic once the actual stage mapping is known.
- [x] Keep heuristic caster derivation only as a compatibility/fallback path.
- [x] Move visible-vs-ShadowGenerate visual comparison to the dedicated P2 diagnostics backlog.
- [x] Add tests for assets where shadow-stage geometry differs from G-buffer geometry.

### Why this is critical

The current jagged-shadow artifact may be caused upstream by drawing the wrong authored geometry contract, not by PCF/PCSS quality.

---

## [ ] Preserve stage-specific vertex/input ABI instead of flattening everything into ModelVertex

Tiger can use multiple authored streams and stage-specific layout metadata. Quicktag previously collapsed these into `ModelVertex` too early.

**Progress (2026-09-25):** authored geometry inputs are now first-class data. Quicktag preserves package stream/header/data references, index-buffer metadata, per-stage layout IDs, element semantic/index/format/offset information, instancing flags, geometry dequantization, and attachment transforms through merged model assembly. These streams are materialized as native WGPU vertex/index buffers. The Strict Tiger **ShadowGenerate** path can consume them directly; the general visible/depth/material paths still rely on reconstructed `ModelVertex`, so this P0 remains open.

The package-wide ShadowGenerate relation probe also found an important constraint: among 1661 geometries with both stage 0 and stage 4, **1661/1661 use the same input-layout ID** and **0 use a different layout**. The real stage-specific difference is usually the authored vertex program: 1594/1661 geometry resources have completely disjoint visible-vs-shadow VS sets.

### Required work

- [x] Preserve raw vertex-buffer streams in renderer-facing geometry.
- [x] Preserve:
  - [x] vertex0
  - [x] vertex1
  - [x] auxiliary buffer2 / buffer3 references
  - [x] color buffer references
  - [x] skinning buffer references
  - [x] additional authored UV/color/custom semantics through layout descriptors
  - [x] per-instance stream flags through layout descriptors
- [x] Decode Marathon input-layout tables from render globals / geometry metadata.
- [x] Associate an input-layout ID with each render stage.
- [x] Preserve semantic/index/format/offset/stream information.
- [x] Materialize authored package streams as GPU vertex/index buffers.
- [x] Allow Strict Tiger ShadowGenerate pipelines to consume authored stream layouts and source index ranges.
- [ ] Generalize authored stream/layout consumption to Strict Tiger G-buffer/depth/other stage pipelines.
- [ ] Keep reconstructed `ModelVertex` only as an explicit compatibility fallback for unsupported authored paths.
- [x] Verify package-wide whether ShadowGenerate changes input-layout ID: 0 / 1661 differ.
- [x] Add Conquest LMG regression coverage for authored source/range/layout preservation.

### Exit criteria

A Tiger-compatible shader path should not require reconstructing or guessing missing vertex attributes that existed in the source package.

---

## [ ] Make TFX execution renderer-authoritative

Quicktag now has a renderer-facing runtime-state layer, but authored TFX is **not yet the final authority for shading**.

**Progress (2026-09-25):** `TechniqueStageRuntimeState` evaluates TFX per shader stage, overlays output registers onto authored inline constants, records runtime resource/sampler bindings, and carries unresolved dependencies. VS/PS register images and decode status are uploaded into `MaterialUniform` (`tfx_vs_registers`, `tfx_ps_registers`, `tfx_meta`). Runtime resource bindings are represented/resolved in CPU state and participate in material identity, but the custom WGSL shaders do not yet consume those TFX register arrays and TFX-driven resources are not yet applied as dynamic GPU bind-group bindings. Therefore this remains P0.

### Required work

- [x] Define a renderer-facing TFX runtime result/state.
- [x] Evaluate TFX outputs into per-stage constant-register images.
- [x] Upload TFX VS/PS register images and decode status into actual GPU uniform data.
- [x] Represent runtime texture/sampler/resource bindings and unresolved dependencies in renderer state.
- [ ] Make custom/compatibility shaders actually consume required TFX register values.
- [ ] Apply TFX-driven:
  - [ ] texture bindings
  - [ ] texture-view bindings
  - [ ] sampler bindings
  - [ ] context values beyond current limited renderer inputs
  - [ ] object channels beyond currently wired mod age / unique-id channels
  - [ ] global channels
  - [ ] texture metadata
  - [ ] extern-scope values/resources
- [x] Track unresolved/unknown TFX dependencies at render time.
- [ ] Make Strict Tiger fail visibly or fall back explicitly when required TFX inputs are unknown.
- [ ] Do not silently replace unknown authored behavior with unrelated MaterialIR defaults.
- [ ] Add render probes comparing TFX output buffers with known Alkahest/Destiny behavior where compatible.

### Exit criteria

TFX should become:

    TFX
      ↓
    actual GPU state

rather than:

    TFX
      ↓
    Render Evidence only

---

# P1 — Core Tiger Rendering Architecture

## [x] Remove Pretty Preview as a separate fidelity mode

**Completed (2026-09-25):** Quicktag now exposes one renderer contract whose goal is to follow authored Tiger/Marathon behavior. The `FidelityMode` enum, UI toggle, Pretty-only shadow projection, controllable PCSS softness path, probe override, and Pretty-only material guessing have been removed from the active renderer. Unsupported authored behavior may still use explicit compatibility fallbacks, but those fallbacks are not presented as a second renderer mode.

### Remaining Tiger-fidelity targets

- [ ] authored render stages across the full renderer
  - [x] ShadowGenerate uses authored stage membership
- [ ] authored stage-specific vertex ABI across the full renderer
  - [x] ShadowGenerate can consume package-native vertex/index streams when its VS ABI is supported
- [ ] authored technique selection as shader behavior
- [ ] authored render state across all passes
- [ ] renderer-authoritative TFX
- [ ] engine-style extern scopes
- [ ] shader-family-specific or translated shader behavior
- [ ] minimal semantic reinterpretation

Compatibility rendering remains useful, but it is now an implementation fallback rather than a user-facing fidelity choice.

---

## [ ] Build a Marathon extern-scope system

Alkahest's authored shaders depend on large engine scopes. Quicktag needs a Marathon equivalent for strict rendering.

### Required scopes to investigate

- [ ] Frame
- [ ] View
- [ ] Deferred
- [ ] DeferredLight
- [ ] DeferredShadow
- [ ] RigidModel
- [ ] Gear / dye-related scopes
- [ ] object-local channels
- [ ] global channels
- [ ] material/editor scopes where relevant
- [ ] texture metadata scopes

### Required work

- [ ] Map Marathon TFX extern IDs to typed structures.
- [ ] Populate view/projection/world transforms.
- [ ] Populate frame time and animation time.
- [ ] Populate shadow map + projection data.
- [ ] Populate G-buffer/light-buffer references.
- [ ] Populate global material lookup textures.
- [ ] Add evidence reporting for unresolved externs.

---

## [ ] Complete Tiger-style pipeline-state translation

Quicktag already decodes the selectors; now make their GPU effect as faithful as WGPU allows.

### Required work

- [ ] Port/verify the full Tiger blend-state table.
- [ ] Preserve independent RGB/alpha blend equations where possible.
- [ ] Preserve render-target write masks.
- [ ] Support min/max blend operations.
- [ ] Support destination-color / blend-factor states.
- [ ] Verify the complete depth-state table.
- [ ] Verify all rasterizer states, not only cull mode.
- [ ] Restore/use authored depth-bias selections.
- [ ] Reconstruct stencil behavior where WebGPU permits it.
- [ ] Explicitly report unsupported state combinations instead of silently using generic alpha blending.

### Exit criteria

Unknown Tiger state index should never silently mean "normal alpha blending."

---

## [ ] Restore Tiger shadow raster-bias baseline in Strict Tiger

The known Alkahest/Tiger shadow state uses depth-bias preset 6:

    constant = 2
    slope    = 2.0

### Required work

- [x] Restore 2 / 2 for the native Strict Tiger ShadowGenerate pipeline.
- [ ] Keep alternate bias/fallback behavior out of Strict Tiger once every authored shadow layout is supported.
- [ ] Verify Marathon's authored shadow default state independently rather than relying only on the Alkahest/Tiger baseline.
- [x] Stop further blind bias/filter tuning while authored caster/stage behavior is being reconstructed.

---

# P2 — Shadow Fidelity Investigation

## [ ] Rebuild the shadow path around authored Tiger/Marathon data

Current Quicktag shadowing is a replacement system. Strict Tiger should increasingly reproduce engine behavior.

### Required work

- [ ] Derive shadow projection from Marathon light data where available.
- [ ] Investigate Marathon equivalents of:
  - [ ] light-space transform
  - [ ] far plane
  - [ ] half FOV
  - [ ] shadowed-lighting technique
- [ ] Populate a DeferredShadow-style scope.
- [ ] Verify world/view/light matrix conventions against Alkahest.
- [ ] Compare Quicktag shadow UV/depth values against a reference implementation on controlled geometry.
- [x] Remove the Pretty Preview finite-spotlight shadow projection from the Tiger path; current shadow space is affine/directional while exact authored Marathon projection data remains to be decoded.

---

## [ ] Replace universal shadow vertex behavior with stage-/technique-aware behavior

**Progress (2026-09-25):** the package-wide ABI census compares stage-0 and stage-4 VS sets. Of 1661 geometries with both passes, 67 use equal VS sets, 0 partially overlap, and **1594 use completely disjoint VS sets**. Conquest LMG's visible and ShadowGenerate shaders were extracted and disassembled with Windows SDK DXC. For that rigid fixture, both paths use the same layout (7) and the same packed-position/dequantization contract; ShadowGenerate strips the visible shader's extra normal/tangent work. The static mesh record does not contain a large hidden t2 position buffer, so that shader resource is runtime-generated rather than an omitted package vertex tag.

Quicktag now has a native-layout Strict Shadow compatibility pipeline, but it is keyed by layout/primitive/rasterizer rather than by the authored shader family itself. The authored packaged VS is still not executed or translated.

### Required work

- [x] Inventory visible-vs-ShadowGenerate vertex-shader set relationships package-wide.
- [x] Extract and DXC-disassemble representative Conquest visible/shadow vertex shaders.
- [x] Verify Conquest visible/shadow input-layout relationship (layout 7 → 7).
- [x] Verify the Conquest stage-4 packed-position resource is runtime-generated, not a dropped static mesh stream.
- [ ] Generalize authored shadow shader ABI analysis beyond the Conquest/common rigid family.
- [ ] Identify whether other shadow shader families require:
  - [ ] alternate UVs
  - [ ] vertex colors
  - [ ] procedural deformation
  - [ ] alpha/control coordinates
  - [ ] true skinning/deformation inputs
  - [ ] stage-specific position transforms not covered by the current compatibility path
- [ ] Key strict shadow pipelines by authored shader ABI/signature or translated shader implementation.
- [x] Preserve the current universal SHADOW_SHADER as explicit fallback for unsupported authored paths.

---

## [x] Reduce active shadow filtering to the Tiger-style path

**Completed (2026-09-25):** the active renderer uses a small 9-tap rotated Poisson comparison kernel with a small texel radius and fixed compare bias. The PCSS/contact-hardening path is no longer user-selectable and `Shadow softness` has been removed. Combined with the affine directional shadow projection, this eliminated the reported smooth-surface spike/teeth artifact in the live viewer.

### Completed work

- [x] Replace the active PCSS path with a small Tiger/Alkahest-style PCF kernel.
- [x] Use a small Poisson kernel.
- [x] Use a small texel radius.
- [x] Use a simple compare bias.
- [x] Remove the Pretty Preview / controllable-softness renderer path.

### Optional diagnostics still useful

- [ ] Add side-by-side diagnostic output:
  - [ ] raw hard shadow
  - [ ] Tiger-style PCF

---

## [ ] Add authoritative shadow diagnostics

### Debug views

- [ ] ShadowGenerate geometry only
- [ ] visible/G-buffer geometry only
- [ ] overlay comparison between the two
- [ ] shadow-map depth visualization
- [ ] shadow UV visualization
- [ ] reference-depth minus sampled-depth heatmap
- [ ] authored stage ID color view
- [ ] input-layout ID color view
- [ ] cull mode / depth-bias state visualization
- [ ] hard-compare-only shadow mode

### Goal

Make future shadow bugs diagnosable from data instead of visual guessing.

---

# P3 — Shader Fidelity

## [ ] Inventory Marathon packaged shader families/signatures

Before choosing a shader strategy, understand the actual population.

### Required work

- [ ] Catalog vertex/pixel shader hashes used by all visible model techniques.
- [ ] Group identical or near-identical shader ABI signatures.
- [ ] Record for the full shader population:
  - [ ] input semantics
  - [ ] constant-buffer slots
  - [ ] texture slots
  - [ ] sampler slots
  - [ ] extern dependencies
  - [x] render-stage association in the stage ABI census
- [ ] Determine how many shader families cover the majority of weapon/runner assets.
- [x] Add package-wide visible-vs-shadow VS relationship census.
- [ ] Fully classify shadow-specific shader families beyond the representative Conquest/common rigid path.

### Goal

Find out whether shader-family compatibility shaders can cover most assets without requiring universal DXBC→WGSL translation immediately.

---

## [ ] Decide Strict Tiger shader strategy

Evaluate three paths:

### [ ] A. Shader bytecode translation/decompilation

- [x] Investigate Marathon packaged shader form and extract representative shader blobs.
- [x] Disassemble representative Marathon DXIL with Windows SDK `dxc -dumpbin`.
- [x] Identify the optional `hlsldecompiler` build/toolchain incompatibility encountered on this MSVC setup.
- [ ] Determine whether SPIR-V/intermediate translation is practical.
- [ ] Determine WGSL feature gaps for representative shader families.
- [ ] Prototype one known simple packaged shader translation.

### [ ] B. Native D3D backend

- [ ] evaluate direct packaged-shader execution on Windows
- [ ] estimate architecture cost
- [ ] determine coexistence with WGPU UI/rendering
- [ ] determine portability consequences

### [ ] C. Shader-signature compatibility library

- [ ] key compatibility shaders by actual Tiger shader ABI/signature
- [ ] reproduce the most common families first
- [ ] drive bindings from TFX and authored state
- [ ] use MaterialIR only as fallback

### Likely near-term direction

Option C appears to be the most practical incremental strategy.

---

# P4 — BRDF / Lighting / G-Buffer Fidelity

## [ ] Reconstruct Tiger material G-buffer encoding

### Required work

- [ ] Determine exact Marathon G-buffer outputs for common model shaders.
- [ ] Compare with Destiny 2 Alkahest:
  - [ ] albedo encoding
  - [ ] normal encoding
  - [ ] material parameter packing
  - [ ] stencil/flags
- [ ] Map Marathon shader outputs to strict G-buffer targets.
- [ ] Stop forcing all strict materials into Quicktag's current surface-property contract.

---

## [ ] Investigate Tiger BRDF lookup resources

Alkahest exposes engine lookup textures such as:

- specular_lobe_lookup
- specular_lobe_3d_lookup
- specular_tint_lookup
- iridescence_lookup

### Required work

- [ ] Find Marathon equivalents.
- [ ] Identify package hashes/classes.
- [ ] Determine which material shader families consume them.
- [ ] Bind them through strict extern/resource paths.
- [ ] Compare against Quicktag's current custom GGX.

---

## [ ] Reconstruct authored local-light behavior

### Required work

- [ ] Decode Marathon light structures.
- [ ] Reconstruct light-space transforms.
- [ ] Reconstruct spot/area-light parameters.
- [ ] Reconstruct authored light-volume techniques.
- [ ] Separate engine-authored lighting from asset-viewer studio lighting.

---

## [ ] Reconstruct authored cubemap / IBL behavior

### Required work

- [ ] Find Marathon cubemap/probe components.
- [ ] Identify:
  - [ ] specular cubemap
  - [ ] diffuse/voxel data
  - [ ] alpha/fade
  - [ ] relighting parameters
  - [ ] probe volume behavior
- [ ] Implement strict authored probe selection/binding.
- [ ] Keep current fallback cubemap for Pretty Preview.

---

# P5 — Material and TFX Coverage

## [ ] Expand TFX opcode coverage

### Required work

- [ ] Enumerate all currently unknown/stopped Marathon opcodes.
- [ ] Rank them by frequency in real visible model techniques.
- [ ] Implement highest-frequency missing operations first.
- [ ] Add per-opcode regression tests.
- [ ] Track decode coverage percentage across installed packages.

---

## [ ] Improve extern/global-channel coverage

### Required work

- [ ] Record unresolved global channel IDs during package-wide scans.
- [ ] Correlate names/values with Alkahest/Destiny equivalents when possible.
- [ ] Add Marathon-specific channel names and semantics.
- [ ] Report required-but-unresolved values per technique.

---

## [ ] Keep MaterialIR as fallback and evidence model

### Required work

- [ ] Do not delete MaterialIR.
- [ ] Clearly distinguish:
  - [ ] authored/strict material behavior
  - [ ] inferred compatibility behavior
  - [ ] fallback behavior
- [ ] Surface evidence confidence in Render Evidence.
- [ ] Make fallback activation explicit in diagnostics.

---

# P6 — Geometry / Deformation Completeness

Raw authored stream/layout preservation has moved forward substantially under P0. The items below now distinguish **preserving the package ABI** from actually interpreting/executing deformation semantics.

## [ ] Preserve skinning inputs

- [ ] decode bone indices
- [ ] decode bone weights
- [ ] reconstruct skinning-buffer semantics
- [x] preserve skinning-buffer package reference/payload for renderer-facing authored inputs
- [x] preserve correct per-stage input-layout metadata
- [ ] implement strict skinned vertex transform behavior

## [ ] Preserve soft deformation / morph inputs

- [ ] identify Marathon deformation buffers/semantics
- [ ] decode deformation layout semantics
- [x] preserve otherwise-unknown auxiliary buffer2 / buffer3 payload references through renderer-facing authored inputs
- [ ] support shader-family-specific deformation

## [ ] Preserve additional UV/color/custom streams

- [x] preserve TEXCOORD1+ / additional semantic descriptors when present
- [x] preserve color-buffer package data
- [x] preserve packed custom attributes as raw authored stream + format descriptors
- [x] preserve per-instance stream classification/step mode
- [ ] make non-shadow Strict Tiger shader families consume these additional authored streams

---

# P7 — Postprocess Fidelity

These are lower priority for the model viewer than authored geometry/material/shadow correctness.

## [ ] Reconstruct Tiger-like autoexposure

- [ ] luminance sampling
- [ ] geometric/linear luminance blend
- [ ] asymmetric adaptation speeds
- [ ] authored exposure channels

## [ ] Reconstruct Tiger-like bloom chain

- [ ] authored downsample stages
- [ ] weighted/Gaussian blur families
- [ ] weighted combination passes
- [ ] engine postprocess constants

## [ ] Separate strict postprocess from viewer controls

- [ ] Strict Tiger authored pipeline
- [ ] Pretty Preview brightness/contrast/gamma/exposure controls

---

# P8 — Verification Infrastructure

## [ ] Build renderer conformance probes

For representative assets, capture and compare:

- [x] authored stage membership
- [x] vertex/input-layout IDs and descriptors
- [x] technique hashes
- [x] shader hashes / shader-stage signatures
- [x] authored state-selector distributions
- [ ] TFX output register images against a reference
- [ ] actual bound texture slots after runtime TFX application
- [ ] actual bound sampler slots after runtime TFX application
- [ ] G-buffer values
- [ ] shadow-map values

Current probes include the reusable package-wide render-stage ABI JSON report plus Conquest-specific authored shadow/source/shader regressions.

---

## [ ] Add package-wide fidelity reports

Generate reports for:

- [ ] percentage of techniques with fully decoded TFX
- [ ] percentage of vertex ABIs fully represented
- [ ] unknown blend/depth/rasterizer states
- [x] unknown / unresolved render stages through the package-wide stage ABI census
- [ ] unresolved externs
- [ ] fallback MaterialIR usage
- [ ] compatibility shader coverage

---

## [ ] Add reference-render comparison tooling

Where a known-good external/reference renderer is available:

- [ ] render the same asset/camera/light configuration
- [ ] save intermediate buffers when possible
- [ ] compare:
  - [ ] silhouette
  - [ ] normals
  - [ ] albedo
  - [ ] roughness/material channels
  - [ ] shadow depth
  - [ ] final lighting
- [ ] calculate image-difference metrics in addition to visual comparison

---

# P9 — Cleanup / Naming / UX

## [ ] Rename misleading fidelity labels where necessary

Until Strict Tiger is actually authored-contract-driven:

- [ ] make clear that current Tiger GGX modes are approximations
- [ ] avoid implying that custom Quicktag shadowing exactly reproduces Tiger
- [ ] distinguish compatibility/inferred/evidence-backed behaviors

## [ ] Improve Render Evidence for strict-vs-fallback state

Show:

- [x] authored stage
- [x] authored input layout / source geometry presence
- [x] technique
- [ ] complete authored pipeline state
- [x] TFX decode status
- [ ] shader compatibility implementation used
- [ ] fallback reason in normal inspection output
- [ ] unresolved extern/resource count in normal inspection output

---

# Suggested Execution Sequence

The recommended implementation sequence is:

    1. Marathon render-stage ABI
           ↓
    2. exact ShadowGenerate participation
           ↓
    3. per-stage input layouts / raw vertex streams
           ↓
    4. renderer-facing TFX
           ↓
    5. complete authored pipeline states
           ↓
    6. strict shadow projection + stage-aware shadow shaders
           ↓
    7. shader-family inventory / compatibility strategy
           ↓
    8. Tiger G-buffer + BRDF resources
           ↓
    9. authored IBL / local lights
           ↓
    10. postprocess fidelity
           ↓
    11. minor polish / diagnostics

The first four items should be treated as the architectural foundation. Work below them should not be allowed to introduce new assumptions that will later have to be discarded.

---

# Immediate Next Work for the Current Shadow Bug

The original upstream-caster checklist is mostly complete. Do not return to blind filter tuning until the remaining authored-light/shader questions are resolved.

- [x] Restore native Strict Tiger ShadowGenerate raster bias to Tiger preset 6 (2 / 2).
- [x] Run the Marathon render-stage ABI probe across installed packages.
- [x] Identify the Marathon ShadowGenerate stage as raw stage 4.
- [x] For the problematic Conquest LMG, dump/verify:
  - [x] visible-stage and shadow-stage part/index ranges
  - [x] technique hashes
  - [x] visible/shadow vertex-shader hashes
  - [x] input-layout IDs
  - [x] LOD/category/source metadata
  - [ ] weapon-specific rasterizer/depth-bias selector summary
- [x] Render the exact authored ShadowGenerate geometry from package-native source index ranges in Strict Tiger when supported.
- [ ] Add an explicit visible-vs-ShadowGenerate silhouette/overlay diagnostic.
- [x] Verify Conquest stage-specific vertex layout/transform contract:
  - stage 0 and stage 4 both use layout 7;
  - representative packaged VS binaries were DXC-disassembled;
  - stage-4's packed-position resource is runtime-generated rather than a dropped static mesh stream.
- [ ] Verify the remaining shadow shader families, not only Conquest/common rigid.
- [ ] Reconstruct/verify authored Marathon shadow projection/light data.
- [ ] Only after those checks, compare hard shadow / Tiger-style PCF / Pretty Preview PCSS.

This should replace further blind tweaking of filter radius, sample count, or arbitrary bias constants.

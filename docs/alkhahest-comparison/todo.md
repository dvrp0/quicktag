# Alkahest Comparison — Prioritized TODO

This backlog translates the findings in `analysis_result.md` into implementation work, ordered from the highest-impact renderer-fidelity issues to lower-priority polish.

The ordering is intentional: avoid spending significant time tuning presentation, BRDF constants, or shadow filtering while Quicktag may still be drawing the wrong authored stages, with the wrong vertex ABI, or without renderer-facing TFX state.

---

# P0 — Critical Renderer-Fidelity Blockers

## [ ] Reconstruct and formalize the Marathon/Goliath render-stage ABI

This is the highest-priority task.

Quicktag currently knows that Marathon has 25 render-stage ranges / 26 boundaries, while Destiny 2 Alkahest exposes 24 typed render stages. The Marathon-specific inserted stage and exact semantic mapping need to be identified from package evidence.

### Required work

- [ ] Identify the exact meaning of all 25 Marathon raw render-stage indices.
- [ ] Determine where Marathon's additional stage was inserted relative to Destiny 2.
- [ ] Identify the true Marathon equivalent of:
  - [ ] GenerateGbuffer
  - [ ] Decals
  - [ ] InvestmentDecals
  - [ ] ShadowGenerate
  - [ ] LightingApply
  - [ ] LightProbeApply
  - [ ] DecalsAdditive
  - [ ] Transparents
  - [ ] Distortion
  - [ ] SkinPrepass
  - [ ] DepthPrepass
  - [ ] Volumetrics
  - [ ] Cubemaps
  - [ ] postprocess-related stages
- [ ] Correlate stage boundaries with:
  - [ ] technique hashes
  - [ ] render states
  - [ ] index ranges
  - [ ] overlap with stage 0 geometry
  - [ ] input-layout IDs
  - [ ] observed shader stages
- [ ] Replace speculative names such as "stage 4 shadow-only" with verified semantic names.
- [ ] Introduce a typed Marathon render-stage representation.
- [ ] Add regression tests proving the mapping against representative package samples.

### Exit criteria

Quicktag should be able to answer:

> "Which exact authored part range is used for Marathon ShadowGenerate?"

without using heuristic material classification.

---

## [ ] Drive shadow caster selection from authored ShadowGenerate participation

Do not infer shadow participation primarily from visible-stage material behavior once the Marathon stage ABI is known.

### Required work

- [ ] Read the exact authored ShadowGenerate part range from geometry metadata.
- [ ] Preserve its exact:
  - [ ] part indices
  - [ ] index ranges
  - [ ] techniques
  - [ ] variant shader indices
  - [ ] LOD categories
  - [ ] input-layout ID
- [ ] Stop treating ordinary visible draws as automatic shadow casters in Strict Tiger mode.
- [ ] Remove/rework the current stage-4 proxy override heuristic once the actual stage mapping is known.
- [ ] Keep heuristic caster derivation only as a compatibility/fallback path.
- [ ] Add a diagnostic showing visible-stage vs ShadowGenerate geometry side-by-side.
- [ ] Add tests for assets where shadow-stage geometry differs from G-buffer geometry.

### Why this is critical

The current jagged-shadow artifact may be caused upstream by drawing the wrong authored geometry contract, not by PCF/PCSS quality.

---

## [ ] Preserve stage-specific vertex/input ABI instead of flattening everything into ModelVertex

Tiger can use different input layouts per render stage. Quicktag currently loses information by repacking geometry too early.

### Required work

- [ ] Preserve raw vertex-buffer streams in renderer-facing geometry.
- [ ] Preserve:
  - [ ] vertex0
  - [ ] vertex1
  - [ ] auxiliary buffers
  - [ ] color buffer
  - [ ] skinning buffer
  - [ ] additional UV streams
  - [ ] per-instance streams
- [ ] Decode Marathon input-layout tables from render globals / geometry metadata.
- [ ] Associate an input-layout ID with each render stage.
- [ ] Preserve semantic/index/format/stream information.
- [ ] Allow Strict Tiger pipelines to consume authored stream layouts.
- [ ] Keep reconstructed `ModelVertex` only for compatibility/Pretty Preview.
- [ ] Add tests for meshes whose ShadowGenerate input layout differs from GenerateGbuffer.

### Exit criteria

A Tiger-compatible shader path should not require reconstructing or guessing missing vertex attributes that existed in the source package.

---

## [ ] Make TFX execution renderer-authoritative

Quicktag already parses and partially executes Marathon TFX, but its outputs currently function mainly as diagnostics.

### Required work

- [ ] Define a renderer-facing TFX runtime result.
- [ ] Feed TFX output registers into actual GPU constant-buffer data.
- [ ] Support TFX-driven:
  - [ ] texture bindings
  - [ ] texture-view bindings
  - [ ] sampler bindings
  - [ ] context values
  - [ ] object channels
  - [ ] global channels
  - [ ] texture metadata
  - [ ] extern-scope values
- [ ] Track unresolved/unknown TFX operations at render time.
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

# P1 — Core Strict-Tiger Rendering Architecture

## [ ] Separate Strict Tiger and Pretty Preview into genuinely different renderer contracts

The current modes should stop being mostly parameter variations inside the same custom renderer.

### Strict Tiger target

- [ ] authored render stages
- [ ] authored stage-specific vertex ABI
- [ ] authored technique selection
- [ ] authored render state
- [ ] renderer-facing TFX
- [ ] engine-style extern scopes
- [ ] shader-family-specific or translated shader behavior
- [ ] minimal semantic reinterpretation

### Pretty Preview target

- [ ] MaterialIR
- [ ] generic ModelVertex
- [ ] custom GGX
- [ ] studio lighting
- [ ] PCSS / controllable softness
- [ ] supersampling
- [ ] enhanced postprocessing
- [ ] graceful fallback rendering

### Exit criteria

"Strict Tiger" should mean "follow authored engine behavior," not "use slightly less stylized Quicktag lighting."

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

- [ ] Restore 2 / 2 for Strict Tiger shadow generation.
- [ ] Keep alternate bias controls only in Pretty Preview/debug modes.
- [ ] Verify whether Marathon uses the same shadow default state.
- [ ] Avoid further bias tuning until authored caster/stage behavior is verified.

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
- [ ] Separate authored shadow projection from Pretty Preview's custom spotlight controls.

---

## [ ] Replace universal shadow vertex behavior with stage-/technique-aware behavior

### Required work

- [ ] Identify which Marathon shadow techniques have unique vertex shaders.
- [ ] Identify whether shadow generation uses:
  - [ ] alternate UVs
  - [ ] vertex colors
  - [ ] procedural deformation
  - [ ] alpha/control coordinates
  - [ ] skinning
  - [ ] stage-specific position transforms
- [ ] Key strict shadow pipelines by authored shader ABI/signature.
- [ ] Preserve the current universal SHADOW_SHADER only as fallback.

---

## [ ] Reduce shadow filtering complexity in Strict Tiger

Do not use a more sophisticated filter as a substitute for an incorrect caster/projection contract.

### Required work

- [ ] Once stage/projection behavior is correct, compare simple PCF against current PCSS.
- [ ] Add a strict filter patterned after known Tiger/Alkahest behavior:
  - [ ] small Poisson kernel
  - [ ] small texel radius
  - [ ] simple compare bias
- [ ] Keep PCSS/contact hardening as a Pretty Preview feature.
- [ ] Add side-by-side diagnostic output:
  - [ ] raw hard shadow
  - [ ] Tiger-style PCF
  - [ ] Pretty Preview PCSS

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

- [ ] Catalog vertex/pixel shader hashes used by visible model techniques.
- [ ] Group identical or near-identical shader ABI signatures.
- [ ] Record:
  - [ ] input semantics
  - [ ] constant-buffer slots
  - [ ] texture slots
  - [ ] sampler slots
  - [ ] extern dependencies
  - [ ] render stages
- [ ] Determine how many shader families cover the majority of weapon/runner assets.
- [ ] Identify shadow-specific shader families.

### Goal

Find out whether shader-family compatibility shaders can cover most assets without requiring universal DXBC→WGSL translation immediately.

---

## [ ] Decide Strict Tiger shader strategy

Evaluate three paths:

### [ ] A. Shader bytecode translation/decompilation

- [ ] investigate DXBC/DXIL form used by Marathon
- [ ] determine whether SPIR-V/intermediate translation is practical
- [ ] determine WGSL feature gaps
- [ ] prototype one known simple shader

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

## [ ] Preserve skinning inputs

- [ ] bone indices
- [ ] bone weights
- [ ] skinning buffer semantics
- [ ] correct per-stage layouts
- [ ] strict vertex transform path

## [ ] Preserve soft deformation / morph inputs

- [ ] identify Marathon deformation buffers
- [ ] decode layout semantics
- [ ] preserve data through renderer
- [ ] support shader-family-specific deformation

## [ ] Preserve additional UV/color streams

- [ ] TEXCOORD1+
- [ ] vertex color channels
- [ ] packed custom attributes
- [ ] per-instance data

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

- [ ] authored stage membership
- [ ] vertex layouts
- [ ] technique hashes
- [ ] shader hashes
- [ ] state selectors
- [ ] TFX outputs
- [ ] bound texture slots
- [ ] bound sampler slots
- [ ] G-buffer values
- [ ] shadow-map values

---

## [ ] Add package-wide fidelity reports

Generate reports for:

- [ ] percentage of techniques with fully decoded TFX
- [ ] percentage of vertex ABIs fully represented
- [ ] unknown blend/depth/rasterizer states
- [ ] unknown render stages
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

- [ ] authored stage
- [ ] authored input layout
- [ ] technique
- [ ] pipeline state
- [ ] TFX decode status
- [ ] shader compatibility implementation used
- [ ] fallback reason
- [ ] unresolved extern/resource count

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

Before making another visual shadow-tuning pass:

- [ ] Restore Strict Tiger shadow raster bias to Tiger preset 6 (2 / 2).
- [ ] Run the Marathon render-stage ABI probe across installed packages.
- [ ] identify the actual Marathon ShadowGenerate stage.
- [ ] For the problematic weapon, dump:
  - [ ] visible-stage part ranges
  - [ ] shadow-stage part ranges
  - [ ] technique hashes
  - [ ] shader hashes
  - [ ] input-layout IDs
  - [ ] LOD categories
  - [ ] rasterizer/depth-bias states
- [ ] Render only the exact authored shadow-stage geometry.
- [ ] Compare its silhouette to the visible geometry.
- [ ] Verify the stage-specific vertex transform/layout.
- [ ] Only after those checks, compare hard shadow / Tiger-style PCF / Pretty Preview PCSS.

This should replace further blind tweaking of filter radius, sample count, or arbitrary bias constants.

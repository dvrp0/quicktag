# Alkahest 6 vs. Quicktag Renderer — Source-Level Analysis

## Purpose

This document records the source-level comparison between Quicktag's current Marathon/Goliath model renderer and alkahest-6, with special focus on alkahest-6/crates/render.

Alkahest is useful as a reference because it targets Destiny 2, which uses the same Tiger engine family, and its renderer attempts to preserve the engine's original rendering contract as directly as possible. It is not a perfect implementation of Tiger and it contains several TODOs and approximations, but it provides an unusually strong example of how Tiger's authored rendering data can be treated as authoritative instead of being reinterpreted into a generic material renderer.

The most important high-level conclusion is:

> **Alkahest reconstructs Tiger's rendering contract. Quicktag currently reconstructs what it believes the resulting image should mean.**

That distinction explains most of the fidelity gap identified below.

---

# 1. Architectural Overview

## 1.1 Alkahest's rendering model

The general Alkahest path for a rigid/dynamic model is approximately:

    Tiger mesh
      ↓
    actual per-render-stage part range
      ↓
    actual stage-specific input layout
      ↓
    actual authored technique
      ↓
    actual packaged Tiger shader bytecode
      ↓
    actual Tiger blend/depth/rasterizer/depth-bias state
      ↓
    execute TFX expression bytecode
      ↓
    populate authored constant buffers / extern scopes / textures / samplers
      ↓
    draw

Relevant source areas:

- alkahest-6/crates/render/src/feature/rigid_model.rs
- alkahest-6/crates/render/src/tfx/technique.rs
- alkahest-6/crates/render/src/tfx/dynamic_constants.rs
- alkahest-6/crates/render/src/tfx/scope.rs
- alkahest-6/crates/render/src/tfx/externs/*
- alkahest-6/crates/render/src/gpu/global_state.rs
- alkahest-6/crates/data/tfx/features/dynamic.rs

Alkahest does not first convert a Tiger material into a small semantic PBR structure and then shade it with one universal shader. It preserves the original technique/shader/state pipeline whenever possible.

## 1.2 Quicktag's current rendering model

Quicktag's current path is closer to:

    Tiger mesh
      ↓
    decode/repack geometry into ModelVertex
      ↓
    inspect technique + TFX + resources
      ↓
    classify into MaterialIR
      ↓
    infer a DrawPassPlan
      ↓
    translate resources into semantic material roles
      ↓
    render through Quicktag's custom WGSL shaders

Relevant source areas:

- src/gui/model_renderer.rs
- src/render/material.rs
- src/render/technique.rs
- src/render/tfx.rs
- src/render/pass_plan.rs
- src/render/adapter.rs
- src/material.rs
- src/geometry/mod.rs

This approach has real advantages for portability, debugging, UI inspection, and rendering incomplete reverse-engineered assets. However, every place where Quicktag translates or classifies Tiger behavior becomes a possible fidelity loss.

---

# 2. What Quicktag Is Already Doing Well

The comparison did not show that Quicktag is fundamentally primitive. Several systems are already strong and should be preserved.

## 2.1 High-detail LOD selection

Quicktag's LOD selection matches the high-detail categories recognized by Alkahest.

Quicktag tests explicitly recognize:

    highest detail: 0, 1, 2, 3, 10
    lower detail:   4, 7, 8, 9

Quicktag also has fallback logic when no normal highest-detail range exists.

Relevant Quicktag source:

- src/geometry/mod.rs
- is_highest_detail_lod
- geometry_primary_index_ranges

Alkahest similarly checks part.lod_category.is_highest_detail() before drawing rigid-model parts.

Relevant Alkahest source:

- alkahest-6/crates/render/src/feature/rigid_model.rs

### Conclusion

The previously observed jagged/triangular shadow artifacts are unlikely to be caused simply by Quicktag rendering lower LODs on top of high-detail meshes.

## 2.2 Technique render-state decoding

Quicktag correctly decodes Tiger's four packed state selectors:

    blend
    depth_stencil
    rasterizer
    depth_bias

from the packed technique state word.

Relevant Quicktag source:

- src/material.rs
- TechniqueRenderState::from_raw

This matches Alkahest's PipelineState representation.

Relevant Alkahest source:

- alkahest-6/crates/data/tfx/technique.rs

## 2.3 Depth-bias preset table

Quicktag's nine depth-bias presets match the Tiger table extracted by Alkahest.

Quicktag currently contains:

    0: constant  0, slope  0
    1: constant  0, slope  0
    2: constant  5, slope  2
    3: constant 10, slope  4
    4: constant 15, slope  6
    5: constant 20, slope  8
    6: constant  2, slope  2
    7: constant -1, slope -2
    8: constant 51, slope  2

This matches:

- alkhahest-6/crates/render/builtin/gpu/depth_biases.bin

Relevant Quicktag source:

- src/gui/model_renderer.rs
- depth_bias_state

This is a strong piece of reverse engineering that should be retained.

## 2.4 Marathon-specific TFX decoding

Quicktag contains Marathon-specific TFX opcode remapping and new opcodes that Alkahest's Destiny 2 implementation obviously does not target.

Examples include Quicktag's handling of:

- Marathon context values
- Marathon indexed values
- Marathon texture views
- Marathon shader-resource assignment
- Marathon texture metadata
- shifted legacy opcode ranges

Relevant Quicktag source:

- src/material.rs
- parse_marathon_tfx_bytecode_op
- marathon_tfx_legacy_opcode

This is valuable engine-version-specific knowledge and should become part of a stricter runtime path rather than being discarded.

## 2.5 Material inspection and provenance

Quicktag's MaterialIR, evidence levels, classification reports, TFX traces, dependency reporting, and render-evidence UI are better suited to reverse-engineering than Alkahest's runtime-first approach.

Relevant Quicktag source:

- src/render/material.rs
- src/render/evidence.rs
- src/render/technique.rs
- src/render/tfx.rs

This should remain an inspection/debug/fallback layer even if a more engine-faithful renderer is added.

## 2.6 Shadow alpha/cutout handling exists

Quicktag's shadow pass is not simply a raw depth-only triangle caster.

SHADOW_SHADER samples authored material data and performs discard for:

- alpha-mask materials
- decals
- generic base-color alpha/mask conditions

Relevant Quicktag source:

- src/gui/model_renderer.rs
- SHADOW_SHADER

Therefore the shadow issue is not explained by the simplistic hypothesis that all masked geometry is casting its complete underlying triangles.

## 2.7 High shadow/render resolution

Quicktag's model preview is supersampled and its shadow map uses a high resolution.

The renderer includes:

- model render supersampling
- bounded large offscreen targets
- a high-resolution shadow-map path in the current implementation

Therefore the visible shadow teeth/jagged geometry are unlikely to be solved simply by increasing map resolution.

---

# 3. Largest Fidelity Gap: Packaged Tiger Shaders Do Not Drive Quicktag Rendering

## 3.1 Alkahest executes authored shaders

Alkahest loads the shader referenced by each Tiger technique and creates the corresponding native D3D11 shader object.

Its Technique can contain:

- vertex stage
- hull stage
- domain stage
- geometry stage
- pixel stage
- compute stage

It supports Tiger bind modes such as:

- Vertex + Pixel
- Vertex only
- Vertex + Geometry + Pixel
- tessellated Vertex + Hull + Domain + Pixel
- tessellated Vertex-only path
- Compute

Relevant source:

- alkhahest-6/crates/render/src/tfx/technique.rs

The package shader bytecode is treated as the real program.

## 3.2 Quicktag interprets the material, then shades with custom WGSL

Quicktag decodes and inspects the technique and referenced shaders, but the final model rendering is performed by Quicktag-defined shader modules such as:

- MODEL_SHADER
- LIGHTING_SHADER
- SHADOW_SHADER
- bloom shader
- present shader
- distortion/coating paths

Relevant source:

- src/gui/model_renderer.rs

The result is effectively:

    Tiger technique
          ↓
    Quicktag's understanding of the technique
          ↓
    Quicktag shader implementation

rather than:

    Tiger technique
          ↓
    Tiger-authored shader behavior

## 3.3 Consequence

Even a perfectly decoded Tiger technique can still render incorrectly if the semantic mapping into Quicktag's material model is incomplete.

This creates a hard fidelity ceiling for:

- unusual shader math
- packed texture interpretations
- dynamic resource bindings
- uncommon vertex inputs
- variant-specific BRDF behavior
- deformation
- custom shadow behavior
- procedural masking
- specialized effects

---

# 4. TFX: Quicktag Has a Runtime, but It Is Not Yet Renderer-Authoritative

## 4.1 Alkahest TFX execution

Alkahest's dynamic constants system evaluates TFX bytecode at bind time.

The interpreter can:

- read authored constants
- read object channels
- read global channels
- read extern scopes
- write output constant-buffer elements
- bind texture views
- bind samplers
- bind UAVs
- access texture metadata
- evaluate math/trig/random/spline operations

Relevant source:

- alkhahest-6/crates/render/src/tfx/dynamic_constants.rs
- alkhahest-6/crates/render/src/tfx/expression_vm/interpreter.rs
- alkhahest-6/crates/render/src/tfx/expression_vm/opcodes.rs

The result directly changes the GPU state used by the packaged technique.

## 4.2 Quicktag TFX execution

Quicktag has a TfxExecutionResult containing:

- outputs
- dependencies
- trace
- undecoded offset
- undecoded bytes

and a useful partial runtime in:

- src/render/tfx.rs
- execute_preview

However, inspection of all call sites showed that tfx_execution.outputs are currently used mainly for:

- render-evidence/status output
- inspection UI
- diagnostic JSON/probes
- decode status reporting

They are not currently used as the authoritative source for Quicktag's GPU material constant/resource state.

## 4.3 Consequence

The current flow is approximately:

    TFX runtime
      ↓
    diagnostic information

instead of:

    TFX runtime
      ↓
    actual material constant buffers / resource bindings
      ↓
    render

This is one of the largest concrete differences between Quicktag and Alkahest.

## 4.4 Recommended direction

Quicktag's existing Marathon-aware TFX interpreter should be promoted into a renderer-facing runtime.

The long-term renderer should allow TFX outputs to populate:

- per-technique constant buffers
- dynamic texture bindings
- samplers
- object/global/context channels
- engine extern equivalents

MaterialIR should then become an explanation/fallback layer, not the only source of material behavior.

---

# 5. Vertex/Input ABI Is Being Flattened in Quicktag

## 5.1 Alkahest preserves Tiger stream structure

Destiny 2 dynamic meshes contain multiple authored buffers such as:

    vertex0_buffer
    vertex1_buffer
    buffer2
    buffer3
    index_buffer
    color_buffer
    skinning_buffer

Alkahest loads the relevant model buffers and uses Tiger's input-layout table.

Relevant sources:

- alkhahest-6/crates/data/tfx/features/dynamic.rs
- alkhahest-6/crates/render/src/feature/rigid_model.rs
- alkhahest-6/crates/render/src/gpu/global_state.rs

Alkahest also reads the engine's input layouts from render_globals, reconstructing their:

- semantic name
- semantic index
- format
- input stream
- per-vertex/per-instance classification

The input layout can be different per render stage through input_layout_per_render_stage.

## 5.2 Quicktag repacks into a fixed ModelVertex

Quicktag's WGPU renderer uses a fixed vertex layout containing a limited reconstructed set of values such as:

- position
- normal
- UV
- tangent / reconstructed auxiliary values

Quicktag's own VertexAbiDescriptor explicitly reports missing areas such as:

    additional UV/color streams not yet preserved
    morph/soft-deformation streams not yet preserved
    bone indices/weights not yet preserved

Relevant source:

- src/render/technique.rs
- src/gui/model_renderer.rs

## 5.3 Consequence

Even if a future Quicktag shader perfectly reproduces a Tiger pixel shader, it still cannot behave correctly if the original vertex shader expected data that Quicktag discarded.

Possible affected features include:

- TEXCOORD1+
- vertex color
- procedural vertex data
- packed deformation values
- instance data
- skinning data
- per-stage alternate input layouts
- shadow-stage-specific vertex behavior

This is especially important for engine-faithful shadow generation.

---

# 6. Render Stages: Authored Dispatch vs. Semantic Inference

## 6.1 Destiny 2 Alkahest stage model

Alkahest exposes a typed Tiger RenderStage enum.

In the inspected D2 code:

    0  GenerateGbuffer
    1  Decals
    2  InvestmentDecals
    3  ShadowGenerate
    4  LightingApply
    5  LightProbeApply
    6  DecalsAdditive
    7  Transparents
    8  Distortion
    9  LightShaftOcclusion
    10 SkinPrepass
    11 LensFlares
    12 DepthPrepass
    13 WaterReflection
    14 PostprocessTransparentStencil
    15 Impulse
    16 Reticle
    17 WaterRipples
    18 MaskSunLight
    19 Volumetrics
    20 Cubemaps
    21 PostprocessScreen
    22 WorldForces
    23 ComputeSkinning

Relevant source:

- alkhahest-6/crates/data/tfx/enums.rs

Each dynamic mesh stores:

- part_range_per_render_stage
- input_layout_per_render_stage

Alkahest therefore renders ShadowGenerate by asking the mesh for the exact authored part range for that stage.

Relevant source:

- alkhahest-6/crates/data/tfx/features/dynamic.rs
- alkhahest-6/crates/render/src/feature/rigid_model.rs

## 6.2 Marathon differs

Quicktag has package-level evidence that Marathon/Goliath contains:

    25 render-stage ranges
    26 boundaries

Quicktag explicitly notes:

    Marathon adds one render stage: 26 boundaries for 25 stage ranges.

Relevant source:

- src/geometry/mod.rs

Therefore the raw Marathon stage IDs cannot simply be assumed to match Destiny 2's enum numerically.

For example, Quicktag currently treats values such as:

- 4 as observed authored shadow-only geometry
- 13 as observed depth-only geometry

whereas Destiny 2 uses:

- 3 for ShadowGenerate
- 12 for DepthPrepass

This looks like an off-by-one mismatch until the Marathon-specific extra stage is taken into account.

## 6.3 Current Quicktag approach

Quicktag preserves raw stage IDs but assigns semantics through GoliathAdapter, then derives DrawPassPlan using:

- raw render stage
- technique state
- material classification
- observed behavior

Relevant sources:

- src/render/adapter.rs
- src/render/pass_plan.rs

This is a reasonable reverse-engineering bridge, but it remains inferential.

## 6.4 Recommended direction

The real target should be a formally reconstructed Marathon render-stage ABI, backed by package evidence.

Once the Marathon stage table is understood, renderer dispatch should follow authored stage ranges directly instead of asking heuristics whether a visible material should cast a shadow.

---

# 7. Pipeline-State Fidelity

## 7.1 Alkahest uses the actual Tiger state tables

Alkahest reconstructs/loads Tiger's full GPU state tables:

- blend states
- rasterizer states
- depth-bias states
- depth states
- stencil states
- input layouts

Relevant source:

- alkhahest-6/crates/render/src/gpu/global_state.rs
- alkhahest-6/crates/render/builtin/gpu/*

The technique's state selection is applied directly.

## 7.2 Quicktag state handling

Quicktag does well on several pieces:

- packed selectors are decoded correctly
- the depth-bias preset table matches Tiger
- depth-state mapping is relatively broad
- rasterizer culling distinctions are preserved for common cases

Relevant source:

- src/gui/model_renderer.rs
- ModelPipelineKey
- blend_state
- rasterizer_cull_mode
- depth_stencil_state
- depth_bias_state

## 7.3 Blend-state loss

Alkahest's Tiger blend table contains many specialized states, including:

- normal opaque
- additive
- destination-color multiplication
- min/max blend operations
- independent render-target blending
- per-target write masks
- nonstandard alpha equations
- blend-factor driven states

Quicktag currently explicitly maps only a subset.

Unknown/unhandled blend-state indices can fall back to a generic alpha-blend approximation.

This is a significant fidelity loss for unusual techniques and MRT passes.

## 7.4 Rasterizer/stencil loss

Quicktag largely reduces authored rasterizer behavior to culling mode.

Alkahest preserves the full rasterizer/depth/stencil state table structure.

A specific Alkahest limitation should also be noted:

> Alkahest currently prints that stencil testing is disabled in its reconstructed state setup.

Therefore Alkahest itself is not a perfect Tiger state implementation.

---

# 8. G-Buffer and Deferred Lighting Architecture

## 8.1 Alkahest's Tiger-shaped buffers

Alkahest creates a G-buffer with approximately:

    albedo:
      R8G8B8A8 typeless
      sRGB render/sample view

    normal:
      R10G10B10A2

    third material buffer:
      R8G8B8A8

    depth:
      R32G8X24 typeless
      D32Float + stencil depth view

Relevant source:

- alkhahest-6/crates/render/src/renderer/submit/buffers.rs

Lighting is accumulated into separate buffers including:

- diffuse light
- direct specular light
- specular IBL
- vertex AO
- distortion
- volumetrics
- SSAO

This is much closer to Tiger's deferred pipeline structure.

## 8.2 Quicktag's model preview buffers

Quicktag is also multipass and is not merely a forward renderer.

It contains separate resources including:

- surface albedo
- surface normal
- surface properties
- surface emissive
- surface flags
- scene/lit color
- distortion
- depth
- shadow depth
- bloom intermediates

Relevant source:

- src/gui/model_renderer.rs
- ModelTargetResources

This is a real deferred-ish asset-viewer pipeline.

## 8.3 Main difference

Quicktag's buffers represent **Quicktag's material contract**.

Alkahest's buffers and authored shaders represent **Tiger's material contract**.

Therefore Tiger data must be translated before Quicktag lighting can consume it, and that translation can lose information.

---

# 9. BRDF, Lighting and Engine Lookup Resources

## 9.1 Alkahest exposes Tiger global lookup textures

The inspected Tiger extern definitions include resources such as:

    specular_lobe_lookup
    specular_lobe_3d_lookup
    specular_tint_lookup
    iridescence_lookup

Relevant source:

- alkhahest-6/crates/render/src/tfx/externs/definitions.rs

Alkahest passes these resources into the actual packaged shaders through the TFX/extern system.

## 9.2 Quicktag uses a custom BRDF approximation

Quicktag contains custom WGSL functions such as:

- Fresnel-Schlick
- GGX specular
- custom roughness/metalness response
- custom indirect diffuse/specular
- custom environment Fresnel
- custom IBL intensity controls

Relevant source:

- src/gui/model_renderer.rs

The UI appropriately names modes such as:

- TigerGgxCompatibility
- TigerGgxApproximation

These are approximations rather than original Tiger shader execution.

## 9.3 Consequence

Differences may remain in:

- roughness response
- grazing reflections
- specular tint
- multi-lobe highlights
- iridescence
- energy distribution
- material-specific lighting behavior

even when texture decoding and material classification are correct.

---

# 10. Cubemaps / IBL

## 10.1 Alkahest

Alkahest has an authored cubemap feature renderer.

The data includes resources such as:

- specular cubemap
- cubemap alpha
- voxel diffuse
- cubemap volume extents
- fade behavior
- relighting data
- global sun/cubemap channels

Relevant source:

- alkhahest-6/crates/render/src/feature/cubemap.rs

The cubemap application is itself tied to Tiger pipeline/technique behavior.

## 10.2 Quicktag

Quicktag provides an asset-viewer-oriented environment system with:

- a fallback cubemap
- custom IBL intensity
- custom studio-lighting controls

This is useful for a model viewer, but it is not equivalent to authored Tiger probe/cubemap behavior.

---

# 11. Extern Scopes

Alkahest exposes a very large set of Tiger engine-side extern scopes.

Inspected examples include:

- Frame
- View
- Deferred
- DeferredLight
- DeferredUberLight
- DeferredShadow
- Atmosphere
- RigidModel
- EditorMeshMaterial
- many additional engine scopes

Relevant source:

- alkhahest-6/crates/render/src/tfx/externs/definitions.rs

These contain:

- frame/game time
- exposure values
- world/view/projective matrices
- target-pixel transforms
- G-buffer textures
- light buffers
- shadow depth maps
- shadow projection matrices
- atmospheric textures
- material/global lookup tables
- rigid-model transforms
- object data

Quicktag's TFX runtime resolves only a much smaller semantic subset of extern concepts.

That is another major reason why some TFX programs can decode successfully but cannot yet produce engine-faithful rendering.

---

# 12. Shadow Rendering — Most Important Findings

The Alkahest comparison materially changes the diagnosis of Quicktag's current shadow artifacts.

## 12.1 Alkahest shadow architecture

For local shadowing lights, Alkahest carries authored light data including:

    light_space_transform
    far_plane
    half_fov
    lighting technique
    shadowed-lighting technique
    volumetric technique
    shadowed-volumetric technique

Relevant source:

- alkhahest-6/crates/data/tfx/features/light.rs
- alkhahest-6/crates/render/src/feature/light.rs

The render flow is broadly:

    SShadowingLight
      ↓
    shadow view / shadow map
      ↓
    render Tiger ShadowGenerate stage
      ↓
    bind shadow map through DeferredShadow extern
      ↓
    supply authored shadow projection matrix
      ↓
    bind packaged lighting_apply_shadowing technique
      ↓
    draw light volume

The important point is that the shadow map and its resolve are integrated into the same authored Tiger technique/extern architecture.

## 12.2 Quicktag shadow architecture

Quicktag currently performs:

    Quicktag light representation
      ↓
    Quicktag light projection math
      ↓
    Quicktag-inferred shadow caster participation
      ↓
    Quicktag SHADOW_SHADER
      ↓
    Quicktag shadow filtering / PCSS-style path
      ↓
    Quicktag custom lighting shader

This is a replacement shadow system rather than a direct reconstruction of Tiger's shadow contract.

## 12.3 Tiger/Alkahest shadow raster bias baseline

A highly relevant discovery:

Alkahest begins the shadow view with a pipeline state equivalent to:

    PipelineState::new(
        Some(0),
        Some(2),
        Some(0),
        Some(6),
    )

The fourth value is depth-bias preset 6.

From the extracted Tiger table, preset 6 is:

    constant bias = 2
    slope scale   = 2.0

Quicktag originally used:

    constant: 2
    slope_scale: 2.0

During recent shadow debugging this was changed to approximately:

    constant: 1
    slope_scale: 1.0

to reduce visible Peter-panning.

### Conclusion

From a Tiger-fidelity standpoint, 1 / 1 moves away from the known Tiger baseline.

This does not prove that 2 / 2 alone fixes the visual artifact, but the engine-faithful starting point should be restored before further tuning.

## 12.4 Alkahest sun-shadow filtering is simpler than Quicktag

Alkahest's custom sun-shadow resolve uses a relatively simple PCF pattern:

- 9 rotated Poisson taps
- approximately 2.5 texel spread
- approximately 0.0001 compare bias

It does not use PCSS-style blocker search/contact hardening there.

Relevant source:

- alkhahest-6/crates/render/builtin/shaders/shadow_map.hlsl
- alkhahest-6/crates/render/src/renderer/submit/sun_shadows.rs

### Important inference

Quicktag has already experimented with a substantially more sophisticated soft-shadow filter.

If the shadow remains geometrically wrong despite high resolution and many taps, the strongest suspicion should move **upstream of filtering**.

## 12.5 Current strongest shadow suspects

After this comparison, the strongest likely causes are:

1. **Wrong Marathon shadow-stage participation**
   - Quicktag currently infers shadow participation through observed raw-stage semantics and pass planning.
   - The exact Marathon render-stage ABI is not yet formally reconstructed.

2. **Wrong stage-specific vertex behavior**
   - Tiger can use a shadow-stage-specific input layout and actual authored vertex shader.
   - Quicktag uses a common reconstructed ModelVertex and common shadow vertex shader.

3. **Discarded input streams**
   - A shadow shader may depend on data not preserved by Quicktag's fixed vertex ABI.

4. **Shadow-space transform mismatch**
   - Alkahest feeds an authored/engine-derived shadow projection matrix into DeferredShadow.
   - Quicktag rebuilds the light projection math itself.

5. **Wrong caster draw contract**
   - Even correct visible geometry does not guarantee that the same parts/techniques/input layouts are supposed to participate in ShadowGenerate.

6. **Material-specific shadow behavior**
   - Quicktag handles ordinary alpha/decal discard, which is good.
   - However, specialized authored vertex/material behavior may still differ.

### Lower-priority suspects

These now look less likely as the fundamental cause:

- shadow-map resolution
- number of PCF/PCSS taps
- simply mixing lower LODs
- generic alpha cutout omission

---

# 13. Quicktag Shadow Shader: What Is Good

Quicktag's SHADOW_SHADER has several sensible characteristics worth preserving in a compatibility/preview renderer:

- perspective light-space projection
- authored UV transform usage
- material texture sampling
- alpha-mask discard
- decal-mask discard
- general alpha threshold discard
- cull-mode-specific shadow pipelines

Therefore the shadow shader is not poorly written in the conventional rendering sense.

The important problem is that it is still a **universal substitute** for whatever Tiger's authored ShadowGenerate techniques actually do.

---

# 14. Postprocessing and Exposure

## 14.1 Alkahest

Alkahest reconstructs a large postprocessing graph including:

- bloom initial downsample
- multiple reduction levels
- Gaussian/weighted blur stages
- weighted combination stages
- GPU luminance sampling
- CPU-side exposure adaptation
- asymmetric light/dark adaptation speeds
- engine postprocess externs

Relevant sources:

- alkhahest-6/crates/render/src/renderer/submit/bloom.rs
- alkhahest-6/crates/render/src/renderer/autoexposure.rs

## 14.2 Quicktag

Quicktag has its own model-viewer post stack including:

- bloom
- tone mapping
- exposure
- FXAA
- SSAO
- presentation controls

Relevant source:

- src/gui/model_renderer.rs

## 14.3 Conclusion

For an asset viewer, Quicktag's custom postprocess path is useful and does not need to be replaced immediately.

It should be considered a presentation layer rather than evidence of Tiger renderer fidelity.

---

# 15. Known Alkahest Limitations

Alkahest should not be treated as infallible ground truth.

Important limitations visible directly in source include:

- stencil testing is explicitly disabled in the reconstructed global state setup
- several extern fields remain unknown
- several TFX opcodes are still named unknown
- some texture/expression behaviors are TODO
- some cubemap parameters are approximated or hardcoded
- SSAO/global-lighting paths include incomplete/commented sections
- some culling behavior is known to be imperfect
- various engine constants are taken from captures or educated reconstruction
- some shader/decompiler paths are experimental/commented

Therefore the correct use of Alkahest is:

> a highly valuable Tiger-architecture reference and source of proven state/ABI behavior, not a claim that every rendered pixel exactly matches Destiny 2.

---

# 16. Recommended Quicktag Architecture

The comparison suggests that Quicktag should separate two goals that are currently partially mixed together.

## 16.1 Strict Tiger path

A future Strict Tiger path should increasingly mean:

    authored render-stage dispatch
    authored stage-specific input ABI
    authored technique selection
    authored pipeline state
    runtime TFX outputs
    engine-style extern scopes
    shader-faithful execution or translation
    minimal semantic reinterpretation

## 16.2 Pretty Preview / Compatibility path

Quicktag's current semantic renderer remains very useful as:

    MaterialIR
    Quicktag material families
    Quicktag GGX
    studio/environment lighting
    soft PCSS shadows
    enhanced AA
    custom bloom/exposure
    robust fallback rendering

This path can intentionally prioritize usability and attractive asset previews over exact engine fidelity.

## 16.3 Clean distinction

Eventually the two modes could be conceptually:

    Strict Tiger
        "render the authored contract"

    Pretty Preview
        "render the asset well"

This would make both modes clearer and more maintainable.

---

# 17. Recommended Priority Order

## Priority 1 — Formalize the Marathon render-stage ABI

Investigate the 25 Marathon stages and determine:

- where the additional stage was inserted relative to Destiny 2
- exact stage meanings
- exact shadow-generation stage
- exact depth-prepass stage
- stage-specific input-layout tables
- which parts belong to each stage

Do not build additional long-term behavior around names like "stage 4 shadow-only" until this is backed by a formal Marathon stage mapping.

## Priority 2 — Restore Tiger's shadow raster-bias baseline

Use Tiger preset 6 as the engine-faithful starting point:

    constant = 2
    slope    = 2.0

Do not continue tuning filtering/bias heuristically until caster participation and projection semantics are verified.

## Priority 3 — Make TFX execution renderer-facing

Quicktag already contains valuable Marathon-aware TFX decoding.

The next step should be to make TFX outputs actually populate rendering state:

- constant buffers
- texture bindings
- sampler bindings
- object/global channels
- extern-derived values

This provides a path toward authored behavior without immediately solving native shader execution.

## Priority 4 — Preserve raw Tiger vertex streams/layout information

Avoid reducing all renderer-facing geometry to one universal ModelVertex too early.

A stricter path should preserve:

- original buffers
- stream IDs
- format declarations
- semantic/index information
- per-stage layout ID
- skinning/deformation inputs
- additional UV/color data

This is necessary for shader-faithful rendering.

## Priority 5 — Complete Tiger pipeline-state mapping

Quicktag should move from approximate state mapping toward direct table-driven state translation for:

- blend states
- write masks
- alpha equations
- rasterizer state
- depth state
- depth bias
- stencil where WebGPU permits equivalent behavior

The existing correctly decoded state selectors should remain the source of truth.

## Priority 6 — Shader-fidelity strategy

This is the hardest long-term problem.

Alkahest can directly bind Destiny's D3D shader bytecode because it renders through D3D11.

Quicktag uses WGPU/WGSL.

Possible strategies:

### A. Shader translation/decompilation

Translate packaged Tiger shader bytecode into WGSL or an intermediate representation.

Highest theoretical fidelity, highest implementation complexity.

### B. Native Windows/D3D backend

Add a strict rendering backend capable of directly executing compatible packaged shaders.

Potentially very faithful, but weakens Quicktag's WGPU portability and introduces major architecture cost.

### C. Technique/shader-signature compatibility library

Keep WGPU, but reconstruct compatibility shaders keyed by actual Tiger shader ABI/signature rather than broad MaterialIR families.

This is likely the most realistic incremental path.

It can coexist with MaterialIR and gradually replace broad semantic approximation with shader-family-specific behavior.

---

# 18. Systems to Keep

The comparison strongly supports keeping the following Quicktag work:

- Marathon/Goliath package parsing
- geometry decoding
- high-detail LOD logic
- gear/model/mod assembly
- material provenance
- MaterialIR as explanation/fallback
- Render Evidence UI
- evidence-level tracking
- Marathon TFX parsing
- current technique/state parsing
- exact depth-bias table
- diagnostic probes/tests
- custom pretty-preview renderer
- supersampling
- studio-light controls

These are complementary to a strict renderer rather than wasted work.

---

# 19. Systems That Should Become Fallback/Compatibility Layers

The following should not remain the final authority for a future strict renderer:

- broad MaterialIR family classification as the sole material implementation
- heuristic DrawPassPlan stage semantics
- generic ModelVertex as the only vertex ABI
- custom GGX as Tiger ground truth
- custom universal shadow shader as strict ShadowGenerate behavior
- generic fallback blend-state interpretation
- inspection-only TFX execution

They remain useful for fallback rendering and reverse-engineering.

---

# 20. Final Assessment

Quicktag has already reverse-engineered a significant amount of Marathon's rendering data and is stronger than a conventional "extract textures + apply PBR" model viewer.

Its main limitation is architectural rather than cosmetic:

> Quicktag frequently understands authored Tiger data, but then converts that data into its own renderer contract before drawing.

Alkahest demonstrates the opposite approach:

> preserve the authored render contract for as long as possible, and let Tiger's own techniques, states, shaders, stages, and TFX define the rendering behavior.

For future engine-level fidelity, Quicktag should move incrementally in that direction.

The current shadow artifact is a good example of why. High resolution, PCSS, careful alpha discard, and bias tweaking cannot guarantee a correct result if the wrong stage, input ABI, caster set, vertex behavior, or shadow-space contract is being rendered.

The immediate shadow investigation should therefore prioritize:

    Marathon render-stage ABI
            ↓
    exact ShadowGenerate participation
            ↓
    stage-specific input/vertex behavior
            ↓
    Tiger shadow state/projection
            ↓
    only then filtering/tuning

rather than continuing to treat the problem primarily as a shadow-filter-quality issue.

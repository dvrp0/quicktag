# Quicktag Tiger-Fidelity Renderer

## Product Requirements Document v2.1 — Evidence-Gated Revision

Status: implemented and package-audited for the supported Marathon weapon/mod preview corpus
Primary target: Marathon / Goliath asset preview  
Secondary reference: Destiny-era Tiger behavior, never assumed equivalent

## Implementation status — revamp branch

The architecture and required preview graph are operational. The canonical regression fixture is BR33 Volley Rifle: Vibrant Sport (`80A9FF17`) with Deluxe Cold Vigilance Scope (`80A60FED`) and Deluxe Impulse Brake (`80A60608`), captured at yaw -24°, pitch +14°, 1024×640 in Strict Tiger mode.

Completed gates:

- raw LOD/stage/technique/source provenance survives through GPU scheduling and capture metadata
- versioned `GoliathAdapter`, per-stage technique/resource ABI, explicit unsupported vertex ABI
- Strict/Pretty evidence split, typed MaterialIR, deterministic family reports, dependency graph
- typed/partial TFX runtime with trace, dependencies, raw scopes, and undecoded-tail preservation
- validated shadow/opaque/alpha/decal/lighting/transparent pass plan and expanded pipeline keys
- logical albedo, normal/roughness, metal/AO, emissive, `R32Uint` flags, depth, and debug captures
- replaceable deferred Tiger GGX approximation with reconstructed-position shadow resolve; compatibility-forward remains selectable
- separate Investment Decal, emissive, integer-flags, and sorted forward-transparent passes
- confirmed six-slot Goliath Gear baseline, Gear Pattern, Weapon Mod Condition, compact-hair classification
- deterministic schema-2 capture manifest with package inventory, adapter/schema, GPU/backend/driver, scene, fidelity, channel, draw plans, and unknown TFX counts
- all 25 raw Marathon render ranges are retained; observed stage 4/9/13/15/17 geometry is routed as authored shadow, max-blend forward, depth-only, auxiliary, and forward-special work instead of being discarded or duplicated as opaque
- Marathon TFX byte alignment is verified across 175,244 package stages; every program decodes to its end, while 18,160 stages with still-unclassified semantic operations remain explicitly Partial
- live TFX execution re-evaluates constants, frame/view/object/global/Gear/context inputs, time, temporaries, comparisons, and outputs instead of replaying parse-time expression metadata
- package catalog GearDye audits cover every skin palette, 399 mod associations, and 2,883 skin/mod combinations with zero missing/invalid channels
- GPU gates cover BR33 Vibrant Sport Deluxe, Bully Transmit Engine, V85 Vox Nocturna, and Syntax Disrupt after authored-stage scheduling

Current canonical capture retains all authored ranges, including auxiliary and forward-special draws. Unknown families and the remaining semantic-partial TFX operations remain visible rather than guessed.

Evidence-gated exclusions remain exclusions by design, not unfinished acceptance items: unsupported Runner/skinned ABI, unobserved dynamic effect families, anisotropic hair, speculative decal writes, distortion/energy/hologram behavior, and unconfirmed Destiny-style Gear fields.

---

# 0. Executive Summary

Quicktag already decodes substantial Tiger data:

- geometry, vertex/index buffers, authored UV/normal/tangent
- raw LOD category and render-stage range
- techniques and per-shader-stage resources
- blend/depth/rasterizer/depth-bias state
- TFX bytecode, bindings, expressions, extern references, and partial evaluation
- Gear Dye subset, Gear Pattern, Weapon Mod Condition, and Investment Decal
- texture mips, formats, color space, and asynchronous caching
- shadow, forward HDR, bloom, presentation, inspectors, and GPU visual probes

Current failure is architectural. Geometry retains stage/LOD metadata, then GPU draw creation drops it. Material behavior converges into one large forward shader containing confirmed decoding, family-specific branches, and visual heuristics.

Current path:

```text
Tiger asset
→ geometry/material decoders
→ WireframeMaterialTextures
→ ModelDraw (stage/LOD lost)
→ one MODEL_SHADER
→ forward HDR
→ bloom/present
```

Target path:

```text
Tiger asset graph
→ raw draw packet + provenance
→ per-stage TechniqueDescriptor
→ TFX runtime
→ deterministic material-family classification
→ typed MaterialIR / dependency graph
→ validated pass plan
→ logical surface or forward-special output
→ MRT/decal/lighting/transparent passes
→ post/debug output
```

Goal: reconstruct authored semantics and pass structure where evidence exists; expose uncertainty where it does not. Visual similarity must not silently override decoded evidence.

---

# 1. Scope and Fidelity

## 1.1 Product goal

For every rendered range, Quicktag should answer:

- source tag/range and raw LOD category
- raw render-stage ID and current semantic interpretation
- technique and shader-stage resources
- vertex/resource ABI requirements
- TFX inputs, outputs, dependencies, and incomplete execution
- material-family classification and evidence
- authored versus fallback surface values
- pass routing, final surface contribution, and provenance

## 1.2 Fidelity levels

### A — Structural fidelity

Distinct authored stages remain distinct when confirmed. Unknown stage IDs remain raw/unknown; they are not forced into Destiny names.

### B — Semantic fidelity

Bindings/channels gain meaning only from technique, shader, asset correlation, or confirmed decoder evidence. Raw values remain inspectable.

### C — Visual fidelity

Matched scene/camera output approaches reference after A/B. Pretty Preview may use disclosed heuristics; Strict Tiger may not.

## 1.3 Non-goals

- full Marathon world renderer
- full Destiny 2 renderer reproduction
- world streaming, Umbra, particles, atmosphere, gameplay lighting
- exact proprietary G-buffer packing
- full automatic DXIL-to-WGSL decompiler
- binary-identical or bit-perfect HDR output
- unsupported Destiny semantics presented as Marathon facts

---

# 2. Evidence Model

All reverse-engineered interpretations carry evidence:

```rust
enum EvidenceLevel {
    Confirmed,
    StronglyCorrelated,
    Probable,
    Heuristic,
    Unknown,
}
```

Do not wrap every scalar in a heap-heavy `Decoded<T>`. Store compact provenance IDs referencing interned records:

```rust
struct ProvenanceId(u32);

struct ProvenanceRecord {
    evidence: EvidenceLevel,
    source_spans: Vec<SourceSpan>,
    technique: Option<TagHash>,
    shader_stage: Option<ShaderStage>,
    notes: Vec<String>,
}

struct SourceSpan {
    tag: TagHash,
    offset: u64,
    size: Option<u32>,
}
```

Requirements:

- preserve raw value/bytes beside interpretations
- unknown values remain visible and serializable
- decoder outputs and derived material expressions reference provenance
- Strict Tiger rejects hidden heuristic substitution
- one unknown binding/material/opcode must not prevent partial asset rendering

---

# 3. Core Data Model

## 3.1 Raw draw packet

P0 requirement. Stage/LOD must survive geometry decode through GPU scheduling.

```rust
struct TigerDrawPacket {
    geometry: GeometrySlice,
    raw_lod: RawLodCategory,
    raw_stage: RawRenderStageId,
    technique: TechniqueId,
    material: MaterialInstanceId,
    transform: Mat4,
    source: ProvenanceId,
}
```

`RawLodCategory` and `RawRenderStageId` are game/build-specific integers. Semantic mappings live in versioned adapters. Marathon currently exposes 26 stage boundaries/25 ranges; Destiny's documented 19-stage model must not be hardcoded into Goliath.

LOD selection uses decoded membership categories. Existing highest-detail hardcoding becomes adapter policy, not geometry truth.

## 3.2 Per-stage TechniqueDescriptor

Current per-stage extraction is preserved and expanded. Flat technique-level texture/TFX arrays are invalid because ABI differs by shader stage.

```rust
struct TechniqueDescriptor {
    technique_hash: TagHash,
    render_state: TechniqueRenderState,
    stages: Vec<TechniqueStageDescriptor>,
    alpha_mode: AlphaMode,
    raw_scope_mask: u64,
    semantic_stage_hints: Vec<EvidencedStageHint>,
    provenance: ProvenanceId,
}

struct TechniqueStageDescriptor {
    stage: ShaderStage,
    shader_hash: Option<TagHash>,
    shader_signature: ShaderSignature,
    resources: Vec<ResourceBinding>,
    samplers: Vec<SamplerBinding>,
    constants: Vec<ConstantBinding>,
    inline_constants: Vec<Vec4>,
    tfx_program: Option<TfxProgram>,
    raw_scope_mask: u64,
}
```

Resource ABI records:

- slot/register and shader stage
- texture dimension: 2D/array/3D/cube/unknown
- source format and intended color space
- sampler state per binding
- required/optional/unknown status
- missing-resource fallback policy

Vertex ABI records:

- stream/offset/stride/format
- position, normal, tangent
- every required UV/color stream
- bone indices/weights
- skinning, morph, or soft-deformation requirements
- unknown semantics/raw declaration

Technique pipeline-cache keys include family, pass, technique/permutation, vertex layout, render-target/depth formats, and GPU state—not only four render-state indices.

## 3.3 Shader analysis

TFX does not describe every compiled shader operation. Add evidence-producing shader analysis:

- shader hash/signature registry
- available reflection/resource declarations
- optional DXIL disassembly notes
- known output/packing signatures
- correlations across representative assets

Full shader decompilation remains non-goal.

## 3.4 Typed MaterialIR

Avoid another optional-field god object:

```rust
enum MaterialIR {
    Surface(SurfaceIR),
    Decal(DecalIR),
    ForwardSpecial(ForwardSpecialIR),
    Unknown(UnknownMaterialIR),
}

struct SurfaceIR {
    family: SurfaceFamily,
    layers: MaterialDependencyGraph,
    opacity: Expression,
    evidence: ProvenanceId,
}

struct SurfaceOutput {
    albedo: Vec3,
    normal: Vec3,
    perceptual_roughness: f32,
    metalness: f32,
    ambient_occlusion: f32,
    emissive: Vec3,
    opacity: f32,
    subsurface: f32,
    flags: u32,
}
```

Family-specific typed payloads remain separate. Logical `SurfaceOutput` is common contract, not proof of Tiger's packed G-buffer.

## 3.5 Deterministic family registry

No fuzzy winner chosen only by `match_score`.

```rust
trait MaterialFamilyDecoder {
    fn classify(&self, ctx: &ClassificationContext) -> ClassificationEvidence;
    fn decode(&self, ctx: &MaterialDecodeContext) -> Result<MaterialIR>;
}
```

Classification uses deterministic signatures/rules and returns all supported candidates, conflicts, evidence, and rejected conditions. `Unknown` is first-class. New family does not add another boolean/branch to universal shader.

Initial confirmed/migratable families:

- standard opaque / textureless solid
- static Gear Pattern
- Weapon Mod Condition
- Investment Decal
- compact hair
- alpha-tested/transparent only where render state and assets prove behavior

Speculative hologram, distortion, energy, multiplicative, and other families remain research candidates.

---

# 4. Functional Requirements

## FR-01 — Geometry, LOD, render stage, and vertex fidelity

Priority: P0; required.

- retain raw LOD/stage/source span through `TigerDrawPacket`
- inspect raw and semantic values independently
- route draws through validated planner
- use adapter-based LOD membership policy
- add vertex streams/skinning required by preview corpus
- never infer Destiny stage names as Goliath truth

Acceptance:

- every GPU draw traces to source, raw LOD, raw stage, technique, material
- stage/LOD survive CPU and capture dumps
- unsupported vertex ABI yields explicit partial/unsupported result
- Runner/skinned corpus is gated until bone ABI and transforms are implemented

## FR-02 — TFX Runtime v2

Priority: P0 foundation; extend current parser/evaluator.

- typed stack/value model
- preserve concrete raw scope IDs/masks
- registry for opcodes/externs with game/build ownership
- frame/view/object/material/Gear inputs when confirmed
- time, object/global/Gear channels, texture dimensions, instance inputs
- execution trace and expression dependency graph
- explicit `Complete`, `Partial`, `StoppedAtUnknown`, `Invalid` result
- no silent unknown-op no-op
- parser preserves undecoded remainder after unknown opcode

Stepping, pause, and time scrub are useful inspector features but may land after runtime correctness.

## FR-03 — Goliath Gear Dye

Priority: research-gated.

Confirmed baseline currently includes six palette outputs plus decoded color, roughness remap, and metalness remap. Implement only fields supported by Goliath evidence.

```rust
struct GoliathGearDyeIR {
    slots: [ConfirmedGearSlot; 6],
    raw_parameters: Vec<RawParameter>,
    evidence: ProvenanceId,
}
```

Destiny-style default/custom/locked, primary/secondary, clean/worn, detail transforms, specular AA, emissive, and subsurface fields remain adapter-specific hypotheses until observed in Goliath ABI. Data model may support extension without making these acceptance requirements.

## FR-04 — Gearstack/control-channel expressions

Priority: required where used.

Gearstack is per-fragment evaluation, not a CPU `decode(sample: Vec4)` operation.

IR describes:

- raw RGBA bindings
- per-family channel expressions/remaps
- masks, thresholds, inversion, branch selection
- provenance for each interpretation
- raw-channel debug output

Unknown mapping preserves RGBA without semantic labels. Heuristic mappings are Pretty-only.

## FR-05 — Gear wear and Weapon Mod Condition

Priority: required separation.

Rename:

```text
WeaponModWearMaterial → WeaponModConditionMaterial
```

Gear worn state describes target material response. Weapon Mod Condition describes scratches/grime/damage/projection/control behavior. They may share dependency nodes but must not be forced through a universal resolver or formula. Combination operators are family/shader evidence.

Existing Weapon Mod Condition decoding and procedural projections migrate without visual regression.

## FR-06 — Detail surface layers

Priority: conditional.

- support detail diffuse/normal when bindings and shader signature prove them
- combine normals in correct authored space/TBN
- derive transform and scale only from evidence
- do not require `base * tint + detail`, SSS, or specular AA algorithms without Goliath proof
- expose unclassified bound resources for research

## FR-07 — Static Gear Pattern

Priority: required migration.

Move confirmed current pattern behavior into typed static procedural layer. Preserve field texture, projection, transforms, stripe/contour controls, colors, material-ID gating, and provenance. Time-dependent behavior belongs to dynamic dependency system.

## FR-08 — TFX-driven dynamic surfaces

Priority: architecture required; individual effects research-gated.

First represent:

- TFX time/state dependencies
- coordinate/projection dependencies
- texture/field dependencies
- material-ID gating
- affected logical output expression

Names such as swirl, pulse, breathing, roughness flow, emissive flow, animated normal, or distortion are assigned only after shader/asset evidence. Pause/time scrub/speed override/debug become acceptance criteria when first confirmed dynamic family lands.

## FR-09 — Investment Decal

Priority: required.

Migrate confirmed selector/atlas/mask/detail/threshold/remap decoder from universal surface branch into `DecalIR` and authored stage routing.

```rust
struct DecalIR {
    family: DecalFamily,
    writes: Vec<EvidencedDecalWrite>,
    opacity: Expression,
    evidence: ProvenanceId,
}
```

Albedo is baseline confirmed target. Normal/roughness/metalness/emissive writes are capabilities, not requirements until proven. Ordinary/additive families likewise require observed assets/stages. Depth test/write/bias derive from technique state.

## FR-10 — Logical research MRT/G-buffer

Priority: required after MaterialIR/pass planner.

Initial logical layout may use:

```text
G0 RGBA16F: linear albedo.rgb, opacity.a
G1 RGBA16F: encoded/linear normal.rgb, roughness.a
G2 RGBA16F: metalness.r, AO.g, subsurface.b, auxiliary.a
G3 RGBA16F: emissive.rgb, reserved.a
GF R32Uint: material/debug flags
Depth: Depth32Float
```

Exact formats may change based on wgpu support/performance. Flags never alias floating opacity. This layout is a research contract, not claimed Tiger packing.

## FR-11 — Lighting and IBL

Priority: required after logical MRT.

- material evaluation separated from lighting
- directional light and shadow resolve
- environment rotation
- diffuse/specular environment approximation
- roughness-dependent specular response
- replaceable lighting model: Tiger approximation/reference GGX/debug Lambert

Direct and IBL may share one initial lighting pass. BRDF LUT/prefiltered cube are implementation choices, not Tiger-fidelity claims. Shadow participation derives from authored stage/state/pass plan, not every opaque draw.

## FR-12 — Alpha test and transparency

Priority: basic framework required; special families gated.

- alpha test remains opaque/MRT-capable and may participate in shadow when proven
- true transparency uses forward path after lighting
- honor authored blend/depth state
- deterministic transparent sort using camera depth plus stable tie-breaker
- scene color/depth sampling only for confirmed families

Hologram, distortion, multiplicative, energy, and premultiplied special handling remain candidates until signatures/assets prove them.

## FR-13 — Hair

Priority: confirmed compact-hair family migration.

Preserve current mask, alpha/cutout, normal, and roughness behavior in separate family. Anisotropic specular is deferred until evidence.

---

# 5. Strict and Pretty Modes

Strict Tiger:

- authored/decoded values only
- neutral documented fallback for unknowns
- visible evidence and unsupported state
- deterministic captures

Pretty Preview:

- may use current luminance/chroma roughness/metalness heuristics
- every heuristic reported in provenance/debug UI
- never overwrites decoded values or capture metadata

Strict is default for reverse-engineering captures. Pretty may remain default for casual preview only if UI clearly labels it.

---

# 6. Pass Planning and Render Graph

Free bitflags can encode invalid combinations. Use validated plan derived from raw stage, technique state, family, and adapter evidence:

```rust
struct DrawPassPlan {
    passes: SmallVec<[PlannedPass; 3]>,
    evidence: ProvenanceId,
    warnings: Vec<PlanWarning>,
}
```

Minimum justified graph:

```text
Prepare resources / evaluate TFX
→ optional confirmed shadow pass
→ opaque + alpha-tested logical MRT
→ confirmed decal passes
→ lighting + IBL
→ forward-special + transparent
→ bloom / tone map / present
→ selected debug overlay/output
```

Deferred until evidence/performance requires:

- depth prepass
- ordinary/additive decal families
- distortion composite
- fog/atmosphere
- particles
- UI HDR composition

Surface layers use dependency graph. No universal Gear resolution order. Cycles and missing dependencies produce explicit errors/partial results.

---

# 7. Preview Scene and Reproducibility

Visual comparison requires deterministic `PreviewScene`:

```rust
struct PreviewScene {
    camera: CameraParameters,
    framing: FramingPolicy,
    model_transform: Mat4,
    lights: Vec<PreviewLight>,
    environment: EnvironmentConfig,
    exposure: f32,
    background: BackgroundConfig,
    output_size: UVec2,
}
```

Capture manifest records:

- package/game build and adapter version
- source asset/tag
- Quicktag revision and decoder/schema version
- GPU, backend, driver when available
- scene/camera/environment/exposure
- strict/pretty mode and debug channel
- incomplete/unknown decoder states

Current GPU probe harness is foundation. Formalize it; do not rebuild from zero.

---

# 8. Inspectors and Debug Views

Existing Technique and TFX inspectors are extended.

Required early views:

- final, base/albedo, diffuse, AO, specular, HDR, normal
- raw bound texture RGBA and sampler/resource metadata
- raw/semantic LOD and stage
- material family candidates/evidence
- Strict fallback/provenance warnings
- draw pass plan

Add as pipeline lands:

- SurfaceOutput fields
- each MRT and integer flags
- direct diffuse/specular, IBL diffuse/specular
- decal writes
- alpha/transparent sort
- TFX live values, trace, dependencies, partial stop point

Do not block architecture on dozens of speculative views.

---

# 9. Code Architecture

Extract by stable ownership; avoid premature directory explosion.

```text
src/render/
    draw_packet.rs
    preview_scene.rs
    evidence.rs
    pass_plan.rs

    technique/
        descriptor.rs
        abi.rs
        signature.rs

    tfx/
        program.rs
        runtime.rs
        trace.rs

    material/
        ir.rs
        registry.rs
        gear_pattern.rs
        weapon_condition.rs
        investment_decal.rs
        unknown.rs

    gpu/
        compatibility_forward.rs
        logical_mrt.rs
        lighting.rs
        transparent.rs
        post.rs

    debug/
        views.rs
        capture.rs
```

Rules:

- preserve existing decoders, texture/cache code, inspector logic, and probe harness
- first extraction keeps compatibility-forward pixels stable
- `TigerCore` owns raw common structures
- `GoliathAdapter` owns Marathon mappings/signatures
- Destiny adapter exists only when supported by code/data; docs alone do not create runtime obligations
- asset graph resolution uses small typed resolvers, not new graph god object

---

# 10. Implementation Sequence

## PR-00 — Baseline and capture manifest

Freeze current reference output. Formalize existing GPU probes. Initial curated corpus may be ~20 assets; grow based on coverage, not arbitrary count.

Exit: deterministic scene/capture metadata, repeatable CPU/GPU dumps, known environment-dependent tests documented.

## PR-01 — Raw draw packet

Retain raw stage/LOD/source through GPU preparation. No intended visual change.

## PR-02 — Technique, resource, vertex ABI

Per-stage descriptor, raw scopes, shader signatures, binding dimensions/formats/samplers, vertex requirements, expanded cache key.

## PR-03 — Provenance + Strict/Pretty

Interned evidence, unknown propagation, neutral strict fallbacks, visible heuristic provenance.

## PR-04 — Typed MaterialIR + family registry

Deterministic classification. Migrate Gear Pattern, Weapon Mod Condition, Investment Decal. Keep compatibility-forward output.

## PR-05 — Validated pass planner

Derive schedules from stage/state/family evidence. Preserve forward renderer while pass semantics become inspectable.

## PR-06 — TFX runtime completion

Typed values, runtime inputs, unknown preservation, partial states, trace/dependencies.

## PR-07 — Logical SurfaceOutput + MRT

Implement logical outputs, `R32Uint` flags, MRT/debug inspection, compatibility fallback.

## PR-08 — Lighting/IBL

Move opaque lighting out of material shader; add replaceable approximation and shadow participation.

## PR-09 — Confirmed Goliath Gear

Implement only evidence-backed Gear slots/control expressions/detail fields. Research findings update requirements before code.

## PR-10 — Investment Decal stage

Remove confirmed Investment Decal work from universal forward branch; route and inspect writes.

## PR-11 — Alpha/hair/transparency

Cutout/shadow semantics, compact hair migration, transparent sorting and forward path.

## PR-12 — Confirmed dynamic effects

Implement evidence-backed TFX surface families, time controls, and debug outputs.

## PR-13+ — Additional families and optimization

Only after signatures/corpus prove need. Then profile batching, caches, packing, resource aliasing, and compilation.

---

# 11. Corpus, Metrics, and Gates

Corpus coverage categories:

- base/non-skin weapons
- weapon skins
- multi-material weapons
- confirmed Gear Dye variants
- Weapon Mod Condition
- Gear Pattern
- Investment Decal
- alpha-tested/hair/transparent
- TFX/time-dependent assets
- Runner/skinned geometry after vertex ABI support
- unknown/negative-control techniques

No fixed 30–50 count blocks early architecture. Grow corpus until each confirmed family/stage/ABI has positive and negative controls.

Metrics:

- 100% draws retain raw stage/LOD/source
- 100% techniques report per-stage ABI or explicit unsupported fields
- no silent unknown TFX opcode/binding/family
- no Strict-mode hidden luma/chroma material guess
- family/stage coverage reported by corpus and game build
- image thresholds calibrated per asset/channel after deterministic camera/exposure alignment

SSIM constants such as 0.97/0.92 are not global gates. Establish baselines from stable captures, then set per-channel tolerances. Performance gates require named target GPU/backend/resolution; until then record frame time, GPU memory, compile time, and stalls without arbitrary 1080p60/4K30 acceptance.

---

# 12. Anti-Patterns

- adding effect booleans to universal material uniform/shader
- creating material structs directly from one unexplained binding
- naming systems from appearance before semantics are known
- treating Weapon Mod Condition as Gear worn state
- reducing Gear Dye to RGB tint
- routing every raw stage through opaque forward path
- silently interpreting unknown channels/opcodes
- mapping Destiny semantics directly onto Marathon
- flattening per-stage shader ABI into technique-level arrays
- discarding raw IDs after semantic mapping
- using fuzzy family score as authoritative selection
- encoding flags and opacity in same floating G-buffer channel
- tuning visuals before deterministic PreviewScene exists

---

# 13. Phase Definition of Done

## Foundation complete

- raw draw packet/provenance reaches GPU
- per-stage technique/resource/vertex ABI inspectable
- Strict/Pretty split operational
- unknown/partial states explicit
- deterministic capture manifest operational

## Material architecture complete

- typed MaterialIR and deterministic registry
- existing Pattern/Condition/Investment Decal migrated
- compatibility-forward output retained during transition
- logical layer dependencies inspectable

## Deferred preview complete

- opaque/alpha logical MRT and integer flags
- lighting/IBL separated from material evaluation
- shadow participation planned from evidence
- Investment Decal routed outside universal material shader
- transparent forward path and stable sorting

## Goliath fidelity complete for supported corpus

- every supported asset reports raw stage/LOD/ABI/material provenance
- confirmed Gear semantics implemented; unconfirmed fields remain raw/unknown
- confirmed dynamic families use TFX runtime
- supported skinned assets use decoded vertex/skeleton ABI
- no hidden heuristic in Strict captures
- remaining unknown families/signatures quantified

---

# 14. Final Target Flow

```text
Tiger package
→ typed tag/resource resolution
→ geometry + raw LOD/stage + source span
→ per-stage TechniqueDescriptor + vertex/resource ABI
→ TFX execution/dependency result
→ deterministic material classification
→ typed MaterialIR / dependency graph
→ validated DrawPassPlan
→ logical SurfaceOutput or Decal/ForwardSpecial output
→ opaque/alpha MRT
→ confirmed decals
→ lighting/IBL/shadows
→ forward-special/transparent
→ bloom/tone map/present/debug
→ reproducible capture + provenance report
```

Core principle:

> Preserve raw Tiger facts first. Interpret through versioned evidence. Render partial truth visibly. Tune beauty only after semantics and scene are reproducible.

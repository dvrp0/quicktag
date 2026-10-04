# Tiger renderer roadmap

### Open KASHA chest-mask follow-up — October 4

- [x] Reproduce with the current production-renderer probe. Isolate opaque
  draws 0/1 (`80B152AA`) as the broad chest/shoulder pattern owner; isolated
  draw 7 (`80B152EC`) is hands. Stage2 omission preserves the broad pattern.
  Four small live probes only; no general test suite or new native helper.
  Evidence: `target/kasha-mask-isolation.json` and
  `docs/alkhahest-comparison/kasha-chest-mask-oct4.md`.
- [x] Trace omitted source operations: t7 RGBA layer selectors, derivative
  edge rejection, original composition order, and the explicit raw-position
  `TEXCOORD6.y <= 0` gate. Current chest draws remain `StandardSurface`;
  recent B152BE draw8 admission did not integrate this chest shader.
- [ ] Integrate the complete authored chest layer-mask/composition contract
  by source shader ABI, with packaged parameters/resources and exact input
  coordinates. Prove the reported side mask in displayed output. Full opaque
  shading's unresolved high View provider is separate from these known masks.
  Finding one side gate is not permission to cut primary color or every decal.
- [ ] Ship and verify that mask fix. No renderer change/build delivered by
  this investigation; this defect remains open.

### Delivered Channel view — October 4

- [x] Human clarifies four editable components: X/Y/Z/W.
- [x] Add Channel beside Lighting in the Models viewport. Discover used
  object/global channels from package technique programs and associated
  compute programs; hide unused declarations. Resolve names from package
  strings, with the existing name dictionary where available.
- [x] Apply finite vec4 edits through the original TFX programs, including
  every assembled geometry's object scope. Retain camera, lighting, base
  vertex/index buffers and source streams; rebuild parameter-dependent
  compute outputs. Reset restores original scope values.
- [x] Verify Rook Aftermarket Scratch: all 15 object declarations are read,
  plus three global slots. Package hash `7325BDBA` differs from the supplied
  `8325BDBA`; keep the package identity. Empty expressions, nine zero-bound
  values, six engine-initialized ones. No production channel IDs hardcoded.
  Evidence: `target/channel-probe/rook-aftermarket-scratch.json`.
- [x] One focused live control changes `7EC60CCB` to
  `[0.03125, 0.0625, 0.125, 0.25]`: 482 albedo and 475 lit pixels change.
  CPU/GPU mesh and source buffers remain exact; camera/light state stays exact;
  Reset restores all displayed RGBA bytes. NaN edits leave values/revision intact.
  Source wiring retains edits across mod rebuilds and carries them into exports.
  Evidence: `target/quicktag-model-probe/runner-rook-aftermarket-scratch-combined---channels.channels.json`.
- [x] Ship default release `target/release/quicktag.exe`, October 4 15:23:33 KST,
  95,127,552 bytes; SHA256
  `487c03b1dd63c5a6e712abe924150fe80fd8e306b31312c7b43752d7337a5f3e`.
  Guarded `--help` launch passes. Two focused package/GPU probes passed;
  ordinary test suites were not run.
  Evidence: `target/channel-release-delivery.json`.

### Delivered B152BE static source pass — October 4

- [x] Correct source audit: decode selectors11 differ from batch mode0.
  Angle-axis NaNs are clamped before the angle tests; a separate zero
  distance-gated quaternion axis still reaches Rsqrt(0) and NaN before row13.
  Evidence: `docs/alkhahest-comparison/b152be-source-oct4.md`.
- [x] Reproduce the original31-row/12-buffer compute adapter. Original
  math/control and exact typed widths preserved; Vulkan validation passes.
  Independently audit its paired B152B5 pixel program with no unresolved rows.
  Evidence: `target/runner-b152be-source-audit/artifact-proof.json`,
  `target/runner-b152be-pixel-audit`.
- [x] Human approves a mesh-derived influence radius for this exact static
  source. Preserve every other authored simulation parameter and original
  shader math. Derive the finite radius from the real packed positions and
  both authored centers; require original source-pose/frame proof.
- [x] Pass one focused source-pose/GPU tail/draw-omission control. All45,911
  positions match source packing exactly;91,822 frame directions stay within
  one unit. GPU radius2.3499825 covers both centers and every source point;
  all other tail bits are exact. Draw8 changes118 displayed albedo/lit pixels;
  native RGB is finite; restoring all image bits is exact.
  Evidence: `target/quicktag-model-probe/runner-thief-kasha-yokai-combined--b152be.b152be.json`.
- [x] Ship default build14:37:43 KST,120,363,520 bytes, SHA256
  `cd67d77c42b2b70f9de946e84c0147decf967f23b14e402fea496c73877315ed`;
  guarded launch passes.116/116 roots load;1,353/1,813 opaque and606/606
  stage2 admitted. Exactly one new draw; all prior admissions, geometry,
  stages, index formats and pipeline owners are unchanged. Remaining460
  opaque gaps:457 PS, three VS;84 partial receiver roots remain.
  Evidence: `target/runner-b152be-{release,population-delivery}.json`.
- [x] Audit remaining E778/E7A6 container/interface identity independently.
  E778 omits TC8; E7A6 has a distinct80-row resource graph. No B761/B762
  alias or copied simulation tail is justified. Original compute math still
  needs tracing; its paired E77B PS also lacks a shared numeric provider.
  Evidence: `docs/alkhahest-comparison/e778-source-oct4.md`.
- [ ] Complete the remaining universal material/decal fidelity gaps.
  Next shared blockers: eight View vectors at+0x470..+0x4E0, indexed value1,
  dimension-W/metadata selector contracts, and the three distinct VS/CS
  families. Current source audits do not recover their numeric writers.
  The approved B152BE radius policy resolves only that compute source; it is
  not permission to invent these other inputs. All606 stage2 operations are
  registered, but84 roots still have incomplete native opaque receiver coverage.

### Delivered C9C5/C9DD source pass — October 4

- [x] Audit distinct C9C5 interface and byte-identical C9DD/B762 compute graph.
  Independently parse paired C9C8 PS: TC8 is declared but has no executable
  reference, so the existing source-inert auxiliary image remains valid.
  Evidence: `docs/alkhahest-comparison/vox-source-oct4.md`.
- [x] Reproduce source-pinned adapters and independently audit C9C8 PS.
  Original math/control/I/O and exact typed widths preserved; Vulkan validation
  passes. C9DD adapter is byte-identical to B762, with separate source identity.
  Registry548 preserves all547 earlier contract objects and source prefix.
  Evidence: `target/runner-vox-source-audit/{artifact-proof,pixel-registry-preservation}.json`.
- [x] Integrate and verify through a focused normal-renderer probe.
  Vox draws 10/11 use their original VS/CS/PS and real PS texture. All 42,632
  positions match source quantization exactly; 85,264 frame directions stay
  within one unit; evaluated geometry-scoped TFX tail bits match the GPU.
  Omission changes 1,771 displayed albedo/lit pixels; restore is bit-exact.
  The initial tail oracle incorrectly compared evaluated TFX to raw inline
  rows; the corrected oracle checks actual authored geometry-scoped evaluation.
  Evidence: `target/quicktag-model-probe/runner-vandal-vox-nocturna-combined--c9c5-shared.c9c5.json`.
- [x] Share identical static producer inputs within each geometry. Match exact
  executable/cache ABI and every constant bit, including signed zero/NaN;
  all source/object buffers already share ownership within that geometry.
  Vox dispatches 2→1; exported RGBA has zero changed pixels. Authored draw order
  stays exact. Evidence: `target/runner-vox-source-audit/shared-producer-rgba.json`.
- [x] Ship verified default build and refresh remaining-draw census.
  Built 14:16:39 KST, 93,723,136 bytes, SHA256
  `277fe2003a36621a0a489be224ee34dc5f529f669935a941c1c6e3d3fe9460fb`;
  guarded `--help` passes. Exactly two new opaque draws; every prior admission
  and geometry/stage owner stays exact. Census: 116/116 roots load,
  1,352/1,813 opaque, 606/606 stage2. Remaining 461: 458 PS and three VS;
  84 partial receiver roots remain.
  Evidence: `target/runner-c9c5-{release,population-delivery}.json`.
- [x] Integrate KASHA's distinct B152BE producer under the approved finite
  static-radius policy; delivered in the newer pass above.
- [ ] Continue shared numeric providers. Full material/decal fidelity remains
  incomplete; registration of every stage2 does not prove complete receivers.

### Delivered EC03/EC0B source pass — October 4

- [x] Audit exact six-resource vertex interface, authored zero displacement
  row and unused declaration padding. EC0B matches A2D3 executable math;
  retain its distinct 15-row source/module identity.
  Evidence: `docs/alkhahest-comparison/astrophage-source-oct4.md`.
- [x] Reproduce exact vertex/compute adapters and independently audit EC60 PS.
  Original math/control, output stores, interface and typed widths preserved;
  both adapters pass Vulkan validation. EC60 has no unresolved numeric input.
  Registry now has 547 PS contracts; all 546 prior contracts/prefix stay exact.
  Evidence: `target/runner-astrophage-audit/artifact-proof.json`,
  `target/runner-astrophage-audit/pixel-registry-preservation.json`.
- [x] Integrate the source-paired static path and prove actual live output.
  Draws18/19 execute their exact VS/CS/PS through normal renderer bindings.
  All 46,729 source positions match exactly, 93,458 frame directions stay
  within one packing unit, guards stay zero. Omission changes 91 displayed
  albedo/lit pixels; native RGB is finite; image restoration is bit-exact.
  Evidence: `target/quicktag-model-probe/runner-triage-astrophage-combined--ec03.ec03.json`.
- [x] Ship the verified default build and refresh remaining-draw census.
  Built13:56:13 KST,93,690,368 bytes, SHA256
  `42c5986e47d9e3f3cff4a69ebd54e345f5c735b22a2313d878ba64e3ac4eaade`;
  guarded `--help` passes. Exactly two new opaque draws; all prior admissions
  and geometry/stage ownership stay exact. Census:116/116 roots load,
  1,350/1,813 opaque,606/606 stage2. Remaining463:458 PS and five VS;
  84 partial receiver roots remain.
  Evidence: `target/runner-ec03-{release,population-delivery}.json`.
  Shader admission does not establish complete skin appearance.
- [ ] Continue remaining source families and shared numeric providers.

### Delivered second cloth source pass — October 4

- [x] Audit B15221/B144DD source pairing. The B152 vertex interface differs
  from B1CB; B144DD raw and adapted compute bytes are identical to verified
  A9C4. Retain each exact payload identity. Pixel B144D4 passes independent
  resource/numeric audit; TC8 is declared but unused.
  Evidence: `docs/alkhahest-comparison/acid-cloth-b152-b144dd-oct4.md`,
  `target/runner-acid-cloth-pixel-audit`.
- [x] Reproduce source-pinned B152 adapter and B144DD artifacts; preserve
  math/control, the original vertex interface and correct buffer strides.
  Evidence: `target/runner-cloth-acid-audit/artifact-proof.json`.
- [x] Add a shader-cache invariant: shared descriptor identity requires the
  same executable stage and SPIR-V hash. Distinct math must have a distinct
  cache identity; exact translated aliases still retain package source hashes.
- [x] Integrate source-paired programs and verify draws7/17/20 through one
  normal-renderer probe. All 43,084/8,713/10,010 source positions match exactly,
  frame packing differs by at most one unit, guards stay zero. Omission changes
  1,526 displayed albedo/lit pixels; native RGB is finite; restoration is
  bit-exact. Evidence:
  `target/quicktag-model-probe/runner-thief-acid-abyss-combined--cloth-variant.cloth-variant.json`.
- [x] Ship with AA060B below. All 545 previous PS contracts and registry
  prefix remain exact; registry now has 546. One final combined census after
  both live proofs preserves every prior admission and geometry/stage owner.

### Delivered AA060B producer pass — October 4

- [x] Correct the earlier zero-axis audit. Row24 reaches only an unordered
  angle clamp; the actual rotation axes come from authored row14 and rows
  29/30. Original static row13 zero-weight output is source-invariant with
  preserved authored tail. No guessed row24 override is required.
  Evidence: `docs/alkhahest-comparison/aa060b-static-invariance-oct4.md`.
- [x] Reproduce exact original compute artifact and the 13-buffer ABI.
  Keep 34 authored rows; pad only its unused 35th declared row.
  Evidence: `target/runner-aa060b-audit/artifact-proof.json`.
- [x] Focused live proof: 47,411 source positions exact, frame packing within
  one unit, authored tail GPU bits exact, original zero row24 retained. Actual
  317-record color decode/clamp is exact. Original PS does not read TC8:
  complementing color must preserve all image bits. Separate draw omission
  changes 8,901 albedo / 8,902 lit pixels; restore is bit-exact. The initial
  control incorrectly required color influence for an unused source varying;
  the corrected generic probe checks source use, not a character/skin exception.
  Evidence: `docs/alkhahest-comparison/aa060b-live-source-contract.md`.
- [x] Default-feature shipping build and guarded `--help` pass. Built
  09:50:29 KST, 93,604,352 bytes, SHA256
  `f9a7a31513b84117bc6a29dce9b9fdb91a50df6e368c8d4e1d5a3846ea1177bd`.
  Record: `target/runner-acid-aa-release.json`.
- [x] Combined census: four new draws, all prior admissions/owners exact.
  116/116 roots load, 1,348/1,813 opaque and all 606 stage2 draws admit.
  All 22/22 C107 draws now admit. Remaining 465 opaque gaps: 458 PS, seven VS.
  Still 84 partial receiver roots; whole-skin appearance remains incomplete.
  Evidence: `target/runner-acid-aa-population-delivery.json`.
- [ ] Continue the seven vertex gates and shared high-View/non-View providers.
  Source audit and shader admission are not complete material/decal fidelity.

### Active hair-variant pass — October 4

- [x] Audit all 12 remaining VS-gated draws: seven distinct source families;
  none may be inferred from a character, row count or shared IA layout.
  Evidence: `docs/alkhahest-comparison/remaining-vertex-oct4.md`.
- [x] Audit original 14582/1458D pixel contracts for source-paired 14580 VS.
  Both pass independent resource/numeric source gates; no guessed extern fill.
  Evidence: `target/runner-acid-hair-pixel-audit`.
- [x] Reproduce exact 14580 VS artifact, validate source I/O and math, confirm
  TC8 is inert in both paired PS. Keep a distinct program identity/cache key;
  reuse the verified A9DE producer only through the actual source pairing.
- [x] Verify new source draws 21/22 through the normal renderer. All 10,010
  positions match source quantization exactly; 20,020 frame directions differ
  by at most one packing unit; guards stay zero. Draw omission changes 1,301
  displayed albedo and 1,302 lit pixels; restoration is bit-exact. Native RGB
  is finite. One focused live probe; no ordinary suite.
  Evidence: `docs/alkhahest-comparison/hair-14580-source-contract.md`.
- [x] Rebuild default-feature shipping app and pass guarded `--help`. Built
  09:19:42 KST, 93,488,128 bytes, SHA256
  `727484ef75e36275f9658701917f194969d086cb8aafcb33611ffa73f2c3835d`.
  Record: `target/runner-acid-hair-release.json`. Registry now has 545 PS
  contracts; all 543 previous contracts and registry prefix remain exact.
- [x] Refresh census only after live proof. Exactly two opaque draws add;
  prior admissions and geometry/stage ownership remain exact. All 116 roots
  load, 1,344/1,813 opaque and all 606 stage2 draws admit. Remaining 469 opaque
  gaps: 458 PS, 10 VS and one AA060B producer/input gate. Still 84 partial
  receiver roots. Evidence: `target/runner-acid-hair-audit/population-delivery.json`.
- [x] Extend the shared high-View audit through the installed executable,
  read-only. The protected on-disk binary yields no coefficient table or
  writer. B032's high rows feed a shared directional lighting lookup; they
  cannot be treated as an inert mask or filled with zero/unit vectors.
  Evidence: `docs/alkhahest-comparison/high-view-provider-oct4.md`.
- [x] Audit the four non-View pixel gaps and KASHA's B152B5 handoff. Existing
  arithmetic is sufficient; missing inputs are indexed-value 1, texture
  dimension W, selector 13 and the distinct 31-row B152BE producer. These
  audits are source evidence only, not live renderer fixes.
  Evidence: `docs/alkhahest-comparison/nonview-pixel-oct4.md`.
- [ ] Continue remaining exact source-pair families and shared provider work.
  Eligibility is not image parity; full material/decal fidelity stays open.

### Active receiver-coverage pass — October 4

- [x] Audit zero-class source outputs. Packed-normal material class is not
  coverage: six still-unregistered draws explicitly write RT1.a=0 while their
  RGB carries a normal; other families have dynamic class expressions.
  Source probe: `target/runner-cloth-audit/remaining-receiver-alpha-probe.json`.
- [x] Remove class-alpha coverage assumptions from generic receiver import,
  native receiver mask, displayed source color and viewer material projection.
  Detect the all-zero cleared packed-normal MRT; keep authored class unchanged.
  A modified decal pixel must already have an opaque receiver. No new target
  allocation, material defaults or original shader edits.
- [x] Focused retained-frame probe: 28,095 receiver pixels retain coverage and
  projected normal/properties/albedo exactly when only class is zeroed.
  A synthetic decal normal write into cleared background stays excluded.
  Restore all native MRT and displayed bits exactly. Original 1024x640
  REFRESH_RESIST decoded RGBA remains exact (zero changed pixels).
  Evidence: `target/quicktag-model-probe/runner-thief-refresh-resist-combined--zero-class.zero-class.json`,
  `target/runner-cloth-audit/zero-class-rgba-preservation.json`.
- [x] Rebuild default-feature executable; guarded `--help` passes. Built
  09:00:09 KST, 93,376,000 bytes, SHA256
  `890614968701005ceaf0452d408b534cff9ebb64745ed91cea97d5dc7c47ea00`.
  Record: `target/runner-zero-class-release.json`. Material registry and
  admission counts are unchanged; no redundant CPU census or ordinary suite.
- [ ] Finish shared high-View provider, remaining vertex and non-View pixel
  integration. These projection fixes do not admit the remaining 465 opaque draws
  or establish that all decals/materials now look correct.

### Active cloth pass — October 4

- [x] Independently audit B15080 (45 rows) and A9C4 (46 rows): preserve
  authored rows14 onward; original row13.x selects zero deformation. Distinct
  source identities select each producer, never a character or skin name.
- [x] Reproduce Vulkan-valid B1CB VS and both compute artifacts. Independently
  compare original math/control instructions and vertex I/O; validate dense
  descriptor layouts and all typed buffer strides, including uint4 t3.
  Evidence: `scripts/build_runner_cloth.py`,
  `target/runner-cloth-audit/cloth-artifact-proof.json`.
- [x] Audit E40B/1451C material contracts paired with B1CB. Other B1CB
  materials remain withheld pending their independent high-View proof.
- [x] Verify both producer families through normal renderer: source pose,
  finite outputs, draw influence on displayed albedo/lit images, exact
  restoration. Three focused probes: MIDNIGHT DECAY, REFRESH_RESIST and
  ACID ABYSS. All 46,128/43,157/43,084 source positions match exactly;
  normal/tangent packing differs by at most one unit; guards stay zero.
  Omission affects 6,992/15,405/12,478 displayed albedo and
  7,011/15,421/12,508 lit pixels. Restoring draws restores all image bits.
  Evidence: `docs/alkhahest-comparison/cloth-static-source-contract.md`.
- [x] Rebuild default-feature shipping executable; guarded `--help` passes.
  Built 08:38:31 KST, 93,375,488 bytes, SHA256
  `3777e76ea48f6e3ba30020aa97c08a4d8069ba6e49d1db42c65ead2de30a7f5d`.
  Record: `target/runner-cloth-release.json`. All 541 prior material contracts
  and registry prefix remain exact; registry now contains 543 PS contracts.
- [x] Refresh CPU admission census after live proof; previous admissions stay
  exact. Seven additional opaque draws admit across three roots. All 116 roots
  load; 1,342/1,813 opaque and all 606 stage2 draws admit. Remaining 471 opaque
  gaps: 458 PS, 12 VS, one AA060B producer/input gap. The 19 unproved B1CB
  materials now correctly report the PS gate. Still 84 partial receiver roots;
  zero whole-preview rejection candidates. Evidence:
  `target/runner-cloth-audit/population-delivery.json`.
- [x] Resolve AA060B separately in the delivered producer pass above. The
  earlier row24 output-poison claim was incorrect; original clamps make that
  intermediate NaN inert. Full source pose and displayed influence now pass.
  This still cannot claim all skins or decals visually complete.

### Active vertex-color pass — October 4

- [x] Audit C107 ownership: 22 draws, six roots, three distinct source compute
  producers. Color data belongs to each geometry; the six record counts differ.
- [x] Receive explicit user approval for RGBA8-UNORM decoding and per-buffer
  `record_count - 1` static-preview clamp. Preserve authored final records. This
  is a viewer policy, not a claim about an observed native runtime writer.
- [x] Audit all 11 paired pixel programs independently: nine consume TC8;
  E007/AA0601 do not. Exact source pairing and numeric gate proofs pass.
  Evidence: `target/runner-vertex-color-pixel-audit`.
- [x] Deliver source-pinned six-buffer C107 VS and B4CB static producer through
  normal renderer; upload real geometry color data, retain original shader math.
- [x] Verify actual GPU color decode/clamp, generated source pose, live displayed
  color influence and exact restoration with focused Assassin/Vandal CRYO SHIFT
  probes. Rebuild default-feature shipping executable after they pass.
  Assassin: all 52,367 positions exact; Vandal: all 45,441 positions exact.
  Normals/tangents differ by at most one packing unit; zero guards preserved.
  GPU decode and cb1[4].w bits match real 2,770/1,812-record buffers, including
  their final `000000FF` record. Complementing only uploaded RGB changes
  1,392/10,420 displayed albedo and 1,437/10,923 lit pixels; RT1/RT3 remain
  unchanged and restoring actual colors restores every MRT/displayed bit.
  Default-feature executable rebuilt 04:07:00 KST, guarded `--help` passes:
  93,047,808 bytes, SHA256
  `2262823efc3e4dfef4eb2db205d0abdecdb3b1c9458211f7fbce1b6b3447cc9d`.
  Record `target/runner-vertex-color-release.json`; 541 surface programs,
  all 530 prior contracts exactly preserved. Two focused GPU probes; no
  ordinary test-suite run. This proves source consumption, not total fidelity.
- [x] Refresh CPU admission census and correct its counting gate to mirror the
  live vertex-color source/producer check. C107 admits 17/22 draws across four
  B4CB roots. Five AA060B/B15080 draws remain withheld. All 116 roots load;
  1,335/1,813 opaque and all 606 stage-2 draws admit, with 84 partial receiver
  roots and zero whole-preview rejection candidates. Remaining 478 opaque gaps:
  439 PS, 34 VS, five producer/input gaps. Prior admissions are preserved.
  Evidence: `target/runner-vertex-color-audit/population-delivery.json`.
- [x] Resolve B15080 independently: four source-owned MIDNIGHT DECAY draws
  pass live original-producer/static-pose proof. Never substitute B4CB.
- [x] Resolve remaining AA060B draw independently; its static normalization
  source trace and full live outputs pass with the original zero row24.
  See the delivered AA060B producer pass above; no axis override is used.
- [x] Resolve B1CB/A9C4 original static graph for two independently audited
  material contracts. Preserve original row13.x zero-delta mask and authored
  tail; live proof establishes finite pose/material output. TC8 remains unused
  by paired PS. The 19 other B1CB draws still need material input proof.

### Active procedural-body pass — October 4

- [x] Identify source-selected B761/B762 pair: 36 draws across 14 catalogue
  roots. Authored IA strides are 24/4 bytes; catalogue's 76-byte value describes
  the generic reconstructed vertex, not the native input streams.
- [x] Translate and validate the distinct seven-buffer B761 VS and 81-row,
  16-binding B762 compute shader. Source t4 fetch reads all four uint lanes;
  its scalar reflection base type must not select scalar SSBO lowering.
- [x] Audit 14 paired PS programs independently; BA69/BA74 remain unsupported
  by the high-View SSA proof. A decoded Complete status does not mean admission.
- [x] Deliver B761/B762 through the normal renderer with 14 original PS
  contracts. All 16 source PS interfaces leave TC8 unused; signature ID 7 is
  TC7 from authored NORMAL.xyz. Keep the inert auxiliary float4 image bounded;
  do not mistake a declared TC8 for a read. Preserve the original authored
  procedural tail, including zero normalization axes. The existing explicit
  static wind direction makes the delta finite before row20 selects zero motion.
- [x] Verify focused WEAVErunner/FINELINE-XS live body controls: all
  42,061/42,071 source positions exact; normals and tangents within one packing
  unit, guards zero. Source draws affect 6,230/6,243 displayed albedo pixels and
  6,245/6,246 lit pixels; omission/restoration restores all MRT/displayed bits.
  Existing WEAVErunner hair control still passes. Three focused GPU probes;
  no ordinary test-suite or full GPU catalogue rerun.
- [x] Rebuild default-feature executable October 4 03:30:29 KST; guarded
  `--help` passes. 91,897,344 bytes, SHA256
  `c3f795184c7f806b0d18f806ee3886e3a58c2c5089a0c8bdd64393bc4ccfd44f`.
  Record: `target/runner-procedural-body-release.json`. The 530-entry surface
  registry preserves all 516 prior contracts exactly.
- [x] Refresh one CPU catalogue probe: all 116 roots load; 1,318/1,813 opaque
  draws and all 606 stage-2 draws admit. B761 family admits 33/36 draws; remaining
  three are Vandal SHADOW INDEX, PS BA69/BA74. Source admission does not prove
  every skin's final appearance. Remaining 495 opaque gaps: 439 PS / 56 VS;
  84 roots still mix native and generic opaque receivers.
- [ ] Resolve BA69/BA74 independently; high-View SSA proof remains unproved.
- [ ] Integrate remaining source-selected VS families, starting with C107
  (22 draws / six roots, including Assassin/Vandal CRYO SHIFT) and B1CB
  (22 draws / ten roots). Keep exact source identities, producers, and constants.

### Active appearance audit — October 3

**Latest user validation before the 22:00 build:** Thief hair is fully resolved;
Vandal hair remains incorrect. Keep the shared hair work open. Investigate
the geometry-selected compute/vertex/pixel contracts and their real auxiliary
inputs, not character names. Production dispatch must follow payload identity
and authored resource ownership; distinct handling requires source evidence.

- [x] Identify the separate source-selected B84F/B7BC family in Vandal's hair
  geometry: 14 draws across 14 catalogue roots. B84F writes 120 vertex constants
  and declares 149; B7BC has 133 constants, raw source streams, typed palette/frame
  buffers and a Frame texture/sampler. It does **not** have B762's cb1. Equal
  descriptor shapes or character ownership must not select a different program.
- [x] Audit five additional exact pixel contracts paired with B84F: B860,
  DB93, BBBB, B54B and D201. Source metadata dependencies have independent
  zero-gate proofs; retain original arithmetic and texture/sampler bindings.
  `target/runner-secondary-hair-pixel-audit/audited-records.json`. Generated
  registry now has 516 programs including BA67's two pixel contracts; all 509
  prior contracts remain exactly unchanged.
- [x] Deliver B84F/B7BC and these pixel contracts through the normal renderer,
  verify packed static source pose and authored color influence with focused
  probes, rebuild the shipping executable. CRYO SHIFT and WEAVErunner submit
  their original source-paired hair programs. All 9,306/9,294 positions match
  exactly; 18,612/18,588 directions match within one packing unit. Color controls
  change 1,681/2,424 displayed albedo and lit pixels, keep RT1/2/3 identical and
  restore every output bit. These checks prove live consumption, not every
  remaining skin's final fidelity.
  The first WEAVErunner GPU readback exposed five unlowered raw source buffers:
  SPIR-V retained texel-buffer descriptors while the renderer bound storage
  buffers. Corrected all 12 buffer-image resources to storage buffers; the
  builder now rejects any retained texel-buffer variable. All 16 B7BC catalogue
  draws share the exact same authored rows 21..132 and unresolved input rows;
  `target/runner-secondary-hair-audit/source-population.json`.
  The Cryo fixture also used its component tag for Pattern runtime inputs;
  probe scope now follows the selected assembly Pattern, as the live UI does.
- [x] Admit the distinct BA67 hair vertex source with B7BC for SHADOW INDEX
  and Achromatic Rush. Equal 120-row images do not establish B84F equivalence.
  BA67 omits TEXCOORD7; BB6A/E8A8 do not read TEXCOORD8. Their own authored
  textures, samplers and RGB rows 121..123 remain intact. Focused controls verify
  all 9,294/10,726 source vertices, 2,481/1,560 displayed color changes, unchanged
  protected MRTs and exact restoration.
  Monoclast and WHITE RABBIT have body-owned draws, not missing B84F hair draws.
- [x] Rebuild default-feature shipping executable at 22:00:01 KST; build and
  guarded `--help` pass. 90,614,784 bytes, SHA256
  `ecce76aa573b569eac72c640e197c8457688225c9ab9209afa26a2b9e70ea115`.
  Record: `target/runner-secondary-hair-release.json`. Four focused live probes;
  no full GPU catalogue or ordinary test-suite rerun.
- [ ] Resolve B84F-paired DE21's required high View constants independently.
  Its failed SSA gate remains visible; do not inject a guessed universal vector
  or mark every Vandal hair variant fixed from the five admitted contracts.

The user reports remaining defects in KASHA YŌKAI masking/decals, Sentinel
texturing, Thief/Vandal hair, Assassin/Vandal CRYO SHIFT bottoms and Triage
decals. The other working renderer is closed source. Reopen the audit around
actual texture ownership, material evaluation, receiver coverage and displayed
composition. Missing capture evidence does not establish that these defects
require game runtime data; investigate packaged resources and shared renderer
logic first. The full goal remains incomplete: the latest October 4 CPU census
records 478 opaque source-contract gaps and 84 partial receiver roots. The
03:31 census before C107 delivery recorded 495 gaps. The historical
census before MIDNIGHT/B7BC integration recorded 544 opaque gaps. Its two WHITE RABBIT
hair input failures were overstrict validation: a referenced zero tangent is
valid under original source packing. Both draws now pass the focused live probe.
Source admission and intermediate
MRT changes are not evidence of correct final appearance.

- [ ] Audit shared hair texture selection and authored color handling.
  The generic hair evaluator forces gray/white and ignores authored colors.
  Native strand integration is delivered for the exact A9DB/A9DE vertex/compute
  pair and original A9D5 pixel shader. The first live probe exposed a rejected
  constant image: TFX writes 51 vertex rows, while SPIR-V declares 80 and
  reads only row50. Preserve those 51 rows and pad the unused tail. Cryo
  also uses solid-hair PS A9D3, now integrated with its own six-texture/170-row
  proof; other hair programs still need source/live verification.
  The corrected focused live probe encodes Cryo strand draw15. Decoded RGBA
  changes2,095 pixels within head bounds `[496,48,558,142]`; the remaining
  653,265 pixels are identical. Existing census data shows all19 A9D5 draws
  share the zero row142 gate and51-row VS image. Proof and limitations:
  [static hair source contract](hair-static-source-contract.md). The refreshed
  census below tracks admission separately from complete hair appearance.
- [x] Integrate solid-hair A9D3 with its original170-row/six-texture contract,
  3D slot5, source-pinned descriptor/bias-only artifact and metadata gate136/145.
  Both hair draws now use their exact A9DE producer. One8.10-second focused
  live control changes1,312 native RT0 pixels and895 displayed albedo/lit
  pixels; RT1/2/3 and restored outputs are bit-identical. Independent readback
  checks all9,503 source vertices: packed positions match exactly and19,006
  normal/tangent directions match rational octahedral projection within one
  quantization unit; output guards stay zero. Report:
  `target/quicktag-model-probe/runner-thief-cryo-shift-combined--solid-hair-control.hair-control.json`.
- [x] Refresh116-root CPU admission census once (70.20seconds): zero load
  failures,1,269/1,813 opaque source contracts,606/606 stage2 contracts,
  544 remaining opaque contracts across104 skins and84 partial receivers.
  The35 hair contracts include19 strands and16 solid draws;33 pass current
  static input checks, while Thief WHITE RABBIT's two hair draws fail them.
  Withhold invalid hair before GPU construction so one unproved input image
  cannot reject the whole model. CPU source admission is not live appearance.
  Log: `target/runner-appearance-solid-hair-catalogue.log`;
  [updated remaining source draws](remaining-runner-draws.md).
- [x] Resolve WHITE RABBIT's overstrict static input gate. Its palette bounds
  are valid; vertex8156 has one referenced zero tangent. Preserve original
  numeric min/max packing rather than synthesize a tangent. The6.32-second
  live probe submits both hair draws and independently checks11,310 vertices:
  positions exact,22,620 directions within one quantization unit, zero tangent
  exactly `0x80018001`, finite RGB and bit-exact restoration. Hair colors change
  1,419 native RT0 pixels and1,417 displayed albedo/lit pixels; RT1/2/3 unchanged.
  Report: `target/quicktag-model-probe/runner-thief-white-rabbit-combined--white-hair-control.hair-control.json`.
  Genuine invalid source images are withheld before GPU construction.
- [x] Deliver the verified solid/strand hair increment: default release build
  and `--help` pass, October3,16:03:59 KST,65,550,848bytes, SHA256
  `f96fd270878e0e090d903e79190308b70130607d27f7ac76c5fee38019500ad3`.
  Record: `target/runner-appearance-solid-hair-release.json`. Complete appearance
  and MIDNIGHT DECAY's distinct hair program remain unverified.
- [x] Independently audit MIDNIGHT DECAY's E0B5 source:213 authored rows,
  ten textures,3D slot8 and three original samplers. Unknown row157 is proved
  output/control inert only behind authored zero row166.x. Generate its original
  PS through the shared surface registry, retaining the exact Hair VS/CS pair;
  all508 existing surface contracts remain unchanged. Source audit/artifact logs:
  `target/runner-appearance-midnight-hair-{source,artifacts}.log`.
- [x] Verify MIDNIGHT DECAY through the normal renderer: original E0B5 submits
  draw22 with all ten textures, shared A9DB VS and exact A9DE producer. Its
  original PS does not read TEXCOORD8, so the existing inert auxiliary policy
  remains valid. The6.02-second color control changes1,022 native RT0 pixels
  and854 displayed albedo/lit pixels; protected MRTs and restored outputs are
  bit-identical. All8,363 source positions match exactly and16,726 directions
  differ by at most one packing unit. Report:
  `target/quicktag-model-probe/runner-thief-midnight-decay-combined--midnight-hair-control.hair-control.json`.
  The first attempts stopped at pending textures: the fixture preloaded generic
  selections, not the newly admitted resource graph. Fix fixture preloading
  from native surface/decal inputs; live UI already retries pending resources.
- [x] Deliver E0B5 plus the preceding solid/strand hair fixes. Default release
  build and `--help` pass, October3,20:35:36 KST,65,611,776bytes, SHA256
  `09b149f7feccea10c9ca47295f7cb4b652191ebe82ba3da3b02daf9f4bfcbf66`.
  Record: `target/runner-appearance-midnight-hair-release.json`. The116-root
  census above predates this one additional live PS; it is not a new census.
- [x] Verify the live strand color path with one focused9.65-second control:
  changing only authored hair colors changes1,285 native RT0 pixels and876
  displayed albedo/lit pixels. RT1/2/3 remain identical, changed RGB is finite,
  and restoration recovers all native/displayed bits. The encoded draw uses
  A9DB/A9D5 with its exact A9DE compute dispatch. Report:
  `target/quicktag-model-probe/runner-thief-cryo-shift-combined--hair-control.hair-control.json`.
- [x] Rebuild and ship the verified strand increment. Default release build and
  `--help` pass: October3,15:25:24 KST,89,822,720bytes, SHA256
  `358894614a8c48c39bb02c645319d9d8b4b2b56e074a8283694f9a991cee4bd8`.
  Record: `target/runner-appearance-hair-release.json`. Solid hair, other
  distinct hair shaders and the remaining body/decal appearance defects stay
  open. This executable supersedes the13:47 receiver-only delivery.
- [ ] Audit decal masking and final composition over partial native receivers.
  KASHA's B152F7 is already the original live stage2 program: it reads receiver
  RGB, not receiver class alpha. The imported `.67` class matches the original
  opaque programs. The broad shoulder pattern persists with stage2 disabled;
  opaque owners B152AA/C3/E0/EC still use incomplete generic evaluation and
  consume high View mask fields. Investigate those original opaque programs;
  changing the already-live decal dispatch or receiver class has no source basis.
- [x] Remove the live no-native-opaque decal suppression that still remained in
  `ModelRenderer`, despite earlier candidate reports. Create native targets for
  decal-only graphs; import generic opaque receivers into the native material
  ABI before ordered stage2 execution. Snapshot receivers and project only
  native opaque or stage2-modified pixels, preserving untouched generic output.
- [x] Verify displayed stage2 influence with three focused live probes: Recon
  White Rabbit gains 478 albedo/lit pixels, Triage WEAVErunner gains 3,388 and
  Thief KASHA YŌKAI gains 585. All three have zero changes outside projection
  coverage and bit-exact restoration. These checks prove decal composition,
  not complete material fidelity or the reported KASHA chest mask.
  Reports: `target/quicktag-model-probe/*--receiver-bridge.receiver-composition.json`;
  logs: `target/runner-appearance-{receiver,triage,kasha}-live.log`.
- [x] Preserve Thief WEAVErunner decoded RGBA exactly against the Float48
  checkpoint with one simple probe (`target/runner-appearance-preserved.log`).
- [x] Initialize all 256 static global TFX channels as Alkahest does: authored
  prefix followed by ONE, retaining explicit inputs (including Unknown).
  Absent interpreter values remain Unknown; no unknown runtime writer is
  replaced. The registered native contracts have no global-channel gaps.
- [ ] Audit shared generic material evaluation for black/mistextured surfaces.
- [x] Map108 unregistered vertex draws to their source producers:51 geometries,
  35 roots; all108 paired PS are also unregistered. B1CB/A9A9C4 owns22 draws
  across10 skins with46-row/13-descriptor producer topology. B761/B762 owns36
  with81-row cb0 plus cb1/sampler/t10, while B84F/B7BC owns14 with133 rows.
  C107 owns22 split across13/45/34-row producers. Do not alias these by layout.
- [ ] Integrate remaining vertex/pixel/producer pairs with real authored
  auxiliary inputs where PS consumes TEXCOORD8. B1CB/A9A9C4 is the smallest
  broad family; current zeroed Hair auxiliary cannot be reused for active TC8.
- [x] Ship the demonstrated receiver/global-table fixes after the four focused
  probes above. Default-feature release build and `--help` pass: October 3,
  13:47:32 KST, 89,777,152 bytes, SHA256
  `bce773a4a313883820508e65ede9c3208ff3b2680f415369b0aab79fb08764bc`.
  Record: `target/runner-appearance-receiver-release.json`. Mixed receiver
  snapshots add 80 bytes/pixel, included in the existing 256 MiB target cap.
- [ ] Complete the remaining appearance defects. The delivered receiver fix
  does not fix generic hair/material evaluation or prove the KASHA chest mask.

### Remaining non-View numeric inputs — source diagnosis

- [x] Group all144 first-gate vertex/runtime failures into13 VS identities from the existing census, including36 hair,36 B761,22 B1CB and22 C107 draws. Preserve associated producer and unresolved input lists in `target/runner-view-owner-audit/remaining-vertex-contracts.json`; no additional renderer test/census.
- [ ] Obtain missing runtime evidence for faithful numerical writers, color-buffer cap/typed views and cloth/31-row resource handoff. A new game frame capture was requested; no existing capture or symbols are available. [Exact target/binding checklist](runtime-capture-requirements.md). Do not replace these values with guessed defaults or mark catalogue appearance complete.
- [x] Run one focused1.02s CPU probe for four remaining non-View PS contracts. Midnight Decay ADD9/E0EE share opcode0x57 indexed runtime value1; The Severed B143AC and Acid Abyss B145BA consume opcode0x67 dimension W through exact indexed texture owners. Acid Abyss's original DXIL feeds that value into cube-map minimum LOD. Log `target/runner-completion-nonview-numeric.log`; [source contract and limits](nonview-numeric-source-contract.md).
- [ ] Recover dimension W's exact engine value and opcode0x57 indexed writer. Do not equate a LOD consumer with a proven mip-count/residency value. The Severed additionally retains unowned opcode0x64 selector13.
- [x] Verify KASHA's current CPU candidate already rejects its unregistered31-row producer; index bounds are valid. The failed provisional live probe was not a current CPU false-positive. Withhold its independently proved PS until the producer runtime is implemented.

### October3 indexed atlas metadata — current delivery

- [x] Resolve shared resource ownership and retain exact header metadata through common stage TFX execution. Focused packaged-data probe verifies output4, original phase math, missing-owner and unsupported-lane negatives (`target/runner-completion-texture-metadata-tfx.log`).
- [x] Independently validate/register final B15812 decal shader; all70 previous contracts remain exact. White Rabbit encodes six ordered decals. Atlas row4 control changes853 native RT0 pixels; protected channels, snapshot, repeat and restoration pass (`target/runner-completion-texture-metadata-live.log`).
- [x] Preserve unaffected Thief WEAVE decoded RGBA exactly (`target/runner-completion-texture-metadata-{preserved,exact}.log`).
- [x] Refresh116-root census:1234/1813 opaque,606/606 stage2, zero load failures and stage2 order splits.579 opaque draws remain across109 skins;89 roots retain partial receivers. [Updated per-skin/source-draw mapping](remaining-runner-draws.md).
- [x] Rebuild default-feature executable and verify `--help`: 2026-10-03 05:28:12 KST, 90,380,800bytes, SHA256 `3cd9db396fce7b14781e6d163a5d2b2e7d247320e7c5667521c71867e001eecb`. Record `target/runner-completion-texture-metadata-release.json`.
- [ ] Complete native opaque receivers and displayed skin appearance. White Rabbit fixture has zero native opaque coverage; encoded head draws and853 native decal control pixels do not prove body rendering or final projected decal visibility. Keep source admission, live influence and appearance evidence separate.
- [ ] Complete remaining579 opaque contracts, including unknown View numeric values and distinct cloth/31-row producer runtime inputs. Full goal remains incomplete.

### October 2 helper startup repair — historical

- [x] Repair translator/compute/pixel-binding helper startup too: bundle translator object/libraries and the complete MinGW runtime. All four helper usage checks succeed; a real translation is byte-identical to its prior artifact. Native launcher retries must stop immediately on startup failure. Translation entry point now launches hidden child processes, logs failures, enforces 45-second timeouts and restores process error mode; PowerShell syntax validation passes.
- [x] Fix typed-buffer adapter startup `0xC0000022`: static-link the MinGW runtime, including winpthread. Usage startup succeeds; original body conversion retains SHA-256 `5daa6b17e30824aa32e88eb3dcef27ed9bf557e54efc6ba17177e02bb99e0b40`.
- [x] Integrate 414 source-identity-pinned opaque PS artifacts and 68 decal PS artifacts into production registries (including six exact E95B vertex-paired programs). Every artifact passed descriptor/bias-only instruction comparison and Vulkan validation. `cargo check --bin quicktag` passes.
- [x] Expand opaque registry to 469 exact PS artifacts (55 added, including 38 Eyes-vertex programs). Every generated artifact passes descriptor/bias-only source comparison and Vulkan validation. Remove obsolete fixed eyes/mouth registrations that shadowed the audited contracts. This is source integration; fresh live admission is tracked separately.
- [x] Verify per-part compute producer binding on mixed Body/Head geometry. Production retains each exact registered producer separately and selects its storage outputs by authored strip range; no layout-based producer substitution. Thief WEAVErunner submits 12 opaque draws, including new eyes/mouth, plus four decals. Focused live repeat, source controls and restoration pass (`target/runner-completion-mixed-live.log`).
- [x] Verify Triage WEAVErunner's six E95B-paired decals through the normal renderer. Immutable snapshot, protected outputs, zero-gate and exact restoration checks pass (`target/runner-completion-triage-live.log`). Four opaque receiver parts are admitted; remaining body materials are explicitly incomplete.
- [x] Verify focused live receivers at the earlier October2 checkpoint: Thief WEAVErunner preserves prior PNG bytes; Assassin WEAVErunner submits11 opaque parts and six decals through the normal renderer. Source-gate, repeat and restoration controls pass. Later border-sampler delivery increases Assassin to12/six; the latest delivery block supersedes this historical count.
- [x] Verify exact A2D3 head producer packing on 44,950 source vertices against independent CPU decoding and prior full-output hash; one focused GPU probe passes. Head pixel admission remains separate.
- [x] Integrate exact A60035/B8BD 15-row producer images. Full row-normalized DXIL graph proofs select the distinct source modules; focused GPU probe checks all three head producers on the same 44,950 vertices, independent CPU position/frame decoding and preserved full-output SHA. One probe passes in 1.33s (`target/runner-completion-head-pack.log`). No 31-row deformation producer or hair simulation admission claim.
- [x] Rebuild shipping executable for the verified 469-program increment. Default-feature release build and `--help` pass; 2026-10-02 13:24:41 KST, 86,943,232 bytes, SHA-256 `673b4d7918be32dc2905ef492817143a1efd4145a1a45925722e5d59a3125f8e`. Includes per-part producers and exact clamp/border samplers. Logs: `target/runner-completion-release-{build,help}.log`; record: `target/runner-completion-release.json`. Global catalogue appearance remains incomplete.

### October 3 point samplers and native index width — current delivery checkpoint

- [x] Map every remaining admission gap to named skins and exact source draws from the existing04:21 census:579 opaque draws across109 skins plus Recon White Rabbit's one stage2 draw. [Complete per-skin/source-draw report](remaining-runner-draws.md); reproducible script `scripts/report_remaining_runner_draws.py` and JSON `target/runner-renderer-audit/remaining-draws.json`. This identifies ownership; it does not mark those rendering defects fixed.

- [x] Resolve allnine newly exposed sampler gates across three exact PS contracts: F94D (six draws), B7A11F (two), B7A2A6 (one). Source filter0 means min/mag/mip point sampling; require authored anisotropy1 and retain each original address mode and shader-embedded mip bias. Opaque and decal submission share the exact sampler factory. [D3D11 filter semantics](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_filter).
- [x] Fix the actual native GPU construction failure exposed by Sentinel Syntax Disrupt:65,634 vertices and an original Uint32 index buffer were rejected by a Uint16-only guard. Retain source width through opaque, decal and C827 pipeline creation, cache keys, byte bounds, strip restart state, encoding and binding audits. Pipelines compile only for model-required formats. No index reconstruction, narrowing or shader math changes.
- [x] Verify live Sentinel Syntax Disrupt through the normal renderer: all12 opaque parts/three decals submit with original32-bit strips; the affected point material retains filters0x15/0x15/0/0 and authored address/bias. Source controls, repeat, snapshot/protected channels and restoration pass9.32s (`target/runner-completion-native-index-width-live.log`). Vandal WEAVE's16 opaque/three decals pass5.75s and decoded Uint16 RGBA remains exactly equal to the Float48 checkpoint (`target/runner-completion-native-index-width-{preserved,exact}.log`).
- [x] Fresh116-root CPU census passes74.27s: **1,234/1,813 opaque**, **605/606 stage2**, no load failures. Allnine sampler gates close.579 opaque draws remain across109 skins:435 pixel contracts,108 vertex contracts,36 runtime ABI gates. Partial native receivers remain89 roots. One Recon White Rabbit stage2 contract/order split remains; complete catalogue appearance is still unchecked. Logs `target/runner-completion-point-sampler-{catalogue,summary}.log`. CPU candidates did not detect the preceding Uint32 construction failure; candidate flags are not a live-render proof.
- [x] Ship this increment: default-feature build and `--help` pass.2026-10-03 **04:14:48 KST**,90,286,080 bytes, SHA-256 `9ed27d331d1ea0cca80929a2da6d0bfcce1f50b41ee5670cd1553c2c2d5f50f9`; record `target/runner-completion-native-index-width-release.json`.
- [x] Finish and ship index-bound follow-up: checked C827 range addition, format-aware encoding bounds, and CPU input candidates requiring a valid index view/in-bounds range. Thief WEAVE's12 opaque/four decals plusC827 retain exact decoded RGBA; source controls/restoration pass6.73s. Fresh116-root CPU probe passes75.97s with unchanged1,234 opaque/605 stage2 admissions and no load failures (`target/runner-completion-native-index-bounds-{preserved,exact,catalogue,summary}.log`). Default-feature rebuild/`--help` pass:2026-10-03 **04:21:07 KST**,90,288,128bytes, SHA-256 `d62528378874795daa2a02fac74cb3cf1c7898977bb034f3c791fbee1d95d43f`; record `target/runner-completion-native-index-bounds-release.json`. Supersedes04:14 executable.

### October 3 static analytic coverage — previous delivery checkpoint

- [x] Apply the user-authorized analytic cutoff input to release Marathon static previews only. Explicit Frame/scoped inputs, including Unknown, retain precedence. Register six source-preserving PS contracts (BE22, C951, F94D, B14EF0, B7A11F, B7A2A6), bringing the registry to508; all502 prior identities, indices, instructions and dependencies remain unchanged. This is viewer policy, not a recovered game Frame writer.
- [x] Verify through the normal renderer: Vandal Fineline XS submits19 opaque draws, including BE22 parts9/10/23 with zero Frame+0x1E0. Raising the original cutoff changes1,099 native RT0 pixels; restoration recovers every native MRT bit. Source/UV controls and repeat pass7.40s (`target/runner-completion-static-coverage-live.log`). Vandal WEAVE's16 opaque/three decals pass6.96s and decoded RGBA remains exactly unchanged (`target/runner-completion-static-coverage-{preserved,exact}.log`). No catalogue GPU test matrix.
- [x] Run one fresh116-root CPU census:0 failures, **1,225/1,813 opaque**, **586 source decals +19 C827 =605/606 stage2**. This adds107 opaque draws;588 remain unadmitted across113 skins. First gates:435 pixel contracts,108 vertex contracts,36 vertex runtime ABIs,9 sampler contracts. Partial native receivers remain93 roots; no no-opaque decal suppression or whole-preview input rejection candidates. Recon White Rabbit remains the sole unadmitted stage2/mixed-order root. Logs `target/runner-completion-static-coverage-{catalogue,summary}.log`.
- [x] Rebuild shipping executable with this increment:2026-10-03 **03:49:06 KST**,65,891,840 bytes, SHA-256 `b6f0433695c6fac1cb00a001fdf9f9400455079cf25084164cdde95f385b6383`. Default-feature release build and `--help` pass; record `target/runner-completion-static-coverage-release.json`. Full catalogue appearance remains incomplete.
- [x] Verify and ship the nine sampler contracts newly exposed after cutout registration, including the additional Uint32 native input failure found by the live probe. Exact nearest sampling and source index-width handling are delivered in the October3,04:14 checkpoint above.

### October 2 completion pass — previous admission checkpoint
Fresh CPU catalogue probe passes in63.16s: all116 roots load, opaque **1,118/1,813**, stage2 **586 source decals +19 C827 =605/606** (`target/runner-completion-float48-catalogue.log`). Float48 integration adds32 opaque draws;695 remain unadmitted across113 skins. The sole unadmitted stage2 operation remains B15812 on Recon White Rabbit. Focused Vandal WEAVErunner live submission/source controls pass16 opaque/three decals; Thief WEAVE retains its preserved RGBA exactly. This is admission and scoped live evidence, not complete catalogue appearance or receiver coverage. Shipping23:35 includes this increment; default-feature build and `--help` pass.

- [x] Identify all 30 sampler rejections: filter0x15, address mode3 on every axis, bias-0.5, LOD0..float32-max, anisotropy1. Production now supports the exact ClampToEdge address contract for opaque and decal samplers; previous wrap/mirror mappings are unchanged. [D3D11 enum semantics](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_texture_address_mode).
- [x] Verify actual clamped-sampler material submission: Thief THE SEVERED focused normal-renderer probe passes, including source controls and exact restoration (`target/runner-completion-clamp-live.log`). App rebuild remains separate.
- [x] Implement exact border sampler support: parse D3D11 border color at bytes28..43, bind matching WGPU border mode/color and require native border feature; reject unsupported colors. Focused Assassin WEAVErunner normal-renderer probe passes (12 opaque draws, six decals, source controls and exact restoration), and shipping build includes it (`target/runner-completion-border-live.log`). Prior census has 15 border sampler rejections; no fresh all-catalogue admission count claimed after this change. [D3D11 sampler layout](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_sampler_desc).
- [x] Verify all15 previous border sampler rejects through targeted production CPU admission across14 roots; exact draw indices/techniques retained, all15 now admitted. One9.90s probe (`target/runner-completion-border-admission.{log,json}`). This closes those sampler rejects without claiming full appearance or a second full-catalogue census.
- [x] Compare Thief WEAVErunner before/after mixed-producer integration: exactly62 RGBA pixels change, bounds x503..526/y96..102 (eyes/mouth); alpha and every other pixel remain exact. Record: `target/runner-completion-protected-pixels.json`. This preservation observation does not establish full appearance parity.
- [ ] Resolve shared View+0x470..0x4E0 numeric input consumers rather than suppressing active shader math. These inputs block many opaque source contracts.
- [x] Implement shared Marathon TFX opcode0x49 (canonical unk3b) as exact portable scalar vector-ramp evaluation over11 authored constants. Independent original Alkahest SSE oracle matches bit-for-bit across1,024 finite fixtures, boundary/degenerate/NaN cases; truncated constants remain unresolved (`target/runner-completion-piecewise-reference.log`). No zero placeholder or architecture fallback added.
- [x] Re-audit and integrate Rook Aftermarket Scratch EA4A/EA6E plus six opaque PS contracts using the now-resolved shared opcode. Registry now475 opaque/70 decal programs. Focused live probe encodes every eight opaque receiver parts and allfour formerly missing decals; snapshot, protected channels, source controls and exact restoration pass (`target/runner-completion-rook-live.log`). Shipping rebuild tracked separately.
- [x] Fix surface generator reload provenance: preserve each source's actual `buffer_slots` instead of inventing cb1 for shaders that omit it. Regeneration validates all475 source artifacts without arithmetic/interface edits (`target/runner-completion-piecewise-surface-generation.log`).
- [x] Verify Cryo Shift's11 admitted opaque parts, including eyelashes, eyes and mouth, plus four decals through the normal renderer. Focused source controls, repetition and exact restoration pass in6.61s (`target/runner-completion-cryo-head-live.log`). The fixture identifies opaque eye draw9; draw10 is a shadow-only vertex stage. Hair and remaining two opaque contracts stay incomplete.
- [x] Rebuild delivered executable with opcode0x49,475 opaque/70 decal artifacts and Rook integration. Default-feature release build and `--help` pass; 2026-10-02 18:17:30 KST,64,229,888 bytes, SHA-256 `d2fb202ddf06b36a6f096028503fa1068a2824937c20454bc046988ee9497a2b`. Record: `target/runner-completion-piecewise-release.json`; build/help logs share that prefix. This supersedes the13:24 shipping identity above, not the unchecked catalogue fidelity goal.
- [x] Prove Recon White Rabbit B15812 metadata ownership: TFX indices1/2/3 resolve atlas80B15810 through the shared stage+0x40 WideHash table; index0 is its sampler. Raw header supplies the exact0x10 float4 and0x2A tile count9. Shader slots and metadata resource indices remain separate. [Source contract](texture-metadata-source-contract.md); raw technique/header exports preserved.
- [x] Finish metadata TFX/live integration and rebuilt delivery: exact output4/phase and unresolved negatives, independent B15812 registration, six live ordered decals,853-pixel metadata influence/restoration and exact Thief WEAVE RGBA. Complete opaque receivers and displayed skin parity remain unchecked above.
- [x] Identify an additional KASHA YŌKAI analytic PS contract80B152B5 from current census and independently audit its exact source successfully. Focused live probe exposes unsupported vertex producer/input and aborts GPU construction; withhold registration rather than break the preview. Prior508-program registry restored. Logs `target/runner-completion-kasha-cutoff-{source,live}.log`; independently proved PS remains in `target/runner-cutoff-contract-audit`. This is source progress, not a fixed live material.
- [x] Re-audit remaining known-vertex PS contracts after shared TFX fixes:485 independently supported source contracts,206 active/unproved numeric gates andseven absent/invalid row observations. The latter audit observations do not revoke separately validated fixed programs. Integrate ten newly proved opaque programs without replacing shader arithmetic; source registry prefix remains stable. Log: `target/runner-completion-remaining-surface-audit.log`. Fresh live/census delivery tracked below.
- [x] Prove IA13 VS80A9B7D7 uses original Float48 POSITION/NORMAL/TANGENT, Snorm16x2 UV and cb1/cb12 only. It requires no compute producer. Vertex adaptation changes exactly two binding decoration words, passes Vulkan validation and preserves all math/interfaces (`target/runner-completion-direct-ia-vertex.log`). Nine paired PS contracts pass source/metadata proof; three retain active View inputs (`target/runner-completion-direct-ia-source.log`).
- [x] Verify and ship direct-IA production path plus494 opaque contracts. Vandal WEAVErunner submits14 opaque parts/three decals, including both B857 original triangle-list draws and the new BE20 material. Explicit no-compute assertions, source controls and exact restoration pass in6.84s; RGBA against the preserved initial direct-IA capture stays exact (MAE/p99/max/changed fraction0). Fresh116-root CPU census passes; no whole-preview input rejection candidate is detected. Nine direct-IA PS contracts are integrated into production; the focused GPU fixture proves B857's two live draws, not all nine shaders' appearance. Three contracts retain unresolved active View inputs. Log: `target/runner-completion-direct-ia-live.log`.
- [x] Rebuild current delivered executable:2026-10-02 18:52:16 KST,64,821,760 bytes, SHA-256 `8755279a6a48b5cbff70fcaf7b4fd81036d928a711a5b87b2c51962805393cfd`. Default-feature release build and `--help` pass. Contains494 opaque/70 decal artifacts and direct-IA integration. Record/logs: `target/runner-completion-direct-ia-release{.json,-build.log,-help.log}`. Supersedes18:17 executable; complete catalogue appearance remains unchecked.
- [ ] Resolve active View numeric ownership. High fields0x470..0x4E0 feed a procedural weighted mask/quantizer through TFX writes into material CB0; compact View cb12 does not own those destination rows. The source mapping is proven, but Alkahest definitions/writers and exported scope bytecode provide no numerical writer. Current04:21 census:210 PS/454 remaining opaque draws carry this dependency; largestB032 contributes54 draws/53 skins. Keep upstream values unresolved; no spherical-harmonic guess or zero default. [Corrected source contract](docview-numeric-source-contract.md).
- [x] Audit B1CB VS and A9C4/B762 cloth CS independently with pinned22621 DXC, no GPU/Cargo matrix. B1CB's packed t2/t3/t4 and t0 fetches are active; only cb0 row31 is output-inert. It shares normalized math with B761 but has a distinct constant ABI and no pose gate. A9C4/B762 both retain active cloth streams, tables/weights and output stores; their animation gates do not establish a rigid static branch. Source dumps: `target/runner-hair-owner-audit/{80A9B1CB-Vertex,80A9A9C4-80A9A9C5-cs,80A9B762-80A9B763-cs}.dumpbin.txt`. This is source diagnosis; exact producer ownership, buffers and live hair appearance remain incomplete.
- [x] Audit Pattern-channel initialization independently: Alkahest initializes declared channels to `Vec4::ONE` and uses the same default for missing object-channel lookup. Retain serialized AF85 overrides and branch-local ownership; this verifies initialization policy, not captured Marathon runtime colors. Named runtime updates and non-local binding ownership remain separate.
- [x] Verify and ship Frame+4 render-clock resolution. Production TFX supplies game/render clocks from the existing static preview time, matching Alkahest's packet-time writer; explicit frame/scoped values, including Unknown, override preview policy. Existing clock/dependency regression passes. Focused Triage Overdrive live probe submits14 opaque/six decals with source controls, repeat and restoration (`target/runner-completion-frame-clock-live.log`). This preserves prior admission after dependency refresh; it adds no net draws above18:52. Clock remains static viewer policy, not captured game time/animation. Frame+0x1E0 and View+0x470..0x4E0 remain unresolved.
- [x] Fix stale runtime-proof reuse: registered payload identity is insufficient when current unresolved dependencies or constant extent differ. Re-audit those observations; refresh existing generator entries in place from independently proved source metadata. Sixteen render-clock contracts now have current dependency/zero-gate lists; every source SHA, compiled SPIR-V SHA and registry index remains unchanged. Source refresh/generation logs: `target/runner-completion-frame-clock-{source-refresh,generation}.log`. Current catalogue and focused live probe pass; shipped22:35.
- [x] Verify and ship LOD-owned compute lookup. Focused raw stage24 export proves duplicate index ranges have distinct LOD0/LOD3 producers (CA3E/B7C5 versus B7E4). Production lookup verifies visible source part/range and matches its LOD category/run; retained producer map keys authored part index, preventing overlapping-range collisions. Matching probe passes (`target/runner-completion-compute-membership-owned.log`); Vandal WEAVE14 opaque/three decals preserve prior RGBA exactly (`target/runner-completion-runtime-owner-preserved.log`). Shipped22:35. This resolves ownership selection, not B7FC producer/runtime integration.
- [x] Rebuild default-feature shipping executable after runtime/ownership fixes:2026-10-02 22:35:12 KST,64,813,568 bytes, SHA-256 `d40b6dccddd23d2ea9926a0fd3fa07216db0190c9abb3df599b39b9288e45947`. Build and `--help` pass; record `target/runner-completion-runtime-owner-release.json`. Coverage remains1,086/1,813 opaque and605/606 stage2. Global appearance goal remains unchecked.
- [x] Integrate B7FC Float48 generated-position/frame path using actual LOD0 CA3E/B7C5 producers, exact source graphs and independently checked rigid-pose inputs. Focused live Vandal WEAVErunner verifies both generated parts and exact B7C5 dispatch. All116-root census adds32 opaque draws. This is live source consumption, not complete catalogue appearance.
- [x] **SOURCE / PRODUCER PROBE, NOT NEW LIVE DRAWS:** Package CA3E/B7C5 after full1,419-instruction DXIL graph comparison. CA3E's payload itself duplicates registered BE34; B7C5 retains row indices with15-row extent, unlike shifted head producers. Its exact module/storage ABI is registered. Extended existing focused GPU probe passes all44,950 source vertices, independent position/frame checks, guards, repeat and preserved full-output SHA (`target/runner-completion-float-producer-pack.log`,1.90s). This source-only checkpoint preceded integration. Subsequent actual Float48 packing/live controls are recorded below; B7C5 and Float48 consumer shipped23:35.
- [x] **TRANSLATOR INPUT SUPPORT:** Map declared BLENDWEIGHT/BLENDINDICES to locations4/5; exact B7FC translation and Vulkan validation pass. These inputs are unused by its DXIL function, but previously caused remapper rejection.
- [x] **SOURCE / NATIVE VERTEX PROBE:** Pin B7FC DXIL/raw SPIR-V identities; replace only typed-buffer fetches with scalar-storage loads. All463 other instructions remain unchanged. Native original/lowered instrumentation preserves31 output scalars bit-exactly in five vertex/base/current/previous cases, with negative binding controls (`target/runner-completion-float-vertex-native.log`). This proves shader adaptation, not native-game appearance.
- [x] **ACTUAL FLOAT48 PRODUCER PROBE:** Export real DACF/B52B48-byte source streams; independently verify both14/15-row producers' packed positions, oct frames, guards and repeat (`target/runner-completion-float-runtime-pack.log`,1.20s). Raw positions are not pretransformed. Palette fetch is unconditional: share one bounded3.75MiB zero-initialized image per preview; poisoned/zero image values produce identical rigid-branch output. No captured native palette claim.
- [x] **SHARED PIXEL CONTRACT:** Identical PS payload CA2A/B141C3 serves both Body and Float48 source VS pairings. Replace singular PS-to-VS assumption with explicit source-proved allowed ABI lists, including pipeline selection and actual-vertex audit topology. Generate502 unique PS artifacts; all494 prior indices, shader identities, math and dependency lists remain unchanged. Eight new PS contracts pass; B513/D30E/F65E remain unresolved. Generator log `target/runner-completion-float-pixel-generation.log`. Shipped23:35 after focused live verification.
- [x] **FLOAT48 LIVE CONSUMPTION / PRESERVATION:** Vandal WEAVErunner encodes16 opaque/three decal draws, including both B7FC parts with exact LOD-owned B7C5 dispatch; source controls, finite outputs, repeat and restoration pass6.75s (`target/runner-completion-float48-live.log`). Direct parts share geometry with generated parts; no-compute assertion now verifies part ownership. B85A legitimately stores cb0[89].x=0.75 in RT3.z, so its source material payload replaces an obsolete zero assertion. Thief WEAVE remains RGBA-exact (all deltas0) against preserved mixed-producer baseline (`target/runner-completion-float48-preserved.log`,6.06s). Fresh63.16s census admits1,118/1,813 opaque and605/606 stage2. Shipped23:35; remaining full appearance goal stays unchecked.

### Historical expanded registry live checkpoint
116/116 roots load; opaque admission **845/1,813**, stage2 **537 decals + 19 C827 / 606**. No admitted decal is now suppressed by the zero-native-opaque guard. Remaining: 968 opaque draws (376 unregistered PS, 257 unregistered VS, 229 unsupported VS ABI, 76 unsupported state, 30 sampler contracts); 19 skins still have unadmitted stage2 operations. Partial receivers remain incomplete. Fresh census passes in 58.54s. One live Thief WEAVErunner final probe passes; PNG bytes remain exact (`08ef880e510e61cfd6414e4ff0cdad34165d3f4c45733addcf7e9abb2c7ca325`). Shipping executable has not yet been rebuilt for these changes.

## Active all-Runner repair goal — October 2, 2026

User requests complete fixes for the confirmed catalogue and renderer audit defects. This is delivery work, not admission-only completion. Track all 116 roots; preserve unaffected RGBA and source shader operations. An active goal now tracks complete fixes across all 116 roots, complete receiver coverage and authored order, protected pixels, accurate checklist evidence, focused probes and the delivered executable. No completion claim until those requirements are fulfilled.

- [x] Export every distinct catalogue shader payload and geometry-local runtime image; group by source SHA, not tag aliases. 894 payloads across 116 roots; associated source VS/CS exported.
- [ ] Supply complete source-faithful opaque and stage2 contracts across the catalogue, including common unsupported vertex/producer ABIs. Refresh coverage after integration.
- [ ] Eliminate zero/partial native receivers and split stage2 execution; retain authored operation order over complete receiver attachments.
- [ ] Keep resource loading from silently changing admitted materials into generic evaluation.
- [ ] Preserve normal/emissive-only material resources independently of color texture presence. Implementation in progress; probe pending.
- [ ] Remove texture-pixel heuristics that suppress geometry. Implementation in progress; probe pending.
- [ ] Validate raw authored vertex/index views, required streams and native strip ranges before GPU submission; report rejected inputs concretely. Implementation in progress; probe pending.
- [ ] Resolve C827 through its geometry-local TFX scope. Implementation in progress; probe pending.
- [ ] Verify actual submissions and protected outputs with focused probes; rebuild delivered executable; do not equate registration, admission or build success with appearance completion.

# Marathon renderer fidelity plan

Updated 2026-10-02 after [shared generic material correction](generic-runner-material-contract.md), live decal delivery and the [renderer audit](analysis_result.md).

**Historical initial catalogue audit:** all116 authored Runner skins in the actual Models catalogue load in one58.48s CPU pass. Production admission accepts146/1,813 opaque draws and132 decals +19 C827 operations out of606 stage2 draws; generic submissions are not labelled visually correct.79 skins have unadmitted stage2 operations. Ten roots lose52 otherwise admitted decals through the zero-native-opaque guard; Destroyer WEAVErunner is one. Five roots can split native/generic stage2 order. [Every-skin ledger](runner-renderer-census.md), [structural review and reported cases](runner-renderer-review.md).
- [x] **AUDIT / NOT APPEARANCE DELIVERY:** Complete all-skin material/decal census using production rejection reasons, accounting for all116 roots and zero loader failures. One focused live preservation probe keeps previous Thief WEAVErunner PNG bytes and all four native MRT hashes exact. No broad suite, all-skin GPU matrix or appearance-fix/release claim.
- [ ] Replace unsupported generic material/decal evaluation by shared complete source contracts, prioritized from the catalogue ledger; include the shared E95B decal vertex path used by15 skins. Per-skin examples are regression targets, not majority-coverage evidence.
- [ ] Repair zero/partial native receivers and split stage2 composition/order over one complete material receiver contract; address readiness-dependent path changes. Removing the no-opaque guard alone would expose an empty snapshot.
- [ ] Separate asset discovery/classification, native admission, residency and encoded submission diagnostics; validate raw IA/producer inputs before GPU creation and report whole-model rejection concretely.

**Previous scoped material increment:** **Thief WEAVErunner `80A9C2C4`**. Investigation proved omitted material color/response math. Shared exact program evaluation replaces nine generic body draws for this target; it did not complete WEAVErunner on other shells or the majority path. Verification remains limited to focused probes at the user's request.
- [x] **LIVE / SCOPED:** Fix shared C244/C430/14701/14706 generic material omissions. Complete source programs consume geometry-local TFX, full textures and samplers through the normal renderer; ten total material draws, including existing eye-detail, plus four live decals. UV/light influence, protected channels, source gates and repeat/restoration pass. [Contract and evidence](generic-runner-material-contract.md).
- [ ] Cover other unsupported generic material/decal families and specialized hair/head inputs. These four shared corrections are not completion of all skins; no skin-ID recoloring or texture guesses.

- [x] **LIVE / SCOPED:** Submit six identity-verified stage2 decal programs over an immutable native packed-normal snapshot; preserve authored order, state26/27 masks and ordered C827 before viewer lighting. Focused final probes pass for Thief Cryo4/Digital6/Assassin Cryo7/Vandal WEAVE3/Cyber8/Thief WEAVE4 decal draws. Disabled gate, protected-channel, snapshot and exact-restoration checks pass. Cyber's final focused rerun checks state27 and its original opaque color-domain oracle. Release00:18 KST contains this integration. Unknown families and native inherited-state parity remain open.

**Historical expanded priority (accepted examples):** Thief's improved body is not completion. Diagnose and fix shared material/ownership failures across Assassin Digital Prowl `80A9DC79`, Assassin Cryo Shift `80A9DD33`, Vandal WEAVErunner `80A9DAD3` and Destroyer Cyber Red `80A9D869`; retain Thief as a regression case. Initial audit found that only Thief's five PS identities were admitted and merged geometry lost nested Pattern ownership. Avoid skin-ID color overrides, shader selection by layout alone, or claiming universal fidelity from one corrected skin.

**October1 cross-skin material pass verified (historical):** 18 additional exact opaque PS contracts, branch-owned Pattern channels, both audited cull states and texture-readiness admission are applied to the normal renderer. Encoded draws: Thief9, Digital Prowl13, Assassin Cryo8, WEAVErunner10, Cyber6. Shared Face/EyeDetail now run live; Cyber armor uses distinct VS80A9D7E6/PS80A9D820. All15 normal-renderer GGX/Compatibility/Lambert captures pass source-gate, UV/light, independent projection, repeat and exact-restoration checks. Four authored output hashes stay identical across lighting modes. Cyber's16 red NaNs are independently predicted from original D852 fractional-power math; production arithmetic is unchanged and displayed HDR remains finite. Cyber original/lowered VS parity passes31 scalars bit-exact with negative controls. Thief's prior seven-draw body pass and unrelated Conquest LMG preserve their prior final RGBA exactly. The new face/eye-detail correction intentionally changes head appearance. Ordinary binary193/0/143ignored, portable8/0. Fresh default-feature release and `--help` pass; the16:27 release below is historical. Hair, other material variants, decals and native final lighting remain incomplete. [Full contract and limits](cross-skin-material-contract.md).

- [x] **LIVE / SCOPED:** Verify the18 additional opaque PS contracts and branch-owned TFX inputs; integrate exact VS/PS/producer admission without skin-ID color overrides.
- [x] **LIVE / TESTED:** Run all five shells in all three viewer shaded modes; verify actual encoded draws, source-gate/UV/light controls and exact restoration. Preserve prior supported Thief body and unrelated weapon RGBA; head evaluation is an intentional correction.
- [ ] Resolve remaining hair/head/eyelash input contracts, unsupported decal families and native runtime fidelity; admitted decals now compose live, but global skin fidelity remains incomplete.
- [x] **LIVE / PIXEL-PRESERVED:** Build material pipelines only when resident draws need their exact VS/PS/cull key; retain them in the renderer cache. Fresh fixtures require Thief7/Digital6/Cryo4/WEAVErunner5/Cyber6 instead of50 eager material pipelines. Six unsubmitted D0F2/hair/head probe pipelines now compile only in tests. All15 final/Compatibility/Lambert PNGs and unrelated Conquest LMG preserve prior RGBA exactly; ordinary193/0/143ignored. Aggregate capture verifier passes. Release rebuilt22:02 KST; `--help` passes. Logs:`target/*-demand-pipeline.log`, `target/cross-skin-demand-pipeline-ordinary.log`, `target/cross-skin-demand-pipeline-unrelated.log`, `target/cross-skin-demand-pipeline-release-build.log`. Pipeline counts establish removed construction work; overall opening/switching latency is not yet benchmarked.
- [ ] Complete Assassin Cryo's remaining C107 material family after proving VS color-buffer interpretation/index cap and PS input influence. Normal loader exports geometry80B14E77 → header80A9DD2B/data80A9DD2C, stride4/type5,2770 records; exact CS80A9B4CB arithmetic/ABI matches BodyCS, but its native dispatch/runtime is not captured. All three PS variants have reachable TC8.x paths into RT0.rgb/RT2.g; DD0F's zero rows10/11 do not gate that path. Alkahest's RGBA8 UNORM view is reference evidence only. Do not bind all-ones t0 or guess a native cap. [Source/input contract](assassin-cryo-color-contract.md). Audit:`target/cross-skin-audit/material-admission.json`; test:`target/cross-skin-color-buffer-audit.log`.

**Current rebuilt executable:** [target/release/quicktag.exe](../../target/release/quicktag.exe),2026-10-03 05:28:12 KST, 90,380,800bytes, SHA-256 `3cd9db396fce7b14781e6d163a5d2b2e7d247320e7c5667521c71867e001eecb`. Includes508 opaque/71 decal contracts and indexed atlas metadata. Build/`--help` pass; `target/runner-completion-texture-metadata-release.json`. Admission1234/1813 opaque and606/606 stage2;579 opaque draws and complete receiver/catalogue appearance remain unresolved. Earlier checkpoints are historical.

**Current live delivery:** supported body/material, shared face/eye-detail and audited Eyes-vertex pixel families feed Models **Tiger GGX**, **Compatibility**, **Debug Lambert**, **Albedo** and **Albedo MRT**. Fresh final-mode probes submit Thief WEAVE12 material/four decal draws, Assassin WEAVE12/six and Triage WEAVEfour/six; no fresh all-mode matrix claimed. Source-pinned decal contracts compose before viewer conversion/lighting, retaining ordered C827. Native final lighting, specialized hair/headwear, other head variants and unadmitted material/decal families remain incomplete. [Checked-entry live audit](live-delivery-audit.md) separates checked experiments/infrastructure from actual consumption.

| Target correction | Normal renderer | Current evidence |
| --- | --- | --- |
| Top-face C827 band | Live; replaces old submission | 10 HDR / 8 albedo pixels differ in fixed fixture |
| Body, jacket, yellow chest, hands/hardware | Live in Albedo / Albedo MRT and Tiger GGX / Compatibility / Debug Lambert | Seven encoded authored draws, four FP32 outputs, independent GPU conversion oracle, UV/light controls and exact restoration |
| Face and eye-detail | Live for verified exact contracts | Thief11/12; one shared eye-detail draw each for both Assassins and WEAVErunner |
| Audited eyes/mouth families | Live through exact per-part head producer | Thief WEAVE's two new draws consume the audited source PS; focused controls and restoration pass |
| Dark hair, headwear and remaining head variants | Incomplete | Hair simulation/resource requirements remain unresolved |
| Atlas markings / missing decal detail | Live for admitted exact contracts; incomplete receiver coverage elsewhere | 605/606 stage2 operations admitted in current census; B15812 and many opaque receiver materials remain incomplete |

**Current visible status (October1):** compiled material evaluation now corrects substantial body colors/textures across all five fixtures, including Cyber armor. Shared face/eyelash material evaluation is live; hair remains pale. Four native FP32 outputs stay intact. `ViewerMaterialProjection` converts normal direction to viewer coordinates, applies explicit viewer roughness `1-saturate(4*radius-3)`, maps metalness/AO and preserves uncovered pixels. Native RT1.a=.67 is coverage/class, never roughness; native RT3.xy is signed motion, never emission. Viewer lights, shadows, exposure and ambient remain viewer policy. Exact native deferred/global lighting is not connected; M1/M2 remain incomplete.

**Historical seven-draw shaded delivery verification:** ordinary binary192 passed, zero failed,142 ignored (`target/roadmap-live-lit-ordinary.log`). Normal-loader actual GPU captures pass in Tiger GGX, Compatibility and Debug Lambert (`target/roadmap-live-lit-final-gpu.log`, `target/roadmap-live-lit-compatibility-gpu.log`, `target/roadmap-live-lit-lambert-gpu.log`). All46,211 covered pixels pass the independent f64 normal-rotation/property/UNORM conversion oracle, observed maximum1byte against the predeclared1byte limit; uncovered poisoned receivers remain exact. UV controls change38,625 lit pixels in GGX/Compatibility and40,125 in Lambert; controlled lighting changes23,090 and41,698 respectively, preserving native material outputs. Restoring uniforms restores every native/receiver HDR bit. All four native FP32 output files are independently hash-verified and bit-identical to prior delivered Albedo across current Albedo and all three shaded modes (`target/cryo-runtime-audit/viewer-lit-delivery-verification.json`). Prior Albedo and unrelated weapon final PNGs remain exactly RGBA-identical; MAE/p99/max/changed fraction0 (`target/roadmap-live-lit-base-rgba.log`, `target/roadmap-live-lit-other-model-rgba.log`). Fresh GGX metadata capture also preserves the new shaded PNG exactly (`target/roadmap-live-lit-final-rgba.log`). This demonstrates actual material consumption and controlled viewer lighting, not native in-game scene parity. The separate incomplete native decal receiver gate remains unchecked.

**Historical seven-draw executable:** `target/release/quicktag.exe`,2026-10-01 16:27:07 KST,40,184,832 bytes, SHA-256 `fdb1d50dce53641f266382013e141e73eba71ac65741b6d739b087d36938bf3f`. Normal default-feature build and `--help` pass (`target/roadmap-live-lit-release-build.log`, `target/roadmap-live-lit-release-help.log`). Record:`target/cryo-runtime-audit/viewer-lit-release-build.json`. This release includes the viewer-lit material path; actual consumption is proven by the normal-renderer GPU captures above, not a release UI screenshot. Restart the app to load this executable. Earlier12:02/12:34 build records below are historical.

**Historical seven-draw source-color evidence:** `target/quicktag-model-probe/runner-thief-cryo-shift-combined-base-color-roadmap-live-surface-shipped.authored-surfaces.json` records seven original strips, actual four FP32 target hashes and removal of their old opaque submissions. The normal-loader GPU fixture verifies46,211 covered pixels, bit-exact repeated outputs, +.125 cb1[6].w changing27,850 native color pixels and25,657 displayed albedo pixels, preserved motion, and exact restoration of every native and viewer attachment. Source-color PNG intentionally differs from the previous C827-only Base Color capture in43,950 pixels; alpha remains exact (`target/cryo-runtime-audit/live-source-color-image-change.json`). This proves real consumption, not native runtime or final-lit fidelity.

**Explicit limits:** PS view row17 remains a zero branch policy; skinning row5.w=1 is a viewer texture-frequency policy. Both can affect RT0. Bind pose/camera, storage view intent, native active variant/inherited state and dynamic runtime values still need independent proof. Unresolved metadata is allowed only under each source program's audited zero gate, with a finite inert row; other unresolved dependencies reject admission. FP32 native intermediates preserve probe outputs; RGB projection into RGBA16F/RGBA8 is a display conversion, not bit-exact RT0 export. Interactive native targets use the bounded allocation estimate; large source-color PNG export memory/parity remains unverified.

**Evidence:** `scripts/verify_c827_live_receivers.py` verifies actual normal-renderer MRT readbacks and encoded native submissions against the same scene without draw19; report `target/cryo-runtime-audit/c827-live-receiver-comparison.json`. Original strip1719..1785 and both native projections are encoded; no old indexed C827 draw remains. A +.125 live cb1[6].w control changes both color receivers, preserves other attachments/alpha, and normal uniform restoration restores every MRT bit without rebuilding groups. The GPU fixture passes (`target/roadmap-live-c827-controls.log`, `target/roadmap-live-c827-controls-receivers.log`); two independent camera/field-ownership regressions join the ordinary192-test suite. Native inherited state/animation/lighting remain unclaimed. Release executable rebuild is tracked below; the September30 executable did not contain this integration.

**Native lighting source progress:** the earlier empty inventories were an exporter omission, not absent packaged consumers. Export and pinned SDK22621 disassembly now preserve212 PS/CS payloads. Exact source traces identify the packed-normal radius decoder and separate global/direct/environment response math. Direct `deferred_uber_light` consumes length differently from global lighting; debug response-squared is not a BRDF. Active variant, attachment binding/format and runtime light/probe values remain required. [Consumer evidence](thief-material-consumer-contract.md#packaged-lighting-consumers).

**Current native-lighting increment:** a fresh production-parser TFX export covers ten global/deferred techniques (`target/cryo-runtime-audit/render-globals/lighting-consumer-bindings.json`; `target/roadmap-lighting-consumer-bindings.log`, one pass). Earlier absence of texture-source lists was an exporter gap, not absent TFX. For the full `global_lighting_and_shading` candidate, exact TFX supplies t0=Deferred+B8, t1=A8, t2=B0, t3=98, plus the other Deferred/Atmosphere/ShadowMask/Frame inputs. Alkahest's Destiny offsets/field names do not establish Marathon allocations. s0 is shader-consumed but not assigned by the technique's TFX;122 empty-runtime dependencies remain. [Final-consumer contract](thief-global-lighting-contract.md) records every resource, used row and missing ownership proof.

The exact global-light accumulation VS80A0620F/PS80A06213 pair is now identity-pinned, embedded and executed against actual GPU body RT1/RT2. Independent descriptor-edit validation preserves all arithmetic/interfaces/sampling and rejects structurally valid unknown identities (`scripts/test_native_global_light.py`). The controlled ambient branch passes all4,171 body pixels against a predeclared2e-6 f64 oracle bound; maximum error8.533854e-8. Direction reversal changes all4,171 pixels; exact zero colors, literal alpha/inactive specular, repeat and restoration pass. No WGSL vertex bridge or CPU input reupload. Report:`body-global-light-execution.json`; log:`target/roadmap-native-global-light-body-gpu.log`. This is accumulation evidence, not final shading, direct BRDF, native sampler/runtime parity, a live encoded operation or another release appearance correction. `GlobalLightPipeline` uses explicitly controlled nearest sampling; production sampler proof remains required.

**Historical native-lighting probe verification:** ordinary binary192 passed, zero failed,142 ignored (`target/roadmap-native-global-light-ordinary.log`); all8 established native/WGPU gates passed (`target/roadmap-native-global-light-established-gates.log`). The separate incomplete receiver test remained excluded explicitly. All4 preserved body/material/UV/layered report hashes remained exact (`target/roadmap-native-global-light-preservation.json`). These are historical probe checks, not current shaded delivery evidence.

**Historical source-color-only build recheck:** before the current shaded integration, native body admission was limited to diagnostic1 / `SurfaceAlbedo`; default shaded modes still used old body materials. Its Base Color regression matched the delivered source-color capture exactly (`target/roadmap-native-global-light-live-rgba.log`). This explains the user screenshot and is superseded by current viewer-lit adoption. The user has no native GPU capture or shader symbols; packaged data and explicit viewer policies remain the available evidence.

The previous executable was separately observed at2026-10-01 12:34:07 KST,40,161,280 bytes, SHA-256 `c76a685ec476b5dc2d8d687dd02561a8dae684014badbef1e48f57a72a437322`. Record:`target/cryo-runtime-audit/observed-release-status.json`. That observation did not establish build provenance or default shaded adoption. The earlier12:02 source-color build below is historical; current viewer-lit delivery is recorded separately.

**Hair input gate:** exact hair VS/PS are embedded. Nested Pattern80A9D191 owns component80A9E5F5/geometry80A9D18F; sibling component80A9D939 declares and serializes AF85 zero vectors for object inputs feeding hair rows65/67. Assembly now retains that path and resolves geometry-local channels without sibling merging. This ownership correction does not supply HairCS simulation inputs or metadata. Row133 remains unresolved without a zero gate. Hair CS80A9A9DE producer ownership/dispatch and headwear PS80A9A9D3 remain unimplemented. Replacing the verified compatibility whitening expression with a tint heuristic would not complete authored hair evaluation.

**Packaged lighting sampler owner recovered:** earlier technique-only exports omitted inherited Frame resources. The fresh scope export now records exact sampler tags. Frame80A06026 binds PS s0 to80A06025; Frame80A74028 binds it to80A74027. Independent raw-array and exact TFX-byte audit matches the production parser, confirms Frame/View mask3 for all ten lighting techniques, and rejects any technique/View overwrite of s0 (`scripts/audit_lighting_sampler_inheritance.py`, `target/cryo-runtime-audit/render-globals/lighting-frame-sampler-ownership.json`). The focused scope-export test passes once (`target/roadmap-lighting-frame-sampler-export.log`). This closes packaged s0 owner provenance; active native binding history, filtering/formats and exact native final-lighting consumption remain open. This sampler audit itself produces no appearance change; current viewer-lit adoption is separate.

**Child hair-scope audit:** root80A9D134 → component80A9E5DF → Pattern80A9D191 → component80A9E5F5 → geometry80A9D18F establishes the hair/headwear owner branch; Pattern80A9D191 also points to channel component80A9D939. That component is shared and not hair-exclusive. Root80A9E636 lacks the two object declarations. Preserve the exact branch path through assembly and material ranges before local resolution. Row133 metadata remains unresolved independently. [Audit](live-delivery-audit.md) records scope limits.

September30 and earlier October1 checkpoints below are historical. Their zero-live-draw or C827-only counts describe those captures. Supported materials and six stage2 decal contracts now encode through the normal renderer. Native global/deferred final lighting and specialized hair/headwear/eyes/mouth still lack live consumption.

**Historical source-color delivery checks:**

- [x] Ordinary binary suite:192 passed, zero failed,141 ignored (`target/roadmap-live-surface-ordinary.log`).
- [x] All8 established ignored native/WGPU gates pass after sharing dense raw-IA draw mechanics (`target/roadmap-live-surface-established-gates.log`). The incomplete receiver gate is excluded explicitly, not treated as a success.
- [x] All4 established body/material/UV/layered report hashes remain exact (`target/roadmap-live-surface-preservation.json`).
- [x] Normal-loader Base Color and MRT albedo captures pass after module extraction, with actual encoded draws, resources, output controls and restoration (`target/roadmap-live-surface-shipped-gpu.log`, `target/roadmap-live-surface-shipped-albedo-gpu.log`).
- [x] Default shaded normal-loader capture passes; only C827 is encoded (`target/roadmap-live-surface-shipped-lit-gpu.log`). No exact prior default-lit RGBA baseline is claimed for this pass.
- [x] Rebuild `target/release/quicktag.exe` with normal default features after current GPU checks. Build and `--help` pass (`target/roadmap-live-surface-release-build.log`, `target/roadmap-live-surface-release-help.log`). New executable:2026-10-01 12:02:00 KST,40,142,336 bytes, SHA-256 `8c4c6dd3577ba6f59c16fe0441568b291cf343dca634c844c22eb95d1dca659b`. Record: `target/cryo-runtime-audit/source-color-release-build.json`. This identifies the compiled release; actual consumption evidence comes from the normal-renderer GPU captures, not a release UI screenshot. The earlier C827-only executable below did not contain this integration.
- [x] Independent Luna read-only review confirms production Models controls reach the same mode gate and encode path. Native source-color activation is `model_renderer.rs` diagnostic1 or `SurfaceAlbedo`; default diagnostic0/TigerGGX retains compatibility body draws. The separate TagView CPU fallback does not expose this native integration.

**Historical C827 delivery checks:**

- [x] Fresh ordinary binary suite:192 passed, zero failed,141 ignored (`target/roadmap-live-c827-final-ordinary.log`).
- [x] Removing unused C827 compatibility pipeline allocation preserves the validated live RGBA capture exactly (`target/roadmap-live-c827-final-rgba.log`, zoom2.0).
- [x] Refresh both receiver captures with explicit matching camera metadata and rerun the strict C827 comparison. Schema2 final captures pass with yaw−58.1°, pitch2.864789°, zoom2.0, zero pan (`target/roadmap-live-c827-final-rgba.log`, `target/roadmap-live-c827-final-receivers.log`, `target/cryo-runtime-audit/c827-live-receiver-comparison.json`). The Thief fixture's test-only default zoom is now2.0 so the complete model and face-band control remain in frame. A zoom3.1 control failed due to coverage; it is not accepted evidence.
- [x] Rebuild `target/release/quicktag.exe` with normal default features (`wordlist`, `xbox-one-deswizzler`). Build passes (`target/roadmap-live-c827-release-build.log`); `--help` exits successfully. New executable:2026-10-01 05:45:44 KST,40,012,800 bytes, SHA-256 `a0c515744ae3e645370fd8b1254e9eec118867f117a1030d10b68a3a450abced`. Production C827 binding/receiver labels are present in the binary; normal-renderer GPU tests establish actual consumption separately. Record: `target/cryo-runtime-audit/c827-release-build.json`. The September30 executable was stale for C827.

**Goal:** reproduce Marathon's authored geometry, materials and rendering, starting with Thief Cryo Shift `80A9D134`, then generalize by proven shader behavior. Deliver working increments. Do not tune colors to one screenshot or preserve incorrect pixels as fidelity truth.

This replaces shadow-first ordering and obsolete Pretty Preview proposals. Alkahest is architectural/reference evidence; Marathon data and controlled outputs establish Marathon behavior.

**Visible renderer status (September 30):** material appearance corrections are not yet active in the normal UI. It still submits compatibility shaders; live authored material draw count is 0. Embedded shader registry, shared pipelines and passing GPU probes are infrastructure/evidence, not visible adoption. The user's latest-build check confirms no visible Thief correction. Prioritize a complete consuming path and normal-renderer submissions; do not present further probe-only work as a visual fix. Remove each replaced compatibility draw once its live authored path passes its gate.

**Remaining native-fidelity integration (historical decal limitation superseded above):** body material outputs now feed existing viewer lighting with retained mutable camera/UV scopes. Exact native final-consumer runtime/state/sampler/attachment ownership remains open; the viewer conversion is not native BRDF parity. Hair/head need branch-owned inputs and actual producers. D0F2 needs source-proven depth/snapshot/receiver composition. Complete these specific consuming paths before replacing their old draws; registry counts and pipeline creation are not completion gates.

**Historical pipeline-only decal refactor (superseded by live delivery above):** `src/render/decal_draw.rs` now owns the exact D0F2 dense bindings, original strip submission and state-26 blend/write masks; both cached UI pipeline creation and actual-source GPU probes use this component. The scene-normal `Texture.Load` binding permits an unfilterable view, matching the audited read instead of requiring filtering. Initial actual-source graphics and real-device normal UI pipeline creation each passed once (`target/roadmap-shared-decal-draw-gpu.log`, `target/roadmap-shared-decal-draw-pipeline.log`). This consolidates code; it does not add a live native draw.

**Historical receiver experiment rejected:** body draw0 alone was an insufficient receiver fixture. Without depth/culling, exact marking strips changed pixel13428 outside that draw's coverage (`target/roadmap-body-decal-receiver-gpu.log`). This establishes a fixture/precondition failure, not the owning surface or native depth rule. Do not weaken the coverage assertion, infer ownership, clip pixels on CPU, or report a composition pass. The incomplete experiment was removed from the working harness and preserved at `target/cryo-runtime-audit/decal-receiver-incomplete-experiment.rs.txt`. A replacement must use independently established receiver ownership and depth/state coverage; receiver composition remains unchecked.

**Historical receiver experiment — not native parity evidence:** the separate experiment now includes all six opaque body-source draws: body0, hardware1, chest3–4, hands5 and sleeve7. Their exact shaders execute before markings16–18, using the GPU normal receiver directly, read-only depth and state26 masks. Draw16 and draw18 preserve uncovered pixels; draw17 still writes one uncovered pixel23493 (x197,y91) in the256x256 fixture (`target/roadmap-complete-body-decal-receiver-coverage.log`, `target/cryo-runtime-audit/body-decal-receiver-coverage.json`). Missing hardware/hands no longer explains this failure. Source geometry, edge precision and inherited state remain under investigation; nearest-triangle candidate tracing does not establish native ownership. The strict assertion is unchanged. `actual_body_decal_receiver_composition_preserves_coverage` retains this incomplete gate separately from established material/UV tests. No live D0F2 integration or release appearance fix results from this work.

**Receiver edge audit:** corrected barycentric signs and independent triangle self-tests in `scripts/audit_thief_decal_pixel.py`. Across all68,072 opaque triangle windows, no checked opaque draw covers pixel center(197.5,91.5). Draw17 covers it via vertices32626/32627/32628; closest opaque sleeve triangle lies0.02609095 pixels away, beyond the1/256 edge band. This identifies projected source overhang under the tested policy, not a missing checked draw or proven native receiver ownership. Report: `target/cryo-runtime-audit/body-decal-uncovered-pixel.json`. Native inherited depth/stencil/receiver behavior remains open; strict coverage assertion is unchanged.

**Receiver contract correction under review:** D0F2 has no RT0-presence test; blend26 is not its depth selector. Its authored depth selector15 maps to raw depth3 with writes disabled; reference compare depends inherited Forward/Reverse mode. Reference code supplies a pre-decal normal snapshot and prior depth. Consequently the probe's requirement that every marked pixel have an opaque RT0 pixel is not yet an established native invariant. Preserve the historical failing test while proving correct inherited state, clear values, producer assignment and a source-based composition oracle; do not clip the one overhang pixel or silently relax an assertion to manufacture a pass. Head20's nearest hardware13 candidates also remain geometric evidence, not captured receiver ownership.

Reference details: Alkahest `gpu/global_state.rs:310–338` maps combo15 to depth3/stencil1; `builtin/gpu/depth_states.txt:61–80` declares disabled depth writes and Less. `gpu/command_list.rs:147–170,334–338` selects Less for Reverse and LessEqual for Forward; `feature/decals.rs:107–128,151–177` explicitly enters Forward for decals. Unset technique state inherits current state. `renderer/submit/buffers.rs:170–180` clears albedo/normal to zero, the third buffer to `[0,.5,0,0]`, and depth to zero. These are Destiny/Alkahest reference facts, not independently established Marathon native defaults. Marking16–18 declares compute80A9A2E2 while the six opaque body parts declare80A9BE34; both validated identities currently lower to the same embedded compute artifact. The shared44,950-vertex fixture correctly dispatches703 groups; authored ranges are index windows, not per-part vertex counts. Different source identity alone does not prove the one-pixel overhang's cause (`target/cryo-runtime-audit/thief-compute-parts.json`).

**Historical receiver-material live-path recheck before source-color integration:** the normal renderer constructs `_authored_body_pipeline` and `_authored_d0f2_pipeline` but does not encode either. Only `authored_c827_pipeline` is encoded. Shared dense descriptor handling and added hands/hardware programs therefore produce no new UI appearance correction. Fresh normal-loader Base Color capture remains RGBA-identical to the live-C827 baseline: MAE/p99/max/changed fraction all zero (`target/roadmap-receiver-material-live-rgba.log`). Ordinary binary tests pass192, fail0, ignore141 (`target/roadmap-receiver-material-ordinary.log`); preserved body/graphics/UV/layered reports remain byte-identical (`target/roadmap-receiver-material-preservation.json`). Hands/hardware actual-source execution and conditional runtime gates are documented in [the receiver material contract](thief-receiver-material-contract.md), with native-runtime limits explicit. These are probe results, not delivered material fixes. Release executable remains the October1 05:45:44 KST build recorded above; no subsequent rebuild is claimed.

**Historical C827-only build investigation validation:** all8 established ignored native/WGPU tests pass (`target/roadmap-live-status-established-gates.log`), explicitly excluding the incomplete receiver gate. That gate was run separately and still fails at the same uncovered pixel (`target/roadmap-live-status-receiver-gate.log`); it is not an expected-panic success. All4 preserved body/material/UV/layered reports remain byte-identical (`target/roadmap-live-status-preservation.json`). The existing live C827 receiver verifier passes again. These checks clarify delivery status and preserve evidence; no additional normal-renderer material correction or new release build is claimed.

**Historical validation after decal refactor:** ordinary binary suite 190 passed, zero failed, 141 ignored (`target/roadmap-shared-decal-final-ordinary.log`), including real-device pipeline creation. Actual-source body/layered/marking GPU graphics test passed once (`target/roadmap-shared-decal-final-gpu.log`). Preserved body material/graphics report hashes remain byte-identical. Normal-loader Thief RGBA comparison has zero MAE, p99, maximum difference and changed pixels (`target/roadmap-shared-decal-final-rgba.log`). This is a pixel-preserving infrastructure change, not a fidelity correction. Live authored UI material draws remain 0; M1/M2 remain incomplete.

## Completion rules

- Distinguish parsed, stored, resolved, bound and consumed correctly. Each checkbox covers its stated scope only. Separately report whether the normal renderer submits the change and whether the delivered executable contains it.
- TFX Complete, resource/layout counts, nonblank images and histogram scores are not fidelity proofs.
- Each implementation needs shader/source evidence, a reproducer, intermediate-output checks and explicit unknowns.
- Keep unsupported behavior explicit during replacement. Remove obsolete branches after validation; do not maintain permanent parallel old/new modes.
- Retain bounded caches and warm-switch performance. Pixel-preserving refactors and intentional fidelity corrections have different acceptance rules.

## Existing foundation

| System | Implemented | Still open |
| --- | --- | --- |
| Stages | Typed 25-stage ABI, graded census and strong anchors | Stage 3 and weaker feature-stage meanings |
| Shadow membership | Authored ranges/LOD/layout/techniques and separate selection | Packaged shader behavior, fallbacks, exact light/receiver contract |
| Native inputs | Raw payloads/descriptors and IA ranges; compatible depth/shadow consumers | Visible VS, runtime resources and full stream use |
| TFX | Register images, binding and unresolved-dependency records | Required values and actual consumers |
| Materials | Selected weapon/runner/decal reconstructions | Thief body, hair and target decals |
| States | Selectors, bias table, broad depth mapping | Full blend/rasterizer/stencil, inherited defaults and MRT effects |
| Shadows | Affine projection and 9-tap filter; earlier artifact reported fixed | Independent Marathon projection/reference proof |
| Residency | Bounded caches/targets, deduplication, unused TFX upload removed | Long-session validation as fidelity grows |
| UI | Pretty Preview toggle removed | Honest unsupported-path/evidence reporting |

Historical census totals and September 25 tests are not new full-suite results from this audit. Alkahest-equivalent settings are not independently proven Marathon defaults.

## M0 — Reproducible target evidence

- [x] Add exact `runner-thief-cryo-shift-combined` fixture through the normal shell loader.
- [x] Capture final/base-color output and CPU material/draw reports.
- [x] Distinguish Thief from the Vandal Cryo Shift fixture.
- [x] Save PNGs before pixel assertions to prevent stale images beside new metadata.
- [x] Persist a versioned per-draw manifest: root/component/geometry, IA ranges, transforms, LOD/variant, layout, bytecode hashes and authored state. Target capture streams SHA-256 over every present patch in referenced package families through the selected patch, since older blocks may reside outside the latest file. Version/engine/platform and unresolved tag references remain explicit.
- [x] Map jacket/chest/body/face/hair/marks to exact source ranges using isolated captures. All 21 normal isolates are mapped in [the draw contract](thief-draw-contract.md); comma-separated receiver captures locate C827's effect on the top-face/head band.
- [x] Record authored bindings, resolved TFX values, actual GPU bindings and shader reads separately. The manifest records authored resources/samplers and policy-resolved TFX values; the GPU sidecar records actual WGPU descriptors,18 visible indexed draws,30 shadow encodings and348 texture-view bindings joined to56 created views. Pinned DXIL SSA records shader reads for all t0..t7 independently. Controlled-pixel substitutions prove output influence for t0/t1/t2/t3/t5; Controlled branch-driving cases now prove output influence for t4/t6/t7 too; actual runtime branch ownership remains an M1 gate.
- [x] **TEST DIAGNOSTICS:** Separate decode completeness, input resolution, implemented semantics and consumption in diagnostics; exclude absent stages from coverage. Each present stage reports four independent contract fields; absent stages are `not applicable` and coverage-ineligible. Decoded operations remain under `tfx`, while `execution` now evaluates the explicit model-preview input policy instead of serializing the descriptor's empty-input snapshot.
- [x] Replace stale “not preserved” vertex messages with explicit visible-ABI decoding/implementation limitations.
- [x] Record capture success/failure atomically, including failures before readback. Journal starts before target loading, attaches the completed manifest, and records panic/drop or soft visual failures; four focused journal tests pass.

**Gate:** every visible range has traceable ownership and an explicit supported/unsupported contract. List required unknowns; Confirmed is not a rendered-correctness label.

## M1 — One complete body material

Start with `80A9D0E9` / PS `80A9D0E1`, then jacket/chest ranges identified in M0.

- [x] Extract/disassemble body VS/PS with SDK 22621 DXC; record binary SHA-256, shader model and input/resource/constant/output ABI in [the draw contract](thief-draw-contract.md). SDK 19041 remains unsuitable.
- [ ] Trace channels, resource views, UV/procedural transforms, sampler/filter/address/LOD, masks, coverage, detail normals and response through shader math. [Body SSA contract](thief-body-contract.md) maps all16 output stores to texture/constant/input dependencies; it is static value-flow evidence, not branch reachability or GPU consumption.
- [ ] Identify local texture/constant colors versus object/gear/global/context inputs. Do not presume every runner uses weapon six-color GearDye.
- [ ] Resolve required skin/root/component inheritance. Test nested/shared components and a sibling skin without cross-skin contamination.
- [ ] Prototype the same material using feasible translation, native execution or faithful reconstruction; compare demonstrated correctness and complexity. [Pinned translation experiment](shader-translation-experiment.md) validates body VS/PS and C827 SPIR-V; controlled C827 pixel execution and state-76 receiver blending pass four FP32 cases; controlled body VS arithmetic also matches original native texel-buffer execution bit-for-bit after scalar-SSBO lowering; the packaged compute producer also passes controlled rigid/matrix/remap/packing tests and feeds both VS paths directly with bit-exact outputs; compute scalar/uint4/float4 storage-buffer lowering also matches native producer outputs in those cases; dense production artifacts execute exact embedded body VS+PS together through WGPU, produce four finite/fresh MRTs, repeat bit-exactly and respond independently to generated-vertex and texture substitutions. Native packed-byte palette versus expanded SSBO reads also match for192 actual two/four-weight vertices with controlled identity/translated bones; another test uploads all eight actual packaged textures with exact full mip sizes, including the BC7 D3 volume. Actual sampler payloads supply address/filter/anisotropy; fixed -0.5 bias is represented in40 implicit sample instructions and changes a minification fixture. Live values, native sampler-bias parity, view/branch consumption and authored composition remain gates. See [runtime producers](thief-vertex-runtime.md).
- [x] Select the production strategy from that experiment. Hash-pinned offline DXIL-to-SPIR-V translation with identity-gated, separately validated ABI adaptations executes through WGPU Vulkan passthrough; the application embeds audited artifacts and never depends on runtime SDK/compiler files. See [the decision record](shader-translation-experiment.md#production-strategy-decision-2026-09-28).
- [ ] Implement the material together with required vertex behavior, runtime values, views/samplers and MRT outputs. No generic TFX uploader without consumers.
- [ ] Identify implementations by verified program semantics/identity, not resource counts/signatures alone.
- [ ] Check vector/scalar math, normalization, interpolation, precision, matrices and texture decode/sampling against authored behavior.
- [ ] Implement required stage/default/render states; diagnose unsupported states instead of guessing silently.
- [ ] Establish the body authored MRT consumer before live submission. Decode normal direction and its independent radius/response; do not treat literal RT1.a=.67 as proven roughness. Preserve signed RT3.xy motion separately from emissive, and use depth/coverage rather than RT0.a=0 as an opaque-presence gate. Verify Marathon attachment formats/views and downstream channel meanings; the current compatibility layout/lighting and Alkahest debug shader do not establish them. This M1 dependency precedes the full M4 light graph.

**M1 implementation ledger:**

- [x] **PROBE / NOT LIVE:** Export actual global/deferred technique TFX and execute exact paired global-light accumulation stages on actual body MRTs with an independent ambient oracle, source identity and descriptor-only adaptation proof. See current native-lighting increment. Final shaded consumer, native resource/sampler/runtime ownership and direct BRDF validation remain unchecked.

- [x] Connect audited body/chest/sleeves/hands/hardware programs through normal renderer in existing Base Color / MRT albedo modes. Replace seven old opaque submissions with exact CS → raw IA/VS → PS → four FP32 targets → scoped RT0 RGB inspection consumer. Shared dense draw mechanics live in `src/render/body_draw.rs`; admission/view scopes in `src/gui/model_surfaces.rs`; target ownership/projection in `src/render/surface_targets.rs`. Complete actual texture graphs, exact supported authored samplers, mutable view/skinning uniforms and bounded interactive targets are consumed. GPU controls and restored bits are verified in the current delivery checks. This checked increment does not complete native runtime/state, large export, final-lit consumer or the broader M1/M2 gates.

- [x] Consume those seven actual authored material outputs in normal Tiger GGX, Compatibility and Debug Lambert. Retained GPU-only `ViewerMaterialProjection` preserves uncovered pixels, independently decodes world normal/radius and maps native property/color channels into existing viewer lighting. A separate native coverage binding scopes Compatibility's GGX consumer to replaced surfaces. All three normal-loader GPU captures pass; independent f64 matrix/UNORM oracle maximum error1byte, UV and light controls alter intended lit pixels, every restored HDR/native attachment matches exactly. Prior Albedo and unrelated weapon final RGBA remain exact. This is applied viewer shading, not exact native final-lighting parity. Delivery checks and live audit above distinguish the remaining gaps.

- [x] **INFRA / PARTIAL LIVE:** Embed body mesh compute, body VS/PS, C827 VS/PS, D0F2 PS, hair VS/PS, eyes VS/PS, eye-detail PS and face PS as audited artifacts; pin original payload and SPIR-V SHA-256; reject wrong-stage and unknown identities. September 30 normal-loader capture resolves 28 of 39 present visible stage instances by package-payload identity, including two hair VS / one hair PS, two eyes VS / one eyes PS, face and eye-detail PS, five exact C827 VS uses and four D0F2 PS uses. Eleven instances remain unsupported; selection never relies on tag/layout guesses. Manifest: `target/quicktag-model-probe/runner-thief-cryo-shift-combined-base-color-roadmap-head-registry.draw-manifest.json`; summary: `target/cryo-runtime-audit/head-registry-selection.json`. Capture passes; decoded 1024x640 RGBA differs from the September 28 base-color capture in zero channels (2,621,440 compared, maximum difference 0; `head-registry-rgba-diff.json`). Selection and pixel preservation do not prove authored draw submission.
- [x] Enable Vulkan SPIR-V passthrough on UI and headless renderer devices; remove dependence on runtime SDK/compiler files.
- [x] **INFRA / PARTIAL LIVE:** Instantiate every embedded authored module in the renderer's immutable pipeline-resource cache; production-format GPU pipeline smoke test passes.
- [x] Replace the production texture loader's D3-to-D2 collapse with true depth-shrinking D3 mip upload; retain exact 2D behavior, add focused BC7 mip tests and retain a bounded first-slice D2 UI preview. Fresh Thief capture passes with BC sliced-3D enabled.
- [x] **INFRA / LIVE BODY:** Create the authored body VS/PS pipeline and descriptor layouts from the identity-gated registry. Body and shared layout-7 VS resources are compacted to dense bindings0-4; sparse high-numbered WGPU resources were proven to read zero and were removed. A real-device Vulkan smoke test validates ABI-compatible pipeline creation, dense PS/scalar-storage VS bindings, four MRT formats and target depth/raster state. Current normal-renderer cases submit the actual raw IA under explicit static viewer policy; native runtime/state parity remains open.
- [x] **LIVE / SCOPED:** Bind the authored body pipeline to verified target draws under the explicit viewer static-pose/runtime policy. Actual GPU producer outputs and exact admitted descriptors feed original strips; replaced opaque submissions are removed only after texture readiness. All five skins' verified shaded cases execute this path. Native game runtime/view/state parity remains in the separate unchecked ownership tasks.
- [x] Unify retained body draw submission with the production pipeline. `body_draw.rs` owns pipeline construction, stable PS/VS descriptor/IA identities and the original Uint16 strip range; `body_vertex.rs` owns explicit31/27-row VS inputs and layout7 IA. The renderer cache and actual-source material fixture use the same constructor; the fixture submits through the production encoder with bindings retained across runs. Actual-source GPU execution and normal-renderer pipeline creation each pass once (`target/roadmap-shared-body-draw-gpu.log`, `target/roadmap-shared-body-draw-pipeline.log`). Both complete material/graphics reports remain byte-identical (`target/body-draw-refactor-baseline/comparison.json`), and strict normal-loader RGBA has zero MAE/p99/max/changed fraction (`target/roadmap-shared-body-draw-rgba.log`). This does not resolve missing native values or replace live compatibility draws.
- [x] **EVIDENCE / NOT LIVE:** Locate Marathon's own named material consumers instead of assigning meanings from Alkahest. Both packaged `render_globals` tables (`80A06EB1`, `80A74EA8`) export42 named scopes and700 named pipelines apiece, with16 explicit null sentinels and no missing nonnull references. Nine named debug PS consumers from each table disassemble successfully with hash-pinned SDK22621; actual packaged TFX bindings export in one passing ignored test (`target/roadmap-native-mrt-consumer-bindings.log`). Smoothness/world-normal read Deferred+0xB0; metalness/transmission/emissive-intensity read Deferred+0xB8; source-color/overcoat-ID read Deferred+0xA8; AO/emissive read A8 and B8 together. These are native extern identities, not a captured assignment of attachments or formats. Reports: `target/cryo-runtime-audit/render-globals/named-pipelines.json`, `debug-consumer-disassembly.json`, `debug-consumer-bindings.json`.
- [x] **PROBE / NOT LIVE:** Execute the exact Marathon smoothness/world-normal consumers against controlled packed normals and actual-source body RT1. Sixteen controls include the negative-response diagnostic branch; alpha0/1 and equal-radius/different-direction smoothness outputs stay bit-exact. Actual GPU body RT1 feeds both decoders directly, without CPU reupload. All4,171 covered pixels pass baseline/zero-response-cap independent DXIL oracles; maximum smoothness error6.85e-7 (limit2e-6), normal error1.52e-7 (limit5e-6). All four normal display modes pass, fresh outputs repeat exactly, and complete material/graphics reports remain byte-identical. One focused GPU test and the structural identity/one-word descriptor-compaction/rejection script pass (`target/roadmap-native-normal-decoder-gpu.log`, `target/roadmap-native-normal-consumer-structural.log`). [Native consumer contract](thief-material-consumer-contract.md). Native attachment ownership/formats, complete light BRDF and live values remain separate gates.
- [ ] Execute AO/emissive packing and material-class gates separately; native source tracing exists but does not prove GPU output or the full light graph.
- [x] Prove controlled body t7 branch consumption: branch-driving cb0 rows 79-84 make t7 substitution alter RT0/RT1. Keep ordinary target inputs separate; the newer persisted t4/t6 reachability regression is recorded below.
- [x] Prove controlled output consumption for body t4 and t6. September 30 `packaged_body_detail_procedural_volume_textures_reach_outputs` passes: explicit synthetic constants activate t4 detail selection and t6's previously missed outer t2.B interval. Single-slot t4 substitution changes RT1; t6 substitution changes RT0; a persisted t7 case changes RT0. Every MRT channel remains fresh/finite, repeats bit-exactly, retains authored literal channels and keeps RT3 unchanged under each texture substitution. Reports persist all cb0 rows and per-slot changed channels (`target/cryo-shader-audit/body-pixel-t{4,6,7}-branch-output.json`); log: `target/roadmap-body-remaining-branches.log`. These synthetic controls establish reachable output influence, not the target's native metadata value or actual branch state.
- [ ] Prove native body constant-buffer/runtime, texture-view and sampler parity. Live draws already receive admitted TFX constants, textures and authored samplers under explicit viewer scope/view policies; native dynamic values and BC7 view intent remain unobserved.
- [x] Share and verify production body pixel binding. `render/body_material.rs` supplies the dense cb0/cb1/cb12, eight texture and three sampler ABI to renderer and both existing body GPU fixtures, with explicit125/31/29-row extents and true D3 t7. At this historical refactor checkpoint, five existing GPU tests passed; all five preserved pre-refactor JSON reports and output bits remained byte-identical (`target/body-binding-refactor-baseline/comparison.json`). Ordinary binary suite passed190 tests, zero failures,138 ignored, including pipeline creation (`target/roadmap-shared-body-material-regression.log`). Strict normal-loader target RGBA had zero MAE/p99/max/changed fraction (`target/roadmap-shared-body-material-rgba.log`). That checkpoint had no live body submission; the current five-skin path above does.
- [x] Execute actual draw0 material through direct CS/VS/PS. The exact embedded programs consume pinned actual IA and19,040-index strip, GPU-generated position/frame buffers, all eight actual complete mip chains (including shrinking-depth BC7 D3) and decoded sampler settings. Four FP32 MRTs cover the same4,171 pixels as the independent diagnostic VS test; all covered channels are fresh/finite, literal channels stay exact and fresh complete runs repeat bit-exactly. t0 substitution changes albedo while normal/motion targets remain exact. The explicit static pose/camera, synthetic procedural-frequency/PS View and neutral state are experiment inputs, not native resolution. Log: `target/roadmap-actual-body-material-gpu.log`; report: `target/cryo-runtime-audit/body-static-preview-material.json`. This closes actual-source full material execution, not native runtime/state/view parity or live submission.
- [x] Establish body row75's conditional output gate. Actual target preview cb0[84].x is0; pinned original SSA places every row75.x use behind the row84 multiply/saturate. Explicit row75 values0/.25/.5/.75/1 yield identical full actual-source outputs. Synthetic row84=.25 makes row75 endpoints0/1 change output, rejecting a dead-binding false pass. This proves finite-input output insensitivity at the current zero gate; it does not resolve opcode0x64 or prove the in-game gate. If row84 changes, row75 may be required. Trace and limits: [body gate contract](thief-body-contract.md#body-row75-effective-output-gate-2026-09-30).
- [x] Verify full actual-body static output against the original texel-buffer program. `full_actual_body_static_outputs_match_native_texel_buffers` dispatches all44,950 vertices with the original R8G8B8A8_UINT palette view and compares both complete output buffers to the exact embedded dense adapter and preserved WGPU hash. Four freshly poisoned dispatches preserve every active word and both guard bands; all outputs match bit-for-bit, SHA-256 `db941e89c625c89043ea56a0d7c5e6899f592d8c378d2afe798fffff561b526e`. This establishes static mode0 output parity despite21,907 unused rigid palette OOB fetches; it does not observe their fetched values or establish native Direct3D/animation semantics. One focused test and the eight-test native/WGPU regression pass, zero failures. Report: `target/cryo-runtime-audit/body-static-native-parity.json`; logs: `target/roadmap-full-body-native-parity.log`, `target/roadmap-full-body-native-regression.log`.
- [x] Prove normal-radius encoding and signed motion on actual-source body MRTs. A zero cb0[35].x cap yields decoded normal length.75 at all4,171 covered pixels (max error1.59e-7, limit2e-6), preserves direction (1.44e-7, limit1e-5), and leaves other targets/alpha bit-exact. Opposite synthetic previous-projection translations produce signed RT3 deltas(.0625,.125)/(-.0625,-.125) with max error7.46e-9 (limit1e-6); all material targets remain exact. Baseline full MRT hash is unchanged. Focused GPU test passes (`target/roadmap-body-mrt-semantic-controls.log`); full inputs/errors live in the material report's `mrt_semantic_controls`. This proves shader packing, not native response naming, attachment formats or live composition.
- [ ] Complete body scene/runtime ownership: PS cb0 has 125 inline rows with all current policy outputs resolved except row 75; shaders require a 31-row cb1 prefix and separate 27-row VS / 29-row PS View images. The fresh target technique declares used/compatible scope mask131 (`0x83`, decoded Frame/View/Skinning), consistent with DXIL's `scope_skinning` cb1 name. Skinning declares4098 zero rows without TFX; its dynamic writer and consumed31-row prefix remain unowned. The 15-row RigidModel image is comparison evidence, not the body binding. Package geometry proves UV/scale/offset candidates, but current/previous matrices and generated-buffer bases remain unowned. View numeric source offsets are traced; live values and semantic meanings remain unresolved.
- [ ] Prove BC7 linear/sRGB view intent for body t2-t7 and native sampler-bias parity. t0 sRGB and t1 linear-data intent are known; exact s1/s2/s3 address/filter/anisotropy values are decoded, while the embedded PS carries the required -0.5 bias in 40 sample instructions.
- [ ] Recover Marathon runtime-selector semantics for TFX opcode `0x64`; target body uses selector13. Quicktag tentatively labels it texture metadata. The census rejects direct local texture slot13, but neither the producer's semantic class nor field table/value is proven; cb0 row75 stays explicitly unresolved. Its exact target equation is `saturate((1-max(A590EEC6.x,714FE9CA.x))*M13.x)`; the subsequent cubic coefficients `[0,0,1,0]` leave this value unchanged. The bounded read-only installed-binary probe found no usable evaluator/table or matching PDB; build/hash and limitations are recorded in [the body contract](thief-body-contract.md#opcode0x64-source-limitation-2026-09-30). The user confirms no existing native capture or shader symbols. Other packaged-data work continues; no native value is guessed.
- [ ] Establish native runtime ownership/dispatch for the packaged body producer. Live packaged CS→VS consumption now works under explicit viewer static-pose policy; native compute scope/remap/batch/current-previous parity remains unproven.
- [x] Prove direct GPU body producer-to-VS consumption with actual IA. September 30 `actual_body_producer_feeds_authored_vertex_graphics` chains exact embedded CS and body VS in one encoder, with GPU outputs bound directly as t2/t3 and static t4 alias. Pinned D127/D128/D12B streams submit authored draw0's19,040-index Uint16 strip. Two diagnostic passes expose all eight varyings through four FP32 MRTs. All covered channels are fresh/finite; full fresh runs repeat bit-exactly. Independent CPU strip coverage matches all4,171 covered pixels with zero edge disagreements; raw IA reconstruction error4.26e-7 (limit2e-6), pixel-center projection error1.51e-5 (limit2e-5), previous/current error1.79e-7 (limit2e-6; authored precision paths differ). Producer position offset changes coverage; generated-frame substitution changes normals only; raw UV substitution changes UV only and zero UV yields declared offsets. Three focused tests pass, zero failures (`target/roadmap-actual-body-graphics.log`); report `target/cryo-runtime-audit/body-static-preview-graphics.json`. This is viewer-owned static pose/camera, diagnostic PS and explicit neutral state. Later ledger entries close the rigid frame oracle, actual material PS execution and original-versus-dense static output parity. Native runtime/OOB fetched values, inherited state and live submission remain open.
- [x] Prove full-source static-preview producer output. The ignored WGPU regression executes the exact embedded CS on all44,950 actual body vertices with pinned source/palette hashes, raw layout7 inputs, rigid mode0, identity remap and one viewer-owned batch. Dispatch703x1x1 follows packaged local size64; 42 trailing invocations are rejected. Both outputs are fresh for every vertex, repeated bits match, and four guard words per buffer remain poisoned. Independent signed16 decode and21/21/22 packing match all134,850 position components exactly (zero allowed/observed quantization delta). Original controlled producer test also passes: two tests, zero failures (`target/roadmap-full-source-compute-gpu.log`); report `target/cryo-runtime-audit/body-static-preview-output.json`. Weighted palette ranges are checked. Rigid records still issue an unused out-of-range palette fetch before mode selection; the later full-source native Vulkan fixture proves static mode0 output parity; fetched OOB values and native Direct3D semantics remain open. This is a static preview policy, not native bones, animation, remap/batch ownership or current/previous resource capture; actual VS handoff now passes under the separate static-policy consumer.
- [x] Embed and create the dense production compute pipeline. The body producer now embeds the validated WGPU artifact (`a095c0d2b92798c6e892c3c82a0c0270d0462e3ba40537cb0e3f6a0299357ab7`) rather than sparse native-test descriptors. Production/UI/headless device limits request its eleven storage bindings. Structural validation and explicit rejection of eye/hair programs pass (`target/roadmap-compute-structural.log`). Cached-pipeline creation passes once (`target/roadmap-body-compute-pipeline-gpu.log`); the controlled WGPU dispatch now reads the exact embedded asset and passes once (`target/roadmap-embedded-compute-execution.log`). This closes the immutable compute ABI gate, not live scope108 inputs, dispatch selection or generated-buffer aliases.
- [ ] Resolve compute scope 108 before dispatch: package layout strongly identifies t0/t1/t2 with interleaved stream0 and t3 with packed skin records; u0 semantically produces VS t2 positions and u1 produces VS t3 normal/tangent. Runtime t4 previous-position ownership, t4 bone matrices, t5 batch table, t8 remap, descriptor aliases, cb0 values and dispatch-Y batches remain unproven.
- [x] Share and verify production body compute execution. `render/body_mesh.rs` owns dense pipeline/layout creation, explicit original-slot buffer binding and dispatch; renderer and actual-source producer/graphics fixtures use that same component. Immutable bind groups are retained while input buffers may change in place. Cached body graphics topology was corrected from reconstructed triangle lists to the target's authored Uint16 strip. Three focused GPU tests pass after extraction, including a pinned pre-refactor full-output SHA-256 assertion. Ordinary binary suite passes190 tests, zero failures,138 ignored, including renderer pipeline creation (`target/roadmap-shared-body-regression.log`). Strict normal-loader target RGBA remains zero MAE/p99/max/changed fraction (`target/roadmap-shared-body-rgba.log`). This closes shared compute code and target topology, not native input values or live authored submission.
- [x] Add independent full-source normal/tangent packing oracle. An f64 analytic L1 oct projection checks all179,800 signed16 output components from44,950 actual source vertices, with the declared maximum difference1 and observed maximum1. It analytically cancels Euclidean normalization instead of duplicating the shader's f32 fast rsqrt/reciprocal chain. SNORM clamp, hemisphere fold/sign and signed rounding are explicit. Exact entire-output SHA-256 and fresh-run equality remain strict, independent preservation gates. Three producer/graphics tests pass (`target/roadmap-body-frame-golden-gpu.log`); structured results are in `body-static-preview-output.json`. This proves rigid-mode source packing, not deformed/native animation frames.
- [x] Verify controlled full-graphics body pixels: exact embedded body VS+PS execute together with dense cb1/cb12/generated buffers and dense PS descriptors; all four MRTs are finite/fresh, repeat bit-exactly, retain audited literal channels and respond to generated-normal and t0 substitutions.
- [ ] Route authored MRT outputs through the verified native final composition graph. Existing viewer GGX/Lambert conversion now consumes body materials live; exact native lighting and remaining-family composition are still open.

**Gate:** albedo, coverage, normal and material channels agree with independent authored-program evaluation/capture at controlled pixels/inputs. Runtime changes affect intended GPU outputs. Colors follow data, not skin-hash tint overrides.

**Strategy decision record:** supported bytecode/stages, missing features, resource/root/descriptor ABI, encoding, precision, portability, WGPU integration, residency and maintenance. Native execution must support these Marathon programs; Alkahest D3D11 is not a drop-in DXIL solution. Reconstructed families must implement relevant math/branches, not merely resemble the screenshot.

## M2 — Hair, face, decals and remaining target surfaces

- [ ] Replace CompactHair's neutral `mix(0.62, 1.0, mask)` with the verified `80A9D171` contract, including coverage, color, tangent/normal and response.
- [ ] Verify other hair/face/eye ranges independently.
- [ ] Resolve stage-2 `80A9D0F2` and `80A9C827`; identify markings belonging to body layers instead.
- [x] Inspect existing target decal disassemblies and match their binary SHA-256 values to fresh exports: D0F2 is scene-normal/color/mask; C827 is a textureless procedural multiplier.
- [x] Implement C827's exact TEXCOORD3.y/cb0-driven mask and state76 multiplier in normal viewer color receivers; preserve full external cb0, original paired VS/PS and alpha/non-color attachments. This scope uses the explicit viewer pose/camera; native inherited-state parity remains open.
- [x] Locate C827 buffer `80A9D945` and preserve its nine registers. Fix Marathon slot/tag offsets, external precedence and final partial register; assert the target buffer/gate/program identity in the GPU fixture.
- [ ] Verify Decal externs, scene-normal reads, atlas/mask channels, UVs, selectors, thresholds, gates, tint, blend/write masks and encoding.
- [ ] Diagnose missing detail as absent geometry/binding, skipped branch, wrong sampling or wrong composition using isolated outputs.
- [x] Join authored texture tags to selected resources, created GPU views and encoded draws using stable draw IDs. `scripts/audit_thief_texture_bindings.py` verifies18 visible submissions in the preserved normal-renderer capture. All four D0F2 marking draws16–18/20 bind exact color80A9D137 and mask80A9D135, with correct BC1 sRGB/BC4 views; none of their direct packaged textures is missing. Body draw0 selects only two of eight authored tags; six are absent from compatibility material bindings. Full report: `target/cryo-texture-audit/thief-live-texture-bindings.json`; log: `target/roadmap-thief-live-texture-bindings.log`. This separates binding from authored evaluation; it does not prove UV placement, missing native branches or decal composition.
- [x] **PROBE / NOT LIVE:** Prove actual body marking ranges16–18 through the exact paired80A9A2D7 VS: original raw IA, GPU producer buffers and cb1[6] -> fragment TEXCOORD3. Diagnostic coverage is313/526/130 pixels; original source UV influences every covered pixel, zero UV returns the declared offsets, and offset controls(+.125,-.25) reach the fragment unchanged within1.79e-7 (predeclared limit1e-6). Independent CPU coverage has zero interior disagreements; draw17 has one allowed1/256-pixel edge disagreement. Fresh outputs repeat bit-exactly. One focused GPU test passes (`target/roadmap-thief-body-marking-uv-gpu.log`). Static viewer inputs and diagnostic PS only; native pose/culling/depth and head20 remain open.
- [x] **PROBE / NOT LIVE:** Execute actual body marking IA through exact paired80A9A2D7 VS /80A9B4C7 PS with packaged full-mip atlas80A9D137/mask80A9D135 and decoded sampler80A60020. The production full-mip export passes once (`target/roadmap-thief-decal-texture-export.log`). Three body ranges produce79/156/31 marked receiver pixels; isolated atlas substitution changes only RT0, scene-normal substitution only RT1, and zero mask preserves all four seeded targets exactly. Literal state26 masks preserve alpha, RT2 G/A and allRT3; fresh outputs repeat exactly. One actual-source GPU test passes (`target/roadmap-thief-body-marking-material-gpu.log`); full UV/material report: `target/cryo-runtime-audit/body-marking-vertex-uv.json`. This is actual geometry/textures and exact programs under neutral depth/culling, controlled Decal/View/scene-normal and FP32 receiver inputs. Native attachments/runtime, live composition, head20 and game-pose parity remain open.
- [ ] Implement remaining target body/head families.
- [ ] Complete remaining headwear/live head-family integration. Historical registry gap: `80A9D0F7` (draws1/13 head hardware), `80A9D10F` (draw5 hands) and `80A9A9D3` (draw14 headwear). After chest/sleeve registration, 32/39 present stage instances are registered: four unsupported PS uses from these three families and three shadow-only VS `80A9A317` uses on draws2/6/10. Hardware `80A9D0F7`, hands `80A9D10F`, chest/pelvis `80A9D103` and sleeves `80A9D11A` now have live material submission in Albedo and all three viewer shaded modes. Headwear `80A9A9D3` remains unsupported; the quoted32/39 selection below is historical. Registry selection alone does not implement these surfaces. Similar descriptor counts must not substitute for each program's payload identity.
- [x] Adapt/embed distinct chest PS80A9D103 and sleeve PS80A9D11A; execute actual strips3/4/7, original layout7 IA and body CS GPU outputs with all nine packaged full-mip textures. Source hashes, all15 descriptor remaps, exactly42/51 −0.5 sample biases, Vulkan validation and unrelated-program rejection pass. All4344/1475/2430 covered pixels are fresh/finite with exact repetition and independently traced literal channels; CPU coverage has zero interior disagreements. Each t1..t6 isolated substitution affects RT1, t0 changes only RT0, and raw UV controls affect material placement without changing motion. cb0 rows76/79 are conditionally output-inert under gates85/88=0; a synthetic active gate proves reachability without claiming native resolution. These programs now also feed live Albedo and viewer shaded modes; native runtime/state parity remains open. Full [contract](thief-layered-material-contract.md); actual GPU report `target/cryo-runtime-audit/thief-layered-material-execution.json`; serialized native/WGPU regression8 passed (`target/roadmap-thief-layered-native-regression.log`). The original probe did not close native inputs/state/composition; current UI adoption is the separate live delivery above.
- [x] **INFRA / NOT LIVE:** Adapt and embed mouth PS `80A9AA27` by its own payload identity. Fresh DXIL SHA-256 `bd135d1b7ca9ce2f18f65c0204839ce5b08f2d3f30cf7d97ba4474bb1cf4f621`; adapted SPIR-V `b61961f71148389e83e166e54fbe5a27567829af09c4fad2d231ae135238e7b9`. Dense ABI retains cb0/cb1/cb12 (49/31/29 rows), two D2 textures and s1. Exactly two implicit sample sites retain authored -0.5 bias; the earlier inventory's three-sample count was wrong. Structural validation, pinned hashes, unrelated-program rejection and two registry tests pass. Cached eyes-VS/mouth-PS pipeline creation passes on Vulkan (one test; `target/roadmap-mouth-pipeline-gpu.log`). Fresh normal-loader capture selects 29 of 39 present visible stage instances, ten still unsupported (`target/cryo-runtime-audit/mouth-registry-selection.json`); strict decoded RGBA matches the preserved matrix-runtime capture exactly, with zero MAE/p99/max/changed fraction (`target/roadmap-mouth-registry-rgba.log`). Inherited states, controlled paired graphics/native parity and live submission remain open.
- [x] **PROBE / NOT LIVE:** Execute exact embedded mouth PS with controlled varyings and dense resources. The ignored `mouth_spirv_synthetic_fullscreen_preserves_mrt_contract` passes once: all four FP32 MRTs are fresh, finite, repeated bit-exactly and preserve independently traced literal channels. Single-slot t0 substitution changes only RT0.rgb/RT2.g; t1 changes only RT1.rgb. The first expectation inverted the texture handles; original DXIL `CreateHandleFromBinding` and translated descriptors independently establish the corrected mapping. Unrelated channels must remain bit-identical. Log: `target/roadmap-mouth-controlled-gpu.log`; structured constants/output bits: `target/cryo-shader-audit/mouth-controlled-output.json`. This is synthetic PS execution, not the eyes-VS pair, native lowering parity or live mouth fidelity.
- [ ] Validate mip/array/3D/cubemap selection, source decode and linear/sRGB views for each shader input; storage format alone does not prove intent.
- [ ] Test that unrelated shaders sharing resource layouts do not enter the implementation.

**M2 implementation ledger:**

- [x] Embed and identity-gate the translated C827 VS/PS; preserve their separately audited descriptor ABIs. Native texel-buffer versus scalar-SSBO execution matches all 25 C827 VS output scalars bit-for-bit across valid ranges; controlled output verifies `TEXCOORD3.y = cb1[6].y * TEXCOORD0.y + cb1[6].w`, and a binding-swap negative rejects wrong ownership.
- [ ] Complete independent actual-target C827 fragment-varying readback. Live exact VS/PS now consume original TEXCOORD0 and cb1[6]; the GPU +.125 offset control changes both receivers and restores bit-exactly. This establishes live UV influence, alongside controlled native VS parity; a standalone target-fragment numeric oracle is still required for this broader gate.
- [x] Add the native C827 pipeline with exact paired programs, state-76 RT0 RGB destination multiplication, RT0 alpha preservation and zero write masks for RT1-RT3; real-device pipeline creation passes.
- [x] Bind live C827 cb0/vertex resources and route only identity-verified eligible draws through the native pipeline. Actual target draw19 receiver verification passes:10 HDR /8 albedo pixels change in the independently located top-face band, normals/properties/alpha remain bit-exact, and no compatibility C827 draw remains. Selection uses program payload SHA, authored layout/stage/state and exact per-strip compute ownership, never the skin hash. Viewer bind pose/camera/color projection are explicit; native runtime/light parity is not claimed.
- [x] **INFRA / NOT LIVE:** Adapt and embed identity-gated D0F2 PS `80A9B4C7` with its two fixed -0.5 sample biases; reject unrelated shader identities. Compact its six descriptors to one dense WGPU bind group: the previous sparse texture bindings 18-20 returned zero on the controlled Vulkan execution despite verified uploads. Cache its exact `80A9A2D7` VS pair, triangle-strip IA, state-26 blend and RT0 RGB / RT1 RGB / RT2 R+B / RT3 none masks. Structural gate, Vulkan validation, registry hashes and real-device pipeline creation pass; compatibility submission remains active.
- [x] **PROBE / NOT LIVE:** Execute exact embedded D0F2 PS in a controlled Vulkan graphics pass. Fixed scene-normal/color/mask substitutions alter their audited writable outputs; threshold `0.125` gates the mask; state-26 preserves RT0/RT1 alpha, RT2 G/A and every RT3 channel bit-exact. Bridge VS and controlled cb0/cb12 remain probe-only.
- [ ] Resolve and implement live D0F2 scene-normal/color/mask resources, paired authored UV path and production cb0/cb12 values.
- [ ] Register and execute identity-gated hair, eyes, eye-detail and face programs with their exact runtime inputs. All four families are registered. Shared Face/EyeDetail now execute through actual normal-renderer GPU captures; separate hair, eyes and mouth producers/live inputs remain open.
- [x] **INFRA / NOT LIVE:** Translate, identity-gate and embed hair PS `80A9A9D5` and its exact paired VS `80A9A9DB`. The PS adapter compacts cb0/cb1/cb12, five textures and two samplers to dense bindings and inserts exactly fifteen fixed -0.5 sample-bias operands. The VS adapter lowers float4 t0 plus scalar-U32 t2/t3/t4 to four dense read-only storage buffers without changing index arithmetic, and compacts its three cbuffers. Vulkan 1.2 validation, pinned structural scripts, registry hash tests and real-device pipeline creation pass. The cached pipeline preserves authored layout 7 and two-sided rasterizer 1; inherited blend/depth state remains explicit and unresolved.
- [x] **PROBE / NOT LIVE:** Execute the exact hair VS/PS pair with controlled inputs; prove all MRTs fresh/finite/repeatable plus texture and front-face influence. On September 30 the ignored `hair_pixel_probe::tests::packaged_hair_graphics_program_executes_controlled_two_sided_mrt` passes (one test, zero failures): exact embedded VS/PS, synthetic layout-7 IA, dense resources and four FP32 MRTs. Every channel is written, finite and repeatable bit-for-bit; independent authored literal channels remain exact. Texture substitutions change material outputs; reversed index winding preserves vertex IDs and changes front-face-sensitive outputs. Probe cb1/cb12 extents were corrected to 31/29 rows. Log: `target/roadmap-hair-controlled-gpu.log`. This is controlled execution, not native lowering parity or live hair fidelity.
- [x] **INFRA / PARTIAL LIVE:** Adapt, identity-gate and embed eyes VS `80A9A2C7` / PS `80A9D1D1`, eye-detail PS `80A9BE2C` and face PS `80A9D0DF`; cache eyes with its exact VS and eye-detail/face with body VS `80A60DAD`. Dense pixel ABIs retain respectively 43, 4 and 27 fixed -0.5 sample-bias operands, exact cbuffer extents and D2/D3 dimensions. Pinned structural scripts and registry hashes pass; September 30 `creates_model_preview_pipelines` passes on a real Vulkan device (one test, zero failures; `target/roadmap-head-pipeline-gpu.log`). The original neutral pipeline states were ABI gates only. Face/EyeDetail now use exact admitted states through the shared live material path; separate eyes live submission, native runtime/inherited-state parity remain open.
- [ ] Implement and bind distinct hair producer `80A9A9DE`, live VS cb0/t0/t2/t3/t4/cb1/cb12, PS cb0/cb1/cb12, five exact textures including D3 t4, both authored samplers and inherited render state. Do not route hair through the body producer/layout.

**Gate:** target surfaces have validated material outputs/coverage. No required input or silent generic substitute is unexplained. Blue jacket, dark hair, yellow chest and white marks are regional reference checks, not substitutes for program proof.

## M3 — Runtime contract and representative coverage

Expand a working slice rather than starting an all-engine rewrite.

- [ ] Build typed value/resource/sampler resolution in dependency order: Frame, View, RigidModel, Gear/TextureSet, Decal, Deferred/Light/Shadow, global/object/context, editor/material and texture metadata.
- [ ] Resolve required TFX operations and runtime-generated resources; add operation/output-binding tests and report required unknowns.
- [x] Correct Marathon matrix-store opcode `0x55` and execute four-column extern pushes/stores with exact column order. Missing/unknown columns invalidate the matrix dependency; scoped overrides apply to every column. Independent nonidentity transform, stack-balance and partial-column regressions pass. Remove unverified `0x54`/`0x56` legacy aliases. September 30 ordinary binary suite: 190 passed, zero failed, 138 ignored (`target/roadmap-matrix-runtime-regression.log`); fresh package scope export confirms six PS / five VS View matrix stores. Lighting GPU harness now requests the same required SPIR-V device feature as the renderer. Exact target RGBA capture remains unchanged: MAE/p99/max/changed fraction all zero (`target/roadmap-matrix-runtime-rgba.log`). This is CPU scope semantics and pixel preservation, not live authored GPU consumption.
- [x] Preserve full external constant-buffer payloads, including partial registers; initialize external data before TFX writes and report unavailable buffers/unresolved numeric outputs. Unknown extern labels now retain scope IDs; scoped numeric inputs resolve independently and Unknown values remain unresolved. CPU contract only; visible consumers remain open.
- [ ] Keep dynamic GPU state separate from immutable cache identity. No unused arrays or per-frame bind-group/render-bundle churn.
- [ ] Make visible stage/variant selection authoritative and reconstruct inherited defaults.
- [ ] Consume extra UV/color/custom and per-instance inputs, bones/weights, skinning resources, morph/soft deformation and stage-specific transforms where read.
- [ ] Verify full RGB/alpha equations, independent MRT/write masks, blend constants, depth/stencil comparisons/references/masks, rasterizer states and bias. Unknown must not silently mean alpha blending.
- [ ] Inventory visible VS/PS identities/semantics across runners and weapons; prioritize fixture impact plus population coverage.
- [ ] Add sibling Thief skin, another runner, Vandal Cryo Shift, representative weapon and multi-mod weapon fixtures.
- [ ] Report separate denominators for decoded TFX, resolved required inputs, implemented programs, consumed vertex ABI and fallback draw/pixel coverage.

**Gate:** multiple distinct families work without per-asset patches; ownership and unrelated contracts remain correct.

## M4 — G-buffer, lighting and shadows

Material MRT outputs needed by M1/M2 are early work; the complete light graph follows.

- [ ] Reconstruct target formats, MRT packing, normals/material flags, views and producer/consumer behavior together.
- [ ] Locate/verify specular-lobe/tint/iridescence and other required lookups; replace custom GGX using Marathon evidence.
- [ ] Decode lights, volumes, transforms and shadowed-lighting techniques; verify matrix conventions.
- [ ] Reconstruct probe/cubemap selection, diffuse/voxel data, fade/extents, relighting and globals.
- [ ] Derive exact shadow projection/DeferredShadow data and independently verify Marathon bias defaults.
- [ ] Implement remaining shadow shader families, runtime resources, coverage and deformation; equal layouts do not prove equal programs.
- [ ] Add visible/shadow range and silhouette overlays, raw depth, shadow UV, receiver-depth delta, stage/layout/state and hard-compare/active-PCF diagnostics.
- [ ] Resolve feature stages, including raw stage 3, from runtime/non-mesh evidence rather than sequence alone.

**Gate:** controlled geometry/material scenes match verified light/shadow intermediates. Removing one historical artifact does not close shadow fidelity.

## M5 — Reference scene and final parity

- [ ] Match game pose/animation, camera/projection, lights/probes, resolution, time, exposure and transfer function; record unavailable inputs.
- [ ] Reconstruct required tone mapping/exposure, bloom, SSAO and AA. Separate adjustable viewer controls from reference settings.
- [ ] Compare silhouette, coverage, albedo, normals, material channels, shadow depth, linear HDR and final output in order.
- [ ] Compare jacket, chest, hair, face, leg marks and boots separately; record provenance and registration/cropping.
- [ ] Use exact RGBA diffs for fixed-environment refactors, documented tolerances for independently implemented floating-point paths. Set thresholds before accepting corrections.
- [ ] Review fidelity baseline updates against independent evidence; never automatically bless current wrong output.
- [ ] Keep unmatched-screenshot claims qualitative. Histogram similarity is not a percentage of engine fidelity.

**Gate:** the agreed reference scenario meets explicit intermediate/final criteria and remaining differences are explained. Report achieved scope until then, not “100% faithful.”

## M6 — Performance and cleanup throughout

- [ ] Check cold/warm opens and switching after material milestones; retain cache bounds and correct identities/invalidation.
- [ ] Measure CPU/GPU residency, upload and resource-creation counts over long browsing sessions. A 256 MiB target-set limit is not total memory.
- [ ] Allocate consumed inputs only; share immutable package resources and account for asynchronous loads.
- [ ] Remove replaced shader branches and family guesses after validation.
- [ ] Separate parsing, runtime inputs, immutable program/pipeline state, submission and diagnostics.
- [ ] Preserve vanilla/default-mod lighting scale across mod switches as a viewer control.
- [ ] Keep UI evidence/fallback labels accurate without a second fidelity mode.

## Reproduce this audit

PowerShell from repository root, with unrelated `QUICKTAG_PROBE_*` overrides cleared:

```powershell
$env:QUICKTAG_MARATHON_PACKAGES = 'D:\SteamLibrary\steamapps\common\Marathon\packages'
$env:QUICKTAG_MODEL_PROBE_CASE = 'runner-thief-cryo-shift-combined'
$env:QUICKTAG_MODEL_MATERIAL_REPORT = '1'
$env:QUICKTAG_PROBE_ZOOM = '2.0'
$env:QUICKTAG_PROBE_PASS = 'base-color'
cargo test --no-default-features --features wordlist gui::model_renderer::tests::renders_weapon_mod_skin_visual_comparisons -- --ignored --exact --nocapture
$env:QUICKTAG_PROBE_PASS = 'final'
cargo test --no-default-features --features wordlist gui::model_renderer::tests::renders_weapon_mod_skin_visual_comparisons -- --ignored --exact --nocapture
```

Outputs: `target/quicktag-model-probe`. A pass checks basic rendering, not game fidelity. The material report exports shader binaries too.

For a pixel-preserving refactor, set `QUICKTAG_PROBE_BASELINE` to the preserved
capture before running, and set `QUICKTAG_PROBE_MAX_MAE`,
`QUICKTAG_PROBE_MAX_P99`, `QUICKTAG_PROBE_CHANGED_THRESHOLD` and
`QUICKTAG_PROBE_MAX_CHANGED_FRACTION` to `0`. Use a new output suffix; never
overwrite the baseline to accept a change. September 30's
`base-color-roadmap-matrix-runtime.metrics.json` records all-zero differences
against `base-color-roadmap-head-registry.png` under these exact thresholds.

The first Thief run exposed a test assumption that every combined runner uses the common character-surface ABI. The unresolved Thief diagnostic is excluded from that unrelated assertion; existing supported-family checks remain. The ordinary nonblank assertion is unchanged. Zoom 1.0's small dark base-color image failed its fixed 5,000-pixel threshold; documented zoom 2.0 passes both captures. This is a framing/probe issue, not a material fix.

## Old backlog coverage

| Old workstream | New location |
| --- | --- |
| P0 stages/ranges/raw inputs | Foundation, M0, M3, M4 |
| P0 TFX / P1 externs | M1 consuming slice, M3 expansion |
| P1 states | M1 required subset, M3 full mapping |
| P2 shadow fidelity/diagnostics | M4 |
| P3 shader inventory/strategy | M1 executable decision, M3 population |
| P4 G-buffer/BRDF/lights/probes | M1 outputs, M4 graph |
| P5 material/TFX coverage | M0–M3 |
| P6 deformation/streams | M1 required ABI, M3 expansion, M5 pose |
| P7 postprocess | M5 |
| P8 conformance/reference tools | M0 onward, M5 |
| P9 naming/UX | M0, M6 |

**Next implementation — user priority:** implement remaining cross-skin material variants through exact compute/VS/PS/resource contracts; establish HairCS active simulation/bind-pose inputs and separate eyes/mouth producers; resolve native decal scene-normal/depth/snapshot composition across compiled families. Preserve scoped exact refactor controls. Native final-lighting attachment/runtime/sampler ownership remains separate. M1/M2 stay incomplete.

**Historical October1 priority checkpoint:** Thief9, Digital13, Assassin Cryo8, WEAVErunner10, Cyber6 authored material draws reached shaded output. Counts are superseded by the current delivery block/census. Face/eye-detail was live; hair remained pale. Assassin Cryo C107 and additional Vandal variants had unsupported producers/vertex inputs. Correct atlas/mask tags alone do not establish native decal composition. The earlier marking overhang does not prove an RT0-coverage restriction; its failed oracle remains visible. HairCS default branch still reads t8 and simulation rows14–64, so zero-filled buffers are not a proven finite pose. Native final-lighting ownership remains open.

**Historical chest/sleeve probe pass before live source-color integration:** structural, actual-source graphics and texture/UV influence gates pass; the new [layered material contract](thief-layered-material-contract.md) records all evidence and limits. Fresh serialized native/WGPU harness8 passed, zero failures; fresh ordinary binary suite190 passed, zero failures,141 ignored (`target/roadmap-thief-layered-ordinary-regression.log`). Production full-mip export passes once. Prior body material/graphics reports remain byte-identical. Fresh normal-loader RGBA matches the preserved shared-body-draw capture exactly: MAE/p99/max/changed fraction all0 (`target/roadmap-thief-layered-rgba.log`). Fresh manifest selects32/39 present stage instances (`target/cryo-runtime-audit/layered-registry-selection.json`); remaining seven are four visible PS uses (head hardware1/13, hands5, headwear14) and three shadow-only VS uses2/6/10. cb0 rows76/79 remain unresolved, conditional output-inactivity proved separately. Live authored UI submission remains0; M1/M2 incomplete. Next user priority: actual decal scene-normal/Decal/View resolution and receiver composition, plus head20/C827 ownership and remaining visible surfaces.

**Historical native decoder/decal validation before source-color integration:** the native normal/decal increment passes all eight ignored tests in the serialized native/WGPU harness, zero failures (`target/roadmap-native-decoder-decal-regression.log`). The original seven-family pixel-adapter regression also passes (`target/roadmap-native-consumer-adapter-regression.log`). Separate named-consumer binding export and actual decal texture export each pass once. Complete pre-existing body material/graphics JSON reports remain byte-identical after these probes. The prior ordinary190-test /138-ignored result below predates the two added export tests; it is historical, not a fresh full-suite result. Shared draw refactor's strict normal-loader RGBA gate remains all-zero; this later increment changes probes/diagnostics only and claims no new UI visual correction. Live authored submissions remain0; M1/M2 remain incomplete.

**Runtime evidence (September 30):** actual Marathon `s_scope` export passes (one test, zero failures): 10 View/RigidModel/Skinning/Frame candidates from 84 scopes, with fresh hashes, constant-buffer slots/images and decoded TFX operations/resources. Report: `target/cryo-runtime-audit/preview-runtime-scopes.json`; log: `target/roadmap-runtime-scope-export.log`. View declares different PS/VS cb12 images (29/27 rows); RigidModel declares 15 cb1 rows; Skinning declares 4098 zero rows with no TFX. Opcode `0x55` now decodes/executes as four-row matrix output. Source offsets for all View rows are traced, while their live values, matrix meanings and actual scope-selection ownership remain open. Candidate strings alone do not prove live binding ownership.

**Historical body-runtime verification before source-color integration:** shared production body compute/pixel bindings execute through existing controlled fixtures and the new actual-source full CS/VS/PS test. Five existing body GPU tests pass with all preserved reports byte-identical; actual-source material execution passes once with exact repetition, MRT coverage/literals and isolated t0 influence. The conditional row84 gate rejects row75 influence in current preview inputs; enabling a synthetic gate proves row75 remains a reachable shader input. Full original-texel versus embedded-dense parity now passes all44,950 vertices with fresh guards and the preserved WGPU hash; the eight-test native/WGPU regression passes (`target/roadmap-full-body-native-regression.log`). The later actual-source MRT-control test passes (`target/roadmap-body-mrt-semantic-controls.log`): normal radius responds independently of direction/alpha, RT3 preserves signed projective deltas, and the baseline full material hash remains unchanged. Prior position/coverage/frame-oracle and exact producer-buffer preservation checks remain active in this consumer. Fresh ordinary suite passes190 tests, zero failures,138 ignored, including pipeline creation (`target/roadmap-shared-body-material-regression.log`); fresh normal-loader RGBA is exact (`target/roadmap-shared-body-material-rgba.log`). Selection is29/39 present stages; live authored submission remains0. Native Skinning/View ownership, view/sampler/state parity and remaining families stay open. Opcode0x64 is unresolved; native capture/symbol availability remains pending, with its zero-gated current material effect distinguished from actual value resolution. Luna quota remains unavailable; root continues implementation. M1/M2 are incomplete.

- [x] Correct census static-input candidate policy for generated Float48. Old diagnostic assumed24-byte streams plus palette and falsely flagged16 roots. Match production48-byte source/UV streams, absent palette, normalized placement and exact14/15-row Body producer. Existing116-root CPU probe passes101.24s, retaining1,118 opaque/605 stage2; whole-preview rejection candidates16→0, partial native receivers remain93 (`target/runner-completion-float-audit-policy-{catalogue,summary}.log`). Audit-only cfg(test) change; shipping renderer already had those contracts.

- [x] Audit Frame+0x1E0 independently with Luna: TFX explicitly emits numeric vec4 into PS80A9BE22 row2 across100 draws; raw source reads x as analytic-cutoff/stochastic-coverage switch. Stochastic hash uses UV derivatives/mip scale, not a time seed. No local Marathon writer/default is established; Alkahest texture mapping and serialized unresolved zeros are not evidence. [Source contract](frame-numeric-source-contract.md). User explicitly authorizes analytic cutoff for static previews. Production input policy selects this branch for release Marathon model previews; generic runtime defaults remain unresolved and explicit frame/scoped values (including Unknown) win. Six contracts and focused live controls shipped October3,03:49; game's runtime writer remains unresolved.

- [x] Integrate authorized static-preview analytic coverage into the six independently verified source contracts and live renderer, preserving stochastic shader math. Typed TFX overrides, live cutoff influence and unaffected RGBA passed; shipped October3,03:49 and retained in04:21 delivery. This supersedes the stale unchecked23:35-build note. KASHA's additional affected source and exact compute dependency ship in the October4 B152BE pass above. Complete catalogue appearance remains separately unchecked.

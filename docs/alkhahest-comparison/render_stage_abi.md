# Marathon / Goliath Render-Stage ABI

## Status

Quicktag now treats Marathon's geometry render-stage table as a 25-stage ABI with 26 u16 part boundaries.

The package-backed census over the current installed Marathon build parsed the stage ABI successfully for **2313 / 2313 geometry resources**.

The strongest current conclusion is that Marathon retains the Destiny 2 Tiger stage ordering reconstructed by Alkahest, but inserts **one additional Marathon-only slot at raw stage 3**. From raw stage 4 onward, the Destiny 2 sequence therefore shifts by +1.

Raw stage 3 is intentionally left semantically unnamed. It was empty across the geometry-resource census, and no package evidence collected so far justifies inventing a purpose for it.

Implementation source of truth:

- `src/render/stage.rs`
- `src/render/adapter.rs`
- `src/geometry/mod.rs::geometry_render_stage_abi`

Reusable census:

- hidden CLI: `--probe-render-stage-abi <path>`
- schema: `quicktag.goliath-render-stage-abi.v1`

## Stage table

| Raw | Marathon interpretation | Evidence | Geometry census |
| ---: | --- | --- | --- |
| 0 | GenerateGbuffer | Strongly correlated | 1666 geometries / 7290 parts |
| 1 | Decals | Probable | empty in geometry-resource census |
| 2 | InvestmentDecals | Confirmed | 455 geometries / 975 parts |
| 3 | Marathon-specific unknown | Unknown | empty |
| 4 | ShadowGenerate | Confirmed | 1880 geometries / 6696 parts |
| 5 | LightingApply | Probable | empty |
| 6 | LightProbeApply | Probable | empty |
| 7 | DecalsAdditive | Probable | empty |
| 8 | Transparents | Strongly correlated | 682 geometries / 1142 parts |
| 9 | Distortion | Strongly correlated | 83 geometries / 134 parts |
| 10 | LightShaftOcclusion | Strongly correlated | 638 geometries / 1098 parts |
| 11 | SkinPrepass | Probable | empty |
| 12 | LensFlares | Probable | empty |
| 13 | DepthPrepass | Confirmed | 1839 geometries / 5638 parts |
| 14 | WaterReflection | Strongly correlated | 44 geometries / 44 parts |
| 15 | PostprocessTransparentStencil | Strongly correlated | 82 geometries / 82 parts |
| 16 | Impulse | Probable | empty |
| 17 | Reticle | Strongly correlated | 53 geometries / 66 parts |
| 18 | WaterRipples | Strongly correlated | 44 geometries / 44 parts |
| 19 | MaskSunLight | Probable | empty |
| 20 | Volumetrics | Probable | empty |
| 21 | Cubemaps | Probable | empty |
| 22 | PostprocessScreen | Probable | empty |
| 23 | WorldForces | Probable | empty |
| 24 | ComputeSkinning | Confirmed | 836 geometries / 5574 parts |

"Empty" means no highest-detail authored parts were observed for this stage in the scanned `CLASS_GEOMETRY_RESOURCE` population. It does **not** prove that the engine stage is globally unused; feature renderers or other resource classes may still submit it.

## ABI layout

For the geometry buffer-set record used by the scanned Marathon resources:

- record size: `0x80`
- stage boundary table: `0x30`
- boundary count: `26`
- boundary element type: `u16`
- stage-specific input-layout table: `0x64`
- layout count: `25`

Quicktag now decodes this once through `GeometryRenderStageAbi` instead of duplicating the offsets in multiple callers.

## Package evidence

### Stage 2 — InvestmentDecals

This stage contains technique `80A9A2E0`, which Quicktag independently identifies through the investment-decal material path.

Observed census:

- 455 geometries
- 975 highest-detail parts
- blend states dominated by authored decal states 26 / 27
- no stage-0 geometry overlap

This is a direct semantic anchor.

### Stage 4 — ShadowGenerate

Stage 4 has the characteristic authored shadow shape:

- 1880 geometries
- 6696 highest-detail parts
- approximately 73.6% of its index volume overlaps stage-0 geometry
- representative techniques are predominantly **vertex-only**
- representative techniques use distinct shadow vertex shaders rather than the visible-stage shaders

Examples:

- `80A447A1`: VS `80A4479B`, no PS
- `80A44C4F`: VS `80A455ED`, no PS
- `80A498AE`: VS `80A498A8`, no PS

This invalidates Quicktag's former "stage-4 shadow proxy" policy. Stage 4 is now treated as the actual authored ShadowGenerate range.

### Stage 13 — DepthPrepass

Stage 13 is another large high-overlap geometry pass, but with its own depth-prepass vertex shaders:

- 1839 geometries
- 5638 highest-detail parts
- approximately 72.6% stage-0 index overlap
- representative techniques are predominantly vertex-only

Examples:

- `80A447A7`: VS `80A447A2`, no PS
- `80A44C52`: VS `80A455F5`, no PS
- `80A498B4`: VS `80A498AF`, no PS

Together with the +1 sequence, this strongly anchors raw 13 as DepthPrepass.

### Stage 24 — ComputeSkinning

Stage 24 is the clearest late-stage anchor.

Representative techniques use Tiger bind mode 6 and are **compute-only**:

- `80A447BD`: CS `80A447B7`
- `80A44B6C`: CS `80A447B7`
- `80A44CE0`: CS `80A44CDC`

Observed census:

- 836 geometries
- 5574 highest-detail parts
- approximately 97.2% stage-0 index overlap

This directly identifies raw 24 as ComputeSkinning and strongly supports the single insertion at raw 3.

### Stages 14 and 18 — WaterReflection / WaterRipples

The stage-14 and stage-18 census populations have the same representative geometry tags in the same order, including:

- `80ADA7B6`
- `80ADA7BE`
- `80ADA7C6`
- `80ADA7CE`
- `80ADA7D6`

Their technique families commonly share the same VS while selecting different PS variants.

That paired geometry behavior plus the shifted Tiger ordering strongly correlates them with WaterReflection and WaterRipples respectively.

### Stage 17 — Reticle

Quicktag already had an asset-level regression demonstrating that stage 17 contains camera-facing helper-card geometry for weapon-mod assets, including technique `80A60AC2`.

That behavior is a much better fit for Tiger's Reticle stage than the old generic "forward special" label.

Quicktag still suppresses resource-free stage-17 helper cards from becoming visible white quads in the generic preview path, but the stage itself is now semantically identified as Reticle.

### Stages 8 / 9 / 10

The old adapter constants in this region were shifted incorrectly.

The package census plus Alkahest's Tiger order support:

- raw 8 = Transparents
- raw 9 = Distortion
- raw 10 = LightShaftOcclusion

Stage 8 is dominated by blend state 8, matching the premultiplied-alpha style state Tiger uses broadly for transparent submission. Stages 8 and 10 also frequently appear as paired geometry families with shared vertex shaders and different pixel shaders.

Raw stage 9 is a smaller specialized population dominated by blend state 10 (MAX blend), distinguishing it from ordinary transparent submission.

These are marked StronglyCorrelated rather than Confirmed because Quicktag is not yet executing the packaged shaders against the full Marathon extern-scope contract.

## Renderer changes made from this result

- Added typed `MarathonRenderStage`.
- Centralized the 25-stage count.
- Corrected raw adapter aliases:
  - DecalsAdditive: 7
  - Transparents: 8
  - Distortion: 9
  - LightShaftOcclusion: 10
- Corrected InvestmentDecal warning logic to expect raw stage 2.
- Removed the old stage-4 "shadow proxy" suppression heuristic.
- Split raw authored geometry-part collection from preview deduplication so ABI analysis cannot erase repeated shadow/depth/compute ranges.
- Added a reusable JSON census that records:
  - stage boundaries
  - input-layout IDs
  - geometry/part/index counts
  - stage-0 overlap
  - LOD categories
  - primitive types
  - render-state distributions
  - representative geometry tags
  - technique and shader-stage signatures

## Remaining uncertainty

The ABI is not fully "solved" yet.

Most importantly:

1. **Raw stage 3 remains unknown.**
2. Several stages are empty in the geometry-resource census, so their names currently derive from the strongly supported +1 Tiger sequence rather than direct Marathon geometry evidence.
3. Stages 8 / 9 / 10 are strongly correlated but should eventually be verified against full shader outputs / extern scopes.
4. The current renderer still deduplicates repeated source index ranges for visible preview geometry; Strict Tiger shadow/depth execution must consume the authored stage ranges directly rather than the preview-deduplicated representation.

The next highest-value task is therefore the TODO immediately following this one: drive shadow caster selection from the exact authored ShadowGenerate part range and preserve its stage-specific input ABI.

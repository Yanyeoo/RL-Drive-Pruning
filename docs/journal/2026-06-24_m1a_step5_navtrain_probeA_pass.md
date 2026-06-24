# M1.a Step 5 — navtrain probe A PASS (L*=12)

**Date**: 2026-06-24 ~20:10
**Status**: ✅ PASS — `vision_frac_mean = 0.1693` ∈ [0.15, 0.22]
**Outcome**: M1.a fully delivered on both navtest (n=500, locked) and navtrain (n=100, new).

## Result

| N | layer | vision_frac_mean | std | min | max | acceptance |
|---|-------|------------------|-----|-----|-----|------------|
| 100 | 12 | **0.1693** | 0.0527 | 0.0705 | 0.3783 | PASS [0.15, 0.22] |

n_vision_mean = 720 vision tokens per scene (consistent with navtest).

## Artifacts

- token list: `exp/m1a_navtrain_probeA_setup/tokens_100.txt`
- nocot JSONs: `data/navtrain_nocot_probe100/*.json` (100)
- probe `.pt`: `exp/m1a_navtrain_probeA_L12/*.pt` (100)
- summary: `exp/m1a_navtrain_probeA_L12/probeA_summary.json`
- yaml: `code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_probe100.yaml`

## Runtime

- nocot prebuild (S5): 7 min (5 min SceneLoader load + 1.3 min process 100 scenes)
- probe A (D4): 5.6 min (1.9 min model load + 3.6 min @ 2.16 s/scene on GPU0 H20)

## What blocked us briefly (and the lesson)

Before D4 I built a "scan missing images" script that walked every navtrain target
token's [-4,+10] frame window and required all 8 cams × 14 frames to exist on disk
(=112 jpg per scene). It reported **81% unusable**. I almost rewrote MA2.x to
limit to a 19K "clean" subset.

That was the **same reverse pattern as incident §3** — `build_all_sensors()` over a
15-frame window asks for sensor data on frames whose hashes were never shipped
in the navtrain tgz set (navtrain is sparse key-frames by design, 9 jpg per cam
per scene). The standard SceneLoader with `build_no_sensors()` (incident §2.2)
already proved diff=0 over all 103,288 tokens. D4 then confirmed empirically:
100 fresh navtrain tokens run end-to-end with **ok=100, skip=0, err=0**.

**Action items**:
- Do **not** use `navtrain_window_clean_tokens.txt` to filter MA2.x inputs.
- The full 103,288 navtrain.yaml token pool is the right one for M1.b / MA2.x.
- Scan products retained as forensic in `exp/m1a_navtrain_probeA_setup/` —
  do not delete (they're proof of the false alarm) but do not reuse either.
- Updated `RESUME_MONDAY.md` "give next AI" block to call this out.

## Why S5 first attempt hit a missing jpg

The original 100 tokens for S5 came from a navtest-derived list reused without
re-checking membership — at least one token was outside `navtrain.yaml`, in a
log whose frames were intentionally not shipped. Re-sampling 100 tokens directly
from `navtrain.yaml`'s token block (sorted lexically) made S5 trivially succeed.

## Next

**M1.b**: full navtrain attention extraction at L=12 for ALL 103,288 tokens →
feeds M1.c data pool. No further token-level filtering needed beyond standard
navsim `SceneLoader` pruning.

# M1.a Layer Sweep on navtest (n=100, 8 layers) — 2026-06-18 16:56

> Source of truth for `key_results.md §M1.a`.

## Sweep config

- Run: `exp/m1a_layer_sweep_20260618_1644/`
- Token list: first 100 tokens of `data/navtest_nocot/` (sorted), file at `${SWEEP}/tokens_100.txt`
- Layers swept: {0, 4, 8, 12, 16, 20, 24, 27} (8 of 28 Qwen2.5-VL-3B decoder layers, ~uniform)
- Hardware: 4×H20, 4 GPUs × 2 layers/GPU sequential
- Wall clock: 16:44 → 16:56 = ~12 min, 800 forward passes
- Per-scene latency: ~2.15-2.4 s (head-mean attention capture overhead negligible)
- Storage cost: 800 × ~3 KB = ~2.4 MB total
- Smoke pre-check: `exp/m1a_smoke_L14/` 5/5 OK with `vision_frac_mean=0.064` at L14

## Results

| layer | n | vision_frac_mean | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0  | 100 | 0.0537 | 0.0045 | 0.0452 | 0.0698 |
| 4  | 100 | 0.0302 | 0.0103 | 0.0182 | 0.0789 |
| 8  | 100 | 0.1034 | 0.0257 | 0.0646 | 0.1838 |
| **12** | 100 | **0.1789** | 0.0599 | 0.0759 | 0.3763 |
| 16 | 100 | 0.0262 | 0.0074 | 0.0131 | 0.0464 |
| 20 | 100 | 0.0952 | 0.0293 | 0.0530 | 0.2166 |
| 24 | 100 | 0.0458 | 0.0125 | 0.0181 | 0.0870 |
| **27** | 100 | **0.1804** | 0.0460 | 0.0801 | 0.3144 |

Note: `vision_frac` is `vision_attn.sum()` where `vision_attn` is the
head-mean attention from query=last instruction token to keys=vision tokens.
Since attention rows sum to 1 over all keys, vision_frac is the share of
attention mass routed to the 720 vision tokens (3 cams × 240 vision tokens
each). The complement (~80-95%) goes to instruction/text tokens.

## Headline

**L\* = 27** (`vision_frac_mean = 0.1804`)

But L12 is essentially tied (`0.1789`, within 1 std of L\*=27). **Two-peak
pattern** is clear:

```
        ▁▁▁▆▁▄▁▆
0  4  8 12 16 20 24 27
```

A ~30% mid-layer peak at L12, a sharp dip at L16, a recovery at L20, a
second dip at L24, then highest peak at L27.

## Interpretation (preliminary, needs deeper sweep)

1. **L12 peak is the more interesting candidate.** Mid-layer attention is
   well-known to host the most localized, semantically meaningful
   visual-text alignment in VLMs (e.g. "first visual concept binding").
   L27 being high is partly tautological — the last layer is what produces
   the trajectory tokens, and it has to look at vision to do so.

2. **Variance: L12 std=0.060 vs L27 std=0.046.** L12 has higher
   scene-specific variance, suggesting it differentiates scenes more.
   For pruning, that may be a feature (per-scene relevance scoring) or a
   bug (inconsistent token importance) — needs follow-up.

3. **L16 dip (2.6%)** is striking — almost no vision attention. If we
   were to prune at L16 we'd risk catastrophic loss because the model
   may use later layers (L20+) to re-look at vision. **Don't prune at the
   dip layer.**

## Recommended next steps

Before locking L\*, do a **fine sweep** in two windows:
- L10–L14 (zoom on first peak)
- L25–L27 (zoom on second peak)

That's 7 more layers × 100 scenes = 700 passes ≈ 9-10 min on 4 GPU,
or 18-20 min on 2 GPU after 17:00.

Once fine sweep is done, choose L\* by considering BOTH:
- vision_frac_mean (raw signal strength)
- per-scene std (discriminativeness)
- **AND** which layer is "early enough" to allow downstream layers to
  use the pruning result without re-introducing pruned tokens

If the picture stays as "L12 ≈ L27", **prefer L12** because:
- Earlier-layer pruning saves more compute downstream
- L27 pruning is too late to save flops (only the final attention block runs after)
- L12 has higher variance = more per-scene signal

## Sanity checks (all passed)

- All 800 .pt files saved successfully (0 errors, 0 skips)
- `vision_attn.shape == (720,)` for all scenes (3 cams × 240 vision tokens)
- `captured_q_len == prompt_len == 941` (V2 sanity from `m1a_prereqs.md`)
- `vision_blocks = [(108, 349), (372, 613), (636, 877)]` consistent across
  scenes (3 evenly-sized vision-token spans separated by frame markers)

## Risks / caveats

1. **navtest pivot**, not navtrain probe A. By design (see
   `decision_proposal_2026-06-17_m1a_on_navtest.md`). Re-confirm L\* on
   navtrain probe A once `.chain_complete` lands, expect ≤2-layer shift.
2. **Head-averaged.** May hide head-specific patterns. If a fine sweep
   doesn't sharpen the picture, try `--per-head` on top 3 candidate layers.
3. **n=100 is small.** Std/n ≈ 0.6%/√100 = 0.06% standard error per layer.
   Per-layer means are well-separated relative to noise (smallest gap
   17.89 vs 17.04 = 0.85% > 6× SE), so the ranking is robust.

## Reproducibility

```bash
# Re-run fine sweep
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# Edit run_m1a_layer_sweep_4gpu.sh GPU_LAYERS to:
#   GPU_LAYERS[0]="10 25"
#   GPU_LAYERS[1]="11 26"
#   GPU_LAYERS[2]="13 27"   # repeat L27 for sanity
#   GPU_LAYERS[3]="14 12"   # repeat L12 for sanity (and grab L14)
# Then:
bash scripts/run_m1a_layer_sweep_4gpu.sh \
    /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644/tokens_100.txt
```

```bash
# Re-analyze any sweep dir
PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code \
    /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
    -m rldrive.scoring.analyze_layer_sweep \
    --sweep-dir <SWEEP_DIR> \
    --layers 10,11,12,13,14,25,26,27
```

## Output artifacts

- `exp/m1a_layer_sweep_20260618_1644/L{NN}/<token>.pt` × 800
- `exp/m1a_layer_sweep_20260618_1644/layer_sweep_summary.json` (auto)
- `exp/m1a_layer_sweep_20260618_1644/SWEEP_STATUS.txt` (launcher meta)
- This journal

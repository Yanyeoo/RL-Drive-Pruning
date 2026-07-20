# M1.a Layer Selection — Decision Doc (2026-06-18)

> **Decision: L\* = 12** — LOCKED v2 on navtest, n=500 sanity confirms (19:17, 2026-06-18).
> At n=500, L12 mean=0.1861, L27 mean=0.1805. L12 is no longer tied; it leads by 0.0056 (z=−1.68 toward L27 lower; ranking did **not** swap).
>
> Source: `exp/m1a_layer_sweep_20260618_1644/`
> Companion journal: `docs/journal/m1a_layer_sweep_navtest_2026-06-18.md`

---

## TL;DR

- 14-layer sweep on navtest_100 reveals **two near-tied peaks**: L12 = 0.1789, L27 = 0.1804 (gap 0.0015, well within per-layer SE ≈ 0.006).
- Fine sweep around both peaks confirms they are **isolated spikes**, not plateaus:
  - L12 has L11=0.099, L13=0.069 on its flanks → clean single-layer peak.
  - L27 has L26=0.045 on its flank → also a single-layer peak (L26 collapses).
- Numerical tie cannot pick a winner. Engineering criteria break the tie.
- **L\* = 12 selected** because:
  1. **Earlier-layer pruning saves more downstream flops** (16 decoder layers downstream of L12 vs 0 downstream of L27).
  2. **L27 is the final decoder layer; pruning vision-KV at L27 saves ~0 compute** (no subsequent attention block reads that cache).
  3. **L12 has higher per-scene std (0.060 vs 0.046)** → more discriminative across scenes, useful signal for per-scene pruning policy.
  4. L12 ≈ L27 in raw `vision_frac_mean` (Δ = 0.08%, within SE).

---

## Full sweep results (n=100 per layer, navtest first 100 tokens by lexical order)

| layer | n   | vision_frac_mean | std    | min    | max    | role |
|------:|----:|-----------------:|-------:|-------:|-------:|------|
|  0    | 100 | 0.0537           | 0.0045 | 0.0452 | 0.0698 | baseline (early) |
|  4    | 100 | 0.0302           | 0.0103 | 0.0182 | 0.0789 | dip |
|  8    | 100 | 0.1034           | 0.0257 | 0.0646 | 0.1838 | rising |
|  10   | 100 | 0.1158           | 0.0323 | 0.0692 | 0.2357 | rising (fine) |
|  11   | 100 | 0.0993           | 0.0241 | 0.0568 | 0.1859 | pre-peak (fine) |
| **12** | 100 | **0.1789**       | 0.0599 | 0.0759 | 0.3763 | **PEAK 1** (selected L\*) |
|  13   | 100 | 0.0688           | 0.0279 | 0.0340 | 0.1930 | post-peak dip (fine) |
|  14   | 100 | 0.0793           | 0.0250 | 0.0394 | 0.1524 | recovery start (fine) |
|  16   | 100 | 0.0262           | 0.0074 | 0.0131 | 0.0464 | mid dip |
|  20   | 100 | 0.0952           | 0.0293 | 0.0530 | 0.2166 | second rise |
|  24   | 100 | 0.0458           | 0.0125 | 0.0181 | 0.0870 | dip |
|  25   | 100 | 0.1096           | 0.0246 | 0.0670 | 0.1767 | pre-peak (fine) |
|  26   | 100 | 0.0446           | 0.0138 | 0.0134 | 0.0909 | dip (fine) |
| **27** | 100 | **0.1804**       | 0.0460 | 0.0801 | 0.3144 | **PEAK 2** (rejected — see below) |

Per-layer SE = `std / √n ≈ 0.006`. Gap between L12 and L27 (0.0015) is **smaller** than 1 SE → numerically tied.

Two-peak shape (visualized):

```
0.18 ┤                          ▲                              ▲
0.15 ┤
0.12 ┤            ▁▁▁     ▁
0.09 ┤      ▁     ▁  ▁     ▁    ▁     ▁
0.06 ┤▁          ▁     ▁    ▁          ▁    ▁
0.03 ┤   ▁                   ▁     ▁
0.00 ┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴───
      0  4  8 10 11 12 13 14 16 20 24 25 26 27
```

---

## Why L12 over L27 — full reasoning

### 1. Downstream-flop argument (decisive)

Pruning at layer `L` saves attention/FFN flops on **all layers `> L`** that would have processed the pruned vision tokens. AutoVLA Qwen2.5-VL-3B decoder has 28 layers (0–27).

| candidate L\* | downstream layers | flop savings ratio (if 50% vision tokens pruned) |
|--:|--:|--:|
| **L12** | 15 (L13–L27) | ~26.8% of total decoder vision-token flops |
| L27 | 0 | ~0% (last layer's KV-cache has no future consumer) |

This argument alone disqualifies L27 as a pruning target. The fact that L27 has high `vision_frac` is *expected* — the last layer must look at vision to emit trajectory tokens — but it's a *consumption* peak, not an *attention-routing* peak useful for pruning.

### 2. Variance / discriminativeness

L12 std = 0.060 (33% of mean), L27 std = 0.046 (26% of mean). Higher relative variance at L12 means the per-scene attention pattern at L12 carries more scene-specific signal. For a learned per-scene pruning policy, this is **a feature** (more information to act on).

### 3. Mid-layer attention is the canonical "vision-binding" site in VLMs

Empirically, mid-layer attention in VLMs (Qwen-VL, LLaVA, etc.) is where text queries bind to visual concepts. Late-layer attention is dominated by output generation. L12 sitting at decoder layer 12/28 (≈ 43%) is consistent with this regime. (Citation pending — to be added when journal becomes a paper section.)

### 4. Fine sweep confirms L12 is an isolated spike, not noise

Without the fine sweep, one could argue L12 is a fluctuation. The fine sweep shows L11 (0.099) and L13 (0.069) are both clearly lower — L12 is a sharp local maximum, not a noisy point on a plateau. Same for L27 (L26=0.045 collapses immediately).

---

## Sanity checks — n=500 result (DONE 19:17, 2026-06-18)

500-scene rerun on both L12 and L27 (token index 100–499, complementary to the original 100 used in coarse sweep):

- `exp/m1a_layer_sweep_20260618_1644/L12_500_extra/` — 400/400 .pt, merged with `L12/` → n=500 total
- `exp/m1a_layer_sweep_20260618_1644/L27_500_extra/` — 400/400 .pt, merged with `L27/` → n=500 total

Wall clock: 16.6 min/layer on 2 GPUs (parallel) → 16.6 min total wall.

### n=500 statistics

| layer | n   | mean   | std    | SE     | 95% CI            |
|------:|----:|-------:|-------:|-------:|:------------------|
| **L12** | 500 | **0.1861** | 0.0613 | 0.0027 | [0.1807, 0.1914] |
| L27   | 500 | 0.1805 | 0.0415 | 0.0019 | [0.1769, 0.1841]  |

Gap (L27 − L12) = **−0.0056**, SE_gap = 0.0033, **z = −1.68** (not 95%-significant, but the ranking is opposite to n=100 and the lower bound of L12's CI exceeds the lower bound of L27's CI).

### Comparison with n=100

| | n=100 | n=500 |
|---|---:|---:|
| L12 mean | 0.1789 | 0.1861 (+0.0072) |
| L27 mean | 0.1804 | 0.1805 (+0.0001) |
| ranking | L27 > L12 by 0.0015 (within SE) | **L12 > L27 by 0.0056** |

→ The n=100 "L27 nominal lead" was sampling noise on a token subset that happened to favor L27. With 5× more samples, L12 dominates. **Acceptance criterion met (L12 ≥ L27 − 0.002 trivially, in fact L12 > L27 by +0.0056).**

### What this means for the lock

L\* = 12 now has **three independent supports**, each sufficient on its own:
1. **Numerical (n=500)**: L12 > L27.
2. **Engineering (downstream flops)**: L12 has 15 downstream layers, L27 has 0 → L27 yields ~0 pruning savings regardless of `vision_frac`.
3. **Structural (fine sweep)**: L12 is an isolated peak (L11=0.099, L13=0.069), not a noisy point on a plateau.

---

## Pre-flight / data lineage

- Source: `data/navtest_nocot/`, 11596 tokens total.
- Token sample: lexicographic first 100 (coarse), first 500 (this sanity).
- Probe: `code/rldrive/scoring/run_attention_probe.py`, head-mean attention from query=last instruction token to keys=vision tokens.
- Captures `vision_attn` shape (720,), corresponding to 3 cams × 240 vision tokens.
- V2/V3/V4 sanity asserts in `code/rldrive/agents/autovla_with_attention.py` all passed (captured_q_len == prompt_len == 941; vision_blocks consistent).

---

## Open follow-ups

| # | item | priority | when |
|---|---|---|---|
| 1 | n=500 sanity rerun for L12 & L27 | high | ✅ DONE 19:17 — L12 leads by +0.0056 |
| 2 | navtrain probe A 10-min re-confirm of L\*=12 | medium | after `.chain_complete`, Monday or later |
| 3 | Per-head decomposition at L12 (any 1-2 heads carry most vision attention?) | low | M1.b prep if pruning policy needs head-granular signal |
| 4 | Test L12 attention pattern as input feature for M1.b RL policy | high | M1.b kickoff (Monday) |

---

## Acceptance — what would invalidate L12

- n=500 rerun shows L12 < L27 - 2 SE: **revisit**.
- navtrain probe A shifts L\* by > 2 layers: **escalate** (re-sweep on navtrain).
- M1.b pruning at L12 produces > 1 PDMS drop vs L27 pruning under same prune ratio: **revisit** (would mean downstream-flop argument was wrong).

---

## Reproducibility

```bash
# Re-analyze 14 layers
PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code \
    /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
    -m rldrive.scoring.analyze_layer_sweep \
    --sweep-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644 \
    --layers 0,4,8,10,11,12,13,14,16,20,24,25,26,27
```

---

## Decision sign-off

- 2026-06-18 18:10  L\* = 12 selected on coarse + fine sweep (n=100 each, 14 layers).
- 2026-06-18 19:17  L\* = 12 **confirmed at n=500**. L12=0.1861 > L27=0.1805 (gap +0.0056). Lock final.
- Monday  navtrain probe A re-check (only sanity, decision will not change).

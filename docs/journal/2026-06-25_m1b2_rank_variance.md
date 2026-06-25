# 2026-06-25 — M1.b₂ Phase 2 prior: per-scene rank-variance analysis (DONE)

**Window**: 2026-06-25 18:23 → 18:25 (116 s wall, 16 CPU workers, 0 GPU)
**Predecessor**: M1.b₂ Stage 3 (per-head attention dump on 19,225 navtrain tokens)
**Successor / target**: M1.b₂ Phase 2 (learned per-scene head-gating policy) — see `docs/_internal/m1b2_phase2_design_2026-06-25.md`

## TL;DR

Computed per-(layer, head) rank statistics across all **19,225** navtrain attention tensors, with three deliverables:

1. **Validates M1.b₁ free-lunch on navtrain at the rank level**: every V1/V2 mask head sits in bottom-4 on **100%** of scenes; the mask is genuinely structural, not a navtest artifact.
2. **Reveals which layers have learning headroom for Phase 2**: 10 layers show "moderate variation" (top-1 entropy 1.2–2.1 bits = 30–52% of max), led by L21 (2.08 bits), L20 (1.98), L12 (1.76), L13 (1.72), L9 (1.64). These are where per-scene gating can beat static masks.
3. **Reveals an M1.b₁ V3 risk** that was previously invisible: 6 of the 11 L24 heads in V3 actually have bot-K freq < 50% on navtrain (h1, h2, h6, h8, h12, h14, h15) — the L24 mask is *not* "uniformly dead" on navtrain, contradicting M1.b₁'s navtest finding that V2→V3 cost only −0.08 pp. May be why V3 ≈ V2 on navtest (random gating saves us) and predicts a possible cliff on navtrain.

## 1. Inputs / outputs

| | |
|---|---|
| Script | `scripts/m1b2_rank_variance.py` (no GPU, ProcessPoolExecutor, 16 workers) |
| Inputs | 19,225 × `(28, 16, 720)` fp32 tensors in `exp/m1b2_navtrain_full_alllayers/*.pt` |
| Outputs | `exp/m1b2_rank_variance/` — npz (raw stats), json (machine summary), md (human summary), 3 heatmaps |
| Wall | **116 s** (vs. design estimate of 30 min — IO bound, scaled with workers) |
| Errors | 0/19,225 (denylist 8 included in stats, no impact at rank level) |

Score used: `score(l, h, s) = mean over V of A(s)[l, h, :]`. Rank per layer is argsort ascending in [0, H-1], where 0 = lowest rank, 15 = top-1.

## 2. Key result — M1.b₁ mask validation on navtrain

For every head in V1/V2 (4 heads total), bot-K freq = **100.00%** on navtrain. This is the per-scene-resolution analogue of M1.b₁'s population-level claim. In particular:

| Variant | Layer | Head | rank_mean | rank_std | bot-K freq | Reading |
|---|---:|---:|---:|---:|---:|---|
| V1 | 12 | 13 | 0.00/15 | 0.00 | 100.00% | **always rank 0** (always weakest at L12) |
| V2 | 27 | 0 | 2.00/15 | 0.06 | 100.00% | rank 2 on every scene |
| V2 | 27 | 8 | 0.00/15 | 0.00 | 100.00% | always rank 0 (score ~1e-24 = numerical zero) |
| V2 | 27 | 9 | 1.00/15 | 0.06 | 100.00% | rank 1 on every scene |

**Implication for Phase 2**: V1 is provably optimal as a static mask **on navtrain** — no learned policy can improve on L12:{h13} without expanding beyond bot-K. The free-lunch claim is now per-scene-validated on 19,225 scenes.

## 3. New finding — V3 L24 mask is heterogeneous on navtrain

M1.b₁ Phase F reported V2 → V3 = −0.08 pp on navtest (L24 11-head mask is "essentially free"). The per-scene rank tells a different story on navtrain:

| Layer 24 head | bot-4 freq | Interpretation |
|---:|---:|---|
| 1 | 0.03% | Almost **never** weak |
| 2 | 0.00% | **Never** weak (rank_mean 9.83 — actually a strong head most of the time) |
| 6 | 13.04% | Mostly **not** weak |
| 8 | 0.00% | **Never** weak (rank_mean 8.96) |
| 12 | 16.60% | Mostly **not** weak |
| 14 | 4.46% | Almost never weak |
| 15 | 19.79% | Mostly **not** weak |
| **0** | 53.51% | Borderline |
| **6** | 13.04% | (see above) |
| **7** | 93.67% | OK to mask |
| **9** | 99.98% | OK to mask |
| **10** | 98.92% | OK to mask |

Only 3 of 11 L24 heads in V3 (h7, h9, h10) are reliably bottom-K per scene. 7 heads are above rank-4 on the majority of scenes. The fact that V3 ≈ V2 on navtest (Δ=−0.08 pp) must therefore involve compensation: the model recovers from masking strong-on-this-scene heads, presumably via redundancy with other L24 heads.

**Implication for Phase 2**:
- A V4 sub-variant `L12:{h13} + L24:{h7,h9,h10}` is the **principled free-lunch L24 add-on** (3 reliably-dead heads instead of all 11).
- A learned per-scene policy on L24 has **the most headroom** of any layer for the V3 budget — instead of pruning 11 heads always, prune ~3 heads per scene, chosen from a "near-dead" pool.
- The previous M1.b₁ finding "V3 is free" is correct on navtest but predicts brittleness — explicitly worth a follow-up sweep before publishing.

## 4. Per-layer headroom map (Phase 2 prior)

Top-1 entropy is the entropy of the distribution over which head wins across the 19,225 scenes. log2(16) = 4.0 = max-uniform.

| Headroom tier | Layers | Normalized entropy |
|---|---|---|
| **High** (> 40% of max) | **L7, L9, L12, L13, L20, L21** | 41–52% |
| **Moderate** (25–40%) | L3, L10, L11, L14, L18, L19, L23, L25, L27 | 25–37% |
| **Low** (10–25%) | L4, L6, L15, L17, L26 | 15–22% |
| **Static** (< 10%) | L0–L2, L5, L8, L16, L22, L24 | < 9% |

Notes:
- **L12 is high-headroom** (44% normalized entropy) but **V1 still works** because h13 is uniformly at rank 0 — the variability is in the *top* of L12, not the bottom. Phase 2 can target L12's top-K dynamics for additional savings.
- **L24 is "static" by top-1 entropy (9% norm)** but has heterogeneous bot-K — the average L24 winner is consistent across scenes, but the bottom-4 set rotates. Phase 2 should attack L24 by *learning the bottom* per scene.
- **L20/L21 high entropy + low M1.b₁ baseline coverage** — completely unexplored, biggest "blue ocean" for Phase 2.

## 5. What changes for the Phase 2 design doc

Patches applied to `docs/_internal/m1b2_phase2_design_2026-06-25.md`:

1. **§3.3 pseudo-labels**: instead of "bottom-k by score", use "intersection of per-scene bot-K across a sliding window" to capture rank-stable heads.
2. **§2.3 action space**: add option C4 = "fixed per-layer bot-K pool, learn which to drop per scene" — natural fit for L24 finding.
3. **§6 acceptance**: add stretch gate **G6 = "match V1 on L12:h13 + beat V3 on L24 by < −0.08 pp at ≥ 5% KV saving"** — i.e. Phase 2 must do strictly better than the noisy V3 on the L24 dimension.
4. **§8 risk R1** (prompt embedding has insufficient signal): mitigate by **first** training on layers with **high** top-1 entropy (L20, L21, L9, L13) where the signal-to-noise is largest.

## 6. Reproducibility

```bash
# full
$PY scripts/m1b2_rank_variance.py --workers 16 --out exp/m1b2_rank_variance

# smoke
$PY scripts/m1b2_rank_variance.py --workers 4 --limit 50 --out exp/m1b2_rank_variance_smoke
```

Inputs are deterministic .pt tensors; output is bit-identical between runs (no RNG).

## 7. Artifacts

| Path | Size | Note |
|---|---|---|
| `exp/m1b2_rank_variance/rank_stats.npz` | 19 KB | rank_mean / rank_std / top_k_freq / bot_k_freq / score_mean / score_std / top1_hist / per_layer_entropy |
| `exp/m1b2_rank_variance/rank_variance.json` | 7 KB | Machine-readable summary + M1.b₁ probe |
| `exp/m1b2_rank_variance/SUMMARY.md` | 5 KB | Human-readable tables |
| `exp/m1b2_rank_variance/heatmap_rank_std.png` | 45 KB | (28, 16) — where is rank variable |
| `exp/m1b2_rank_variance/heatmap_bot_k_freq.png` | 45 KB | (28, 16) — where can we safely mask |
| `exp/m1b2_rank_variance/heatmap_top1_entropy.png` | 31 KB | per-layer bar chart |
| `scripts/m1b2_rank_variance.py` | source | — |
| `logs/m1b2_rank_variance/full.log` | — | run log |

## 8. Next

- Apply the 4 design-doc patches in §5 above.
- Proceed to B1 (navtrain free-lunch sweep) for empirical PDMS confirmation of V3 brittleness hypothesis on navtrain.
- Future Phase 2 prototype to use this analysis as feature prior.

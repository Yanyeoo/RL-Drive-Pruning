# Released scorer checkpoints

This directory ships the **token scorer + budget head** weights trained in this
project. Each checkpoint is a ~6.6MB MLP (0.6M params), small enough to live in
git so that `git clone` gives you directly usable models.

**Not redistributed here**: the frozen VLA backbones (AutoVLA `AutoVLA_PDMS_89.ckpt`
16GB, Qwen2.5-VL-3B/7B, ImpromptuVLA-7B, ~140GB total). Those are third-party
pretrained models — download them from their official sources.

## What is in a checkpoint

| file | content |
|---|---|
| `checkpoint.pt` | `TokenScorerWithBudget` state dict (`token_net` + `budget_net` + optional `value_net`) |
| `feature_norm.pt` | per-dim mean/std used to normalize vision features |
| `config.json` | `emb_dim` / `n_cam` / `hidden` / `min_keep_ratio` / `max_keep_ratio` / `use_value` |
| `manifest.json` | full training recipe (reward terms, lr, selection surrogate, scene list, ...) |
| `budget_params.pt` | learned `budget_log_std` of the Gaussian budget policy |

## Results (navtest PDMS)

Reference points: no-prune (`attn_L12` r=1.0) = **0.898845**; learned SFT scorer
= 0.89199; SFT r=0.75 = 0.898353. `full` = 4 shards, N≈11576. `gate` = shards 0+1.

| checkpoint | method | gate | full 4-shard | vs no-prune |
|---|---|---|---|---|
| `sft_scorer_base` | supervised scorer, fixed keep ratio (RL init) | — | 0.89199 | -0.006855 |
| `v7_st_topk` | straight-through Top-K surrogate + delta reward | — | 0.894798 | -0.004047 |
| `v10_value_only` | + value baseline only (max_kr 0.70) | 0.894235 | — | — |
| `v10_floor_init` | + efficiency floor + kr init 0.65 (no value head) | 0.899157 | — | — |
| `v10_full` | value baseline + floor + init 0.65 | 0.897519 | — | — |
| `v10_full_hi` | value baseline + floor + init 0.70 + max_kr 0.85 | 0.899729 | 0.898274 | -0.000571 |
| `v11_hard50` | `v10_full_hi` retrained on 50% hard + 50% normal | 0.899520 | 0.898360 | -0.000485 |
| **`v11_hard50_kl`** | same mix, `kl_beta` 0.02 (tighter token_net KL) | 0.898909 | **0.898719** | **-0.000126** |
| `v11_hard50_lr` | same mix, `budget_lr` 2e-4 | 0.898425 | — | — |
| `v11_hard75` | 75% hard scenes (over-weighted, worse generalization) | 0.897437 | — | — |

**Best checkpoint: `v11_hard50_kl` at 0.898719**, i.e. **-0.000126** from the
no-prune upper bound — statistically level with it, but *not* above it. The gap
narrowed across cycles (-0.000571 → -0.000485 → -0.000126) yet was never closed.

Two things worth knowing before you trust the gate column:

1. **Gate ranking inverted against full.** `v11_hard50` had the best 2-shard gate
   score (0.899520, above no-prune) but lost to `v11_hard50_kl` on the full
   4-shard set. Always conclude on full.
2. **`v11_hard50_kl` wins by losing less, not by rescuing more.** Per-scene
   attribution over the aligned 11572 scenes: catastrophic scenes are 82
   (`hard50`) vs 83 (`hard50_kl`) — essentially identical — while the count of
   scenes that *beat* no-prune is 550 vs 606 (baseline `v10_full_hi`: 639).
   Raising `keep_ratio` globally buys safety on dangerous scenes at the cost of
   the pruning gains everywhere else; tightening the KL on `token_net` preserves
   the supervised token ranking and gives most of that back. Note `hard50_kl`'s
   keep_ratio barely moved (0.691 → 0.686) yet it scored best — training reward
   and keep_ratio are *not* reliable model-selection signals here.

See `STATUS.md` for the full attribution and the resulting v12 direction.

## Efficiency

`v11_hard50_kl` settles at `keep_ratio ≈ 0.69`, `v11_hard50` at `≈ 0.75` and
`v10_full_hi` at `≈ 0.72` (keeping ~500-540 of 720 vision tokens). At r=0.72 the
theoretical saving is **-19.0% total FLOPs / -22.3% LLM prefill**
(`scripts/compute_flops_table.py`; the ViT cost of 949.6G is constant and only
LLM prefill shrinks).
Measured wall-clock latency is **not yet verified** — `scripts/profile_wallclock.py`
still fails at trajectory decoding; treat the FLOPs number as theoretical.

## Usage

```python
from rldrive.scoring.token_scorer_budget import BudgetScorerRunner

# v11_hard50_kl is the best checkpoint (full 4-shard 0.898719)
runner = BudgetScorerRunner("release_ckpt/v11_hard50_kl", device="cuda:0")
token_scores, keep_ratio = runner.score_budget(
    vision_feat, vision_token_positions, vision_blocks
)
# keep_ratio is the per-scene budget chosen by the learned head (policy mean);
# keep the top-B tokens where B = round(keep_ratio * n_vision_tokens).
```

For evaluation through the navsim harness, point the agent at a checkpoint dir:

```bash
EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" \
  bash scripts/eval_v7_folds_4gpu.sh <CYCLE_ID> full <arm>
# internally: +agent.selector=scorer_budget +agent.scorer_ckpt=<ckpt dir>
```

`EVAL_WORKERS=1` is required — multi-worker model loading hits a meta-tensor
race on H20 and silently drops scenes.

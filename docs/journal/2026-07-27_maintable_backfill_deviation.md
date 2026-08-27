# Main-table backfill deviation record — 2026-07-27

## Observed facts (verified from CSV row counts, not from memory)

Per-family scene coverage under `results/raw/tokenprune_S3_full/` (full navtest = 11,576/11,580 rows across 4 shards):

- `varBsafe_scorer_r075`: only sh0 present (2950 rows). Missing sh1/sh2/sh3.
- `rl_shaped_r025`: only sh0 present (2950 rows). Missing sh1/sh2/sh3.
- `rl_shaped_r075`: only sh0 present (2950 rows). Missing sh1/sh2/sh3.
- `rl_taucut_kr060`: only sh0 present (2950 rows). Missing sh1/sh2/sh3.

All other main-table families already have full 4-shard coverage.

## Original sh0 configuration (source of truth for comparability)

Located exact original launch configs (do not deviate when backfilling sh1/2/3):

- `varBsafe_scorer_r075` — `scripts/run_sparsevlm_r075_gpu4.sh:74-101`:
  selector=scorer, keep_ratio=0.75, scorer_ckpt=`ckpt/s3_token_scorer`,
  prune_variant=drop, no denylist, no safety_net.
- `rl_shaped_r025` / `rl_shaped_r075` — `scripts/run_rl_eval_4gpu.sh:111-141`:
  selector=scorer, keep_ratio=0.25/0.75,
  scorer_ckpt=`ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh0/ckpt_best`,
  prune_variant=drop.
- `rl_taucut_kr060` — `scripts/run_rl_eval_4gpu.sh:143-167`:
  selector=scorer_taucut, keep_ratio=0.5, tau=-0.1668, tau_min_keep=36,
  same rl_shaped ckpt_best, prune_variant=drop.

The rl_shaped ckpt is pinned to the absolute path above (not `ls -dt`) so
sh1/2/3 use the exact checkpoint that produced sh0.

## Decision and rationale

During the Safe-HTPO main-line evaluation phase (GPU 0-3 occupied), the idle
GPU 4-7 will run an unattended backfill of the four incomplete families'
sh1/sh2/sh3, then aggregate each to full navtest. This uses otherwise-idle
compute to complete pre-existing main-table comparison rows. All backfill runs
use Variant-B true token drop, raw scores, and no denylist / no fallback,
matching each family's original sh0 exactly. This does not alter the
Safe-HTPO research protocol and does not touch the running training.

The backfill is deliberately Variant-B raw/no-denylist so its numbers are
compatible with the AAAI no-denylist protocol; the `+fallback` column in the
old main table is NOT used or reproduced here.

## Changed / added files

- Added: `scripts/backfill_maintable_gpu4_7.sh` (new unattended backfill runner).
- Added: `scripts/backfill_aggregate.py` (per-family full-navtest aggregation, raw only).
- Not modified: `results/main_table.md`, `run_rl_eval_4gpu.sh`,
  `run_sparsevlm_r075_gpu4.sh` (backed up under
  `backups/20260727_231053_maintable_backfill/` regardless).

## Reverse instruction

To revert: delete `scripts/backfill_maintable_gpu4_7.sh`,
`scripts/backfill_aggregate.py`, and any newly produced
`results/raw/tokenprune_S3_full/MT_{varBsafe_scorer_r075,rl_shaped_r025,rl_shaped_r075,rl_taucut_kr060}_sh{1,2,3}.csv`
plus `results/raw/maintable_backfill/`. The original sh0 CSVs and scripts are
preserved in `backups/20260727_231053_maintable_backfill/`; restore from there
if any original file is disturbed.

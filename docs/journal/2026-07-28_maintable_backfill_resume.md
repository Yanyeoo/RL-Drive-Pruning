# Main-table backfill RESUME — 2026-07-28 21:22 (+0800)

## Observed facts (verified, not from memory)

- Safe-HTPO night cycle finished `DONE` at 05:39:46; GPU pool shrank from 8→4
  (`nvidia-smi` now shows only indices 0-3, all 0 MiB, 0% util at 21:17).
- No live `run_pdm_score_cot.py` / backfill process (`pgrep` clean).
- Previous backfill runner (pid 37131) is DEAD; its `status.json` was frozen at
  `RUNNING ... on GPU 4 5 6 7`. GPU 4-7 were recycled ~09:01, which killed the
  four then-in-flight jobs mid-run (no CSV written).
- CSV coverage under `results/raw/tokenprune_S3_full/` (verified via `wc -l`):
  - `varBsafe_scorer_r075`: sh0/1/2/3 present → COMPLETE.
  - `rl_shaped_r025`: sh0/1/2/3 present → COMPLETE.
  - `rl_shaped_r075`: sh0/1/2 present, **sh3 MISSING**.
  - `rl_taucut_kr060`: sh0 only, **sh1/sh2/sh3 MISSING**.
  - Shard sizes are consistent: sh0=2950, sh1=2797, sh2=2964, sh3=2869 (≈11580).
- Stale killed-job exp dirs exist under `exp/MT_{rl_shaped_r075_sh3,
  rl_taucut_kr060_sh{1,2,3}}/` but contain **no CSV** → safe to re-run; the runner
  copies only a freshly-produced CSV (`ls -t .../$exp/*/*.csv | head -1`).
- Pinned checkpoints exist: `ckpt/s3_token_scorer/checkpoint.pt` and
  `ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh0/ckpt_best/checkpoint.pt`.
- `STOP_DRIVER` (empty, dated 2026-07-03) is unrelated; the backfill runner only
  honors `STOP_BACKFILL`, which is absent.

## Decision and rationale

Resume the SAME idempotent runner `scripts/backfill_maintable_gpu4_7.sh` to
complete the 4 remaining shards, with two env overrides:

- `BACKFILL_GPUS="0 1 2 3"` — GPU 4-7 no longer exist; the current idle pool is 0-3.
- `BACKFILL_WAIT_FOR_EVAL=0` — the Safe-HTPO main line is already `DONE`; there is
  no eval stage to wait for.

Everything else is byte-identical to the original launch config recorded in
`docs/journal/2026-07-27_maintable_backfill_deviation.md`: Variant-B true token
drop, raw scores, NO denylist / NO fallback, pinned scorer checkpoints, same
shard filters. This preserves comparability with the already-landed sh0-2 rows
and the AAAI no-denylist protocol.

Launched 21:22:12. Runner correctly SKIP-ped the 8 existing shards and started
exactly 4 jobs, one per GPU:
- GPU0 `MT_rl_shaped_r075_sh3`
- GPU1 `MT_rl_taucut_kr060_sh1`
- GPU2 `MT_rl_taucut_kr060_sh2`
- GPU3 `MT_rl_taucut_kr060_sh3`

Per-shard wall ≈ 3.3h (observed from the earlier cycle). With 4 shards on 4 GPUs
in parallel, ETA ≈ 00:45, well inside the tomorrow-14:00 recycle window.

## Changed / added files

- None modified. Only new shard CSVs will be produced:
  `results/raw/tokenprune_S3_full/MT_{rl_shaped_r075_sh3,rl_taucut_kr060_sh{1,2,3}}.csv`,
  then `results/raw/maintable_backfill/MT_<family>_full.csv` + `backfill_summary.json`.
- Original sh0 CSVs and scripts remain preserved under
  `backups/20260727_231053_maintable_backfill/` (from the prior deviation record).

## Reverse instruction

To revert this resume: `touch STOP_BACKFILL` (graceful) or kill the runner
process group, then delete the four newly produced CSVs listed above and
`results/raw/maintable_backfill/`. No original artifact is touched, so nothing
else needs restoring.

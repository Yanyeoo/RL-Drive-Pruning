# Overnight Table-1 orchestrator — decision & deviation record (2026-07-28 21:52)

## Observed facts (verified)
- GPU pool = 4× H20 (idx 0-3); reclaim tomorrow 14:00.
- RL backfill (pid 3615) running on GPU0-3: rl_shaped_r075 sh3 + rl_taucut_kr060 sh1/2/3.
- Table-1 gap (from `gen_table1_draft.py` on landed drop CSVs): baselines FastV(all 3
  ratios, existing CSVs are MASK not drop), Random r=0.25/0.75, PruMerge(all 3),
  SparseVLM r=0.25/0.5, and ours SFT r=0.25 are missing; all other cells present.
- 7B assets present; the earlier `image_processor` loader blocker is resolved
  (finetune `preprocessor_config.json`, 7/23, now has `image_processor_type`).

## Decision (per user 07-28 rulings)
- All-drop Table 1; PruMerge re-included; baselines report raw, ours report +fallback.
- Fill gaps shard0-first (fast draft), then upgrade to full 4-shard as time allows.
- 7B is best-effort after the main table: attempt, on failure report and continue,
  never change the research protocol.

## Added files (no existing file modified)
- `scripts/gen_table1_draft.py` — Table-1 generator (baseline raw, ours +fallback,
  coverage-tagged, atomic write).
- `scripts/run_maintable_overnight.sh` — unattended orchestrator; phases
  A(wait for RL backfill) → B(baseline drop shard0 + SFT-ours r=0.25) →
  C(best-effort 7B) → B2(upgrade to full shards); hard stop 2026-07-29 13:45;
  idempotent (skip existing CSV); single-instance lock; STOP_MAINTABLE honored;
  regenerates `results/table1_draft.md` after every job.
- `scripts/run_7b_besteffort.sh` — smoke → 4-GPU parallel ratios → step2 metrics;
  non-blocking; writes `results/impromptu7b/BESTEFFORT_REPORT.md`.

Launched: orchestrator pid 13472 at 21:51, currently WAIT_A.

## New CSV/artifact outputs (all additive)
- `results/raw/tokenprune_S3_full/MT_{fastv_l2,random,prumerge_cls,sparsevlm_text}_drop_r*_sh*.csv`
- `results/raw/tokenprune_S3_full/MT_sft_varB_drop_r025_sh*.csv`
- `results/table1_draft.md`, `results/impromptu7b/pred_*.jsonl` + `eval_*.json`.

## Reverse instruction
- Graceful stop: `touch STOP_MAINTABLE` (workers exit at next job boundary).
- Hard stop: kill orchestrator pid (13472) + its child `run_pdm_score_cot.py`.
- Full revert: delete the three added scripts and all `*_drop_*` / `MT_sft_varB_drop_*`
  CSVs listed above, plus `results/table1_draft.md` and `results/impromptu7b/pred_*`,
  `eval_*`, `BESTEFFORT_REPORT.md`. No pre-existing artifact is touched.

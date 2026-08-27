# Overnight main-table window — final report (2026-07-29 12:05)

## Window
- 2026-07-28 21:22 → 2026-07-29 11:04 (orchestrator DONE); hard deadline 13:45.
- 4× H20 (idx 0-3). Operator: stateless execution agent.
- Orchestrator: `scripts/run_maintable_overnight.sh` (pid 15133). Single dispatcher
  after 22:10 (old orchestrator 13472 + backfill dispatcher were retired).

## Completed (verified from CSV row counts / logs)
- **RL Scorer r=0.75 -> full navtest**: `MT_rl_shaped_r075_sh0-3`, +fb 0.8944 (N=11576).
- **Baseline drop shard0 (N=2949)** for all four methods × r=0.75/0.5/0.25:
  - FastV 0.8736 / 0.8191 / 0.6783
  - Random 0.8731 / (full 0.8635) / 0.7670
  - PruMerge 0.8511 / 0.7887 / 0.6511
  - SparseVLM (full 0.8991) / 0.8774 / 0.8293
- Candidate table `docs/results/2026-07-29_physical_drop_main_table.md`, live draft
  `results/table1_draft.md`, key_results §14 all written.

## Headline (verified)
At r=0.5, **SFT (ours) +fallback = 0.9045 dominates every drop baseline** (SparseVLM
0.877, FastV 0.819, PruMerge 0.789) and ≈ no-prune 0.89879 (100.6%). Advantage widens
at r=0.25 (ours RL 0.826 vs PruMerge 0.651 / FastV 0.678).

## Not completed / failures (honest)
1. **Phase B2 (baseline full-shard upgrade) ran ZERO jobs.** Bug: `FULLSH_NEED=13000s`
   headroom gate; at 11:04 only 9618s remained, so every full-shard job was skipped and
   B2 exited immediately — even though 4 idle GPUs + 2.7h were available. Baselines are
   therefore **shard0-only** (except random r0.5 / sparsevlm r0.75 which were full before).
   Fix for next window: lower `FULLSH_NEED` and/or size per-job timeout to remaining time.
2. **7B ImpromptuVLA / nuScenes = INVALID.** Smoke was 0 ok / 8 err (grep gate miscounted
   "0\n0" as pass), full runs produced 6019 empty predictions. Root cause: image library
   is packed in `data/nuscenes_impromptu_val/raw/*.tar` (2.2G, 6 shards) as
   `<sceneid>.CAM_*.png`, but QA json refs are nuScenes-native
   `nuscenes/samples/CAM_FRONT/n015-...jpg`; no unpack + no name mapping + no image root
   in the eval script (TODO left at line 186). Not rescued (unverified mapping work,
   unsafe within remaining time; not a headline). See `results/impromptu7b/BESTEFFORT_REPORT.md`.
3. **SFT r=0.25** intentionally not run (user: ours fixed-ratio non-headline).
4. rl_taucut sh1/2/3 stopped earlier per user; RL+τ-cut remains shard0 only.

## `0.9107` statement
`0.9107` was neither generated nor used anywhere this window.

## Artifacts
- CSVs: `results/raw/tokenprune_S3_full/MT_*_drop_r*_sh0.csv`, `MT_rl_shaped_r075_sh0-3`.
- Tables: `results/table1_draft.md`, `docs/results/2026-07-29_physical_drop_main_table.md`,
  `docs/results/key_results.md` §14.
- Logs: `logs/maintable_overnight/`. Scripts backed up in
  `backups/20260728_220446_taucut_stop/`.

## Recommended next steps
1. Upgrade baseline r=0.5 (then 0.75/0.25) shard0 -> full 4-shard: rerun the four
   `MT_*_drop_r*` families for sh1/2/3 (each ~3.3h/shard, 4-GPU parallel). Fix the B2 gate first.
2. If 7B needed: unpack the 6 tars, build `samples/<CAM>/<file>.jpg -> <sceneid>.<CAM>.png`
   map, add `--image-root`+mapping to `run_impromptu7b_nuscenes_eval.py`, re-smoke (8 scenes)
   and require non-empty preds before full runs.
3. Paper table remains untouched pending review of the candidate table.

## Reverse
- New baseline CSVs are additive under `tokenprune_S3_full/` with `_drop_*` names;
  delete `MT_{fastv_l2,random,prumerge_cls,sparsevlm_text}_drop_r*_sh0.csv` to revert.
- No existing artifact or paper file was modified.

# Safe-HTPO Night Cycle Report — 2026-07-28

## Scope and run identity

- Final run ID: `safehtpo_20260727_224834`
- Training: synchronized 8-rank Safe-HTPO-Lite, one epoch, 19,200 scenes total (2,400/rank).
- Evaluation: full navtest, Variant-B physical token drop, no denylist fallback, four independent shards.
- Training action: random logistic-normal scene budget in `[0.2, 0.9]` plus the Top-K token-selection surrogate.
- Evaluation action: deterministic budget-head policy mean. `BudgetScorerRunner.score_budget()` calls the budget head directly under `torch.no_grad()` and does not sample.
- Same-scene unpruned VLA rollouts were used for the training reward baseline; no external navtest baseline-score JSON was used.

## Prior failed launch

`safehtpo_20260727_221408` failed during its DDP smoke at `2026-07-27 22:23:07 +08:00`: every rank raised `RuntimeError: Distributed Safe-HTPO requires CUDA/NCCL`. No training checkpoint, DDP audit, or result CSV was produced. This run is invalid and was not used for results.

## Final run integrity

- DDP smoke passed before final training: a smoke checkpoint, joint-policy records, and an 8-rank `ddp_sync_audit.json` were produced.
- Final DDP audit has identical two-value parameter fingerprints across all eight ranks.
- No training/evaluation traceback, OOM, NaN/Inf, or NCCL failure was found. The only NCCL message was the non-fatal deprecation warning for `NCCL_ASYNC_ERROR_HANDLING`.
- Final checkpoint: `ckpt/safehtpo_20260727_224834/checkpoint.pt` (written `02:03:57 +08:00`).
- Training completion marker: `ckpt/safehtpo_20260727_224834/TRAIN_DONE` (written `02:04:01 +08:00`).
- Evaluation completion marker: `results/raw/safe_htpo_safehtpo_20260727_224834/EVAL_DONE` (written `05:39:46 +08:00`).
- Final status: `DONE` at `2026-07-28 05:39:46 +08:00`.

## Training observations

Configured objective parameters:

- `efficiency_beta=0.15`, `safety_beta=0.50`, `selection_pg_weight=1.0`, token `lr=3e-5`, budget `lr=1e-4`.
- Final epoch average reward: `0.5870`; best reward selected by training: `0.6074`.
- At final step 150: reward `0.57246`, training keep ratio `0.35770` (local std `0.10525`), safety loss `0.03125`, budget/selection log-prob `-0.52770/-17.13688`, gradient norm `4.30730`.

The logged train keep ratio decreased from roughly `0.546` at step 1 to roughly `0.358` at step 150 while remaining stochastic and inside the configured bounds. This is an observation, not evidence of convergence or an optimal policy.

## Full-navtest result

Coverage validation passed on four output CSVs: `11,576` unique scenario tokens, matching the expected full navtest coverage.

| Metric | Value |
|---|---:|
| Raw PDMS | `0.84133` |
| 95% bootstrap CI | `[0.83623, 0.84625]` |
| PDMS P5 / median | `0.00000` / `0.93968` |
| No-at-fault collisions | `0.97551` |
| Drivable-area compliance | `0.92389` |
| Ego progress | `0.77850` |
| TTC within bound | `0.94238` |
| Comfort | `0.99810` |
| Driving-direction compliance | `0.95866` |
| Evaluation keep-ratio mean / std | `0.34813` / `0.07923` |
| Evaluation keep-ratio P10 / P50 / P90 | `0.24550` / `0.34900` / `0.45250` |
| Mean retained visual tokens | `250.7 / 720` |

The matching existing full-navtest SFT fixed-ratio reference is `r=0.35`, raw PDMS `0.85515`; the Safe-HTPO dynamic result is `0.01382` lower. It is also below the unpruned baseline (`0.89879`) by `0.05746`. Therefore this run must not be presented as an improvement over either baseline.

## Metadata deviation

The evaluation shell script and generated `protocol.json` say `random-budget` / `samples its own budget head ratio`. This wording is inaccurate for evaluation. The executed scorer path uses the deterministic budget-head output, as described above. The result is nevertheless dynamic across scenes: the deterministic policy mean varies with each scene feature. No source files were changed during this cycle; the wording should be corrected in a separate backed-up, journaled metadata-only patch before any paper-facing use.

## Evidence locations

- Configuration and final status: `ckpt/safehtpo_20260727_224834/run_config.json`, `logs/safehtpo_20260727_224834/status.json`
- Training records: `logs/safehtpo_20260727_224834/train.log`, `ckpt/safehtpo_20260727_224834/train_log.jsonl`
- DDP audit: `ckpt/safehtpo_20260727_224834/ddp_sync_audit.json`
- Evaluation launcher and shard logs: `logs/safehtpo_20260727_224834/eval_orchestrator.log`, `logs/safe_htpo_safehtpo_20260727_224834/eval/`
- Aggregates and raw shard CSVs: `results/raw/safe_htpo_safehtpo_20260727_224834/`

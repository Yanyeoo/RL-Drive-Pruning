# Handoff: Safe-HTPO AAAI Unattended Run

**Created:** 2026-07-27 21:xx +0800
**Owner:** Stateless execution agent
**Run window:** after 22:00 tonight through 18:00 tomorrow

## Mission

Run and monitor the AAAI Safe-HTPO-Lite experiment without fabricating a result or silently changing the research protocol. The goal is a valid, no-denylist, full-navtest dynamic-policy result or an evidence-rich failure report that identifies the blocking cause.

## Mandatory reading order

1. `docs/journal/2026-07-27_safe_htpo_preflight.md`
2. `refine-logs/SAFE_HTPO_22H_RUNBOOK.md`
3. `refine-logs/2026-07-27_safe_htpo_experiment_plan.md` (especially M0--M3)
4. `scripts/run_safe_htpo_aaai_unattended.sh`
5. `scripts/train_scorer_budget_rl.py`
6. `scripts/eval_safe_htpo_dynamic_4gpu.sh`
7. `paper/aaai2027/main.tex` Method section
8. Historical results only for context: `results/main_table.md` and `docs/journal/2026-07-22_window_end_summary.md`

## Facts that must not be overwritten by memory or assumption

- There is no currently live Safe-HTPO, BudgetRL, torchrun, or NAVSIM evaluation process as of the handoff preflight. Recheck before every launch.
- Historical dynamic BudgetRL is **not** evidence for the new method: raw PDMS 0.84865, no-denylist comparison below no-prune. Do not reuse it as the new result.
- The historical `results/baseline_sub_scores.json` contains navtest keys only: overlap with navtrain is 0/19,225. It must not be used as navtrain delta/safety supervision.
- The code has been corrected: the first full-token feature-capture pass now retains the same scene's unpruned trajectory and computes its PDM submetrics at runtime. The pruned rollout is compared against that exact baseline.
- Training ratio is stochastic and scene-specific in `[0.2, 0.9]`. Never force `k_s=0.5`. Fixed ratios are baselines only.
- Dynamic full evaluation uses the policy mean per scene, Variant-B physical token drop, and no denylist.
- The current implementation is AAAI Safe-HTPO-Lite: Gaussian budget policy + deterministic Top-K execution with a multi-label selection-policy surrogate. Do not claim exact Plackett--Luce sampling, PPO, GSPO, CVaR constraints, G-OPD, or ExOPD.

## Required operating rules

1. Before every decision, verify facts from code, artifact, or log. Never rely on an old journal as current state.
2. Before starting any process, inspect `ps` and relevant logs for existing `torch.distributed`, `train_scorer_budget_rl.py`, `run_safe_htpo`, and `run_pdm_score_cot.py` processes. Do not double launch.
3. Before editing a critical artifact, make `cp -a` backup under `backups/<timestamp>_*`; never overwrite the original without a backup.
4. Any deviation from this handoff, runbook, or plan must be written immediately to a new dated journal entry. Include: observed fact, decision, rationale, exact changed files/parameters, and a reverse instruction.
5. Maintain the active session todo list. Mark each completed item immediately.
6. Do not use a manual denylist, post-hoc fallback score, or unsupported result in the paper.
7. Do not update Introduction/Experiments/Conclusion claims until the final summary confirms full 11,576-scene raw Variant-B coverage.

## Launch protocol

At or after 22:00, only if no active run exists:

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
mkdir -p logs/safe_htpo_launcher
nohup bash scripts/launch_safe_htpo_after_22h.sh \
  > logs/safe_htpo_launcher/nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The launcher waits for time and eight idle GPUs where `nvidia-smi` is available, then executes:

1. 8-GPU DDP smoke;
2. 1-epoch synchronized Safe-HTPO-Lite training;
3. 4-shard full-navtest dynamic Variant-B evaluation;
4. aggregation and `DONE`/`FAILED` state write.

Do not manually start `run_safe_htpo_aaai_unattended.sh` if `launch_safe_htpo_after_22h.sh` is active.

## Smoke acceptance gate

Before formal training, require all of:

- `ckpt/<run-id>/smoke/checkpoint.pt`;
- `ckpt/<run-id>/smoke/ddp_sync_audit.json`;
- `ckpt/<run-id>/smoke/train_log.jsonl` contains `selection_log_prob_mean`;
- no CUDA OOM, NCCL error, NaN, or DDP desynchronization in smoke log;
- logged ratios are within `[0.2, 0.9]` and have nonzero standard deviation.

If smoke fails, do not continue to formal training. Diagnose the log, make only an evidence-based repair with backup and journal, then repeat smoke.

## Monitoring schedule

- Immediately after launch: verify launcher PID, selected run ID, and `status.json`.
- During smoke: inspect `logs/<run-id>/smoke.log` and smoke artifacts.
- During training: inspect `logs/<run-id>/train.log`, `ckpt/<run-id>/train_log.jsonl`, `status.json`, and process table every 20--30 minutes.
- During evaluation: inspect `logs/<run-id>/eval_orchestrator.log`, `logs/<run-id>/eval/`, and completed CSV count every 20--30 minutes.
- At 18:00: collect the final state, summary, logs, and a factual report.

## Health signals and stop conditions

Expected training log fields: `reward_mean`, `keep_ratio_mean`, `safety_loss_mean`, `budget_log_prob_mean`, `selection_log_prob_mean`, and finite gradient norm.

Stop and journal instead of improvising if:

- runtime same-scene baseline scoring fails repeatedly;
- all valid scenes are skipped;
- NaNs/infs occur;
- DDP fingerprint mismatch occurs;
- model ratio collapses to a bound for sustained updates;
- projected remaining runtime exceeds the 12-hour window;
- evaluation has missing/duplicate scene coverage.

A one-off code bug may be fixed only after backup, root-cause evidence, a journal entry, and a new smoke test. Do not tune research hyperparameters ad hoc to chase a score.

## Final acceptance criteria

A result is valid only if all are true:

1. `status.json` is `DONE`.
2. `safe_htpo_summary.md` reports 11,576 unique scenes.
3. Protocol is explicitly no-denylist and Variant-B true token drop.
4. Keep-ratio distribution is present (mean/std/P10/P50/P90).
5. Raw PDMS, P5 PDMS, six mean submetrics, and 95% bootstrap CI are present.
6. The dynamic policy is compared against fixed SFT controls at matched mean retention before any claim is edited.

## Required 18:00 report

Write `docs/journal/2026-07-28_safe_htpo_cycle_report.md` with:

- run ID, exact git diff / config, start/end timestamps;
- smoke outcome;
- final training status and checkpoint paths;
- full evaluation coverage and summary paths;
- raw PDMS, CI, P5, submetrics, ratio distribution;
- matched-retention baseline comparison;
- whether the AAAI claim gate passed, failed, or remains incomplete;
- all deviations and exact reverse instructions;
- recommended next action.

## Chat handoff procedure

If a chat-dev service is available, first verify the actual `scripts/dev-up.sh` path and API endpoint. This repository did not contain that script during preflight; do not invent an endpoint or a `projectId`.

Use the user-provided user ID only in the verified chat request. Discover the current `projectId` from the verified API response or active project context, then send this entire handoff document as the stateless agent's task prompt. Preserve the returned conversation/session ID in the cycle report.

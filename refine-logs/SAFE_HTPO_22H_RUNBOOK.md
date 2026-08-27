# Safe-HTPO Unattended 22:00 Runbook

## What is prepared

1. `scripts/train_scorer_budget_rl.py`
   - synchronized multi-process (`torchrun`) training with manual gradient all-reduce;
   - one random logistic-normal keep-ratio policy per scene in `[0.2, 0.9]`;
- one driving reward jointly updates budget and token-selection policy terms;
- the feature-capture pass reuses its same-scene unpruned trajectory as the delta/safety baseline, avoiding navtest baseline leakage into navtrain;
- explicit collision/drivable/TTC degradation penalty;
   - DDP parameter-fingerprint audit before final checkpoint.

2. `scripts/run_safe_htpo_aaai_unattended.sh`
   - syntax/data/checkpoint preflight;
   - automatic eight-GPU DDP smoke;
   - resumable one-epoch synchronized training;
   - automatic four-shard no-denylist Variant-B dynamic evaluation;
   - aggregate PDMS, submetrics, P5 PDMS, bootstrap CI, and keep-ratio distribution;
   - status file: `logs/<run-id>/status.json`.

3. `scripts/launch_safe_htpo_after_22h.sh`
   - waits for local time `2026-07-27 22:00:00` by default;
   - waits for eight GPUs below 1 GiB memory / 10% utilization when `nvidia-smi` is available;
   - then executes the unattended pipeline.

## Launch command

Run once before leaving:

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
mkdir -p logs/safe_htpo_launcher
nohup bash scripts/launch_safe_htpo_after_22h.sh \
  > logs/safe_htpo_launcher/nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The launcher waits; it does **not** start training before 22:00. To override the start time:

```bash
SAFE_START_AT='2026-07-27 22:10:00' nohup bash scripts/launch_safe_htpo_after_22h.sh > logs/safe_htpo_launcher/nohup_manual.log 2>&1 &
```

## Default final-run configuration

- 8 synchronized GPU ranks;
- one epoch after a small automatic DDP smoke;
- random `k_s` in `[0.2, 0.9]`; no scene is fixed to 0.5;
- `selection_pg_weight=1.0`;
- `efficiency_beta=0.15`;
- `safety_beta=0.50` and no allowed safety degradation margin;
- training uses Variant-A attention-mask proxy;
- evaluation uses Variant-B physical token drop;
- no `varB_denylist` argument is ever passed.

## Expected artifacts

- Training: `ckpt/safehtpo_<timestamp>/`
- Training logs / state: `logs/safehtpo_<timestamp>/`
- Raw evaluation and summary: `results/raw/safe_htpo_safehtpo_<timestamp>/`
- Final readable result: `safe_htpo_summary.md`

## Autonomous failure behavior

- failed smoke blocks final training;
- strict 12-hour defaults do not automatically retry a failed train; a resume checkpoint is preserved for the next window;
- missing checkpoint, manifest, evaluation CSV, or incomplete 11,576-scene coverage marks status `FAILED`;
- completed pipeline marks status `DONE`.

## Interpretation rule

The run is evidence only if `safe_htpo_summary.md` reports full 11,576 unique scenes and no-denylist Variant-B protocol. It must still be compared against fixed SFT controls at matched **mean** retention before any paper claim is updated.

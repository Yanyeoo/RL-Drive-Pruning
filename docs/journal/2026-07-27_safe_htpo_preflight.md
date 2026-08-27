# Safe-HTPO Preflight Journal — 2026-07-27

## Decision

Use a **runtime same-scene unpruned trajectory** as the baseline for the Stage-2 delta reward and safety-degradation penalty. Do not use `results/baseline_sub_scores.json` during navtrain policy learning.

## Fact check

Before the unattended run, the following overlap was measured:

- `results/baseline_sub_scores.json`: 11,574 scene keys;
- overlap with `data/navtrain_nocot`: **0 / 19,225**;
- overlap with `data/navtest_nocot`: **11,574 / 11,596**.

Therefore the old external JSON is a navtest-only artifact. Passing it to navtrain training silently caused `baseline_scores=None`, which reduced the shaped reward to its absolute-only fallback and made the safety penalty zero. It cannot support the intended delta/safety method claim.

## Implemented correction

In `scripts/train_scorer_budget_rl.py`:

1. The existing first, unpruned `autovla.predict` pass used to capture layer-0 features now retains its predicted poses.
2. These poses form `baseline_trajectory` for the exact same scene.
3. PDM submetrics are computed from this baseline trajectory.
4. The pruned trajectory is scored against those runtime baseline submetrics for both the delta reward and collision/drivable/TTC safety-degradation penalty.
5. If either trajectory or complete submetric set is unavailable, the scene is skipped while the DDP rank still participates in a zero-gradient synchronized step.

This adds one PDM scoring call per valid scene but does **not** add another AutoVLA forward pass.

## Reverse instruction

If runtime baseline PDM scoring materially violates the 12-hour budget or introduces instability, do **not** silently restore the navtest JSON. Instead:

1. stop the run or set status `FAILED`;
2. journal measured throughput and failure evidence;
3. either generate a clean navtrain no-prune baseline artifact under a separate protocol, or remove delta/safety claims from the AAAI method and paper before relaunch.

## Launcher consequence

`scripts/run_safe_htpo_aaai_unattended.sh` no longer requires or passes `results/baseline_sub_scores.json` to training. Its immutable `run_config.json` records that the baseline source is the same-scene full-token rollout.

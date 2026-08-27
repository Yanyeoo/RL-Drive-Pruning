# 4×H20 Physical-Token-Removal Recovery Runbook

**Owner:** execution agent with four H20 GPUs and access to this workspace  
**Run window:** start immediately; hard recovery deadline **2026-07-29 14:00 Asia/Shanghai**  
**Scope:** complete missing, reproducible NAVSIM physical-token-removal evidence for the main table.  
**Out of scope:** new Safe-HTPO training, 8-GPU jobs, hyperparameter sweeps, external-server results, and unsupported paper numbers.

---

## 1. Non-negotiable rules

1. Use only **AutoVLA-3B** and `+agent.prune_variant=drop` for all NAVSIM evaluations in this recovery run.
2. Do **not** launch `Safe-HTPO`, `torchrun`, or any 8-GPU training job. The completed local Safe-HTPO result is already archived and is not a candidate for this recovery queue.
3. Do **not** overwrite existing CSVs under `results/raw/tokenprune_S3_full/`. Historical files may be mask-based or protocol-mixed. All new jobs must use fresh experiment names and a fresh recovery output directory.
4. Do **not** claim or recreate `0.9107`; there is no local artifact chain for it.
5. Treat `0.90452`, `0.90399`, and `0.87010` only as **post-hoc fallback estimates** until a real four-shard `safety_net=True` evaluation completes.
6. Never report a partial shard as a full-navtest result. An arm is reportable only when all four shards exist, tokens are unique, scores are finite, and the aggregate records its valid unique-scene count.
7. Do not modify `paper/prism/main.tex` or `paper/aaai2027/main.tex` during the run. Write a candidate table and evidence first; the paper update requires a separate review.

---

## 2. Required reading and authoritative local facts

Read these before launch:

```text
results/main_table.md
docs/journal/2026-07-21_decisions_backup.md
docs/journal/2026-07-21_main_table_plan.md
docs/journal/2026-07-22_window_end_summary.md
docs/journal/2026-07-28_safe_htpo_cycle_report.md
docs/results/README.md
```

Known physical-token-removal raw evidence already on disk:

| Arm | Raw PDMS | Valid unique scenes | Evidence |
|---|---:|---:|---|
| SFT scorer, `r=0.50` | `0.872533` | `11,576` | `MT_varBsafe_scorer_r05_sh{0,1,2,3}.csv` |
| SFT scorer, `r=0.75` | `0.881638` | `11,576` | `MT_varBsafe_scorer_r075_sh{0,1,2,3}.csv` |
| RL-shaped scorer, `r=0.25` | `0.811271` | `11,576` | `MT_rl_shaped_r025_sh{0,1,2,3}.csv` |
| RL-shaped scorer, `r=0.50` | `0.890938` | `11,571` | `MT_rl_shaped_r05_sh{0,1,2,3}.csv` |
| BudgetRL fixed control, `r=0.50` | `0.872273` | `11,576` | `MT_budget_rl_r050_sh{0,1,2,3}.csv` |
| BudgetRL dynamic | `0.848654` | `11,576` | `MT_budget_rl_dynamic_sh{0,1,2,3}.csv` |

There is an unfinished historical backfill:

```text
logs/backfill_maintable/status.json
logs/backfill_maintable.lock
scripts/backfill_maintable_gpu4_7.sh
```

Before touching its lock, confirm that neither its PID nor a `run_pdm_score_cot.py` child is alive. The remaining useful arm is `rl_shaped r=0.75, shard 3`; the `rl_taucut` shards are lower priority.

---

## 3. Required output locations

Create one isolated run root, never reuse historical raw filenames:

```text
logs/physical_drop_recovery_20260728/
results/raw/physical_drop_recovery_20260728/
results/raw/physical_drop_recovery_20260728/summary.json
results/raw/physical_drop_recovery_20260728/summary.md
docs/journal/2026-07-29_4gpu_physical_drop_recovery.md
docs/results/2026-07-29_physical_drop_main_table.md
```

Each completed arm must have:

```text
<run-root>/<arm>/shard_<0..3>.csv
<run-root>/<arm>/shard_<0..3>.log
<run-root>/<arm>/manifest.json
```

`manifest.json` must include: exact command, Git revision if available, checkpoint path, selector, keep ratio, `prune_variant=drop`, `safety_net`, denylist path or `null`, GPU ID, start/end time, exit code, source NAVSIM experiment directory, and copied CSV path.

The top-level `summary.json` must contain only completed four-shard arms and, for each arm: `n_rows`, `n_unique_tokens`, `duplicate_rows`, `nonfinite_scores`, `mean_pdms`, per-shard counts, selector, ratio/dynamic policy, checkpoint, safety-net state, and denylist state.

---

## 4. Preflight and recovery procedure

Run from:

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
source scripts/setup_navsim_env_vars.sh
export PYTHONPATH="$PWD/code:$PWD/code/third_party/AutoVLA/navsim:$PWD/code/third_party/AutoVLA:${PYTHONPATH:-}"
```

### 4.1 Verify GPUs and avoid duplicate jobs

1. Locate the four assigned H20 GPU IDs using the execution agent's working GPU tool/environment.
2. Check for active jobs:

```bash
ps -eo pid=,ppid=,etime=,stat=,args= | grep -E '[r]un_pdm_score_cot|[t]orchrun|[t]rain_scorer|[b]ackfill' || true
```

3. If a live `run_pdm_score_cot.py` job exists, identify its experiment name and **resume/monitor it** instead of launching a duplicate.
4. If `logs/backfill_maintable.lock` exists, test its PID with `kill -0`. Only if the PID is absent and no matching evaluator is live may the stale lock be removed. Keep all old logs and status files.
5. Start a fresh recovery status file with stage, active arm, timestamps, GPU allocation, and next queued arm. Update it after every completed shard.

### 4.2 Base evaluation command

Every new recovery shard must use the following pinned base configuration:

```bash
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
AUTOVLA_ROOT=$ROOT/code/third_party/AutoVLA
NAVSIM_ROOT=$AUTOVLA_ROOT/navsim
CKPT=$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt
YAML=$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml
SENSOR=$ROOT/data/navsim_v2_local

cd "$NAVSIM_ROOT"
CUDA_VISIBLE_DEVICES=<GPU_ID> timeout --signal=TERM --kill-after=5m 40000 \
  "$PY" navsim/planning/script/run_pdm_score_cot.py \
  experiment_name=<FRESH_EXPERIMENT_NAME> \
  train_test_split=navtest_local_filtered_shard<SHARD>_20260616_154858 \
  metric_cache_path=$ROOT/data/navtest_metric_cache \
  +json_data_path=$ROOT/data/navtest_nocot \
  agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
  +agent.config_path=$YAML \
  +agent.checkpoint_path=$CKPT \
  +agent.sensor_data_path=$SENSOR \
  +agent.codebook_cache_path=$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl \
  +agent.lora_conf.use_lora=false \
  +agent.keep_ratio=<RATIO> \
  +agent.selector=<SELECTOR> \
  +agent.prune_variant=drop \
  +agent.prune_verbose=false \
  worker=single_machine_thread_pool worker.max_workers=1
```

For `selector=scorer`, add:

```bash
+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer
```

For RL-shaped scorer evaluation, add:

```bash
+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh0/ckpt_best
```

For the real safety evaluation, add both:

```bash
+agent.scorer_ckpt=$ROOT/ckpt/s3_token_scorer \
+agent.safety_net=true \
+agent.varB_denylist=$ROOT/results/varB_catastrophic_tokens.json
```

Use a fresh experiment name such as:

```text
PDREC_20260728_<selector>_r050_raw_sh0
PDREC_20260728_sft_r050_safetynet_denylist_sh0
```

After a process returns, harvest exactly one CSV from:

```text
$NAVSIM_EXP_ROOT/<FRESH_EXPERIMENT_NAME>/*/*.csv
```

Copy it into the recovery output directory. A non-zero exit code is not automatically fatal if a complete CSV exists, but record the exit code and investigate the log before accepting the shard.

---

## 5. Priority queue through the deadline

Use all four GPUs continuously. Assign one independent shard to each GPU. Do not mix two dispatchers that might both launch `run_pdm_score_cot.py`.

### P0 — recover already-started high-value work

1. Finish **RL-shaped scorer, `r=0.75`, shard 3** using the pinned RL-shaped checkpoint and raw physical-token-removal configuration.
2. Complete the four-shard **SFT scorer, `r=0.50`, `safety_net=true` + denylist** evaluation. This replaces the current post-hoc fallback estimate with a directly executed result.

Recommended packing:

- GPU A: `RL-shaped r=.75 sh3`;
- GPUs B/C/D: safety evaluation shards `0/1/2`;
- as soon as a GPU becomes free: safety evaluation shard `3`.

Do not spend early deadline hours on `rl_taucut` unless P0 is already complete.

### P1 — complete the main baseline comparison at the central operating point

Run these as separate full four-shard raw physical-token-removal arms, one arm at a time across all four GPUs:

1. `fastv_l2`, `r=0.50`;
2. `random`, `r=0.50`;
3. `prumerge_cls`, `r=0.50`;
4. `sparsevlm_text`, `r=0.50`.

Use `safety_net=false` and no denylist for all baseline arms. Do not reuse historical output names such as `MT_fastv_l2_r05_sh0.csv`; those names already have protocol ambiguity.

### P2 — only after P0 and P1 finish

Use remaining time in this order:

1. FastV `r=0.75`, four shards;
2. SparseVLM `r=0.75`, four shards;
3. Random `r=0.75`, four shards;
4. PruMerge `r=0.75`, four shards;
5. FastV / Random / PruMerge / SparseVLM `r=0.25`, each only if all higher-priority work is complete.

### Stop rule

At **2026-07-29 12:00**, stop starting new full four-shard arms. Use the final two hours for harvesting, validation, aggregation, journal updates, and the recovery report. Let already-running shards finish unless they cannot complete before 14:00.

---

## 6. Validation gate for each completed arm

Before an arm enters any result table:

1. Confirm four shard CSV files exist.
2. Remove only rows with empty `token`, `token == "average"`, non-finite score, or explicitly invalid score records; preserve a count for every exclusion.
3. Confirm no duplicated scene tokens after merging shards.
4. Report `N`, mean PDMS, relative PDMS against `0.89879`, per-shard row counts, and every exclusion.
5. Inspect each log for `variant=drop` / physical-token-removal evidence and for `OOM`, `NCCL`, traceback, timeout, or repeated decode fallback.
6. Confirm exact selector, ratio, checkpoint, `safety_net`, and denylist state from the command/manifest.

Never convert a partial result into a full-navtest mean. Place incomplete arms under a separate `Incomplete / not reportable` heading in the journal.

---

## 7. Required written updates before 14:00

### 7.1 Journal: mandatory

Create/update:

```text
docs/journal/2026-07-29_4gpu_physical_drop_recovery.md
```

Required sections:

1. Window, GPU IDs, start/finish time, operator identity;
2. preflight and stale-lock decision;
3. exact queue and actual execution order;
4. each full arm's command/config and artifact paths;
5. completed results table with `N`, PDMS, Rel., and safety state;
6. incomplete/failed arms with final log line and recovery recommendation;
7. explicit statement that `0.9107` was neither generated nor used;
8. deviations from this runbook and why.

### 7.2 Result source of truth: mandatory for completed four-shard arms only

Update `docs/results/key_results.md` only after passing the validation gate. Add a dated section and changelog rows with:

- headline result;
- raw physical-token-removal protocol;
- checkpoint and safety state;
- `N` and PDMS;
- comparison to the `0.89879` reference;
- paths to the recovery CSVs, manifests, summary JSON, and journal.

### 7.3 Candidate table: mandatory

Write:

```text
docs/results/2026-07-29_physical_drop_main_table.md
```

This must contain the new raw physical-token-removal matrix and a separate safety-evaluated SFT row. It is a review artifact; do not change the paper table automatically.

### 7.4 Final recovery package

Write both:

```text
results/raw/physical_drop_recovery_20260728/summary.json
results/raw/physical_drop_recovery_20260728/summary.md
```

The markdown summary must explicitly distinguish:

- raw physical-token-removal results;
- directly executed safety-net + denylist results;
- historical post-hoc fallback estimates;
- incomplete or missing arms.

---

## 8. Monitoring and failure handling

- Check the active processes and latest four logs every 20--30 minutes.
- If a shard ends without a CSV, retry that exact shard once with the same fresh experiment name plus `_retry1`; record both attempts.
- If retry fails, move to the next queued arm and record the failure; do not block all GPUs.
- If an OOM/NCCL error occurs, stop only the affected worker, preserve its log, and do not silently lower model precision, batch behavior, timeout, or research settings.
- If storage or `NAVSIM_EXP_ROOT` harvesting fails, fix only the copy/paths and preserve the source experiment output before rerunning inference.
- If no GPU telemetry command is available in the shell, use the executor's GPU management interface; do not infer idleness solely from an absent `nvidia-smi` binary.

---

## 9. Final handoff at 14:00

The final message must contain only verified facts:

- completed full arms and their PDMS/N;
- directly executed safety result, if completed;
- incomplete arms and reason;
- exact artifact and journal paths;
- whether `docs/results/key_results.md` was updated;
- whether the candidate table is ready for paper review.

Do not claim an SOTA, a fallback gain, dynamic-policy improvement, or end-to-end speedup without the corresponding newly archived artifact.

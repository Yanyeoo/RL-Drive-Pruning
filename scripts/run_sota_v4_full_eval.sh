#!/usr/bin/env bash
# run_sota_v4_full_eval.sh — 全量 navtest eval (4 shard, 4 GPU)
# 用法：bash scripts/run_sota_v4_full_eval.sh <CKPT_DIR> [TAG_PREFIX]
#   例如：bash scripts/run_sota_v4_full_eval.sh ckpt/s3_token_scorer_budget_rl_20260804_XXXXXX_sh0 SOTAV4_R1
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT_DIR="${1:?usage: run_sota_v4_full_eval.sh <CKPT_DIR> [TAG_PREFIX]}"
TAG_PREFIX="${2:-SOTAV4_FULL}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
PREFIX="navtest_local_filtered_shard"; SUFFIX="_20260616_154858"
LOGDIR="$ROOT/logs/sota_v4_full_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

log(){ echo "[sota-v4-full $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

log "=== Full navtest eval ==="
log "ckpt_dir: $CKPT_DIR"
log "tag_prefix: $TAG_PREFIX"
log "logdir: $LOGDIR"

# Use the provided ckpt_dir directly (should have checkpoint.pt)
if [[ ! -f "$CKPT_DIR/checkpoint.pt" ]]; then
  log "FATAL: checkpoint.pt not found in $CKPT_DIR"
  exit 2
fi
log "ckpt: $CKPT_DIR ($(ls -lh "$CKPT_DIR/checkpoint.pt" | awk '{print $5}'))"

PIDS=""
for SH in 0 1 2 3; do
  local EXP="${TAG_PREFIX}_sh${SH}"
  local CSV="$ROOT/results/raw/${EXP}.csv"
  local JLOG="$LOGDIR/_${EXP}.log"
  if [[ -f "$CSV" ]]; then
    log "SKIP $EXP (csv exists)"
    continue
  fi
  local GPU=$SH
  log "GPU$GPU START $EXP shard=$SH"
  (
    export CUDA_VISIBLE_DEVICES=$GPU
    cd "$NAVSIM_ROOT"
    $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$EXP" \
      train_test_split="${PREFIX}${SH}${SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$CKPT_DIR" \
      +agent.keep_ratio=0.5 \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$JLOG" 2>&1 &
  PIDS="$PIDS $!"
done

log "PIDs: $PIDS"
wait

# Aggregate
log "=== Aggregating ==="
$PY - <<'PYEOF'
import glob, pandas as pd, statistics
rows = []
for f in sorted(glob.glob("/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/SOTAV4_FULL_sh*.csv")):
    d = pd.read_csv(f)
    d = d[d['token'] != 'average']
    rows.append(d['score'].mean())
    print(f"  {f.split('/')[-1]}: N={len(d)} PDMS={d['score'].mean():.5f} NC={d['num_penalties_collisions'].mean():.4f} EP={d['ego_progress'].mean():.4f}")
if len(rows) > 1:
    print(f"\n  OVERALL: n={len(rows)} mean={statistics.mean(rows):.5f} std={statistics.stdev(rows):.5f}")
elif len(rows) == 1:
    print(f"\n  OVERALL: n=1 PDMS={rows[0]:.5f}")
PYEOF

log "=== Done ==="

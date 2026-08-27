#!/usr/bin/env bash
# run_budget_rl_eval_quick.sh — 快速 eval 刚训练完的 Budget RL ckpt
# 用法：bash run_budget_rl_eval_quick.sh <OUT_DIR_ROOT>
#   例如：bash run_budget_rl_eval_quick.sh ckpt/s3_token_scorer_budget_rl_20260803_153000
# 自动选择 best ckpt（sh0/ckpt_best），在 navtest 全量 4 shard 上并行 eval（4 GPU）
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

OUT_DIR="${1:?usage: run_budget_rl_eval_quick.sh <OUT_DIR_ROOT>}"
# OUT_DIR 形如 ckpt/s3_token_scorer_budget_rl_20260803_153000，取 sh0/ckpt_best
BCKPT="${OUT_DIR}_sh0/ckpt_best"
if [[ ! -f "$BCKPT/checkpoint.pt" ]]; then
    echo "[eval] ckpt not found: $BCKPT"
    BCKPT=$(ls -dt ${OUT_DIR}_sh0/ckpt_step* 2>/dev/null | head -1)
    [[ -z "$BCKPT" ]] && { echo "[eval] FATAL: no ckpt"; exit 2; }
fi
echo "[eval] Using ckpt: $BCKPT"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
PREFIX="navtest_local_filtered_shard"; SUFFIX="_20260616_154858"
TAG="SOTA_budgetrl_dynamic_raw"
OUTDIR="$ROOT/results/raw"
LOGDIR="$ROOT/logs/sota_rl_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOGDIR"

log(){ echo "[sota-eval $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

eval_worker(){
  local gpu="$1" sh="$2"
  local EXP="${TAG}_sh${sh}"
  local CSV="$OUTDIR/${EXP}.csv"
  local JLOG="$LOGDIR/_${EXP}.log"
  if [[ -f "$CSV" ]]; then log "SKIP $EXP (csv exists)"; return; fi
  log "GPU$gpu START $EXP shard=$sh"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$EXP" \
      train_test_split="${PREFIX}${sh}${SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$BCKPT" \
      +agent.keep_ratio=0.5 \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$JLOG" 2>&1
  local rc=$?
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$EXP"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$CSV"
    local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$CSV');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f} N={len(d)}')" 2>/dev/null)
    log "DONE $EXP rc=$rc PDMS=$pdms"
  else
    log "FAIL $EXP rc=$rc"
  fi
}

PIDS=()
for sh in 0 1 2 3; do
    eval_worker "$sh" "$sh" &
    PIDS+=($!)
done

printf '%s\n' "${PIDS[@]}" > "$LOGDIR/pids.txt"
wait "${PIDS[@]}"

log "=== Aggregating ==="
$PY - <<EOF 2>&1 | tee -a "$LOGDIR/run.log"
import glob, re, pandas as pd
fs = sorted(glob.glob("$OUTDIR/${TAG}_sh*.csv"))
d = pd.concat([pd.read_csv(f) for f in fs])
d = d[d['token'] != 'average'].drop_duplicates('token')
print(f"\n  SOTA Budget RL dynamic raw (full navtest)")
print(f"  N={len(d)} PDMS={d['score'].mean():.5f} NC={d['no_at_fault_collisions'].mean():.4f} EP={d['ego_progress'].mean():.4f}")
EOF
log "=== Done. Logs: $LOGDIR ==="

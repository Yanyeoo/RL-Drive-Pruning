#!/usr/bin/env bash
# ============================================================================
# eval_v7_folds_4gpu.sh — v7 surrogate ablation 的互斥 fold 评估（4×H20）
#
# 评估策略（避免上周期 200-scene 门控的 selection bias）：
#   - gate 阶段：用 navtest 的 4 个互斥 shard 作为 fold，先评估 4 个 arm 的
#     final checkpoint 在 shard0+shard1（约 5800 scenes）上的表现。
#   - final 阶段：对 gate 胜者跑 full 4-shard（约 11576 scenes）。
#
# 用法：
#   bash scripts/eval_v7_folds_4gpu.sh <CYCLE_ID> [gate|full] [arm1 arm2 ...]
# ============================================================================
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CYCLE_ID=${1:?usage: eval_v7_folds_4gpu.sh <CYCLE_ID> [gate|full] [arms...]}
PHASE=${2:-gate}
RUN_ROOT="$ROOT/ckpt/v7_surrogate_${CYCLE_ID}"
OUTDIR="$ROOT/results/raw/v7_surrogate_${CYCLE_ID}_${PHASE}"
LOGDIR="$ROOT/logs/v7_surrogate_${CYCLE_ID}"
mkdir -p "$OUTDIR" "$LOGDIR"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"

if [[ $# -ge 3 ]]; then
  ARMS=("${@:3}")
else
  ARMS=(softmax_open st_topk gumbel st_topk_tau1)
fi
if [[ "$PHASE" == "gate" ]]; then SHARDS=(0 1); elif [[ "$PHASE" == "gate0" ]]; then SHARDS=(0); else SHARDS=(0 1 2 3); fi
# GPU pool (default 0 1 2 3; override via EVAL_GPUS="0 1 2 3 4 5 6 7")
EVAL_GPUS=${EVAL_GPUS:-"0 1 2 3"}
# per-GPU 并行 worker 数（默认 1；H20 96G 可到 3~4，加速 eval）
EVAL_WORKERS=${EVAL_WORKERS:-1}

log(){ echo "[v7-eval $(date +%H:%M:%S)] $*"; }

# 收集 (arm, shard) 任务，然后按 GPU 循环派发（4 卡并行，任务队列）
declare -a JOBS
for arm in "${ARMS[@]}"; do
  SCORER="$RUN_ROOT/$arm"   # final checkpoint 直接存在 arm 根目录
  [[ -f "$SCORER/checkpoint.pt" ]] || { log "MISSING $SCORER/checkpoint.pt, skip"; continue; }
  for sh in "${SHARDS[@]}"; do
    JOBS+=("$arm|$sh")
  done
done
log "total jobs: ${#JOBS[@]} (arms=${#ARMS[@]} shards=${#SHARDS[@]})"

run_job() {
  local arm=$1 sh=$2
  local exp="v7_${PHASE}_${arm}_sh${sh}"
  local csv="$OUTDIR/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp"; return; }
  local gpu=$3
  log "GPU$gpu START $exp"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES=$gpu
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" \
      +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.keep_ratio=0.5 \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$RUN_ROOT/$arm" \
      +agent.prune_verbose=false \
      worker=single_machine_thread_pool worker.max_workers=$EVAL_WORKERS
  ) > "$LOGDIR/eval_${exp}.log" 2>&1
  # copy result csv
  local found
  found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  [[ -n "$found" ]] && cp -a "$found" "$csv" && log "DONE $exp -> $csv"
}

# 派发到 4 卡（简单的轮询队列）
PIDS=()
GPU_FREE=(0 1 2 3)
idx=0
for job in "${JOBS[@]}"; do
  arm="${job%%|*}"; sh="${job##*|}"
  # 等待有空闲 GPU
  while :; do
    free_gpu=""
    for g in $EVAL_GPUS; do
      if [[ ! -f "$LOGDIR/.lock_gpu$g" ]]; then free_gpu=$g; break; fi
    done
    [[ -n "$free_gpu" ]] && break
    sleep 15
  done
  touch "$LOGDIR/.lock_gpu$free_gpu"
  ( run_job "$arm" "$sh" "$free_gpu"; rm -f "$LOGDIR/.lock_gpu$free_gpu" ) &
done
wait
log "ALL JOBS DONE"

# 汇总 PDMS（每 arm 跨 shard 平均 + 最差 shard）
"$PY" - "$OUTDIR" <<'PY'
import sys, os
import pandas as pd
outdir = sys.argv[1]
files = [f for f in os.listdir(outdir) if f.endswith('.csv')]
print(f"{'arm':16s} {'n_shards':>8s} {'N_total':>8s} {'PDMS_mean':>10s} {'PDMS_min_shard':>16s}")
rows = {}
for f in sorted(files):
    parts = f[:-4].split('_')          # v7_gate_<arm>_sh<sh>
    sh = parts[-1]
    arm = '_'.join(parts[2:-1])
    df = pd.read_csv(os.path.join(outdir, f))
    df = df[df['token'] != 'average']
    rows.setdefault(arm, {})[sh] = (len(df), df['score'].mean())
for arm, shmap in sorted(rows.items()):
    ns = [v[0] for v in shmap.values()]
    pdms = [v[1] for v in shmap.values()]
    mean_p = sum(pdms)/len(pdms)
    min_p = min(pdms)
    print(f"{arm:16s} {len(shmap):8d} {sum(ns):8d} {mean_p:10.6f} {min_p:16.6f}")
PY
log "eval summary done"

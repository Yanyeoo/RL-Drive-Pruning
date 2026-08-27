#!/usr/bin/env bash
# run_budget_rl_sota_v4.sh — ICLR 周期 Budget RL 训练（修复版）
#
# 修复 v3 的问题：
#   1. train() stdout 日志混入返回值 → BCKPT 含多行 → eval/R2 失败
#      修复：train() 日志写 stderr，仅最后 echo 路径到 stdout
#   2. efficiency_beta=0.15 太高 → kr 从 0.55 跌到 0.34
#      修复：efficiency_beta=0.05，取消 adaptive_efficiency（用线性 bonus）
#   3. ckpt_best 停在 step~25 不代表最终性能
#      修复：eval 用 FINAL checkpoint.pt（step 151），同时记录 ckpt_best
#   4. 适配 H20×4
#
# 超参设计：
#   - efficiency_beta=0.05：温和效率激励（原 0.15 的 1/3）
#   - driving_scale=3.0：保持驾驶质量信号强度
#   - safety_beta=0.05：轻度安全惩罚（原 0.1）
#   - budget_log_std_init=-0.5：保持较宽探索
#   - NO adaptive_efficiency：恢复线性 efficiency bonus
#   - 1 epoch，~150 steps/shard，4 GPU 并行
#
# 用法：bash scripts/run_budget_rl_sota_v4.sh
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# === 超参 ===
EPOCHS=1
GROUP_SIZE=16
EFFICIENCY_BETA=0.05       # 温和（原 0.15）
DRIVING_SCALE=3.0
BUDGET_LR=1e-4
LR=3e-5
KL_BETA=0.01
BUDGET_LOG_STD_INIT=-0.5
MIN_KR=0.2; MAX_KR=0.9
SAFETY_BETA=0.05           # 轻度（原 0.1）
SAFETY_MARGIN=0.02
PRUNE_VARIANT=attn_mask    # 训练用 attn_mask（eval 用 drop）
NUM_GPUS=4                 # H20×4

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
BASELINE="$ROOT/results/baseline_sub_scores.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$ROOT/ckpt/s3_token_scorer_budget_rl_${TIMESTAMP}"

echo "=========================================="
echo "[sota-v4] Mild Budget RL (ICLR cycle)"
echo "  gpus=$NUM_GPUS  epochs=$EPOCHS"
echo "  eff_beta=$EFFICIENCY_BETA  drive_scale=$DRIVING_SCALE"
echo "  safety_beta=$SAFETY_BETA  log_std_init=$BUDGET_LOG_STD_INIT"
echo "  adaptive=OFF  (linear efficiency bonus)"
echo "  out=$OUT_BASE"
echo "=========================================="

train(){
  # 日志写 stderr，返回值写 stdout（修复 v3 bug）
  local tag="$1" init_ckpt="$2" out_dir="$3"
  echo "[sota-v4] Training: $tag  init=$init_ckpt" >&2

  PIDS=""
  for ((SH=0; SH<NUM_GPUS; SH++)); do
    local GPU=$SH
    (
      export CUDA_VISIBLE_DEVICES=$GPU
      $PY scripts/train_scorer_budget_rl.py \
        --scorer-ckpt "$init_ckpt" \
        --out-dir "${out_dir}_sh${SH}" \
        --json-dir "$ROOT/data/navtrain_nocot" \
        --metric-cache "$ROOT/data/navtrain_metric_cache" \
        --efficiency-beta $EFFICIENCY_BETA \
        --driving-scale $DRIVING_SCALE \
        --num-epochs $EPOCHS \
        --group-size $GROUP_SIZE \
        --lr $LR --budget-lr $BUDGET_LR \
        --kl-beta $KL_BETA \
        --min-keep-ratio $MIN_KR --max-keep-ratio $MAX_KR \
        --budget-log-std-init $BUDGET_LOG_STD_INIT \
        --safety-beta $SAFETY_BETA --safety-margin $SAFETY_MARGIN \
        --shaped-reward --baseline-scores "$BASELINE" \
        --num-shards $NUM_GPUS --shard-id $SH \
        --seed $((42 + SH)) \
        --prune-variant $PRUNE_VARIANT \
        --device cuda:0
    ) > "$ROOT/logs/sota_v4_${tag}_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
  done
  echo "[sota-v4] $tag PIDs: $PIDS" >&2
  wait
  echo "[sota-v4] $tag DONE" >&2

  # Pick best: prefer FINAL checkpoint.pt, fallback to latest step, then ckpt_best
  local best="${out_dir}_sh0/checkpoint.pt"
  if [[ ! -f "$best" ]]; then
    local latest_step=$(ls -d ${out_dir}_sh0/ckpt_step* 2>/dev/null | sed -E 's/.*ckpt_step([0-9]+)/\1/' | sort -n | tail -1)
    if [[ -n "$latest_step" && -f "${out_dir}_sh0/ckpt_step${latest_step}/checkpoint.pt" ]]; then
      best="${out_dir}_sh0/ckpt_step${latest_step}"
    elif [[ -f "${out_dir}_sh0/ckpt_best/checkpoint.pt" ]]; then
      best="${out_dir}_sh0/ckpt_best"
    fi
  else
    best="${out_dir}_sh0"
  fi
  echo "[sota-v4] $tag best ckpt: $best" >&2
  # ONLY the path goes to stdout (for $() capture)
  echo "$best"
}

# Quick eval on shard0 (~50 min)
eval_one(){
  local ckpt="$1" tag="$2"
  echo "[sota-v4] Eval: $tag  ckpt=$ckpt" >&2
  (
    export CUDA_VISIBLE_DEVICES=0
    cd "$NAVSIM_ROOT"
    $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$tag" \
      train_test_split="navtest_local_filtered_shard0_20260616_154858" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.selector=scorer_budget \
      +agent.scorer_ckpt="$ckpt" \
      +agent.keep_ratio=0.5 \
      +agent.prune_variant=drop +agent.prune_verbose=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$ROOT/logs/sota_v4_eval_${tag}.log" 2>&1
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$tag"/*/*.csv 2>/dev/null | head -1)
  local pdms="N/A"
  if [[ -n "$found" ]]; then
    cp -a "$found" "$ROOT/results/raw/${tag}.csv"
    pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
  fi
  echo "[sota-v4] $tag: PDMS=$pdms (shard0)" >&2
  echo "$pdms"
}

# ============================================================
# Main
# ============================================================
INIT="$SCORER_CKPT"
declare -a ROUND_PDMS

# Phase 1: Mild Budget RL from SFT init
BCKPT=$(train "R1" "$INIT" "$OUT_BASE")
[[ -z "$BCKPT" ]] && { echo "FATAL: Round 1 training failed"; exit 1; }
echo "[sota-v4] R1 ckpt: $BCKPT"

# Eval FINAL ckpt
P1_FINAL=$(eval_one "$BCKPT" "SOTAV4_R1_FINAL")
echo "[sota-v4] R1 FINAL PDMS=$P1_FINAL"

# Also eval ckpt_best for comparison
BEST_CKPT="${OUT_BASE}_sh0/ckpt_best"
if [[ -f "$BEST_CKPT/checkpoint.pt" ]]; then
  P1_BEST=$(eval_one "$BEST_CKPT" "SOTAV4_R1_BEST")
  echo "[sota-v4] R1 BEST  PDMS=$P1_BEST"
fi

echo ""
echo "=========================================="
echo "[sota-v4] SUMMARY"
echo "  R1 FINAL: PDMS=$P1_FINAL (shard0)"
[[ -n "${P1_BEST:-}" ]] && echo "  R1 BEST:  PDMS=$P1_BEST (shard0)"
echo "=========================================="

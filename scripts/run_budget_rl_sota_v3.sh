#!/usr/bin/env bash
# run_budget_rl_sota_v3.sh — 精简版多轮自适应 Budget RL（适配今晚 GPU 窗口）
#
# 时间窗口：Block A 结束(~15:30) → 24:00 回收 = 8.5h
# 单 epoch ≈ 5.5h (8 GPU)，目标：跑 1-2 轮 × 1 epoch 验证改进
#
# 方法改进（vs v1）：
#   1. adaptive_efficiency: reward = -beta*(kr - target_kr)^2 （目标锚定）
#      替代原来的 efficiency = beta*(1-kr)（线性 push 更少）
#   2. 增大 budget_log_std_init (-1.0 → -0.5)：更宽的高斯探索
#   3. safety_beta=0.1：开启安全惩罚，防止 unsafe over-pruning
#   4. driving_scale=3.0：增大驾驶质量信号
#   5. 多轮迭代：每轮从上一轮 best ckpt 初始化，逐轮 refine
#
# 用法：bash scripts/run_budget_rl_sota_v3.sh
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

# === 超参 ===
EPOCHS=1                     # 今晚只跑 1 epoch/round（~5.5h）
GROUP_SIZE=16
EFFICIENCY_BETA=0.15
DRIVING_SCALE=3.0
BUDGET_LR=1e-4
LR=3e-5
KL_BETA=0.01
BUDGET_LOG_STD_INIT=-0.5    # 更宽探索（原 -1.0）
MIN_KR=0.2; MAX_KR=0.9
TARGET_KR=0.355
SAFETY_BETA=0.1
SAFETY_MARGIN=0.02
PRUNE_VARIANT=attn_mask

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SCORER_CKPT="$ROOT/ckpt/s3_token_scorer"
BASELINE="$ROOT/results/baseline_sub_scores.json"
OUT_BASE="$ROOT/ckpt/s3_token_scorer_budget_rl_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "[sota-v3] Adaptive Budget RL (target-centric efficiency)"
echo "  epochs=$EPOCHS  eff_beta=$EFFICIENCY_BETA  drive_scale=$DRIVING_SCALE"
echo "  target_kr=$TARGET_KR  safety_beta=$SAFETY_BETA  log_std_init=$BUDGET_LOG_STD_INIT"
echo "=========================================="

train(){
  local tag="$1" init_ckpt="$2" out_dir="$3"
  echo "[sota-v3] Training: $tag  init=$init_ckpt"

  PIDS=""
  for SH in 0 1 2 3 4 5 6 7; do
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
        --adaptive-efficiency --target-keep-ratio $TARGET_KR \
        --shaped-reward --baseline-scores "$BASELINE" \
        --num-shards 8 --shard-id $SH \
        --seed $((42 + SH)) \
        --prune-variant $PRUNE_VARIANT \
        --device cuda:0
    ) > "$ROOT/logs/budget_rl_sota_${tag}_sh${SH}.log" 2>&1 &
    PIDS="$PIDS $!"
  done
  echo "[sota-v3] $tag PIDs: $PIDS"
  wait
  echo "[sota-v3] $tag DONE"

  # Pick best
  local best="${out_dir}_sh0/ckpt_best"
  if [[ ! -f "$best/checkpoint.pt" ]]; then
    best=$(ls -dt ${out_dir}_sh0/ckpt_step* 2>/dev/null | head -1)
  fi
  echo "$best"
}

# Quick eval (shard0 only, ~50min)
eval_one(){
  local ckpt="$1" tag="$2"
  echo "[sota-v3] Eval: $tag"
  (
    export CUDA_VISIBLE_DEVICES=0
    cd "$NAVSIM_ROOT"
    timeout 10000 $PY navsim/planning/script/run_pdm_score_cot.py \
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
  ) > "$ROOT/logs/sota_v3_eval_${tag}.log" 2>&1
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$tag"/*/*.csv 2>/dev/null | head -1)
  local pdms="N/A"
  if [[ -n "$found" ]]; then
    cp -a "$found" "$ROOT/results/raw/${tag}.csv"
    pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
  fi
  echo "[sota-v3] $tag: PDMS=$pdms (shard0)"
  echo "$pdms"
}

# ============================================================
# Main
# ============================================================
INIT="$SCORER_CKPT"
BEST_PDMS=0.0
declare -a ROUND_PDMS

# Round 1: adaptive efficiency from SFT init
BCKPT=$(train "R1" "$INIT" "$OUT_BASE")
[[ -z "$BCKPT" ]] && { echo "FATAL: Round 1 failed"; exit 1; }
P1=$(eval_one "$BCKPT" "SOTAV3_R1")
ROUND_PDMS+=("$P1")
echo "[sota-v3] Round 1 PDMS=$P1"

# Check time budget for Round 2
NOW=$(date +%s); HARD=$(date -d "today 21:00" +%s)
if [[ $NOW -lt $HARD ]]; then
  INIT_R2="$BCKPT"
  BCKPT2=$(train "R2" "$INIT_R2" "${OUT_BASE}_R2")
  if [[ -n "$BCKPT2" ]]; then
    P2=$(eval_one "$BCKPT2" "SOTAV3_R2")
    ROUND_PDMS+=("$P2")
    echo "[sota-v3] Round 2 PDMS=$P2"

    # Compare
    if [[ "$P1" != "N/A" && "$P2" != "N/A" ]]; then
      if (( $(echo "$P2 > $P1" | bc -l) )); then
        echo "[sota-v3] Round 2 IMPROVED: $P1 → $P2 (+$(echo "$P2 - $P1" | bc -l))"
      else
        echo "[sota-v3] Round 2 DID NOT IMPROVE: $P1 → $P2"
      fi
    fi
  fi
fi

echo ""
echo "=========================================="
echo "[sota-v3] SUMMARY"
for i in "${!ROUND_PDMS[@]}"; do
  echo "  Round $((i+1)): PDMS=${ROUND_PDMS[$i]}"
done
echo "=========================================="

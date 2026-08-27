#!/usr/bin/env bash
# run_baselines_parallel.sh — 并行跑 SparseVLM + PruMerge baseline full eval
# GPU1-4: SparseVLM r=0.5, GPU5-7: PruMerge r=0.5 (shard3等空闲GPU)
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
OUTDIR="$ROOT/results/raw"
LOGDIR="$ROOT/logs/baselines_20260810"
mkdir -p "$LOGDIR"

run_shard(){
  local gpu=$1 exp=$2 sh=$3 selector=$4 kr=$5
  local csv="$OUTDIR/${exp}.csv"
  local jlog="$LOGDIR/_${exp}.log"
  [[ -f "$csv" ]] && { echo "[SKIP] $exp exists" | tee -a "$LOGDIR/master.log"; return; }
  echo "[START $(date +%H:%M:%S)] GPU$gpu $exp" | tee -a "$LOGDIR/master.log"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    cd "$NAVSIM_ROOT"
    $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="navtest_local_filtered_shard${sh}_20260616_154858" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.keep_ratio="$kr" +agent.selector="$selector" \
      +agent.prune_variant=drop +agent.prune_verbose=false \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$jlog" 2>&1
  local found; found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$csv"
    echo "[DONE $(date +%H:%M:%S)] GPU$gpu $exp" | tee -a "$LOGDIR/master.log"
  else
    echo "[FAIL $(date +%H:%M:%S)] GPU$gpu $exp" | tee -a "$LOGDIR/master.log"
  fi
}

echo "=== Baselines Parallel $(date) ===" | tee "$LOGDIR/master.log"

# SparseVLM r=0.5 full on GPU1-4
for sh in 0 1 2 3; do
  run_shard "$((sh+1))" "DEF_sparsevlm_r05_sh${sh}" "$sh" "sparsevlm_text" "0.5" &
done

# PruMerge r=0.5 sh0-2 on GPU5-7 (sh3 after one finishes)
for sh in 0 1 2; do
  run_shard "$((sh+5))" "DEF_prumerge_r05_sh${sh}" "$sh" "prumerge_cls" "0.5" &
done

wait

# PruMerge sh3 on whichever GPU is free (GPU5)
run_shard 5 "DEF_prumerge_r05_sh3" 3 "prumerge_cls" "0.5"

echo "=== All Done $(date) ===" | tee -a "$LOGDIR/master.log"

#!/usr/bin/env bash
# run_pareto_extra.sh — 补全Pareto曲线：SparseVLM r=0.25, PruMerge r=0.75
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
PREFIX="navtest_local_filtered_shard"; SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw"
LOGDIR="$ROOT/logs/pareto_extra_$(date +%Y%m%d)"
mkdir -p "$LOGDIR"

log(){ echo "[pareto $(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/master.log"; }

eval_shard(){
  local gpu="$1" exp="$2" sh="$3" selector="$4" kr="$5"
  local csv="$OUTDIR/${exp}.csv"
  local jlog="$LOGDIR/_${exp}.log"
  if [[ -f "$csv" ]]; then
    log "SKIP $exp (csv exists)"
    return
  fi
  log "GPU$gpu START $exp (sel=$selector kr=$kr sh=$sh)"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    cd "$NAVSIM_ROOT"
    $PY navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="${PREFIX}${sh}${SUFFIX}" \
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
    local pdms; pdms=$($PY -c "import pandas as pd;d=pd.read_csv('$found');d=d[d['token']!='average'];print(f'{d[\"score\"].mean():.5f}')" 2>/dev/null)
    log "GPU$gpu DONE  $exp: PDMS=$pdms"
  else
    log "GPU$gpu FAIL  $exp: no CSV"
  fi
}

log "=== Pareto Extra $(date) ==="

# SparseVLM r=0.25 full 4-shard (GPU2-5)
log "=== SparseVLM r=0.25 full ==="
for sh in 0 1 2 3; do
  eval_shard "$((sh+2))" "DEF_sparsevlm_r025_sh${sh}" "$sh" "sparsevlm_text" "0.25" &
done

# PruMerge r=0.75 full 4-shard (GPU6-7 + wait)
log "=== PruMerge r=0.75 (2 shards first) ==="
eval_shard 6 "DEF_prumerge_r075_sh0" 0 "prumerge_cls" "0.75" &
eval_shard 7 "DEF_prumerge_r075_sh1" 1 "prumerge_cls" "0.75" &

wait

# PruMerge r=0.75 sh2-3
eval_shard 6 "DEF_prumerge_r075_sh2" 2 "prumerge_cls" "0.75" &
eval_shard 7 "DEF_prumerge_r075_sh3" 3 "prumerge_cls" "0.75" &

wait

log "=== ALL DONE $(date) ==="

#!/usr/bin/env bash
# run_taucut_fullnavtest.sh — τ-cut full navtest (4 shards, best τ from shard0 quick test)
#
# ONLY run this AFTER shard0 quick test passes the gate (PDMS > 0.892 at mean_kr≈0.5).
# This runs the winning τ (or top 2) across all 4 shards for the paper.
#
# Launch: nohup bash scripts/run_taucut_fullnavtest.sh <tau> <tag> > logs/_taucut_full.log 2>&1 &
# Example: nohup bash scripts/run_taucut_fullnavtest.sh -0.1487 kr050 > logs/_taucut_full.log 2>&1 &
# Stop:   touch STOP_TAUCUT
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
OUT="$ROOT/results/raw/tokenprune_taucut"; mkdir -p "$OUT" logs
STOP="$ROOT/STOP_TAUCUT"
MSE_CKPT="$ROOT/ckpt/s3_token_scorer_mse"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
NG=$(nvidia-smi -L | wc -l); [[ "$NG" -lt 1 ]] && NG=1
log(){ echo "[taucut-full $(date +%H:%M:%S)] $*"; }

# Arguments
TAU="${1:?Usage: $0 <tau> <tag>. Example: $0 -0.1487 kr050}"
TAG="${2:?Usage: $0 <tau> <tag>}"

# 4 shards (skip shard0 if already done from quick test)
JOBS=()
for s in 0 1 2 3; do JOBS+=("$TAU $TAG $s"); done

run_job(){
  local gpu="$1" tau="$2" tag="$3" sh="$4"
  local exp="TC_mse_tau_${tag}_sh${sh}"
  local csv="$OUT/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp (done)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  log "GPU$gpu start $exp (tau=$tau)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio=0.5 +agent.selector=scorer_taucut \
      +agent.scorer_ckpt="$MSE_CKPT" +agent.tau="$tau" +agent.tau_min_keep=36 \
      +agent.prune_verbose=false worker=single_machine_thread_pool worker.max_workers=1
  ) > "logs/_tc_${exp}.log" 2>&1; local rc=$?
  local c; c=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  [[ -n "$c" ]] && cp -a "$c" "$csv" && log "GPU$gpu done $exp rc=$rc -> csv" || log "GPU$gpu $exp rc=$rc WARN no csv"
}

worker(){ local gpu="$1"; local i="$gpu"
  while [[ "$i" -lt "${#JOBS[@]}" ]]; do
    [[ -f "$STOP" ]] && { log "STOP -> worker$gpu exit"; return; }
    run_job "$gpu" ${JOBS[$i]}
    i=$(( i + NG ))
  done
}

if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: pdm_score already running"; exit 1; fi
log "τ-cut full navtest: 4 shards, τ=$TAU, tag=$TAG. out=$OUT"
for g in $(seq 0 $((NG-1))); do worker "$g" & done
wait
log "τ-cut full dispatch finished. csvs: $(ls "$OUT"/TC_mse_tau_${TAG}_*.csv 2>/dev/null | wc -l)/4"

#!/usr/bin/env bash
# run_fastv_baseline.sh — FastV-selector-at-input baseline (layer-2 attention as selector)
# This is baseline #4 in design_decisions.md Q5.c:
#   "FastV-selector-at-input at matched ratio" — use LLM layer-2 attention as
#   importance score, but prune at ViT→LLM interface (same position as ours).
#   Isolates "selector quality gain" from "pruning position gain."
#
# Uses the same executor (AutoVLAWithTokenPruneAgent, Variant A attn-mask) as
# the main scorer experiments, just with selector='fastv_l2' (which internally
# sets score_layer=2).
#
# Launch: nohup bash scripts/run_fastv_baseline.sh > logs/_fastv_baseline.log 2>&1 &
# Stop:   touch STOP_FASTV
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
OUT="$ROOT/results/raw/tokenprune_S3_full"; mkdir -p "$OUT" logs
STOP="$ROOT/STOP_FASTV"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
NG=$(nvidia-smi -L | wc -l); [[ "$NG" -lt 1 ]] && NG=1
log(){ echo "[fastv $(date +%H:%M:%S)] $*"; }

# Arms: FastV-at-input (layer-2 attention) at r=0.5 and r=0.75
ARMS=("fastv_l2 0.5" "fastv_l2 0.75")
JOBS=()
for arm in "${ARMS[@]}"; do for s in 0 1 2 3; do JOBS+=("$arm $s"); done; done

run_job(){
  local gpu="$1" sel="$2" kr="$3" sh="$4"
  local krtag=$(echo "$kr" | sed 's/\.//'); local exp="MT_${sel}_r${krtag}_sh${sh}"
  local csv="$OUT/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp (done)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  log "GPU$gpu start $exp"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
      +agent.prune_verbose=false worker=single_machine_thread_pool worker.max_workers=1
  ) > "logs/_mt_${exp}.log" 2>&1; local rc=$?
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

if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: pdm_score already running (wait for scorer to finish)"; exit 1; fi
log "FastV baseline dispatch: ${#JOBS[@]} jobs over $NG GPUs. out=$OUT"
for g in $(seq 0 $((NG-1))); do worker "$g" & done
wait
log "FastV baseline dispatch finished. csvs: $(ls "$OUT"/MT_fastv_*.csv 2>/dev/null | wc -l)/${#JOBS[@]}"

#!/usr/bin/env bash
# run_s3_maintable_full_navtest.sh — resumable main-table dispatcher on FULL
# navtest (4 shards). Queue = selector x ratio x shard; round-robin over N GPUs.
# Resumable: skips any job whose harvested CSV already exists (SKIP_DONE), so a
# window boundary + memory loss just re-runs this script to continue.
# Stop: touch STOP_S3.  Launch: nohup bash scripts/run_s3_maintable_full_navtest.sh > logs/_s3_maintable.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"; SCORER="$ROOT/ckpt/s3_token_scorer"
OUT="$ROOT/results/raw/tokenprune_S3_full"; mkdir -p "$OUT" logs
STOP="$ROOT/STOP_S3"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
NG=$(nvidia-smi -L | wc -l); [[ "$NG" -lt 1 ]] && NG=1
log(){ echo "[mt $(date +%H:%M:%S)] $*"; }

# job list: "selector ratio" (r=1.0 -> selector attn_L12 == no-prune ref)
# ARM order can be overridden via env ARMS_SPEC (";"-separated), e.g. to
# prioritize cheap essential arms before a deadline/reclaim.
DEFAULT_ARMS="scorer 0.5;scorer 0.25;scorer 0.75;attn_L12 0.5;random 0.5;attn_L12 1.0"
IFS=';' read -ra ARMS <<< "${ARMS_SPEC:-$DEFAULT_ARMS}"
JOBS=()
for arm in "${ARMS[@]}"; do for s in 0 1 2 3; do JOBS+=("$arm $s"); done; done

run_job(){
  local gpu="$1" sel="$2" kr="$3" sh="$4"
  local krtag=$(echo "$kr" | sed 's/\.//'); local exp="MT_${sel}_r${krtag}_sh${sh}"
  local csv="$OUT/${exp}.csv"
  [[ -f "$csv" ]] && { log "SKIP $exp (done)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  local extra=""; [[ "$sel" == "scorer" ]] && extra="+agent.scorer_ckpt=$SCORER"
  log "GPU$gpu start $exp"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
      +agent.prune_variant=drop $extra \
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

if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: pdm_score already running"; exit 1; fi
log "main-table dispatch: ${#JOBS[@]} jobs over $NG GPUs. out=$OUT"
for g in $(seq 0 $((NG-1))); do worker "$g" & done
wait
log "main-table dispatch finished. csvs: $(ls "$OUT"/*.csv 2>/dev/null | wc -l)/${#JOBS[@]}"

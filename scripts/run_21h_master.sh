#!/usr/bin/env bash
# run_21h_master.sh — 21h window parallel dispatcher (8 H20)
# Reads a queue file: each line = gpu|exp|shard|keep_ratio|selector|tau|scorer_ckpt
#   - gpu: 0-7 (CUDA_VISIBLE_DEVICES)
#   - exp: experiment_name
#   - shard: 0-3
#   - keep_ratio: e.g. 0.5 / 0.75
#   - selector: attn_L12|random|scorer|scorer_taucut|fastv_l2|sparsevlm_text|prumerge_cls
#   - tau: numeric tau for scorer_taucut, else "none"
#   - scorer_ckpt: absolute ckpt dir, or "none"
# Lines starting with # and blank lines are skipped.
# Stop: touch STOP_21H
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
LOGDIR="$ROOT/logs/_21h"; mkdir -p "$LOGDIR"
QUEUE="${1:-$ROOT/_21h_queue.txt}"
STOP="$ROOT/STOP_21H"
log(){ echo "[21h $(date +%H:%M:%S)] $*"; }

# fact-check guard: do not launch if a pdm_score is already running (manual pgrep before call)
if pgrep -f run_pdm_score_cot >/dev/null; then log "ABORT: run_pdm_score_cot already running (avoid double-launch)"; exit 1; fi
[[ ! -f "$QUEUE" ]] && { log "ABORT: queue $QUEUE not found"; exit 1; }

run_one(){
  local gpu="$1" exp="$2" sh="$3" kr="$4" sel="$5" tau="$6" ckpt="${7:-none}"
  local csv
  case "$sel" in
    scorer_taucut) csv="$ROOT/results/raw/tokenprune_taucut/${exp}.csv" ;;
    *) csv="$ROOT/results/raw/tokenprune_S3_full/${exp}.csv" ;;
  esac
  [[ -f "$csv" ]] && { log "SKIP $exp (csv exists)"; return; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return; }
  local tauarg=""; [[ "$tau" != "none" && -n "$tau" ]] && tauarg="+agent.tau=$tau +agent.tau_min_keep=36"
  local ckptarg=""; [[ "$ckpt" != "none" && -n "$ckpt" ]] && ckptarg="+agent.scorer_ckpt=$ckpt"
  log "GPU$gpu START $exp (kr=$kr sel=$sel tau=$tau)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout 40000 "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
      $tauarg $ckptarg +agent.prune_verbose=false worker=single_machine_thread_pool worker.max_workers=1
  ) > "$LOGDIR/_${exp}.log" 2>&1; local rc=$?
  local c; c=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$c" ]]; then
    cp -a "$c" "$csv"; log "GPU$gpu DONE $exp rc=$rc -> $csv"
    "$PY" -c "import pandas as pd; df=pd.read_csv('$csv'); df=df[df['token']!='average']; print(f'[PDMS] $exp N={len(df)} PDMS={df[\"score\"].mean():.6f}')" 2>/dev/null || true
  else
    log "GPU$gpu $exp rc=$rc WARN no csv (see $LOGDIR/_${exp}.log)"
  fi
}

declare -A GPU_JOBS
while IFS='|' read -r gpu exp sh kr sel tau ckpt; do
  [[ -z "$gpu" || "$gpu" == \#* ]] && continue
  GPU_JOBS[$gpu]+="$exp|$sh|$kr|$sel|$tau|$ckpt"$'\n'
done < "$QUEUE"

log "Dispatching queue=$QUEUE over GPUs with jobs: $(echo "${GPU_JOBS[@]}" | grep -c '|')"
for gpu in $(seq 0 7); do
  ( while IFS='|' read -r exp sh kr sel tau ckpt; do
      [[ -z "$exp" ]] && continue
      run_one "$gpu" "$exp" "$sh" "$kr" "$sel" "$tau" "$ckpt"
    done <<< "${GPU_JOBS[$gpu]:-}" ) &
done
wait
log "ALL 21h jobs finished. taucut_csvs=$(ls "$ROOT/results/raw/tokenprune_taucut"/TC_*.csv 2>/dev/null | wc -l); mse_csvs=$(ls "$ROOT/results/raw/tokenprune_S3_full"/MT_scorer_mse*.csv 2>/dev/null | wc -l)"

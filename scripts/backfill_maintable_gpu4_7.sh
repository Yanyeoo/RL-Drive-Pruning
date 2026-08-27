#!/usr/bin/env bash
# backfill_maintable_gpu4_7.sh — Unattended main-table backfill on idle GPUs.
#
# Completes sh1/sh2/sh3 for four families whose only shard0 exists, using each
# family's EXACT original sh0 config (Variant-B drop, raw, NO denylist / NO
# fallback). Aggregates each family to full navtest when its 4 shards exist.
#
# Design:
#   - Idempotent: skips any (family,shard) whose CSV already exists.
#   - Polite: by default WAITS until the Safe-HTPO main line reaches its
#     EVALUATING stage (GPU 0-3 busy) and then uses GPU 4-7. Override with
#     BACKFILL_GPUS / BACKFILL_WAIT_FOR_EVAL.
#   - Safe: refuses to double-launch; never touches the training/eval main line.
#   - Long-lived: can run unattended; writes its own status + logs.
#
# Launch:
#   nohup bash scripts/backfill_maintable_gpu4_7.sh \
#     > logs/backfill_maintable/nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -Eeuo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"
NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"
SHARD_SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw/tokenprune_S3_full"
AGGDIR="$ROOT/results/raw/maintable_backfill"
LOGDIR="$ROOT/logs/backfill_maintable"
STATUS="$LOGDIR/status.json"
LOCK="$ROOT/logs/backfill_maintable.lock"
STOP="$ROOT/STOP_BACKFILL"
mkdir -p "$OUTDIR" "$AGGDIR" "$LOGDIR"

SFT_CKPT="$ROOT/ckpt/s3_token_scorer"
RL_CKPT="$ROOT/ckpt/s3_token_scorer_rl_shaped_20260721_174549_sh0/ckpt_best"

# Config: which GPUs to use, and whether to wait for the main-line eval stage.
BACKFILL_GPUS="${BACKFILL_GPUS:-4 5 6 7}"
BACKFILL_WAIT_FOR_EVAL="${BACKFILL_WAIT_FOR_EVAL:-1}"
BACKFILL_POLL_SEC="${BACKFILL_POLL_SEC:-120}"
BACKFILL_MAX_WAIT_SEC="${BACKFILL_MAX_WAIT_SEC:-43200}"   # 12h safety cap on waiting
PER_JOB_TIMEOUT="${PER_JOB_TIMEOUT:-40000}"
GPU_ARR=($BACKFILL_GPUS)
NG=${#GPU_ARR[@]}

log() { echo "[backfill $(date '+%F %T')] $*" | tee -a "$LOGDIR/backfill.log"; }
write_status() {
  cat > "$STATUS" <<EOF
{"stage":"$1","time":"$(date -Iseconds)","message":"$2"}
EOF
}
fail() { write_status "FAILED" "$1"; log "FATAL: $1"; exit 1; }
trap 'rc=$?; [[ $rc -ne 0 ]] && write_status "FAILED" "exited rc=$rc"; exit $rc' ERR

# --- Single-instance lock ---
if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another backfill instance is alive (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# --- Preflight: verify pinned checkpoints exist ---
[[ -f "$SFT_CKPT/checkpoint.pt" ]] || fail "SFT scorer ckpt missing: $SFT_CKPT"
[[ -f "$RL_CKPT/checkpoint.pt" ]] || fail "RL shaped ckpt missing: $RL_CKPT"

# --- Family table: name|selector|keep_ratio|scorer_ckpt|extra hydra args ---
# All families use prune_variant=drop, no denylist, no safety_net (raw).
FAMILIES=(
  "varBsafe_scorer_r075|scorer|0.75|$SFT_CKPT|"
  "rl_shaped_r025|scorer|0.25|$RL_CKPT|"
  "rl_shaped_r075|scorer|0.75|$RL_CKPT|"
  "rl_taucut_kr060|scorer_taucut|0.5|$RL_CKPT|+agent.tau=-0.1668 +agent.tau_min_keep=36"
)

# Build the pending (family,shard) job list (shards 1..3; sh0 already exists).
JOBS=()
for entry in "${FAMILIES[@]}"; do
  IFS='|' read -r name sel kr ckpt extra <<< "$entry"
  for SH in 1 2 3; do
    CSV="$OUTDIR/MT_${name}_sh${SH}.csv"
    [[ -f "$CSV" ]] && { log "SKIP existing $name sh$SH"; continue; }
    JOBS+=("${name}|${sel}|${kr}|${ckpt}|${extra}|${SH}")
  done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  log "Nothing to backfill; all sh1/2/3 already present. Proceeding to aggregation."
else
  # --- Optionally wait until the main-line eval stage (so GPU 4-7 are the idle ones) ---
  if [[ "$BACKFILL_WAIT_FOR_EVAL" == "1" ]]; then
    write_status "WAITING" "Waiting for Safe-HTPO main line to reach EVALUATING"
    waited=0
    ACTIVE_OUT="$(cat "$ROOT/logs/safe_htpo_active_outdir.txt" 2>/dev/null || true)"
    RID="$(basename "${ACTIVE_OUT:-}")"
    while true; do
      [[ -f "$STOP" ]] && fail "STOP_BACKFILL present before start"
      st="$(cat "$ROOT/logs/${RID}/status.json" 2>/dev/null | grep -o '"stage":"[^"]*"' | cut -d'"' -f4 || true)"
      if [[ "$st" == "EVALUATING" || "$st" == "DONE" || "$st" == "FAILED" ]]; then
        log "Main line stage=$st; starting backfill on GPU: $BACKFILL_GPUS"
        break
      fi
      # Also start if the designated GPUs are actually idle regardless of stage.
      idle=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -v g="$BACKFILL_GPUS" 'BEGIN{split(g,G," ");for(i in G)want[G[i]]=1} want[$1]==1 && $2+0<1024{n++} END{print n+0}')
      if [[ "${idle:-0}" -ge "$NG" ]]; then
        log "Designated GPUs ($BACKFILL_GPUS) all idle; starting backfill (main stage=$st)"
        break
      fi
      (( waited += BACKFILL_POLL_SEC ))
      if (( waited > BACKFILL_MAX_WAIT_SEC )); then
        fail "Timed out waiting ${BACKFILL_MAX_WAIT_SEC}s for eval stage / idle GPUs"
      fi
      log "main stage=${st:-unknown}; idle designated GPUs=${idle:-0}/$NG; wait ${waited}s"
      sleep "$BACKFILL_POLL_SEC"
    done
  fi

  write_status "RUNNING" "Backfilling ${#JOBS[@]} (family,shard) jobs on GPU $BACKFILL_GPUS"

  run_one() {
    local gpu="$1" name="$2" sel="$3" kr="$4" ckpt="$5" extra="$6" sh="$7"
    local exp="MT_${name}_sh${sh}"
    local csv="$OUTDIR/${exp}.csv"
    local jlog="$LOGDIR/_${exp}.log"
    [[ -f "$csv" ]] && { log "SKIP $exp (csv exists)"; return 0; }
    [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return 0; }
    log "GPU$gpu START $exp (sel=$sel kr=$kr ckpt=$(basename "$(dirname "$ckpt")")/$(basename "$ckpt"))"
    ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
      timeout --signal=TERM --kill-after=5m "$PER_JOB_TIMEOUT" \
        "$PY" navsim/planning/script/run_pdm_score_cot.py \
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
        +agent.keep_ratio="$kr" \
        +agent.selector="$sel" \
        +agent.scorer_ckpt="$ckpt" \
        +agent.prune_variant=drop \
        +agent.prune_verbose=false \
        $extra \
        worker=single_machine_thread_pool worker.max_workers=1
    ) > "$jlog" 2>&1
    local rc=$?
    local found
    found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
      cp -a "$found" "$csv"
      local n=$(($(wc -l < "$csv")-1))
      log "GPU$gpu DONE $exp rc=$rc rows=$n -> $csv"
    else
      log "GPU$gpu WARN $exp rc=$rc: no CSV produced (see $jlog)"
    fi
  }

  # Static round-robin: worker w handles jobs at indices w, w+NG, w+2NG, ...
  worker() {
    local w="$1" gpu="${GPU_ARR[$1]}" i="$1"
    while [[ "$i" -lt "${#JOBS[@]}" ]]; do
      [[ -f "$STOP" ]] && { log "STOP -> worker$gpu exit"; return; }
      IFS='|' read -r name sel kr ckpt extra sh <<< "${JOBS[$i]}"
      run_one "$gpu" "$name" "$sel" "$kr" "$ckpt" "$extra" "$sh"
      i=$(( i + NG ))
    done
  }

  for w in $(seq 0 $((NG-1))); do worker "$w" & done
  wait
  log "All backfill jobs dispatched/finished."
fi

# --- Aggregate each family to full navtest (raw only, no fallback) ---
write_status "AGGREGATING" "Aggregating completed families to full navtest"
"$PY" "$ROOT/scripts/backfill_aggregate.py" \
  --result-dir "$OUTDIR" --out-dir "$AGGDIR" \
  --families varBsafe_scorer_r075 rl_shaped_r025 rl_shaped_r075 rl_taucut_kr060 \
  | tee "$AGGDIR/backfill_summary.txt" || fail "Aggregation failed"

write_status "DONE" "Backfill complete; see $AGGDIR/backfill_summary.txt"
log "DONE. Summary: $AGGDIR/backfill_summary.txt"

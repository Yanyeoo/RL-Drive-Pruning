#!/usr/bin/env bash
# run_maintable_overnight.sh — Unattended Table-1 completion, runs until the
# 2026-07-29 GPU reclaim (hard stop 13:45, 15 min before 14:00).
#
# Pipeline (each phase idempotent, table regenerated after every job so a
# mid-run kill always leaves an up-to-date draft):
#   Phase A  : wait for the already-running RL backfill (rl_shaped_r075 sh3,
#              rl_taucut_kr060 sh1/2/3) to finish. Do NOT relaunch it.
#   Phase B  : baseline drop shard0 placeholders + SFT-ours r=0.25 shard0.
#   Phase C  : best-effort 7B ImpromptuVLA/nuScenes (non-blocking; on failure,
#              write a report and continue). Never changes research protocol.
#   Phase B2 : upgrade the shard0 cells to full navtest (sh1/2/3), priority
#              r=0.5 -> r=0.75 -> r=0.25, until the deadline.
#   Finalize : regenerate Table-1 draft + status DONE.
#
# All eval is Variant-B true token drop. Baselines = raw scores, no denylist.
# Ours = scorer + denylist + safety_net (its raw CSV; +fallback applied by the
# table generator post-hoc, reproducing the 0.9045 headline).
#
# Launch:
#   nohup bash scripts/run_maintable_overnight.sh \
#     > logs/maintable_overnight/nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh" >/dev/null
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"

CKPT="$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
YAML="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SENSOR="$ROOT/data/navsim_v2_local"
SHARD_PREFIX="navtest_local_filtered_shard"; SHARD_SUFFIX="_20260616_154858"
OUTDIR="$ROOT/results/raw/tokenprune_S3_full"
LOGDIR="$ROOT/logs/maintable_overnight"; STATUS="$LOGDIR/status.json"
LOCK="$ROOT/logs/maintable_overnight.lock"; STOP="$ROOT/STOP_MAINTABLE"
SFT_CKPT="$ROOT/ckpt/s3_token_scorer"
DENYLIST="$ROOT/results/varB_catastrophic_tokens.json"
GEN="$ROOT/scripts/gen_table1_draft.py"
TABLE="$ROOT/results/table1_draft.md"
mkdir -p "$OUTDIR" "$LOGDIR"

GPUS=(0 1 2 3); NG=${#GPUS[@]}
DEADLINE_EPOCH=$(date -d "2026-07-29 13:45:00" +%s)
SHARD0_NEED=4200      # need >70min headroom to launch a shard0 (~1h) job
FULLSH_NEED=13000     # need >3.6h headroom to launch a single full-navtest shard (~3.3h)

log(){ echo "[maintbl $(date '+%F %T')] $*" | tee -a "$LOGDIR/run.log"; }
write_status(){ printf '{"stage":"%s","time":"%s","message":"%s"}\n' "$1" "$(date -Iseconds)" "$2" > "$STATUS"; }
time_left(){ echo $(( DEADLINE_EPOCH - $(date +%s) )); }
regen(){ "$PY" "$GEN" --out "$TABLE" >/dev/null 2>>"$LOGDIR/gen.log" && log "table regenerated -> $TABLE" || log "WARN table regen failed (see gen.log)"; }

# --- single-instance lock ---
if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another instance alive (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# ---------------------------------------------------------------------------
# run_eval: one (selector,keep_ratio,shard) job. mode=baseline|ours.
# ---------------------------------------------------------------------------
run_eval(){
  local gpu="$1" sel="$2" kr="$3" exp_base="$4" sh="$5" mode="$6"
  local exp="${exp_base}_sh${sh}"
  local csv="$OUTDIR/${exp}.csv"; local jlog="$LOGDIR/_${exp}.log"
  [[ -f "$csv" ]] && { log "SKIP $exp (exists)"; return 0; }
  [[ -f "$STOP" ]] && { log "STOP -> skip $exp"; return 0; }
  local extra=""
  if [[ "$mode" == "ours" ]]; then
    extra="+agent.scorer_ckpt=$SFT_CKPT +agent.varB_denylist=$DENYLIST +agent.safety_net=true"
  fi
  # per-job timeout: min(base, remaining-300)
  local tl; tl=$(time_left); local to=$(( tl - 300 )); [[ $to -gt 20000 ]] && to=20000
  [[ $to -lt 300 ]] && { log "DEADLINE -> skip $exp (tl=${tl}s)"; return 0; }
  log "GPU$gpu START $exp (sel=$sel kr=$kr mode=$mode timeout=${to}s)"
  ( cd "$NAVSIM_ROOT"; export CUDA_VISIBLE_DEVICES="$gpu"
    timeout --signal=TERM --kill-after=3m "$to" "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="${SHARD_PREFIX}${sh}${SHARD_SUFFIX}" \
      metric_cache_path="$ROOT/data/navtest_metric_cache" \
      +json_data_path="$ROOT/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$YAML" +agent.checkpoint_path="$CKPT" +agent.sensor_data_path="$SENSOR" \
      +agent.codebook_cache_path="$AUTOVLA_ROOT/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false +agent.keep_ratio="$kr" +agent.selector="$sel" \
      +agent.prune_variant=drop $extra +agent.prune_verbose=false \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "$jlog" 2>&1
  local rc=$? found
  found=$(ls -t "$NAVSIM_EXP_ROOT/$exp"/*/*.csv 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    cp -a "$found" "$csv"
    log "GPU$gpu DONE $exp rc=$rc rows=$(($(wc -l < "$csv")-1)) -> $csv"; regen
  else
    log "GPU$gpu WARN $exp rc=$rc: no CSV (see $jlog)"
  fi
}

# wait until a physical GPU is idle (<1GiB used) so we never double-book a GPU
# still running the kept rl_shaped_r075_sh3 job.
gpu_wait_idle(){
  local g="$1"
  while :; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
    [[ -z "$used" ]] && return 0                 # no telemetry -> assume usable
    [[ "${used:-0}" -lt 1024 ]] && return 0
    [[ -f "$STOP" ]] && return 1
    [[ $(time_left) -lt 300 ]] && return 1
    log "GPU$g busy (${used}MiB) -> wait"; sleep 120
  done
}
# dispatch a flat job list "sel|kr|exp_base|sh|mode" over GPUS round-robin
dispatch(){
  local -n JL=$1; local minhead=$2
  worker(){ local w="$1" i="$1"
    while [[ $i -lt ${#JL[@]} ]]; do
      [[ -f "$STOP" ]] && { log "STOP -> worker${GPUS[$w]} exit"; return; }
      [[ $(time_left) -lt $minhead ]] && { log "DEADLINE -> worker${GPUS[$w]} stop (tl=$(time_left)s)"; return; }
      gpu_wait_idle "${GPUS[$w]}" || { log "worker${GPUS[$w]} stop (gpu wait aborted)"; return; }
      IFS='|' read -r sel kr eb sh md <<< "${JL[$i]}"
      run_eval "${GPUS[$w]}" "$sel" "$kr" "$eb" "$sh" "$md"
      i=$(( i + NG ))
    done; }
  local pids=(); for w in $(seq 0 $((NG-1))); do worker "$w" & pids+=($!); done
  wait "${pids[@]}"
}

# ===========================================================================
log "=== run_maintable_overnight START (deadline $(date -d @${DEADLINE_EPOCH} '+%F %T'), tl=$(time_left)s) ==="
[[ -f "$STOP" ]] && { log "STOP present at start; abort"; write_status "ABORTED" "STOP present"; exit 0; }

# ---- Phase A: wait for the running RL backfill to finish -------------------
# Scope (2026-07-28 22:05 user ruling): taucut and SFT r=0.25 DROPPED. Only the
# already-running rl_shaped_r075 sh3 is kept from phase A (near-free full row).
# GPU重心 = baseline four methods (FastV/Random/PruMerge/SparseVLM), all drop.
# We do NOT wait for A to finish; the lone rl_shaped_r075_sh3 keeps GPU3 busy and
# the dispatch below simply runs baselines on whatever GPUs are free (idempotent,
# per-GPU serial). run_pdm_score_cot on an already-busy GPU would OOM, so phase B
# starts only on GPUs that are currently idle; the RL job's GPU joins when free.
write_status "WAIT_A" "keeping rl_shaped_r075_sh3; not waiting (baselines start on idle GPUs)"
log "Phase A: rl_shaped_r075_sh3 left running (kept); taucut + SFT r=0.25 dropped per user"
regen

# ---- Phase B: BASELINE drop shard0 placeholders (four methods only) --------
# Scope = FastV / Random / PruMerge / SparseVLM, all drop, raw (no denylist).
# random r=0.5 and sparsevlm r=0.75 already have full drop CSVs -> not re-run.
# ours (SFT/RL fixed-ratio + taucut) are NOT the headline and are NOT re-run.
write_status "PHASE_B" "baseline drop shard0 placeholders (fastv/random/prumerge/sparsevlm)"
B_JOBS=(
  "fastv_l2|0.5|MT_fastv_l2_drop_r05|0|baseline"
  "random|0.5|MT_random_drop_r05|0|baseline"
  "prumerge_cls|0.5|MT_prumerge_cls_drop_r05|0|baseline"
  "sparsevlm_text|0.5|MT_sparsevlm_text_drop_r05|0|baseline"
  "fastv_l2|0.75|MT_fastv_l2_drop_r075|0|baseline"
  "prumerge_cls|0.75|MT_prumerge_cls_drop_r075|0|baseline"
  "random|0.75|MT_random_drop_r075|0|baseline"
  "fastv_l2|0.25|MT_fastv_l2_drop_r025|0|baseline"
  "random|0.25|MT_random_drop_r025|0|baseline"
  "prumerge_cls|0.25|MT_prumerge_cls_drop_r025|0|baseline"
  "sparsevlm_text|0.25|MT_sparsevlm_text_drop_r025|0|baseline"
)
dispatch B_JOBS "$SHARD0_NEED"
regen; log "Phase B done"

# ---- Phase C: best-effort 7B (non-blocking) --------------------------------
write_status "PHASE_C" "best-effort 7B ImpromptuVLA/nuScenes"
if [[ $(time_left) -gt $SHARD0_NEED ]]; then
  bash "$ROOT/scripts/run_7b_besteffort.sh" > "$LOGDIR/_7b_besteffort.log" 2>&1 || \
    log "Phase C: 7B best-effort returned non-zero (see _7b_besteffort.log) -> continuing"
  log "Phase C finished (best-effort)"
else
  log "Phase C skipped (deadline)"
fi

# ---- Phase B2: upgrade baseline shard0 cells to full navtest (sh1/2/3) ------
# Priority r=0.5 (central op point) -> r=0.75 -> r=0.25; shards 1,2,3.
write_status "PHASE_B2" "upgrading baseline cells to full navtest"
B2_JOBS=()
for tag in "0.5:MT_fastv_l2_drop_r05:fastv_l2" \
           "0.5:MT_random_drop_r05:random" \
           "0.5:MT_prumerge_cls_drop_r05:prumerge_cls" \
           "0.5:MT_sparsevlm_text_drop_r05:sparsevlm_text" \
           "0.75:MT_fastv_l2_drop_r075:fastv_l2" \
           "0.75:MT_random_drop_r075:random" \
           "0.75:MT_prumerge_cls_drop_r075:prumerge_cls" \
           "0.25:MT_fastv_l2_drop_r025:fastv_l2" \
           "0.25:MT_random_drop_r025:random" \
           "0.25:MT_prumerge_cls_drop_r025:prumerge_cls" \
           "0.25:MT_sparsevlm_text_drop_r025:sparsevlm_text"; do
  IFS=':' read -r kr eb sel <<< "$tag"
  for sh in 1 2 3; do B2_JOBS+=("$sel|$kr|$eb|$sh|baseline"); done
done
dispatch B2_JOBS "$FULLSH_NEED"
regen; log "Phase B2 done (or deadline-truncated)"

# ---- Finalize --------------------------------------------------------------
regen
write_status "DONE" "Table-1 draft at $TABLE"
log "=== ALL DONE. Table draft: $TABLE ==="

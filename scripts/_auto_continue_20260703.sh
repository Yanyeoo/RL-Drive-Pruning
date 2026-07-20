#!/usr/bin/env bash
# ============================================================================
# _auto_continue_20260703.sh — post-17:50 unattended "next step" (SAFE scope).
# Written 2026-07-03 17:3x before session memory loss. See HANDOFF doc:
#   docs/journal/HANDOFF_2026-07-03_next_step.md
#
# What it does (and ONLY this — deliberately conservative):
#   1. wait for the in-flight L0K4 landscape run to finish (pgrep loop)
#   2. final landscape harvest + cp -a backup
#   3. S1 GPU smoke verify of the NEW token-prune agent on navtest_smoke5_shard0:
#        (a) keep_ratio=1.0  -> must run rc=0 + produce pdms (lossless no-op path)
#        (b) keep_ratio=0.5  -> must run rc=0 + produce pdms (prune path exercised)
#   4. write logs/_s1_verify_marker.txt with PASS/FAIL/ERROR + the two pdms
#   It does NOT run S2 (needs a new sweep script + human confirm of lossless).
#
# Stop: touch STOP_AUTO
# Launch: nohup bash scripts/_auto_continue_20260703.sh > logs/_auto_continue_20260703.log 2>&1 &
# ============================================================================
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="${ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"
STOP="${ROOT}/STOP_AUTO"
MARKER="${ROOT}/logs/_s1_verify_marker.txt"
DEADLINE=$(date -d '2026-07-04 06:00' +%s)   # give L0K4 ample time; safety cap
mkdir -p logs

log(){ echo "[auto $(date +%m-%d\ %H:%M:%S)] $*"; }

log "auto-continue start. waiting for L0K4 (run_pdm_score_cot) to finish ..."
while pgrep -f run_pdm_score_cot >/dev/null; do
  [[ -f "$STOP" ]] && { log "STOP_AUTO; abort before verify."; exit 0; }
  (( $(date +%s) >= DEADLINE )) && { log "deadline; abort wait."; exit 0; }
  sleep 120
done
log "L0K4 done. GPUs should be free."

# ---- final landscape harvest + backup (safe, proven) ----
"$PY" scripts/plot_layer_prunability_landscape.py >/dev/null 2>&1 || true
"$PY" scripts/plot_magnitude_vs_prunability.py     >/dev/null 2>&1 || true
BK="backups/auto_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$BK/aggregates"
for d in results/raw/M1b_freelunch_L*_*; do
  [[ -s "$d/aggregate.json" ]] && cp -a "$d/aggregate.json" "$BK/aggregates/$(basename $d).aggregate.json"
done
cp -a docs/results/key_results.md docs/results/figures/landscape_data.json "$BK/" 2>/dev/null || true
log "landscape harvested + backed up -> $BK"

[[ -f "$STOP" ]] && { log "STOP_AUTO; skip S1 verify."; exit 0; }

# ---- S1 GPU smoke verify (new token-prune agent) ----
source "${ROOT}/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="${ROOT}/code:${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"
CKPT="${ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt"
QWEN_TRAIN_YAML="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"
SMOKE="navtest_smoke5_shard0_20260616_154725"

run_one(){  # $1=keep_ratio $2=exp_name
  local kr="$1" exp="$2" rc=0
  ( cd "$NAVSIM_ROOT"
    CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 1800s \
    "$PY" navsim/planning/script/run_pdm_score_cot.py \
      experiment_name="$exp" \
      train_test_split="$SMOKE" \
      metric_cache_path="${ROOT}/data/navtest_metric_cache" \
      +json_data_path="${ROOT}/data/navtest_nocot" \
      agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
      +agent.config_path="$QWEN_TRAIN_YAML" \
      +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="${ROOT}/data/navsim_v2_local" \
      +agent.codebook_cache_path="${AUTOVLA_ROOT}/codebook_cache/agent_vocab.pkl" \
      +agent.lora_conf.use_lora=false \
      +agent.keep_ratio="$kr" +agent.selector=attn_L12 +agent.prune_verbose=true \
      worker=single_machine_thread_pool worker.max_workers=1
  ) > "logs/_s1_verify_kr${kr}.log" 2>&1 || rc=$?
  echo "$rc"
}

pdms_of(){  # $1=exp_name  -> mean score over valid, or "NA"
  "$PY" - "$1" <<'PY' 2>/dev/null || echo "NA"
import sys, os, glob
import pandas as pd
exp=sys.argv[1]; root=os.environ.get("NAVSIM_EXP_ROOT","")
csvs=glob.glob(os.path.join(root, exp, "**", "*.csv"), recursive=True)
if not csvs: print("NA"); raise SystemExit
m=pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
m=m[m["token"]!="average"]
v=m[m["valid"]] if "valid" in m else m
print(round(float(v["score"].mean()),5) if len(v) else "NA")
PY
}

log "S1 smoke: keep_ratio=1.0 (lossless no-op path) ..."
RC10=$(run_one 1.0 "S1_verify_r10_smoke_$(date +%H%M)")
P10=$(pdms_of "S1_verify_r10_smoke_$(date +%H%M)")
[[ -f "$STOP" ]] && { log "STOP_AUTO after r1.0."; exit 0; }
log "S1 smoke: keep_ratio=0.5 (prune path) ..."
RC05=$(run_one 0.5 "S1_verify_r05_smoke_$(date +%H%M)")
P05=$(pdms_of "S1_verify_r05_smoke_$(date +%H%M)")

# NOTE: exp names use $(date +%H%M) twice; pdms_of may miss if minute rolled.
# The per-run logs logs/_s1_verify_kr*.log are the source of truth regardless.

VERDICT="FAIL"
if [[ "$RC10" == "0" && "$RC05" == "0" ]]; then VERDICT="PASS(ran; verify pdms in logs)"; fi
{
  echo "=== S1 GPU verify marker  $(date -Iseconds) ==="
  echo "keep_ratio=1.0 : rc=$RC10  pdms=$P10   log=logs/_s1_verify_kr1.0.log"
  echo "keep_ratio=0.5 : rc=$RC05  pdms=$P05   log=logs/_s1_verify_kr0.5.log"
  echo "VERDICT: $VERDICT"
  echo "next: (1) confirm r=1.0 pdms == B0 smoke (lossless); (2) r=0.5 pdms differs & rc=0 (prune works);"
  echo "      (3) if both ok -> write scripts/run_tokenprune_sweep.sh + run S2 gate (see HANDOFF §3.B/C)."
  echo "      if rc!=0 -> read the kr logs, fix agent/hydra wiring, DO NOT run S2."
} > "$MARKER"
log "S1 verify done. marker -> $MARKER (rc10=$RC10 rc05=$RC05)"
log "auto-continue STOP here (S2 needs new sweep script + human confirm)."

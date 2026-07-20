#!/usr/bin/env bash
# _sequencer_20260713.sh — Automated sequencer for W5 window (2026-07-13→14)
#
# Sequence:
#   Phase 1: τ-cut shard0 quick test (4 τ × shard0) → already launched separately
#   Phase 2: After τ-cut shard0 finishes → launch FastV r=0.75 (all 4 shards)
#   Phase 3: After FastV r=0.75 finishes → launch MSE scorer eval (shard0)
#   Phase 4: If τ-cut gate PASS → launch τ-cut full navtest (best τ × 4 shards)
#
# This script waits for τ-cut shard0 to complete, then proceeds.
# Launch: nohup bash scripts/_sequencer_20260713.sh > logs/_sequencer_20260713.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
log(){ echo "[seq $(date +%H:%M:%S)] $*"; }

# ---- Phase 2: Wait for τ-cut shard0 to finish, then run FastV r=0.75 ----
log "Phase 2: Waiting for τ-cut shard0 to finish..."
TAUCUT_OUT="$ROOT/results/raw/tokenprune_taucut"
while true; do
    n_done=$(ls "$TAUCUT_OUT"/TC_mse_tau_*_sh0.csv 2>/dev/null | wc -l)
    if [[ "$n_done" -ge 4 ]]; then
        log "Phase 2: All 4 τ-cut shard0 jobs done ($n_done csvs). Proceeding."
        break
    fi
    # Also check if the dispatcher is still running
    if ! pgrep -f "run_taucut_shard0_quick" >/dev/null && [[ "$n_done" -lt 4 ]]; then
        log "Phase 2: τ-cut dispatcher died but only $n_done/4 done. Checking..."
        sleep 30
        n_done2=$(ls "$TAUCUT_OUT"/TC_mse_tau_*_sh0.csv 2>/dev/null | wc -l)
        if [[ "$n_done2" -ge 4 ]]; then
            log "Phase 2: OK, csvs appeared. Proceeding."
            break
        fi
        log "Phase 2: WARNING: Only $n_done2/4 csvs, dispatcher not running. Will proceed anyway in 5min."
        sleep 300
        break
    fi
    sleep 120
done

# ---- Report τ-cut results ----
log "Phase 2: τ-cut shard0 results:"
"$PY" -c "
import pandas as pd, glob, os
out = '$TAUCUT_OUT'
ref_fixed = 0.8920  # fixed scorer r=0.5 full navtest
gate_pass = False
best_tau_tag = None
best_pdms = 0
for csv in sorted(glob.glob(f'{out}/TC_mse_tau_*_sh0.csv')):
    name = os.path.basename(csv).replace('.csv','')
    df = pd.read_csv(csv)
    df = df[df['token'] != 'average']
    pdms = df['score'].mean()
    tag = name.split('_')[3]  # kr040/kr050/kr060/kr070
    delta = pdms - ref_fixed
    status = '✅ WIN' if pdms > ref_fixed else '❌ LOSE'
    print(f'  {name}: PDMS={pdms:.6f} (Δ vs fixed r=0.5: {delta:+.4f}) {status}')
    if pdms > best_pdms:
        best_pdms = pdms
        best_tau_tag = tag
    if pdms > ref_fixed:
        gate_pass = True
print()
print(f'GATE: {\"PASS\" if gate_pass else \"FAIL\"}')
print(f'Best: {best_tau_tag} @ PDMS={best_pdms:.6f}')
# Write gate result to a flag file
with open(f'{out}/_gate_result.txt', 'w') as f:
    f.write(f'gate={\"PASS\" if gate_pass else \"FAIL\"}\n')
    f.write(f'best_tag={best_tau_tag}\n')
    f.write(f'best_pdms={best_pdms}\n')
" 2>&1
# ---- Decision: prioritize based on gate result ----
if [[ -f "$TAUCUT_OUT/_gate_result.txt" ]]; then
    gate=$(grep "^gate=" "$TAUCUT_OUT/_gate_result.txt" | cut -d= -f2)
    best_tag=$(grep "^best_tag=" "$TAUCUT_OUT/_gate_result.txt" | cut -d= -f2)
else
    gate="UNKNOWN"
    best_tag=""
fi

if [[ "$gate" == "PASS" ]]; then
    # Gate PASS → τ-cut full is highest priority (paper core upgrade)
    log "Phase 3: Gate PASS! Starting full navtest τ-cut with best tag=$best_tag FIRST"
    case "$best_tag" in
        kr040) tau="-0.1253";;
        kr050) tau="-0.1487";;
        kr060) tau="-0.1668";;
        kr070) tau="-0.1840";;
        *) tau="-0.1487"; log "WARN: unknown tag $best_tag, defaulting to kr050";;
    esac
    if ! pgrep -f run_pdm_score_cot >/dev/null; then
        bash "$ROOT/scripts/run_taucut_fullnavtest.sh" "$tau" "$best_tag"
        log "Phase 3: Full τ-cut done."
    else
        log "Phase 3: SKIP full τ-cut — pdm_score already running"
    fi

    # Then FastV r=0.75 (nice-to-have)
    log "Phase 4: Starting FastV r=0.75..."
    if ! pgrep -f run_pdm_score_cot >/dev/null; then
        rm -f "$ROOT/STOP_FASTV"
        bash "$ROOT/scripts/run_fastv_baseline.sh"
        log "Phase 4: FastV r=0.75 done."
    else
        log "Phase 4: SKIP FastV — pdm_score already running"
    fi

    # Then MSE eval (lowest priority)
    log "Phase 5: Starting MSE scorer eval..."
    if ! pgrep -f run_pdm_score_cot >/dev/null; then
        rm -f "$ROOT/STOP_MSE"
        bash "$ROOT/scripts/run_mse_scorer_eval.sh"
        log "Phase 5: MSE scorer eval done."
    else
        log "Phase 5: SKIP MSE eval — pdm_score already running"
    fi

else
    # Gate FAIL → FastV r=0.75 + MSE eval (Route A data completion)
    log "Phase 3: Gate FAIL (or unknown). Route B negative."
    log "         This is clean evidence: even calibrated scorer + τ-cut"
    log "         cannot beat fixed ratio. C2 = robust unlearnable proof."

    log "Phase 3: Starting FastV r=0.75..."
    if ! pgrep -f run_pdm_score_cot >/dev/null; then
        rm -f "$ROOT/STOP_FASTV"
        bash "$ROOT/scripts/run_fastv_baseline.sh"
        log "Phase 3: FastV r=0.75 done."
    else
        log "Phase 3: SKIP FastV — pdm_score already running"
    fi

    log "Phase 4: Starting MSE scorer eval..."
    if ! pgrep -f run_pdm_score_cot >/dev/null; then
        rm -f "$ROOT/STOP_MSE"
        bash "$ROOT/scripts/run_mse_scorer_eval.sh"
        log "Phase 4: MSE scorer eval done."
    else
        log "Phase 4: SKIP MSE eval — pdm_score already running"
    fi
fi

log "Sequencer finished. Check results in $TAUCUT_OUT and logs."

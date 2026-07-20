#!/usr/bin/env bash
# _notify_watcher.sh — Watch for experiment completion and send WeChat notifications
# Launch: nohup bash scripts/_notify_watcher.sh > logs/_notify_watcher.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
OUT="$ROOT/results/raw/tokenprune_S3_full"
TAUCUT="$ROOT/results/raw/tokenprune_taucut"
log(){ echo "[notify $(date +%H:%M:%S)] $*"; }

notify(){
    local msg="$1"
    curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
        -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"$msg\"}}" >/dev/null 2>&1
    log "Sent: $msg"
}

# ---- Generic per-experiment PDMS watcher (P1/P2/P0/GPU7) ----
pdms_of(){ "$PY" -c "
import pandas as pd, glob, sys
fs=glob.glob('$1')
if not fs: sys.exit(1)
df=pd.concat([pd.read_csv(f) for f in fs])
print(f'{df[\"score\"].mean():.6f} (N={len(df)})')
" 2>/dev/null; }

NOTIFIED_EXPS=" "   # space-separated set of already-reported experiment names
watch_experiment(){
    local name="$1" glob="$2"
    if [[ "$NOTIFIED_EXPS" == *" $name "* ]]; then return; fi
    if ls $glob >/dev/null 2>&1; then
        local pd
        pd=$(pdms_of "$glob")
        if [[ -n "$pd" ]]; then
            notify "[TokenRL] ✅ $name done — PDMS=$pd"
            NOTIFIED_EXPS="$NOTIFIED_EXPS$name "
            log "reported $name = $pd"
        fi
    fi
}

# Track what we've already notified about
NOTIFIED_FASTV75=""
NOTIFIED_MSE=""
NOTIFIED_SEQ=""

while true; do
    # ---- Generic P1/P2/P0/GPU7 per-experiment PDMS watcher ----
    watch_experiment "P1 τ-cut kr050 sh1-3"  "$TAUCUT/TC_mse_tau_kr050_sh[123].csv"
    watch_experiment "P1 τ-cut kr070 sh1-3"  "$TAUCUT/TC_mse_tau_kr070_sh[123].csv"
    watch_experiment "GPU7 τ-cut kr040 sh1-3" "$TAUCUT/TC_mse_tau_kr040_sh[123].csv"
    watch_experiment "P2 MSE r=0.5 (4sh)"     "$OUT/MT_scorer_mse_r05_sh[0-3].csv"
    watch_experiment "P2 MSE r=0.75 (4sh)"    "$OUT/MT_scorer_mse_r075_sh[0-3].csv"
    watch_experiment "P0 SparseVLM r=0.5 sh0" "$OUT/MT_sparsevlm_text_r05_sh0.csv"
    watch_experiment "P0 PruMerge r=0.5 sh0"  "$OUT/MT_prumerge_cls_r05_sh0.csv"

    # Check FastV r=0.75 completion (need all 4 shards)
    if [[ -z "$NOTIFIED_FASTV75" ]]; then
        n_fastv=$(ls "$OUT"/MT_fastv_l2_r075_sh{0,1,2,3}.csv 2>/dev/null | wc -l)
        if [[ "$n_fastv" -ge 4 ]]; then
            # Compute aggregate
            pdms=$("$PY" -c "
import pandas as pd
dfs=[]
for sh in range(4):
    df=pd.read_csv('$OUT/MT_fastv_l2_r075_sh'+str(sh)+'.csv')
    df=df[df['token']!='average']
    dfs.append(df)
c=pd.concat(dfs)
print(f'{c[\"score\"].mean():.6f}')
" 2>/dev/null)
            notify "[TokenRL] FastV r=0.75 done! PDMS=$pdms (N~11576). Compare: FastV r=0.5=0.8314, scorer r=0.75=0.8983"
            NOTIFIED_FASTV75="done"
        fi
    fi

    # Check MSE scorer eval completion
    if [[ -z "$NOTIFIED_MSE" ]]; then
        if [[ -f "$OUT/MT_scorer_mse_r05_sh0.csv" ]]; then
            pdms=$("$PY" -c "
import pandas as pd
df=pd.read_csv('$OUT/MT_scorer_mse_r05_sh0.csv')
df=df[df['token']!='average']
print(f'{df[\"score\"].mean():.6f}')
" 2>/dev/null)
            notify "[TokenRL] MSE scorer r=0.5 shard0 done! PDMS=$pdms. Compare: LambdaRank scorer r=0.5 shard0~0.8918"
            NOTIFIED_MSE="done"
        fi
    fi

    # Check sequencer completion
    if [[ -z "$NOTIFIED_SEQ" ]]; then
        if ! pgrep -f "sequencer_20260713" >/dev/null 2>&1; then
            if [[ -n "$NOTIFIED_FASTV75" || -n "$NOTIFIED_MSE" ]]; then
                notify "[TokenRL] Sequencer finished. All scheduled experiments complete."
                NOTIFIED_SEQ="done"
            fi
        fi
    fi

    # All done?
    if [[ -n "$NOTIFIED_FASTV75" && -n "$NOTIFIED_MSE" && -n "$NOTIFIED_SEQ" ]]; then
        log "All notifications sent. Exiting."
        break
    fi

    # No GPU processes at all? Might mean sequencer died
    if ! pgrep -f "run_pdm_score_cot" >/dev/null 2>&1 && ! pgrep -f "sequencer" >/dev/null 2>&1; then
        if [[ -z "$NOTIFIED_SEQ" ]]; then
            notify "[TokenRL] WARNING: No GPU processes and no sequencer running. Check logs."
            NOTIFIED_SEQ="warned"
        fi
    fi

    sleep 300  # check every 5 min
done

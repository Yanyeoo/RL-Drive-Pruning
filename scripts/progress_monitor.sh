#!/usr/bin/env bash
# progress_monitor.sh — 每 30 分钟检查进度并做决策，每小时汇报
# Launch: nohup bash scripts/progress_monitor.sh > logs/progress_monitor.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
LOGDIR="$ROOT/logs"
REPORT_FILE="$ROOT/logs/progress_reports.log"
INTERVAL=1800  # 30 minutes
HOUR_COUNTER=0

log(){ echo "[monitor $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$REPORT_FILE"; }

check_progress() {
    log "========== PROGRESS CHECK =========="
    
    # 1. GPU utilization
    log "--- GPU Status ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        log "  GPU $line"
    done
    
    # 2. Check RL training progress
    log "--- RL Training ---"
    for SH in 0 1 2 3; do
        LOG=$(ls -t $ROOT/logs/rl_shaped_sh${SH}.log 2>/dev/null | head -1)
        if [[ -f "$LOG" ]]; then
            LAST_LINE=$(grep "^\[step" "$LOG" 2>/dev/null | tail -1)
            if [[ -n "$LAST_LINE" ]]; then
                log "  shard$SH: $LAST_LINE"
            else
                log "  shard$SH: no step output yet (check for errors)"
                tail -3 "$LOG" 2>/dev/null | while read l; do log "    $l"; done
            fi
        else
            log "  shard$SH: log not found"
        fi
    done
    
    # 3. Check SparseVLM / VarB progress
    log "--- SparseVLM/VarB ---"
    SPARSE_LOG="$ROOT/logs/sparsevlm_r075.log"
    if [[ -f "$SPARSE_LOG" ]]; then
        tail -3 "$SPARSE_LOG" | while read l; do log "  $l"; done
    else
        log "  Not started or log not found"
    fi
    
    # 4. Check τ-cut analysis
    log "--- τ-cut Analysis ---"
    if [[ -f "$ROOT/results/analysis/taucut_dynamic_stats.json" ]]; then
        log "  DONE: $(head -5 $ROOT/results/analysis/taucut_dynamic_stats.json)"
    elif [[ -f "$ROOT/logs/taucut_dynamic_analysis.log" ]]; then
        log "  Running... $(tail -1 $ROOT/logs/taucut_dynamic_analysis.log)"
    else
        log "  Not started"
    fi
    
    # 5. Check completed CSVs
    log "--- Completed Results ---"
    for pattern in "MT_sparsevlm_text_r075" "MT_varBsafe_scorer_r075" "MT_rl_shaped_r05"; do
        count=$(ls $ROOT/results/raw/tokenprune_S3_full/${pattern}*.csv 2>/dev/null | wc -l)
        log "  $pattern: $count/4 shards"
    done
    
    # 6. Active processes
    log "--- Active Processes ---"
    ps aux | grep -E "(train_scorer_grpo|run_pdm_score_cot|analyze_taucut)" | grep -v grep | while read l; do
        log "  $(echo $l | awk '{print $2, $11, $12, $13}')"
    done
    
    log "========== END CHECK =========="
}

make_decisions() {
    # Decision logic based on progress
    
    # If RL training finished (all 4 logs have "DONE"), start eval
    RL_DONE=0
    for SH in 0 1 2 3; do
        if grep -q "DONE" "$ROOT/logs/rl_shaped_sh${SH}.log" 2>/dev/null; then
            RL_DONE=$((RL_DONE+1))
        fi
    done
    
    if [[ $RL_DONE -eq 4 ]]; then
        # Check if eval already running
        if ! pgrep -f "MT_rl_shaped" >/dev/null; then
            if [[ ! -f "$ROOT/results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh0.csv" ]]; then
                log "[DECISION] RL training done! Launching RL eval..."
                bash "$ROOT/scripts/run_rl_eval_4gpu.sh" &
            fi
        fi
    fi
    
    # If RL eval finished, check if 7B eval should start
    RL_EVAL_DONE=0
    for SH in 0 1 2 3; do
        if [[ -f "$ROOT/results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh${SH}.csv" ]]; then
            RL_EVAL_DONE=$((RL_EVAL_DONE+1))
        fi
    done
    
    if [[ $RL_EVAL_DONE -eq 4 ]]; then
        log "[DECISION] RL eval done! Calculating final PDMS..."
        $PY -c "
import pandas as pd
from pathlib import Path
dfs = []
for sh in range(4):
    p = Path('results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh' + str(sh) + '.csv')
    if p.exists():
        df = pd.read_csv(p)
        df = df[df['token']!='average']
        dfs.append(df)
if dfs:
    all_df = pd.concat(dfs)
    pdms = all_df['score'].mean()
    print(f'RL SHAPED RESULT: N={len(all_df)}, PDMS={pdms:.6f}')
    print(f'vs SFT (0.8920): {\"WIN\" if pdms > 0.8920 else \"LOSE\"} (delta={pdms-0.8920:+.4f}pt)')
" 2>/dev/null | while read l; do log "  $l"; done
    fi
}

# Main loop
log "Progress monitor started. Interval=${INTERVAL}s"
while true; do
    check_progress
    make_decisions
    HOUR_COUNTER=$((HOUR_COUNTER + 1))
    
    if [[ $((HOUR_COUNTER % 2)) -eq 0 ]]; then
        log "[HOURLY REPORT] === Hour $((HOUR_COUNTER/2)) ==="
        # Hourly summary for WeChat notification
        echo "---" >> "$ROOT/logs/hourly_reports.log"
        echo "[$(date '+%H:%M')] Hour $((HOUR_COUNTER/2)) Summary:" >> "$ROOT/logs/hourly_reports.log"
        tail -30 "$REPORT_FILE" >> "$ROOT/logs/hourly_reports.log"
    fi
    
    sleep $INTERVAL
done

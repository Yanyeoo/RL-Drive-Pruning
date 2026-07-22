#!/usr/bin/env bash
# wecom_heartbeat_v2.sh — 企业微信心跳 + 进度监控 + 自动决策
# 每 30 分钟轮询进度，每 60 分钟发企业微信汇报
# Launch: nohup bash scripts/wecom_heartbeat_v2.sh > logs/wecom_heartbeat_v2.log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
CHECK_INTERVAL=1800  # 30 minutes
REPORT_INTERVAL=3600 # 60 minutes
LAST_REPORT=0
LOGFILE="$ROOT/logs/monitor_decisions.log"

cd "$ROOT"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

send_wecom() {
    local content="$1"
    local payload
    payload=$($PY -c "
import json, sys
c = sys.stdin.read()
if len(c) > 1800: c = c[:1800] + '\n...(truncated)'
print(json.dumps({'msgtype': 'text', 'text': {'content': c}}, ensure_ascii=False))
" <<< "$content")
    curl -sS -X POST "$WEBHOOK" -H 'Content-Type: application/json' -d "$payload" > /dev/null 2>&1
    log "WeChat report sent (${#content} chars)"
}

check_and_decide() {
    local now=$(date +%s)
    local gpu_info rl_status sparse_status

    # GPU status
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "N/A")

    # RL training progress
    rl_status=""
    local rl_done=0
    for sh in 0 1 2 3; do
        local logf="$ROOT/logs/rl_shaped_sh${sh}.log"
        if [[ -f "$logf" ]]; then
            local last_step=$(grep '^\[step' "$logf" 2>/dev/null | tail -1)
            if grep -q "DONE\|Epoch.*complete" "$logf" 2>/dev/null; then
                rl_done=$((rl_done+1))
                rl_status="${rl_status}sh${sh}: DONE\n"
            elif [[ -n "$last_step" ]]; then
                rl_status="${rl_status}sh${sh}: $last_step\n"
            else
                rl_status="${rl_status}sh${sh}: loading/starting\n"
            fi
        fi
    done

    # SparseVLM progress
    sparse_status="not running"
    if [[ -f "$ROOT/logs/_sparsevlm_r075/_MT_sparsevlm_text_r075_sh0.log" ]]; then
        local last=$(grep "Processing scenario" "$ROOT/logs/_sparsevlm_r075/_MT_sparsevlm_text_r075_sh0.log" 2>/dev/null | tail -1)
        sparse_status="shard0: $last"
    fi
    # Check completed shards
    local sparse_done=0
    for sh in 0 1 2 3; do
        [[ -f "$ROOT/results/raw/tokenprune_S3_full/MT_sparsevlm_text_r075_sh${sh}.csv" ]] && sparse_done=$((sparse_done+1))
    done
    sparse_status="${sparse_status}\nCompleted: ${sparse_done}/4 shards"

    # Results check
    local results=""
    local rl_eval_done=0
    for sh in 0 1 2 3; do
        [[ -f "$ROOT/results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh${sh}.csv" ]] && rl_eval_done=$((rl_eval_done+1))
    done
    results="RL eval: ${rl_eval_done}/4 shards"

    # === DECISIONS ===
    
    # Decision 1: If RL training done → start eval
    if [[ $rl_done -eq 4 ]]; then
        if ! pgrep -f "MT_rl_shaped" > /dev/null 2>&1; then
            if [[ $rl_eval_done -lt 4 ]]; then
                log "[DECISION] RL training complete! Starting RL eval..."
                nohup bash "$ROOT/scripts/run_rl_eval_4gpu.sh" > "$ROOT/logs/rl_eval_auto.log" 2>&1 &
                log "[ACTION] RL eval launched (PID=$!)"
            fi
        fi
    fi

    # Decision 2: If RL eval done → compute final result
    if [[ $rl_eval_done -eq 4 ]]; then
        local pdms=$($PY -c "
import pandas as pd
from pathlib import Path
dfs = []
for sh in range(4):
    p = Path('results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh' + str(sh) + '.csv')
    if p.exists():
        df = pd.read_csv(p); df = df[df['token']!='average']; dfs.append(df)
if dfs:
    all_df = pd.concat(dfs)
    print(f'{all_df[\"score\"].mean():.6f}')
" 2>/dev/null)
        if [[ -n "$pdms" ]]; then
            results="RL PDMS=$pdms (vs SFT 0.8920)"
            log "[RESULT] RL shaped PDMS = $pdms"
        fi
    fi

    # Decision 3: If GPU4 idle and SparseVLM done → can start more work
    local gpu4_used=$(echo "$gpu_info" | grep "^4," | awk -F',' '{print $2}' | tr -d ' MiB')
    if [[ "${gpu4_used:-0}" -lt 1000 && $sparse_done -eq 4 ]]; then
        log "[INFO] GPU4 idle, SparseVLM done. Available for next task."
    fi

    # === BUILD REPORT ===
    local report="【TokenRL 进度 $(date '+%H:%M')】
GPU: 
$gpu_info

RL训练 (GPU0-3):
$(echo -e "$rl_status")
SparseVLM (GPU4):
$(echo -e "$sparse_status")

结果:
$results

τ-cut: std=0.085 ✅ (动态性已证)
目标: RL PDMS > 0.8920 (SFT baseline)"

    log "$report"

    # Send WeChat report every 60 min
    if [[ $((now - LAST_REPORT)) -ge $REPORT_INTERVAL ]]; then
        send_wecom "$report"
        LAST_REPORT=$now
    fi
}

# === MAIN LOOP ===
log "=== WeChat heartbeat + monitor started ==="
log "Check interval: ${CHECK_INTERVAL}s, Report interval: ${REPORT_INTERVAL}s"

# Send initial report immediately
check_and_decide
LAST_REPORT=$(date +%s)
send_wecom "【TokenRL 启动汇报 $(date '+%H:%M')】
5×H20 GPU window active.
- GPU0-3: RL shaped reward 训练 (~step 170, 4.1s/scene)
- GPU4: SparseVLM r=0.75 eval (1414/2949 shard0)
- τ-cut 动态分析: 完成 (std=0.085 ✅)
- 目标: RL PDMS > 0.8920
下次汇报: 1h 后"

while true; do
    sleep $CHECK_INTERVAL
    check_and_decide
done

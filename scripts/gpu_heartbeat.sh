#!/usr/bin/env bash
# 独立 GPU 心跳: 每小时向企业微信汇报 8 卡 GPU 利用率/显存 + 关键进程数。
# 完全独立于编排器/bridge, 它们退出也不影响本脚本。
# 启动: nohup bash scripts/gpu_heartbeat.sh >/dev/null 2>&1 &
# 停止: touch STOP_HEARTBEAT   (或 kill 掉本进程)
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
HB_LOG="$ROOT/logs/gpu_heartbeat.log"
HB_PID="$ROOT/logs/gpu_heartbeat.pid"
STOP="$ROOT/STOP_HEARTBEAT"
INTERVAL=3600   # 每小时

echo $$ > "$HB_PID"

log(){ echo "[hb $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$HB_LOG"; }

notify(){
    local msg="$1" payload
    payload=$("$PY" -c "import json,sys; c=sys.stdin.read(); print(json.dumps({'msgtype':'text','text':{'content':c[:1800]}}, ensure_ascii=False))" <<< "$msg")
    curl -sS -X POST "$WEBHOOK" -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1
    log "WECOM sent"
}

gpu_report(){
    # 逐卡: index util% mem_used/mem_total
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', *' '{printf "  GPU%s: %s%%  %sMiB/%sMiB\n",$1,$2,$3,$4}'
}

log "GPU heartbeat started (pid=$$, interval=${INTERVAL}s). stop: touch STOP_HEARTBEAT"
# 启动即发一条, 确认活着
notify "【GPU心跳】已启动, 每小时汇报一次GPU利用率。
$(gpu_report)"

while true; do
    [[ -f "$STOP" ]] && { log "STOP_HEARTBEAT detected, exit"; notify "【GPU心跳】收到停止标记, 已退出。"; rm -f "$HB_PID"; exit 0; }
    sleep "$INTERVAL"
    [[ -f "$STOP" ]] && { log "STOP_HEARTBEAT detected, exit"; notify "【GPU心跳】收到停止标记, 已退出。"; rm -f "$HB_PID"; exit 0; }

    REPORT=$(gpu_report)
    # 平均利用率 (判断是否全 0 空转)
    AVG=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++} END{if(n>0) printf "%.0f", s/n; else print "NA"}')
    # 关键进程数
    NPDM=$(pgrep -f run_pdm_score_cot 2>/dev/null | wc -l)
    NTRAIN=$(pgrep -f train_scorer_budget_rl 2>/dev/null | wc -l)
    N7B=$(pgrep -f "run_7b_budget_rl_train\|impromptu7b" 2>/dev/null | wc -l)
    FLAG=""
    [[ "$AVG" != "NA" && "$AVG" -lt 5 ]] && FLAG=" ⚠️GPU近乎空转(均${AVG}%)"

    notify "【GPU心跳 $(date '+%m-%d %H:%M')】均利用率 ${AVG}%${FLAG}
进程: pdm评测=${NPDM} 3B训练=${NTRAIN} 7B=${N7B}
${REPORT}"
done

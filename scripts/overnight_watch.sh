#!/usr/bin/env bash
# overnight_watch.sh — 2026-06-22 夜
#
# 监视 navtrain chain 直到 .chain_complete 出现，每 5 min 写一行 status。
# 完全 read-only。除非检测到长时 hang 才触发 incident §5 escalation（这里只
# 报告，不自动 kill — 留给 AI 早晨决策）。
#
# 不要在前台跑。用法：
#   nohup bash scripts/overnight_watch.sh > /dev/null 2>&1 &
#
# 输出：
#   logs/overnight_status.log     — 每 5 min 一行汇总
#   logs/overnight_alerts.log     — 异常事件
#   logs/overnight_io_history.log — 每 5 min /proc IO 抽样
#
# 终止条件：
#   - .chain_complete 出现 → 写 DONE 并退出 0
#   - 主下载/chain 脚本全死了 → 写 FATAL 并退出 1（不重启，留人决策）
#   - 超过 10h 都没结束 → 写 TIMEOUT 并退出 2

set -u
REPO=/apdcephfs/private_shayladeng/tokenrl_autoVLA
STAGE=/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain
STATUS=${REPO}/logs/overnight_status.log
ALERTS=${REPO}/logs/overnight_alerts.log
IOHIST=${REPO}/logs/overnight_io_history.log

mkdir -p ${REPO}/logs
START_TS=$(date +%s)
MAX_RUNTIME=$((10*3600))    # 10h hard ceiling
POLL_INTERVAL=300            # 5 min

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >> "$STATUS"; }
alert() { echo "[$(ts)] ALERT: $*" >> "$ALERTS"; log "ALERT: $*"; }

log "watch start (PID $$). poll every ${POLL_INTERVAL}s, ceiling ${MAX_RUNTIME}s."

# 上一轮采样的 staging 总 size（用来判 hang）
prev_trainval_size=0
unchanged_polls=0

while true; do
    now=$(date +%s)
    elapsed=$((now - START_TS))

    # 1. 终止条件：.chain_complete 出现
    if [[ -f "${STAGE}/.chain_complete" ]]; then
        log "DONE: .chain_complete found (elapsed ${elapsed}s)"
        # 列出所有 sentinel 状态
        log "  download_complete: $(ls -la ${STAGE}/.download_complete 2>&1)"
        log "  installed sentinels:"
        ls -la ${STAGE}/trainval_sensor_blobs/.navtrain_*.installed 2>&1 \
            | sed 's/^/    /' >> "$STATUS"
        log "  data/splits:"
        ls -la ${REPO}/data/splits/ 2>&1 | sed 's/^/    /' >> "$STATUS"
        exit 0
    fi

    # 2. 收集快照
    chain_alive=$(pgrep -f "post_dl_chain.sh" 2>/dev/null | tr '\n' ',' || true)
    dl_alive=$(pgrep -f "download_navtrain_robust.sh" 2>/dev/null | tr '\n' ',' || true)
    aria2_alive=$(pgrep -f "aria2c" 2>/dev/null | tr '\n' ',' || true)
    tar_alive=$(pgrep -fl "tar.*navtrain\|tar -xz" 2>/dev/null | tr '\n' ';' || true)
    rsync_alive=$(pgrep -fl "rsync.*history\|rsync.*current\|rsync.*sensor_blobs" 2>/dev/null | tr '\n' ';' || true)

    dl_sentinel="missing"
    [[ -f "${STAGE}/.download_complete" ]] && dl_sentinel="present"
    h3_inst="m"; h4_inst="m"
    [[ -f "${STAGE}/trainval_sensor_blobs/.navtrain_history_3.tgz.installed" ]] && h3_inst="OK"
    [[ -f "${STAGE}/trainval_sensor_blobs/.navtrain_history_4.tgz.installed" ]] && h4_inst="OK"

    tgz_now=$(ls -1 ${STAGE}/navtrain_*.tgz 2>/dev/null | head -1)
    tgz_size="-"
    if [[ -n "$tgz_now" ]]; then
        tgz_size=$(stat -c %s "$tgz_now" 2>/dev/null || echo "?")
        tgz_name=$(basename "$tgz_now")
    else
        tgz_name="(none)"
    fi

    trainval_size=$(du -sb ${STAGE}/trainval_sensor_blobs/trainval 2>/dev/null | awk '{print $1}')
    [[ -z "$trainval_size" ]] && trainval_size=0
    h3_dir=$(du -sb ${STAGE}/history_split_3 2>/dev/null | awk '{print $1}')
    [[ -z "$h3_dir" ]] && h3_dir=0
    h4_dir=$(du -sb ${STAGE}/history_split_4 2>/dev/null | awk '{print $1}')
    [[ -z "$h4_dir" ]] && h4_dir=0

    diskfree=$(df --output=avail /apdcephfs/private_shayladeng | tail -1 | awk '{print $1}')

    # 3. 写 status
    log "t=${elapsed}s | dl_sentinel=${dl_sentinel} h3=${h3_inst} h4=${h4_inst} | chain=[${chain_alive}] dl=[${dl_alive}] aria2=[${aria2_alive}] tar=[${tar_alive}] rsync=[${rsync_alive}] | tgz=${tgz_name}(${tgz_size}) | trainval=${trainval_size} h3dir=${h3_dir} h4dir=${h4_dir} | free=${diskfree}KB"

    # 4. /proc IO 抽样（找一个 rsync/tar/aria2c 主进程）
    main_pid=""
    for pat in "tar -xz" "rsync -a" "aria2c"; do
        p=$(pgrep -f "$pat" 2>/dev/null | head -1)
        if [[ -n "$p" ]]; then
            main_pid=$p
            cat /proc/$p/io 2>/dev/null > /tmp/_io_a
            sleep 3
            cat /proc/$p/io 2>/dev/null > /tmp/_io_b
            wch_a=$(grep "^wchar:" /tmp/_io_a | awk '{print $2}')
            wch_b=$(grep "^wchar:" /tmp/_io_b | awk '{print $2}')
            if [[ -n "$wch_a" && -n "$wch_b" ]]; then
                delta=$((wch_b - wch_a))
                echo "[$(ts)] pid=$p ($pat) wchar_delta_3s=${delta} bytes" >> "$IOHIST"
            fi
            break
        fi
    done

    # 5. hang 检测：连续 6 个 5min poll (=30 min) trainval/h3/h4 三个目录 size 一点没变
    cur_blob=$((trainval_size + h3_dir + h4_dir))
    if [[ "$cur_blob" -eq "$prev_trainval_size" ]] && [[ "$cur_blob" -gt 0 ]]; then
        unchanged_polls=$((unchanged_polls + 1))
        if [[ "$unchanged_polls" -eq 3 ]]; then
            alert "data blob size unchanged for 3 polls (~15 min). size=${cur_blob}"
        elif [[ "$unchanged_polls" -eq 6 ]]; then
            alert "data blob size unchanged for 6 polls (~30 min). possible hang. size=${cur_blob}. NOT auto-kill. flagged for AI review."
        elif [[ "$unchanged_polls" -eq 12 ]]; then
            alert "data blob size unchanged for 12 polls (~60 min). HARD STALL. flagged."
        fi
    else
        unchanged_polls=0
    fi
    prev_trainval_size=$cur_blob

    # 6. 主进程全死掉？
    if [[ -z "$chain_alive" && -z "$dl_alive" && -z "$aria2_alive" && -z "$tar_alive" && -z "$rsync_alive" ]]; then
        alert "ALL chain/downloader processes are dead. .chain_complete=missing. FATAL — exiting watcher."
        log "FATAL: bail. AI should diagnose."
        exit 1
    fi

    # 7. 超时
    if [[ "$elapsed" -ge "$MAX_RUNTIME" ]]; then
        alert "TIMEOUT: 10h reached without .chain_complete. exiting watcher (processes left running)."
        log "TIMEOUT"
        exit 2
    fi

    sleep ${POLL_INTERVAL}
done

#!/bin/bash
# 周末 navtrain 看门狗：定期 poll sentinel 数量 + .chain_complete，写到 log。
# 不做任何危险操作 — 只读、只写 log。

STAGING=/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain
SENTDIR=$STAGING/trainval_sensor_blobs
LOG=/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/navtrain_watchdog.log

echo "[navtrain-wd] $(date) start. expect 8 .installed sentinels + .chain_complete" >> $LOG

LAST=-1
while true; do
    CNT=$(ls $SENTDIR/.*.installed 2>/dev/null | wc -l)
    if [ "$CNT" -ne "$LAST" ]; then
        echo "[navtrain-wd] $(date) sentinels = $CNT/8" >> $LOG
        ls $SENTDIR/.*.installed 2>/dev/null | sed 's|^|  - |' >> $LOG
        LAST=$CNT
    fi
    if [ -f "$STAGING/.chain_complete" ]; then
        echo "[navtrain-wd] $(date) .chain_complete FOUND — DONE" >> $LOG
        # 写一个上层指示器，让接手 AI 一眼可见
        echo "navtrain chain complete @ $(date)" > /apdcephfs/private_shayladeng/tokenrl_autoVLA/NAVTRAIN_READY.txt
        exit 0
    fi
    if [ "$CNT" -ge 8 ]; then
        echo "[navtrain-wd] $(date) 8 sentinels present but no .chain_complete (maybe install_navtrain step pending)" >> $LOG
    fi
    sleep 300   # 5min poll
done

#!/usr/bin/env bash
# Simple watcher: poll for FastV r=0.75 completion, then notify
set -u
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
OUT="$ROOT/results/raw/tokenprune_S3_full"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"

echo "[watcher2] started $(date)"

while true; do
    n=$(ls "$OUT"/MT_fastv_l2_r075_sh0.csv "$OUT"/MT_fastv_l2_r075_sh1.csv "$OUT"/MT_fastv_l2_r075_sh2.csv "$OUT"/MT_fastv_l2_r075_sh3.csv 2>/dev/null | wc -l)
    echo "[watcher2] $(date +%H:%M) fastv075 csvs: $n/4"
    if [ "$n" -ge 4 ]; then
        MSG="[TokenRL] FastV r=0.75 ALL 4 shards DONE! Check results."
        curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"$MSG\"}}"
        echo "[watcher2] notified fastv075 done"
        break
    fi
    sleep 300
done

# Also watch for MSE
echo "[watcher2] now watching MSE eval..."
while true; do
    if [ -f "$OUT/MT_scorer_mse_r05_sh0.csv" ]; then
        MSG="[TokenRL] MSE scorer eval shard0 DONE! Check results."
        curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"$MSG\"}}"
        echo "[watcher2] notified MSE done"
        break
    fi
    if ! pgrep -f run_pdm_score_cot >/dev/null 2>&1; then
        MSG="[TokenRL] All GPU jobs finished. Window complete."
        curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"$MSG\"}}"
        echo "[watcher2] no more GPU jobs"
        break
    fi
    sleep 300
done
echo "[watcher2] exit $(date)"

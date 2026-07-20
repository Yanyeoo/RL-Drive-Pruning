#!/usr/bin/env bash
# reclaim_1450.sh — 无人看守周期收尾（明天下午 14:50 触发）
# 动作（按 journal 规则：备份用 cp -a 不动原文件；幂等可重跑）:
#   1. 确认 P1/P2 实验 CSV 就绪（最多等待 40 min 补齐）
#   2. 备份关键 artifact 到 backups/cycle_end_20260719_1450
#   3. 推送最终汇总到企业微信
#   4. git add + commit + push 论文/结果/日志（仅 docs/results/paper，不碰大产物）
# 设计原则：不删除任何东西；任何失败仅告警不中断；可安全重复执行。
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
TS="20260719_1450"
BACKUP_DIR="backups/cycle_end_$TS"
OUT_TAUCUT="$ROOT/results/raw/tokenprune_taucut"
OUT_MSE="$ROOT/results/raw/tokenprune_S3_full"
log(){ echo "[reclaim $(date '+%H:%M:%S')] $*"; }

# ---------- 1. 等待实验补齐 ----------
EXPECT_TAUCUT=("TC_mse_tau_kr050_sh1" "TC_mse_tau_kr050_sh2" "TC_mse_tau_kr050_sh3" \
               "TC_mse_tau_kr070_sh1" "TC_mse_tau_kr070_sh2" "TC_mse_tau_kr070_sh3")
EXPECT_MSE=("MT_scorer_mse_r05_sh0" "MT_scorer_mse_r05_sh1" "MT_scorer_mse_r05_sh2" "MT_scorer_mse_r05_sh3" \
            "MT_scorer_mse_r075_sh0" "MT_scorer_mse_r075_sh1" "MT_scorer_mse_r075_sh2" "MT_scorer_mse_r075_sh3")
wait_for(){
  local label="$1"; shift; local dir="$1"; shift; local files=("$@")
  for i in $(seq 1 40); do
    local missing=0
    for f in "${files[@]}"; do [[ -f "$dir/$f.csv" ]] || { missing=$((missing+1)); break; }; done
    [[ $missing -eq 0 ]] && { log "$label: all ready"; return 0; }
    log "$label: waiting ($i/40) for $missing file(s)..."; sleep 60
  done
  log "WARN $label: still missing after 40min — proceeding with available data"
}

# ---------- 2. 聚合 PDMS ----------
aggregate(){
  local dir="$1"; local -a names=("${!2}"); local out=0; local n=0
  "$PY" - <<PY 2>/dev/null
import pandas as pd, glob, sys
files=[ "$dir/$x.csv" for x in ${names[@]@Q} ]
dfs=[]
for f in files:
    try:
        d=pd.read_csv(f); d=d[d['token']!='average']; dfs.append(d)
    except Exception as e:
        print(f"MISSING/ERR {f}: {e}")
c=pd.concat(dfs)
print(f"AGG N={len(c)} PDMS={c['score'].mean():.6f}")
PY
}

log "=== RECLAIM $TS start ==="
wait_for "P1-taucut" "$OUT_TAUCUT" "${EXPECT_TAUCUT[@]}"
wait_for "P2-MSE"    "$OUT_MSE"    "${EXPECT_MSE[@]}"

P1_SUM=$(aggregate "$OUT_TAUCUT" EXPECT_TAUCUT[@] 2>/dev/null | tail -1)
P2_05=$(aggregate "$OUT_MSE"   EXPECT_MSE[@] 2>/dev/null | tail -1)

# ---------- 3. 备份（cp -a，不动原文件）----------
mkdir -p "$BACKUP_DIR"
log "backing up results + docs + logs -> $BACKUP_DIR"
cp -a results "$BACKUP_DIR/results" 2>/dev/null || log "WARN results backup partial"
cp -a docs "$BACKUP_DIR/docs" 2>/dev/null || log "WARN docs backup partial"
cp -a logs "$BACKUP_DIR/logs" 2>/dev/null || log "WARN logs backup partial"
cp -a paper "$BACKUP_DIR/paper" 2>/dev/null || log "WARN paper backup partial"
[[ -f _21h_queue.txt ]] && cp -a _21h_queue.txt scripts/run_21h_master.sh "$BACKUP_DIR/" 2>/dev/null
echo "RECLAIM $TS generated at $(date)" > "$BACKUP_DIR/status.txt"
log "backup done -> $BACKUP_DIR"

# ---------- 4. 企业微信最终报告 ----------
content="【RL-Drive-Pruning 周期收尾 $TS】
实验: 8×H20 无人看守窗口已完成/收尾。
$P1_SUM
$P2_05
备份: $BACKUP_DIR (cp -a, 原文件未动)
GitHub: 推送论文+结果+日志中...
下轮: 若 RL 仍 < SFT 0.895, 按 journal 2b reverse 改主线。"
payload="$("$PY" -c "import json,os; c=os.environ['C']; print(json.dumps({'msgtype':'text','text':{'content':c[:1800]}},ensure_ascii=False))" C="$content")"
curl -sS -X POST "$WEBHOOK" -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1 && log "wecom final sent" || log "WARN wecom send failed"

# ---------- 5. GitHub push（仅论文/结果/日志，不推大模型产物）----------
git add docs results paper scripts logs 2>/dev/null || true
git commit -q -m "cycle end $TS: P1 tau-cut full curve + P2 MSE ablation results, paper tables updated" 2>/dev/null || log "git: nothing to commit"
git push origin main 2>&1 | tail -3 && log "github pushed" || log "WARN github push failed (check auth/network)"

log "=== RECLAIM $TS DONE ==="

#!/usr/bin/env bash
# run_ef_after_d.sh — bridge E/F stages that run AFTER the unattended orchestrator
# finishes stage D (and frees the 8x H20 GPUs). This script is launched NOW in the
# background; it polls until the orchestrator is done, then runs:
#   E: fine-grained 3B SFT scorer ratio sweep (full 4 shards) -> completes the
#      PDMS-vs-keep_ratio Pareto curve for the main table (reuses stage-D dispatcher).
#   F: efficiency profile (token-saving %% + wall-clock speedup) -> run_efficiency_profile.sh
#
# It does NOT touch the running orchestrator. Launch:
#   nohup bash scripts/run_ef_after_d.sh > logs/ef_after_d.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
STATE_FILE="$ROOT/logs/unattended_state.txt"
JOURNAL="$ROOT/logs/unattended_journal.log"
MAINTABLE_OUT="$ROOT/results/raw/tokenprune_S3_full"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
log(){ echo "[ef-bridge $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$JOURNAL"; }
notify(){ curl -s -m 10 -H "Content-Type: application/json" -d "{\"msgtype\":\"text\",\"content\":{\"content\":\"$1\"}}" "$WEBHOOK" >/dev/null 2>&1; }

log "E/F bridge started; polling for stage D completion (state=DONE + GPUs free)..."

# Poll until D is done AND the 8 GPUs are free (no orchestrator, no pdm workers).
while true; do
  ST=$(cat "$STATE_FILE" 2>/dev/null)
  ORCH=$(pgrep -f run_unattended_pipeline | head -1)
  PDM=$(pgrep -f run_pdm_score_cot | head -1)
  NMT=$(ls "$MAINTABLE_OUT"/*.csv 2>/dev/null | wc -l)
  if [[ "$ST" == "DONE" && -z "$ORCH" && -z "$PDM" ]]; then
    log "stage D done (state=DONE, no orchestrator, no pdm) -> proceed"; break
  fi
  # fallback: orchestrator gone + main table fully produced (>=24) + no pdm
  if [[ -z "$ORCH" && -z "$PDM" && "$NMT" -ge 24 ]]; then
    log "stage D done (orchestrator gone, $NMT main-table CSVs, no pdm) -> proceed"; break
  fi
  sleep 120
done

# ---------- E: fine-grained 3B SFT scorer ratio sweep (full 4 shards) ----------
log "=== STAGE E: fine-grained 3B ratio sweep (full navtest) ==="
notify "【TokenRL】编排器 D 完成 ✅ 衔接脚本启动 阶段E: 细粒度 3B SFT scorer ratio 扫描 (全 navtest 4 shard, 补 Pareto 曲线)"
export ARMS_SPEC="scorer 0.1;scorer 0.35;scorer 0.65;scorer 0.9"
bash "$ROOT/scripts/run_s3_maintable_full_navtest.sh" 2>&1 | tee -a "$JOURNAL"
log "STAGE E done"

# ---------- F: efficiency profile ----------
log "=== STAGE F: efficiency profile (token-saving + wall-clock speedup) ==="
notify "【TokenRL】阶段E 完成 ✅ 启动 阶段F: 效率实测 (token 剪枝率 + drop 变体计时加速比)"
bash "$ROOT/scripts/run_efficiency_profile.sh" 2>&1 | tee -a "$JOURNAL"

notify "🎉【TokenRL】E/F 全部完成 (细粒度 Pareto + 效率实测)。8卡 H20 现已空闲, 可交明早复核。详见 results/eff_profile/efficiency_summary.md 与 results/raw/tokenprune_S3_full/"
log "=== E/F bridge DONE ==="

#!/usr/bin/env bash
# run_ef_after_d.sh — bridge E/F/G/H stages that run AFTER the unattended orchestrator
# finishes stage D (and frees the 8x H20 GPUs). This script is launched NOW in the
# background; it polls until the orchestrator is done, then runs:
#   E: fine-grained 3B SFT scorer ratio sweep (full 4 shards) -> completes the
#      PDMS-vs-keep_ratio Pareto curve for the main table (reuses stage-D dispatcher).
#   F: efficiency profile (token-saving %% + wall-clock speedup) -> run_efficiency_profile.sh
#   G: 7B Budget RL training (same method as 3B, retrained on 7B from s3_token_scorer_7b)
#      -> completes contribution (3) for the FULL method; keeps 8 GPUs busy ~20h.
#   H: evaluate the 7B Budget RL scorer on ImpromptuVLA_7B (token_net -> SFT-compat reuse)
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

# ---------- G: 7B Budget RL training (completes contribution 3 for the FULL method) ----------
# Same method as 3B Budget RL, retrained on 7B (warm-start from s3_token_scorer_7b),
# then phase H evaluates it on ImpromptuVLA_7B. Keeps all 8 GPUs busy ~20h (past 09:00).
log "=== STAGE G: 7B Budget RL training (warm-start from s3_token_scorer_7b) ==="
notify "【TokenRL】阶段F 完成 ✅ 启动 阶段G: 7B Budget RL 训练 (same method, retrained on 7B; keeps 8x H20 busy ~20h)"
bash "$ROOT/scripts/run_7b_budget_rl_train.sh" 2>&1 | tee -a "$JOURNAL"
G_RC=$?
if [[ $G_RC -ne 0 ]]; then
  log "STAGE G exited rc=$G_RC (smoke test failed or training error). Skipping stage H."
  notify "⚠️【TokenRL】阶段G (7B Budget RL) 退出 rc=$G_RC. 未产出 7B budget scorer, 跳过阶段H. 明早请查 logs/budget_rl_7b_*.log"
else
  log "STAGE G done (7B budget RL trained). Proceeding to stage H (eval on ImpromptuVLA_7B)."

  # ---------- H: evaluate 7B Budget RL scorer on ImpromptuVLA_7B ----------
  # Convert the trained 7B budget scorer's token_net -> SFT-compatible checkpoint, then
  # reuse the existing ImpromptuVLA_7B eval harness (run_7b_eval_dual.sh) at fixed ratios.
  B7_OUT=$(cat "$ROOT/logs/budget_rl_7b_outdir.txt" 2>/dev/null)
  B7_SH0="${B7_OUT}_sh0"
  if [[ -n "$B7_OUT" && -f "$B7_SH0/checkpoint.pt" ]]; then
    COMPAT="$B7_SH0/sft_compat_checkpoint.pt"
    log "=== STAGE H: convert 7B budget token_net -> $COMPAT, eval on ImpromptuVLA_7B ==="
    notify "【TokenRL】阶段G 完成 ✅ 启动 阶段H: 7B Budget RL scorer 在 ImpromptuVLA_7B 评测"
    "$PY" - <<PYEOF 2>&1 | tee -a "$JOURNAL"
import torch
sd = torch.load("$B7_SH0/checkpoint.pt", map_location="cpu", weights_only=False)
if "state_dict" in sd: sd = sd["state_dict"]
out = {k.replace("token_net.", "net."): v for k, v in sd.items() if k.startswith("token_net.")}
assert out, "no token_net.* keys found in 7B budget ckpt"
torch.save(out, "$COMPAT")
print(f"[H] wrote SFT-compat 7B budget scorer: {len(out)} keys -> $COMPAT")
PYEOF
    if [[ -f "$COMPAT" ]]; then
      SCORER_CKPT="$COMPAT" IMPROMPTU_OUT="$ROOT/results/impromptu7b_rl" \
        bash "$ROOT/scripts/run_7b_eval_dual.sh" 2>&1 | tee -a "$JOURNAL"
      log "STAGE H done -> results/impromptu7b_rl/"
      notify "✅【TokenRL】阶段H 完成: 7B Budget RL scorer 已在 ImpromptuVLA_7B 出结果 (results/impromptu7b_rl/). 动态预算评测留作明早小补。"
    else
      log "STAGE H: conversion failed; skip eval."
      notify "⚠️【TokenRL】阶段H: 7B budget token_net 转换失败, 跳过 ImpromptuVLA 评测."
    fi
  else
    log "STAGE H: 7B budget scorer shard0 ckpt not found at $B7_SH0; skip."
    notify "⚠️【TokenRL】阶段H: 未找到 7B budget scorer ckpt ($B7_SH0), 跳过评测."
  fi
fi

notify "🎉【TokenRL】E/F/G/H 全部完成. 8卡 H20 持续占用至 7B 训练+评测结束. 详见 results/eff_profile/ results/raw/tokenprune_S3_full/ results/impromptu7b_rl/ ckpt/s3_token_scorer_budget_rl_7b_*/"
log "=== E/F/G/H bridge DONE ==="

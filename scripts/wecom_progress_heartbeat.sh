#!/usr/bin/env bash
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
PYTHON_BIN="${PYTHON_BIN:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"

send_report() {
  cd "$ROOT" || exit 1
  local ts proc gpu recent rl_tail content
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"

  proc="$(pgrep -af 'python|train_scorer|navtest|grpo|run_.*eval' | grep -v 'wecom_progress_heartbeat' | head -12 || true)"
  [[ -n "$proc" ]] || proc="无匹配运行进程"

  gpu="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/^/GPU /' | head -8 || true)"
  [[ -n "$gpu" ]] || gpu="nvidia-smi 不可用"

  recent="$(find logs results/raw ckpt -type f -mmin -130 2>/dev/null | sort | tail -20 || true)"
  [[ -n "$recent" ]] || recent="过去约2小时未发现新 artifact"

  if [[ -f ckpt/s3_token_scorer_rl/train_log.jsonl ]]; then
    rl_tail="$(tail -3 ckpt/s3_token_scorer_rl/train_log.jsonl)"
  elif [[ -f ckpt/s3_token_scorer_rl_pilot/train_log.jsonl ]]; then
    rl_tail="pilot tail:\n$(tail -3 ckpt/s3_token_scorer_rl_pilot/train_log.jsonl)"
  else
    rl_tail="未发现 RL train_log"
  fi

  content="【RL-Drive-Pruning 2h进度】$ts
运行进程:
$proc

GPU:
$gpu

RL训练日志:
$rl_tail

近2小时新artifact:
$recent

当前思考:
- 若 GPU 空闲且仍在实验窗口，应立即启动下一优先级任务。
- 若 eval 已完成，应先聚合 CSV，再决定 full / ablation / baseline 下一步。
- 若结果低于 SFT scorer，应回退到 SFT scorer + τ-cut adaptive 主线。

下一步计划:
- 优先补 SparseVLM / ToMe(PruMerge) baseline、τ-cut full curve、MSE fixed-r ablation。
- 偏离计划时先写 journal，并附 reverse 指令。

说明: 本脚本只做状态汇报，不启动/终止实验；关键决策仍以会话内事实核验为准。"

  payload="$(CONTENT="$content" "$PYTHON_BIN" - <<'PY'
import json
import os
content = os.environ["CONTENT"]
if len(content) > 1800:
    content = content[:1800] + "\n...(truncated)"
print(json.dumps({"msgtype": "text", "text": {"content": content}}, ensure_ascii=False))
PY
)"
  curl -sS -X POST "$WEBHOOK" -H 'Content-Type: application/json' -d "$payload"
  echo
}

while true; do
  sleep "$INTERVAL_SECONDS"
  send_report
done

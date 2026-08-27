#!/usr/bin/env bash
# auto_launch_batch2.sh
# 自动监控 batch1 (run_supplement_experiments.sh) 的 GPU0/1/2 完成状态
# 一旦三个 GPU worker 全部完成，自动启动 batch2
# 用法: nohup bash scripts/auto_launch_batch2.sh > logs/auto_batch2.log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"

# 找最新的 supplement_* 日志目录（batch1 的）
BATCH1_LOGDIR=$(ls -dt "$ROOT/logs/supplement_"*/ 2>/dev/null | head -1)
if [[ -z "$BATCH1_LOGDIR" ]]; then
  echo "[auto $(date +%H:%M:%S)] ERROR: No supplement_* log dir found"
  exit 1
fi
BATCH1_RUNLOG="${BATCH1_LOGDIR}run.log"
echo "[auto $(date +%H:%M:%S)] Monitoring batch1: $BATCH1_RUNLOG"

# GPU0/1/2 最后一个 job 的实验名后缀
declare -A LAST_JOB=(
  [0]="SUPP_budgetrl_dynamic_raw_sh3"
  [1]="SUPP_sft_r0355_raw_sh3"
  [2]="SUPP_sparsevlm_r05_fallback_sh3"
)

# 检查某个 GPU 的最后一个 job 是否完成
check_gpu_done() {
  local gpu=$1
  local last_exp="${LAST_JOB[$gpu]}"
  grep -qE "\[GPU${gpu}\] (DONE|FAIL) ${last_exp}" "$BATCH1_RUNLOG" 2>/dev/null
}

echo "[auto $(date +%H:%M:%S)] Waiting for GPU0/1/2 workers to finish..."
echo "[auto $(date +%H:%M:%S)] Last jobs to watch:"
echo "[auto $(date +%H:%M:%S)]   GPU0: ${LAST_JOB[0]}"
echo "[auto $(date +%H:%M:%S)]   GPU1: ${LAST_JOB[1]}"
echo "[auto $(date +%H:%M:%S)]   GPU2: ${LAST_JOB[2]}"

# 轮询等待
POLL_INTERVAL=60  # 每60秒检查一次
while true; do
  gpu0_done=false; gpu1_done=false; gpu2_done=false
  check_gpu_done 0 && gpu0_done=true
  check_gpu_done 1 && gpu1_done=true
  check_gpu_done 2 && gpu2_done=true

  now=$(date +%H:%M:%S)
  echo "[auto $now] Status: GPU0=$gpu0_done GPU1=$gpu1_done GPU2=$gpu2_done"

  if $gpu0_done && $gpu1_done && $gpu2_done; then
    echo "[auto $now] === All GPU0/1/2 workers DONE! Launching batch2 ==="
    break
  fi

  sleep $POLL_INTERVAL
done

# 检查 batch1 主进程是否还在跑（防止误判）
BATCH1_PID=$(ps aux | grep "run_supplement_experiments.sh" | grep -v grep | grep -v "$0" | awk '{print $2}' | head -1)
if [[ -n "$BATCH1_PID" ]]; then
  echo "[auto $(date +%H:%M:%S)] batch1 still running (PID $BATCH1_PID) — GPU3 likely still working."
  echo "[auto $(date +%H:%M:%S)] Killing batch1 GPU0/1/2 worker subprocesses so batch2 can use those GPUs..."
  # 尝试杀掉 GPU0/1/2 的 worker 子进程（GPU3 保留）
  # batch1 的 worker 是 run_supplement_experiments.sh 的子 bash 进程
  # 我们通过查找 run_supplement_experiments.sh 的子进程来识别
  BATCH1_CHILDREN=$(pgrep -P "$BATCH1_PID" 2>/dev/null)
  for child in $BATCH1_CHILDREN; do
    child_cmd=$(ps -o args= -p "$child" 2>/dev/null)
    # GPU0/1/2 的 worker 会跑 SUPP_budgetrl/SUPP_sft/SUPP_sparsevlm
    # GPU3 跑 SUPP_fastv/SUPP_prumerge
    if echo "$child_cmd" | grep -qE "SUPP_(budgetrl|sft|sparsevlm)"; then
      echo "[auto $(date +%H:%M:%S)] Killing GPU0/1/2 worker PID=$child"
      kill "$child" 2>/dev/null
    fi
  done
  sleep 3
fi

# 确认 GPU0/1/2 空闲
echo "[auto $(date +%H:%M:%S)] Waiting for GPU0/1/2 to be free..."
for i in 0 1 2; do
  while true; do
    mem_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$i" 2>/dev/null | awk -F',' '{print $2}' | tr -d ' MiB')
    if [[ -n "$mem_used" ]] && [[ "$mem_used" -lt 1000 ]]; then
      echo "[auto $(date +%H:%M:%S)] GPU$i free ($mem_used MiB)"
      break
    fi
    echo "[auto $(date +%H:%M:%S)] GPU$i still busy ($mem_used MiB), waiting..."
    sleep 10
  done
done

# 启动 batch2
BATCH2_LOG="$ROOT/logs/supplement_batch2_$(date +%Y%m%d_%H%M%S).log"
echo "[auto $(date +%H:%M:%S)] Launching batch2..."
nohup bash "$ROOT/scripts/run_supplement_batch2.sh" > "$BATCH2_LOG" 2>&1 &
BATCH2_PID=$!
echo "[auto $(date +%H:%M:%S)] batch2 launched (PID=$BATCH2_PID, log=$BATCH2_LOG)"

# 短暂等待确认启动
sleep 30
BATCH2_RUNNING=$(ps -p "$BATCH2_PID" -o pid= 2>/dev/null)
if [[ -n "$BATCH2_RUNNING" ]]; then
  echo "[auto $(date +%H:%M:%S)] batch2 confirmed running."
else
  echo "[auto $(date +%H:%M:%S)] WARNING: batch2 process not found after launch! Check $BATCH2_LOG"
fi

echo "[auto $(date +%H:%M:%S)] Auto-launch script exiting. Monitor batch2 with:"
echo "  tail -f $BATCH2_LOG"

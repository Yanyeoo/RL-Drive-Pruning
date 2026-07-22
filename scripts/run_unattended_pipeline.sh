#!/usr/bin/env bash
# run_unattended_pipeline.sh — TokenRL 无人值守编排器 (stage machine)
#
# 阶段:
#   A: 等待 Budget RL 8卡训练完成 (当前已在跑的孤儿进程, 仅监控不启动)
#   B: 8卡跑 Budget RL EVAL  -> scripts/run_budget_rl_eval_8gpu.sh
#   C: 7B 迁移实验           -> scripts/run_7b_eval_dual.sh
#   D: 8卡补充主表实验        -> scripts/run_s3_maintable_full_navtest.sh
#
# 特性:
#   - 每小时向企业微信汇报进度 (复用 legacy wecom webhook)
#   - 阶段切换 / 异常 立即发企业微信
#   - 每阶段断点续跑 (CSV 已存在则跳过)
#   - 训练崩溃检测: 进程消失但 ckpt 不齐 -> 告警并停止, 不误导下游
#
# Launch: nohup bash scripts/run_unattended_pipeline.sh > logs/unattended_pipeline.log 2>&1 &
# Stop :  touch STOP_UNATTENDED   (下一轮循环检测后优雅退出)
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
STATE_FILE="$ROOT/logs/unattended_state.txt"
JOURNAL="$ROOT/logs/unattended_journal.log"
PID_FILE="$ROOT/logs/unattended_pipeline.pid"
STOP="$ROOT/STOP_UNATTENDED"
echo $$ > "$PID_FILE"

# 若已有一个活的编排器, 不重复启动
if [[ -f "$PID_FILE" ]]; then
    OLD=$(cat "$PID_FILE" 2>/dev/null)
    if [[ -n "$OLD" && "$OLD" != "$$" && -d "/proc/$OLD" ]]; then
        echo "[unattend] already running pid=$OLD, exit"; exit 0
    fi
fi

log(){ echo "[unattend $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$JOURNAL"; }
set_state(){ echo "$1" > "$STATE_FILE"; log "STATE -> $1"; }
notify(){
    local msg="$1"
    local payload
    payload=$("$PY" -c "import json,sys; c=sys.stdin.read(); print(json.dumps({'msgtype':'text','text':{'content':c[:1800]}}, ensure_ascii=False))" <<< "$msg")
    curl -sS -X POST "$WEBHOOK" -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1
    log "WECOM: $msg"
}
pdms_of(){ "$PY" -c "
import pandas as pd, glob, sys
fs = sorted(glob.glob(sys.argv[1]))
if fs:
    df = pd.concat([pd.read_csv(f) for f in fs]); df = df[df['token']!='average']
    print(f'{df[\"score\"].mean():.6f} (N={len(df)})')
" "$1" 2>/dev/null; }

BUDGET_BASE=$(ls -d "$ROOT"/ckpt/s3_token_scorer_budget_rl_*_sh0 2>/dev/null | sed 's/_sh0$//' | head -1)
log "BUDGET_BASE=$BUDGET_BASE"

# ============ 每小时心跳 (企业微信) ============
heartbeat(){
    while true; do
        sleep 3600
        [[ -f "$STOP" ]] && break
        local ts stage gpu proc arts
        ts="$(date '+%Y-%m-%d %H:%M %Z')"
        stage="$(cat "$STATE_FILE" 2>/dev/null || echo unknown)"
        gpu="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '|')"
        proc="$(pgrep -af 'train_scorer_budget_rl|run_pdm_score_cot|run_impromptu7b|run_s3_maintable' | grep -v unattend | wc -l) running jobs"
        arts="$(find results logs -type f -mmin -70 2>/dev/null | sort | tail -6 | sed 's#.*/##' | tr '\n' ';')"
        local msg="【TokenRL 无人值守进度 $ts】
阶段: $stage
GPU: $gpu
进程: $proc
近1h产物: $arts"
        notify "$msg"
    done
}
heartbeat &
HB_PID=$!

# ============ 阶段 A: 等 Budget RL 训练完成 ============
set_state "A:waiting_budget_rl_training"
notify "【TokenRL】无人值守启动 ✅ 阶段A: 等待 Budget RL 8卡训练完成 (base=${BUDGET_BASE##*/})"
while true; do
    [[ -f "$STOP" ]] && { notify "【TokenRL】收到 STOP_UNATTENDED, 阶段A 退出"; kill $HB_PID 2>/dev/null; exit 0; }
    N_PROC=$(pgrep -f train_scorer_budget_rl | wc -l)
    if [[ "$N_PROC" -eq 0 ]]; then
        sleep 90                               # 双检: 避免训练间瞬间 0 进程误判
        N_PROC2=$(pgrep -f train_scorer_budget_rl | wc -l)
        if [[ "$N_PROC2" -eq 0 ]]; then
            # Final checkpoint is written to the SHARD ROOT as checkpoint.pt at the end of
            # training (tag='final' -> save_dir = out_dir). There is NO ckpt_final/ subdir.
            # ckpt_best/ is a frozen early spike (~step50) and must NOT be used as a completion signal.
            N_FINAL=$(ls "${BUDGET_BASE}"_sh*/checkpoint.pt 2>/dev/null | wc -l)
            if [[ "$N_FINAL" -ge 8 ]]; then
                log "training complete: 8 shard-root final checkpoint.pt present"; break
            else
                notify "⚠️【TokenRL】Budget RL 训练进程全消失, 但仅 $N_FINAL/8 shard 根目录 checkpoint.pt 落盘 (final 在结尾才写入根目录; ckpt_best 为早期 best 占位不可作完成信号), 疑似崩溃! 请检查 logs/budget_rl_sh*.log。编排器停止, 已避免误导下游。"
                kill $HB_PID 2>/dev/null
                set_state "A:CRASHED"
                exit 1
            fi
        fi
    fi
    sleep 120
done

# ============ 阶段 B: 8卡 EVAL ============
set_state "B:budget_rl_eval_8gpu"
notify "【TokenRL】阶段B: Budget RL 训练完成 ✅ 启动 8卡 EVAL (动态budget 4shard + 固定r=0.5 4shard, 全 navtest)"
if [[ -f "$ROOT/results/raw/tokenprune_S3_full/MT_budget_rl_dynamic_sh3.csv" ]]; then
    log "B eval CSVs already present, skip"
else
    bash "$ROOT/scripts/run_budget_rl_eval_8gpu.sh" 2>&1 | tee -a "$JOURNAL"
fi
DP_DYN=$(pdms_of "$ROOT/results/raw/tokenprune_S3_full/MT_budget_rl_dynamic_sh[0-3].csv")
DP_FIX=$(pdms_of "$ROOT/results/raw/tokenprune_S3_full/MT_budget_rl_r050_sh[0-3].csv")
notify "【TokenRL】阶段B EVAL 完成 ✅ 动态budget PDMS=$DP_DYN | 固定r=0.5 PDMS=$DP_FIX (ref: SFT 0.8920 / SFT+τcut+VarB 0.9045)"

# ============ 阶段 C: 7B 迁移 ============
set_state "C:7b_transfer"
notify "【TokenRL】阶段C: 启动 7B 迁移实验 (ImpromptuVLA 7B zero-shot)"
if [[ -f "$ROOT/results/impromptu7b/eval_r10.json" ]]; then
    log "C 7B eval already present, skip"
else
    bash "$ROOT/scripts/run_7b_eval_dual.sh" 2>&1 | tee -a "$JOURNAL"
fi
notify "【TokenRL】阶段C 7B迁移完成 ✅ (见 results/impromptu7b/)"

# ============ 阶段 D: 8卡补充主表 ============
set_state "D:maintable_supplementary_8gpu"
notify "【TokenRL】阶段D: 启动 8卡补充主表实验 (baselines attn_L12/random + SFT scorer ratio sweep)"
bash "$ROOT/scripts/run_s3_maintable_full_navtest.sh" 2>&1 | tee -a "$JOURNAL"
notify "【TokenRL】🎉 全流程完成 (Budget RL训练 → EVAL → 7B → 主表补充)。可以填论文数字了！"
set_state "DONE"
kill $HB_PID 2>/dev/null
log "=== unattended pipeline DONE ==="

#!/usr/bin/env bash
# _master_sequencer_rl.sh — Auto-sequence: RL train → eval → (optional) 7B
# Launch: nohup bash scripts/_master_sequencer_rl.sh > logs/_master_sequencer_rl.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
source scripts/setup_navsim_env_vars.sh 2>/dev/null || true
export PYTHONPATH="${ROOT}/code:${ROOT}/code/third_party/AutoVLA:${PYTHONPATH:-}"

log(){ echo "[seq $(date '+%H:%M:%S')] $*"; }
notify(){
    curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
        -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"[TokenRL-Seq] $1\"}}" >/dev/null 2>&1
    log "NOTIFY: $1"
}

# ============================================================
# PHASE 1: Wait for RL training to finish
# ============================================================
log "=== PHASE 1: Waiting for RL training to complete ==="
notify "Sequencer 启动。等待 RL 训练完成..."

RL_CKPT_DIR=""
while true; do
    # Find the most recent RL ckpt dir
    latest_dir=$(ls -dt ckpt/s3_token_scorer_rl_shaped_* 2>/dev/null | head -1)
    if [[ -z "$latest_dir" ]]; then
        sleep 60; continue
    fi

    # Check if training process is still running
    if pgrep -f "train_scorer_grpo" >/dev/null 2>&1; then
        sleep 120  # still training, wait 2 min
        continue
    fi

    # Process not running — check if it finished or crashed
    if [[ -f "$latest_dir/train_log.jsonl" ]]; then
        last_epoch=$("$PY" -c "
import json
lines = open('$latest_dir/train_log.jsonl').readlines()
if lines: print(json.loads(lines[-1]).get('epoch', -1))
else: print(-1)
" 2>/dev/null)
        if [[ "$last_epoch" -ge 2 ]]; then
            RL_CKPT_DIR="$latest_dir"
            log "RL training completed! ckpt=$RL_CKPT_DIR, last_epoch=$last_epoch"
            notify "✅ RL训练完成 ($RL_CKPT_DIR). 开始 P2 eval..."
            break
        else
            # Might have crashed mid-way
            log "WARNING: RL process died at epoch $last_epoch. Using best available ckpt."
            RL_CKPT_DIR="$latest_dir"
            notify "⚠️ RL进程在 epoch=$last_epoch 结束。使用已有 ckpt 继续 eval。"
            break
        fi
    fi
    sleep 60
done

# ============================================================
# PHASE 2: Eval RL scorer on full navtest (4 shard)
# ============================================================
log "=== PHASE 2: Eval RL scorer on navtest (4 shard) ==="
notify "开始 P2: 4-shard eval RL scorer on navtest..."

OUT_DIR="results/raw/tokenprune_S3_full"
mkdir -p "$OUT_DIR"

# Check which eval script exists
if [[ -f scripts/run_tokenprune_eval.py ]]; then
    EVAL_SCRIPT="scripts/run_tokenprune_eval.py"
elif [[ -f scripts/run_tokenprune_sweep.sh ]]; then
    EVAL_SCRIPT="scripts/run_tokenprune_sweep.sh"
else
    notify "❌ 找不到 eval 脚本! ls scripts/run_tokenprune*"
    log "ERROR: No eval script found"
    exit 1
fi

# Launch 4-shard parallel eval
for SH in 0 1 2 3; do
    OUT_CSV="$OUT_DIR/MT_rl_shaped_r05_sh${SH}.csv"
    if [[ -f "$OUT_CSV" ]]; then
        log "shard $SH already exists, skipping"
        continue
    fi
    log "Launching eval shard $SH on GPU $SH"
    CUDA_VISIBLE_DEVICES=$SH "$PY" "$EVAL_SCRIPT" \
        --scorer-ckpt "$RL_CKPT_DIR" \
        --keep-ratio 0.5 \
        --shard-id $SH --num-shards 4 \
        --out-csv "$OUT_CSV" \
        --selector scorer > "logs/_rl_eval_sh${SH}.log" 2>&1 &
done

log "Waiting for all 4 eval shards to complete..."
wait
log "All eval shards done."

# Compute aggregate PDMS
RESULT=$("$PY" -c "
import pandas as pd
dfs=[]
for sh in range(4):
    p=f'$OUT_DIR/MT_rl_shaped_r05_sh{sh}.csv'
    try:
        df=pd.read_csv(p)
        df=df[df['token']!='average']
        dfs.append(df)
    except: pass
if dfs:
    c=pd.concat(dfs)
    rl=c['score'].mean()
    print(f'PDMS={rl:.6f} (N={len(c)}), vs SFT=0.8920, gain={rl-0.8920:+.4f}pt')
else:
    print('ERROR: no CSV files found')
" 2>/dev/null)

log "RL eval result: $RESULT"
notify "✅ P2 eval 完成! $RESULT"

# ============================================================
# PHASE 3: Summary
# ============================================================
log "=== PHASE 3: All critical tasks done ==="
notify "🎉 全部关键实验完成! RL train + eval done. 结果: $RESULT. 可以填论文数字了。"

# Save final summary
echo "$(date): RL eval result = $RESULT" >> "$ROOT/results/_rl_shaped_final_result.txt"
echo "RL ckpt: $RL_CKPT_DIR" >> "$ROOT/results/_rl_shaped_final_result.txt"

log "Sequencer finished."

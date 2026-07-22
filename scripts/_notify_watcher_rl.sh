#!/usr/bin/env bash
# _notify_watcher_rl.sh — Watch for RL shaped reward training completion
# Launch: nohup bash scripts/_notify_watcher_rl.sh > logs/_notify_watcher_rl.log 2>&1 &
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82"
log(){ echo "[notify $(date +%H:%M:%S)] $*"; }

notify(){
    local msg="$1"
    curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
        -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"$msg\"}}" >/dev/null 2>&1
    log "Sent: $msg"
}

NOTIFIED_RL_DONE=""
NOTIFIED_RL_EVAL=""

while true; do
    # --- Check RL training completion ---
    if [[ -z "$NOTIFIED_RL_DONE" ]]; then
        # Look for RL ckpt dirs with train_log
        rl_dirs=$(find ckpt -maxdepth 1 -name "s3_token_scorer_rl_shaped*" -type d 2>/dev/null)
        if [[ -n "$rl_dirs" ]]; then
            # Check if any shard has finished (last line of train_log has "epoch":2)
            for d in $rl_dirs; do
                if [[ -f "$d/train_log.jsonl" ]]; then
                    last_epoch=$("$PY" -c "
import json
lines = open('$d/train_log.jsonl').readlines()
if lines:
    last = json.loads(lines[-1])
    print(last.get('epoch', -1))
else:
    print(-1)
" 2>/dev/null)
                    if [[ "$last_epoch" == "2" ]]; then
                        # Get latest reward
                        reward=$("$PY" -c "
import json
lines = open('$d/train_log.jsonl').readlines()
last = json.loads(lines[-1])
print(f\"epoch={last.get('epoch')}, mean_reward={last.get('mean_reward', 'N/A')}, n_scenes={last.get('n_scenes_processed', 'N/A')}\")
" 2>/dev/null)
                        notify "[TokenRL] ✅ RL训练完成! $d — $reward. 请启动P2 eval。"
                        NOTIFIED_RL_DONE="done"
                        break
                    fi
                fi
            done
        fi
    fi

    # --- Check RL eval completion ---
    if [[ -z "$NOTIFIED_RL_EVAL" ]]; then
        n_rl_csv=$(ls results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh{0,1,2,3}.csv 2>/dev/null | wc -l)
        if [[ "$n_rl_csv" -ge 4 ]]; then
            pdms=$("$PY" -c "
import pandas as pd
dfs=[]
for sh in range(4):
    df=pd.read_csv('results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh'+str(sh)+'.csv')
    df=df[df['token']!='average']
    dfs.append(df)
c=pd.concat(dfs)
print(f'{c[\"score\"].mean():.6f} (N={len(c)})')
" 2>/dev/null)
            sft_diff=$("$PY" -c "
import pandas as pd
dfs=[]
for sh in range(4):
    df=pd.read_csv('results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh'+str(sh)+'.csv')
    df=df[df['token']!='average']
    dfs.append(df)
c=pd.concat(dfs)
rl=c['score'].mean()
print(f'RL={rl:.6f}, SFT=0.8920, gain={rl-0.8920:+.4f}pt {\"✅RL>SFT!\" if rl>0.8920 else \"❌RL<SFT\"} ')
" 2>/dev/null)
            notify "[TokenRL] ✅ RL eval 完成! PDMS=$pdms | $sft_diff"
            NOTIFIED_RL_EVAL="done"
        fi
    fi

    # --- Check if no GPU processes (training may have died) ---
    if [[ -z "$NOTIFIED_RL_DONE" ]]; then
        if ! pgrep -f "train_scorer_grpo" >/dev/null 2>&1; then
            # Check if it already finished or died
            rl_dirs=$(find ckpt -maxdepth 1 -name "s3_token_scorer_rl_shaped*" -type d 2>/dev/null)
            if [[ -n "$rl_dirs" ]]; then
                notify "[TokenRL] ⚠️ train_scorer_grpo 进程不在了！检查是完成还是崩溃。logs/ 下查看。"
                NOTIFIED_RL_DONE="warned"
            fi
        fi
    fi

    # All done?
    if [[ -n "$NOTIFIED_RL_DONE" && -n "$NOTIFIED_RL_EVAL" ]]; then
        log "All RL notifications sent. Exiting."
        notify "[TokenRL] 🎉 RL 全流程完成（训练+评测）。可以填论文数字了！"
        break
    fi

    sleep 300  # check every 5 min
done

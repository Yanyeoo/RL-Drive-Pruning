#!/bin/bash
# 看门狗：等 L8/L16/L20/L24 4 个 layer per-head 全部跑完后，自动跑分析 + 写 docs。
# 不会 kill 任何东西。如果哪一个 layer 卡了，会写一条 STALL 行到 log 就退出，不会瞎搞。

set -u
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
SWEEP=exp/m1a_perhead_L12
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
LOG=logs/m1b_autofinalize.log
DEADLINE=$(($(date +%s) + 3600))   # 1h 截止，超时自动放弃

echo "[wd] $(date) waiting for L8/L16/L20/L24 to finish (100 .pt each)" >> $LOG

while true; do
    done_all=1
    for L in L8 L16 L20 L24; do
        c=$(ls $SWEEP/${L}_perhead/*.pt 2>/dev/null | wc -l)
        if [ "$c" -lt 100 ]; then done_all=0; fi
    done
    if [ $done_all -eq 1 ]; then
        echo "[wd] $(date) all 4 layers done, running analyzer" >> $LOG
        break
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[wd] $(date) STALL — deadline reached, aborting auto-finalize" >> $LOG
        exit 0
    fi
    sleep 30
done

# 跑分析
$PY <<'PYEOF' >> logs/m1b_autofinalize.log 2>&1
import torch, glob, numpy as np, json, os
from scipy.stats import spearmanr, pearsonr

SWEEP = "exp/m1a_perhead_L12"
LAYERS = {
    "L8":  f"{SWEEP}/L8_perhead",
    "L12": f"{SWEEP}/L12_perhead",
    "L16": f"{SWEEP}/L16_perhead",
    "L20": f"{SWEEP}/L20_perhead",
    "L24": f"{SWEEP}/L24_perhead",
    "L27": f"{SWEEP}/L27_perhead",
}

def per_head(d):
    files = sorted(glob.glob(d + "/*.pt"))
    A = np.stack([
        torch.load(f, map_location='cpu', weights_only=False)['vision_attn'].sum(dim=-1).numpy()
        for f in files
    ], axis=0)
    return A

summary = {}
print("\n=== M1.b₀ full layer landscape ===")
print(f"{'layer':>5} {'shape':>14} {'g_mean':>8} {'dead_heads':>22} {'near_dead':>22} {'top4_share':>12} {'top12_cum':>10} {'eff_heads':>10}")
for L, d in LAYERS.items():
    if not os.path.isdir(d): continue
    A = per_head(d)
    m = A.mean(0)
    g = A.mean()
    order = np.argsort(-m)
    sm = np.sort(m)[::-1]; cum = np.cumsum(sm)/m.sum()
    dead = [int(h) for h in range(len(m)) if m[h] < 0.001]
    near = [int(h) for h in range(len(m)) if 0.001 <= m[h] < 0.05]
    hi = m/m.sum(); eff = float(np.exp(-(hi*np.log(hi+1e-12)).sum()))
    top4 = float(sm[:4].sum()/m.sum())
    top12 = float(cum[11]) if len(cum)>=12 else float(cum[-1])
    summary[L] = dict(
        shape=list(A.shape), g_mean=float(g),
        head_means=[float(x) for x in m],
        order_desc=[int(x) for x in order],
        dead=dead, near_dead=near,
        top4_share=top4, top12_cum=top12, eff_heads=eff,
    )
    print(f"{L:>5} {str(A.shape):>14} {g:>8.4f} {str(dead):>22} {str(near):>22} {top4:>12.4f} {top12:>10.4f} {eff:>10.2f}")

# 跨 layer top-4 是否重合？
print("\n=== top-4 head overlap across layers ===")
keys = list(summary.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a = set(summary[keys[i]]['order_desc'][:4])
        b = set(summary[keys[j]]['order_desc'][:4])
        print(f"{keys[i]} ∩ {keys[j]}  = {sorted(a&b)}  ({len(a&b)}/4)")

with open("exp/m1a_perhead_L12/landscape_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved → exp/m1a_perhead_L12/landscape_summary.json")
PYEOF

echo "[wd] $(date) analysis done" >> $LOG

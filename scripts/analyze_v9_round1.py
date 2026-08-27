#!/usr/bin/env python3
"""v9 Round1 gate 结果解析 + winner 选择。

用法:
    python3 scripts/analyze_v9_round1.py [cycle_dir]

默认读取 results/raw/v7_surrogate_20260824_v9r1_gate/ 下 4 个 arm 的
v7_gate_<arm>_sh0.csv / sh1.csv，计算 gate PDMS（两 shard 简单平均），
与 v7 st_topk 基线对比，选出 winner。
"""
import os
import sys

import pandas as pd

CYCLE = sys.argv[1] if len(sys.argv) > 1 else "v7_surrogate_20260824_v9r1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_DIR = os.path.join(ROOT, "results", "raw", f"{CYCLE}_gate")

ARMS = ["st_topk_tau005", "st_topk_ep2", "st_topk_pgw2", "st_topk_kl005"]

# 基线（来自 v7 / 历史）
BASELINE = {
    "SFT_scorer_r0.5": 0.89199,
    "v7_st_topk_gate": 0.894656,   # v7 winner gate (sh0+sh1 简单平均)
    "v7_st_topk_full": 0.894798,   # v7 winner full 4-shard
    "SOTA_no_prune": 0.89879,      # 上界
}


def shard_score(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    avg = df[df["token"] == "average"]
    if avg.empty:
        # 退化情况：用非 average 行的 mean
        return float(df[df["token"] != "average"]["score"].mean())
    return float(avg["score"].iloc[0])


def main() -> None:
    rows = []
    for arm in ARMS:
        s0 = os.path.join(GATE_DIR, f"v7_gate_{arm}_sh0.csv")
        s1 = os.path.join(GATE_DIR, f"v7_gate_{arm}_sh1.csv")
        if not (os.path.isfile(s0) and os.path.isfile(s1)):
            rows.append((arm, None, None, None, "MISSING"))
            continue
        v0 = shard_score(s0)
        v1 = shard_score(s1)
        gate = (v0 + v1) / 2.0
        rows.append((arm, v0, v1, gate, ""))

    print(f"=== v9 Round1 gate 结果 ({CYCLE}) ===")
    print(f"{'arm':<16} {'sh0':>10} {'sh1':>10} {'gate_mean':>11}  vs_full0.894798")
    valid = []
    for arm, v0, v1, gate, note in rows:
        if note:
            print(f"{arm:<16} {'-':>10} {'-':>10} {'-':>11}  {note}")
            continue
        delta = gate - BASELINE["v7_st_topk_full"]
        print(f"{arm:<16} {v0:>10.6f} {v1:>10.6f} {gate:>11.6f}  {delta:+.6f}")
        valid.append((arm, gate))

    print("\n=== 基线 ===")
    for k, v in BASELINE.items():
        print(f"  {k:<20} {v:.6f}")

    if not valid:
        print("\n[!] 无完整 gate 结果，评估可能仍在进行。")
        return

    valid.sort(key=lambda x: x[1], reverse=True)
    winner_arm, winner_gate = valid[0]
    print(f"\n=== WINNER: {winner_arm} (gate={winner_gate:.6f}) ===")
    print(f"  超过 v7 full 基线: {winner_gate - BASELINE['v7_st_topk_full']:+.6f}")
    print(f"  距离 SOTA no-prune: {BASELINE['SOTA_no_prune'] - winner_gate:+.6f}")
    print(f"\n  -> full 评估命令: EVAL_WORKERS=1 EVAL_GPUS=\"0 1 2 3\" bash scripts/eval_v7_folds_4gpu.sh {CYCLE} full {winner_arm}")


if __name__ == "__main__":
    main()

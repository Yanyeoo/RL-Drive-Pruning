#!/usr/bin/env python
"""build_v11_scene_list.py — v11 阶段 2：把 mining 结果组装成 hard-enriched 训练场景表。

输入：results/mining_<cycle>/mine_shard*.jsonl（run_v11_mining_4gpu.sh 产出）
      每行 {token, pruned_pdms, noprune_pdms, delta, keep_ratio, N, B, ...}

设计依据（v10 r1 诊断）：
  剪枝的总损失高度集中在少量「灾难场景」（pruned 崩盘而 no-prune 成功）。
  因此训练分布应当：
    1. 显著加权灾难场景（delta <= -catastrophic-thresh），让 budget head 学会
       在这些场景把 keep_ratio 抬上去 / 保住关键 token；
    2. 仍保留一部分正常场景作为「锚」，防止策略退化成「所有场景都不剪」——
       那样 efficiency 就没了，违背 SOTA+efficiency 双目标。

输出：一个 token 列表文本文件（每行一个 token，重复行 = 过采样权重），
      可直接传给 train_scorer_budget_rl.py --scene-list。

用法：
  python scripts/build_v11_scene_list.py \
    --mine-glob 'results/mining_20260826_v11/mine_shard*.jsonl' \
    --out results/mining_20260826_v11/scenes_hard.txt \
    --n-total 512 --hard-frac 0.5 --hard-repeat 2
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")


def load_records(pattern):
    files = sorted(glob.glob(pattern if Path(pattern).is_absolute() else str(ROOT / pattern)))
    if not files:
        raise SystemExit(f"[v11] no mining files match: {pattern}")
    recs, seen = [], set()
    for f in files:
        for line in Path(f).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a reclaimed run
            if r["token"] in seen:
                continue
            seen.add(r["token"])
            recs.append(r)
    print(f"[v11] loaded {len(recs)} unique scenes from {len(files)} shard files")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-glob", default="results/mining_20260826_v11/mine_shard*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-total", type=int, default=512,
                    help="Number of DISTINCT scenes in the training list (before repeats)")
    ap.add_argument("--hard-frac", type=float, default=0.5,
                    help="Fraction of the list drawn from catastrophic/negative-delta scenes")
    ap.add_argument("--hard-repeat", type=int, default=2,
                    help="How many times each hard scene is repeated (oversampling weight)")
    ap.add_argument("--catastrophic-thresh", type=float, default=0.3,
                    help="delta <= -thresh counts as catastrophic")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    recs = load_records(args.mine_glob)
    rng = np.random.RandomState(args.seed)

    deltas = np.array([r["delta"] for r in recs])
    krs = np.array([r["keep_ratio"] for r in recs])
    cat = [r for r in recs if r["delta"] <= -args.catastrophic_thresh]
    neg = [r for r in recs if -args.catastrophic_thresh < r["delta"] < -1e-9]
    zero = [r for r in recs if abs(r["delta"]) <= 1e-9]
    pos = [r for r in recs if r["delta"] > 1e-9]

    print("\n=== mined delta distribution (pruned - no-prune, deterministic policy) ===")
    print(f"  scenes            : {len(recs)}")
    print(f"  mean delta        : {deltas.mean():+.5f}   (sum {deltas.sum():+.2f})")
    print(f"  keep_ratio        : mean {krs.mean():.4f}  min {krs.min():.4f}  max {krs.max():.4f}")
    print(f"  catastrophic (<=-{args.catastrophic_thresh}): {len(cat):5d} "
          f"({len(cat)/len(recs)*100:.2f}%)  sum_delta {sum(r['delta'] for r in cat):+.2f}")
    print(f"  mild negative     : {len(neg):5d} ({len(neg)/len(recs)*100:.2f}%)  "
          f"sum_delta {sum(r['delta'] for r in neg):+.2f}")
    print(f"  exactly zero      : {len(zero):5d} ({len(zero)/len(recs)*100:.2f}%)  <- pruning is free")
    print(f"  positive (beats)  : {len(pos):5d} ({len(pos)/len(recs)*100:.2f}%)  "
          f"sum_delta {sum(r['delta'] for r in pos):+.2f}")
    neg_total = -deltas.clip(max=0).sum()
    if neg_total > 0:
        share = -sum(r["delta"] for r in cat) / neg_total * 100
        print(f"  catastrophic share of total negative delta: {share:.1f}%")

    # ---- hard pool: catastrophic first, then the worst mild-negative scenes ----
    n_hard = int(round(args.n_total * args.hard_frac))
    hard_pool = sorted(cat + neg, key=lambda r: r["delta"])  # most negative first
    hard = hard_pool[:n_hard]
    if len(hard) < n_hard:
        print(f"[v11] WARNING only {len(hard)} negative-delta scenes available "
              f"(wanted {n_hard}); filling the rest with normal scenes")

    # ---- normal pool: sample zero/positive scenes so efficiency is preserved ----
    n_normal = args.n_total - len(hard)
    normal_pool = zero + pos
    if len(normal_pool) <= n_normal:
        normal = normal_pool
    else:
        idx = rng.choice(len(normal_pool), n_normal, replace=False)
        normal = [normal_pool[i] for i in idx]

    lines = []
    for r in hard:
        lines.extend([r["token"]] * args.hard_repeat)
    lines.extend(r["token"] for r in normal)
    rng.shuffle(lines)

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# v11 hard-enriched scene list\n"
        f"# mined={len(recs)} hard={len(hard)}(x{args.hard_repeat}) normal={len(normal)} "
        f"lines={len(lines)}\n"
        f"# hard mean delta={np.mean([r['delta'] for r in hard]):+.4f} "
        f"normal mean delta={np.mean([r['delta'] for r in normal]):+.4f}\n"
    )
    out.write_text(header + "\n".join(lines) + "\n")

    print(f"\n[v11] wrote {out}")
    print(f"  distinct scenes : {len(hard) + len(normal)} "
          f"(hard {len(hard)} + normal {len(normal)})")
    print(f"  total lines     : {len(lines)} (hard repeated x{args.hard_repeat})")
    dup = Counter(lines).most_common(1)
    if dup:
        print(f"  max repeat      : {dup[0][1]}")


if __name__ == "__main__":
    main()

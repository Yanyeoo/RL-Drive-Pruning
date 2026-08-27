#!/usr/bin/env python
"""v10 SOTA headroom diagnosis.

Goal: understand *where* the gap to no-prune comes from, and whether pruning can
ever beat no-prune (delta > 0 scenes), to inform the value-baseline + keep_ratio
uprising round.

Compares (per-scene, token-aligned):
  - no-prune r=1.0   : results/raw/tokenprune_S3_full/MT_attn_L12_r10_sh*.csv
  - SFT  r=0.5       : results/raw/tokenprune_S3_full/MT_scorer_r05_sh*.csv
  - SFT  r=0.75      : results/raw/tokenprune_S3_full/MT_scorer_r075_sh*.csv
  - RL v7 st_topk    : results/raw/v7_surrogate_20260818_184128_full/v7_full_st_topk_sh*.csv

Outputs:
  1. per-shard PDMS table
  2. per-scene delta(v7 - no-prune) distribution + how many scenes beat no-prune
  3. top catastrophic scenes (largest negative delta)
  4. top beneficial scenes (largest positive delta)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = "/apdcephfs/private_shayladeng/tokenrl_autoVLA"


def load_method(pat, label):
    files = sorted(glob.glob(os.path.join(ROOT, pat)))
    if not files:
        raise SystemExit(f"[diag] no files match: {pat}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.token != "average"].drop_duplicates("token")
    df = df.set_index("token")
    df.columns = [f"{label}_{c}" for c in df.columns]
    return df, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-catastrophic", type=int, default=30)
    ap.add_argument("--n-beneficial", type=int, default=30)
    args = ap.parse_args()

    noprune, _ = load_method("results/raw/tokenprune_S3_full/MT_attn_L12_r10_sh*.csv", "noprune")
    sft05, _ = load_method("results/raw/tokenprune_S3_full/MT_scorer_r05_sh*.csv", "sft05")
    sft075, _ = load_method("results/raw/tokenprune_S3_full/MT_scorer_r075_sh*.csv", "sft075")
    v7, v7files = load_method("results/raw/v7_surrogate_20260818_184128_full/v7_full_st_topk_sh*.csv", "v7")

    m = noprune.join(sft05, how="inner").join(sft075, how="inner").join(v7, how="inner")
    print(f"[diag] token-aligned rows: {len(m)}")

    # ---- 1. per-shard PDMS ----
    print("\n=== per-shard PDMS (valid scenes only) ===")
    shard_of = {}
    for f in v7files:
        sid = int(os.path.basename(f).split("_sh")[1].split(".")[0])
        for t in pd.read_csv(f).token.unique():
            if t != "average":
                shard_of[t] = sid

    def pcol(label):
        return f"{label}_score"

    rows = []
    for sid in sorted(set(shard_of.values())):
        tokens = [t for t in m.index if shard_of.get(t) == sid]
        sub = m.loc[tokens]
        sub = sub[sub["v7_valid"] & sub["noprune_valid"]]
        rows.append({
            "shard": sid,
            "n": len(sub),
            "no-prune": sub[pcol("noprune")].mean(),
            "SFT r0.5": sub[pcol("sft05")].mean(),
            "SFT r0.75": sub[pcol("sft075")].mean(),
            "RL v7": sub[pcol("v7")].mean(),
            "v7-no_prune": sub[pcol("v7")].mean() - sub[pcol("noprune")].mean(),
        })
    agg = pd.DataFrame(rows)
    print(agg.round(6).to_string(index=False))

    total = agg
    print("\n--- totals ---")
    print(f"no-prune:  {total['no-prune'].mean():.6f}")
    print(f"SFT r0.5:  {total['SFT r0.5'].mean():.6f}")
    print(f"SFT r0.75: {total['SFT r0.75'].mean():.6f}")
    print(f"RL v7:     {total['RL v7'].mean():.6f}")

    # ---- 2. per-scene delta distribution ----
    valid = m[(m["v7_valid"]) & (m["noprune_valid"])].copy()
    valid["delta_v7"] = valid[pcol("v7")] - valid[pcol("noprune")]
    valid["delta_sft075"] = valid[pcol("sft075")] - valid[pcol("noprune")]

    print("\n=== per-scene delta (pruned - no-prune) on valid scenes ===")
    for col, label in [("delta_v7", "RL v7"), ("delta_sft075", "SFT r0.75")]:
        d = valid[col]
        beats = (d > 0).sum()
        print(f"{label:12s}: mean={d.mean():+.4f}  std={d.std():.4f}  "
              f"min={d.min():+.4f}  max={d.max():+.4f}  "
              f"beats_no_prune={beats}/{len(d)} ({beats/len(d)*100:.1f}%)")

    # catastrophic: delta < -0.3
    cat = valid[valid["delta_v7"] < -0.3].sort_values("delta_v7")
    print(f"\n=== catastrophic scenes (delta_v7 < -0.3): {len(cat)} ===")
    print(cat[["delta_v7", pcol("v7"), pcol("noprune")]].round(4).head(15).to_string())

    # beneficial: delta > +0.05
    ben = valid[valid["delta_v7"] > 0.05].sort_values("delta_v7", ascending=False)
    print(f"\n=== beneficial scenes (delta_v7 > +0.05): {len(ben)} ===")
    print(ben[["delta_v7", pcol("v7"), pcol("noprune")]].round(4).head(15).to_string())

    # ---- 3. histograms ----
    print("\n=== delta_v7 histogram ===")
    hist, edges = np.histogram(valid["delta_v7"], bins=20, range=(-1.0, 0.2))
    for i in range(len(hist)):
        print(f"  [{edges[i]:+.2f}, {edges[i+1]:+.2f})  {hist[i]:5d}")

    # how much of the gap is concentrated in the tail?
    gap_total = -valid["delta_v7"].clip(upper=0).sum()
    gap_tail = -cat["delta_v7"].sum()
    print(f"\n=== gap concentration ===")
    print(f"total negative delta (sum): {gap_total:.3f}")
    print(f"catastrophic (<-0.3) share : {gap_tail:.3f} ({gap_tail/gap_total*100:.1f}%)")


if __name__ == "__main__":
    main()

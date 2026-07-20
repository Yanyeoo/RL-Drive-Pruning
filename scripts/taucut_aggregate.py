"""taucut_aggregate.py — Aggregate τ-cut results and compare with fixed-r baselines.

Usage:
  cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
  python scripts/taucut_aggregate.py [--full]  # --full for 4-shard; default = shard0 only
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
TAUCUT_DIR = ROOT / "results/raw/tokenprune_taucut"
MAIN_DIR = ROOT / "results/raw/tokenprune_S3_full"


def load_arm(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["token"] != "average"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Aggregate all 4 shards (else shard0 only)")
    args = parser.parse_args()

    shards = [0, 1, 2, 3] if args.full else [0]

    # τ-cut results
    print("=" * 70)
    print("τ-cut Results (Route B: calibrated MSE scorer + global threshold)")
    print("=" * 70)
    print()

    tau_tags = ["kr040", "kr050", "kr060", "kr070"]
    tau_vals = {"kr040": -0.1253, "kr050": -0.1487, "kr060": -0.1668, "kr070": -0.1840}
    target_kr = {"kr040": 0.40, "kr050": 0.50, "kr060": 0.60, "kr070": 0.70}

    results = {}
    for tag in tau_tags:
        pdms_list = []
        n_total = 0
        for sh in shards:
            csv = TAUCUT_DIR / f"TC_mse_tau_{tag}_sh{sh}.csv"
            if csv.exists():
                df = load_arm(csv)
                pdms_list.append((df["score"].sum(), len(df)))
                n_total += len(df)
        if pdms_list:
            weighted_pdms = sum(p * n for p, n in pdms_list) / sum(n for _, n in pdms_list) if n_total > 0 else 0
            # Wait — PDMS is per-scene mean; weighted sum needs total sum / total count
            total_sum = sum(p * n for p, n in pdms_list)
            weighted_pdms = total_sum / n_total if n_total > 0 else 0
            results[tag] = {"pdms": weighted_pdms, "n": n_total, "n_shards": len(pdms_list)}

    # Fixed-r baselines (full navtest or shard0 depending on mode)
    baselines = {}
    for sel, ratio, label in [
        ("scorer", "r05", "scorer r=0.5"),
        ("scorer", "r075", "scorer r=0.75"),
        ("attn_L12", "r10", "no-prune r=1.0"),
    ]:
        pdms_list = []
        n_total = 0
        for sh in shards:
            csv = MAIN_DIR / f"MT_{sel}_{ratio}_sh{sh}.csv"
            if csv.exists():
                df = load_arm(csv)
                pdms_list.append((df["score"].sum(), len(df)))
                n_total += len(df)
        if pdms_list:
            total_sum = sum(p * n for p, n in pdms_list)
            baselines[label] = {"pdms": total_sum / n_total, "n": n_total}

    # Print table
    print(f"{'Arm':<30} {'N':>6} {'PDMS':>10} {'Δ vs fixed r=0.5':>18} {'Status':>8}")
    print("-" * 78)

    ref_pdms = baselines.get("scorer r=0.5", {}).get("pdms", 0.8920)

    for tag in tau_tags:
        if tag in results:
            r = results[tag]
            delta = r["pdms"] - ref_pdms
            status = "✅ WIN" if delta > 0 else "❌"
            tau = tau_vals[tag]
            kr = target_kr[tag]
            label = f"τ-cut τ={tau:.4f} (kr≈{kr:.2f})"
            print(f"{label:<30} {r['n']:>6} {r['pdms']:>10.6f} {delta:>+18.4f} {status:>8}")

    print("-" * 78)
    for label, b in baselines.items():
        delta = b["pdms"] - ref_pdms
        print(f"{label:<30} {b['n']:>6} {b['pdms']:>10.6f} {delta:>+18.4f}")

    print()
    print("=" * 70)
    print("Gate Decision:")
    print(f"  Win condition: τ-cut @ mean_kr≈0.5 PDMS > {ref_pdms:.4f} (fixed scorer r=0.5)")
    if "kr050" in results:
        delta = results["kr050"]["pdms"] - ref_pdms
        if delta > 0:
            print(f"  RESULT: ✅ PASS — τ-cut (kr050) beats fixed r=0.5 by {delta:+.4f} pt")
            print(f"  → Proceed with Route B: full navtest τ-cut + paper upgrade")
        else:
            print(f"  RESULT: ❌ FAIL — τ-cut (kr050) loses to fixed r=0.5 by {delta:+.4f} pt")
            print(f"  → Route B negative. Revert to Route A.")
            print(f"  → But this is clean evidence: even calibrated scorer + τ-cut")
            print(f"    cannot beat fixed ratio. C2 = robust unlearnable proof.")
    else:
        print("  kr050 not yet available.")

    # Pareto analysis (if multiple τ values)
    print()
    print("Pareto analysis (τ-cut vs fixed-r):")
    print(f"  {'Method':<30} {'mean_kr':>8} {'PDMS':>10} {'FLOPs saving':>13}")
    print("  " + "-" * 65)
    # Fixed-r references
    if "no-prune r=1.0" in baselines:
        print(f"  {'no-prune r=1.0':<30} {'1.00':>8} {baselines['no-prune r=1.0']['pdms']:>10.6f} {'0%':>13}")
    if "scorer r=0.75" in baselines:
        print(f"  {'scorer r=0.75 (fixed)':<30} {'0.75':>8} {baselines['scorer r=0.75']['pdms']:>10.6f} {'16.9%':>13}")
    for tag in tau_tags:
        if tag in results:
            kr = target_kr[tag]
            saving = f"{(1-kr)*67.2:.1f}%"  # 67.2% of FLOPs are vision-token related
            label = f"τ-cut kr≈{kr:.2f}"
            print(f"  {label:<30} {kr:>8.2f} {results[tag]['pdms']:>10.6f} {saving:>13}")
    if "scorer r=0.5" in baselines:
        print(f"  {'scorer r=0.5 (fixed)':<30} {'0.50':>8} {baselines['scorer r=0.5']['pdms']:>10.6f} {'33.6%':>13}")


if __name__ == "__main__":
    main()

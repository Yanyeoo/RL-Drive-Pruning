#!/usr/bin/env python
"""m1b2 phase 2 v1 gate check.

Reads existing soft_eval.json from v0 runs and applies v1 acceptance gates:
  - G_v1_1 PRIMARY:  max_{h ∈ H_valid} AUROC_h(holdout) ≥ 0.65
                  AND max_{h ∈ H_valid} AUROC_h(shifted) ≥ 0.65
  - G_v1_4 SHIFT:   max_{h ∈ H_valid} AUROC_h(shifted) ≥ 0.60
  - H_valid = {h | 0.05 ≤ pos_rate_train_h ≤ 0.95}

Outputs JSON + markdown summary for the recovery / v1 results report.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

EXP = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_phase2_v0")

CELLS = [
    # (label,         dir,                       feat_dim)
    ("R1pp-P1 (96)",  "p1_full_20260626_154930", 96),
    ("R1pp-P2 (96)",  "p2_full_20260626_154951", 96),
    ("C1-P1 (192)",   "p1_C1_20260629_142400",   192),
    ("C1-P2 (192)",   "p2_C1_20260629_142500",   192),
    ("C2-P1 (768)",   "p1_C2_20260629_144700",   768),
    ("C2-P2 (768)",   "p2_C2_20260629_144800",   768),
]


def auroc_max_valid(per_head_auroc, h_valid):
    """Max AUROC across H_valid, treating NaN as -inf."""
    vals = []
    for h in h_valid:
        v = per_head_auroc[h]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append((h, v))
    if not vals:
        return None, None
    h_best, v_best = max(vals, key=lambda x: x[1])
    return h_best, v_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-lo", type=float, default=0.05)
    ap.add_argument("--pos-hi", type=float, default=0.95)
    ap.add_argument("--thr-primary", type=float, default=0.65, help="G_v1_1 threshold")
    ap.add_argument("--thr-shift", type=float, default=0.60, help="G_v1_4 threshold")
    ap.add_argument("--out-json", type=str, default=str(EXP / "m1b2_phase2_v1_gate_check.json"))
    ap.add_argument("--out-md", type=str, default=str(EXP / "m1b2_phase2_v1_results.md"))
    args = ap.parse_args()

    rows = []
    for label, sub, dim in CELLS:
        p = EXP / sub / "soft_eval.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        d = json.load(open(p))
        hold = d["splits"]["holdout"]
        shft = d["splits"]["shifted"]

        # H_valid 用 holdout 的 per_head_pos_rate (= train pos_rate proxy)
        pos = hold["per_head_pos_rate"]
        h_valid = [i for i, r in enumerate(pos) if args.pos_lo <= r <= args.pos_hi]
        h_drop = [i for i in range(len(pos)) if i not in h_valid]

        h_best_h, v_best_h = auroc_max_valid(hold["per_head_auroc"], h_valid)
        h_best_s, v_best_s = auroc_max_valid(shft["per_head_auroc"], h_valid)

        # Per-head detail for H_valid
        valid_details = []
        for h in h_valid:
            a_h = hold["per_head_auroc"][h]
            a_s = shft["per_head_auroc"][h]
            valid_details.append({
                "head": h,
                "pos_rate": pos[h],
                "auroc_holdout": a_h,
                "auroc_shifted": a_s,
            })

        g_v1_1 = (v_best_h is not None and v_best_h >= args.thr_primary
                  and v_best_s is not None and v_best_s >= args.thr_primary)
        g_v1_4 = (v_best_s is not None and v_best_s >= args.thr_shift)
        verdict = "PASS" if (g_v1_1 and g_v1_4) else "FAIL"

        rows.append({
            "cell": label,
            "dir": sub,
            "feat_dim": dim,
            "h_valid": h_valid,
            "h_dropped": h_drop,
            "auroc_holdout_max_valid": v_best_h,
            "auroc_holdout_max_valid_head": h_best_h,
            "auroc_shifted_max_valid": v_best_s,
            "auroc_shifted_max_valid_head": h_best_s,
            "G_v1_1_primary": g_v1_1,
            "G_v1_4_shift": g_v1_4,
            "verdict": verdict,
            "per_head_detail": valid_details,
        })

    # JSON dump
    json.dump({
        "thresholds": {
            "G_v1_1_primary": args.thr_primary,
            "G_v1_4_shift": args.thr_shift,
            "pos_rate_window": [args.pos_lo, args.pos_hi],
        },
        "cells": rows,
    }, open(args.out_json, "w"), indent=2)
    print(f"[saved] {args.out_json}")

    # Markdown report
    lines = []
    lines.append("# M1-B2 Phase 2 **v1 Results** — gate check verdict\n")
    lines.append(f"**Date**: 2026-06-29\n")
    lines.append(f"**Source**: 6 v0 soft_eval.json (no retraining)\n")
    lines.append(f"**Gates**:")
    lines.append(f"- G_v1_1 PRIMARY: `max_{{h∈H_valid}} AUROC_h ≥ {args.thr_primary}` on **both** holdout & shifted")
    lines.append(f"- G_v1_4 SHIFT: `max_{{h∈H_valid}} AUROC_h(shifted) ≥ {args.thr_shift}`")
    lines.append(f"- H_valid: pos_rate ∈ [{args.pos_lo}, {args.pos_hi}]\n")

    # H_valid derivation (use first cell to read pos_rate, all share same data so H_valid same)
    if rows:
        r0 = rows[0]
        lines.append(f"**H_valid derivation** (from `{r0['cell']}` holdout pos_rate):")
        lines.append(f"- H_valid = {r0['h_valid']} (count={len(r0['h_valid'])})")
        lines.append(f"- dropped = {r0['h_dropped']} (count={len(r0['h_dropped'])}, reason: pos_rate outside window)\n")

    # Summary table
    lines.append("## Summary table (sorted by cell)\n")
    lines.append("| cell | dim | holdout max AUROC (h=?) | shifted max AUROC (h=?) | G_v1_1 | G_v1_4 | **verdict** |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        hh = f"{r['auroc_holdout_max_valid']:.3f} (h{r['auroc_holdout_max_valid_head']})" if r['auroc_holdout_max_valid'] is not None else "n/a"
        ss = f"{r['auroc_shifted_max_valid']:.3f} (h{r['auroc_shifted_max_valid_head']})" if r['auroc_shifted_max_valid'] is not None else "n/a"
        g1 = "✅" if r["G_v1_1_primary"] else "❌"
        g4 = "✅" if r["G_v1_4_shift"] else "❌"
        v = f"**{r['verdict']}**"
        lines.append(f"| {r['cell']} | {r['feat_dim']} | {hh} | {ss} | {g1} | {g4} | {v} |")

    # Per-head detail for P1 cells (the spec-relevant ones)
    lines.append("\n## Per-head AUROC on H_valid (P1 cells, primary)\n")
    for r in rows:
        if "P1" not in r["cell"]:
            continue
        lines.append(f"### {r['cell']}")
        lines.append("| head | pos_rate | AUROC holdout | AUROC shifted |")
        lines.append("|---|---|---|---|")
        for d in r["per_head_detail"]:
            a_h = f"{d['auroc_holdout']:.3f}" if d['auroc_holdout'] is not None and not (isinstance(d['auroc_holdout'], float) and math.isnan(d['auroc_holdout'])) else "nan"
            a_s = f"{d['auroc_shifted']:.3f}" if d['auroc_shifted'] is not None and not (isinstance(d['auroc_shifted'], float) and math.isnan(d['auroc_shifted'])) else "nan"
            lines.append(f"| h{d['head']} | {d['pos_rate']:.3f} | {a_h} | {a_s} |")
        lines.append("")

    # Verdict per path
    lines.append("## Path-level verdict\n")
    paths = {}
    for r in rows:
        # R1pp-P1, R1pp-P2 → path R1pp
        pname = r["cell"].split("-")[0]
        paths.setdefault(pname, []).append(r)
    for pname, cells in paths.items():
        any_pass = any(c["verdict"] == "PASS" and "P1" in c["cell"] for c in cells)
        # Per spec: a path passes if at least one P1 cell passes all gates
        p1_cells = [c for c in cells if "P1" in c["cell"]]
        p1_pass = any(c["verdict"] == "PASS" for c in p1_cells)
        lines.append(f"- **{pname}**: P1 cell pass = {p1_pass}")

    lines.append("")
    Path(args.out_md).write_text("\n".join(lines))
    print(f"[saved] {args.out_md}")

    # Console summary
    print("\n=== verdict ===")
    for r in rows:
        vh = r['auroc_holdout_max_valid']
        vs = r['auroc_shifted_max_valid']
        sh = f"{vh:.3f}" if vh is not None else "nan"
        ss = f"{vs:.3f}" if vs is not None else "nan"
        print(f"  {r['cell']:18s} hold={sh}  shift={ss}  {r['verdict']}")


if __name__ == "__main__":
    main()

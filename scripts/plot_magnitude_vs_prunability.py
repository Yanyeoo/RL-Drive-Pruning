#!/usr/bin/env python
"""
Magnitude vs Prunability (path-A punchline analysis).
======================================================
Tests the core path-A claim: the cost of removing bot-K heads (ΔPDMS) is
predicted by LAYER DEPTH, NOT by attention MAGNITUDE (per-layer vision-attn
g_mean). If magnitude predicted prunability, high-g_mean layers would be the
least prunable. We show it does not:
  - L12 (g_mean=0.179, high) is FREE, but L27 (g_mean=0.180, ~same) is a CLIFF.
  - L16 (g_mean=0.026, low) is FREE, L24 (g_mean=0.046, low) is borderline.

Computes Spearman rank correlation of ΔPDMS vs {layer_idx, g_mean} (no scipy).
Reads ΔPDMS from docs/results/figures/landscape_data.json (auto-refreshed).
g_mean per layer from key_results §5.5.

Output: docs/results/figures/magnitude_vs_prunability.png
Run: <autovla-python> scripts/plot_magnitude_vs_prunability.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/apdcephfs/private_shayladeng/tokenrl_autoVLA"

# per-layer vision-attn g_mean, key_results §5.5
G_MEAN = {8: 0.103, 12: 0.179, 16: 0.026, 20: 0.095, 24: 0.046, 27: 0.180}


def spearman(xs, ys):
    """Spearman rank correlation, no external deps."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx * vy else float("nan")


data = json.load(open(os.path.join(ROOT, "docs/results/figures/landscape_data.json")))
pts = [p for p in data["points"] if p["dPDMS"] is not None and p["layer"] in G_MEAN]
layers = [p["layer"] for p in pts]
dpdms = [p["dPDMS"] for p in pts]
gmean = [G_MEAN[L] for L in layers]

rho_depth = spearman(layers, dpdms)
rho_mag = spearman(gmean, dpdms)
print(f"n={len(pts)} layers={layers}")
print(f"Spearman(ΔPDMS, layer_idx) = {rho_depth:+.3f}   (depth predicts cost)")
print(f"Spearman(ΔPDMS, g_mean)    = {rho_mag:+.3f}   (magnitude does NOT)")

color = {"free": "#2e7d32", "borderline": "#f9a825", "cliff": "#c62828",
         "pending": "#9e9e9e"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

# left: ΔPDMS vs g_mean (magnitude) — should show NO monotone trend
for p in pts:
    a1.scatter([G_MEAN[p["layer"]]], [p["dPDMS"]], s=140, color=color[p["verdict"]],
               edgecolor="black", linewidth=0.6, zorder=3)
    a1.annotate(f"L{p['layer']}", (G_MEAN[p["layer"]], p["dPDMS"]),
                textcoords="offset points", xytext=(6, 6), fontsize=9)
a1.axhspan(-0.001, 0.001, color="#2e7d32", alpha=0.08)
a1.axhline(0, color="gray", lw=0.8, ls="--")
a1.set_xlabel("per-layer vision-attn g_mean (magnitude)")
a1.set_ylabel("ΔPDMS (bot-4 removal)")
a1.set_title(f"magnitude does NOT predict cost\nSpearman(ΔPDMS, g_mean) = {rho_mag:+.3f}")
a1.grid(alpha=0.25)

# right: ΔPDMS vs layer depth — strong monotone
order = sorted(range(len(pts)), key=lambda i: layers[i])
a2.plot([layers[i] for i in order], [dpdms[i] for i in order], "-",
        color="#546e7a", lw=1.3, zorder=1)
for p in pts:
    a2.scatter([p["layer"]], [p["dPDMS"]], s=140, color=color[p["verdict"]],
               edgecolor="black", linewidth=0.6, zorder=3)
    a2.annotate(f"L{p['layer']}", (p["layer"], p["dPDMS"]),
                textcoords="offset points", xytext=(6, 6), fontsize=9)
a2.axhspan(-0.001, 0.001, color="#2e7d32", alpha=0.08)
a2.axhline(0, color="gray", lw=0.8, ls="--")
a2.set_xlabel("decoder layer index (depth)")
a2.set_ylabel("ΔPDMS (bot-4 removal)")
a2.set_title(f"depth DOES predict cost\nSpearman(ΔPDMS, layer_idx) = {rho_depth:+.3f}")
a2.grid(alpha=0.25)

fig.suptitle("Redundancy is layer-position-structural, not attention-magnitude-based (path-A spine)")
fig.text(0.01, 0.01, "some points may be fallback (mixed-protocol) until clean L24K4/L27K4 land; "
         "figure auto-refreshes.", fontsize=7, color="#555")
out = os.path.join(ROOT, "docs/results/figures/magnitude_vs_prunability.png")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(out, dpi=150)
print("saved ->", out)

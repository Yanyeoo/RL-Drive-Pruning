#!/usr/bin/env python
"""
Layer x Prunability Landscape (auto-discovering, path-A spine figure).
=====================================================================
Scans results/raw/M1b_freelunch_L{N}K4_* on disk, computes per-layer bot-4
head-removal ΔPDMS (4-shard weighted mean, dedup latest dir per shard), and
renders x=layer, y=ΔPDMS. Re-run any time to refresh as new variants land.

Fixed reference points (not run as L{N}K4):
  - L12 = Sc4 [0,6,13,14] = +0.0004 (§6.7, 2-shard).
  - L27 fallback = V2 (L12:{h13}+L27:{h0,h8,h9}) = -0.0440 (§6.1) — only used
    if L27K4 (clean) has not landed yet.
  - L24 fallback refs (V4 bot-3 / V3-V2 11-head) shown only if L24K4 absent.

Baseline V0 = 0.8980 (§6.7 2-shard ref).
Outputs:
  docs/results/figures/layer_prunability_landscape.png
  docs/results/figures/landscape_data.json   (machine-readable table)
Run: <autovla-python> scripts/plot_layer_prunability_landscape.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/apdcephfs/private_shayladeng/tokenrl_autoVLA"
V0 = 0.8980
RAW = os.path.join(ROOT, "results/raw")


def four_shard_mean(layer):
    """Weighted mean PDMS over shards for M1b_freelunch_L{layer}K4_*.
    Shards are identified by manifest scene_filter (NOT the g-dir suffix, which
    is the GPU id and only spans {0,1} for 2-GPU runs). Keep latest dir/shard.
    """
    best = {}
    for m in glob.glob(os.path.join(RAW, f"M1b_freelunch_L{layer}K4_*", "manifest.json")):
        d = os.path.dirname(m)
        ag = os.path.join(d, "aggregate.json")
        if not os.path.exists(ag):
            continue
        try:
            man = json.load(open(m))
            a = json.load(open(ag))
        except Exception:
            continue
        sf = man.get("scene_filter", "")
        if "shard" not in sf or a.get("pdms") is None:
            continue
        s = sf.split("shard")[1][0]
        if s not in best or d > best[s][2]:
            best[s] = (a["pdms"], a.get("n_valid", 0), d)
    num = sum(p * n for p, n, _ in best.values())
    den = sum(n for _, n, _ in best.values())
    return (num / den if den else None), den, sorted(best)


def verdict(dp):
    if dp is None:
        return "pending"
    if abs(dp) <= 0.001:
        return "free"
    if dp >= -0.005:
        return "borderline"
    return "cliff"


# --- assemble points ---
CANDIDATE_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24, 25, 26, 27]
points = []  # (layer, dPDMS, n, shards, verdict, source)

for L in CANDIDATE_LAYERS:
    if L == 12:
        points.append((12, 0.0004, 5744, [0, 1], "free", "Sc4 §6.7 (2-shard)"))
        continue
    m, n, shards = four_shard_mean(L)
    if m is not None:
        dp = m - V0
        points.append((L, dp, n, shards, verdict(dp), f"disk L{L}K4"))
    elif L == 27:
        # fallback to V2 mixed cliff point until L27K4 lands
        points.append((27, -0.0440, 11574, [0, 1, 2, 3], "cliff", "V2 §6.1* (fallback)"))
    elif L == 24:
        points.append((24, -0.0032, 11575, [0, 1, 2, 3], "borderline", "V4 §6.1* (fallback)"))

points.sort()

# dump machine-readable
os.makedirs(os.path.join(ROOT, "docs/results/figures"), exist_ok=True)
data = [{"layer": L, "dPDMS": (round(dp, 5) if dp is not None else None),
         "n": n, "shards": sh, "verdict": v, "source": src}
        for (L, dp, n, sh, v, src) in points]
json.dump({"V0": V0, "points": data},
          open(os.path.join(ROOT, "docs/results/figures/landscape_data.json"), "w"),
          indent=2)
for row in data:
    print(row)

# --- plot ---
color = {"free": "#2e7d32", "borderline": "#f9a825", "cliff": "#c62828",
         "pending": "#9e9e9e"}
fig, ax = plt.subplots(figsize=(9, 5.4))
xs = [L for (L, dp, *_r) in points if dp is not None]
ys = [dp for (L, dp, *_r) in points if dp is not None]
ax.plot(xs, ys, "-", color="#546e7a", lw=1.3, zorder=1,
        label="bot-4 (K=4) per-layer")
for (L, dp, n, sh, v, src) in points:
    if dp is None:
        continue
    ax.scatter([L], [dp], s=130, color=color[v], zorder=3,
               edgecolor="black", linewidth=0.6)
    fb = "*" if "fallback" in src else ""
    ax.annotate(f"L{L}{fb}\n{dp:+.4f}", (L, dp), textcoords="offset points",
                xytext=(0, 12 if dp > -0.01 else -26), ha="center", fontsize=8)

ax.axhspan(-0.001, 0.001, color="#2e7d32", alpha=0.08, zorder=0)
ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
ax.text(0.3, 0.0013, "noise floor ±0.001 (free)", fontsize=7.5, color="#2e7d32")
ax.set_xlabel("Decoder layer index")
ax.set_ylabel("ΔPDMS vs V0 (0.8980)")
ax.set_title("Layer × Prunability Landscape — bot-4 head removal per layer\n"
             "cost rises toward output (redundancy is layer-position-structural)")
ax.set_xticks(CANDIDATE_LAYERS)
ax.grid(alpha=0.25)
legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color[k],
                 markeredgecolor="black", markersize=10, label=lab)
          for k, lab in [("free", "free (|Δ|≤0.001)"),
                         ("borderline", "borderline (−0.001..−0.005)"),
                         ("cliff", "cliff (< −0.005)")]]
ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)
fig.text(0.01, 0.01, "* = fallback point (mixed protocol) pending clean L{N}K4 run. "
         "L12 = Sc4 2-shard; others = 4-shard on-disk recompute.",
         fontsize=6.5, color="#555")
out = os.path.join(ROOT, "docs/results/figures/layer_prunability_landscape.png")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(out, dpi=150)
print("saved ->", out)

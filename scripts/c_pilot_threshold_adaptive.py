"""C-Pilot: Threshold-based emergent adaptive ratio (zero-training).

Goal: demonstrate that a single global threshold τ on scorer output naturally
produces scene-varying keep-ratios — and that this "emergent adaptive" strategy
Pareto-dominates fixed-ratio (e.g. r=0.5) or at least shows clear adaptivity.

Inputs (all offline, no GPU eval needed):
  1. Per-scene scorer scores: scorer(feature) → (720,) scores for 1495 navtest scenes.
  2. Per-scene PDMS at r ∈ {0.25, 0.5, 0.75, 1.0}: from existing CSVs.

Method:
  For each global τ (quantile of all scores pooled), compute:
    - per-scene keep_ratio(τ) = fraction of tokens with score > τ
    - per-scene PDMS(τ) ≈ linear interpolation among known {r: PDMS} anchor points
  Then aggregate: mean keep_ratio, mean PDMS.

Outputs:
  1. Figure: per-scene keep-ratio histogram at several τ values → shows adaptivity
  2. Figure: PDMS vs mean_keep_ratio Pareto, with fixed-r points overlaid
  3. Summary stats: variance of keep-ratio across scenes at each τ
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
SCORER_CKPT = ROOT / "ckpt/s3_token_scorer"
FEAT_DIR = ROOT / "data/s3_scorer/features_navtest_sub1500"
OUT_DIR = ROOT / "docs/results/figures/c_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Step 1: Load scorer and compute per-scene scores ----
print("[C-pilot] Loading scorer...")
sys.path.insert(0, str(ROOT / "code"))
from rldrive.scoring.token_scorer import ScorerRunner

scorer = ScorerRunner(str(SCORER_CKPT), device="cpu")

feat_files = sorted(FEAT_DIR.glob("*.pt"))
print(f"[C-pilot] {len(feat_files)} scenes with features")

scene_scores = {}  # token -> (720,) numpy scores
for f in feat_files:
    token = f.stem
    d = torch.load(f, map_location="cpu", weights_only=False)
    s = scorer.score(d["vision_feat"], d["vision_token_positions"], d["vision_blocks"])
    scene_scores[token] = s.numpy().flatten()

print(f"[C-pilot] Scored {len(scene_scores)} scenes")

# ---- Step 2: Load per-scene PDMS at known ratios ----
def load_csv(path):
    df = pd.read_csv(path)
    return df.set_index("token")["score"].to_dict()

pdms_r025 = load_csv(ROOT / "results/raw/tokenprune_S3/S3sub1500_scorer_r025.csv")
pdms_r050 = load_csv(ROOT / "results/raw/tokenprune_S3/S3sub1500_scorer_r050.csv")
pdms_r075 = load_csv(ROOT / "results/raw/tokenprune_S3/S3sub1500_scorer_r075.csv")
pdms_r100 = load_csv(ROOT / "results/raw/tokenprune_S2/S2sub1500_attnL12_r100.csv")

# Common tokens
common_tokens = sorted(
    set(scene_scores.keys()) &
    set(pdms_r025.keys()) &
    set(pdms_r050.keys()) &
    set(pdms_r075.keys()) &
    set(pdms_r100.keys())
)
print(f"[C-pilot] Common tokens with all PDMS: {len(common_tokens)}")

# Build per-scene PDMS lookup: {token: {ratio: pdms}}
ratios_known = np.array([0.25, 0.5, 0.75, 1.0])
per_scene_pdms = {}
for t in common_tokens:
    per_scene_pdms[t] = np.array([
        pdms_r025.get(t, np.nan),
        pdms_r050.get(t, np.nan),
        pdms_r075.get(t, np.nan),
        pdms_r100.get(t, np.nan),
    ])

# ---- Step 3: Pool all scores to define τ quantiles ----
all_scores = np.concatenate([scene_scores[t] for t in common_tokens])
print(f"[C-pilot] Pooled scores: N={len(all_scores)}, "
      f"mean={all_scores.mean():.4f}, std={all_scores.std():.4f}, "
      f"min={all_scores.min():.4f}, max={all_scores.max():.4f}")

# τ sweep: use quantiles of the pooled score distribution
tau_quantiles = np.arange(0.05, 0.95, 0.025)  # 36 points
tau_values = np.quantile(all_scores, tau_quantiles)

# ---- Step 4: For each τ, compute per-scene keep_ratio and interpolated PDMS ----
results = []

for tq, tau in zip(tau_quantiles, tau_values):
    keep_ratios = []
    pdms_interp = []
    
    for t in common_tokens:
        scores = scene_scores[t]
        kr = (scores > tau).mean()  # fraction kept
        kr = max(kr, 0.01)  # at least 1% (avoid 0)
        keep_ratios.append(kr)
        
        # Interpolate PDMS from known anchor points
        # Clamp to [0.25, 1.0] for interp
        kr_clamped = np.clip(kr, 0.25, 1.0)
        pdms_vals = per_scene_pdms[t]
        # numpy interp (needs sorted x)
        pdms_est = np.interp(kr_clamped, ratios_known, pdms_vals)
        pdms_interp.append(pdms_est)
    
    keep_ratios = np.array(keep_ratios)
    pdms_interp = np.array(pdms_interp)
    
    # Handle NaN PDMS
    valid = ~np.isnan(pdms_interp)
    
    results.append({
        "tau_quantile": tq,
        "tau_value": tau,
        "mean_keep_ratio": keep_ratios.mean(),
        "std_keep_ratio": keep_ratios.std(),
        "min_keep_ratio": keep_ratios.min(),
        "max_keep_ratio": keep_ratios.max(),
        "mean_pdms": pdms_interp[valid].mean() if valid.sum() > 0 else np.nan,
        "n_valid": int(valid.sum()),
        "keep_ratios": keep_ratios,  # for histogram
    })

print(f"\n[C-pilot] τ sweep done ({len(results)} points)")

# ---- Step 5: Fixed-ratio reference points ----
fixed_points = []
for r, pdms_dict in [(0.25, pdms_r025), (0.5, pdms_r050), (0.75, pdms_r075), (1.0, pdms_r100)]:
    vals = [pdms_dict[t] for t in common_tokens if t in pdms_dict and not np.isnan(pdms_dict.get(t, np.nan))]
    fixed_points.append({"ratio": r, "pdms": np.nanmean(vals), "n": len(vals)})

print("\n[C-pilot] Fixed-ratio reference points:")
for fp in fixed_points:
    print(f"  r={fp['ratio']:.2f}: PDMS={fp['pdms']:.6f} (N={fp['n']})")

# ---- Step 6: FIGURE 1 — keep-ratio histograms at selected τ ----
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
selected_tqs = [0.15, 0.30, 0.45, 0.55, 0.70, 0.85]

for ax, target_tq in zip(axes.flatten(), selected_tqs):
    # Find closest result
    idx = np.argmin([abs(r["tau_quantile"] - target_tq) for r in results])
    r = results[idx]
    ax.hist(r["keep_ratios"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(r["mean_keep_ratio"], color="red", linestyle="--", linewidth=2,
               label=f"mean={r['mean_keep_ratio']:.3f}")
    ax.set_title(f"τ at q={r['tau_quantile']:.2f} (τ={r['tau_value']:.3f})\n"
                 f"std={r['std_keep_ratio']:.3f}, range=[{r['min_keep_ratio']:.2f},{r['max_keep_ratio']:.2f}]",
                 fontsize=10)
    ax.set_xlabel("per-scene keep_ratio")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)

fig.suptitle("C-Pilot: Per-scene Keep-Ratio Distribution at Different Global Thresholds τ\n"
             "(high std = strong scene-adaptivity = 'emergent dynamic')", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "c_pilot_keepratio_histograms.png", dpi=150, bbox_inches="tight")
print(f"\n[C-pilot] Figure 1 saved: {OUT_DIR / 'c_pilot_keepratio_histograms.png'}")

# ---- Step 7: FIGURE 2 — Pareto: PDMS vs mean_keep_ratio ----
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))

# τ-cut curve
mean_krs = [r["mean_keep_ratio"] for r in results]
mean_pdms = [r["mean_pdms"] for r in results]
ax2.plot(mean_krs, mean_pdms, "o-", color="steelblue", markersize=4, linewidth=1.5,
         label="τ-cut (emergent adaptive)", zorder=3)

# Fixed-ratio points
for fp in fixed_points:
    ax2.scatter(fp["ratio"], fp["pdms"], s=120, marker="D", color="crimson", zorder=5)
    ax2.annotate(f"  r={fp['ratio']:.2f}\n  {fp['pdms']:.4f}", (fp["ratio"], fp["pdms"]),
                 fontsize=9, color="crimson")

ax2.set_xlabel("Mean Keep Ratio (lower = more efficient)", fontsize=12)
ax2.set_ylabel("Mean PDMS", fontsize=12)
ax2.set_title("C-Pilot: PDMS vs Mean Keep-Ratio\n"
              "τ-cut curve vs Fixed-Ratio points (scorer selector, navtest sub1500)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.1, 1.05)

fig2.savefig(OUT_DIR / "c_pilot_pareto.png", dpi=150, bbox_inches="tight")
print(f"[C-pilot] Figure 2 saved: {OUT_DIR / 'c_pilot_pareto.png'}")

# ---- Step 8: Summary table ----
print("\n" + "="*80)
print("[C-pilot] SUMMARY: τ-cut Pareto vs Fixed-ratio")
print("="*80)
print(f"{'τ_q':>6} {'τ_val':>8} {'mean_kr':>8} {'std_kr':>8} {'PDMS':>10} {'dominates r=0.5?':>18}")
print("-"*80)

r05_pdms = fixed_points[1]["pdms"]  # r=0.5 reference
for r in results:
    dominates = "✅ YES" if (r["mean_pdms"] >= r05_pdms and r["mean_keep_ratio"] < 0.5) or \
                            (r["mean_pdms"] > r05_pdms and r["mean_keep_ratio"] <= 0.5) else ""
    print(f"{r['tau_quantile']:>6.3f} {r['tau_value']:>8.4f} {r['mean_keep_ratio']:>8.3f} "
          f"{r['std_keep_ratio']:>8.3f} {r['mean_pdms']:>10.6f} {dominates:>18}")

print("-"*80)
print(f"\nFixed r=0.5 reference: PDMS={r05_pdms:.6f}")
print(f"Fixed r=0.25: PDMS={fixed_points[0]['pdms']:.6f}")
print(f"Fixed r=0.75: PDMS={fixed_points[2]['pdms']:.6f}")
print(f"Fixed r=1.0:  PDMS={fixed_points[3]['pdms']:.6f}")

# Key metric: at what mean_kr does τ-cut match r=0.5 PDMS?
for r in results:
    if r["mean_pdms"] >= r05_pdms and r["mean_keep_ratio"] < 0.5:
        print(f"\n🎯 DOMINATION POINT: τ_q={r['tau_quantile']:.3f}, "
              f"mean_kr={r['mean_keep_ratio']:.3f}, PDMS={r['mean_pdms']:.6f} ≥ r=0.5 ({r05_pdms:.6f})")
        break
else:
    print("\n⚠️ No τ-cut point strictly dominates r=0.5 (lower keep-ratio AND higher PDMS)")
    # Find best trade-off
    best_idx = np.argmax([r["mean_pdms"] - r05_pdms for r in results 
                          if r["mean_keep_ratio"] < 0.5] or [0])
    if best_idx < len(results):
        print(f"   Closest approach: τ_q={results[best_idx]['tau_quantile']:.3f}, "
              f"mean_kr={results[best_idx]['mean_keep_ratio']:.3f}, "
              f"PDMS={results[best_idx]['mean_pdms']:.6f}")

# Adaptivity metric
print(f"\n📊 ADAPTIVITY METRICS (key for paper):")
for tq_target in [0.3, 0.5, 0.7]:
    idx = np.argmin([abs(r["tau_quantile"] - tq_target) for r in results])
    r = results[idx]
    print(f"  τ_q≈{r['tau_quantile']:.2f}: mean_kr={r['mean_keep_ratio']:.3f}, "
          f"std={r['std_keep_ratio']:.3f}, range=[{r['min_keep_ratio']:.2f},{r['max_keep_ratio']:.2f}] "
          f"(CoV={r['std_keep_ratio']/r['mean_keep_ratio']:.2f})")

# Save JSON summary
summary = {
    "tau_sweep": [{k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in r.items() if k != "keep_ratios"} for r in results],
    "fixed_points": [{k: float(v) if isinstance(v, (np.floating, float)) else v 
                      for k, v in fp.items()} for fp in fixed_points],
}
(OUT_DIR / "c_pilot_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[C-pilot] Summary JSON: {OUT_DIR / 'c_pilot_summary.json'}")
print("[C-pilot] DONE.")

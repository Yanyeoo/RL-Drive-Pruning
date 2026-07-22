"""analyze_taucut_dynamic.py — Analyze per-scene keep ratio under τ-cut to demonstrate dynamic pruning.

Proves that τ-cut produces scene-adaptive keep ratios (not uniform),
and correlates with scene difficulty (baseline PDMS).

Usage:
  cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
  python scripts/analyze_taucut_dynamic.py

Output:
  - results/analysis/taucut_dynamic_stats.json
  - results/analysis/taucut_dynamic_histogram.png
  - results/analysis/taucut_dynamic_scatter.png
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))

# ============================================================
# Config
# ============================================================
MSE_SCORER_CKPT = ROOT / "ckpt/s3_token_scorer_mse"
FEATURE_DIR = ROOT / "data/s3_scorer/features"
BASELINE_CSV_DIR = ROOT / "results/raw/tokenprune_S3_full"
OUTPUT_DIR = ROOT / "results/analysis"

# τ values from taucut_aggregate.py (calibrated to target mean keep-ratios)
TAU_VALUES = {
    "kr040": -0.1253,
    "kr050": -0.1487,
    "kr060": -0.1668,
    "kr070": -0.1840,
}

# Default: use kr060 (the best-performing τ-cut point, PDMS=0.8940)
DEFAULT_TAU_TAG = "kr060"


def load_scorer(ckpt_dir):
    """Load the MSE scorer model."""
    import torch
    sys.path.insert(0, str(ROOT / "code"))
    from rldrive.scoring.token_scorer import TokenImportanceScorer

    scorer = TokenImportanceScorer(emb_dim=2048, hidden=256, n_cam=3)
    ckpt_path = Path(ckpt_dir) / "checkpoint.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(ckpt_dir) / "scorer.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(ckpt_dir) / "best_model.pt"
    if not ckpt_path.exists():
        # List what's available, exclude norm files
        files = [f for f in Path(ckpt_dir).glob("*.pt")
                 if "norm" not in f.name]
        if files:
            ckpt_path = files[0]
        else:
            raise FileNotFoundError(f"No .pt file found in {ckpt_dir}")

    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        scorer.load_state_dict(state["model_state_dict"])
    elif "state_dict" in state:
        scorer.load_state_dict(state["state_dict"])
    else:
        scorer.load_state_dict(state)
    scorer.eval()
    print(f"Loaded scorer from {ckpt_path}")
    return scorer


def load_baseline_pdms():
    """Load per-scene baseline PDMS (r=1.0) from CSV files."""
    import pandas as pd

    baseline = {}
    for sh in range(4):
        csv_path = BASELINE_CSV_DIR / f"MT_attn_L12_r10_sh{sh}.csv"
        if not csv_path.exists():
            print(f"  [WARN] {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["token"] != "average"]
        for _, row in df.iterrows():
            if row.get("valid", True):
                baseline[row["token"]] = row["score"]
    print(f"Loaded baseline PDMS for {len(baseline)} scenes")
    return baseline


def compute_per_scene_keep_ratios(scorer, feature_dir, tau):
    """For each scene, compute scorer scores and count tokens above τ."""
    import torch

    feature_files = sorted(Path(feature_dir).glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"No .pt files in {feature_dir}")

    results = {}
    n_total = 720  # total vision tokens per scene

    print(f"Processing {len(feature_files)} scenes with τ={tau:.4f}...")
    for i, fp in enumerate(feature_files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(feature_files)}")

        data = torch.load(fp, map_location="cpu")

        # Extract features - handle different save formats
        if isinstance(data, dict):
            if "vision_features" in data:
                feats = data["vision_features"]  # (720, 2048)
            elif "features" in data:
                feats = data["features"]
            else:
                # Try first tensor-like value
                for v in data.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] == 720:
                        feats = v
                        break
                else:
                    print(f"  [WARN] Cannot parse {fp.name}, skipping")
                    continue
        elif isinstance(data, torch.Tensor):
            feats = data
        else:
            continue

        # Build input: [vision_feat (2048) ; cam_onehot (3)]
        # Camera assignment: 240 tokens per camera (front, front-left, front-right)
        n_tokens = feats.shape[0]
        cam_ids = torch.zeros(n_tokens, 3)
        cam_ids[:240, 0] = 1.0    # front
        cam_ids[240:480, 1] = 1.0  # front-left
        cam_ids[480:720, 2] = 1.0  # front-right

        x = torch.cat([feats, cam_ids], dim=-1)  # (720, 2051)

        with torch.no_grad():
            scores = scorer(x).squeeze(-1)  # (720,)

        # Count tokens above threshold
        keep_mask = scores > tau
        keep_count = keep_mask.sum().item()
        keep_ratio = keep_count / n_tokens

        # Scene token = filename stem
        scene_token = fp.stem
        results[scene_token] = {
            "keep_count": int(keep_count),
            "keep_ratio": float(keep_ratio),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_max": float(scores.max()),
            "score_min": float(scores.min()),
        }

    return results


def analyze_and_save(per_scene, baseline_pdms, tau_tag, output_dir):
    """Compute statistics and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Merge with baseline PDMS
    keep_ratios = []
    baseline_scores = []
    all_data = []

    for token, info in per_scene.items():
        kr = info["keep_ratio"]
        keep_ratios.append(kr)
        row = {"token": token, **info}
        if token in baseline_pdms:
            row["baseline_pdms"] = baseline_pdms[token]
            baseline_scores.append(baseline_pdms[token])
        all_data.append(row)

    keep_ratios = np.array(keep_ratios)

    # === Statistics ===
    stats = {
        "tau_tag": tau_tag,
        "tau_value": TAU_VALUES[tau_tag],
        "n_scenes": len(keep_ratios),
        "mean_keep_ratio": float(np.mean(keep_ratios)),
        "std_keep_ratio": float(np.std(keep_ratios)),
        "min_keep_ratio": float(np.min(keep_ratios)),
        "max_keep_ratio": float(np.max(keep_ratios)),
        "median_keep_ratio": float(np.median(keep_ratios)),
        "p10": float(np.percentile(keep_ratios, 10)),
        "p25": float(np.percentile(keep_ratios, 25)),
        "p75": float(np.percentile(keep_ratios, 75)),
        "p90": float(np.percentile(keep_ratios, 90)),
        "range": float(np.max(keep_ratios) - np.min(keep_ratios)),
    }

    # Binned analysis by baseline PDMS difficulty
    if baseline_scores:
        matched = [(d["keep_ratio"], d["baseline_pdms"])
                   for d in all_data if "baseline_pdms" in d]
        if matched:
            krs = np.array([m[0] for m in matched])
            bps = np.array([m[1] for m in matched])

            # Difficulty bins
            bins = {
                "hard (PDMS<0.7)": bps < 0.7,
                "medium (0.7-0.9)": (bps >= 0.7) & (bps < 0.9),
                "easy (PDMS>=0.9)": bps >= 0.9,
            }
            difficulty_analysis = {}
            for label, mask in bins.items():
                if mask.sum() > 0:
                    difficulty_analysis[label] = {
                        "n_scenes": int(mask.sum()),
                        "mean_keep_ratio": float(krs[mask].mean()),
                        "std_keep_ratio": float(krs[mask].std()),
                    }
            stats["difficulty_analysis"] = difficulty_analysis

            # Correlation
            from scipy import stats as scipy_stats
            corr, pval = scipy_stats.pearsonr(krs, bps)
            spearman_corr, spearman_p = scipy_stats.spearmanr(krs, bps)
            stats["correlation_with_baseline"] = {
                "pearson_r": float(corr),
                "pearson_p": float(pval),
                "spearman_rho": float(spearman_corr),
                "spearman_p": float(spearman_p),
            }

    # Save stats
    stats_path = output_dir / "taucut_dynamic_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("τ-CUT DYNAMIC PRUNING ANALYSIS")
    print("=" * 60)
    print(f"τ tag: {tau_tag} (τ = {TAU_VALUES[tau_tag]:.4f})")
    print(f"N scenes: {stats['n_scenes']}")
    print(f"Keep ratio: mean={stats['mean_keep_ratio']:.3f}, "
          f"std={stats['std_keep_ratio']:.3f}, "
          f"range=[{stats['min_keep_ratio']:.3f}, {stats['max_keep_ratio']:.3f}]")
    print(f"Percentiles: p10={stats['p10']:.3f}, p25={stats['p25']:.3f}, "
          f"median={stats['median_keep_ratio']:.3f}, p75={stats['p75']:.3f}, p90={stats['p90']:.3f}")

    if "difficulty_analysis" in stats:
        print(f"\nBy scene difficulty (baseline PDMS):")
        for label, info in stats["difficulty_analysis"].items():
            print(f"  {label}: N={info['n_scenes']}, "
                  f"mean keep={info['mean_keep_ratio']:.3f} ± {info['std_keep_ratio']:.3f}")

    if "correlation_with_baseline" in stats:
        c = stats["correlation_with_baseline"]
        print(f"\nCorrelation (keep_ratio vs baseline_PDMS):")
        print(f"  Pearson r={c['pearson_r']:.4f} (p={c['pearson_p']:.2e})")
        print(f"  Spearman ρ={c['spearman_rho']:.4f} (p={c['spearman_p']:.2e})")

    # === Plots ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Histogram of keep ratios
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(keep_ratios, bins=50, edgecolor="black", alpha=0.7, color="#2196F3")
        ax.axvline(np.mean(keep_ratios), color="red", linestyle="--",
                   label=f"Mean = {np.mean(keep_ratios):.3f}")
        ax.set_xlabel("Per-Scene Keep Ratio", fontsize=12)
        ax.set_ylabel("Number of Scenes", fontsize=12)
        ax.set_title(f"τ-cut ({tau_tag}): Per-Scene Keep Ratio Distribution\n"
                     f"std={stats['std_keep_ratio']:.3f}, range=[{stats['min_keep_ratio']:.2f}, {stats['max_keep_ratio']:.2f}]",
                     fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = output_dir / "taucut_dynamic_histogram.png"
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"\nSaved histogram to {hist_path}")

        # Scatter: keep_ratio vs baseline PDMS
        if baseline_scores and 'correlation_with_baseline' in stats:
            _matched = [(d["keep_ratio"], d["baseline_pdms"])
                        for d in all_data if "baseline_pdms" in d]
            if _matched:
                _krs = np.array([m[0] for m in _matched])
                _bps = np.array([m[1] for m in _matched])
                _bins = {
                    "hard (PDMS<0.7)": _bps < 0.7,
                    "medium (0.7-0.9)": (_bps >= 0.7) & (_bps < 0.9),
                    "easy (PDMS>=0.9)": _bps >= 0.9,
                }
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.scatter(_bps, _krs, alpha=0.3, s=10, color="#4CAF50")
                for label, mask in _bins.items():
                    if mask.sum() > 0:
                        ax.axhline(_krs[mask].mean(), color="orange", alpha=0.5, linestyle=":")
                ax.set_xlabel("Baseline PDMS (scene difficulty, higher = easier)", fontsize=12)
                ax.set_ylabel("τ-cut Keep Ratio", fontsize=12)
                c = stats["correlation_with_baseline"]
                ax.set_title(f"Scene Difficulty vs Token Retention\n"
                             f"Spearman ρ={c['spearman_rho']:.3f} (p={c['spearman_p']:.2e})",
                             fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                scatter_path = output_dir / "taucut_dynamic_scatter.png"
                plt.savefig(scatter_path, dpi=150)
                plt.close()
                print(f"Saved scatter to {scatter_path}")

    except ImportError as e:
        print(f"[WARN] Plotting skipped (missing library: {e})")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-tag", default=DEFAULT_TAU_TAG, choices=list(TAU_VALUES.keys()))
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--scorer-ckpt", default=str(MSE_SCORER_CKPT))
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit scenes for quick test")
    args = parser.parse_args()

    tau = TAU_VALUES[args.tau_tag]

    # Load scorer
    scorer = load_scorer(args.scorer_ckpt)

    # Compute per-scene keep ratios
    per_scene = compute_per_scene_keep_ratios(scorer, args.feature_dir, tau)

    if args.max_scenes:
        # Truncate for quick test
        keys = list(per_scene.keys())[:args.max_scenes]
        per_scene = {k: per_scene[k] for k in keys}

    # Load baseline PDMS
    try:
        import pandas as pd
        baseline_pdms = load_baseline_pdms()
    except Exception as e:
        print(f"[WARN] Could not load baseline PDMS: {e}")
        baseline_pdms = {}

    # Analyze and save
    analyze_and_save(per_scene, baseline_pdms, args.tau_tag, OUTPUT_DIR)


if __name__ == "__main__":
    main()

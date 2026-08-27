"""Variant B (True Token Drop) vs Variant A (Attn Mask) Speedup Profiling.

Measures per-scene inference latency to quantify REAL wall-clock speedup from
Variant B's sequence shortening (941 -> ~581 tokens at r=0.5).

Configurations compared:
  - r=1.0 (no pruning, 1-pass, baseline)
  - Variant A scorer r=0.5 (2-pass, attn mask, same seq length)
  - Variant B scorer r=0.5 (2-pass, true token drop, shorter seq)
  - Variant A scorer r=0.75 (2-pass, attn mask)
  - Variant B scorer r=0.75 (2-pass, true token drop)

Usage:
  cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
  source scripts/setup_navsim_env_vars.sh
  CUDA_VISIBLE_DEVICES=0 /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
    scripts/profile_varB_speedup.py --n-scenes 30 --warmup 5
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA/navsim"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA"))

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes", type=int, default=30, help="Number of scenes to profile")
    p.add_argument("--warmup", type=int, default=5, help="Warmup scenes (excluded from stats)")
    p.add_argument("--output", type=str, default="results/profiling/varB_speedup_profile.json")
    return p.parse_args()


def build_agent(selector, keep_ratio, prune_variant, scorer_ckpt=None):
    """Build the token-prune agent with given config."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from rldrive.agents.autovla_with_token_prune import AutoVLAWithTokenPruneAgent

    ts = TrajectorySampling(num_poses=8, interval_length=0.5)
    agent = AutoVLAWithTokenPruneAgent(
        trajectory_sampling=ts,
        checkpoint_path=str(ROOT / "models/AutoVLA/AutoVLA_PDMS_89.ckpt"),
        sensor_data_path=str(ROOT / "data/navsim_v2_local"),
        codebook_cache_path=str(ROOT / "code/third_party/AutoVLA/codebook_cache/agent_vocab.pkl"),
        config_path=str(ROOT / "code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"),
        device="cuda:0",
        keep_ratio=keep_ratio,
        selector=selector,
        scorer_ckpt=scorer_ckpt,
        prune_variant=prune_variant,
        prune_verbose=False,
    )
    return agent


def load_scenes(n_scenes):
    """Load scene tokens from navtest_nocot directory (first n_scenes by sorted name)."""
    json_dir = ROOT / "data/navtest_nocot"
    all_jsons = sorted(json_dir.glob("*.json"))
    tokens = [p.stem for p in all_jsons[:n_scenes]]
    print(f"[profile_varB] Using {len(tokens)} tokens from navtest_nocot (sorted)", flush=True)
    return tokens


def profile_config(agent, token_list, json_dir, warmup, config_name):
    """Profile a single configuration, return timing stats."""
    import json as _json
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    latencies = []
    n_errors = 0
    for i, token in enumerate(token_list):
        # Load scene data from JSON
        json_path = json_dir / f"{token}.json"
        with open(json_path, 'r') as f:
            scene_data = _json.load(f)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
            _ = agent.compute_trajectory(scene_data)
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  [{config_name}] scene {i} ERROR: {type(e).__name__}: {e}", flush=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        if i >= warmup:
            latencies.append(elapsed)

        if i < 3:
            print(f"  [{config_name}] scene {i} {'(warmup)' if i < warmup else ''}: {elapsed:.3f}s", flush=True)

    if n_errors:
        print(f"  [{config_name}] {n_errors} scenes had errors (still included in timing)", flush=True)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    latencies = np.array(latencies)
    stats = {
        "config": config_name,
        "n_measured": len(latencies),
        "mean_s": float(latencies.mean()),
        "std_s": float(latencies.std()),
        "p50_s": float(np.percentile(latencies, 50)),
        "p95_s": float(np.percentile(latencies, 95)),
        "min_s": float(latencies.min()),
        "max_s": float(latencies.max()),
        "peak_gpu_mb": float(peak_mem_mb),
    }
    print(f"  [{config_name}] mean={stats['mean_s']:.3f}s, "
          f"std={stats['std_s']:.3f}s, P50={stats['p50_s']:.3f}s, "
          f"peak_mem={stats['peak_gpu_mb']:.0f}MB", flush=True)
    return stats


def main():
    args = parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[profile_varB] Loading {args.n_scenes} scene tokens (warmup={args.warmup})...", flush=True)
    token_list = load_scenes(args.n_scenes)
    json_dir = ROOT / "data/navtest_nocot"
    print(f"[profile_varB] Got {len(token_list)} tokens", flush=True)

    scorer_ckpt = str(ROOT / "ckpt/s3_token_scorer")

    # Configs to profile: name, selector, keep_ratio, prune_variant, scorer_ckpt
    configs = [
        ("baseline_r1.0",       "scorer", 1.0,  "attn_mask", scorer_ckpt),
        ("varA_scorer_r0.5",    "scorer", 0.5,  "attn_mask", scorer_ckpt),
        ("varB_scorer_r0.5",    "scorer", 0.5,  "drop",      scorer_ckpt),
        ("varA_scorer_r0.75",   "scorer", 0.75, "attn_mask", scorer_ckpt),
        ("varB_scorer_r0.75",   "scorer", 0.75, "drop",      scorer_ckpt),
    ]

    all_stats = []

    for config_name, selector, keep_ratio, prune_variant, ckpt in configs:
        print(f"\n[profile_varB] === {config_name} (variant={prune_variant}, r={keep_ratio}) ===", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        agent = build_agent(selector, keep_ratio, prune_variant, ckpt)
        agent.autovla.eval()

        stats = profile_config(agent, token_list, json_dir, args.warmup, config_name)
        all_stats.append(stats)

        del agent
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 90)
    print("[profile_varB] SPEEDUP SUMMARY — Variant B (True Token Drop) vs Variant A (Attn Mask)")
    print("=" * 90)
    print(f"{'Config':<22} {'Mean(s)':<10} {'P50(s)':<10} {'P95(s)':<10} {'Peak(MB)':<12} {'vs r=1.0':<10} {'vs VarA':<10}")
    print("-" * 90)

    baseline_mean = next((s["mean_s"] for s in all_stats if "r1.0" in s["config"]), None)
    varA_r05_mean = next((s["mean_s"] for s in all_stats if "varA" in s["config"] and "r0.5" in s["config"]), None)
    varA_r075_mean = next((s["mean_s"] for s in all_stats if "varA" in s["config"] and "r0.75" in s["config"]), None)

    for s in all_stats:
        speedup_vs_r1 = baseline_mean / s["mean_s"] if baseline_mean and s["mean_s"] > 0 else 0
        # Speedup vs corresponding Variant A
        if "varB" in s["config"] and "r0.5" in s["config"]:
            speedup_vs_varA = varA_r05_mean / s["mean_s"] if varA_r05_mean and s["mean_s"] > 0 else 0
        elif "varB" in s["config"] and "r0.75" in s["config"]:
            speedup_vs_varA = varA_r075_mean / s["mean_s"] if varA_r075_mean and s["mean_s"] > 0 else 0
        else:
            speedup_vs_varA = 0

        varA_str = f"{speedup_vs_varA:.2f}x" if speedup_vs_varA > 0 else "—"
        print(f"{s['config']:<22} {s['mean_s']:<10.3f} {s['p50_s']:<10.3f} "
              f"{s['p95_s']:<10.3f} {s['peak_gpu_mb']:<12.0f} {speedup_vs_r1:<10.2f}x {varA_str:<10}")

    # Save JSON
    import json
    result = {
        "profiling": all_stats,
        "metadata": {
            "n_scenes": args.n_scenes,
            "warmup": args.warmup,
            "gpu": "NVIDIA H20 (97GB)",
            "model": "AutoVLA (Qwen2.5-VL-3B)",
            "n_vision_tokens": 720,
            "scorer": "LambdaRank (ckpt/s3_token_scorer)",
            "note": "Variant A = attention mask (no seq shortening). "
                    "Variant B = true token drop (seq 941 -> ~581 at r=0.5). "
                    "Both use the same LambdaRank scorer for token selection. "
                    "Baseline r=1.0 uses 1-pass (no scorer); pruned arms use 2-pass.",
        },
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[profile_varB] Results saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M1.b₂ Phase 2 prior: Per-scene per-head rank-variance analysis.

Reads all per-scene .pt tensors of shape (L=28, H=16, V=720) from
`exp/m1b2_navtrain_full_alllayers/*.pt`, computes:

  1. Per-scene per-layer head importance score = mean over V of attention.
  2. Per-scene per-layer head rank (argsort) ∈ [0, H-1].
  3. Across-scene statistics:
     - rank mean / rank std per (layer, head)
     - frequency of being in top-K (default K=4) per (layer, head)
     - frequency of being in bottom-K (default K=4) per (layer, head)
     - per-layer entropy of top-1 head distribution (how scene-variable is the winner)

Compares the M1.b₁ frozen-mask heads (L12:h13 / L24:{11}/ L27:{0,8,9}) against
the per-scene distribution to confirm or challenge their "always near-bottom"
property.

Outputs:
  - JSON:  exp/m1b2_rank_variance/rank_variance.json
  - NPZ:   exp/m1b2_rank_variance/rank_stats.npz  (rank_mean, rank_std, top_k_freq, bot_k_freq)
  - MD:    exp/m1b2_rank_variance/SUMMARY.md  (human-readable)
  - PNG:   exp/m1b2_rank_variance/heatmap_rank_std.png
           exp/m1b2_rank_variance/heatmap_top1_entropy.png   (per-layer bar)

Runtime: CPU-only is fine. Estimated 5–15 min for 19,225 × (28*16*720) fp32.

Usage:
  $PY scripts/m1b2_rank_variance.py
  $PY scripts/m1b2_rank_variance.py --src exp/m1b2_navtrain_full_alllayers --topk 4 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
DEFAULT_SRC = PROJECT_ROOT / "exp" / "m1b2_navtrain_full_alllayers"
DEFAULT_OUT = PROJECT_ROOT / "exp" / "m1b2_rank_variance"
DENYLIST_FILE = DEFAULT_SRC / "_stage3_trajectory_err_tokens.txt"

# M1.b₁ frozen masks (from key_results.md §5.4 / §6.1)
M1B1_MASKS = {
    "V1": {12: [13]},
    "V2": {12: [13], 27: [0, 8, 9]},
    "V3": {12: [13],
           24: [0, 1, 2, 6, 7, 8, 9, 10, 12, 14, 15],
           27: [0, 8, 9]},
}


def load_denylist(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def process_one(pt_path_str: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load one .pt, return (token_id, score (L,H), rank (L,H))."""
    pt_path = Path(pt_path_str)
    token_id = pt_path.stem
    # weights_only=True is safe for tensor-only files
    t = torch.load(pt_path, map_location="cpu", weights_only=True)
    # Expect shape (L=28, H=16, V=720). Some files may store dict — guard.
    if isinstance(t, dict):
        # try common keys
        for k in ("per_layer_vision_attn", "attn", "vision_attn"):
            if k in t:
                t = t[k]
                break
    if not torch.is_tensor(t):
        raise RuntimeError(f"{pt_path}: not a tensor (got {type(t)})")
    if t.ndim != 3:
        raise RuntimeError(f"{pt_path}: expected 3-d, got {tuple(t.shape)}")
    L, H, V = t.shape
    # importance = mean over V (raw vision attention magnitude per head)
    score = t.float().mean(dim=-1).numpy()  # (L, H)
    # rank: 0 = lowest, H-1 = highest, per layer
    # argsort ascending -> position 0 is smallest; we want each head's rank.
    order = np.argsort(score, axis=-1, kind="stable")  # (L, H) of head ids, ascending
    rank = np.empty_like(order)
    rows = np.arange(L)[:, None]
    rank[rows, order] = np.arange(H)[None, :]
    # rank.shape == (L, H), rank[l, h] ∈ [0, H-1]
    return token_id, score.astype(np.float32), rank.astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="Directory containing per-scene .pt files")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output directory")
    ap.add_argument("--topk", type=int, default=4,
                    help="K for top-K / bottom-K frequency counters")
    ap.add_argument("--workers", type=int, default=8,
                    help="ProcessPool workers")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only first N files (debug)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pt_files = sorted([p for p in src.glob("*.pt")])
    if args.limit:
        pt_files = pt_files[: args.limit]

    denylist = load_denylist(DENYLIST_FILE)
    # IMPORTANT: per Phase 2 design, we keep ALL .pt for rank-variance
    # (they ARE valid attention tensors; the assert was in downstream pose check).
    # Just record which are flagged.
    flagged = sum(1 for p in pt_files if p.stem in denylist)
    print(f"[rv] src={src}", flush=True)
    print(f"[rv] N files = {len(pt_files)} ({flagged} in denylist, kept anyway)",
          flush=True)
    print(f"[rv] topk = {args.topk}", flush=True)
    print(f"[rv] workers = {args.workers}", flush=True)

    if not pt_files:
        print("[rv] FATAL: no .pt found", file=sys.stderr)
        return 2

    # ---- streaming aggregation ----
    L, H = 28, 16
    K = args.topk

    # Accumulators
    rank_sum = np.zeros((L, H), dtype=np.float64)
    rank_sqsum = np.zeros((L, H), dtype=np.float64)
    top_k_count = np.zeros((L, H), dtype=np.int64)
    bot_k_count = np.zeros((L, H), dtype=np.int64)
    score_sum = np.zeros((L, H), dtype=np.float64)
    score_sqsum = np.zeros((L, H), dtype=np.float64)
    # top-1 head id per (layer) histogram across scenes -> for entropy
    top1_hist = np.zeros((L, H), dtype=np.int64)

    n_done = 0
    n_err = 0
    err_log = []
    t0 = time.time()

    print(f"[rv] launching pool, est mem per worker ~50 MB", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, str(p)): p for p in pt_files}
        for fut in as_completed(futures):
            try:
                token_id, score, rank = fut.result()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                err_log.append(f"{futures[fut].name}: {e}")
                continue
            # Aggregate
            rank_sum += rank
            rank_sqsum += rank.astype(np.float64) ** 2
            score_sum += score
            score_sqsum += score.astype(np.float64) ** 2
            # top-K: rank >= H-K   (rank H-1 is top-1)
            top_k_count += (rank >= (H - K)).astype(np.int64)
            bot_k_count += (rank < K).astype(np.int64)
            # top-1 head per layer
            top1 = np.argmax(rank, axis=-1)  # (L,)  head_id with rank H-1
            top1_hist[np.arange(L), top1] += 1

            n_done += 1
            if n_done % 1000 == 0:
                dt = time.time() - t0
                rate = n_done / dt if dt else 0
                eta = (len(pt_files) - n_done) / rate if rate else 0
                print(f"[rv] {n_done}/{len(pt_files)}  "
                      f"rate={rate:.0f}/s  eta={eta/60:.1f}min  err={n_err}",
                      flush=True)

    n_total = n_done
    if n_total == 0:
        print("[rv] FATAL: 0 files processed", file=sys.stderr)
        return 3

    rank_mean = rank_sum / n_total
    rank_var = (rank_sqsum / n_total) - rank_mean ** 2
    rank_std = np.sqrt(np.clip(rank_var, 0.0, None))
    top_k_freq = top_k_count / n_total
    bot_k_freq = bot_k_count / n_total
    score_mean = score_sum / n_total
    score_var = (score_sqsum / n_total) - score_mean ** 2
    score_std = np.sqrt(np.clip(score_var, 0.0, None))

    # per-layer top-1 entropy (bits)
    top1_prob = top1_hist / top1_hist.sum(axis=-1, keepdims=True).clip(min=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_layer_entropy = -np.where(
            top1_prob > 0,
            top1_prob * np.log2(top1_prob),
            0.0,
        ).sum(axis=-1)  # (L,)

    # ---- save raw stats ----
    np.savez_compressed(
        out / "rank_stats.npz",
        rank_mean=rank_mean,
        rank_std=rank_std,
        top_k_freq=top_k_freq,
        bot_k_freq=bot_k_freq,
        score_mean=score_mean,
        score_std=score_std,
        top1_hist=top1_hist,
        per_layer_entropy=per_layer_entropy,
        n_total=np.int64(n_total),
        K=np.int64(K),
    )
    print(f"[rv] saved: {out/'rank_stats.npz'}", flush=True)

    # ---- M1.b₁ mask probe ----
    mask_probe = {}
    for vname, mdict in M1B1_MASKS.items():
        per_head = []
        for layer_id, heads in mdict.items():
            for h in heads:
                per_head.append({
                    "layer": layer_id,
                    "head": h,
                    "rank_mean": float(rank_mean[layer_id, h]),
                    "rank_std":  float(rank_std[layer_id, h]),
                    "bot_k_freq": float(bot_k_freq[layer_id, h]),
                    "top_k_freq": float(top_k_freq[layer_id, h]),
                    "score_mean": float(score_mean[layer_id, h]),
                    "score_std":  float(score_std[layer_id, h]),
                })
        mask_probe[vname] = per_head

    # ---- summarize ----
    summary = {
        "n_total": int(n_total),
        "n_err": int(n_err),
        "K_topk": int(K),
        "L": int(L),
        "H": int(H),
        "wall_seconds": int(time.time() - t0),
        "src": str(src),
        "global": {
            "rank_std_mean": float(rank_std.mean()),
            "rank_std_max":  float(rank_std.max()),
            "rank_std_argmax": _arg2d(rank_std),
            "rank_std_min":  float(rank_std.min()),
            "rank_std_argmin": _arg2d(rank_std, mode="min"),
            "per_layer_entropy_bits": [float(x) for x in per_layer_entropy.tolist()],
            "max_entropy_bits": math.log2(H),  # = 4.0 for H=16
        },
        "m1b1_mask_probe": mask_probe,
        "denylist_flagged_kept": int(flagged),
        "err_log_first10": err_log[:10],
    }
    (out / "rank_variance.json").write_text(json.dumps(summary, indent=2))
    print(f"[rv] saved: {out/'rank_variance.json'}", flush=True)

    # ---- markdown summary ----
    _write_md(out / "SUMMARY.md", summary, rank_mean, rank_std,
              top_k_freq, bot_k_freq, per_layer_entropy, H, K)
    print(f"[rv] saved: {out/'SUMMARY.md'}", flush=True)

    # ---- heatmaps (optional, only if matplotlib available) ----
    try:
        _save_heatmaps(out, rank_std, per_layer_entropy, top_k_freq, bot_k_freq)
        print(f"[rv] saved heatmaps to {out}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[rv] heatmap skipped: {e}", flush=True)

    dt = time.time() - t0
    print(f"[rv] DONE  n={n_total}  err={n_err}  wall={dt:.1f}s", flush=True)
    return 0


def _arg2d(arr, mode="max"):
    if mode == "max":
        idx = int(np.argmax(arr))
    else:
        idx = int(np.argmin(arr))
    l, h = divmod(idx, arr.shape[-1])
    return {"layer": int(l), "head": int(h), "value": float(arr[l, h])}


def _write_md(path: Path, summary, rank_mean, rank_std,
              top_k_freq, bot_k_freq, per_layer_entropy, H: int, K: int):
    lines: List[str] = []
    a = lines.append
    a(f"# M1.b₂ Phase 2 prior — Per-scene rank-variance analysis")
    a("")
    a(f"- N = **{summary['n_total']:,}** scenes  (err={summary['n_err']})")
    a(f"- Tensor shape per scene: (L={summary['L']}, H={summary['H']}, V=720)")
    a(f"- K (top/bot threshold) = {K}")
    a(f"- Wall: {summary['wall_seconds']} s")
    a(f"- Source: `{summary['src']}`")
    a("")
    a("## 1. Global rank-std signal")
    a("")
    a("Rank-std measures how variable a head's rank is across scenes.")
    a("rank_std ≈ 0 → head is always at the same rank (static optimum is OK).")
    a("rank_std ≈ H/√12 ≈ 4.62 (uniform max) → head's rank is essentially random.")
    a("")
    g = summary["global"]
    a(f"- rank_std mean over (L×H) = **{g['rank_std_mean']:.3f}**")
    a(f"- rank_std max = **{g['rank_std_max']:.3f}** "
      f"at (L={g['rank_std_argmax']['layer']}, H={g['rank_std_argmax']['head']})")
    a(f"- rank_std min = **{g['rank_std_min']:.3f}** "
      f"at (L={g['rank_std_argmin']['layer']}, H={g['rank_std_argmin']['head']})")
    a(f"- max possible entropy of top-1 distribution = log2({H}) = "
      f"**{g['max_entropy_bits']:.2f} bits**")
    a("")
    a("### Per-layer top-1 entropy (bits)")
    a("")
    a("| Layer | top-1 entropy (bits) | normalized | interpretation |")
    a("|---:|---:|---:|---|")
    for l, h_ent in enumerate(g["per_layer_entropy_bits"]):
        norm = h_ent / g["max_entropy_bits"]
        if norm < 0.10:
            interp = "very static (1 winner)"
        elif norm < 0.30:
            interp = "near-static"
        elif norm < 0.60:
            interp = "moderate variation"
        else:
            interp = "highly scene-dependent"
        a(f"| {l} | {h_ent:.3f} | {norm:.2%} | {interp} |")
    a("")
    a("## 2. M1.b₁ frozen-mask heads — per-scene rank stability check")
    a("")
    a("Heads masked by M1.b₁ V1/V2/V3 should be at the **bottom** of their")
    a("layer's rank distribution on most scenes (else the static mask is risky).")
    a("")
    for vname, items in summary["m1b1_mask_probe"].items():
        a(f"### {vname}")
        a("")
        a("| Layer | Head | rank_mean | rank_std | bot-K freq | top-K freq | "
          "score_mean | score_std |")
        a("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for it in items:
            a(f"| {it['layer']} | {it['head']} | "
              f"{it['rank_mean']:.2f}/{H-1} | {it['rank_std']:.2f} | "
              f"{it['bot_k_freq']:.2%} | {it['top_k_freq']:.2%} | "
              f"{it['score_mean']:.2e} | {it['score_std']:.2e} |")
        a("")
    a("**Reading**: `bot-K freq` = fraction of scenes where this head is in the "
      f"bottom-{K}. Should be near 100% for the V1/V2/V3 picks to validate the "
      "static-mask methodology of M1.b₁ on the navtrain distribution.")
    a("")
    a("## 3. Implications for Phase 2")
    a("")
    a("1. Layers with **low top-1 entropy** (< 1 bit normalized) → static mask "
      "is near-optimal; learned policy has little headroom.")
    a("2. Layers with **high entropy** (> 2.5 bits) → biggest opportunity for "
      "per-scene head-gating policy.")
    a("3. M1.b₁ V1 mask (L12:h13) bot-K freq quantifies *how* free-lunch the "
      "static mask actually is on navtrain (not just navtest).")
    a("")
    a("> Auto-generated by `scripts/m1b2_rank_variance.py`.")
    path.write_text("\n".join(lines))


def _save_heatmaps(out: Path, rank_std, per_layer_entropy,
                   top_k_freq, bot_k_freq):
    import matplotlib  # noqa
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # rank_std heatmap (L, H)
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(rank_std, aspect="auto", cmap="viridis")
    ax.set_xlabel("Head id")
    ax.set_ylabel("Layer id")
    ax.set_title("Per-(layer, head) rank std across 19k scenes\n"
                 "(low=stable rank, high=scene-dependent rank)")
    fig.colorbar(im, ax=ax, label="rank_std")
    fig.tight_layout()
    fig.savefig(out / "heatmap_rank_std.png", dpi=120)
    plt.close(fig)

    # per-layer entropy bar
    fig, ax = plt.subplots(figsize=(10, 4))
    L = len(per_layer_entropy)
    ax.bar(range(L), per_layer_entropy)
    ax.axhline(math.log2(16), color="red", ls="--",
               label=f"max = log2(16) = 4.0")
    ax.set_xlabel("Layer id")
    ax.set_ylabel("Top-1 head entropy (bits)")
    ax.set_title("Per-layer top-1 head entropy across 19k scenes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "heatmap_top1_entropy.png", dpi=120)
    plt.close(fig)

    # bot-K heatmap
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(bot_k_freq, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("Head id")
    ax.set_ylabel("Layer id")
    ax.set_title("Frequency of being in bottom-K across scenes\n"
                 "(=1 means always bottom → safe to mask)")
    fig.colorbar(im, ax=ax, label="bot-K freq")
    fig.tight_layout()
    fig.savefig(out / "heatmap_bot_k_freq.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

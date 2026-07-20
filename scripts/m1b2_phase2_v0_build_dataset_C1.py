"""
M1.b₂ Phase 2 v0 — Step 1 (C1 variant): Build supervised dataset cache.

# CREATED 2026-06-29 by autonomous agent (β path, after R1'' (mean-only) failed
# with EM=0.30 ≈ const baseline). NOT in design doc — see docs/journal/2026-06-29.md.

Same source dumps and label as R1'' (`per_layer_vision_attn` shape (28, 16, 720),
target = L12 bot-K=4 head idxs), but **richer feature**:

R1'' (original)  : x = concat(attn[L].mean(dim=-1)) over L∈{0,4,8,16,20,24}, dim=96
C1   (this file) : x = concat(attn[L].std(dim=-1), attn[L].max(dim=-1)[0])
                                over the same L, dim = 6 layers × 2 stats × 16 heads = 192

Scientific question
-------------------
Does mean over 720 vision tokens (R1'') discard the signal that distinguishes
the ~47% of scenes where L12 bot-4 contains h2 or h4 instead of the default h0
(the 4th most-frequent bot head, freq=0.525)? If std/max recovers EM > const 0.30
by ≥0.05, the v0 hypothesis "L12 bot-K predictable from non-neighbour layers"
is true but only via tail-distribution features, not first-moment summaries.

Outputs
-------
  exp/m1b2_phase2_v0/dataset_C1_target12_botK4.pt          (~ same n + 192/96 size)
  exp/m1b2_phase2_v0/dataset_C1_target12_botK4.summary.json

Split: identical hash rule as R1'' build (md5(token_id)[:8] % 100 → 80/10/10).
       Guarantees same train/holdout/shifted partition as R1'' for fair compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DUMP_DIR = REPO / "exp" / "m1b2_navtrain_full_alllayers"
DEFAULT_OUT_DIR = REPO / "exp" / "m1b2_phase2_v0"

TARGET_LAYER = 12
FEATURE_LAYERS = (0, 4, 8, 16, 20, 24)
EXCLUDED_LAYERS = (11, 12, 13)
K = 4
N_HEADS = 16
N_STATS = 2  # std + max


def hash_bucket(token_id: str, mod: int = 100) -> int:
    h = hashlib.md5(token_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def split_label(bucket: int) -> int:
    if bucket < 80:
        return 0
    elif bucket < 90:
        return 1
    else:
        return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=Path, default=DEFAULT_DUMP_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=0,
                    help="if > 0, only process this many files (smoke test)")
    args = ap.parse_args()

    for L in FEATURE_LAYERS:
        assert L not in EXCLUDED_LAYERS, f"feature layer {L} is excluded"
    assert TARGET_LAYER in EXCLUDED_LAYERS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "build_dataset_C1.log"
    log = open(log_path, "w")

    def _say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        log.write(msg + "\n")
        log.flush()

    files = sorted(args.dump_dir.glob("*.pt"))
    if args.limit > 0:
        files = files[: args.limit]
    n = len(files)

    feat_dim = N_HEADS * N_STATS * len(FEATURE_LAYERS)
    _say(f"[build_C1] dump_dir       = {args.dump_dir}")
    _say(f"[build_C1] n_files        = {n}")
    _say(f"[build_C1] TARGET_LAYER   = L{TARGET_LAYER}")
    _say(f"[build_C1] FEATURE_LAYERS = {FEATURE_LAYERS}")
    _say(f"[build_C1] feature        = concat(std, max) over 720 vision tokens, R^{feat_dim}")
    _say(f"[build_C1] EXCLUDED       = {EXCLUDED_LAYERS}")
    _say(f"[build_C1] K (bot)        = {K}")
    if n == 0:
        _say("[build_C1] ERROR: no files found")
        return 1

    token_ids: list[str] = []
    X = torch.zeros((n, feat_dim), dtype=torch.float32)
    y_botK = torch.zeros((n, N_HEADS), dtype=torch.bool)
    split = torch.zeros((n,), dtype=torch.int8)

    t0 = time.time()
    skipped = 0
    bad_shape: list[str] = []
    for i, p in enumerate(files):
        tok = p.stem
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            _say(f"[build_C1] WARN load failed {tok}: {e}")
            skipped += 1
            continue
        attn = d.get("per_layer_vision_attn", None)
        max_layer_needed = max(TARGET_LAYER, max(FEATURE_LAYERS))
        if (attn is None or attn.dim() != 3 or attn.shape[0] <= max_layer_needed
                or attn.shape[1] != N_HEADS):
            bad_shape.append(tok)
            skipped += 1
            continue

        # Feature: per-layer (std, max) over 720 vision tokens → 16 each → concat
        feats = []
        for L in FEATURE_LAYERS:
            s = attn[L].std(dim=-1)              # (16,)
            m = attn[L].max(dim=-1).values       # (16,)
            feats.append(s)
            feats.append(m)
        x = torch.cat(feats, dim=0)              # (16 * 2 * |FEATURE_LAYERS|,) = 192

        # Label: identical to R1'' build → bot-K at TARGET_LAYER via mean
        target_mean = attn[TARGET_LAYER].mean(dim=-1)
        bot_idx = torch.topk(target_mean, k=K, largest=False).indices
        y = torch.zeros(N_HEADS, dtype=torch.bool)
        y[bot_idx] = True

        idx = len(token_ids)
        token_ids.append(tok)
        X[idx] = x
        y_botK[idx] = y
        split[idx] = split_label(hash_bucket(tok))

        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            _say(f"[build_C1] {i+1}/{n}  ({dt:.1f}s, "
                 f"{(i+1)/max(dt,1e-6):.1f} files/s)")

    n_ok = len(token_ids)
    X = X[:n_ok].contiguous()
    y_botK = y_botK[:n_ok].contiguous()
    split = split[:n_ok].contiguous()

    wall = time.time() - t0
    _say(f"[build_C1] DONE  n_ok={n_ok}  skipped={skipped}  wall={wall:.1f}s")
    if bad_shape:
        _say(f"[build_C1] WARN bad_shape ({len(bad_shape)}): {bad_shape[:5]} ...")

    n_train = int((split == 0).sum())
    n_hold = int((split == 1).sum())
    n_shift = int((split == 2).sum())
    _say(f"[build_C1] split  train={n_train}  holdout={n_hold}  shifted={n_shift}")

    head_freq = y_botK.float().mean(dim=0)
    _say(f"[build_C1] L{TARGET_LAYER} bot-{K} per-head freq:")
    for h in range(N_HEADS):
        _say(f"    h{h:02d}: {head_freq[h].item():.4f}")

    # B0 const baseline (identical formula → should equal R1'' baseline if same split)
    const_topK = torch.topk(head_freq, k=K).indices
    pred_const = torch.zeros_like(y_botK)
    pred_const[:, const_topK] = True
    tp = (pred_const & y_botK).sum(dim=0).float()
    fp = (pred_const & ~y_botK).sum(dim=0).float()
    fn = (~pred_const & y_botK).sum(dim=0).float()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    macro_f1 = f1.mean().item()
    em = (pred_const == y_botK).all(dim=1).float().mean().item()
    _say(f"[build_C1] const-baseline (top-{K}) = {const_topK.tolist()}")
    _say(f"[build_C1]   per-head F1 macro = {macro_f1:.4f}")
    _say(f"[build_C1]   exact-match acc   = {em:.4f}")

    # Feature stats sanity (so I can compare against R1'' meta)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    _say(f"[build_C1] X stats: mean(min/max)=({X_mean.min().item():.2e},"
         f" {X_mean.max().item():.2e})  std(min/max)=({X_std.min().item():.2e},"
         f" {X_std.max().item():.2e})")

    out_path = args.out_dir / "dataset_C1_target12_botK4.pt"
    payload = {
        "token_ids": token_ids,
        "X": X,
        "y_botK": y_botK,
        "split": split,
        "meta": {
            "variant": "C1",
            "target_layer": TARGET_LAYER,
            "feature_layers": list(FEATURE_LAYERS),
            "excluded_layers": list(EXCLUDED_LAYERS),
            "K": K,
            "n_heads": N_HEADS,
            "n_stats": N_STATS,
            "n_total": n_ok,
            "n_train": n_train,
            "n_holdout": n_hold,
            "n_shifted": n_shift,
            "feat_dim": feat_dim,
            "feature": f"concat(std, max) over 720 vision tokens for L in {list(FEATURE_LAYERS)}  (R^{feat_dim})",
            "label": f"multi-hot bot-{K} head idxs at L{TARGET_LAYER}",
            "split_rule": "md5(token_id)[:8] % 100  →  <80 train / 80..89 holdout / 90..99 shifted",
            "skipped": skipped,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baselines": {
                "const_topK_head_idxs": const_topK.tolist(),
                "const_macro_f1": macro_f1,
                "const_exact_match": em,
            },
        },
    }
    torch.save(payload, out_path)
    _say(f"[build_C1] wrote {out_path}  "
         f"size={out_path.stat().st_size / 1024 / 1024:.2f} MB")

    summary_path = args.out_dir / "dataset_C1_target12_botK4.summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "meta": payload["meta"],
            "L_target_botK_head_freq": head_freq.tolist(),
        }, f, indent=2)
    _say(f"[build_C1] wrote {summary_path}")

    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

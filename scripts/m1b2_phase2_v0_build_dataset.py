"""
M1.b₂ Phase 2 v0 — Step 1: Build supervised dataset cache (R1'' cross-layer transfer).

Reads the 19,225 per-token attention dumps under
  exp/m1b2_navtrain_full_alllayers/<token_id>.pt
each containing `per_layer_vision_attn` of shape (28, 16, 720) fp32.

R1'' design (post §10.z pivot):
  feature x : concat of attn[L].mean(dim=-1) for L in FEATURE_LAYERS ∈ R^(16 * |L|)
  label   y : multi-hot bot-K head idxs at TARGET_LAYER  ∈ {0,1}^16
TARGET_LAYER (L12) is *excluded* from FEATURE_LAYERS — and so are its immediate
neighbours L11/L13 — to remove residual-stream leakage. The point of v0 is to test
whether other layers' attention summaries can predict L12's bot-K.

Produces a single cache:
  exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt
A dict with:
  token_ids : list[str]                     length N=19,225
  X         : tensor (N, 16 * |FEATURE_LAYERS|) fp32
  y_botK    : tensor (N, 16) bool            — bot-K at TARGET_LAYER
  split     : tensor (N,) int8               — 0=train (80%) / 1=holdout (10%) / 2=shifted (10%)
  meta      : dict (target_layer, feature_layers, K, ts, n_files, ...)

Split is hash-stable on token_id (md5 -> int -> mod 100):
  bucket 0..79  → train
  bucket 80..89 → holdout (G_v0_1, in-distribution)
  bucket 90..99 → shifted holdout (G_v0_3')

Wall: ~70-80 s on warm cephfs.
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
# 6 representative non-L12 layers, with L11/L13 explicitly excluded to remove
# residual-stream neighbour leakage. Covers shallow / mid-shallow / mid-deep / deep.
FEATURE_LAYERS = (0, 4, 8, 16, 20, 24)
EXCLUDED_LAYERS = (11, 12, 13)  # asserted at build time
K = 4
N_HEADS = 16


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

    # sanity: target/excluded must not appear in feature
    for L in FEATURE_LAYERS:
        assert L not in EXCLUDED_LAYERS, f"feature layer {L} is in EXCLUDED_LAYERS"
    assert TARGET_LAYER in EXCLUDED_LAYERS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "build_dataset.log"
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
    _say(f"[build_dataset] dump_dir       = {args.dump_dir}")
    _say(f"[build_dataset] n_files        = {n}")
    _say(f"[build_dataset] TARGET_LAYER   = L{TARGET_LAYER}")
    _say(f"[build_dataset] FEATURE_LAYERS = {FEATURE_LAYERS}  (R^{16*len(FEATURE_LAYERS)})")
    _say(f"[build_dataset] EXCLUDED       = {EXCLUDED_LAYERS}")
    _say(f"[build_dataset] K (bot)        = {K}")
    if n == 0:
        _say("[build_dataset] ERROR: no files found")
        return 1

    feat_dim = N_HEADS * len(FEATURE_LAYERS)

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
            _say(f"[build_dataset] WARN load failed {tok}: {e}")
            skipped += 1
            continue
        attn = d.get("per_layer_vision_attn", None)
        max_layer_needed = max(TARGET_LAYER, max(FEATURE_LAYERS))
        if (attn is None or attn.dim() != 3 or attn.shape[0] <= max_layer_needed
                or attn.shape[1] != N_HEADS):
            bad_shape.append(tok)
            skipped += 1
            continue

        # Feature: concat of per-head mean attn at FEATURE_LAYERS
        feats = [attn[L].mean(dim=-1) for L in FEATURE_LAYERS]  # each (16,)
        x = torch.cat(feats, dim=0)  # (16 * |FEATURE_LAYERS|,)

        # Label: bot-K at TARGET_LAYER
        target_mean = attn[TARGET_LAYER].mean(dim=-1)  # (16,)
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
            _say(f"[build_dataset] {i+1}/{n}  ({dt:.1f}s, "
                 f"{(i+1)/max(dt,1e-6):.1f} files/s)")

    n_ok = len(token_ids)
    X = X[:n_ok].contiguous()
    y_botK = y_botK[:n_ok].contiguous()
    split = split[:n_ok].contiguous()

    wall = time.time() - t0
    _say(f"[build_dataset] DONE  n_ok={n_ok}  skipped={skipped}  wall={wall:.1f}s")
    if bad_shape:
        _say(f"[build_dataset] WARN bad_shape ({len(bad_shape)}): {bad_shape[:5]} ...")

    n_train = int((split == 0).sum())
    n_hold = int((split == 1).sum())
    n_shift = int((split == 2).sum())
    _say(f"[build_dataset] split  train={n_train}  holdout={n_hold}  shifted={n_shift}")

    head_freq = y_botK.float().mean(dim=0)  # (16,)
    _say(f"[build_dataset] L{TARGET_LAYER} bot-{K} per-head freq:")
    for h in range(N_HEADS):
        _say(f"    h{h:02d}: {head_freq[h].item():.4f}")

    # Const baseline (B0): predict the K most-frequent bot heads for every scene.
    const_topK = torch.topk(head_freq, k=K).indices  # (K,)
    pred_const = torch.zeros_like(y_botK)
    pred_const[:, const_topK] = True
    # Per-head F1
    tp = (pred_const & y_botK).sum(dim=0).float()
    fp = (pred_const & ~y_botK).sum(dim=0).float()
    fn = (~pred_const & y_botK).sum(dim=0).float()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    macro_f1 = f1.mean().item()
    # Exact-match (all 4 heads correct)
    em = (pred_const == y_botK).all(dim=1).float().mean().item()
    _say(f"[build_dataset] const-baseline (top-{K} most frequent = {const_topK.tolist()})")
    _say(f"[build_dataset]   per-head F1 macro = {macro_f1:.4f}")
    _say(f"[build_dataset]   exact-match acc   = {em:.4f}")

    # Closed-form on R1'' (rank of x_concat — different action space, mostly diagnostic)
    # We pick the K dims of x with smallest values and remap to head ids by % 16.
    bot_x_idx = torch.topk(X, k=K, dim=1, largest=False).indices  # (N, K), values in [0, feat_dim)
    pred_closed = torch.zeros_like(y_botK)
    for i in range(n_ok):
        for j in range(K):
            pred_closed[i, bot_x_idx[i, j].item() % N_HEADS] = True
    tp = (pred_closed & y_botK).sum(dim=0).float()
    fp = (pred_closed & ~y_botK).sum(dim=0).float()
    fn = (~pred_closed & y_botK).sum(dim=0).float()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1c = 2 * prec * rec / (prec + rec + 1e-9)
    macro_f1_closed = f1c.mean().item()
    em_closed = (pred_closed == y_botK).all(dim=1).float().mean().item()
    _say(f"[build_dataset] closed-form (rank of x_concat, % 16):")
    _say(f"[build_dataset]   per-head F1 macro = {macro_f1_closed:.4f}")
    _say(f"[build_dataset]   exact-match acc   = {em_closed:.4f}")

    out_path = args.out_dir / "dataset_R1pp_target12_botK4.pt"
    payload = {
        "token_ids": token_ids,
        "X": X,
        "y_botK": y_botK,
        "split": split,
        "meta": {
            "target_layer": TARGET_LAYER,
            "feature_layers": list(FEATURE_LAYERS),
            "excluded_layers": list(EXCLUDED_LAYERS),
            "K": K,
            "n_heads": N_HEADS,
            "n_total": n_ok,
            "n_train": n_train,
            "n_holdout": n_hold,
            "n_shifted": n_shift,
            "feat_dim": feat_dim,
            "feature": f"concat attn[L].mean(-1) for L in {list(FEATURE_LAYERS)}  (R1'' / R^{feat_dim})",
            "label": f"multi-hot bot-{K} head idxs at L{TARGET_LAYER}",
            "split_rule": "md5(token_id)[:8] % 100  →  <80 train / 80..89 holdout / 90..99 shifted",
            "skipped": skipped,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baselines": {
                "const_topK_head_idxs": const_topK.tolist(),
                "const_macro_f1": macro_f1,
                "const_exact_match": em,
                "closed_form_macro_f1": macro_f1_closed,
                "closed_form_exact_match": em_closed,
            },
        },
    }
    torch.save(payload, out_path)
    _say(f"[build_dataset] wrote {out_path}  "
         f"size={out_path.stat().st_size / 1024 / 1024:.2f} MB")

    summary_path = args.out_dir / "dataset_R1pp_target12_botK4.summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "meta": payload["meta"],
            "L_target_botK_head_freq": head_freq.tolist(),
        }, f, indent=2)
    _say(f"[build_dataset] wrote {summary_path}")

    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

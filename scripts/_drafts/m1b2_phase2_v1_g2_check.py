#!/usr/bin/env python
"""m1b2 phase 2 v1 — Step 2: G_v1_2 secondary gate.

Evaluates top-K=3 macro-F1 with h13 masked out (set logit = -inf and label set = original
top-3 excluding h13). Computes both:
  - model top-3 prediction macro-F1 (over 15 heads, not 16)
  - const-baseline top-3 (= train head_freq's top-3 excluding h13)

G_v1_2 PASS: model_top3_F1 ≥ const_top3_baseline_F1 + 0.01 on holdout.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn as nn

EXP = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_phase2_v0")
DROP_HEAD = 13          # h13 = always-positive head, removed from G_v1_2
KPRIME = 3              # top-K' after removal


class P2MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def load_model(ckpt_path, feat_dim, n_heads):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if "fc.weight" in sd:
        m = nn.Linear(feat_dim, n_heads)
        m.load_state_dict({"weight": sd["fc.weight"], "bias": sd["fc.bias"]})
        kind = "linear"
        hidden = None
    elif "net.0.weight" in sd:
        hidden = int(ckpt.get("hidden", sd["net.0.weight"].shape[0]))
        m = P2MLP(feat_dim, hidden, n_heads)
        m.load_state_dict(sd)
        kind = "mlp"
    else:
        raise RuntimeError(f"unknown sd keys: {list(sd.keys())[:5]}")
    m.eval()
    return m, kind, hidden


def topk_f1_excl(logits, labels, k, exclude_head):
    """Top-k macro-F1 with `exclude_head` masked out.

    Returns: per_head_f1 (list of n_heads, value=0 for excluded), macro_f1 (over n_heads-1 heads).
    Also returns predicted top-k indices [N, k].
    """
    n_heads = logits.shape[1]
    masked = logits.clone()
    masked[:, exclude_head] = float("-inf")
    pred_topk = torch.topk(masked, k=k, dim=-1).indices                  # [N, k]
    pred_bin = torch.zeros_like(labels, dtype=torch.bool)
    pred_bin.scatter_(1, pred_topk, True)
    tp = (pred_bin & labels).float().sum(dim=0)
    fp = (pred_bin & ~labels).float().sum(dim=0)
    fn = (~pred_bin & labels).float().sum(dim=0)
    f1_per = (2 * tp / (2 * tp + fp + fn + 1e-12))
    f1_per[exclude_head] = 0.0
    # macro over the 15 retained heads
    keep = [h for h in range(n_heads) if h != exclude_head]
    macro = float(f1_per[keep].mean())
    return f1_per.tolist(), macro, pred_topk


def const_topk_excl_baseline(head_freq, labels, k, exclude_head):
    """Compute F1 for the const baseline that always predicts the top-k of head_freq excluding h13."""
    n_heads = head_freq.shape[0]
    freq = head_freq.clone()
    freq[exclude_head] = -1.0
    const_topk = torch.topk(freq, k=k).indices.tolist()
    # constant prediction set
    pred_bin = torch.zeros_like(labels, dtype=torch.bool)
    pred_bin[:, const_topk] = True
    tp = (pred_bin & labels).float().sum(dim=0)
    fp = (pred_bin & ~labels).float().sum(dim=0)
    fn = (~pred_bin & labels).float().sum(dim=0)
    f1_per = (2 * tp / (2 * tp + fp + fn + 1e-12))
    f1_per[exclude_head] = 0.0
    keep = [h for h in range(n_heads) if h != exclude_head]
    macro = float(f1_per[keep].mean())
    return const_topk, f1_per.tolist(), macro


CELLS = [
    ("R1pp-P1 (96)",  "dataset_R1pp_target12_botK4.pt", "p1_full_20260626_154930", 96),
    ("R1pp-P2 (96)",  "dataset_R1pp_target12_botK4.pt", "p2_full_20260626_154951", 96),
    ("C1-P1 (192)",   "dataset_C1_target12_botK4.pt",   "p1_C1_20260629_142400",   192),
    ("C1-P2 (192)",   "dataset_C1_target12_botK4.pt",   "p2_C1_20260629_142500",   192),
    ("C2-P1 (768)",   "dataset_C2_target12_botK4.pt",   "p1_C2_20260629_144700",   768),
    ("C2-P2 (768)",   "dataset_C2_target12_botK4.pt",   "p2_C2_20260629_144800",   768),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.01,
                    help="G_v1_2 PASS threshold: model_f1 ≥ const_f1 + margin")
    ap.add_argument("--out-json", type=str, default=str(EXP / "m1b2_phase2_v1_g2_check.json"))
    args = ap.parse_args()

    results = []
    for label, ds_name, sub, dim in CELLS:
        ds_path = EXP / ds_name
        ck_path = EXP / sub / "model.pt"
        if not ck_path.exists() or not ds_path.exists():
            print(f"[skip] {label}: missing files")
            continue

        payload = torch.load(ds_path, map_location="cpu", weights_only=False)
        X = payload["X"].float()
        y = payload["y_botK"]
        split = payload["split"]
        n_heads = y.shape[1]

        model, kind, hidden = load_model(ck_path, dim, n_heads)

        # Const baseline (use train freq)
        y_train = y[split == 0].float()
        head_freq = y_train.mean(dim=0)

        cell_out = {"cell": label, "dir": sub, "probe": kind, "splits": {}}

        for sid, sname in [(1, "holdout"), (2, "shifted")]:
            mask = split == sid
            Xs = X[mask]; ys = y[mask]
            with torch.no_grad():
                logits = model(Xs)

            f1_model_per, f1_model_macro, pred_topk = topk_f1_excl(
                logits, ys, KPRIME, DROP_HEAD)
            const_set, f1_const_per, f1_const_macro = const_topk_excl_baseline(
                head_freq, ys, KPRIME, DROP_HEAD)

            # Diversity check: does model predict a different top-3 than const?
            const_t = torch.tensor(const_set)
            pred_eq = (pred_topk.sort(dim=-1).values == const_t.sort().values).all(dim=-1)
            frac_pred_eq_const = float(pred_eq.float().mean())

            cell_out["splits"][sname] = {
                "n": int(Xs.shape[0]),
                "const_top3_excl_h13": const_set,
                "f1_model_macro": f1_model_macro,
                "f1_const_macro": f1_const_macro,
                "delta_model_minus_const": f1_model_macro - f1_const_macro,
                "frac_pred_eq_const": frac_pred_eq_const,
                "f1_model_per_head": [float(x) for x in f1_model_per],
                "f1_const_per_head": [float(x) for x in f1_const_per],
            }
        # G_v1_2 PASS: model holdout f1 ≥ const + margin
        ho = cell_out["splits"]["holdout"]
        cell_out["G_v1_2_pass"] = bool(ho["delta_model_minus_const"] >= args.margin)
        cell_out["G_v1_2_margin"] = args.margin
        results.append(cell_out)

    out = {"margin": args.margin, "drop_head": DROP_HEAD, "k_prime": KPRIME, "cells": results}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out_json}\n")

    # Console summary
    print(f"=== G_v1_2 (top-3 macro-F1, h13 excluded) ===")
    print(f"{'cell':22s} {'const_top3':16s} {'f1_const':>9s} {'f1_model':>9s} {'Δ':>7s} {'frac_eq':>8s} {'verdict':>8s}")
    for r in results:
        ho = r["splits"]["holdout"]
        print(f"{r['cell']:22s} {str(ho['const_top3_excl_h13']):16s} "
              f"{ho['f1_const_macro']:>9.4f} {ho['f1_model_macro']:>9.4f} "
              f"{ho['delta_model_minus_const']:>+7.4f} "
              f"{ho['frac_pred_eq_const']:>8.3f} "
              f"{'PASS' if r['G_v1_2_pass'] else 'FAIL':>8s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

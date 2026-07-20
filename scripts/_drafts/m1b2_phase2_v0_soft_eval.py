"""
M1.b2 Phase 2 v0 — soft-eval diagnostic.

# CREATED 2026-06-29 by autonomous agent.

Question
--------
After path (C) {R1, C1, C2} × {P1, P2} all show holdout F1 = const-baseline 0.2203
exactly (4-digit equality), is the signal truly absent, or is the hard top-K=4 EM
metric merely insensitive (i.e. logits are reordering correctly but never enough
to displace the const top-4 head set [13,14,6,0])?

We probe the best-trained model (C2-P2, train_loss ↓ to 0.18) and emit:
  1. per-head AUROC on holdout + shifted  (16 numbers per split)
  2. per-head positive rate (label distribution)
  3. per-sample margin gap:
        logit_sum(predicted top-4) − logit_sum(const top-4)
     distribution stats (mean / median / >0 frac)
  4. Brier score per head + overall
  5. const baseline rank: how often is each const-top-4 head still in predicted top-4

If any head has holdout AUROC > 0.6 (random=0.5) → signal exists, metric too hard.
If all heads ≈ 0.5 → no per-sample signal in attn stats.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


# Same arch as train_p2.py
class P2MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary AUROC. scores: float [N], labels: bool/int [N]."""
    labels = labels.bool()
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-based: AUROC = (sum_rank_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    sum_rank_pos = float(ranks[labels].sum().item())
    return (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[soft] dataset = {args.dataset}")
    print(f"[soft] model   = {args.model}")
    print(f"[soft] K       = {args.k}")

    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    X = payload["X"].float()                      # [N, D]
    y = payload["y_botK"]                         # [N, 16]  bool
    split = payload["split"]                      # [N] int  0=train 1=holdout 2=shifted
    meta = payload.get("meta", {})

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    feat_dim = int(ckpt["feat_dim"])
    n_heads = int(ckpt["n_heads"])
    K = int(ckpt["K"])
    sd = ckpt["state_dict"]
    # Auto-detect arch by state_dict keys.
    if "fc.weight" in sd:
        probe_kind = "linear"
        hidden = None
        model = nn.Linear(feat_dim, n_heads)
        # P1 train script saved under 'fc.*' so unwrap.
        model.load_state_dict({"weight": sd["fc.weight"], "bias": sd["fc.bias"]})
    elif "net.0.weight" in sd:
        probe_kind = "mlp"
        hidden = int(ckpt.get("hidden", sd["net.0.weight"].shape[0]))
        model = P2MLP(feat_dim, hidden, n_heads, dropout=0.0)
        model.load_state_dict(sd)
    else:
        raise RuntimeError(f"unrecognized state_dict keys: {list(sd.keys())[:5]}")
    print(f"[soft] feat_dim={feat_dim} hidden={hidden} probe={probe_kind} n_heads={n_heads} K(train)={K}")
    assert feat_dim == X.shape[1], f"dim mismatch {feat_dim} vs {X.shape[1]}"
    assert K == args.k, f"K mismatch {K} vs {args.k}"
    model.eval()

    out = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "feat_dim": feat_dim,
        "K": K,
        "n_heads": n_heads,
        "splits": {},
    }

    # const top-K from training labels
    y_train = y[split == 0].float()
    head_freq_train = y_train.mean(dim=0)
    const_topk = torch.topk(head_freq_train, k=args.k, largest=True).indices.tolist()
    out["const_topk"] = const_topk
    out["head_freq_train"] = head_freq_train.tolist()
    print(f"[soft] const top-{args.k} = {const_topk}")

    for split_id, name in [(1, "holdout"), (2, "shifted")]:
        mask = split == split_id
        Xs = X[mask]
        ys = y[mask]
        if Xs.shape[0] == 0:
            continue
        with torch.no_grad():
            logits = model(Xs)                 # [N, 16]
            probs = torch.sigmoid(logits)

        # per-head AUROC
        per_head_auroc = [auroc(logits[:, h], ys[:, h]) for h in range(n_heads)]
        # per-head pos rate
        per_head_pos = ys.float().mean(dim=0).tolist()
        # per-head Brier
        per_head_brier = ((probs - ys.float()) ** 2).mean(dim=0).tolist()
        # predicted top-K vs const top-K
        pred_topk = torch.topk(logits, k=args.k, dim=-1).indices  # [N, k]
        # margin gap
        const_idx = torch.tensor(const_topk)
        # per-sample logit sum of const top-K
        const_sum = logits[:, const_idx].sum(dim=-1)              # [N]
        pred_sum = logits.gather(1, pred_topk).sum(dim=-1)        # [N]
        margin = (pred_sum - const_sum)                            # >=0 always
        # fraction with strict pred != const
        const_set = set(const_topk)
        pred_sets = [set(row.tolist()) for row in pred_topk]
        frac_eq_const = sum(1 for s in pred_sets if s == const_set) / len(pred_sets)
        # how often each const head is still in predicted top-K
        const_in_pred = {h: 0 for h in const_topk}
        for s in pred_sets:
            for h in const_topk:
                if h in s:
                    const_in_pred[h] += 1
        const_in_pred = {h: const_in_pred[h] / len(pred_sets) for h in const_topk}

        # exact-match against y top-K (label is bool [16], the "true top-K" set is {h: y[h]==1})
        # since |true|=K, EM = pred_set == true_set
        em = 0
        for i, s in enumerate(pred_sets):
            true_set = set(int(h) for h in torch.where(ys[i])[0].tolist())
            if s == true_set:
                em += 1
        em /= len(pred_sets)

        # macro F1 on top-K predictions
        pred_bin = torch.zeros_like(ys, dtype=torch.bool)
        pred_bin.scatter_(1, pred_topk, True)
        # per-head F1
        tp = (pred_bin & ys).float().sum(dim=0)
        fp = (pred_bin & ~ys).float().sum(dim=0)
        fn = (~pred_bin & ys).float().sum(dim=0)
        f1_per = (2 * tp / (2 * tp + fp + fn + 1e-12)).tolist()
        macro_f1 = sum(f1_per) / len(f1_per)

        split_out = {
            "n": int(Xs.shape[0]),
            "per_head_auroc": per_head_auroc,
            "per_head_pos_rate": per_head_pos,
            "per_head_brier": per_head_brier,
            "per_head_f1_topk": f1_per,
            "macro_f1_topk": macro_f1,
            "em_topk": em,
            "frac_pred_eq_const_topk": frac_eq_const,
            "const_head_retention_in_pred": const_in_pred,
            "margin_gap_stats": {
                "mean": float(margin.mean()),
                "median": float(margin.median()),
                "min": float(margin.min()),
                "max": float(margin.max()),
                "std": float(margin.std()),
            },
            "auroc_max": max(a for a in per_head_auroc if not (a != a)),
            "auroc_mean": sum(a for a in per_head_auroc if not (a != a)) /
                          max(1, sum(1 for a in per_head_auroc if not (a != a))),
        }
        out["splits"][name] = split_out

        print(f"[soft] --- split={name} n={split_out['n']} ---")
        print(f"[soft]   per-head AUROC: {[f'{a:.3f}' for a in per_head_auroc]}")
        print(f"[soft]   AUROC max={split_out['auroc_max']:.4f} mean={split_out['auroc_mean']:.4f}")
        print(f"[soft]   per-head pos_rate: {[f'{p:.3f}' for p in per_head_pos]}")
        print(f"[soft]   macro_f1_topk={macro_f1:.4f} em_topk={em:.4f}")
        print(f"[soft]   frac(pred==const_topk)={frac_eq_const:.4f}")
        print(f"[soft]   const_head_retention={ {h: f'{v:.3f}' for h,v in const_in_pred.items()} }")
        print(f"[soft]   margin gap (pred−const logit sum): "
              f"mean={split_out['margin_gap_stats']['mean']:.4f} "
              f"median={split_out['margin_gap_stats']['median']:.4f} "
              f"std={split_out['margin_gap_stats']['std']:.4f}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"[soft] wrote {args.out}")

    # final verdict
    ho = out["splits"]["holdout"]
    auroc_max = ho["auroc_max"]
    auroc_mean = ho["auroc_mean"]
    print()
    print(f"[soft] VERDICT:")
    print(f"  holdout AUROC max={auroc_max:.4f} mean={auroc_mean:.4f}")
    if auroc_max > 0.6:
        print(f"  → SIGNAL EXISTS (some head has discriminative logits). Hard top-K EM is the bottleneck.")
    elif auroc_max > 0.55:
        print(f"  → WEAK signal. Marginal; would need richer feature or different label.")
    else:
        print(f"  → NO per-sample signal. attn-stats from feature layers do NOT predict L12 bot-K.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

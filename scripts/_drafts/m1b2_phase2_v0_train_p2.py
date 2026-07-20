"""
M1.b₂ Phase 2 v0 — Step 2b: Train P2 (MLP probe) on R1'' dataset.

# DRAFT — 等用户审完 critic / explorer 报告后再批准执行
# Created 2026-06-26 by builder agent (main AI), per design §10 / §10.z.

Source spec : docs/_internal/m1b2_phase2_design_2026-06-25.md §10 (P2 = MLP 96→64→16),
              §10.z (R1'' cross-layer transfer).
Architecture: Linear(96,64) → GELU → Dropout(0.1) → Linear(64,16).
              Otherwise identical to P1 (BCEWithLogitsLoss multi-label, top-K eval).

Outputs     :
  exp/m1b2_phase2_v0/p2_<ts>/
    model.pt       — final state_dict + meta
    metrics.json   — per-epoch + final summary
    train.log      — text log

Acceptance gates (per §10.z; G_v0_2 specifically requires P1 result for delta):
  G_v0_0  : reported B0 const baseline (floor) — printed at startup.
  G_v0_1  : holdout per-head macro-F1 > B0 + 0.05 .
  G_v0_2  : capacity gap P2 − P1 < 0.05 → P1 sufficient ; if ≥ 0.05 → MLP useful.
  G_v0_3' : |F1(shifted) − F1(holdout)| < 0.05 → no spurious in-dist overfit.

Example
-------
    /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \\
        scripts/_drafts/m1b2_phase2_v0_train_p2.py \\
        --dataset exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt \\
        --out_dir exp/m1b2_phase2_v0/p2_$(date +%Y%m%d_%H%M%S) \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers (kept inline + identical to P1 for one-file drop-in)
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("token_ids", "X", "y_botK", "split", "meta")
    for k in required:
        assert k in payload, f"dataset missing field {k}"
    return payload


def split_tensors(payload: dict, split_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    sel = payload["split"] == split_id
    return payload["X"][sel].float(), payload["y_botK"][sel].bool()


def topk_predict(logits: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.topk(logits, k=k, dim=1, largest=True).indices
    pred = torch.zeros_like(logits, dtype=torch.bool)
    pred.scatter_(1, idx, True)
    return pred


def macro_f1_and_em(pred: torch.Tensor, y: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    tp = (pred & y).sum(dim=0).float()
    fp = (pred & ~y).sum(dim=0).float()
    fn = (~pred & y).sum(dim=0).float()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return f1.mean().item(), (pred == y).all(dim=1).float().mean().item(), f1


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, k: int,
             device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device)).cpu()
    pred = topk_predict(logits, k=k)
    macro, em, f1_per_head = macro_f1_and_em(pred, y)
    return {
        "macro_f1": macro,
        "exact_match": em,
        "per_head_f1": [round(v, 4) for v in f1_per_head.tolist()],
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class P2MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train.log"
    log = open(log_path, "w")

    def _say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        log.write(msg + "\n")
        log.flush()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = load_dataset(args.dataset)
    meta = payload["meta"]
    K = int(meta["K"])
    feat_dim = int(meta["feat_dim"])
    n_heads = int(meta["n_heads"])
    baselines = meta.get("baselines", {})
    const_top = baselines.get("const_topK_head_idxs", None)

    Xtr, ytr = split_tensors(payload, 0)
    Xho, yho = split_tensors(payload, 1)
    Xsh, ysh = split_tensors(payload, 2)

    if args.smoke:
        Xtr, ytr = Xtr[:200], ytr[:200]
        args.epochs = 3

    _say(f"[p2] device          = {device}")
    _say(f"[p2] dataset         = {args.dataset}")
    _say(f"[p2] arch            = Linear({feat_dim},{args.hidden})→GELU→"
         f"Dropout({args.dropout})→Linear({args.hidden},{n_heads})")
    _say(f"[p2] feat_dim        = {feat_dim}    n_heads={n_heads}    K={K}")
    _say(f"[p2] split sizes     train={len(Xtr)}  holdout={len(Xho)}  shifted={len(Xsh)}")
    _say(f"[p2] B0 const top-K  = {const_top}")
    _say(f"[p2] B0 macro-F1     = {baselines.get('const_macro_f1')}")
    _say(f"[p2] B0 exact-match  = {baselines.get('const_exact_match')}")
    _say(f"[p2] closed-form F1  = {baselines.get('closed_form_macro_f1')}")
    _say(f"[p2] config          epochs={args.epochs} lr={args.lr} batch={args.batch}"
         f" seed={args.seed} smoke={args.smoke}")

    model = P2MLP(feat_dim, args.hidden, n_heads, dropout=args.dropout).to(device)
    pre_ho = evaluate(model, Xho, yho, K, device)
    pre_sh = evaluate(model, Xsh, ysh, K, device)
    _say(f"[p2] epoch=0 (random init) holdout  macro_f1={pre_ho['macro_f1']:.4f}"
         f"  EM={pre_ho['exact_match']:.4f}")
    _say(f"[p2] epoch=0 (random init) shifted  macro_f1={pre_sh['macro_f1']:.4f}"
         f"  EM={pre_sh['exact_match']:.4f}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    Xtr_d = Xtr.to(device)
    ytr_d = ytr.float().to(device)

    n_train = len(Xtr_d)
    metrics_log: list[dict] = []
    best_ho_f1 = -1.0
    best_state = None
    best_epoch = -1

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch):
            idx = perm[i: i + args.batch]
            xb = Xtr_d[idx]
            yb = ytr_d[idx]
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)
        ho = evaluate(model, Xho, yho, K, device)
        sh = evaluate(model, Xsh, ysh, K, device)

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "holdout_macro_f1": round(ho["macro_f1"], 6),
            "holdout_em": round(ho["exact_match"], 6),
            "shifted_macro_f1": round(sh["macro_f1"], 6),
            "shifted_em": round(sh["exact_match"], 6),
        }
        metrics_log.append(rec)
        _say(f"[p2] epoch={epoch:3d}  loss={train_loss:.4f}"
             f"  holdout F1={ho['macro_f1']:.4f} EM={ho['exact_match']:.4f}"
             f"  shifted F1={sh['macro_f1']:.4f} EM={sh['exact_match']:.4f}")

        if ho["macro_f1"] > best_ho_f1:
            best_ho_f1 = ho["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    wall = time.time() - t0
    _say(f"[p2] DONE wall={wall:.1f}s  best_holdout_f1={best_ho_f1:.4f}"
         f"  at epoch={best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_ho = evaluate(model, Xho, yho, K, device)
    final_sh = evaluate(model, Xsh, ysh, K, device)

    b0_f1 = float(baselines.get("const_macro_f1", float("nan")))
    b0_em = float(baselines.get("const_exact_match", float("nan")))
    delta_f1_ho = final_ho["macro_f1"] - b0_f1
    delta_f1_sh = final_sh["macro_f1"] - b0_f1

    _say(f"[p2] FINAL holdout  macro_f1={final_ho['macro_f1']:.4f}"
         f" EM={final_ho['exact_match']:.4f}"
         f"  Δvs_B0 F1={delta_f1_ho:+.4f}")
    _say(f"[p2] FINAL shifted  macro_f1={final_sh['macro_f1']:.4f}"
         f" EM={final_sh['exact_match']:.4f}"
         f"  Δvs_B0 F1={delta_f1_sh:+.4f}")

    gate_g_v0_1 = bool(final_ho["macro_f1"] > b0_f1 + 0.05)
    gate_g_v0_3p = bool(abs(final_ho["macro_f1"] - final_sh["macro_f1"]) < 0.05)
    _say(f"[p2] gate G_v0_1 (holdout F1 > B0+0.05): "
         f"{'PASS' if gate_g_v0_1 else 'FAIL'}")
    _say(f"[p2] gate G_v0_3' (|holdout-shifted| < 0.05): "
         f"{'PASS' if gate_g_v0_3p else 'FAIL'}")
    _say(f"[p2] G_v0_2 capacity gap = (P2 macro_f1) − (P1 macro_f1) ; compute "
         f"externally from P1 metrics.json")

    out_metrics = args.out_dir / "metrics.json"
    with open(out_metrics, "w") as f:
        json.dump({
            "probe": "P2_mlp_96_64_16",
            "args": vars(args) | {"dataset": str(args.dataset),
                                  "out_dir": str(args.out_dir)},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": round(wall, 1),
            "best_epoch": best_epoch,
            "epoch_metrics": metrics_log,
            "final": {
                "holdout": final_ho,
                "shifted": final_sh,
                "delta_macro_f1_holdout_vs_B0": round(delta_f1_ho, 6),
                "delta_macro_f1_shifted_vs_B0": round(delta_f1_sh, 6),
                "B0_const_macro_f1": b0_f1,
                "B0_const_exact_match": b0_em,
                "closed_form_macro_f1": baselines.get("closed_form_macro_f1"),
                "gate_G_v0_1": gate_g_v0_1,
                "gate_G_v0_3p": gate_g_v0_3p,
            },
            "dataset_meta": meta,
        }, f, indent=2)
    _say(f"[p2] wrote {out_metrics}")

    if not args.smoke and best_state is not None:
        out_model = args.out_dir / "model.pt"
        torch.save({"state_dict": best_state,
                    "feat_dim": feat_dim,
                    "hidden": args.hidden,
                    "n_heads": n_heads,
                    "K": K,
                    "best_epoch": best_epoch,
                    "best_holdout_f1": best_ho_f1}, out_model)
        _say(f"[p2] wrote {out_model}")

    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M1.b₂ Phase 3 Step 2 — Train per-head binary probe (走法 1).

DRAFT (2026-06-29 20:30) — not yet runnable end-to-end. Awaiting:
  (1) Pivot 1 PDMS to confirm L12 (K, PDMS) static curve shape
      → decides λ-sweep range and whether to abort and pivot to L24/L27.
  (2) `code/rldrive/agents/autovla_with_dynamic_mask.py` to be written
      (provides the task-loss surrogate via 2-pass forward).

Pieces present here (already runnable in isolation):
  * Argparse + config plumbing
  * R1pp-P1 dataset loader (reuse Phase 2 features; label column unused)
  * PerHeadBinaryProbe instantiation + Gumbel τ annealing
  * Sparsity / smoothness loss wiring
  * Per-step + per-epoch logging

Pieces TODO (placeholder raises NotImplementedError):
  * `task_loss_fn(mask, feature, ...)`  — needs the dynamic-mask forward path.
    For now we ship a *signal-substitute* mode (`--task_loss surrogate_kl`)
    that uses the R1pp dataset's `label_botK4` as a target to verify the
    sparsity / training loop works before the real backbone integration.

Spec : exp/m1b2_phase2_v0/m1b2_phase3_step2_spec.md
Probe: code/rldrive/probes/per_head_binary_probe.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Imports of project modules.
# Note: `code/` is NOT a proper python package (no __init__.py at code/),
# and the name "code" collides with stdlib `code`. We add the probes
# directory directly to sys.path.
# ---------------------------------------------------------------------------
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBES_DIR = REPO_ROOT / "code" / "rldrive" / "probes"
if str(PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(PROBES_DIR))

from per_head_binary_probe import (  # noqa: E402
    PerHeadBinaryProbe,
    mask_stats,
    smoothness_loss,
    sparsity_loss,
)


# ---------------------------------------------------------------------------
# Dataset loader (reuse Phase 2 R1pp build)
# ---------------------------------------------------------------------------


def load_r1pp_dataset(path: Path) -> Dict[str, Any]:
    """Load Phase 2 R1pp dataset.

    Expected payload keys (from m1b2_phase2_v0_build_dataset.py):
        X       : (N, 96) float32  — 96-d features
        y_botK  : (N, 16) int      — multi-hot bot-K=4 label
        split   : (N,) int         — 0=train, 1=holdout, 2=shifted
        meta    : dict
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Normalize key names: tolerate both "feature_96d" and "X"
    if "X" in payload and "feature_96d" not in payload:
        payload["feature_96d"] = payload["X"]
    if "y_botK" in payload and "label_botK4" not in payload:
        payload["label_botK4"] = payload["y_botK"]
    assert "feature_96d" in payload, f"dataset missing X/feature_96d: keys={list(payload)}"
    return payload


# ---------------------------------------------------------------------------
# Task loss
# ---------------------------------------------------------------------------


def surrogate_kl_loss(
    mask: torch.Tensor,            # (B, 16)   in [0, 1], 1=keep
    label_botK4: torch.Tensor,     # (B, 16)   multi-hot, 1=is-bot-K=4 (i.e. safe-to-drop)
) -> torch.Tensor:
    """Surrogate task loss for training-loop validation.

    Idea: a head that is consistently in the bottom-K (label=1) is "safe to
    drop"; one that is NOT bot-K (label=0) is "should keep". So target_keep
    = 1 - label. We use BCE-with-logits-style stable formulation (clamp mask
    away from 0/1 hard endpoints to keep numerics sane under STE).

    This is NOT the real navtest task loss — it just gives the probe *some*
    gradient so we can iterate on training infra before the dynamic-mask
    forward path lands.
    """
    target_keep = (1.0 - label_botK4.float())
    # Clamp to avoid log(0) when mask hits exact 0/1 via STE.
    m = mask.clamp(1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(m, target_keep)


def real_task_loss(mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Real downstream task loss via dynamic-mask forward pass.

    Will be implemented in `code/rldrive/agents/autovla_with_dynamic_mask.py`.
    Sketch:
        1. Pass 1: backbone forward, capture L0/4/8/16/20/24 mean attn → 96-d
        2. Probe → mask (per token, per head)
        3. Pass 2: backbone forward with patch_head_mask(per_token=True, mask=...)
        4. KL between Pass-1 and Pass-2 action logits (or L1 on trajectory)
    """
    raise NotImplementedError(
        "real_task_loss requires autovla_with_dynamic_mask.py — TODO Step 2."
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def anneal_tau(epoch: int, total_epochs: int, tau_start: float = 1.0, tau_end: float = 0.5) -> float:
    """Linear anneal of Gumbel-sigmoid τ over the first 50% of training."""
    half = max(1, total_epochs // 2)
    if epoch >= half:
        return tau_end
    return tau_start + (tau_end - tau_start) * (epoch / half)


def train_one_run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    payload = load_r1pp_dataset(Path(args.dataset))
    feat = payload["feature_96d"]
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    feat = feat.float()
    label = payload.get("label_botK4")
    if label is None or args.task_loss == "none":
        label = torch.zeros(feat.shape[0], 16)
    elif not torch.is_tensor(label):
        label = torch.tensor(label)
    label = label.float()
    split = payload.get("split", torch.zeros(feat.shape[0], dtype=torch.long))
    if not torch.is_tensor(split):
        split = torch.tensor(split, dtype=torch.long)

    train_mask = split == 0
    val_mask = split == 1
    train_ds = TensorDataset(feat[train_mask], label[train_mask])
    val_ds = TensorDataset(feat[val_mask], label[val_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- probe ---
    probe = PerHeadBinaryProbe(
        d_in=96, n_heads=16, hidden=args.hidden, init_logit_bias=args.init_logit_bias
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.task_loss == "surrogate_kl":
        task_loss_fn = surrogate_kl_loss
    elif args.task_loss == "real":
        task_loss_fn = real_task_loss  # will raise NotImplementedError
    elif args.task_loss == "none":
        task_loss_fn = lambda m, y: torch.tensor(0.0, device=m.device)  # noqa: E731
    else:
        raise ValueError(f"unknown task_loss={args.task_loss}")

    history = []

    for epoch in range(args.epochs):
        probe.tau = anneal_tau(epoch, args.epochs, tau_start=args.tau_start, tau_end=args.tau_end)
        probe.train()
        epoch_l_task, epoch_l_sp, epoch_l_sm, n_seen = 0.0, 0.0, 0.0, 0
        epoch_K_eff = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = probe(x, hard=True)   # (B, 16) hard STE — for sparsity/smoothness/(real fwd)
            if args.task_loss == "surrogate_kl":
                # BUGFIX 2026-06-30: BCE on STE-hard {0,1} mask kills the gradient
                # (every value sits exactly on the clamp(1e-6,1-1e-6) boundary, so
                # clamp's local grad = 0 for all heads → probe never updates,
                # val_task frozen at ln(2), eval K_eff=0). Feed the *soft* keep prob
                # (deterministic sigmoid) so the surrogate actually trains.
                keep_p = torch.sigmoid(probe.net(x))
                l_task = surrogate_kl_loss(keep_p, y)
            else:
                l_task = task_loss_fn(mask, y)
            l_sp = sparsity_loss(mask)
            l_sm = smoothness_loss(mask) if args.gamma > 0 else torch.tensor(0.0, device=device)
            loss = l_task + args.lambda_ * l_sp + args.gamma * l_sm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = x.shape[0]
            epoch_l_task += l_task.item() * bsz
            epoch_l_sp += l_sp.item() * bsz
            epoch_l_sm += l_sm.item() * bsz
            n_seen += bsz
            # K_eff stats on the hard mask
            with torch.no_grad():
                drop_cnt = (mask < 0.5).float().sum(dim=-1).cpu()
                epoch_K_eff.append(drop_cnt)

        # --- val ---
        probe.eval()
        with torch.no_grad():
            val_K_eff, val_l_task = [], 0.0
            v_seen = 0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = probe(x, hard=False)  # soft for stable val
                val_l_task += task_loss_fn(mask, y).item() * x.shape[0]
                v_seen += x.shape[0]
                mask_hard = probe.predict_hard(x)
                val_K_eff.append((mask_hard == 0).float().sum(dim=-1).cpu())

        epoch_K_eff_t = torch.cat(epoch_K_eff) if epoch_K_eff else torch.zeros(0)
        val_K_eff_t = torch.cat(val_K_eff) if val_K_eff else torch.zeros(0)
        rec = {
            "epoch": epoch,
            "tau": probe.tau,
            "train_l_task": epoch_l_task / max(n_seen, 1),
            "train_l_sp": epoch_l_sp / max(n_seen, 1),
            "train_l_sm": epoch_l_sm / max(n_seen, 1),
            "train_avg_K_eff": float(epoch_K_eff_t.mean().item()) if len(epoch_K_eff_t) else 0.0,
            "train_K_eff_std": float(epoch_K_eff_t.std(unbiased=False).item()) if len(epoch_K_eff_t) else 0.0,
            "val_l_task": val_l_task / max(v_seen, 1),
            "val_avg_K_eff": float(val_K_eff_t.mean().item()) if len(val_K_eff_t) else 0.0,
            "val_K_eff_std": float(val_K_eff_t.std(unbiased=False).item()) if len(val_K_eff_t) else 0.0,
        }
        history.append(rec)
        print(
            f"[ep {epoch:02d}] tau={probe.tau:.2f}  "
            f"train  task={rec['train_l_task']:.4f}  sp={rec['train_l_sp']:.4f}  K={rec['train_avg_K_eff']:.2f}  "
            f"val    task={rec['val_l_task']:.4f}  K={rec['val_avg_K_eff']:.2f}±{rec['val_K_eff_std']:.2f}"
        )

    # --- save ---
    torch.save({"state_dict": probe.state_dict(), "args": vars(args)}, out_dir / "model.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "args": vars(args), "wall_sec": time.time() - t0}, f, indent=2)

    return {"out_dir": str(out_dir), "wall_sec": time.time() - t0, "final": history[-1] if history else None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--lambda_", "--lambda", dest="lambda_", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=None, help="MLP hidden dim; None=linear")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init_logit_bias",
        type=float,
        default=2.0,
        help="initial per-head bias; sigmoid(b)=initial keep prob. "
        "2.0 (default)=keep-all start (degenerate eval K_eff=0 under spec lr/epochs); "
        "0.0=symmetric start (lets surrogate/sparsity move heads across 0.5 quickly).",
    )
    p.add_argument(
        "--task_loss",
        choices=["surrogate_kl", "real", "none"],
        default="surrogate_kl",
        help="surrogate_kl: BCE vs (1-bot_K=4) for infra test; real: dynamic-mask forward (TODO); none: pure sparsity.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(vars(args), indent=2))
    result = train_one_run(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

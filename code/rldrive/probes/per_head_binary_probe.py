"""M1.b₂ Phase 3 Step 2 — Per-head Binary Probe (走法 1).

Single-stage probe that, per token, predicts an independent 0/1 mask over the
16 attention heads of a target layer (default L12). Replaces the original
two-stage R1 (decide K) + R1pp (pick K heads) design.

Probe outputs `keep` mask (1 = keep, 0 = drop). The drop set is then translated
into `head_mask_layers={layer: [head_ids]}` and passed to
`code.rldrive.agents.head_mask_patch.patch_head_mask`.

Input feature   : R1pp-P1 compatible 96-d (6 layers × 16 heads, mean over kv).
                  Reuse `exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt`
                  (feature column unchanged, label column unused here).
Loss            : task_loss + λ * sparsity (training script supplies task_loss).
Training script : scripts/_drafts/m1b2_phase3_step2_train_probe.py
Spec            : exp/m1b2_phase2_v0/m1b2_phase3_step2_spec.md
Status          : DRAFT scaffolding (2026-06-29 20:25) — not yet trained.

Status note
-----------
This file is *scaffolding*. The training loop (and the dynamic-mask eval
forward path in autovla_with_dynamic_mask.py) are still TODO. The
PerHeadBinaryProbe module here is complete and self-contained, suitable for
unit-testing on synthetic features before the Pivot-1 PDMS results return.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Probe module
# ---------------------------------------------------------------------------


class PerHeadBinaryProbe(nn.Module):
    """16 independent sigmoids over a 96-d token feature.

    Convention: output `mask[t, h] = 1` means KEEP head h at token t.
    The training loop converts this to `head_mask_layers={L12: [h where
    mask==0]}` for the inference hook.

    Architecture:
        - Linear baseline (matches R1pp-P1):  nn.Linear(96, 16)
        - Optional MLP:                       Linear-GELU-Linear with `hidden`

    Forward path:
        - Training: Gumbel-sigmoid with straight-through estimator (STE) when
          `hard=True`. Without `hard`, returns the soft (continuous) sigmoid.
        - Eval: deterministic sigmoid; threshold at 0.5 when `hard=True`.

    Temperature `tau` controls Gumbel-sigmoid sharpness; anneal externally
    (e.g. 1.0 -> 0.5 over the first 50% of epochs).
    """

    def __init__(
        self,
        d_in: int = 96,
        n_heads: int = 16,
        hidden: Optional[int] = None,
        init_logit_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_heads = n_heads
        self.hidden = hidden
        if hidden is None:
            self.net: nn.Module = nn.Linear(d_in, n_heads)
            # Bias init so initial keep prob ~ sigmoid(init_logit_bias).
            # init_logit_bias=2.0 => initial keep prob ~ 0.88 (start near "keep all").
            nn.init.zeros_(self.net.weight)
            nn.init.constant_(self.net.bias, init_logit_bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_heads),
            )
            # bias on last layer
            last_linear = self.net[-1]
            assert isinstance(last_linear, nn.Linear)
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, init_logit_bias)
        self.tau = 1.0  # caller anneals via `probe.tau = ...`

    # -- inference helpers ---------------------------------------------------

    @torch.no_grad()
    def predict_hard(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Deterministic hard mask for eval. Returns 0/1 LongTensor, shape (..., n_heads)."""
        self.eval()
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).long()

    @torch.no_grad()
    def keep_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head keep probability (sigmoid of logits), shape (..., n_heads)."""
        self.eval()
        return torch.sigmoid(self.net(x))

    # -- training forward ----------------------------------------------------

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Return mask in [0, 1] (soft) or {0, 1} (hard with STE).

        Args:
            x: (..., d_in) input features.
            hard: if True, use Gumbel-sigmoid + straight-through. Required for
                  end-to-end training where downstream consumer wants binary.

        Returns:
            mask: (..., n_heads). 1 = keep head, 0 = drop head.
        """
        logits = self.net(x)
        if self.training:
            # Gumbel-sigmoid: g = log(u) - log(1-u), u ~ U(0,1)
            u = torch.rand_like(logits).clamp_(1e-8, 1.0 - 1e-8)
            g = torch.log(u) - torch.log1p(-u)
            y_soft = torch.sigmoid((logits + g) / max(self.tau, 1e-3))
        else:
            y_soft = torch.sigmoid(logits)
        if hard:
            y_hard = (y_soft > 0.5).float()
            # straight-through: forward = hard, backward = soft
            return y_hard + (y_soft - y_soft.detach())
        return y_soft


# ---------------------------------------------------------------------------
# Loss helpers (kept here so the training script stays thin)
# ---------------------------------------------------------------------------


def sparsity_loss(mask: torch.Tensor) -> torch.Tensor:
    """Sparsity loss = -avg_K_dropped / n_heads.

    Larger lambda * sparsity_loss => more heads dropped on average.
    mask: (..., n_heads) in [0, 1]. 1 = keep, 0 = drop.
    """
    n_heads = mask.shape[-1]
    drop_frac = (1.0 - mask).mean()  # mean over (..., heads) -> scalar
    return -drop_frac  # negate so that LOWER loss = MORE drop


def smoothness_loss(mask: torch.Tensor) -> torch.Tensor:
    """Variance of per-head mask across the token dim.

    Encourages mask to be relatively stable across tokens (avoid flicker).
    Expects mask shape (T, n_heads) or (B, T, n_heads).
    """
    if mask.dim() == 2:
        # (T, H) -> variance across T per head, then mean over H
        return mask.var(dim=0, unbiased=False).mean()
    if mask.dim() == 3:
        # (B, T, H) -> variance across T per (B,H), then mean
        return mask.var(dim=1, unbiased=False).mean()
    raise ValueError(f"smoothness_loss expects 2D or 3D mask, got shape {tuple(mask.shape)}")


def mask_stats(mask: torch.Tensor) -> dict:
    """Return summary stats on a (..., n_heads) mask for logging."""
    n_heads = mask.shape[-1]
    # K_eff (per row) = number of dropped heads
    drop_count = (mask < 0.5).float().sum(dim=-1)  # (...,)
    return {
        "avg_K_eff": float(drop_count.mean().item()),
        "std_K_eff": float(drop_count.std(unbiased=False).item()),
        "min_K_eff": float(drop_count.min().item()),
        "max_K_eff": float(drop_count.max().item()),
        "frac_keep_per_head": (mask >= 0.5).float().mean(dim=tuple(range(mask.dim() - 1))).tolist(),
        "n_heads": n_heads,
    }


# ---------------------------------------------------------------------------
# Smoke check (run as: python -m code.rldrive.probes.per_head_binary_probe)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    torch.manual_seed(0)
    probe = PerHeadBinaryProbe(d_in=96, n_heads=16, hidden=None)
    x = torch.randn(8, 96)
    probe.train()
    m_soft = probe(x, hard=False)
    m_hard = probe(x, hard=True)
    assert m_soft.shape == (8, 16)
    assert m_hard.shape == (8, 16)
    # STE: hard forward values must be in {0, 1}
    assert torch.all((m_hard.detach() == 0) | (m_hard.detach() == 1))
    probe.eval()
    m_pred = probe.predict_hard(x)
    assert m_pred.shape == (8, 16)
    # sparsity_loss differentiable (clear stale grads from previous calls first)
    probe.zero_grad(set_to_none=True)
    m_train = probe(x, hard=True)
    loss = F.mse_loss(m_train.float(), torch.zeros_like(m_train)) + 0.01 * sparsity_loss(m_train)
    loss.backward()
    assert probe.net.weight.grad is not None
    stats = mask_stats(m_pred)
    print("smoke OK", stats)


if __name__ == "__main__":
    _smoke()

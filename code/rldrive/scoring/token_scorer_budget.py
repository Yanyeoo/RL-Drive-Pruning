"""token_scorer_budget.py — Token scorer with learnable budget head.

Extension of TokenImportanceScorer that adds a scene-level budget prediction:
  - Per-token scores: which tokens to keep (same as before)
  - Scene budget: how many tokens to keep (learned from driving reward + efficiency)

The budget head takes the MEAN of all token features as scene-level representation,
and outputs a single scalar (sigmoid → keep_ratio in [min_kr, max_kr]).

RL training:
  reward = α * driving_quality + β * efficiency_bonus
  efficiency_bonus = (1 - keep_ratio) * scale  # reward for pruning more
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from rldrive.scoring.token_scorer import cam_id_from_blocks, cam_onehot


class TokenScorerWithBudget(nn.Module):
    """Scorer that outputs both per-token importance AND scene-level budget."""

    def __init__(self, emb_dim: int = 2048, n_cam: int = 3, hidden: int = 256,
                 min_keep_ratio: float = 0.2, max_keep_ratio: float = 0.9):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cam = n_cam
        self.min_kr = min_keep_ratio
        self.max_kr = max_keep_ratio
        d = emb_dim + n_cam

        # Per-token importance scorer (same architecture as TokenImportanceScorer)
        self.token_net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Scene-level budget head (takes mean-pooled features → keep_ratio)
        self.budget_net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),  # raw logit → sigmoid → [min_kr, max_kr]
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, emb_dim + n_cam) per-token features

        Returns:
            token_scores: (N,) per-token importance
            keep_ratio: scalar in [min_kr, max_kr], scene-level budget
            budget_logit: raw logit (for log_prob computation)
        """
        # Per-token scores
        token_scores = self.token_net(x).squeeze(-1)  # (N,)

        # Scene-level budget: mean-pool all tokens → single budget decision
        scene_feat = x.mean(dim=0, keepdim=True)  # (1, d)
        budget_logit = self.budget_net(scene_feat).squeeze()  # scalar

        # Map to [min_kr, max_kr] via sigmoid
        keep_ratio = self.min_kr + (self.max_kr - self.min_kr) * torch.sigmoid(budget_logit)

        return token_scores, keep_ratio, budget_logit

    def forward_token_only(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility: just return per-token scores (for fixed-r eval)."""
        return self.token_net(x).squeeze(-1)

    @classmethod
    def from_pretrained_scorer(cls, base_scorer: nn.Module, **kwargs):
        """Initialize from a pretrained TokenImportanceScorer (copy token_net weights)."""
        model = cls(
            emb_dim=base_scorer.emb_dim,
            n_cam=base_scorer.n_cam,
            **kwargs,
        )
        # Copy token_net weights from base scorer
        model.token_net.load_state_dict(base_scorer.net.state_dict())
        return model


class BudgetScorerRunner:
    """Loads a trained TokenScorerWithBudget; returns per-token scores AND the
    scene-level learned keep_ratio (deterministic, policy mean — for EVAL).

    This is the missing eval path for Budget RL: instead of pruning at a fixed
    global ratio, each scene gets its own keep_ratio from the budget head, then
    the top-B tokens (by token_scores) are pruned at that per-scene ratio.
    """

    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        self.n_cam = int(cfg["n_cam"])
        self.emb_dim = int(cfg["emb_dim"])
        self.min_kr = float(cfg.get("min_keep_ratio", 0.2))
        self.max_kr = float(cfg.get("max_keep_ratio", 0.9))
        self.model = TokenScorerWithBudget(
            emb_dim=self.emb_dim, n_cam=self.n_cam,
            hidden=int(cfg["hidden"]),
            min_keep_ratio=self.min_kr, max_keep_ratio=self.max_kr,
        )
        sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
        self.model.load_state_dict(sd)
        self.model.eval().to(device)
        norm = torch.load(ckpt_dir / "feature_norm.pt", map_location=device, weights_only=False)
        self.mean = norm["mean"].to(device)
        self.std = norm["std"].to(device)
        self.device = device

    def build_input(self, vision_feat, vision_token_positions, vision_blocks):
        emb = (vision_feat.to(self.device) - self.mean) / self.std
        cam = cam_id_from_blocks(vision_token_positions, vision_blocks)
        coh = cam_onehot(cam, self.n_cam).to(self.device)
        return torch.cat([emb, coh], dim=-1)

    @torch.no_grad()
    def score_budget(self, vision_feat, vision_token_positions, vision_blocks):
        """Returns (token_scores_cpu, keep_ratio_float)."""
        x = self.build_input(vision_feat, vision_token_positions, vision_blocks)
        token_scores, keep_ratio, _ = self.model(x)
        return token_scores.detach().to("cpu", torch.float32), float(keep_ratio.item())

"""token_scorer.py — S3 per-token Importance Scorer (shared by train + inference).

Spec: docs/specs/s3_token_scorer_spec.md.
Input per vision token: [standardized ViT->LLM emb (H) ; cam_id one-hot (n_cam)].
Output: scalar importance (higher = keep). Ranking-only (LambdaRank trained).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn


def cam_id_from_blocks(vision_token_positions: torch.Tensor,
                       vision_blocks: List[Tuple[int, int]]) -> torch.Tensor:
    """Map each vision token position to its camera block index (0..n_cam-1)."""
    vp = vision_token_positions.flatten()
    cam = torch.zeros(vp.numel(), dtype=torch.long)
    for ci, (s, e) in enumerate(vision_blocks):
        cam[(vp > s) & (vp < e)] = ci
    return cam


def cam_onehot(cam_id: torch.Tensor, n_cam: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(cam_id.clamp(0, n_cam - 1), n_cam).float()


class TokenImportanceScorer(nn.Module):
    def __init__(self, emb_dim: int = 2048, n_cam: int = 3, hidden: int = 256):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cam = n_cam
        d = emb_dim + n_cam
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ScorerRunner:
    """Loads a trained scorer + feature-norm; scores live-captured features.

    Used by AutoVLAWithTokenPruneAgent (selector='scorer')."""

    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        self.n_cam = int(cfg["n_cam"])
        self.emb_dim = int(cfg["emb_dim"])
        self.model = TokenImportanceScorer(self.emb_dim, self.n_cam, int(cfg["hidden"]))
        sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
        self.model.load_state_dict(sd)
        self.model.eval().to(device)
        norm = torch.load(ckpt_dir / "feature_norm.pt", map_location=device, weights_only=False)
        self.mean = norm["mean"].to(device)   # (emb_dim,)
        self.std = norm["std"].to(device)      # (emb_dim,)
        self.device = device

    def build_input(self, vision_feat: torch.Tensor, vision_token_positions: torch.Tensor,
                    vision_blocks) -> torch.Tensor:
        emb = (vision_feat.to(self.device) - self.mean) / self.std
        cam = cam_id_from_blocks(vision_token_positions, vision_blocks)
        coh = cam_onehot(cam, self.n_cam).to(self.device)
        return torch.cat([emb, coh], dim=-1)

    @torch.no_grad()
    def score(self, vision_feat, vision_token_positions, vision_blocks) -> torch.Tensor:
        x = self.build_input(vision_feat, vision_token_positions, vision_blocks)
        return self.model(x).detach().to("cpu", torch.float32)


__all__ = ["TokenImportanceScorer", "ScorerRunner", "cam_id_from_blocks", "cam_onehot"]

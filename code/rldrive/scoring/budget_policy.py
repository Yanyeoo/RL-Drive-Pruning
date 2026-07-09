"""budget_policy.py — S3 Stage B Budget Policy (design Q3.b/Q4.2).

Maps per-scene driving context -> a keep-ratio class in {0.25,0.5,0.75,1.0}.
scene_ctx is CPU-extractable from the navsim nocot json (velocity/acceleration/
instruction/his_trajectory) — no GPU needed to featurize.

Shared by training (scripts/s3_budget_policy_phaseA.py) and live inference
(AutoVLAWithTokenPruneAgent budget_policy_ckpt path).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

RATIOS = [0.25, 0.5, 0.75, 1.0]

# canonical instruction vocabulary (extended lazily at featurize time via hashing
# fallback); keep a small fixed set for the common navsim commands.
_INSTR_VOCAB = [
    "keep forward", "turn left", "turn right", "stop",
    "change lane to the left", "change lane to the right", "reverse",
]


def _instr_onehot(instr: str) -> List[float]:
    instr = (instr or "").strip().lower()
    v = [0.0] * (len(_INSTR_VOCAB) + 1)
    if instr in _INSTR_VOCAB:
        v[_INSTR_VOCAB.index(instr)] = 1.0
    else:
        v[-1] = 1.0  # OOV bucket
    return v


def build_scene_ctx(d: Dict) -> List[float]:
    """Deterministic scene-context feature vector from a nocot json dict."""
    vel = d.get("velocity", [0.0, 0.0]) or [0.0, 0.0]
    acc = d.get("acceleration", [0.0, 0.0]) or [0.0, 0.0]
    vx, vy = float(vel[0]), float(vel[1] if len(vel) > 1 else 0.0)
    ax, ay = float(acc[0]), float(acc[1] if len(acc) > 1 else 0.0)
    speed = math.hypot(vx, vy)
    # history curvature proxy: net lateral drift over history
    his = d.get("his_trajectory", []) or []
    lat = float(his[-1][1]) if his and len(his[-1]) > 1 else 0.0
    feats = [vx, vy, speed, ax, ay, lat] + _instr_onehot(d.get("instruction", ""))
    return feats


CTX_DIM = 6 + len(_INSTR_VOCAB) + 1  # 6 kinematic + instr one-hot


class BudgetPolicy(nn.Module):
    def __init__(self, ctx_dim: int = CTX_DIM, n_class: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x):
        return self.net(x)


class BudgetPolicyRunner:
    """Live inference: nocot json dict -> keep_ratio in {0.25,0.5,0.75,1.0}."""

    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        self.ratios = cfg.get("ratios", RATIOS)
        self.model = BudgetPolicy(cfg["ctx_dim"], len(self.ratios), cfg.get("hidden", 64))
        self.model.load_state_dict(
            torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False))
        self.model.eval().to(device)
        norm = torch.load(ckpt_dir / "ctx_norm.pt", map_location=device, weights_only=False)
        self.mean, self.std = norm["mean"].to(device), norm["std"].to(device)
        self.device = device

    @torch.no_grad()
    def keep_ratio(self, scene_json: Dict) -> float:
        x = torch.tensor(build_scene_ctx(scene_json), dtype=torch.float32, device=self.device)
        x = (x - self.mean) / self.std
        cls = int(self.model(x.unsqueeze(0)).argmax(-1).item())
        return float(self.ratios[cls])


__all__ = ["BudgetPolicy", "BudgetPolicyRunner", "build_scene_ctx", "RATIOS", "CTX_DIM"]

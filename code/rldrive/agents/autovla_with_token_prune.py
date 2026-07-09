"""S1 — AutoVLAWithTokenPruneAgent: per-scene 2-pass vision token pruning (EVAL).

Spec: docs/specs/dynamic_token_pruning_S1_spec.md
Used by the S2 headroom gate (docs/specs/dynamic_headroom_gate_S2_spec.md).

Per scene:
  pass-1  capture L*=12 vision attention (query=last instr token) -> score s in R^N
          (reuses patch_attention_capture; selector='attn_L12').
  select  keep top-B = round(keep_ratio * N) by score; prune the rest.
  pass-2  generate() under patch_vision_token_prune(prune_positions) -> trajectory.

Selectors:
  'attn_L12' (default) : model's own L12 last-instr->vision attention (no training).
  'random'             : per-scene fixed-seed random score (design baseline #2).
  'scorer'             : learned MLP scorer (S3), with optional safety-net fallback.

keep_ratio = 1.0 -> pass-1 skipped, no prune, bit-identical to upstream (lossless).
safety_net = True -> if scorer uncertainty high (entropy/gap), skip pruning for that scene.

Does NOT modify code/third_party/AutoVLA.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

import math

import torch
import torch.nn.functional as F

from navsim.agents.autovla_agent import AutoVLAAgent  # noqa: E402
from navsim.common.dataclasses import Trajectory      # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from rldrive.agents.autovla_with_attention import AutoVLAWithAttentionAgent
from rldrive.agents.token_prune_patch import (
    patch_vision_token_prune,
    select_prune_positions,
)
from rldrive.scoring.attention_capture import patch_attention_capture, patch_vision_feature_capture
from rldrive.scoring.token_scorer import ScorerRunner


class AutoVLAWithTokenPruneAgent(AutoVLAWithAttentionAgent):
    """2-pass vision-token pruning agent for the S2 headroom gate."""

    requires_scene = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        sensor_data_path: Optional[str] = None,
        codebook_cache_path: Optional[str] = None,
        lora_conf: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: str = "cuda",
        skip_model_load: bool = False,
        # ---- token-prune knobs ----
        keep_ratio: float = 1.0,
        selector: str = "attn_L12",
        score_layer: int = 12,
        prune_variant: str = "attn_mask",   # 'attn_mask' (A). 'drop' (B) = S3, not impl.
        prune_verbose: bool = False,
        # ---- S3 learned-scorer selector ----
        scorer_ckpt: Optional[str] = None,   # dir with checkpoint.pt/config.json/feature_norm.pt
        scorer_feat_layer: int = 0,
        # ---- safety-net fallback (scorer only) ----
        safety_net: bool = False,            # enable uncertainty-based fallback to r=1.0
        safety_entropy_thresh: float = 0.92, # normalized entropy threshold (0-1, higher=more flat)
        safety_gap_thresh: float = 0.01,     # boundary gap threshold (lower=less confident)
        safety_temperature: float = 1.0,     # softmax temperature for entropy computation
    ):
        # Force parent's attention capture OFF (we drive capture manually in
        # pass-1 so we control the 2-pass flow); head_mask off.
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=checkpoint_path,
            sensor_data_path=sensor_data_path,
            codebook_cache_path=codebook_cache_path,
            lora_conf=lora_conf,
            config_path=config_path,
            device=device,
            skip_model_load=skip_model_load,
            attention_enabled=False,
            head_mask_layers=None,
        )
        self._keep_ratio = float(keep_ratio)
        self._selector = str(selector)
        self._score_layer = int(score_layer)
        self._prune_variant = str(prune_variant)
        self._prune_verbose = bool(prune_verbose)
        self._scorer_feat_layer = int(scorer_feat_layer)

        # safety-net config
        self._safety_net = bool(safety_net) and (self._selector == "scorer")
        self._safety_entropy_thresh = float(safety_entropy_thresh)
        self._safety_gap_thresh = float(safety_gap_thresh)
        self._safety_temperature = float(safety_temperature)
        self._safety_net_triggers = 0  # counter for monitoring

        if self._prune_variant != "attn_mask":
            raise NotImplementedError(
                f"prune_variant='{self._prune_variant}' not implemented in S1. "
                f"Only 'attn_mask' (Variant A) is available; true-drop (Variant B) "
                f"is S3 and needs M-RoPE position recompute."
            )
        if self._selector not in ("attn_L12", "random", "scorer", "fastv_l2"):
            raise ValueError(f"unknown selector '{self._selector}'")
        # FastV-at-input baseline: uses layer-2 attention as selector at ViT→LLM
        # interface (same position as ours). This isolates "selector quality" gain
        # from "pruning position" gain vs vanilla FastV (which prunes internally).
        if self._selector == "fastv_l2":
            self._score_layer = 2

        self._scorer = None
        if self._selector == "scorer":
            if not scorer_ckpt:
                raise ValueError("selector='scorer' requires scorer_ckpt=<dir>")
            dev = "cuda" if (not skip_model_load) else "cpu"
            self._scorer = ScorerRunner(scorer_ckpt, device=dev)
            print(f"[AutoVLAWithTokenPruneAgent] loaded scorer from {scorer_ckpt} "
                  f"(feat_layer={self._scorer_feat_layer})", flush=True)

        print(
            f"[AutoVLAWithTokenPruneAgent] keep_ratio={self._keep_ratio} "
            f"selector={self._selector} score_layer={self._score_layer} "
            f"variant={self._prune_variant} safety_net={self._safety_net}",
            flush=True,
        )

    # ------------------------------------------------------------------

    def _should_fallback(self, score: torch.Tensor) -> bool:
        """Check if scorer is uncertain and should fallback to no-prune (r=1.0).

        Uses two signals:
        1. Normalized entropy of softmax(score/τ) — high entropy = flat scores = low confidence.
        2. Top-B boundary gap — small gap = unstable boundary = risky pruning.
        """
        if not self._safety_net:
            return False
        n = score.numel()
        b = max(1, int(round(self._keep_ratio * n)))

        # (1) Normalized entropy
        p = F.softmax(score / self._safety_temperature, dim=0)
        entropy = -(p * (p + 1e-10).log()).sum().item()
        max_entropy = math.log(n)
        norm_entropy = entropy / max_entropy  # [0, 1]

        # (2) Boundary gap: difference between the last-kept and first-pruned score
        sorted_scores, _ = score.sort(descending=True)
        if b < n:
            boundary_gap = (sorted_scores[b - 1] - sorted_scores[b]).item()
        else:
            boundary_gap = float("inf")

        return (norm_entropy > self._safety_entropy_thresh) or (boundary_gap < self._safety_gap_thresh)

    def _score_for(self, scene_token: Optional[str], n: int,
                   features: Dict[str, Any], prompt_index) -> torch.Tensor:
        """Return (n,) importance score aligned with vision_token_positions."""
        if self._selector == "random":
            seed = int(hashlib.md5((scene_token or "x").encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            return torch.rand(n, generator=g)
        if self._selector == "scorer":
            # pass-1: capture layer-`scorer_feat_layer` vision features, then MLP
            fbucket: Dict[str, Any] = {}
            with patch_vision_feature_capture(
                vlm=self.autovla.vlm,
                layer_idx=self._scorer_feat_layer,
                prompt_index=prompt_index,
                bucket=fbucket,
            ):
                with torch.no_grad():
                    self.autovla.predict(features)  # pass-1: trajectory discarded
            if "vision_feat" not in fbucket:
                raise RuntimeError(
                    f"[token_prune] scorer pass-1 feature capture did not fire "
                    f"(scene={scene_token}); check feat layer {self._scorer_feat_layer}."
                )
            return self._scorer.score(
                fbucket["vision_feat"],
                prompt_index.vision_token_positions,
                prompt_index.vision_blocks,
            ).flatten()
        # selector == 'attn_L12' or 'fastv_l2': capture attention at self._score_layer
        bucket: Dict[str, Any] = {}
        with patch_attention_capture(
            vlm=self.autovla.vlm,
            layer_idx=self._score_layer,
            prompt_index=prompt_index,
            bucket=bucket,
            average_heads=True,
        ):
            with torch.no_grad():
                self.autovla.predict(features)  # pass-1: trajectory discarded
        if "vision_attn" not in bucket:
            raise RuntimeError(
                f"[token_prune] pass-1 capture did not fire at L{self._score_layer} "
                f"(scene={scene_token}); cannot score. Check pre-fill / eager attn."
            )
        return bucket["vision_attn"].flatten()  # (N_vision,) cpu float32

    def compute_trajectory(self, scene_data):  # type: ignore[override]
        self.autovla.eval()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene_data))
        if self.sensor_data_path:
            features.update({"sensor_data_path": self.sensor_data_path})

        prune_positions: Optional[torch.Tensor] = None

        if self._keep_ratio < 1.0:
            prompt_index, _input_ids = self._build_prompt_index(features)
            n = prompt_index.n_vision
            scene_token = self._extract_scene_token(scene_data)
            score = self._score_for(scene_token, n, features, prompt_index)

            # Safety-net: skip pruning if scorer is uncertain
            if self._should_fallback(score):
                self._safety_net_triggers += 1
                if self._prune_verbose:
                    print(
                        f"[token_prune] SAFETY-NET scene={scene_token} "
                        f"(trigger #{self._safety_net_triggers}) -> skip prune, use r=1.0",
                        flush=True,
                    )
                prune_positions = None  # no pruning for this scene
            else:
                prune_positions = select_prune_positions(
                    vision_token_positions=prompt_index.vision_token_positions,
                    score=score,
                    keep_ratio=self._keep_ratio,
                )
                if self._prune_verbose:
                    print(
                        f"[token_prune] scene={scene_token} N={n} keep_ratio={self._keep_ratio} "
                        f"-> prune {int(prune_positions.numel())} tokens",
                        flush=True,
                    )

        # pass-2: generate under prune mask (no-op if prune_positions empty)
        with ExitStack() as stack:
            stack.enter_context(
                patch_vision_token_prune(
                    vlm=self.autovla.vlm,
                    prune_positions=prune_positions,
                    verbose=self._prune_verbose,
                )
            )
            with torch.no_grad():
                poses, cot_results = self.autovla.predict(features)

        return (
            Trajectory(poses[: self._trajectory_sampling.num_poses, :], self._trajectory_sampling),
            cot_results,
        )

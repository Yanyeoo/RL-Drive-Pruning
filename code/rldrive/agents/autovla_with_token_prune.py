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

keep_ratio = 1.0 -> pass-1 skipped, no prune, bit-identical to upstream (lossless).

Does NOT modify code/third_party/AutoVLA.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

import torch

from navsim.agents.autovla_agent import AutoVLAAgent  # noqa: E402
from navsim.common.dataclasses import Trajectory      # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from rldrive.agents.autovla_with_attention import AutoVLAWithAttentionAgent
from rldrive.agents.token_prune_patch import (
    patch_vision_token_prune,
    select_prune_positions,
)
from rldrive.scoring.attention_capture import patch_attention_capture


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

        if self._prune_variant != "attn_mask":
            raise NotImplementedError(
                f"prune_variant='{self._prune_variant}' not implemented in S1. "
                f"Only 'attn_mask' (Variant A) is available; true-drop (Variant B) "
                f"is S3 and needs M-RoPE position recompute."
            )
        if self._selector not in ("attn_L12", "random"):
            raise ValueError(f"unknown selector '{self._selector}'")

        print(
            f"[AutoVLAWithTokenPruneAgent] keep_ratio={self._keep_ratio} "
            f"selector={self._selector} score_layer={self._score_layer} "
            f"variant={self._prune_variant}",
            flush=True,
        )

    # ------------------------------------------------------------------

    def _score_for(self, scene_token: Optional[str], n: int,
                   features: Dict[str, Any], prompt_index, device) -> torch.Tensor:
        """Return (n,) importance score aligned with vision_token_positions."""
        if self._selector == "random":
            seed = int(hashlib.md5((scene_token or "x").encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            return torch.rand(n, generator=g)
        # selector == 'attn_L12': capture L12 last-instr->vision attention (pass-1)
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
            score = self._score_for(scene_token, n, features, prompt_index, self.device)
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

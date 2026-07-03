"""AutoVLAWithDynamicMaskAgent — per-scene dynamic L12 head mask via trained probe.

M1.b₂ Phase 3 Step 2 (走法 1) — EVAL path.

Per driving scene, runs TWO forward passes:
  pass1 (prefill-only): capture FEATURE_LAYERS per-head vision attention, reduce
        to a 96-d feature   x = concat([attn[L].mean(-1) for L in FEATURE_LAYERS])
        — byte-aligned with scripts/m1b2_phase2_v0_build_dataset.py:135-136.
  probe: PerHeadBinaryProbe(x) -> 16-d keep mask (1=keep, 0=drop) -> drop_heads.
  pass2 (full generate): patch_head_mask({12: drop_heads}) around autovla.predict().

GRANULARITY = PER-SCENE (one 16-d mask per frame), matching the R1pp dataset
(token = scene). This is NOT per-generated-token: the R1pp feature/label are
per-scene, so per-token has no training-data support (see
docs/journal/2026-06-30.md 偏离 #2). Consequently this agent does NOT need a
`per_token=True` mode in head_mask_patch — a per-scene constant mask is exactly
the existing patch_head_mask, just with a scene-specific head set.

Constraints honored:
  * 硬规则 #3 — does NOT modify autovla_with_attention.py / attention_capture.py /
    head_mask_patch.py. It subclasses AutoVLAWithAttentionAgent and only reads
    the (locked) capture + mask context managers.

Hydra usage (eval):
  agent._target_=rldrive.agents.autovla_with_dynamic_mask.AutoVLAWithDynamicMaskAgent
  +agent.probe_ckpt_path=/abs/path/probe_<lambda>_<seed>/model.pt
  +agent.mask_log_dir=/abs/path/maskstats        # optional, for avg_K_eff aggregation
  # backbone / checkpoint / sensor args identical to AutoVLAWithAttentionAgent.

Spec: exp/m1b2_phase2_v0/m1b2_phase3_step2_spec.md
Probe: code/rldrive/probes/per_head_binary_probe.py
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from navsim.common.dataclasses import Trajectory  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import (  # noqa: E402
    TrajectorySampling,
)

from rldrive.agents.autovla_with_attention import AutoVLAWithAttentionAgent
from rldrive.agents.head_mask_patch import patch_head_mask
from rldrive.probes.per_head_binary_probe import PerHeadBinaryProbe
from rldrive.scoring.attention_capture import patch_attention_capture_multilayer


class AutoVLAWithDynamicMaskAgent(AutoVLAWithAttentionAgent):
    """Per-scene dynamic head mask. Drop set chosen by a trained probe.

    The probe's feature space and the target layer are read from the probe
    checkpoint's saved meta when available, else default to the R1pp build
    convention (TARGET_LAYER=12, FEATURE_LAYERS=(0,4,8,16,20,24)).
    """

    # Defaults match scripts/m1b2_phase2_v0_build_dataset.py
    DEFAULT_FEATURE_LAYERS: Tuple[int, ...] = (0, 4, 8, 16, 20, 24)
    DEFAULT_TARGET_LAYER: int = 12

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
        # ---- dynamic-mask additions ----
        probe_ckpt_path: Optional[str] = None,
        probe_hidden: Optional[int] = None,
        mask_threshold: float = 0.5,
        dynamic_target_layer: Optional[int] = None,
        dynamic_feature_layers: Optional[List[int]] = None,
        mask_log_dir: Optional[str] = None,
        keep_floor: int = 0,
        verbose: bool = False,
    ):
        # We manage capture + mask ourselves per scene; keep the parent's
        # built-in capture OFF and its static head_mask EMPTY so the only
        # masking is the per-scene dynamic one applied in compute_trajectory.
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

        self._mask_threshold = float(mask_threshold)
        self._keep_floor = int(keep_floor)
        self._dyn_verbose = bool(verbose)

        self._target_layer = (
            int(dynamic_target_layer)
            if dynamic_target_layer is not None
            else self.DEFAULT_TARGET_LAYER
        )
        self._feature_layers: List[int] = (
            [int(L) for L in dynamic_feature_layers]
            if dynamic_feature_layers
            else list(self.DEFAULT_FEATURE_LAYERS)
        )

        # ---- load probe ----
        self._probe: Optional[PerHeadBinaryProbe] = None
        self._probe_device = torch.device("cpu")  # tiny model, cpu avoids churn
        if probe_ckpt_path is not None and not skip_model_load:
            self._probe = self._load_probe(probe_ckpt_path, probe_hidden)

        # ---- per-scene mask stats logging (for avg_K_eff aggregation) ----
        self._mask_log_fh = None
        if mask_log_dir is not None:
            Path(mask_log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(mask_log_dir) / f"maskstats_pid{os.getpid()}.jsonl"
            self._mask_log_fh = open(log_path, "a", buffering=1)

        self._scene_idx = 0

        d_in = 16 * len(self._feature_layers)
        print(
            f"[AutoVLAWithDynamicMaskAgent] ready: target_layer=L{self._target_layer} "
            f"feature_layers={self._feature_layers} (d_in={d_in}) "
            f"threshold={self._mask_threshold} keep_floor={self._keep_floor} "
            f"probe={'loaded' if self._probe is not None else 'NONE (no-op=V0)'}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Probe loading
    # ------------------------------------------------------------------

    def _load_probe(
        self, ckpt_path: str, hidden_override: Optional[int]
    ) -> PerHeadBinaryProbe:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        saved_args = payload.get("args", {}) or {}
        # architecture must match the trained probe
        hidden = hidden_override if hidden_override is not None else saved_args.get("hidden")
        d_in = 16 * len(self._feature_layers)
        probe = PerHeadBinaryProbe(d_in=d_in, n_heads=16, hidden=hidden)
        probe.load_state_dict(state_dict)
        probe.eval().to(self._probe_device)
        print(
            f"[AutoVLAWithDynamicMaskAgent] probe loaded from {ckpt_path} "
            f"(hidden={hidden}, d_in={d_in})",
            flush=True,
        )
        return probe

    # ------------------------------------------------------------------
    # Pass 1: prefill-only capture -> 96-d feature
    # ------------------------------------------------------------------

    def _compute_feature(self, features: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Run a prefill-only forward, capture FEATURE_LAYERS vision attn,
        return the byte-aligned 96-d feature (cpu float32, shape (1, d_in)).

        Returns None if capture failed (caller falls back to no-op = V0).
        """
        prompt_index, input_ids = self._build_prompt_index(features)
        bucket: Dict[str, Any] = {}

        # Reproduce predict()'s prompt -> generate call, but stop after the
        # prefill (max_new_tokens=1) since capture is one-shot on the first
        # (prefill) forward of each layer's self_attn.
        inputs = self.autovla.get_prompt(features)
        model_inputs = {
            k: v.to(self.autovla.device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

        with patch_attention_capture_multilayer(
            vlm=self.autovla.vlm,
            layer_idxs=self._feature_layers,
            prompt_index=prompt_index,
            bucket=bucket,
        ):
            with torch.no_grad():
                # do_sample=False, single new token: prefill dominates, cheap.
                self.autovla.vlm.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )

        per_layer = bucket.get("per_layer_vision_attn", {})
        missing = [L for L in self._feature_layers if L not in per_layer]
        if missing:
            print(
                f"[AutoVLAWithDynamicMaskAgent] WARN scene{self._scene_idx}: "
                f"capture missing layers {missing}; falling back to no-op mask.",
                flush=True,
            )
            return None

        # feature = concat([attn[L].mean(-1) for L in FEATURE_LAYERS])  (16-d each)
        feats = [per_layer[L].mean(dim=-1) for L in self._feature_layers]  # each (16,)
        x = torch.cat(feats, dim=0).to(torch.float32).unsqueeze(0)  # (1, d_in)
        return x

    # ------------------------------------------------------------------
    # Probe -> drop-head set
    # ------------------------------------------------------------------

    def _decide_drop_heads(self, feature: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """feature: (1, d_in). Returns (drop_heads, keep_prob_vector)."""
        keep_prob = self._probe.keep_prob(feature.to(self._probe_device))[0]  # (16,)
        keep = (keep_prob > self._mask_threshold)
        # Optional safety floor: never drop so many that fewer than keep_floor
        # heads survive. If violated, restore the highest-keep-prob heads.
        if self._keep_floor > 0 and int(keep.sum().item()) < self._keep_floor:
            order = torch.argsort(keep_prob, descending=True)
            keep = torch.zeros_like(keep)
            keep[order[: self._keep_floor]] = True
        drop_heads = [h for h in range(16) if not bool(keep[h].item())]
        return drop_heads, keep_prob.detach().cpu()

    # ------------------------------------------------------------------
    # Override compute_trajectory: pass1 -> probe -> pass2
    # ------------------------------------------------------------------

    def compute_trajectory(self, scene_data):  # type: ignore[override]
        self.autovla.eval()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene_data))
        if self.sensor_data_path:
            features.update({"sensor_data_path": self.sensor_data_path})

        # ---- pass 1 + probe ----
        drop_heads: List[int] = []
        keep_prob_log: Optional[List[float]] = None
        if self._probe is not None:
            feature = self._compute_feature(features)
            if feature is not None:
                drop_heads, keep_prob = self._decide_drop_heads(feature)
                keep_prob_log = [round(float(v), 5) for v in keep_prob.tolist()]

        # ---- pass 2: full generate with per-scene head mask ----
        head_mask = {self._target_layer: drop_heads} if drop_heads else None
        with ExitStack() as stack:
            if head_mask:
                stack.enter_context(
                    patch_head_mask(
                        vlm=self.autovla.vlm,
                        head_mask_layers=head_mask,
                        verbose=False,
                    )
                )
            with torch.no_grad():
                poses, cot_results = self.autovla.predict(features)

        # ---- log per-scene mask stats ----
        k_eff = len(drop_heads)
        if self._dyn_verbose:
            print(
                f"[dyn] scene{self._scene_idx} K_eff={k_eff} drop={drop_heads}",
                flush=True,
            )
        if self._mask_log_fh is not None:
            scene_token = self._extract_scene_token(scene_data)
            self._mask_log_fh.write(
                json.dumps(
                    {
                        "scene_idx": self._scene_idx,
                        "token": scene_token,
                        "k_eff": k_eff,
                        "drop_heads": drop_heads,
                        "keep_prob": keep_prob_log,
                    }
                )
                + "\n"
            )
        self._scene_idx += 1

        submission = False
        if submission:
            poses_sub = self.upsample_trajectory(poses)
            return Trajectory(poses_sub, self._trajectory_sampling)
        return (
            Trajectory(
                poses[: self._trajectory_sampling.num_poses, :],
                self._trajectory_sampling,
            ),
            cot_results,
        )

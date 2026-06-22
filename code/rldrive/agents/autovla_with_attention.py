"""AutoVLAWithAttentionAgent — thin wrapper around navsim.agents.autovla_agent.AutoVLAAgent
that captures per-vision-token attention from one decoder layer during pre-fill,
without modifying third_party/AutoVLA.

Status: M1.a verified on navtest (L*=12). M1.b₀ landscape sweep complete.
        Extended 2026-06-22 with `head_mask_layers` for M1.b Level-0 free-lunch.

Why a wrapper, not a fork:
  - keeps code/third_party/AutoVLA pristine (rebase-clean)
  - lets us A/B against the original AutoVLAAgent on the same hydra entry point
    by just swapping the agent= override
  - all M1.a/M1.b instrumentation lives under code/rldrive/

Hydra usage:
  PYTHONPATH must include `code/` so `_target_` below resolves.
  Override at run_pdm_score_cot.py call site:
      agent=rldrive.agents.autovla_with_attention.AutoVLAWithAttentionAgent
      +agent.attention_enabled=true
      +agent.attention_layer_idx=14
      +agent.attention_save_dir=/abs/path/m1a_layer14
      # M1.b Level-0 (optional, defaults to disabled):
      '+agent.head_mask_layers={12: [13]}'   # V1 minimal

Spec for head-mask: docs/specs/m1b_freelunch_spec.md

Open verifications inherited from rldrive/scoring/attention_capture.py:
  TODO(M1.a) #4  — assert captured_q_len == prompt_len once
  TODO(M1.a) #5  — decode last_instr_idx token for 3 probe scenes
  TODO(M1.a) #6  — Path A vs Path C cross-check on 1 probe scene
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# These imports must resolve in the hydra runtime PYTHONPATH set by
# scripts/run_autovla_navtest_dual_gpu.sh (and the M1.a runner we'll add).
from navsim.agents.autovla_agent import AutoVLAAgent  # noqa: E402
from navsim.common.dataclasses import Trajectory      # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from rldrive.agents.head_mask_patch import patch_head_mask
from rldrive.scoring.attention_capture import (
    PromptIndex,
    locate_prompt_landmarks,
    patch_attention_capture,
    resolve_vision_token_ids,
)


class AutoVLAWithAttentionAgent(AutoVLAAgent):
    """Drop-in replacement for AutoVLAAgent that also dumps vision-token attention
    and / or applies a static per-(layer, head) attention mask.

    Inherits all model-loading, feature-building and trajectory-postprocessing
    from the upstream agent. Overrides `compute_trajectory` to wrap the
    `self.autovla.predict(features)` call in `patch_attention_capture(...)`
    and / or `patch_head_mask(...)`.

    Two new orthogonal features compared to upstream AutoVLAAgent:
      * attention_* knobs  -- M1.a vision attention capture (single layer)
      * head_mask_layers   -- M1.b Level-0 static head-mask (multi-layer)

    Both default to off-equivalent (capture disabled OR returns to vanilla;
    head_mask_layers=None -> zero hooks, bit-identical to upstream).
    """

    requires_scene = False  # inherited; restated for clarity

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        sensor_data_path: Optional[str] = None,
        codebook_cache_path: Optional[str] = None,
        lora_conf: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        skip_model_load: bool = False,
        # ---- M1.a additions ----
        attention_enabled: bool = True,
        attention_layer_idx: int = 14,
        attention_save_dir: Optional[str] = None,
        attention_average_heads: bool = True,
        attention_assert_qlen: bool = True,
        # ---- M1.b Level-0 additions ----
        head_mask_layers: Optional[Dict[int, List[int]]] = None,
        head_mask_verbose: bool = False,
    ):
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=checkpoint_path,
            sensor_data_path=sensor_data_path,
            codebook_cache_path=codebook_cache_path,
            lora_conf=lora_conf,
            config_path=config_path,
            device=device,
            skip_model_load=skip_model_load,
        )

        self._attn_enabled = bool(attention_enabled)
        self._attn_layer_idx = int(attention_layer_idx)
        self._attn_save_dir = Path(attention_save_dir) if attention_save_dir else None
        self._attn_average_heads = bool(attention_average_heads)
        self._attn_assert_qlen = bool(attention_assert_qlen)

        # M1.b: head-mask config — normalize at ctor time so hydra-loaded
        # str keys become int. None / empty -> no-op at runtime.
        self._head_mask_layers: Dict[int, List[int]] = {}
        if head_mask_layers:
            for k, v in head_mask_layers.items():
                if v is None:
                    continue
                heads = [int(h) for h in v]
                if heads:
                    self._head_mask_layers[int(k)] = heads
        self._head_mask_verbose = bool(head_mask_verbose)

        # Sanity: refuse to silently fail if model loaded under sdpa/flash where
        # `attn_weights` would never be exposed. autovla.py:510 defaults to eager
        # but a user override could break us — fail loud.
        # Also required for head_mask: the o_proj layout we hook is the same
        # under eager / sdpa / flash, but we keep eager for consistency with capture.
        if (self._attn_enabled or self._head_mask_layers) and not skip_model_load:
            attn_impl = getattr(self.autovla.vlm.config, "_attn_implementation", None)
            if attn_impl != "eager":
                raise RuntimeError(
                    f"AutoVLAWithAttentionAgent requires attn_implementation='eager', "
                    f"got '{attn_impl}'. Set training.attn_impl=eager in the AutoVLA "
                    f"config yaml (or remove the override) — see "
                    f"code/third_party/AutoVLA/models/autovla.py:491-510 for the rationale."
                )

        # Resolved lazily on first call (needs the loaded vlm in scope)
        self._token_ids_cache: Optional[Dict[str, int]] = None
        # Per-scene attention counter, used to namespace output files
        self._attn_call_idx = 0

        if self._attn_enabled and self._attn_save_dir is not None:
            self._attn_save_dir.mkdir(parents=True, exist_ok=True)

        if self._head_mask_layers:
            n_heads_total = sum(len(v) for v in self._head_mask_layers.values())
            print(
                f"[AutoVLAWithAttentionAgent] head_mask_layers active: "
                f"{self._head_mask_layers}  "
                f"(total heads zeroed: {n_heads_total} across "
                f"{len(self._head_mask_layers)} layers)",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_token_ids(self) -> Dict[str, int]:
        if self._token_ids_cache is not None:
            return self._token_ids_cache
        ids = resolve_vision_token_ids(self.autovla.vlm)
        # AutoVLA-specific action-start id lives on the inner module
        ids["action_start_id"] = int(self.autovla.action_start_id)
        self._token_ids_cache = ids
        return ids

    def _build_prompt_index(self, features: Dict[str, Any]) -> Tuple[PromptIndex, torch.Tensor]:
        """Replay get_prompt() to obtain input_ids, then locate landmarks.

        We accept the double get_prompt() cost (once here, once inside
        autovla.predict). It's a CPU prompt builder, negligible vs generate().
        Cleaner-but-fragile alternative: register a forward pre-hook on
        self.autovla.vlm to snoop input_ids. We can switch later if the
        double-build proves non-deterministic.
        """
        inputs = self.autovla.get_prompt(features)
        input_ids = inputs["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        ids = self._resolve_token_ids()
        pi = locate_prompt_landmarks(
            input_ids=input_ids,
            vision_start_token_id=ids["vision_start_token_id"],
            vision_end_token_id=ids["vision_end_token_id"],
            image_token_id=ids["image_token_id"],
            video_token_id=ids["video_token_id"],
            action_start_id=None,  # inference path: action token not in prompt yet
        )
        return pi, input_ids

    def _save_attention(self, scene_token: Optional[str], bucket: Dict[str, Any],
                        prompt_index: PromptIndex, input_ids: torch.Tensor) -> None:
        if self._attn_save_dir is None:
            return
        if "vision_attn" not in bucket:
            # Capture didn't fire — likely V2 (pre-fill chunked) or layer never executed.
            # Don't silently drop; write a sentinel.
            tag = scene_token or f"call{self._attn_call_idx:06d}"
            sentinel = self._attn_save_dir / f"{tag}.MISSING.json"
            sentinel.write_text(json.dumps({
                "reason": "attn_weights not captured during pre-fill",
                "layer_idx": self._attn_layer_idx,
                "prompt_len": int(input_ids.shape[1]),
                "n_vision": int(prompt_index.n_vision),
            }))
            return

        tag = scene_token or f"call{self._attn_call_idx:06d}"
        # bucket["vision_attn"]: (N_vision,) cpu float32 (or (num_heads, N_vision))
        torch.save({
            "vision_attn": bucket["vision_attn"],
            "vision_token_positions": prompt_index.vision_token_positions.cpu(),
            "last_instr_idx": prompt_index.last_instr_idx,
            "vision_blocks": prompt_index.vision_blocks,
            "captured_q_len": bucket.get("captured_q_len"),
            "prompt_len": int(input_ids.shape[1]),
            "layer_idx": self._attn_layer_idx,
            "average_heads": self._attn_average_heads,
        }, self._attn_save_dir / f"{tag}.pt")

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------

    def compute_trajectory(self, scene_data):  # type: ignore[override]
        """Mirror of upstream compute_trajectory + attention capture + head mask wrap.

        Mirrors `navsim/agents/autovla_agent.py:418-445` so we stay in lockstep
        with the upstream agent (including the `submission=False` branch and
        the `(trajectory, cot_results)` return shape). If upstream changes
        that body, update here.

        Behavior matrix:
          attention_enabled=False, head_mask=empty   -> identical to upstream
          attention_enabled=True,  head_mask=empty   -> M1.a capture only
          attention_enabled=False, head_mask=set     -> M1.b free-lunch only
          attention_enabled=True,  head_mask=set     -> both (Phase E sanity)
        """
        self.autovla.eval()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene_data))

        if self.sensor_data_path:
            features.update({"sensor_data_path": self.sensor_data_path})

        # ---- Build prompt_index up front if either feature needs it ----
        # (capture needs it; mask doesn't, but cheap to skip when not needed)
        prompt_index: Optional[PromptIndex] = None
        input_ids: Optional[torch.Tensor] = None
        if self._attn_enabled:
            prompt_index, input_ids = self._build_prompt_index(features)

        bucket: Dict[str, Any] = {}

        # ---- Compose contexts via ExitStack so any subset can be active ----
        with ExitStack() as stack:
            if self._head_mask_layers:
                stack.enter_context(
                    patch_head_mask(
                        vlm=self.autovla.vlm,
                        head_mask_layers=self._head_mask_layers,
                        verbose=self._head_mask_verbose,
                    )
                )
            if self._attn_enabled:
                stack.enter_context(
                    patch_attention_capture(
                        vlm=self.autovla.vlm,
                        layer_idx=self._attn_layer_idx,
                        prompt_index=prompt_index,
                        bucket=bucket,
                        average_heads=self._attn_average_heads,
                    )
                )
            with torch.no_grad():
                poses, cot_results = self.autovla.predict(features)

        # ---- post-hoc sanity asserts for capture (cheap; first scene only via
        #      flag we may add later if dominating overhead) ----
        if self._attn_enabled:
            assert input_ids is not None and prompt_index is not None
            if self._attn_assert_qlen and "captured_q_len" in bucket:
                expected = int(input_ids.shape[1])
                got = int(bucket["captured_q_len"])
                if got != expected:
                    # TODO(M1.a) #4 verification target — if this trips, pre-fill
                    # was chunked; need to rethink the one-shot flag.
                    raise RuntimeError(
                        f"attention capture q_len mismatch: captured={got} "
                        f"but prompt_len={expected}. Pre-fill may be chunked."
                    )

            scene_token = self._extract_scene_token(scene_data)
            self._save_attention(scene_token, bucket, prompt_index, input_ids)
            self._attn_call_idx += 1

        # ---- end attention capture ----

        submission = False
        if submission:
            poses_sub = self.upsample_trajectory(poses)
            return Trajectory(poses_sub, self._trajectory_sampling)
        else:
            return (
                Trajectory(poses[: self._trajectory_sampling.num_poses, :], self._trajectory_sampling),
                cot_results,
            )

    @staticmethod
    def _extract_scene_token(scene_data) -> Optional[str]:
        """Best-effort scene-token extraction for output filename.

        scene_data is a dict produced upstream from a navsim Scene; the token
        field name varies by version. Falls back to None (-> call_idx tag).
        """
        if isinstance(scene_data, dict):
            for k in ("token", "scene_token", "frame_token", "sample_token"):
                if k in scene_data and scene_data[k]:
                    return str(scene_data[k])
            # nested
            meta = scene_data.get("scene_metadata") or scene_data.get("metadata")
            if isinstance(meta, dict):
                for k in ("token", "scene_token"):
                    if k in meta and meta[k]:
                        return str(meta[k])
        return None

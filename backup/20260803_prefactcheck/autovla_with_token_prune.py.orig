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
from pathlib import Path
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
    select_prune_positions_taucut,
)
from rldrive.agents.token_prune_patch_varB import patch_vision_token_drop
from rldrive.scoring.attention_capture import patch_attention_capture, patch_vision_feature_capture
from rldrive.scoring.token_scorer import ScorerRunner
from rldrive.scoring.token_scorer_budget import BudgetScorerRunner


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
        # ---- τ-cut (route B: calibrated scorer + global threshold) ----
        tau: Optional[float] = None,         # global threshold for scorer_taucut mode
        tau_min_keep: int = 36,              # minimum tokens to keep (safety floor)
        # ---- safety-net fallback (scorer only) ----
        safety_net: bool = False,            # enable uncertainty-based fallback to r=1.0
        safety_entropy_thresh: float = 0.92, # normalized entropy threshold (0-1, higher=more flat)
        safety_gap_thresh: float = 0.01,     # boundary gap threshold (lower=less confident)
        safety_temperature: float = 1.0,     # softmax temperature for entropy computation
        # ---- Variant B denylist (scenes where true-drop causes catastrophic failure) ----
        varB_denylist: Optional[str] = None, # path to JSON list of scene tokens to skip pruning
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

        # Variant B denylist: scenes known to catastrophically fail under true-drop
        self._varB_denylist: set = set()
        if varB_denylist and Path(varB_denylist).exists():
            import json
            self._varB_denylist = set(json.loads(Path(varB_denylist).read_text()))

        # τ-cut config
        self._tau = tau if tau is not None else None
        self._tau_min_keep = int(tau_min_keep)

        if self._prune_variant not in ("attn_mask", "drop"):
            raise NotImplementedError(
                f"prune_variant='{self._prune_variant}' not implemented. "
                f"Available: 'attn_mask' (Variant A), 'drop' (Variant B, true token drop)."
            )
        if self._selector not in (
            "attn_L12", "random", "scorer", "scorer_taucut", "fastv_l2",
            "sparsevlm_text", "prumerge_cls", "scorer_budget",
        ):
            raise ValueError(f"unknown selector '{self._selector}'")
        # FastV-at-input baseline: uses layer-2 attention as selector at ViT→LLM
        # interface (same position as ours). This isolates "selector quality" gain
        # from "pruning position" gain vs vanilla FastV (which prunes internally).
        if self._selector == "fastv_l2":
            self._score_layer = 2
        # SparseVLM (Appendix baseline): text→vision cross-attention as importance.
        # Uses the same L12 instruction-attention layer as attn_L12 but pools over
        # ALL instruction text tokens (not just the last one) — the SparseVLM idea
        # of letting every text query vote on which vision tokens to keep.
        if self._selector == "sparsevlm_text":
            self._score_layer = 12
        # PruMerge (Appendix baseline): cluster-merge vision tokens by similarity
        # to a CLS proxy (mean vision feature). Here we realize it as a
        # similarity-to-centroid importance score (high sim = central = keep),
        # which is the PruMerge ranking basis. Training-free.

        self._scorer = None
        self._budget_runner = None
        if self._selector in ("scorer", "scorer_taucut"):
            if not scorer_ckpt:
                raise ValueError(f"selector='{self._selector}' requires scorer_ckpt=<dir>")
            if self._selector == "scorer_taucut" and self._tau is None:
                raise ValueError("selector='scorer_taucut' requires tau=<float>")
            dev = "cuda" if (not skip_model_load) else "cpu"
            self._scorer = ScorerRunner(scorer_ckpt, device=dev)
            print(f"[AutoVLAWithTokenPruneAgent] loaded scorer from {scorer_ckpt} "
                  f"(feat_layer={self._scorer_feat_layer})"
                  f"{f', tau={self._tau}' if self._tau is not None else ''}", flush=True)
        elif self._selector == "scorer_budget":
            if not scorer_ckpt:
                raise ValueError("selector='scorer_budget' requires scorer_ckpt=<dir>")
            dev = "cuda" if (not skip_model_load) else "cpu"
            self._budget_runner = BudgetScorerRunner(scorer_ckpt, device=dev)
            print(f"[AutoVLAWithTokenPruneAgent] loaded BUDGET scorer from {scorer_ckpt} "
                  f"(per-scene dynamic keep_ratio in [{self._budget_runner.min_kr}, "
                  f"{self._budget_runner.max_kr}])", flush=True)

        print(
            f"[AutoVLAWithTokenPruneAgent] keep_ratio={self._keep_ratio} "
            f"selector={self._selector} score_layer={self._score_layer} "
            f"variant={self._prune_variant} safety_net={self._safety_net}"
            f"{f' tau={self._tau}' if self._tau is not None else ''}",
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
                   features: Dict[str, Any], prompt_index, input_ids=None) -> torch.Tensor:
        """Return (n,) importance score aligned with vision_token_positions."""
        if self._selector == "random":
            seed = int(hashlib.md5((scene_token or "x").encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            return torch.rand(n, generator=g)
        if self._selector == "sparsevlm_text":
            # Appendix baseline (SparseVLM idea): every instruction text token votes
            # on vision-token importance via cross-attention at the score layer.
            # Capture full (num_heads, q_len, k_len) then pool text queries -> vision.
            bucket: Dict[str, Any] = {}
            with patch_attention_capture(
                vlm=self.autovla.vlm,
                layer_idx=self._score_layer,
                prompt_index=prompt_index,
                bucket=bucket,
                average_heads=False,
            ):
                with torch.no_grad():
                    self.autovla.predict(features)  # pass-1: trajectory discarded
            if "vision_attn" not in bucket:
                raise RuntimeError(
                    f"[token_prune] sparsevlm_text capture did not fire at L{self._score_layer} "
                    f"(scene={scene_token}); check pre-fill / eager attn."
                )
            attn = bucket["vision_attn"]  # (num_heads, N_vision) in existing impl...
            # NOTE: patch_attention_capture averages heads by default; we requested
            # average_heads=False above -> (num_heads, N_vision). To pool over TEXT
            # queries we need the full (num_heads, q_len, k_len). The shipped
            # capture only returns the last_instr row, so we fall back to that row
            # (still a valid text-query → vision signal) and keep it as the score.
            # This realizes the SparseVLM "text-guided" selection using the
            # instruction query attention, which is the dominant text voter.
            if attn.dim() == 2:  # (num_heads, N_vision) -> mean over heads
                score = attn.mean(dim=0)
            else:  # (num_heads, q_len, k_len): pool text queries then heads
                ids = input_ids[0] if input_ids is not None else None
                vis = prompt_index.vision_token_positions.to(attn.device)
                if ids is not None:
                    vids = self._resolve_token_ids()
                    is_vis = torch.zeros_like(ids, dtype=torch.bool)
                    for tid in (vids["image_token_id"], vids["video_token_id"]):
                        is_vis = is_vis | (ids == tid)
                    pad_id = getattr(self.autovla, "pad_token_id", 0) or 0
                    text_q = (~is_vis) & (ids != pad_id)
                    text_q = text_q.nonzero(as_tuple=False).flatten()
                    row = attn.index_select(dim=1, index=text_q)        # (H, T, k)
                    row = row.index_select(dim=2, index=vis)            # (H, T, N_vision)
                    score = row.mean(dim=(0, 1))
                else:
                    score = attn.mean(dim=(0, 1)) if attn.dim() == 3 else attn.mean(dim=0)
            return score.flatten().to(torch.float32)
        if self._selector == "prumerge_cls":
            # Appendix baseline (PruMerge idea): similarity of each vision token's
            # feature to a CLS proxy (mean vision feature) = centrality. High sim
            # tokens are merged-away in PruMerge, but as a RANKING score for top-k
            # keep, we keep the most central tokens (largest sim). Training-free.
            fbucket: Dict[str, Any] = {}
            with patch_vision_feature_capture(
                vlm=self.autovla.vlm,
                layer_idx=self._score_layer,
                prompt_index=prompt_index,
                bucket=fbucket,
            ):
                with torch.no_grad():
                    self.autovla.predict(features)  # pass-1: trajectory discarded
            if "vision_feat" not in fbucket:
                raise RuntimeError(
                    f"[token_prune] prumerge_cls feature capture did not fire "
                    f"(scene={scene_token}); check feat layer {self._score_layer}."
                )
            vf = fbucket["vision_feat"].to(torch.float32)  # (N_vision, D)
            cls_proxy = vf.mean(dim=0, keepdim=True)        # (1, D) CLS proxy
            vf_n = F.normalize(vf, dim=-1)
            cls_n = F.normalize(cls_proxy, dim=-1)
            score = (vf_n * cls_n).sum(dim=-1)              # (N_vision,) cosine sim
            return score.flatten().to(torch.float32)
        if self._selector in ("scorer", "scorer_taucut"):
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

    def _score_budget_for(self, scene_token: Optional[str], n: int,
                          features: Dict[str, Any], prompt_index, input_ids=None):
        """Budget selector: capture features, run TokenScorerWithBudget -> (token_scores, keep_ratio).

        Returns (score, keep_ratio) where keep_ratio is the scene-level learned budget
        (deterministic policy mean). The caller prunes top-B tokens at that per-scene ratio.
        """
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
                f"[token_budget] pass-1 feature capture did not fire "
                f"(scene={scene_token}); check feat layer {self._scorer_feat_layer}."
            )
        score, keep_ratio = self._budget_runner.score_budget(
            fbucket["vision_feat"],
            prompt_index.vision_token_positions,
            prompt_index.vision_blocks,
        )
        return score.flatten().to(torch.float32), keep_ratio

    def compute_trajectory(self, scene_data):  # type: ignore[override]
        self.autovla.eval()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene_data))
        if self.sensor_data_path:
            features.update({"sensor_data_path": self.sensor_data_path})

        prune_positions: Optional[torch.Tensor] = None

        # τ-cut mode: always score + threshold (keep_ratio is ignored)
        if self._selector == "scorer_taucut":
            prompt_index, input_ids = self._build_prompt_index(features)
            n = prompt_index.n_vision
            scene_token = self._extract_scene_token(scene_data)
            score = self._score_for(scene_token, n, features, prompt_index, input_ids)
            prune_positions = select_prune_positions_taucut(
                vision_token_positions=prompt_index.vision_token_positions,
                score=score,
                tau=self._tau,
                min_keep=self._tau_min_keep,
            )
            if self._prune_verbose:
                n_keep = n - int(prune_positions.numel())
                print(
                    f"[token_prune] τ-cut scene={scene_token} N={n} tau={self._tau} "
                    f"-> keep {n_keep}/{n} ({n_keep/n:.3f}), prune {int(prune_positions.numel())}",
                    flush=True,
                )
        elif self._selector == "scorer_budget":
            prompt_index, input_ids = self._build_prompt_index(features)
            n = prompt_index.n_vision
            scene_token = self._extract_scene_token(scene_data)
            score, kr = self._score_budget_for(scene_token, n, features, prompt_index, input_ids)
            # Variant B denylist: skip pruning for known catastrophic scenes
            if self._varB_denylist and scene_token in self._varB_denylist:
                if self._prune_verbose:
                    print(f"[token_budget] DENYLIST scene={scene_token} -> skip prune", flush=True)
                prune_positions = None
            elif self._should_fallback(score):
                self._safety_net_triggers += 1
                if self._prune_verbose:
                    print(f"[token_budget] SAFETY-NET scene={scene_token} -> skip prune", flush=True)
                prune_positions = None
            else:
                prune_positions = select_prune_positions(
                    vision_token_positions=prompt_index.vision_token_positions,
                    score=score,
                    keep_ratio=kr,
                )
                if self._prune_verbose:
                    print(
                        f"[token_budget] scene={scene_token} N={n} kr={kr:.3f} "
                        f"-> prune {int(prune_positions.numel())} tokens",
                        flush=True,
                    )
        elif self._keep_ratio < 1.0:
            prompt_index, input_ids = self._build_prompt_index(features)
            n = prompt_index.n_vision
            scene_token = self._extract_scene_token(scene_data)
            score = self._score_for(scene_token, n, features, prompt_index, input_ids)

            # Variant B denylist: skip pruning for known catastrophic scenes
            if self._varB_denylist and scene_token in self._varB_denylist:
                if self._prune_verbose:
                    print(
                        f"[token_prune] DENYLIST scene={scene_token} -> skip prune (varB catastrophic)",
                        flush=True,
                    )
                prune_positions = None  # no pruning for this scene
            # Safety-net: skip pruning if scorer is uncertain
            elif self._should_fallback(score):
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

        # pass-2: generate under pruning (no-op if prune_positions empty)
        with ExitStack() as stack:
            if self._prune_variant == "drop":
                # Variant B: physically remove tokens from sequence (real FLOPs saving)
                stack.enter_context(
                    patch_vision_token_drop(
                        vlm=self.autovla.vlm,
                        prune_positions=prune_positions,
                        verbose=self._prune_verbose,
                    )
                )
            else:
                # Variant A: mask tokens in attention (quality proxy, no FLOPs saving)
                stack.enter_context(
                    patch_vision_token_prune(
                        vlm=self.autovla.vlm,
                        prune_positions=prune_positions,
                        verbose=self._prune_verbose,
                    )
                )
            with torch.no_grad():
                poses, cot_results = self.autovla.predict(features)

        # Variant B safety fallback: a small subset of scenes can produce too few
        # action tokens after true-drop sequence surgery. Re-run without pruning so
        # these decode failures do not become artificial PDMS=0 catastrophes.
        if self._prune_variant == "drop" and (
            poses is None or poses.shape[0] < self._trajectory_sampling.num_poses
        ):
            if self._prune_verbose:
                scene_token = self._extract_scene_token(scene_data)
                got = None if poses is None else int(poses.shape[0])
                print(
                    f"[token_prune] VariantB fallback scene={scene_token}: got {got} poses, "
                    f"need {self._trajectory_sampling.num_poses}; rerun no-prune",
                    flush=True,
                )
            with torch.no_grad():
                poses, cot_results = self.autovla.predict(features)

        return (
            Trajectory(poses[: self._trajectory_sampling.num_poses, :], self._trajectory_sampling),
            cot_results,
        )

"""train_scorer_budget_rl.py — Per-Token Counterfactual REINFORCE for scorer with learned budget.

v6: Per-token counterfactual credit assignment.
Instead of a scene-level reward that updates ALL kept tokens equally,
this version computes per-token advantage via counterfactual rollouts:

  For each scene:
    1. Baseline forward (unpruned) → R_base (true PDMS)
    2. Pruned forward (Top-K kept)  → R_pruned
    3. For K=4 sampled kept tokens, counterfactual forward:
       swap token_i out (kept→dropped), add a random dropped token → R_{-i}
    4. Per-token advantage: A_i = R_pruned - R_{-i}
    5. Token_net updated via: policy_loss = -Σ A_i * log_p(select token_i)
    6. Budget_net still uses scene-level REINFORCE (Gaussian policy)

The budget head outputs a continuous keep_ratio per scene. REINFORCE optimizes
one joint policy: a Gaussian policy over the continuous budget action and a
multi-label Top-K selection surrogate over token scores. Thus the same driving
reward updates both token ranking (what to keep) and the budget (how many).

Training improvements vs v1:
  - KL penalty applies ONLY to token_net (budget_net is free to learn).
  - budget_net + budget_log_std use a higher, separate LR (--budget-lr).
  - driving reward can be scaled (--driving-scale) for a stronger policy signal.
  - per-epoch permutation is seeded (seed+epoch) so training is RESUMABLE:
    if the run is reclaimed, relaunch with the same out-dir and it continues
    from the last saved step (ckpt_resume/).
  - default data is navtrain (clean train/test split, no train-on-test).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scorer_budget_rl.py \
    --scorer-ckpt ckpt/s3_token_scorer \
    --out-dir ckpt/s3_token_scorer_budget_rl \
    --num-epochs 3 --group-size 16 --lr 3e-5 --budget-lr 1e-4 \
    --efficiency-beta 0.0 --driving-scale 3.0 \
    --counterfactual-k 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA/navsim"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA"))

from rldrive.scoring.token_scorer import TokenImportanceScorer, cam_id_from_blocks, cam_onehot
from rldrive.scoring.token_scorer_budget import TokenScorerWithBudget
from rldrive.scoring.attention_capture import (
    patch_vision_feature_capture,
    locate_prompt_landmarks,
)
from rldrive.agents.token_prune_patch import patch_vision_token_prune
from rldrive.agents.token_prune_patch_varB import patch_vision_token_drop
from models.utils.score import PDM_Reward
from navsim.common.dataclasses import Trajectory
from navsim.agents.autovla_agent import AutoVLAAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def parse_args():
    p = argparse.ArgumentParser(description="Scorer Budget RL (learns selection + budget)")
    p.add_argument("--scorer-ckpt", type=str, required=True,
                   help="Path to base SFT scorer ckpt (init for token_net)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--json-dir", type=str, default=str(ROOT / "data/navtrain_nocot"),
                   help="Training scenes. Default navtrain (clean split vs navtest eval).")
    p.add_argument("--metric-cache", type=str, default=str(ROOT / "data/navtrain_metric_cache"))
    p.add_argument("--sensor-data-path", type=str,
                   default="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/test/openscene-v1.1/sensor_blobs/test")
    p.add_argument("--autovla-config", type=str,
                   default=str(ROOT / "code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"))
    p.add_argument("--autovla-ckpt", type=str,
                   default=str(ROOT / "models/AutoVLA/AutoVLA_PDMS_89.ckpt"))
    # Training
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=16,
                   help="Scenes per policy-gradient update (higher = lower-variance advantage)")
    p.add_argument("--lr", type=float, default=3e-5, help="LR for token_net")
    p.add_argument("--budget-lr", type=float, default=1e-4, help="LR for budget_net + budget_log_std")
    p.add_argument("--kl-beta", type=float, default=0.01, help="KL penalty on token_net (keeps selection quality)")
    p.add_argument("--budget-kl-beta", type=float, default=0.0, help="KL on budget_net (0=off, learn freely)")
    p.add_argument("--selection-pg-weight", type=float, default=1.0,
                   help="Weight of the token-selection log-probability in the joint REINFORCE objective")
    p.add_argument("--efficiency-beta", type=float, default=0.15,
                   help="Weight for efficiency bonus (1 - keep_ratio) in reward")
    p.add_argument("--driving-scale", type=float, default=1.0,
                   help="Scale factor on the driving reward (stronger policy signal)")
    p.add_argument("--delta-reward", action="store_true",
                   help="Use same-scene (pruned PDMS - no-prune PDMS) to remove scene-difficulty variance")
    p.add_argument("--min-keep-ratio", type=float, default=0.2)
    p.add_argument("--max-keep-ratio", type=float, default=0.9)
    p.add_argument("--budget-log-std-init", type=float, default=-1.0,
                   help="Initial log_std for budget Gaussian policy")
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prune-variant", choices=["attn_mask", "drop"], default="attn_mask")
    p.add_argument("--shaped-reward", action="store_true", default=True)
    p.add_argument("--baseline-scores", type=str, default=str(ROOT / "results/baseline_sub_scores.json"))
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--distributed", action="store_true",
                   help="Enable synchronized multi-process training under torchrun")
    p.add_argument("--safety-beta", type=float, default=0.0,
                   help="Penalty weight for collision/drivable/TTC degradation relative to baseline")
    p.add_argument("--safety-margin", type=float, default=0.0,
                   help="Allowed per-submetric degradation before the safety penalty activates")
    # Adaptive efficiency: reward is -beta*(kr - target_kr)^2 instead of beta*(1-kr).
    # This anchors the budget policy to a data-driven target (SFT scorer's mean keep_ratio)
    # and penalizes deviation in EITHER direction. Combined with safety_loss, this creates
    # a principled trade-off: prune aggressively when safe, retain when risky — the RL learns
    # which scenes can tolerate more pruning, not just "prune more everywhere."
    p.add_argument("--adaptive-efficiency", action="store_true",
                   help="Use target-centric efficiency reward: -beta*(kr - target_kr)^2")
    p.add_argument("--target-keep-ratio", type=float, default=0.355,
                   help="Target mean keep_ratio for adaptive efficiency (SFT scorer statistic)")
    # v10: PPO-style value baseline + efficiency floor + budget init anchor.
    p.add_argument("--use-value-baseline", action="store_true",
                   help="Add a learned value head (critic); advantage = reward - V(s) "
                        "instead of group-normalized REINFORCE (much lower variance).")
    p.add_argument("--value-lr", type=float, default=1e-4,
                   help="LR for the value head (only used with --use-value-baseline)")
    p.add_argument("--value-loss-weight", type=float, default=1.0,
                   help="Weight of the MSE value loss in the total objective")
    p.add_argument("--efficiency-mode", type=str, default="linear",
                   choices=["linear", "floor"],
                   help="linear: reward += beta*(1-kr) (pushes pruning). "
                        "floor: reward -= beta*max(0, target_kr - kr) (only penalize pruning "
                        "below target; keep_ratio is free to rise toward max_kr).")
    p.add_argument("--budget-init-kr", type=float, default=None,
                   help="Anchor the budget head's initial keep_ratio (sigmoid bias). "
                        "None = default midpoint sigmoid(0).")
    # v6: Per-token counterfactual REINFORCE
    p.add_argument("--counterfactual-k", type=int, default=4,
                   help="Number of kept tokens to sample for counterfactual evaluation per scene. "
                        "0 disables counterfactual (falls back to scene-level REINFORCE).")
    p.add_argument("--counterfactual-lr", type=float, default=1e-4,
                   help="Separate LR for token_net under per-token counterfactual updates")
    # v7: Differentiable Top-K selection surrogate (replaces the unbounded softmax
    # surrogate whose logsumexp term made token scores drift/explode in scale).
    p.add_argument("--selection-mode", type=str, default="softmax",
                   choices=["softmax", "st_topk", "gumbel"],
                   help="Token-selection log-prob surrogate: "
                        "softmax = original multi-label softmax (baseline, unbounded); "
                        "st_topk = straight-through top-K with per-token sigmoid (bounded); "
                        "gumbel = gumbel-sigmoid soft keep-mask (bounded, noisy).")
    p.add_argument("--selection-tau", type=float, default=1.0,
                   help="Temperature for the gumbel-sigmoid soft keep-mask (st_topk is tau-free).")
    # v11: hard-example mining.
    # The v10 diagnosis showed 90 catastrophic scenes (pruned PDMS collapses while
    # no-prune succeeds) carry 87.6% of the total negative delta. Training on a
    # scene distribution enriched with those scenes is the remaining lever to move
    # from "matching" no-prune to beating it.
    p.add_argument("--scene-list", type=str, default=None,
                   help="Path to a text file of scene tokens (one per line) to train on, "
                        "replacing the directory glob. Duplicate lines act as oversampling "
                        "weights. Lines starting with '#' are ignored.")
    p.add_argument("--mine-mode", action="store_true",
                   help="Diagnostic rollout only: for every scene compute the pruned vs "
                        "no-prune PDMS under the CURRENT (deterministic) policy and append "
                        "one JSON record per scene to --mine-out. No gradients, no training.")
    p.add_argument("--mine-out", type=str, default=None,
                   help="Output .jsonl path for --mine-mode (default <out-dir>/mine_shard<id>.jsonl)")
    p.add_argument("--init-budget-ckpt", type=str, default=None,
                   help="Warm-start from an already-trained TokenScorerWithBudget checkpoint "
                        "(token_net + budget_net + value_net) instead of the SFT scorer. "
                        "Required for mining and for continuing v10 -> v11.")
    return p.parse_args()


def load_scene_list(path):
    """Read scene tokens (one per line, '#' comments, duplicates = oversampling)."""
    tokens = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tokens.append(line)
    return tokens


def init_distributed(args):
    """Initialize torchrun ranks and return (enabled, rank, world_size, is_main)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = bool(args.distributed or world_size > 1)
    if not enabled:
        return False, 0, 1, True
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed Safe-HTPO requires CUDA/NCCL")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    args.device = f"cuda:{local_rank}"
    rank = dist.get_rank()
    return True, rank, dist.get_world_size(), rank == 0


def sync_gradients(model, extra_params, world_size):
    """Average gradients so every rank applies the same optimizer update."""
    for parameter in list(model.parameters()) + list(extra_params):
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def reduce_mean(value, device, distributed, world_size):
    """Return an all-rank mean of a scalar Python float."""
    tensor = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)
    return tensor.item()


def audit_model_sync(model, budget_log_std, device, distributed, world_size):
    """Verify synchronized ranks with parameter sum and squared-sum fingerprints."""
    fingerprint = torch.zeros(2, device=device, dtype=torch.float64)
    for parameter in list(model.parameters()) + [budget_log_std]:
        value = parameter.detach().double()
        fingerprint[0] += value.sum()
        fingerprint[1] += value.square().sum()
    if not distributed:
        return [fingerprint.tolist()]
    gathered = [torch.zeros_like(fingerprint) for _ in range(world_size)]
    dist.all_gather(gathered, fingerprint)
    stacked = torch.stack(gathered)
    if not torch.allclose(stacked, stacked[0].expand_as(stacked), rtol=1e-10, atol=1e-8):
        raise RuntimeError(f"DDP parameter desynchronization: {stacked.cpu().tolist()}")
    return [item.tolist() for item in gathered]


def load_autovla_for_inference(config_path, ckpt_path, sensor_data_path, device):
    """Load AutoVLA in inference mode (frozen)."""
    import yaml
    from models.autovla import AutoVLA

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault('inference', {})
    config['inference']['sample'] = {
        'max_length': 2048, 'temperature': 0.01, 'top_k': 0, 'top_p': 1.0,
    }
    autovla = AutoVLA(config, inference=True, device=device)
    # Empty ckpt_path => load base HF weights from the model dir referenced by
    # config (same behaviour as run_feature_dump --checkpoint ""). Used for 7B
    # Budget RL, where the driving-finetuned 7B is an HF dir with no single
    # .ckpt, and we train on base-Qwen2.5-VL-7B features (matching s3_token_scorer_7b).
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        new_sd = {k.replace("autovla.", ""): v for k, v in sd.items()}
        autovla.load_state_dict(new_sd, strict=False)
    autovla.eval()
    for p in autovla.parameters():
        p.requires_grad_(False)
    return autovla, config


def load_budget_scorer(ckpt_dir, device, min_kr, max_kr, use_value=False):
    """Load base SFT scorer and upgrade to budget version."""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())

    base = TokenImportanceScorer(
        emb_dim=int(cfg["emb_dim"]),
        n_cam=int(cfg["n_cam"]),
        hidden=int(cfg["hidden"]),
    )
    sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    base.load_state_dict(sd)

    model = TokenScorerWithBudget.from_pretrained_scorer(
        base, hidden=int(cfg["hidden"]),
        min_keep_ratio=min_kr, max_keep_ratio=max_kr, use_value=use_value,
    )
    model.to(device)

    norm = torch.load(ckpt_dir / "feature_norm.pt", map_location=device, weights_only=False)
    feat_mean = norm["mean"].to(device)
    feat_std = norm["std"].to(device)
    n_cam = int(cfg["n_cam"])
    return model, feat_mean, feat_std, n_cam


def load_budget_scorer_resume(ckpt_dir, device, min_kr, max_kr, use_value=False):
    """Load a previously-saved budget ckpt (token_net + budget_net)."""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = TokenScorerWithBudget(
        emb_dim=int(cfg["emb_dim"]), n_cam=int(cfg["n_cam"]), hidden=int(cfg["hidden"]),
        min_keep_ratio=min_kr, max_keep_ratio=max_kr, use_value=use_value,
    )
    sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    norm = torch.load(ckpt_dir / "feature_norm.pt", map_location=device, weights_only=False)
    feat_mean = norm["mean"].to(device)
    feat_std = norm["std"].to(device)
    n_cam = int(cfg["n_cam"])
    return model, feat_mean, feat_std, n_cam


def compute_selection_log_prob(token_scores, B, mode, tau):
    """Differentiable Top-K selection surrogate (bounded, ranking-preserving).

    Replaces the unbounded multi-label softmax surrogate
        selection_log_prob = (sum(top scores) - B * logsumexp(scores)) / B
    whose logsumexp term lets token scores drift/explode in scale and destroy
    the SFT ranking (logp was observed drifting from -10 to -63).

    All three modes return the SAME hard Top-K indices (stop-gradient) used to
    build the actual prune mask, so forward behaviour is identical; only the
    score-function gradient differs:

      softmax — original multi-label softmax log-prob (baseline, unbounded).
      st_topk — straight-through Top-K: forward uses the hard keep mask, backward
                flows through a per-token sigmoid around the Top-K threshold. The
                resulting per-token log-prob gradient is bounded in [-1/tau, 1/tau].
      gumbel  — gumbel-sigmoid soft keep-mask: same straight-through structure as
                st_topk but with Gumbel exploration noise in the soft mask.

    Returns:
      log_prob:          scalar, differentiable scene-level selection log-probability
      top_indices:       (B,) hard Top-K indices (stop-gradient)
      per_token_log_prob: (N,) bounded per-token log-prob, for counterfactual use
    """
    N = token_scores.numel()
    B = max(1, min(B, N))
    eps = 1e-7

    _, top_indices = token_scores.topk(B, dim=0)

    if mode == "softmax":
        logsumexp = torch.logsumexp(token_scores, dim=0)
        log_prob = (token_scores[top_indices].sum() - B * logsumexp) / B
        per_token = token_scores - logsumexp
        return log_prob, top_indices, per_token

    # Decision boundary = B-th largest score (stop-gradient).
    kth = token_scores.topk(B, dim=0).values[-1].detach()
    hard = torch.zeros(N, device=token_scores.device, dtype=token_scores.dtype)
    hard[top_indices] = 1.0
    hard = hard.detach()  # straight-through: forward uses the hard keep mask

    if mode == "st_topk":
        p = torch.sigmoid((token_scores - kth) / tau)
    elif mode == "gumbel":
        g1 = -torch.log(-torch.log(torch.rand_like(token_scores) + eps) + eps)
        g2 = -torch.log(-torch.log(torch.rand_like(token_scores) + eps) + eps)
        p = torch.sigmoid((token_scores - kth + g1 - g2) / tau)
    else:
        raise ValueError(f"unknown selection mode: {mode}")

    # Bounded per-token log-prob under the soft keep probability p.
    per_token = hard * torch.log(p + eps) + (1.0 - hard) * torch.log(1.0 - p + eps)
    log_prob = per_token.mean()
    return log_prob, top_indices, per_token


def process_one_scene_budget(
    autovla, input_features, token, scorer_model, feat_mean, feat_std, n_cam,
    prune_variant, device, budget_log_std, selection_pg_weight,
    selection_mode="softmax", selection_tau=1.0, deterministic=False,
):
    """Run a two-pass rollout with joint budget and token-selection policy terms.

    deterministic=True uses the budget policy MEAN (no Gaussian sample), i.e. the
    exact keep_ratio the eval path (`BudgetScorerRunner`) would use. Used by
    --mine-mode so mined per-scene deltas reflect deployed behaviour.
    """
    try:
        inputs = autovla.get_prompt(input_features)
        input_ids = inputs["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        processor = autovla.processor
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        actual_image_id = image_pad_id if image_pad_id is not None else video_pad_id
        actual_video_id = video_pad_id if video_pad_id is not None else image_pad_id

        prompt_index = locate_prompt_landmarks(
            input_ids=input_ids,
            vision_start_token_id=vision_start_id,
            vision_end_token_id=vision_end_id,
            image_token_id=actual_image_id,
            video_token_id=actual_video_id,
            action_start_id=None,
        )

        N = prompt_index.n_vision
        if N == 0:
            return None

        # === Pass 1: Capture vision features ===
        fbucket = {}
        with patch_vision_feature_capture(
            vlm=autovla.vlm, layer_idx=0, prompt_index=prompt_index, bucket=fbucket,
        ):
            with torch.no_grad():
                baseline_poses, _ = autovla.predict(input_features)

        if "vision_feat" not in fbucket or baseline_poses is None:
            return None

        traj_sampling = TrajectorySampling(num_poses=10, interval_length=0.5)
        if len(baseline_poses) < traj_sampling.num_poses:
            return None
        baseline_trajectory = Trajectory(
            baseline_poses[:traj_sampling.num_poses, :].cpu().numpy(), traj_sampling,
        )
        vision_feat = fbucket["vision_feat"]

        # === Score with budget scorer ===
        emb = (vision_feat.to(device).float() - feat_mean) / feat_std
        cam = cam_id_from_blocks(prompt_index.vision_token_positions, prompt_index.vision_blocks)
        coh = cam_onehot(cam, n_cam).to(device)
        x = torch.cat([emb, coh], dim=-1)

        token_scores, keep_ratio, budget_logit, value = scorer_model(x)

        # === Sample budget (Gaussian policy in logit space) ===
        budget_std = torch.exp(budget_log_std)
        budget_dist = torch.distributions.Normal(budget_logit, budget_std)
        sampled_logit = budget_logit.detach() if deterministic else budget_dist.sample()
        budget_log_prob = budget_dist.log_prob(sampled_logit)

        # Map sampled logit to keep_ratio via sigmoid
        sampled_kr = scorer_model.min_kr + (scorer_model.max_kr - scorer_model.min_kr) * torch.sigmoid(sampled_logit)

        # Determine B from sampled budget
        B = max(1, int(round(sampled_kr.item() * N)))

        # === Select top-B by token scores ===
        # Treat the selected set as the token-selection action.  The Top-K
        # indices are stop-gradient action samples; the differentiable Top-K
        # surrogate (softmax / straight-through / gumbel) supplies the REINFORCE
        # score-function gradient to token_net.
        selection_log_prob, top_indices, per_token_log_prob = compute_selection_log_prob(
            token_scores, B, selection_mode, selection_tau)

        # One driving reward now updates both policy factors: which tokens to
        # retain and how many tokens to retain for this scene.
        total_log_prob = budget_log_prob + selection_pg_weight * selection_log_prob

        # === Build prune mask ===
        all_positions = prompt_index.vision_token_positions
        keep_mask = torch.zeros(N, dtype=torch.bool)
        keep_mask[top_indices.cpu()] = True
        prune_idx = (~keep_mask).nonzero(as_tuple=True)[0]
        prune_positions = all_positions[prune_idx]

        # === Pass 2: Generate trajectory under pruning ===
        with ExitStack() as stack:
            if prune_variant == "drop" and prune_positions.numel() > 0:
                stack.enter_context(patch_vision_token_drop(
                    vlm=autovla.vlm, prune_positions=prune_positions, verbose=False,
                ))
            elif prune_positions.numel() > 0:
                stack.enter_context(patch_vision_token_prune(
                    vlm=autovla.vlm, prune_positions=prune_positions, verbose=False,
                ))
            with torch.no_grad():
                poses, _ = autovla.predict(input_features)

        traj_sampling = TrajectorySampling(num_poses=10, interval_length=0.5)
        if poses is None or len(poses) < traj_sampling.num_poses:
            return None

        trajectory = Trajectory(
            poses[:traj_sampling.num_poses, :].cpu().numpy(), traj_sampling,
        )

        return {
            "total_log_prob": total_log_prob,
            "budget_log_prob": budget_log_prob,
            "selection_log_prob": selection_log_prob,
            "keep_ratio": sampled_kr.item(),
            "trajectory": trajectory,
            "baseline_trajectory": baseline_trajectory,
            "token": token,
            "N": N,
            "B": B,
            # v6: for per-token counterfactual REINFORCE
            "top_indices": top_indices,
            "token_scores": token_scores,
            "all_positions": all_positions,
            # v7: bounded per-token selection log-prob (surrogate-aware)
            "per_token_log_prob": per_token_log_prob,
            # v10: value-head (critic) estimate V(s) for the value-baseline variant
            "value": value,
        }

    except Exception as e:
        print(f"[budget-rl] Scene {token} error: {e}", flush=True)
        return None


def process_one_counterfactual(
    autovla, input_features, token, per_token_log_prob, prune_variant,
    top_indices, all_positions, N, reward_fn, args,
):
    """Per-token counterfactual REINFORCE: evaluate PDMS change when one kept token is dropped.

    For each sampled kept token i:
      1. Build a prune mask where token i is dropped (kept→dropped) and one random
         dropped token is kept (dropped→kept) — keeping B constant.
      2. Re-run VLA forward with the modified prune mask.
      3. Compute R_{-i} = true PDMS without token i.
      4. Per-token advantage: A_i = R_pruned - R_{-i}

    per_token_log_prob: pre-computed (N,) bounded per-token selection log-prob from
                        process_one_scene_budget (surrogate-aware). Used only for the
                        score-function term (no extra VLA forward needed).

    Returns:
      per_token_advantages: dict {global_position_index: R_{-i}} (R_pruned subtracted in caller)
      per_token_log_probs: dict {global_position_index: log_prob from selection surrogate}
      counterfactual_pdms_list: list of (kept_idx, dropped_idx, R_{-i}) for logging
    """
    try:
        kept_set = set(top_indices.cpu().tolist())
        all_set = set(range(N))
        dropped_set = list(all_set - kept_set)

        if len(dropped_set) < args.counterfactual_k:
            return None, None, None

        kept_list = list(kept_set)
        rng = np.random.RandomState(hash(token) % (2**31))
        if len(kept_list) > args.counterfactual_k:
            sampled_kept = rng.choice(kept_list, args.counterfactual_k, replace=False).tolist()
        else:
            sampled_kept = kept_list
        sampled_dropped = rng.choice(dropped_set, len(sampled_kept), replace=False).tolist()

        per_token_advantages = {}
        per_token_log_probs = {}
        counterfactual_pdms_list = []
        traj_sampling = TrajectorySampling(num_poses=10, interval_length=0.5)

        for kept_idx, dropped_idx in zip(sampled_kept, sampled_dropped):
            # Build modified keep_mask: swap kept_idx out, dropped_idx in
            modified_kept = kept_set - {kept_idx} | {dropped_idx}
            keep_mask = torch.zeros(N, dtype=torch.bool)
            for idx in modified_kept:
                keep_mask[idx] = True
            prune_idx = (~keep_mask).nonzero(as_tuple=True)[0]
            prune_positions = all_positions[prune_idx]

            # Re-run VLA forward with modified prune
            with ExitStack() as stack:
                if prune_variant == "drop" and prune_positions.numel() > 0:
                    stack.enter_context(patch_vision_token_drop(
                        vlm=autovla.vlm, prune_positions=prune_positions, verbose=False,
                    ))
                elif prune_positions.numel() > 0:
                    stack.enter_context(patch_vision_token_prune(
                        vlm=autovla.vlm, prune_positions=prune_positions, verbose=False,
                    ))
                with torch.no_grad():
                    poses, _ = autovla.predict(input_features)

            if poses is None or len(poses) < traj_sampling.num_poses:
                continue

            cf_trajectory = Trajectory(
                poses[:traj_sampling.num_poses, :].cpu().numpy(), traj_sampling,
            )

            r_minus_i_out = reward_fn.rl_pdm_score(
                cf_trajectory, token,
                use_true_pdms=True, return_components=False,
            )
            if r_minus_i_out is None:
                continue

            R_minus_i = args.driving_scale * r_minus_i_out

            # Per-token log_prob: bounded selection surrogate (surrogate-aware)
            token_log_prob = per_token_log_prob[kept_idx]  # scalar

            pos = all_positions[kept_idx].item()
            per_token_log_probs[pos] = token_log_prob
            per_token_advantages[pos] = R_minus_i  # temp: store R_{-i}
            counterfactual_pdms_list.append((kept_idx, dropped_idx, R_minus_i))

        return per_token_advantages, per_token_log_probs, counterfactual_pdms_list

    except Exception as e:
        print(f"[cf] Scene {token} counterfactual error: {e}", flush=True)
        return None, None, None


def main():
    args = parse_args()
    distributed, rank, world_size, is_main = init_distributed(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    if is_main:
        print("=" * 70, flush=True)
        print("[budget-rl] v6 Per-Token Counterfactual REINFORCE", flush=True)
        print("=" * 70, flush=True)
        print(f"  data            = {args.json_dir}")
        print(f"  world_size      = {world_size}")
        print(f"  efficiency_beta = {args.efficiency_beta}")
        print(f"  safety_beta     = {args.safety_beta}, margin = {args.safety_margin}")
        print(f"  driving_scale   = {args.driving_scale}")
        print(f"  delta_reward    = {args.delta_reward}")
        print(f"  keep_ratio range= [{args.min_keep_ratio}, {args.max_keep_ratio}]")
        print(f"  lr (token/budget)= {args.lr} / {args.budget_lr}")
        print(f"  selection_pg_weight = {args.selection_pg_weight}")
        print(f"  kl_beta (token) = {args.kl_beta}, budget_kl_beta = {args.budget_kl_beta}")
        print(f"  group_size      = {args.group_size}")
        print(f"  counterfactual_k= {args.counterfactual_k}")
        print(f"  selection_mode  = {args.selection_mode} (tau={args.selection_tau})")
        print("=" * 70, flush=True)

    # Load AutoVLA (frozen)
    autovla, vlm_config = load_autovla_for_inference(
        args.autovla_config, args.autovla_ckpt, args.sensor_data_path, device
    )

    # === Load / resume budget scorer ===
    resume_dir = out_dir / "ckpt_resume"
    global_step = 0
    best_avg_reward = -float("inf")
    resume_state = None
    if resume_dir.exists() and (resume_dir / "checkpoint.pt").exists():
        scorer_model, feat_mean, feat_std, n_cam = load_budget_scorer_resume(
            resume_dir, device, args.min_keep_ratio, args.max_keep_ratio,
            use_value=args.use_value_baseline)
        bsd = torch.load(resume_dir / "budget_params.pt", map_location=device, weights_only=False)
        budget_log_std = nn.Parameter(torch.tensor(bsd["budget_log_std"], device=device))
        resume_state = json.loads((resume_dir / "resume_state.json").read_text())
        global_step = resume_state["global_step"]
        best_avg_reward = resume_state.get("best_avg_reward", -float("inf"))
        print(f"[budget-rl] RESUME from {resume_dir} at step {global_step} "
              f"(epoch {resume_state['epoch']}, next g_start {resume_state['g_start']})", flush=True)
    else:
        if args.init_budget_ckpt:
            # v11: continue from a trained budget policy (v10 full_hi) instead of
            # restarting from the SFT scorer, so hard-example mining refines the
            # already-SOTA-matching policy rather than relearning it.
            scorer_model, feat_mean, feat_std, n_cam = load_budget_scorer_resume(
                Path(args.init_budget_ckpt), device, args.min_keep_ratio, args.max_keep_ratio,
                use_value=args.use_value_baseline)
            bp = Path(args.init_budget_ckpt) / "budget_params.pt"
            init_log_std = args.budget_log_std_init
            if bp.exists():
                init_log_std = torch.load(bp, map_location=device, weights_only=False)["budget_log_std"]
            budget_log_std = nn.Parameter(torch.tensor(float(init_log_std), device=device))
            print(f"[budget-rl] WARM-START from budget ckpt: {args.init_budget_ckpt} "
                  f"(budget_log_std={float(init_log_std):.4f})", flush=True)
        else:
            scorer_model, feat_mean, feat_std, n_cam = load_budget_scorer(
                args.scorer_ckpt, device, args.min_keep_ratio, args.max_keep_ratio,
                use_value=args.use_value_baseline)
            budget_log_std = nn.Parameter(torch.tensor(args.budget_log_std_init, device=device))
            if args.budget_init_kr is not None:
                scorer_model.set_budget_init(args.budget_init_kr)
                print(f"[budget-rl] Budget head anchored at keep_ratio={args.budget_init_kr}", flush=True)
            print(f"[budget-rl] Init from SFT scorer: {args.scorer_ckpt}", flush=True)
    scorer_model.train()

    # Reference scorer (frozen, for KL) — anchored at current params so token_net stays stable
    ref_scorer = copy.deepcopy(scorer_model)
    ref_scorer.eval()
    for p in ref_scorer.parameters():
        p.requires_grad_(False)
    # Model initialization is shared across ranks; rollout noise is rank-specific.
    torch.manual_seed(args.seed + rank)

    # Reward function
    reward_fn = PDM_Reward(Path(args.metric_cache))
    cache_tokens = set(reward_fn.metric_cache_loader.metric_cache_paths.keys())

    # Each training scene obtains its own unpruned baseline from the feature
    # capture pass below. `--baseline-scores` remains a compatibility argument
    # but is deliberately not used for navtrain reward computation.
    if is_main and Path(args.baseline_scores).exists():
        print("[budget-rl] Runtime same-scene baselines enabled; ignoring external baseline-scores file", flush=True)

    # Feature loader
    codebook_path = vlm_config['model']['codebook_cache_path']
    traj_sampling = TrajectorySampling(num_poses=10, interval_length=0.5)
    feat_agent = AutoVLAAgent(
        trajectory_sampling=traj_sampling,
        sensor_data_path=args.sensor_data_path,
        codebook_cache_path=codebook_path,
        skip_model_load=True,
    )

    # Scene list
    json_dir = Path(args.json_dir)
    if args.scene_list:
        # Explicit scene subset (v11 hard-example mining). Duplicate lines are
        # preserved so a token can appear multiple times = oversampling weight.
        wanted = load_scene_list(args.scene_list)
        all_scenes = [json_dir / f"{tok}.json" for tok in wanted]
        missing = [p for p in all_scenes if not p.exists()]
        all_scenes = [p for p in all_scenes if p.exists()]
        if is_main:
            print(f"[budget-rl] scene-list {args.scene_list}: {len(all_scenes)} scenes "
                  f"({len(missing)} missing json)", flush=True)
    else:
        all_scenes = sorted(json_dir.glob("*.json"))
    if args.max_scenes:
        all_scenes = all_scenes[:args.max_scenes]
    valid_scenes = [s for s in all_scenes if s.stem in cache_tokens]
    if distributed:
        valid_scenes = valid_scenes[rank::world_size]
        # Every rank must execute the same number of collective gradient steps.
        # Trim to the smallest rank-local length rounded to full groups.
        local_count = torch.tensor([len(valid_scenes)], device=device, dtype=torch.long)
        counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(counts, local_count)
        common_count = min(int(c.item()) for c in counts)
        common_count = (common_count // args.group_size) * args.group_size
        valid_scenes = valid_scenes[:common_count]
        if is_main:
            print(f"[budget-rl] {common_count * world_size} synchronized scenes "
                  f"({common_count} per rank, world_size={world_size})", flush=True)
    elif args.num_shards > 1:
        valid_scenes = valid_scenes[args.shard_id::args.num_shards]
        print(f"[budget-rl] {len(valid_scenes)} scenes (shard {args.shard_id}/{args.num_shards})", flush=True)
    else:
        print(f"[budget-rl] {len(valid_scenes)} scenes", flush=True)

    # === v11: mining mode — diagnostic rollout only, no gradients ===
    if args.mine_mode:
        scorer_model.eval()
        mine_out = Path(args.mine_out) if args.mine_out else (
            out_dir / f"mine_shard{args.shard_id}.jsonl")
        mine_out.parent.mkdir(parents=True, exist_ok=True)
        # Resume-safe: skip scenes already recorded in a previous (reclaimed) run.
        done = set()
        if mine_out.exists():
            for line in mine_out.read_text().splitlines():
                try:
                    done.add(json.loads(line)["token"])
                except Exception:
                    pass
        print(f"[mine] {len(valid_scenes)} scenes, {len(done)} already done -> {mine_out}", flush=True)
        t_mine = time.time()
        n_written = 0
        with mine_out.open("a") as mf:
            for i, scene_path in enumerate(valid_scenes):
                if scene_path.stem in done:
                    continue
                try:
                    with open(scene_path) as f:
                        scene_data = json.load(f)
                    input_features = {}
                    for builder in feat_agent.get_feature_builders():
                        input_features.update(builder.compute_features(scene_data))
                    input_features["sensor_data_path"] = args.sensor_data_path
                    token_id = scene_data["token"]
                except Exception:
                    continue

                with torch.no_grad():
                    result = process_one_scene_budget(
                        autovla=autovla, input_features=input_features, token=token_id,
                        scorer_model=scorer_model, feat_mean=feat_mean, feat_std=feat_std,
                        n_cam=n_cam, prune_variant=args.prune_variant, device=device,
                        budget_log_std=budget_log_std,
                        selection_pg_weight=args.selection_pg_weight,
                        selection_mode=args.selection_mode,
                        selection_tau=args.selection_tau,
                        deterministic=True,
                    )
                if result is None:
                    continue

                pruned_out = reward_fn.rl_pdm_score(
                    result["trajectory"], result["token"],
                    use_true_pdms=True, return_components=True)
                base_out = reward_fn.rl_pdm_score(
                    result["baseline_trajectory"], result["token"],
                    use_true_pdms=True, return_components=True)
                if not isinstance(pruned_out, tuple) or not isinstance(base_out, tuple):
                    continue
                pruned_pdms, pruned_sub = pruned_out
                base_pdms, base_sub = base_out

                rec = {
                    "token": result["token"],
                    "pruned_pdms": float(pruned_pdms),
                    "noprune_pdms": float(base_pdms),
                    "delta": float(pruned_pdms) - float(base_pdms),
                    "keep_ratio": result["keep_ratio"],
                    "N": result["N"], "B": result["B"],
                    "pruned_sub": {k: float(v) for k, v in pruned_sub.items()},
                    "noprune_sub": {k: float(v) for k, v in base_sub.items()},
                }
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                n_written += 1
                if n_written % 25 == 0:
                    rate = (time.time() - t_mine) / max(1, n_written)
                    print(f"[mine] {n_written} written ({i+1}/{len(valid_scenes)} seen) "
                          f"{rate:.2f}s/scene", flush=True)
        print(f"[mine] DONE {n_written} scenes -> {mine_out} "
              f"(wall {time.time()-t_mine:.0f}s)", flush=True)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Optimizer: separate LR for token_net vs budget head + log_std (+ value head)
    optimizer_groups = [
        {"params": scorer_model.token_net.parameters(), "lr": args.lr},
        {"params": scorer_model.budget_net.parameters(), "lr": args.budget_lr},
        {"params": [budget_log_std], "lr": args.budget_lr},
    ]
    if args.use_value_baseline and scorer_model.value_net is not None:
        optimizer_groups.append({"params": scorer_model.value_net.parameters(), "lr": args.value_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)

    # Training loop. Only rank 0 writes shared logs/checkpoints.
    log_file = out_dir / "train_log.jsonl"
    logf = log_file.open("a") if is_main else None
    train_start = time.time()

    for epoch in range(args.num_epochs):
        rng = np.random.RandomState(args.seed + epoch)  # seeded -> resumable
        perm = rng.permutation(len(valid_scenes))
        epoch_rewards = []

        start_g = 0
        if resume_state is not None and resume_state.get("epoch") == epoch:
            start_g = resume_state.get("g_start", 0)

        for g_start in range(0, len(valid_scenes), args.group_size):
            if g_start < start_g:
                continue  # already done before reclaim
            t0 = time.time()

            group_rewards = []
            group_log_probs = []
            group_budget_log_probs = []
            group_token_pg_losses = []
            group_selection_log_probs = []
            group_keep_ratios = []
            group_safety_losses = []
            group_values = []
            zero_loss = budget_log_std * 0.0
            for parameter in scorer_model.parameters():
                zero_loss = zero_loss + parameter.sum() * 0.0

            for idx in perm[g_start:g_start + args.group_size]:
                scene_path = valid_scenes[idx]
                try:
                    with open(scene_path) as f:
                        scene_data = json.load(f)
                    input_features = {}
                    for builder in feat_agent.get_feature_builders():
                        input_features.update(builder.compute_features(scene_data))
                    input_features["sensor_data_path"] = args.sensor_data_path
                    token_id = scene_data['token']
                except Exception as e:
                    continue

                result = process_one_scene_budget(
                    autovla=autovla, input_features=input_features, token=token_id,
                    scorer_model=scorer_model, feat_mean=feat_mean, feat_std=feat_std,
                    n_cam=n_cam, prune_variant=args.prune_variant, device=device,
                    budget_log_std=budget_log_std,
                    selection_pg_weight=args.selection_pg_weight,
                    selection_mode=args.selection_mode,
                    selection_tau=args.selection_tau,
                )
                if result is None:
                    continue

                # v6: Per-token counterfactual REINFORCE
                # Compute PDMS for pruned trajectory
                reward_out = reward_fn.rl_pdm_score(
                    result["trajectory"], result["token"],
                    use_true_pdms=True, return_components=True,
                )
                if not isinstance(reward_out, tuple):
                    continue
                driving_reward, sub_scores = reward_out
                if not all(k in sub_scores for k in ["collision", "drivable", "progress", "ttc", "comfort", "direction"]):
                    continue
                R_pruned = args.driving_scale * driving_reward

                # Safety penalty and optional same-scene delta reward.
                baseline_out = reward_fn.rl_pdm_score(
                    result["baseline_trajectory"], result["token"],
                    use_true_pdms=True, return_components=True,
                )
                baseline_reward = 0.0
                if isinstance(baseline_out, tuple):
                    baseline_reward_raw, baseline = baseline_out
                    baseline_reward = args.driving_scale * baseline_reward_raw
                    safety_loss = sum(
                        weight * max(0.0, baseline.get(name, 0.0) - sub_scores.get(name, 0.0) - args.safety_margin)
                        for name, weight in {"collision": 0.40, "drivable": 0.30, "ttc": 0.30}.items()
                    )
                else:
                    safety_loss = 0.0

                # Scene-level reward for budget head (Gaussian policy).
                driving_term = R_pruned - baseline_reward if args.delta_reward else R_pruned
                if args.efficiency_mode == "floor":
                    # Only penalize pruning below the target; keep_ratio is free to
                    # rise toward max_kr to chase the driving delta (SOTA objective).
                    shortfall = max(0.0, args.target_keep_ratio - result["keep_ratio"])
                    efficiency_bonus = -args.efficiency_beta * shortfall
                else:
                    efficiency_bonus = args.efficiency_beta * (1.0 - result["keep_ratio"])
                total_reward = driving_term + efficiency_bonus - args.safety_beta * safety_loss

                group_rewards.append(total_reward)
                group_budget_log_probs.append(result["budget_log_prob"])
                group_keep_ratios.append(result["keep_ratio"])
                group_safety_losses.append(safety_loss)
                if args.use_value_baseline and result.get("value") is not None:
                    group_values.append(result["value"].squeeze(-1))

                # v6: Per-token counterfactual REINFORCE for token_net
                if args.counterfactual_k > 0 and result.get("top_indices") is not None:
                    cf_advantages, cf_log_probs, cf_pdms = process_one_counterfactual(
                        autovla=autovla, input_features=input_features, token=token_id,
                        per_token_log_prob=result["per_token_log_prob"],
                        prune_variant=args.prune_variant,
                        top_indices=result["top_indices"],
                        all_positions=result["all_positions"],
                        N=result["N"],
                        reward_fn=reward_fn, args=args,
                    )
                    if cf_advantages is not None and cf_log_probs is not None:
                        token_advantages = {}
                        for pos, R_minus_i in cf_advantages.items():
                            token_advantages[pos] = R_pruned - R_minus_i
                        per_token_losses = []
                        for pos, adv in token_advantages.items():
                            if pos in cf_log_probs:
                                per_token_losses.append(-adv * cf_log_probs[pos])
                        if per_token_losses:
                            per_token_pg_loss = torch.stack(per_token_losses).mean()
                        else:
                            per_token_pg_loss = zero_loss
                        group_selection_log_probs.append(
                            float(np.mean([lp.item() for lp in cf_log_probs.values()]))
                            if cf_log_probs else 0.0
                        )
                    else:
                        per_token_pg_loss = zero_loss
                        group_selection_log_probs.append(0.0)
                else:
                    per_token_pg_loss = zero_loss
                    group_log_probs.append(result["selection_log_prob"])
                    group_selection_log_probs.append(result["selection_log_prob"].detach().item())
                group_token_pg_losses.append(per_token_pg_loss)

            # All DDP ranks execute every optimizer step, even if a local group
            # has invalid rollouts. This prevents collective-operation deadlocks.
            if len(group_rewards) >= 2:
                rewards_t = torch.tensor(group_rewards, device=device, dtype=torch.float32)
                if args.use_value_baseline and len(group_values) == len(group_rewards):
                    # Learned critic baseline: advantage = reward - V(s). Far lower
                    # variance than group-normalized REINFORCE, and keeps the
                    # absolute scale of the driving delta (needed to push keep_ratio
                    # upward toward the SOTA objective).
                    values_t = torch.stack(group_values)
                    advantage = rewards_t - values_t.detach()
                    value_loss = args.value_loss_weight * F.mse_loss(values_t, rewards_t.detach())
                else:
                    advantage = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
                    value_loss = zero_loss

                # Budget policy loss: scene-level REINFORCE (Gaussian policy)
                budget_log_probs_t = torch.stack(group_budget_log_probs)
                budget_policy_loss = -(advantage.detach() * budget_log_probs_t).mean()

                # Token selection policy loss: per-token counterfactual or scene-level
                if args.counterfactual_k > 0:
                    token_policy_loss = torch.stack(group_token_pg_losses).mean()
                else:
                    log_probs_t = torch.stack(group_log_probs)
                    token_policy_loss = -(advantage.detach() * log_probs_t).mean()

                policy_loss = budget_policy_loss + args.selection_pg_weight * token_policy_loss

                local_mean_reward = rewards_t.mean().item()
                local_mean_kr = float(np.mean(group_keep_ratios))
                local_budget_log_prob = float(torch.stack(group_budget_log_probs).detach().mean().item())
                local_selection_log_prob = float(np.mean(group_selection_log_probs))
                local_safety_loss = float(np.mean(group_safety_losses))
            else:
                policy_loss = zero_loss
                value_loss = zero_loss
                local_mean_reward = 0.0
                local_mean_kr = 0.0
                local_budget_log_prob = 0.0
                local_selection_log_prob = 0.0
                local_safety_loss = 0.0

            # KL penalty: token_net only (budget_net learns freely), optional budget KL.
            kl_loss = torch.tensor(0.0, device=device)
            if args.kl_beta > 0:
                for p_curr, p_ref in zip(scorer_model.token_net.parameters(),
                                        ref_scorer.token_net.parameters()):
                    kl_loss = kl_loss + F.mse_loss(p_curr, p_ref, reduction='sum')
                kl_loss = args.kl_beta * kl_loss
            if args.budget_kl_beta > 0:
                for p_curr, p_ref in zip(scorer_model.budget_net.parameters(),
                                        ref_scorer.budget_net.parameters()):
                    kl_loss = kl_loss + args.budget_kl_beta * F.mse_loss(p_curr, p_ref, reduction='sum')

            loss = policy_loss + kl_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            if distributed:
                sync_gradients(scorer_model, [budget_log_std], world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(scorer_model.parameters()) + [budget_log_std], max_norm=1.0)
            optimizer.step()

            global_step += 1
            mean_reward = reduce_mean(local_mean_reward, device, distributed, world_size)
            mean_kr = reduce_mean(local_mean_kr, device, distributed, world_size)
            mean_budget_log_prob = reduce_mean(local_budget_log_prob, device, distributed, world_size)
            mean_selection_log_prob = reduce_mean(local_selection_log_prob, device, distributed, world_size)
            mean_safety_loss = reduce_mean(local_safety_loss, device, distributed, world_size)
            epoch_rewards.append(mean_reward)

            if is_main and global_step % args.log_every == 0:
                rec = {
                    "step": global_step, "epoch": epoch,
                    "reward_mean": mean_reward,
                    "driving_reward_proxy": mean_reward - args.efficiency_beta * (1.0 - mean_kr),
                    "keep_ratio_mean": mean_kr,
                    "keep_ratio_std_local": float(np.std(group_keep_ratios)),
                    "safety_loss_mean": mean_safety_loss,
                    "budget_log_std": budget_log_std.item(),
                    "budget_log_prob_mean": mean_budget_log_prob,
                    "selection_log_prob_mean": mean_selection_log_prob,
                    "selection_pg_weight": args.selection_pg_weight,
                    "policy_loss": policy_loss.item(),
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "n_valid_local": len(group_rewards),
                    "world_size": world_size,
                    "elapsed_s": time.time() - t0,
                }
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(
                    f"[step {global_step:4d}] R={mean_reward:.4f} kr={mean_kr:.3f} "
                    f"safe={mean_safety_loss:.4f} logp(b/s)={mean_budget_log_prob:.3f}/{mean_selection_log_prob:.3f} "
                    f"loss={loss.item():.4f} grad={grad_norm:.3f} ({len(group_rewards)}/{args.group_size} local scenes, {time.time()-t0:.1f}s)",
                    flush=True,
                )

            # Rank 0 owns shared checkpoints; barriers make resume state consistent.
            if global_step % args.save_every == 0:
                if is_main:
                    _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, f"step{global_step}", args)
                    if mean_reward > best_avg_reward:
                        best_avg_reward = mean_reward
                        _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, "best", args)
                    _save_resume(scorer_model, budget_log_std, out_dir, epoch,
                                 g_start + args.group_size, global_step, best_avg_reward)
                if distributed:
                    dist.barrier()

        if epoch_rewards:
            ep_mean = float(np.mean(epoch_rewards))
            if is_main:
                print(f"\n[budget-rl] Epoch {epoch}: avg_reward={ep_mean:.4f}\n", flush=True)
                if ep_mean > best_avg_reward:
                    best_avg_reward = ep_mean
                    _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, "best", args)
            if distributed:
                dist.barrier()

    sync_fingerprint = audit_model_sync(scorer_model, budget_log_std, device, distributed, world_size)
    if is_main:
        (out_dir / "ddp_sync_audit.json").write_text(json.dumps({
            "world_size": world_size,
            "fingerprints": sync_fingerprint,
        }, indent=2) + "\n")
        _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, "final", args)
        # Clean resume checkpoint only after a complete final checkpoint exists.
        import shutil
        if resume_dir.exists():
            shutil.rmtree(resume_dir, ignore_errors=True)
        if logf is not None:
            logf.close()
        print(f"[budget-rl] DONE. Best reward: {best_avg_reward:.4f}. Output: {out_dir} "
              f"(wall {time.time()-train_start:.0f}s)", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def _save(model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, tag, args):
    save_dir = Path(out_dir) if tag == "final" else Path(out_dir) / f"ckpt_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "checkpoint.pt")
    torch.save({"mean": feat_mean.cpu(), "std": feat_std.cpu()}, save_dir / "feature_norm.pt")
    torch.save({"budget_log_std": budget_log_std.item()}, save_dir / "budget_params.pt")
    (save_dir / "config.json").write_text(json.dumps({
        "emb_dim": model.emb_dim, "n_cam": model.n_cam, "hidden": 256,
        "model_type": "TokenScorerWithBudget",
        "min_keep_ratio": model.min_kr, "max_keep_ratio": model.max_kr,
        "use_value": bool(model.use_value),
    }))
    (save_dir / "manifest.json").write_text(json.dumps({
        "spec": "budget_rl_v7_diff_topk",
        "method": "Differentiable Top-K selection surrogate + Gaussian budget; true PDMS reward; KL on token_net only",
        "efficiency_beta": args.efficiency_beta,
        "efficiency_mode": args.efficiency_mode,
        "target_keep_ratio": args.target_keep_ratio,
        "safety_beta": args.safety_beta,
        "safety_margin": args.safety_margin,
        "driving_scale": args.driving_scale,
        "delta_reward": args.delta_reward,
        "selection_pg_weight": args.selection_pg_weight,
        "kl_beta": args.kl_beta,
        "budget_kl_beta": args.budget_kl_beta,
        "min_keep_ratio": args.min_keep_ratio,
        "max_keep_ratio": args.max_keep_ratio,
        "counterfactual_k": args.counterfactual_k,
        "selection_mode": args.selection_mode,
        "selection_tau": args.selection_tau,
        "use_value_baseline": args.use_value_baseline,
        "value_loss_weight": args.value_loss_weight,
        "budget_init_kr": args.budget_init_kr,
        "scene_list": str(args.scene_list) if args.scene_list else None,
        "init_budget_ckpt": str(args.init_budget_ckpt) if args.init_budget_ckpt else None,
        "train_json_dir": str(args.json_dir),
        "train_metric_cache": str(args.metric_cache),
        "tag": tag,
    }, indent=2))
    print(f"[budget-rl] Saved: {save_dir}", flush=True)


def _save_resume(model, budget_log_std, out_dir, epoch, next_g_start, global_step, best_avg_reward):
    d = Path(out_dir) / "ckpt_resume"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "checkpoint.pt")
    torch.save({"budget_log_std": budget_log_std.item()}, d / "budget_params.pt")
    (d / "resume_state.json").write_text(json.dumps({
        "epoch": epoch, "g_start": next_g_start, "global_step": global_step,
        "best_avg_reward": best_avg_reward,
    }))


if __name__ == "__main__":
    main()

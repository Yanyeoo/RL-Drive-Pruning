"""train_scorer_budget_rl.py — RL training for scorer with learned budget.

The scorer simultaneously learns:
  1. Which tokens to keep (per-token importance)
  2. How many tokens to keep (scene-level budget)

Reward = α * shaped_driving_reward + β * efficiency_bonus
  - shaped_driving_reward: same as train_scorer_grpo.py (sub-metric weighted delta)
  - efficiency_bonus: (1 - keep_ratio) — encourages pruning more

The budget head outputs a continuous keep_ratio per scene, and we use
REINFORCE with a Gaussian policy for the budget action.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scorer_budget_rl.py \
    --scorer-ckpt ckpt/s3_token_scorer \
    --out-dir ckpt/s3_token_scorer_budget_rl \
    --efficiency-beta 0.05 \
    --num-epochs 3 --group-size 8 --lr 3e-5
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
    p.add_argument("--json-dir", type=str, default=str(ROOT / "data/navtest_nocot"))
    p.add_argument("--metric-cache", type=str, default=str(ROOT / "data/navtest_metric_cache"))
    p.add_argument("--sensor-data-path", type=str,
                   default="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/test/openscene-v1.1/sensor_blobs/test")
    p.add_argument("--autovla-config", type=str,
                   default=str(ROOT / "code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"))
    p.add_argument("--autovla-ckpt", type=str,
                   default=str(ROOT / "models/AutoVLA/AutoVLA_PDMS_89.ckpt"))
    # Training
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--efficiency-beta", type=float, default=0.05,
                   help="Weight for efficiency bonus in reward (higher = prune more aggressively)")
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
    return p.parse_args()


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
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    new_sd = {k.replace("autovla.", ""): v for k, v in sd.items()}
    autovla.load_state_dict(new_sd, strict=False)
    autovla.eval()
    for p in autovla.parameters():
        p.requires_grad_(False)
    return autovla, config


def load_budget_scorer(ckpt_dir, device, min_kr, max_kr):
    """Load base SFT scorer and upgrade to budget version."""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())

    # Load base scorer
    base = TokenImportanceScorer(
        emb_dim=int(cfg["emb_dim"]),
        n_cam=int(cfg["n_cam"]),
        hidden=int(cfg["hidden"]),
    )
    sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    base.load_state_dict(sd)

    # Upgrade to budget version (copies token_net weights)
    model = TokenScorerWithBudget.from_pretrained_scorer(
        base, hidden=int(cfg["hidden"]),
        min_keep_ratio=min_kr, max_keep_ratio=max_kr,
    )
    model.to(device)

    # Load feature normalization
    norm = torch.load(ckpt_dir / "feature_norm.pt", map_location=device, weights_only=False)
    feat_mean = norm["mean"].to(device)
    feat_std = norm["std"].to(device)
    n_cam = int(cfg["n_cam"])
    return model, feat_mean, feat_std, n_cam


def process_one_scene_budget(
    autovla, input_features, token, scorer_model, feat_mean, feat_std, n_cam,
    prune_variant, device, budget_log_std,
):
    """Run 2-pass pipeline with LEARNED budget."""
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
                autovla.predict(input_features)

        if "vision_feat" not in fbucket:
            return None

        vision_feat = fbucket["vision_feat"]

        # === Score with budget scorer ===
        emb = (vision_feat.to(device).float() - feat_mean) / feat_std
        cam = cam_id_from_blocks(prompt_index.vision_token_positions, prompt_index.vision_blocks)
        coh = cam_onehot(cam, n_cam).to(device)
        x = torch.cat([emb, coh], dim=-1)

        token_scores, keep_ratio, budget_logit = scorer_model(x)

        # === Sample budget (Gaussian policy in logit space) ===
        budget_std = torch.exp(budget_log_std)
        budget_dist = torch.distributions.Normal(budget_logit, budget_std)
        sampled_logit = budget_dist.sample()
        budget_log_prob = budget_dist.log_prob(sampled_logit)

        # Map sampled logit to keep_ratio via sigmoid
        sampled_kr = scorer_model.min_kr + (scorer_model.max_kr - scorer_model.min_kr) * torch.sigmoid(sampled_logit)

        # Determine B from sampled budget
        B = max(1, int(round(sampled_kr.item() * N)))

        # === Select top-B by token scores (deterministic given scores) ===
        _, top_indices = token_scores.topk(B, dim=0)

        # Only budget action is stochastic; token selection is deterministic topk
        # token_net gradients flow through the budget interaction (B depends on scores indirectly)
        total_log_prob = budget_log_prob

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
            "keep_ratio": sampled_kr.item(),
            "trajectory": trajectory,
            "token": token,
            "N": N,
            "B": B,
        }

    except Exception as e:
        print(f"[budget-rl] Scene {token} error: {e}", flush=True)
        return None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("[budget-rl] Scorer Budget RL — learns WHAT + HOW MANY to prune", flush=True)
    print("=" * 70, flush=True)
    print(f"  efficiency_beta = {args.efficiency_beta}", flush=True)
    print(f"  keep_ratio range = [{args.min_keep_ratio}, {args.max_keep_ratio}]", flush=True)
    print(f"  lr = {args.lr}, kl_beta = {args.kl_beta}", flush=True)
    print("=" * 70, flush=True)

    # Load AutoVLA (frozen)
    autovla, vlm_config = load_autovla_for_inference(
        args.autovla_config, args.autovla_ckpt, args.sensor_data_path, device
    )

    # Load budget scorer (initialized from SFT scorer)
    scorer_model, feat_mean, feat_std, n_cam = load_budget_scorer(
        args.scorer_ckpt, device, args.min_keep_ratio, args.max_keep_ratio
    )
    scorer_model.train()

    # Learnable budget exploration std
    budget_log_std = nn.Parameter(torch.tensor(args.budget_log_std_init, device=device))

    # Reference scorer (frozen, for KL)
    ref_scorer = copy.deepcopy(scorer_model)
    ref_scorer.eval()
    for p in ref_scorer.parameters():
        p.requires_grad_(False)

    # Reward function
    reward_fn = PDM_Reward(Path(args.metric_cache))
    cache_tokens = set(reward_fn.metric_cache_loader.metric_cache_paths.keys())

    # Baseline sub-scores
    baseline_sub_scores = {}
    if args.shaped_reward and Path(args.baseline_scores).exists():
        baseline_sub_scores = json.loads(Path(args.baseline_scores).read_text())
        print(f"[budget-rl] Loaded baseline sub-scores for {len(baseline_sub_scores)} scenes", flush=True)

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
    all_scenes = sorted(json_dir.glob("*.json"))
    if args.max_scenes:
        all_scenes = all_scenes[:args.max_scenes]
    valid_scenes = [s for s in all_scenes if s.stem in cache_tokens]
    if args.num_shards > 1:
        valid_scenes = valid_scenes[args.shard_id::args.num_shards]
    print(f"[budget-rl] {len(valid_scenes)} scenes (shard {args.shard_id}/{args.num_shards})", flush=True)

    # Optimizer (includes budget_log_std)
    all_params = list(scorer_model.parameters()) + [budget_log_std]
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)

    # Training loop
    log_file = out_dir / "train_log.jsonl"
    logf = log_file.open("w")
    global_step = 0
    best_avg_reward = -float("inf")

    for epoch in range(args.num_epochs):
        perm = np.random.permutation(len(valid_scenes))
        epoch_rewards = []

        for g_start in range(0, len(valid_scenes), args.group_size):
            g_end = min(g_start + args.group_size, len(valid_scenes))
            group_idx = perm[g_start:g_end]
            t0 = time.time()

            group_rewards = []
            group_log_probs = []
            group_keep_ratios = []

            for idx in group_idx:
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
                )
                if result is None:
                    continue

                # Compute reward
                driving_reward = reward_fn.rl_pdm_score(
                    result["trajectory"], result["token"],
                    shaped=True,
                    baseline_scores=baseline_sub_scores.get(result["token"]),
                )
                if driving_reward is None:
                    driving_reward = 0.0

                # Efficiency bonus: reward for pruning more
                efficiency_bonus = 1.0 - result["keep_ratio"]

                # Combined reward
                total_reward = driving_reward + args.efficiency_beta * efficiency_bonus

                group_rewards.append(total_reward)
                group_log_probs.append(result["total_log_prob"])
                group_keep_ratios.append(result["keep_ratio"])

            if len(group_rewards) < 2:
                continue

            # Advantage (group-normalized)
            rewards_t = torch.tensor(group_rewards, device=device, dtype=torch.float32)
            advantage = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

            # Policy gradient
            log_probs_t = torch.stack(group_log_probs)
            policy_loss = -(advantage.detach() * log_probs_t).mean()

            # KL penalty
            kl_loss = torch.tensor(0.0, device=device)
            if args.kl_beta > 0:
                for p_curr, p_ref in zip(scorer_model.parameters(), ref_scorer.parameters()):
                    kl_loss = kl_loss + F.mse_loss(p_curr, p_ref, reduction='sum')
                kl_loss = args.kl_beta * kl_loss

            loss = policy_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            global_step += 1
            mean_reward = rewards_t.mean().item()
            mean_kr = np.mean(group_keep_ratios)
            epoch_rewards.append(mean_reward)

            if global_step % args.log_every == 0:
                rec = {
                    "step": global_step, "epoch": epoch,
                    "reward_mean": mean_reward,
                    "driving_reward": mean_reward - args.efficiency_beta * (1.0 - mean_kr),
                    "keep_ratio_mean": mean_kr,
                    "keep_ratio_std": float(np.std(group_keep_ratios)),
                    "budget_log_std": budget_log_std.item(),
                    "policy_loss": policy_loss.item(),
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "n_valid": len(group_rewards),
                    "elapsed_s": time.time() - t0,
                }
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(
                    f"[step {global_step:4d}] R={mean_reward:.4f} kr={mean_kr:.3f}±{np.std(group_keep_ratios):.3f} "
                    f"loss={loss.item():.4f} grad={grad_norm:.3f} "
                    f"({len(group_rewards)}/{len(group_idx)} scenes, {time.time()-t0:.1f}s)",
                    flush=True,
                )

            if global_step % args.save_every == 0:
                _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, f"step{global_step}", args)
                if mean_reward > best_avg_reward:
                    best_avg_reward = mean_reward
                    _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, "best", args)

        if epoch_rewards:
            ep_mean = np.mean(epoch_rewards)
            print(f"\n[budget-rl] Epoch {epoch}: avg_reward={ep_mean:.4f}\n", flush=True)
            if ep_mean > best_avg_reward:
                best_avg_reward = ep_mean
                _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, "best", args)

    _save(scorer_model, feat_mean, feat_std, n_cam, budget_log_std, out_dir, "final", args)
    logf.close()
    print(f"[budget-rl] DONE. Best reward: {best_avg_reward:.4f}. Output: {out_dir}", flush=True)


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
    }))
    (save_dir / "manifest.json").write_text(json.dumps({
        "spec": "budget_rl_v1",
        "method": "REINFORCE with driving reward + efficiency bonus",
        "efficiency_beta": args.efficiency_beta,
        "min_keep_ratio": args.min_keep_ratio,
        "max_keep_ratio": args.max_keep_ratio,
        "tag": tag,
    }, indent=2))
    print(f"[budget-rl] Saved: {save_dir}", flush=True)


if __name__ == "__main__":
    main()

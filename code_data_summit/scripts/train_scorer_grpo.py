"""train_scorer_grpo.py — GRPO/REINFORCE training for the token importance scorer.

Paper contribution: "First RL-optimized adaptive token pruning for AD-VLA"
  RL (REINFORCE with PDMS reward) directly optimizes which vision tokens to keep
  for driving trajectory quality — the scorer learns task-optimal token selection.

Pipeline per scene:
  1. Load scene → build input_features (same as eval agent)
  2. VLM pass-1: feature capture at layer 0 → vision_feat (720, 2048)
  3. Scorer(vision_feat + cam_onehot) → scores (720,) [WITH GRAD]
  4. Select top-B by score → prune mask
  5. VLM pass-2: generate trajectory under prune mask
  6. PDMS reward from metric cache
  7. REINFORCE: policy_loss = -advantage * log_prob(selection | scores)

Log-probability: We use the multi-label softmax approximation:
  log_prob(top-B | scores) = Σ_{i∈B} scores[i] - B * logsumexp(all scores)

Advantage: Group-normalized (gather K scenes, subtract mean, divide by std).

Usage:
  cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scorer_grpo.py \
    --scorer-ckpt ckpt/s3_token_scorer_mse --out-dir ckpt/s3_token_scorer_rl \
    --keep-ratio 0.5 --num-epochs 3 --group-size 8 --lr 1e-4

  Smoke test (20 scenes):
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scorer_grpo.py \
    --scorer-ckpt ckpt/s3_token_scorer_mse --out-dir ckpt/s3_token_scorer_rl_smoke \
    --keep-ratio 0.5 --max-scenes 20 --num-epochs 1 --group-size 4 --lr 1e-4
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
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA/navsim"))
sys.path.insert(0, str(ROOT / "code/third_party/AutoVLA"))

from rldrive.scoring.token_scorer import TokenImportanceScorer, cam_id_from_blocks, cam_onehot
from rldrive.scoring.attention_capture import (
    patch_vision_feature_capture,
    locate_prompt_landmarks,
    PromptIndex,
)
from rldrive.agents.token_prune_patch import patch_vision_token_prune, select_prune_positions
from rldrive.agents.token_prune_patch_varB import patch_vision_token_drop
from models.utils.score import PDM_Reward
from navsim.common.dataclasses import Trajectory
from navsim.agents.autovla_agent import AutoVLAAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Scorer GRPO/REINFORCE training")
    p.add_argument("--scorer-ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--json-dir", type=str, default=str(ROOT / "data/navtest_nocot"))
    p.add_argument("--metric-cache", type=str, default=str(ROOT / "data/navtest_metric_cache"))
    p.add_argument("--sensor-data-path", type=str,
                   default="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/test/openscene-v1.1/sensor_blobs/test")
    p.add_argument("--autovla-config", type=str,
                   default=str(ROOT / "code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml"))
    p.add_argument("--autovla-ckpt", type=str,
                   default=str(ROOT / "models/AutoVLA/AutoVLA_PDMS_89.ckpt"))
    # Training hyperparams
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1, help="Split valid scenes into this many shards")
    p.add_argument("--shard-id", type=int, default=0, help="Shard id in [0, num_shards)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prune-variant", choices=["attn_mask", "drop"], default="attn_mask")
    # Shaped reward (Option 3: sub-metric weighted delta)
    p.add_argument("--shaped-reward", action="store_true", default=True,
                   help="Use shaped sub-metric delta reward instead of raw PDMS product")
    p.add_argument("--baseline-scores", type=str, default=None,
                   help="Path to JSON with per-scene baseline sub-metric scores (for delta reward)")
    # Logging / checkpointing
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# ===========================================================================
# Model loading
# ===========================================================================

def load_autovla_for_inference(config_path: str, ckpt_path: str, sensor_data_path: str, device: str):
    """Load AutoVLA in inference mode (frozen, for pass-1 feature capture + pass-2 generation).

    We reuse the existing AutoVLAWithTokenPruneAgent architecture, but load just the
    core AutoVLA model (not the full agent) for direct manipulation.
    """
    import yaml
    from models.autovla import AutoVLA

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set inference config
    config.setdefault('inference', {})
    config['inference']['sample'] = {
        'max_length': 2048,
        'temperature': 0.01,  # near-greedy for reward stability
        'top_k': 0,
        'top_p': 1.0,
    }

    print(f"[scorer-grpo] Loading Qwen2.5-VL model...", flush=True)
    autovla = AutoVLA(config, inference=True, device=device)

    # Load SFT checkpoint (same logic as AutoVLAAgent)
    print(f"[scorer-grpo] Loading checkpoint: {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    # Remap keys: "autovla.vlm.xxx" → "vlm.xxx"
    new_sd = {}
    for k, v in sd.items():
        k2 = k.replace("autovla.", "")
        new_sd[k2] = v
    missing, unexpected = autovla.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[scorer-grpo] Missing keys: {len(missing)} (first 3: {missing[:3]})", flush=True)
    if unexpected:
        print(f"[scorer-grpo] Unexpected keys: {len(unexpected)}", flush=True)

    autovla.eval()
    for p in autovla.parameters():
        p.requires_grad_(False)

    print(f"[scorer-grpo] AutoVLA loaded and frozen. Device={device}", flush=True)
    return autovla, config


def load_scorer(ckpt_dir: str, device: str):
    """Load scorer model + normalization."""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = TokenImportanceScorer(
        emb_dim=int(cfg["emb_dim"]),
        n_cam=int(cfg["n_cam"]),
        hidden=int(cfg["hidden"]),
    )
    sd = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.to(device)

    norm = torch.load(ckpt_dir / "feature_norm.pt", map_location=device, weights_only=False)
    feat_mean = norm["mean"].to(device)
    feat_std = norm["std"].to(device)
    n_cam = int(cfg["n_cam"])
    return model, feat_mean, feat_std, n_cam


# ===========================================================================
# Scene data loading (reuse AutoVLAAgent's feature builders)
# ===========================================================================

def build_feature_loader(sensor_data_path: str, codebook_path: str):
    """Create a lightweight agent (no model) just for feature building."""
    traj_sampling = TrajectorySampling(num_poses=10, interval_length=0.5)
    agent = AutoVLAAgent(
        trajectory_sampling=traj_sampling,
        sensor_data_path=sensor_data_path,
        codebook_cache_path=codebook_path,
        skip_model_load=True,
    )
    return agent


def load_scene_features(agent: AutoVLAAgent, scene_json_path: Path, sensor_data_path: str) -> Dict[str, Any]:
    """Load a scene JSON and convert to input_features dict (same as RFTDataset)."""
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)

    input_features: Dict[str, Any] = {}
    for builder in agent.get_feature_builders():
        input_features.update(builder.compute_features(scene_data))
    input_features["sensor_data_path"] = sensor_data_path
    return input_features, scene_data['token']


# ===========================================================================
# Core: one scene pass (feature capture → score → prune → generate → reward)
# ===========================================================================

def process_one_scene(
    autovla,
    input_features: Dict[str, Any],
    token: str,
    scorer_model: nn.Module,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    n_cam: int,
    keep_ratio: float,
    prune_variant: str,
    device: str,
) -> Optional[Dict[str, Any]]:
    """Run 2-pass pipeline for one scene. Returns scores (with grad), reward, log_prob."""

    try:
        # === Build PromptIndex (need token IDs for vision detection) ===
        inputs = autovla.get_prompt(input_features)
        input_ids = inputs["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        processor = autovla.processor
        # Get special token IDs
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        # video_pad is the actual vision placeholder
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # Use whichever exists (Qwen2.5-VL uses video_pad for video mode)
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

        B = max(1, int(round(keep_ratio * N)))

        # === Pass 1: Capture vision features ===
        fbucket: Dict[str, Any] = {}
        with patch_vision_feature_capture(
            vlm=autovla.vlm,
            layer_idx=0,
            prompt_index=prompt_index,
            bucket=fbucket,
        ):
            with torch.no_grad():
                autovla.predict(input_features)  # pass-1, trajectory discarded

        if "vision_feat" not in fbucket:
            return None

        vision_feat = fbucket["vision_feat"]  # (N, 2048) on device

        # === Score with trainable scorer ===
        emb = (vision_feat.to(device).float() - feat_mean) / feat_std
        cam = cam_id_from_blocks(prompt_index.vision_token_positions, prompt_index.vision_blocks)
        coh = cam_onehot(cam, n_cam).to(device)
        x = torch.cat([emb, coh], dim=-1)  # (N, 2051)

        scores = scorer_model(x)  # (N,) — HAS GRAD through scorer params

        # === Select top-B ===
        _, top_indices = scores.topk(B, dim=0)

        # Compute log_prob: multi-label softmax approximation (normalized by B)
        # log_prob(top-B | scores) = [sum(scores[top-B]) - B * logsumexp(all_scores)] / B
        # Division by B keeps gradients stable regardless of keep_ratio
        log_prob = (scores[top_indices].sum() - B * torch.logsumexp(scores, dim=0)) / B

        # === Build prune mask ===
        all_positions = prompt_index.vision_token_positions
        keep_mask = torch.zeros(N, dtype=torch.bool)
        keep_mask[top_indices.cpu()] = True
        prune_idx = (~keep_mask).nonzero(as_tuple=True)[0]
        prune_positions = all_positions[prune_idx]

        # === Pass 2: Generate trajectory under pruning ===
        with ExitStack() as stack:
            if prune_variant == "drop" and prune_positions.numel() > 0:
                stack.enter_context(
                    patch_vision_token_drop(
                        vlm=autovla.vlm,
                        prune_positions=prune_positions,
                        verbose=False,
                    )
                )
            elif prune_positions.numel() > 0:
                stack.enter_context(
                    patch_vision_token_prune(
                        vlm=autovla.vlm,
                        prune_positions=prune_positions,
                        verbose=False,
                    )
                )
            with torch.no_grad():
                poses, _ = autovla.predict(input_features)

        # Build Trajectory object
        traj_sampling = TrajectorySampling(num_poses=10, interval_length=0.5)
        if poses is None or len(poses) < traj_sampling.num_poses:
            return None

        trajectory = Trajectory(
            poses[:traj_sampling.num_poses, :].cpu().numpy(),
            traj_sampling,
        )

        return {
            "log_prob": log_prob,
            "scores": scores,
            "trajectory": trajectory,
            "token": token,
            "N": N,
            "B": B,
        }

    except Exception as e:
        print(f"[scorer-grpo] Scene {token} error: {e}", flush=True)
        return None


# ===========================================================================
# Training loop
# ===========================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("[scorer-grpo] Scorer GRPO Training — RL-optimized token pruning", flush=True)
    print("=" * 70, flush=True)
    print(f"  scorer_ckpt  = {args.scorer_ckpt}", flush=True)
    print(f"  keep_ratio   = {args.keep_ratio}", flush=True)
    print(f"  group_size   = {args.group_size}", flush=True)
    print(f"  lr           = {args.lr}", flush=True)
    print(f"  kl_beta      = {args.kl_beta}", flush=True)
    print(f"  prune_variant= {args.prune_variant}", flush=True)
    print(f"  out_dir      = {out_dir}", flush=True)
    print("=" * 70, flush=True)

    # --- Load AutoVLA (frozen) ---
    autovla, vlm_config = load_autovla_for_inference(
        args.autovla_config, args.autovla_ckpt, args.sensor_data_path, device
    )

    # --- Load scorer (trainable) ---
    scorer_model, feat_mean, feat_std, n_cam = load_scorer(args.scorer_ckpt, device)
    scorer_model.train()

    # Reference scorer (frozen, for KL)
    ref_scorer = copy.deepcopy(scorer_model)
    ref_scorer.eval()
    for p in ref_scorer.parameters():
        p.requires_grad_(False)

    # --- Reward function ---
    reward_fn = PDM_Reward(Path(args.metric_cache))
    cache_tokens = set(reward_fn.metric_cache_loader.metric_cache_paths.keys())

    # --- Baseline sub-scores for shaped reward (Option 3) ---
    baseline_sub_scores = {}
    if args.shaped_reward and args.baseline_scores and Path(args.baseline_scores).exists():
        import json as _json
        baseline_sub_scores = _json.loads(Path(args.baseline_scores).read_text())
        print(f"[scorer-grpo] Loaded baseline sub-scores for {len(baseline_sub_scores)} scenes", flush=True)
    elif args.shaped_reward:
        print("[scorer-grpo] WARNING: shaped_reward=True but no --baseline-scores provided. "
              "Using absolute weighted sum (no delta). For best results, provide per-scene "
              "baseline from r=1.0 eval.", flush=True)

    # --- Feature loader (lightweight agent, no VLM) ---
    codebook_path = vlm_config['model']['codebook_cache_path']
    feat_agent = build_feature_loader(args.sensor_data_path, codebook_path)

    # --- Scene list ---
    json_dir = Path(args.json_dir)
    all_scenes = sorted(json_dir.glob("*.json"))
    if args.max_scenes:
        all_scenes = all_scenes[:args.max_scenes]
    valid_scenes = [s for s in all_scenes if s.stem in cache_tokens]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    n_valid_before_shard = len(valid_scenes)
    if args.num_shards > 1:
        valid_scenes = valid_scenes[args.shard_id::args.num_shards]
    print(
        f"[scorer-grpo] Scenes: {len(all_scenes)} total, {n_valid_before_shard} with metric cache, "
        f"shard {args.shard_id}/{args.num_shards} -> {len(valid_scenes)} scenes",
        flush=True,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(scorer_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Training ---
    log_file = out_dir / "train_log.jsonl"
    logf = log_file.open("w")
    global_step = 0
    running_reward = 0.0
    best_avg_reward = -float("inf")

    for epoch in range(args.num_epochs):
        perm = np.random.permutation(len(valid_scenes))
        epoch_rewards = []
        epoch_losses = []

        for g_start in range(0, len(valid_scenes), args.group_size):
            g_end = min(g_start + args.group_size, len(valid_scenes))
            group_idx = perm[g_start:g_end]
            t0 = time.time()

            # Collect group results
            group_rewards = []
            group_log_probs = []

            for idx in group_idx:
                scene_path = valid_scenes[idx]
                # Load features
                try:
                    input_features, token = load_scene_features(
                        feat_agent, scene_path, args.sensor_data_path
                    )
                except Exception as e:
                    print(f"[scorer-grpo] Failed to load {scene_path.stem}: {e}", flush=True)
                    continue

                # Run 2-pass pipeline
                result = process_one_scene(
                    autovla=autovla,
                    input_features=input_features,
                    token=token,
                    scorer_model=scorer_model,
                    feat_mean=feat_mean,
                    feat_std=feat_std,
                    n_cam=n_cam,
                    keep_ratio=args.keep_ratio,
                    prune_variant=args.prune_variant,
                    device=device,
                )

                if result is None:
                    continue

                # Compute reward (shaped=True uses sub-metric weighted delta)
                reward = reward_fn.rl_pdm_score(
                    result["trajectory"], result["token"],
                    shaped=getattr(args, 'shaped_reward', True),
                    baseline_scores=baseline_sub_scores.get(result["token"]),
                )
                if reward is None:
                    reward = 0.0

                group_rewards.append(reward)
                group_log_probs.append(result["log_prob"])

            # Need at least 2 samples for advantage normalization
            if len(group_rewards) < 2:
                continue

            # --- Compute advantage (group-normalized) ---
            rewards_t = torch.tensor(group_rewards, device=device, dtype=torch.float32)
            advantage = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

            # --- Policy gradient loss ---
            log_probs_t = torch.stack(group_log_probs)
            policy_loss = -(advantage.detach() * log_probs_t).mean()

            # --- KL penalty (weight-space L2 to reference) ---
            kl_loss = torch.tensor(0.0, device=device)
            if args.kl_beta > 0:
                for p_curr, p_ref in zip(scorer_model.parameters(), ref_scorer.parameters()):
                    kl_loss = kl_loss + F.mse_loss(p_curr, p_ref, reduction='sum')
                kl_loss = args.kl_beta * kl_loss

            loss = policy_loss + kl_loss

            # --- Backward + update ---
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(scorer_model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            mean_reward = rewards_t.mean().item()
            epoch_rewards.append(mean_reward)
            epoch_losses.append(loss.item())
            running_reward = 0.95 * running_reward + 0.05 * mean_reward if global_step > 1 else mean_reward

            elapsed = time.time() - t0

            # --- Logging ---
            if global_step % args.log_every == 0:
                rec = {
                    "step": global_step, "epoch": epoch,
                    "reward_mean": mean_reward,
                    "reward_std": rewards_t.std().item(),
                    "reward_max": rewards_t.max().item(),
                    "reward_min": rewards_t.min().item(),
                    "running_reward": running_reward,
                    "policy_loss": policy_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "n_valid": len(group_rewards),
                    "elapsed_s": elapsed,
                    "s_per_scene": elapsed / max(len(group_rewards), 1),
                }
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(
                    f"[step {global_step:4d}] R={mean_reward:.4f} "
                    f"(run={running_reward:.4f}) loss={loss.item():.6f} "
                    f"grad={grad_norm:.4f} {len(group_rewards)}/{len(group_idx)} scenes "
                    f"({elapsed:.1f}s = {elapsed/max(len(group_rewards),1):.1f}s/sc)",
                    flush=True,
                )

            # --- Save checkpoint ---
            if global_step % args.save_every == 0:
                _save(scorer_model, feat_mean, feat_std, n_cam, out_dir, f"step{global_step}", args)
                if mean_reward > best_avg_reward:
                    best_avg_reward = mean_reward
                    _save(scorer_model, feat_mean, feat_std, n_cam, out_dir, "best", args)

        # Epoch summary
        if epoch_rewards:
            ep_mean = np.mean(epoch_rewards)
            print(f"\n[scorer-grpo] Epoch {epoch} complete: avg_reward={ep_mean:.4f} "
                  f"avg_loss={np.mean(epoch_losses):.6f} groups={len(epoch_rewards)}\n", flush=True)
            if ep_mean > best_avg_reward:
                best_avg_reward = ep_mean
                _save(scorer_model, feat_mean, feat_std, n_cam, out_dir, "best", args)

    # Final save
    _save(scorer_model, feat_mean, feat_std, n_cam, out_dir, "final", args)
    logf.close()
    print(f"[scorer-grpo] DONE. Best avg reward: {best_avg_reward:.4f}. Output: {out_dir}", flush=True)


def _save(model, feat_mean, feat_std, n_cam, out_dir, tag, args):
    """Save in ScorerRunner-compatible format."""
    save_dir = Path(out_dir) if tag == "final" else Path(out_dir) / f"ckpt_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "checkpoint.pt")
    torch.save({"mean": feat_mean.cpu(), "std": feat_std.cpu()}, save_dir / "feature_norm.pt")
    (save_dir / "config.json").write_text(json.dumps({
        "emb_dim": model.emb_dim, "n_cam": model.n_cam, "hidden": 256, "label_layer": 12,
    }))
    (save_dir / "manifest.json").write_text(json.dumps({
        "spec": "s3_token_scorer_rl_v1",
        "method": "REINFORCE with PDMS reward (scorer GRPO)",
        "init_from": str(args.scorer_ckpt),
        "keep_ratio": args.keep_ratio,
        "prune_variant": args.prune_variant,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "lr": args.lr, "kl_beta": args.kl_beta, "group_size": args.group_size,
        "tag": tag,
    }, indent=2))
    print(f"[scorer-grpo] Saved: {save_dir}", flush=True)


if __name__ == "__main__":
    main()

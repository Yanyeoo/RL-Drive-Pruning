"""run_feature_dump.py — S3 per-token feature dumper.

Mirrors run_attention_probe.py but captures the ViT->LLM interface hidden state
at the vision positions (decoder layer `--feature-layer`, default 0) instead of
attention. Forward-only, no metric_cache dependency, shardable across GPUs.

Output: per-scene <token>.pt with keys {vision_feat (N,H), vision_token_positions,
vision_blocks, ...} under --save-dir. Labels come separately from the existing
m1b2 attention dump (docs/specs/s3_token_scorer_spec.md §2).

Usage (one GPU shard):
    PYTHONPATH=code:navsim:AutoVLA python -m rldrive.scoring.run_feature_dump \
        --save-dir /abs/data/s3_scorer/features --gpu 0 \
        --json-dir /abs/data/navtrain_nocot \
        --shard-stride 4 --shard-index 0 --max-scenes 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = "/apdcephfs/private_shayladeng/tokenrl_autoVLA"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3 per-token feature dumper")
    p.add_argument("--save-dir", required=True, type=Path)
    p.add_argument("--feature-layer", type=int, default=0,
                   help="decoder layer whose INPUT hidden state is captured (0 = LLM-input emb)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--checkpoint", default=f"{ROOT}/models/AutoVLA/AutoVLA_PDMS_89.ckpt")
    p.add_argument("--config",
                   default=f"{ROOT}/code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml")
    p.add_argument("--codebook",
                   default=f"{ROOT}/code/third_party/AutoVLA/codebook_cache/agent_vocab.pkl")
    p.add_argument("--sensor-data", default=f"{ROOT}/data/navsim_v2_local")
    p.add_argument("--json-dir", default=f"{ROOT}/data/navtrain_nocot")
    p.add_argument("--token-list", type=Path, default=None)
    p.add_argument("--shard-stride", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--skip-done", action="store_true",
                   help="skip tokens whose <token>.pt already exists in save-dir")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    for pth in (args.checkpoint, args.config, args.codebook):
        if pth and not Path(pth).exists():
            raise FileNotFoundError(pth)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from rldrive.agents.autovla_with_attention import AutoVLAWithAttentionAgent

    traj_samp = TrajectorySampling(time_horizon=5, interval_length=0.5)
    print(f"[featdump] save_dir={args.save_dir} feat_layer={args.feature_layer} "
          f"gpu={args.gpu} shard={args.shard_index}/{args.shard_stride}", flush=True)
    t0 = time.time()
    agent = AutoVLAWithAttentionAgent(
        trajectory_sampling=traj_samp,
        checkpoint_path=args.checkpoint,
        sensor_data_path=args.sensor_data,
        codebook_cache_path=args.codebook,
        lora_conf={"use_lora": False},
        config_path=args.config,
        device="cuda:0",
        attention_enabled=False,
        head_mask_layers=None,
        feature_capture_enabled=True,
        feature_layer_idx=args.feature_layer,
        feature_save_dir=str(args.save_dir),
    )
    agent.initialize()
    print(f"[featdump] model loaded in {time.time()-t0:.1f}s", flush=True)
    if args.dry_run:
        print("[featdump] DRY RUN", flush=True)
        return 0

    json_dir = Path(args.json_dir)
    if args.token_list is not None:
        tokens = [l.strip() for l in args.token_list.read_text().splitlines()
                  if l.strip() and not l.strip().startswith("#")]
    else:
        tokens = sorted(p.stem for p in json_dir.glob("*.json"))
    if args.shard_stride > 1:
        tokens = [t for i, t in enumerate(tokens) if i % args.shard_stride == args.shard_index]
    if args.max_scenes is not None:
        tokens = tokens[: args.max_scenes]
    print(f"[featdump] num_tokens={len(tokens)}", flush=True)
    if not tokens:
        print("[featdump] ERROR: no tokens", flush=True)
        return 2

    n_ok = n_skip = n_err = 0
    t_loop = time.time()
    for idx, tok in enumerate(tokens):
        out_pt = args.save_dir / f"{tok}.pt"
        if args.skip_done and out_pt.exists():
            n_skip += 1
            continue
        jp = json_dir / f"{tok}.json"
        if not jp.exists():
            n_skip += 1
            continue
        try:
            with jp.open("r") as f:
                scene_data = json.load(f)
            scene_data.setdefault("token", tok)
            agent.compute_trajectory(scene_data)
            n_ok += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                avg = (time.time() - t_loop) / (idx + 1)
                eta = avg * (len(tokens) - idx - 1)
                print(f"[featdump] [{idx+1}/{len(tokens)}] {tok} OK "
                      f"avg={avg:.2f}s eta={eta/60:.1f}min ok={n_ok} skip={n_skip} err={n_err}",
                      flush=True)
        except Exception as e:
            n_err += 1
            print(f"[featdump] [{idx+1}/{len(tokens)}] {tok} ERROR: {type(e).__name__}: {e}",
                  flush=True)
            if n_err > 8 and n_err > n_ok:
                print("[featdump] FATAL: too many errors, aborting", flush=True)
                return 3
    el = time.time() - t_loop
    print(f"[featdump] DONE: {n_ok} ok, {n_skip} skip, {n_err} err in {el/60:.1f}min "
          f"({el/max(n_ok,1):.2f}s/scene) -> {args.save_dir}", flush=True)
    return 0 if n_ok > 0 else 4


if __name__ == "__main__":
    sys.exit(main())

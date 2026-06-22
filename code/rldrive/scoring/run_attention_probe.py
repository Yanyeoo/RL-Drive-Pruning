"""run_attention_probe.py — minimal M1.a runner.

NOT a hydra entry point; talks directly to AutoVLAWithAttentionAgent so we
can use either navtest (data already on disk) or navtrain (after chain_complete)
without rewriting navsim's pdm-score pipeline.

Usage:
    PYTHONPATH=/path/to/code:/path/to/navsim:/path/to/AutoVLA \
    python -m rldrive.scoring.run_attention_probe \
        --scene-filter navtest_smoke5 \
        --save-dir /abs/path/m1a_layer14 \
        --layer-idx 14 \
        --gpu 0 \
        --max-scenes 10

Status: DRAFT — wire-up is import-tested. First real GPU smoke is gated on
        user confirming the navtest pivot proposal
        (docs/_internal/decision_proposal_2026-06-17_m1a_on_navtest.md).

What it deliberately AVOIDS:
  - Running navsim's metric pipeline (PDMS). M1.a does not need a score,
    only the captured attention tensors per scene. PDMS would just add
    cost and metric-cache dependencies that are missing for navtrain.
  - Using hydra. The override surface for run_pdm_score_cot.py is large
    and we'd lose visibility into what config actually got loaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1.a attention layer-probe runner")
    p.add_argument("--scene-filter", required=True,
                   help="scene_filter yaml stem (e.g. navtest_smoke5) — must already exist"
                        " under navsim's scene_filter dir")
    p.add_argument("--save-dir", required=True, type=Path,
                   help="absolute path; per-scene attention .pt files written here")
    p.add_argument("--layer-idx", type=int, default=14,
                   help="decoder layer to capture from (0..27 for Qwen2.5-VL-3B)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-scenes", type=int, default=None,
                   help="stop after N scenes (probe-set A size = 100 by default)")
    p.add_argument("--checkpoint", default="/apdcephfs/private_shayladeng/tokenrl_autoVLA/models/AutoVLA/AutoVLA_PDMS_89.ckpt")
    p.add_argument("--config",
                   default="/apdcephfs/private_shayladeng/tokenrl_autoVLA/code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml")
    p.add_argument("--codebook",
                   default="/apdcephfs/private_shayladeng/tokenrl_autoVLA/code/third_party/AutoVLA/codebook_cache/agent_vocab.pkl")
    p.add_argument("--sensor-data",
                   default="/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navsim_v2_local",
                   help="placeholder; agent reads absolute paths from per-scene json")
    p.add_argument("--json-dir",
                   default="/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtest_nocot",
                   help="dir of per-scene pre-tokenized json files (navtest_nocot or navtrain_nocot)")
    p.add_argument("--per-head", action="store_true",
                   help="store per-head attention (num_heads, N_vision) instead of head-mean")
    p.add_argument("--dry-run", action="store_true",
                   help="parse args + load model, do not run any scenes")
    p.add_argument("--token-list", type=Path, default=None,
                   help="path to text file with one scene token per line. If set, "
                        "overrides --scene-filter and only these tokens (looked up "
                        "under --json-dir as <token>.json) are processed. Used for "
                        "GPU-shard parallel sweeps.")
    p.add_argument("--shard-stride", type=int, default=1,
                   help="if >1, process tokens with index%%shard_stride == --shard-index")
    p.add_argument("--shard-index", type=int, default=0,
                   help="see --shard-stride")
    return p.parse_args(argv)


def _set_gpu(gpu: int) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))


def _validate_inputs(args: argparse.Namespace) -> None:
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not Path(args.config).exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    if not Path(args.codebook).exists():
        raise FileNotFoundError(f"codebook not found: {args.codebook}")
    args.save_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _set_gpu(args.gpu)
    _validate_inputs(args)

    # Imports kept inside main so --help doesn't pay for them
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from rldrive.agents.autovla_with_attention import AutoVLAWithAttentionAgent

    print(f"[probe] scene_filter   = {args.scene_filter}", flush=True)
    print(f"[probe] save_dir       = {args.save_dir}", flush=True)
    print(f"[probe] layer_idx      = {args.layer_idx}", flush=True)
    print(f"[probe] gpu            = {args.gpu}", flush=True)
    print(f"[probe] per_head       = {args.per_head}", flush=True)
    print(f"[probe] max_scenes     = {args.max_scenes}", flush=True)

    traj_samp = TrajectorySampling(time_horizon=5, interval_length=0.5)

    print("[probe] loading AutoVLAWithAttentionAgent...", flush=True)
    t0 = time.time()
    agent = AutoVLAWithAttentionAgent(
        trajectory_sampling=traj_samp,
        checkpoint_path=args.checkpoint,
        sensor_data_path=args.sensor_data,
        codebook_cache_path=args.codebook,
        lora_conf={"use_lora": False},
        config_path=args.config,
        device=f"cuda:{0}",   # CUDA_VISIBLE_DEVICES already filtered
        attention_enabled=True,
        attention_layer_idx=args.layer_idx,
        attention_save_dir=str(args.save_dir),
        attention_average_heads=(not args.per_head),
        attention_assert_qlen=True,
    )
    agent.initialize()
    print(f"[probe] model loaded in {time.time() - t0:.1f}s", flush=True)

    if args.dry_run:
        print("[probe] DRY RUN — skipping scene loop", flush=True)
        return 0

    # ---------------------------------------------------------------
    # Scene loop  (Path 1: per-scene json under --json-dir)
    # ---------------------------------------------------------------
    # Each json under --json-dir is a token-keyed scene_data payload that
    # AutoVLAAgent.compute_trajectory(scene_data) consumes directly.
    # Token resolution order:
    #   1. --token-list (explicit list file, one token per line)
    #   2. --scene-filter == "navtest_smoke_seed"  -> use data/navtest_nocot_smoke_seed/
    #      (pre-baked 5-token seed dir; file stems are tokens)
    #   3. otherwise: enumerate args.json_dir, sorted by stem
    # Then optionally shard (--shard-stride/--shard-index) and cap (--max-scenes).
    json_dir = Path(args.json_dir)
    if not json_dir.is_dir():
        raise FileNotFoundError(f"--json-dir not found: {json_dir}")

    if args.token_list is not None:
        if not args.token_list.exists():
            raise FileNotFoundError(f"--token-list not found: {args.token_list}")
        tokens = [
            line.strip() for line in args.token_list.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        token_source = f"token-list:{args.token_list.name} ({len(tokens)} tokens)"
    elif args.scene_filter == "navtest_smoke_seed":
        # Pre-baked 5-scene smoke under data/navtest_nocot_smoke_seed/
        seed_dir = Path(
            "/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtest_nocot_smoke_seed"
        )
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"smoke seed dir not found: {seed_dir}")
        tokens = sorted(p.stem for p in seed_dir.glob("*.json"))
        json_dir = seed_dir  # override
        token_source = f"smoke-seed-dir:{seed_dir} ({len(tokens)} tokens)"
    else:
        tokens = sorted(p.stem for p in json_dir.glob("*.json"))
        token_source = f"json-dir:{json_dir} ({len(tokens)} tokens)"

    # Sharding: distribute tokens across GPUs by index modulo
    if args.shard_stride > 1:
        tokens = [t for i, t in enumerate(tokens) if i % args.shard_stride == args.shard_index]
        token_source += f" [shard {args.shard_index}/{args.shard_stride}: {len(tokens)} tokens]"

    # Cap
    if args.max_scenes is not None:
        tokens = tokens[: args.max_scenes]

    print(f"[probe] token_source   = {token_source}", flush=True)
    print(f"[probe] num_tokens     = {len(tokens)}", flush=True)
    if not tokens:
        print("[probe] ERROR: no tokens to process", flush=True)
        return 2

    import json
    n_ok = 0
    n_skip = 0
    n_err = 0
    t_loop = time.time()
    for idx, tok in enumerate(tokens):
        json_path = json_dir / f"{tok}.json"
        if not json_path.exists():
            print(f"[probe] [{idx+1}/{len(tokens)}] {tok} SKIP (json not found)", flush=True)
            n_skip += 1
            continue
        try:
            with json_path.open("r") as f:
                scene_data = json.load(f)
            if "token" not in scene_data:
                scene_data["token"] = tok
            t_scene = time.time()
            _traj, _cot = agent.compute_trajectory(scene_data)
            dt = time.time() - t_scene
            n_ok += 1
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(tokens) - 1:
                avg = (time.time() - t_loop) / (idx + 1)
                eta = avg * (len(tokens) - idx - 1)
                print(
                    f"[probe] [{idx+1}/{len(tokens)}] {tok} OK ({dt:.2f}s) "
                    f"| avg={avg:.2f}s/scene eta={eta/60:.1f}min "
                    f"ok={n_ok} skip={n_skip} err={n_err}",
                    flush=True,
                )
        except Exception as e:
            n_err += 1
            print(f"[probe] [{idx+1}/{len(tokens)}] {tok} ERROR: {type(e).__name__}: {e}",
                  flush=True)
            # Continue to next scene; one bad json should not kill the whole sweep
            if n_err > 5 and n_err > n_ok:
                print(f"[probe] FATAL: too many consecutive errors ({n_err}), aborting",
                      flush=True)
                return 3

    elapsed = time.time() - t_loop
    print(f"[probe] DONE: {n_ok} ok, {n_skip} skip, {n_err} err in {elapsed/60:.1f}min "
          f"({elapsed/max(n_ok,1):.2f}s/scene)", flush=True)
    print(f"[probe] outputs in: {args.save_dir}", flush=True)
    return 0 if n_ok > 0 else 4


if __name__ == "__main__":
    sys.exit(main())

"""Build M0.2 navtrain splits.

After `download_navtrain_robust.sh` + `install_navtrain.sh` finish, this
script enumerates all navtrain scenes via navsim's standard SceneLoader,
then produces:

    data/splits/probe_A.txt    -- 100 tokens, stratified by driving_command (4 x 25)
    data/splits/train_pool.txt -- 90% of (navtrain \\ probe_A), random
    data/splits/val_pool.txt   -- 10% of (navtrain \\ probe_A), random

driving_command is a 4-dim one-hot in each frame dict (Q3.c (i)):
    [follow, left, right, ???]    -- exact ordering verified at runtime

Deterministic via --seed (default 0).

Usage (single shot, ~10 min for full navtrain enumeration):
    OPENSCENE_DATA_ROOT=/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2 \\
    NAVSIM_DEVKIT_ROOT=.../AutoVLA/navsim \\
    python scripts/build_m02_splits.py \\
        --out-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/data/splits \\
        --seed 0

Outputs are deterministic given (seed, navtrain snapshot).
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--probe-per-class", type=int, default=25,
        help="how many tokens per driving_command class in probe set A",
    )
    ap.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="val / (train+val) ratio for the train_pool split",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- import navsim lazily; must be on PYTHONPATH ----------------------
    try:
        from navsim.common.dataclasses import SceneFilter, SensorConfig
        from navsim.common.dataloader import SceneLoader
    except ImportError as e:
        print("FATAL: navsim not importable. Add NAVSIM_DEVKIT_ROOT to "
              "PYTHONPATH first.", file=sys.stderr)
        print(f"  underlying error: {e}", file=sys.stderr)
        return 2

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    log_path = data_root / "navsim_logs" / "trainval"
    blobs_path = data_root / "sensor_blobs" / "trainval"
    if not log_path.is_dir():
        print(f"FATAL: navtrain logs missing at {log_path}", file=sys.stderr)
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- enumerate scenes via standard SceneFilter ------------------------
    print(f"[1/4] enumerating navtrain scenes from {log_path}")
    scene_filter = SceneFilter()  # defaults match agents/training
    loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = loader.tokens
    print(f"       {len(tokens)} evaluable tokens")

    # ---- read driving_command per token via scene_frames_dicts ------------
    print("[2/4] extracting driving_command per token")
    # The frame at index (num_history_frames - 1) is the "current" frame
    # whose token names the scene. SceneLoader exposes scene_frames_dicts.
    nh = scene_filter.num_history_frames
    cls_of: dict[str, int] = {}
    for tok in tokens:
        frames = loader.scene_frames_dicts[tok]
        dc = np.asarray(frames[nh - 1]["driving_command"]).reshape(-1)
        cls_of[tok] = int(np.argmax(dc))
    dist = collections.Counter(cls_of.values())
    print(f"       driving_command class distribution: {dict(dist)}")

    # ---- probe set A: stratified sample ----------------------------------
    print(f"[3/4] sampling probe_A: {args.probe_per_class} per class")
    probe: list[str] = []
    by_cls: dict[int, list[str]] = collections.defaultdict(list)
    for tok, c in cls_of.items():
        by_cls[c].append(tok)
    for c in sorted(by_cls):
        bucket = sorted(by_cls[c])  # deterministic order before shuffle
        rng.shuffle(bucket)
        take = bucket[: args.probe_per_class]
        if len(take) < args.probe_per_class:
            print(f"  WARN: class {c} has only {len(bucket)} tokens, "
                  f"using all of them", file=sys.stderr)
        probe.extend(take)
    probe_set = set(probe)
    print(f"       probe_A size = {len(probe)}")

    # ---- train/val split on (all \\ probe) -------------------------------
    print(f"[4/4] random 90/10 split on (navtrain \\ probe_A)")
    rest = sorted(set(tokens) - probe_set)
    rng.shuffle(rest)
    n_val = int(round(len(rest) * args.val_ratio))
    val = rest[:n_val]
    train = rest[n_val:]
    print(f"       train={len(train)}  val={len(val)}")

    # ---- write outputs ----------------------------------------------------
    def _write(name: str, items: list[str]) -> None:
        p = args.out_dir / name
        with p.open("w") as f:
            for t in items:
                f.write(f"{t}\n")
        print(f"       wrote {p}  ({len(items)} tokens)")

    _write("probe_A.txt", sorted(probe))
    _write("train_pool.txt", sorted(train))
    _write("val_pool.txt", sorted(val))

    # ---- write distribution sidecar --------------------------------------
    side = args.out_dir / "m02_split_stats.txt"
    with side.open("w") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"total_tokens={len(tokens)}\n")
        f.write(f"driving_command_dist={dict(dist)}\n")
        f.write(f"probe_size={len(probe)} (per class={args.probe_per_class})\n")
        f.write(f"train_size={len(train)}\n")
        f.write(f"val_size={len(val)}\n")
        f.write(f"val_ratio={args.val_ratio}\n")
    print(f"       wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

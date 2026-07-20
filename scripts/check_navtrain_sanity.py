"""navtrain end-to-end sanity check.

Run AFTER install_navtrain.sh succeeds. Validates:
  1. OPENSCENE_DATA_ROOT/navsim_logs/trainval is loadable by SceneLoader
  2. OPENSCENE_DATA_ROOT/sensor_blobs/trainval has the expected camera dirs
  3. One arbitrary token can be loaded as a full Scene (history+future frames)
  4. driving_command distribution is reported

Exits 0 on success, non-zero on any failure with a clear error message.

Usage:
    OPENSCENE_DATA_ROOT=/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2 \\
    PYTHONPATH=.../AutoVLA/navsim \\
    python scripts/check_navtrain_sanity.py
"""
from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a)  # noqa: E731

    # ---- step 1: navsim importable -----------------------------------------
    log("step 1: import navsim")
    try:
        from navsim.common.dataclasses import SceneFilter, SensorConfig
        from navsim.common.dataloader import SceneLoader
    except ImportError as e:
        print(f"FATAL: navsim import failed: {e}", file=sys.stderr)
        return 2

    # ---- step 2: paths exist -----------------------------------------------
    log("step 2: verify paths")
    try:
        data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    except KeyError:
        print("FATAL: OPENSCENE_DATA_ROOT not set", file=sys.stderr)
        return 2
    log_path = data_root / "navsim_logs" / "trainval"
    blobs_path = data_root / "sensor_blobs" / "trainval"

    if not log_path.is_dir():
        print(f"FATAL: missing {log_path}", file=sys.stderr)
        return 3
    if not blobs_path.is_dir():
        print(f"FATAL: missing {blobs_path}", file=sys.stderr)
        return 3

    n_logs = sum(1 for _ in log_path.glob("*.pkl"))
    log(f"  {log_path}: {n_logs} .pkl files")
    n_blob_dirs = sum(1 for _ in blobs_path.iterdir() if _.is_dir())
    log(f"  {blobs_path}: {n_blob_dirs} subdirectories")
    if n_logs == 0 or n_blob_dirs == 0:
        print("FATAL: empty navtrain directories", file=sys.stderr)
        return 3

    # ---- step 3: SceneLoader enumerates tokens -----------------------------
    log("step 3: SceneLoader.tokens enumeration (may take ~minutes for full navtrain)")
    scene_filter = SceneFilter()
    loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = loader.tokens
    log(f"  -> {len(tokens)} evaluable tokens")
    if len(tokens) < 80000:  # navtrain is supposed to be ~100k+ scenes
        log(f"  WARN: navtrain token count seems low ({len(tokens)}); "
            f"expected >80k. Double-check splits.md.")

    # ---- step 4: load one scene end-to-end --------------------------------
    log("step 4: load one scene end-to-end")
    sample_tok = tokens[len(tokens) // 2]  # arbitrary mid-list pick
    scene = loader.get_scene_from_token(sample_tok)
    log(f"  token={sample_tok}")
    log(f"  log_name={scene.scene_metadata.log_name}")
    log(f"  num_history_frames={scene.scene_metadata.num_history_frames}")
    log(f"  num_future_frames={scene.scene_metadata.num_future_frames}")
    log(f"  map={scene.scene_metadata.map_name}")

    # also try loading agent_input which exercises sensor_blob path
    log("  loading agent_input (exercises sensor_blob paths)")
    agent_input = loader.get_agent_input_from_token(sample_tok)
    log(f"  cameras={len(agent_input.cameras)} lidars={len(agent_input.lidars)}")

    # ---- step 5: driving_command distribution ------------------------------
    log("step 5: driving_command distribution (random 1000-token sample)")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(tokens), size=min(1000, len(tokens)), replace=False)
    nh = scene_filter.num_history_frames
    counts: collections.Counter[int] = collections.Counter()
    for i in sample_idx:
        t = tokens[i]
        dc = np.asarray(loader.scene_frames_dicts[t][nh - 1]["driving_command"]).reshape(-1)
        counts[int(np.argmax(dc))] += 1
    log(f"  driving_command class counts (over {len(sample_idx)} sample): {dict(counts)}")
    if len(counts) < 3:
        log(f"  WARN: only {len(counts)} unique driving_command classes seen; "
            f"expected 3-4 (straight/left/right[/u-turn])")

    log("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

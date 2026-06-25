"""
Scan navtrain logs + sensor_blobs to find tokens whose CAM_F0 jpg actually
exists on disk (partial download case). Output one token per line.

Created 2026-06-24 21:16 to unblock M1.b2 Stage 2 — the sensor_blobs/trainval
directory has only ~30% of frames per log actually downloaded, so the full
103,288-token navtrain split must be filtered to the disk-available subset.

Assumption (verified): if a log's CAM_F0 has jpg X, then CAM_L0/R0/B0 also
have the corresponding jpg X for the same frame (8 cams downloaded together).
So checking only CAM_F0 is sufficient.

Usage:
  python tools/scan_available_tokens.py \
      --log-dir /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/navsim_logs/trainval \
      --sb-dir /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/trainval \
      --out exp/m1b2_navtrain_available_tokens.txt \
      --workers 32
"""
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


def scan_log(args_tuple):
    """Return list of (token,) for which CAM_F0 jpg exists on disk."""
    log_path, sb_dir = args_tuple
    log_stem = os.path.basename(log_path).replace(".pkl", "")
    cam_dir = os.path.join(sb_dir, log_stem, "CAM_F0")
    if not os.path.isdir(cam_dir):
        return log_stem, []
    try:
        present = set(os.listdir(cam_dir))
    except OSError:
        return log_stem, []
    try:
        with open(log_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return log_stem, []
    if not isinstance(data, list):
        return log_stem, []
    out = []
    for scene in data:
        if not isinstance(scene, dict):
            continue
        cams = scene.get("cams")
        if not isinstance(cams, dict):
            continue
        cf = cams.get("CAM_F0")
        if not isinstance(cf, dict):
            continue
        dp = cf.get("data_path", "")
        base = os.path.basename(dp)
        if base and base in present:
            tok = scene.get("token")
            if tok:
                out.append(tok)
    return log_stem, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--sb-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    log_files = sorted(
        os.path.join(args.log_dir, f)
        for f in os.listdir(args.log_dir)
        if f.endswith(".pkl")
    )
    print(f"[scan] {len(log_files)} log files to scan, workers={args.workers}", flush=True)

    available = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_log, (lp, args.sb_dir)) for lp in log_files]
        for fut in as_completed(futs):
            log_stem, toks = fut.result()
            available.extend(toks)
            completed += 1
            if completed % 100 == 0 or completed == len(log_files):
                print(
                    f"[scan] {completed}/{len(log_files)} logs done, "
                    f"running available token count = {len(available)}",
                    flush=True,
                )
    available = sorted(set(available))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for t in available:
            f.write(t + "\n")
    print(f"[scan] DONE: {len(available)} unique available tokens -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

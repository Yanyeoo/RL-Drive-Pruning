#!/usr/bin/env python
"""
Scan navtrain (trainval split) for scene-level missing-image issues.

Strategy:
  - For each of 1310 navsim_logs/trainval/<log>.pkl:
      - readdir each CAM_* subdir under sensor_blobs/trainval/<log>/ → set of existing jpg basenames
      - iterate the 398 frames in the pkl
      - for each frame: check all 8 cams' data_path basename ∈ existing set
      - if any cam's image missing → frame.token is BAD
  - Output:
      navtrain_missing_report.json : per-log stats + global BAD token list
      navtrain_clean_tokens.txt    : sorted list of GOOD scene tokens

This makes only ~10500 readdir calls (1310 logs × 8 cams), avoiding per-jpg stat storms.
"""
from __future__ import annotations
import os
import json
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

NAVSIM_LOGS = Path("/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/navsim_logs/trainval")
SENSOR_BLOBS = Path("/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/trainval")
CAM_NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

OUT_DIR = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_navtrain_probeA_setup")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "navtrain_missing_report.json"
CLEAN_TOKENS = OUT_DIR / "navtrain_clean_tokens.txt"


def scan_one_log(log_pkl: Path) -> dict:
    """Scan one log_pkl, return {log_name, total_frames, bad_tokens, missing_jpgs_sample, log_dir_exists}."""
    log_name = log_pkl.stem
    log_sensor_dir = SENSOR_BLOBS / log_name
    result = {
        "log_name": log_name,
        "total_frames": 0,
        "bad_tokens": [],
        "good_tokens": [],
        "missing_jpgs_sample": [],
        "log_dir_exists": log_sensor_dir.is_dir(),
        "cam_dirs_missing": [],
    }
    if not log_sensor_dir.is_dir():
        # full log dir missing → all frames bad (but we still try to read pkl for token list)
        try:
            with open(log_pkl, "rb") as f:
                frames = pickle.load(f)
            result["total_frames"] = len(frames)
            result["bad_tokens"] = [fr["token"] for fr in frames]
        except Exception as e:
            result["error"] = f"pkl_load_failed: {e}"
        return result

    # readdir per CAM_* → set of basenames
    cam_existing: dict[str, set[str]] = {}
    for cam in CAM_NAMES:
        cam_dir = log_sensor_dir / cam
        if not cam_dir.is_dir():
            result["cam_dirs_missing"].append(cam)
            cam_existing[cam] = set()
        else:
            try:
                cam_existing[cam] = set(os.listdir(cam_dir))
            except OSError as e:
                result["cam_dirs_missing"].append(f"{cam}:{e}")
                cam_existing[cam] = set()

    # load pkl
    try:
        with open(log_pkl, "rb") as f:
            frames = pickle.load(f)
    except Exception as e:
        result["error"] = f"pkl_load_failed: {e}"
        return result

    result["total_frames"] = len(frames)

    for fr in frames:
        token = fr.get("token")
        if not token:
            continue
        cams = fr.get("cams") or {}
        missing = []
        for cam_name in CAM_NAMES:
            cam_data = cams.get(cam_name)
            if not cam_data:
                missing.append(f"{cam_name}:no_dict_entry")
                continue
            data_path = cam_data.get("data_path", "")
            # data_path = "<log_name>/CAM_X/<hash>.jpg" → basename
            basename = os.path.basename(data_path)
            if basename not in cam_existing.get(cam_name, set()):
                missing.append(f"{cam_name}/{basename}")
        if missing:
            result["bad_tokens"].append(token)
            if len(result["missing_jpgs_sample"]) < 3:
                result["missing_jpgs_sample"].append({"token": token, "missing": missing[:3]})
        else:
            result["good_tokens"].append(token)

    return result


def main():
    log_pkls = sorted(NAVSIM_LOGS.glob("*.pkl"))
    print(f"[scan] found {len(log_pkls)} log pkls", flush=True)
    t0 = time.time()

    all_good: list[str] = []
    all_bad: list[str] = []
    per_log_stats = []
    err_logs = []

    # ProcessPool with 16 workers
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(scan_one_log, lp): lp for lp in log_pkls}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                lp = futs[fut]
                err_logs.append({"log": lp.stem, "error": str(e)})
                continue
            all_good.extend(r["good_tokens"])
            all_bad.extend(r["bad_tokens"])
            per_log_stats.append({
                "log_name": r["log_name"],
                "total": r["total_frames"],
                "bad": len(r["bad_tokens"]),
                "good": len(r["good_tokens"]),
                "log_dir_exists": r["log_dir_exists"],
                "cam_dirs_missing": r["cam_dirs_missing"],
                "missing_jpgs_sample": r["missing_jpgs_sample"],
                "error": r.get("error"),
            })
            if done % 50 == 0 or done == len(log_pkls):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(log_pkls) - done) / max(rate, 1e-6)
                print(f"[scan] {done}/{len(log_pkls)}  good={len(all_good)} bad={len(all_bad)}  "
                      f"elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n[scan] DONE in {elapsed:.1f}s", flush=True)
    print(f"[scan] total good tokens: {len(all_good)}", flush=True)
    print(f"[scan] total bad  tokens: {len(all_bad)}", flush=True)
    print(f"[scan] logs with bad>0: {sum(1 for s in per_log_stats if s['bad']>0)}", flush=True)
    print(f"[scan] logs with whole dir missing: {sum(1 for s in per_log_stats if not s['log_dir_exists'])}", flush=True)
    print(f"[scan] logs with pkl_load_failed: {sum(1 for s in per_log_stats if s.get('error'))}", flush=True)

    # write clean tokens (sorted)
    all_good_sorted = sorted(set(all_good))
    with open(CLEAN_TOKENS, "w") as f:
        for tok in all_good_sorted:
            f.write(tok + "\n")
    print(f"[scan] wrote {len(all_good_sorted)} clean tokens -> {CLEAN_TOKENS}", flush=True)

    # write report
    report = {
        "scan_time_sec": elapsed,
        "num_logs": len(log_pkls),
        "total_good_tokens": len(all_good_sorted),
        "total_bad_tokens": len(set(all_bad)),
        "logs_with_bad": sum(1 for s in per_log_stats if s["bad"] > 0),
        "logs_dir_missing": sum(1 for s in per_log_stats if not s["log_dir_exists"]),
        "logs_pkl_failed": sum(1 for s in per_log_stats if s.get("error")),
        "err_logs": err_logs,
        "per_log": per_log_stats,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[scan] wrote report -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()

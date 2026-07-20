#!/usr/bin/env python
"""
For each navtrain.yaml target token, check its temporal window (history_frames=4, future_frames=10).
A target token is USABLE iff ALL frames within [idx-4, idx+10] (inclusive) have ALL 8 cams' jpgs on disk.

Uses the readdir-cached good-jpg-set from scan_navtrain_missing_images.py output.
"""
from __future__ import annotations
import os, json, pickle, time, yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

NAVSIM_LOGS = Path("/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/navsim_logs/trainval")
SENSOR_BLOBS = Path("/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/trainval")
NAVTRAIN_YAML = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
CAM_NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

# These MUST match the SceneFilter in navtrain.yaml
NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 10
# Window: [idx - NUM_HISTORY_FRAMES, idx + NUM_FUTURE_FRAMES]
WINDOW_BACK = NUM_HISTORY_FRAMES
WINDOW_FWD = NUM_FUTURE_FRAMES

OUT_DIR = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_navtrain_probeA_setup")


def scan_one_log(log_pkl_path_str: str, target_token_set_serial: list) -> dict:
    """Per-log: load pkl, build {token -> idx}, build per-cam jpg-existence set, check each target token's window."""
    target_set = set(target_token_set_serial)
    log_pkl = Path(log_pkl_path_str)
    log_name = log_pkl.stem
    log_sensor_dir = SENSOR_BLOBS / log_name
    r = {"log_name": log_name, "log_dir_exists": log_sensor_dir.is_dir(),
         "usable_tokens": [], "unusable_tokens": [], "unusable_reasons_sample": []}

    if not log_sensor_dir.is_dir():
        # Need to mark all target tokens in this log as unusable
        try:
            with open(log_pkl,"rb") as f:
                frames = pickle.load(f)
        except Exception as e:
            r["error"] = f"pkl_load_failed: {e}"
            return r
        for fr in frames:
            t = fr.get("token")
            if t in target_set:
                r["unusable_tokens"].append(t)
        return r

    # cam jpg existence sets
    cam_existing = {}
    for cam in CAM_NAMES:
        cam_dir = log_sensor_dir / cam
        try:
            cam_existing[cam] = set(os.listdir(cam_dir)) if cam_dir.is_dir() else set()
        except OSError:
            cam_existing[cam] = set()

    try:
        with open(log_pkl,"rb") as f:
            frames = pickle.load(f)
    except Exception as e:
        r["error"] = f"pkl_load_failed: {e}"
        return r

    n = len(frames)
    # For each frame in this log, precompute its "all cams ok" boolean
    frame_ok = [False]*n
    for i, fr in enumerate(frames):
        cams = fr.get("cams") or {}
        ok = True
        for cam in CAM_NAMES:
            cd = cams.get(cam)
            if not cd:
                ok = False; break
            base = os.path.basename(cd.get("data_path",""))
            if base not in cam_existing.get(cam, set()):
                ok = False; break
        frame_ok[i] = ok

    # iterate each frame; if it's a target token, check window
    for i, fr in enumerate(frames):
        t = fr.get("token")
        if t not in target_set:
            continue
        lo = max(0, i - WINDOW_BACK)
        hi = min(n-1, i + WINDOW_FWD)
        # require window contains both ends (otherwise navsim's history loader may fail)
        window_ok = (i - WINDOW_BACK >= 0) and (i + WINDOW_FWD <= n-1)
        if window_ok:
            for j in range(lo, hi+1):
                if not frame_ok[j]:
                    window_ok = False
                    bad_j = j
                    break
        if window_ok:
            r["usable_tokens"].append(t)
        else:
            r["unusable_tokens"].append(t)
            if len(r["unusable_reasons_sample"]) < 3:
                r["unusable_reasons_sample"].append({"token": t, "frame_idx": i, "reason": "window_incomplete_or_oob"})

    return r


def main():
    # 1) load navtrain.yaml target tokens
    with open(NAVTRAIN_YAML) as f:
        cfg = yaml.safe_load(f)
    target_tokens = set(cfg.get("tokens") or [])
    print(f"[scan2] navtrain target tokens: {len(target_tokens)}")
    print(f"[scan2] window: [-{WINDOW_BACK}, +{WINDOW_FWD}]")

    # 2) which logs do these tokens span? Use log_names from yaml if present, else scan all
    log_names = cfg.get("log_names")
    if log_names:
        log_pkls = [NAVSIM_LOGS / f"{ln}.pkl" for ln in log_names]
        log_pkls = [p for p in log_pkls if p.exists()]
        print(f"[scan2] using yaml log_names: {len(log_pkls)} pkls")
    else:
        log_pkls = sorted(NAVSIM_LOGS.glob("*.pkl"))
        print(f"[scan2] scanning all {len(log_pkls)} pkls")

    target_list = list(target_tokens)
    t0 = time.time()

    all_usable, all_unusable = [], []
    per_log = []
    err_logs = []

    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(scan_one_log, str(p), target_list): p for p in log_pkls}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                err_logs.append({"log": str(futs[fut]), "error": str(e)})
                continue
            all_usable.extend(r["usable_tokens"])
            all_unusable.extend(r["unusable_tokens"])
            per_log.append({
                "log_name": r["log_name"],
                "usable": len(r["usable_tokens"]),
                "unusable": len(r["unusable_tokens"]),
                "log_dir_exists": r["log_dir_exists"],
                "unusable_reasons_sample": r["unusable_reasons_sample"],
                "error": r.get("error"),
            })
            if done % 50 == 0 or done == len(log_pkls):
                elapsed = time.time()-t0
                rate = done/max(elapsed,1e-6)
                eta = (len(log_pkls)-done)/max(rate,1e-6)
                print(f"[scan2] {done}/{len(log_pkls)}  usable={len(all_usable)} unusable={len(all_unusable)} "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    elapsed = time.time()-t0
    print(f"\n[scan2] DONE in {elapsed:.1f}s")
    print(f"[scan2] usable:   {len(all_usable)}")
    print(f"[scan2] unusable: {len(all_unusable)}")
    print(f"[scan2] total target: {len(target_tokens)}  matched: {len(all_usable)+len(all_unusable)}")

    usable_sorted = sorted(set(all_usable))
    out_clean = OUT_DIR / "navtrain_window_clean_tokens.txt"
    with open(out_clean,"w") as f:
        for t in usable_sorted: f.write(t+"\n")
    print(f"[scan2] wrote {len(usable_sorted)} usable -> {out_clean}")

    report = {
        "scan_time_sec": elapsed,
        "target_total": len(target_tokens),
        "usable": len(usable_sorted),
        "unusable": len(set(all_unusable)),
        "window": [WINDOW_BACK, WINDOW_FWD],
        "logs_dir_missing": sum(1 for s in per_log if not s["log_dir_exists"]),
        "err_logs": err_logs,
        "per_log": per_log,
    }
    out_report = OUT_DIR / "navtrain_window_report.json"
    with open(out_report,"w") as f: json.dump(report, f, indent=2)
    print(f"[scan2] wrote report -> {out_report}")


if __name__ == "__main__":
    main()

"""
M1.b2 stage 2 prep — scan navtrain trigger tokens to find ones with the FULL
context window (history+future+current frame) of jpgs downloaded.

navsim Scene loading requires: num_history_frames=4 + num_future_frames=10 +
current = 15 frames per scene, 8 cams each. If ANY of the 15×8=120 jpgs is
missing, Scene.from_scene_dict_list raises FileNotFoundError.

We restrict to (a) trigger token ∈ navtrain.yaml's 103,288 set AND
(b) all 15-frame neighbors have CAM_F0 jpg on disk
(verified earlier that 8 cams are downloaded together — checking CAM_F0
is a faithful proxy for all 8).

Output: one valid trigger token per line.

Usage:
  python tools/scan_navtrain_full_window.py \
    --log-dir /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/navsim_logs/trainval \
    --sb-dir /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/trainval \
    --navtrain-yaml code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml \
    --out exp/m1b2_navtrain_full_window_tokens.txt \
    --workers 32 \
    --num-history 4 --num-future 10
"""
import argparse
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_navtrain_tokens(path):
    s = set()
    pat = re.compile(r"^\s*-\s*'([a-f0-9]+)'\s*$")
    with open(path) as f:
        for ln in f:
            m = pat.match(ln)
            if m:
                s.add(m.group(1))
    return s


def scan_log(args_tuple):
    log_path, sb_dir, nav_token_set, num_hist, num_fut = args_tuple
    log_stem = os.path.basename(log_path).replace(".pkl", "")
    cam_dir = os.path.join(sb_dir, log_stem, "CAM_F0")
    if not os.path.isdir(cam_dir):
        return []
    try:
        present = set(os.listdir(cam_dir))
    except OSError:
        return []
    try:
        with open(log_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    # build basename for each frame
    bases = []
    for s in data:
        b = None
        if isinstance(s, dict):
            cf = s["cams"].get("CAM_F0") if isinstance(s.get("cams"), dict) else None
            if isinstance(cf, dict):
                b = os.path.basename(cf.get("data_path", ""))
        bases.append(b)

    valid = []
    for i, s in enumerate(data):
        if not isinstance(s, dict):
            continue
        tok = s.get("token")
        if not tok or tok not in nav_token_set:
            continue
        ok = True
        # check frames i-num_hist .. i+num_fut inclusive
        for off in range(-num_hist, num_fut + 1):
            j = i + off
            if j < 0 or j >= len(data):
                ok = False
                break
            b = bases[j]
            if not b or b not in present:
                ok = False
                break
        if ok:
            valid.append(tok)
    return valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--sb-dir", required=True)
    p.add_argument("--navtrain-yaml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--num-history", type=int, default=4)
    p.add_argument("--num-future", type=int, default=10)
    args = p.parse_args()

    nav_set = load_navtrain_tokens(args.navtrain_yaml)
    print(f"[scan] navtrain.yaml tokens: {len(nav_set)}", flush=True)

    log_files = sorted(
        os.path.join(args.log_dir, f)
        for f in os.listdir(args.log_dir)
        if f.endswith(".pkl")
    )
    print(f"[scan] {len(log_files)} logs, workers={args.workers} window=±{args.num_history}/+{args.num_future}", flush=True)

    valid = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(scan_log, (lp, args.sb_dir, nav_set, args.num_history, args.num_future))
            for lp in log_files
        ]
        for fut in as_completed(futs):
            r = fut.result()
            valid.extend(r)
            done += 1
            if done % 100 == 0 or done == len(log_files):
                print(f"[scan] {done}/{len(log_files)} logs, valid running = {len(valid)}", flush=True)
    valid = sorted(set(valid))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for t in valid:
            f.write(t + "\n")
    print(f"[scan] DONE: {len(valid)}/{len(nav_set)} valid ({len(valid)/len(nav_set)*100:.1f}%) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

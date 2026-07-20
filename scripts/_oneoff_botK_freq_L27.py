"""
One-off: compute L27 bot-4 head frequency from the 24GB navtrain attention dump,
using the IDENTICAL methodology as _oneoff_botK_freq_alllayers.py (which only
captured layers {0,4,8,12,16,20,24} — L27 was missing).

Per scene: mean = attn[27].mean(dim=-1) (per-head mean over 720 vision tokens);
bot_idx = topk(mean, k=4, largest=False); head_freq = fraction in bot-4 across scenes;
const_topK = top-4 heads by freq. This gives the L27 bot-4 mask for the
layer×prunability landscape clean point (path A, 2026-07-01).

Output: exp/m1b2_phase2_v0/botK_freq_L27.json
Run (CPU/IO only, no GPU):
  /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
      scripts/_oneoff_botK_freq_L27.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

REPO = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
DUMP_DIR = REPO / "exp" / "m1b2_navtrain_full_alllayers"
OUT_PATH = REPO / "exp" / "m1b2_phase2_v0" / "botK_freq_L27.json"
L = 27
K = 4
N_HEADS = 16


def main() -> int:
    files = sorted(DUMP_DIR.glob("*.pt"))
    n = len(files)
    print(f"[scan] n_files={n} layer=L{L} K={K}", flush=True)
    y = torch.zeros((n, N_HEADS), dtype=torch.bool)
    n_ok = 0
    skipped = 0
    t0 = time.time()
    for idx, p in enumerate(files):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            skipped += 1
            continue
        attn = d.get("per_layer_vision_attn", None)
        if (attn is None or attn.dim() != 3 or attn.shape[0] <= L
                or attn.shape[1] != N_HEADS):
            skipped += 1
            continue
        mean = attn[L].mean(dim=-1)  # (16,)
        bot_idx = torch.topk(mean, k=K, largest=False).indices
        y[n_ok, bot_idx] = True
        n_ok += 1
        if (idx + 1) % 2000 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"[scan] {idx+1}/{n} ({rate:.1f} f/s) n_ok={n_ok} skipped={skipped}",
                  flush=True)

    wall = time.time() - t0
    y = y[:n_ok]
    head_freq = y.float().mean(dim=0)
    const_topK = torch.topk(head_freq, k=K).indices
    pred = torch.zeros_like(y)
    pred[:, const_topK] = True
    tp = (pred & y).sum(dim=0).float()
    fp = (pred & ~y).sum(dim=0).float()
    fn = (~pred & y).sum(dim=0).float()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = (2 * prec * rec / (prec + rec + 1e-9)).mean().item()
    em = (pred == y).all(dim=1).float().mean().item()
    topK_freqs = head_freq[const_topK].tolist()

    out = {
        "n_ok": n_ok, "skipped": skipped, "K": K, "n_heads": N_HEADS, "layer": L,
        "wall_seconds": round(wall, 1),
        "head_freq": [round(v, 4) for v in head_freq.tolist()],
        "const_topK_head_idxs": const_topK.tolist(),
        "const_topK_freqs": [round(v, 4) for v in topK_freqs],
        "const_macro_f1": round(f1, 6),
        "const_exact_match": round(em, 6),
        "topK_mass": round(float(sum(topK_freqs)), 4),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[scan] DONE n_ok={n_ok} skipped={skipped} wall={wall:.1f}s", flush=True)
    print(f"[scan] L27 bot-4 (const_topK) = {sorted(const_topK.tolist())} "
          f"freqs={[round(v,3) for v in topK_freqs]}", flush=True)
    print(f"[scan] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

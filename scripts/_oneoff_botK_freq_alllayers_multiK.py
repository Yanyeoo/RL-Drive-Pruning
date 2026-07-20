"""
One-off probe (multi-K): For each layer L in LAYERS and each K in KS, scan all
token .pt dumps and compute bot-K head frequency + const-baseline macro-F1 / EM.

Purpose
-------
Extends the K=4 scan (see _oneoff_botK_freq_alllayers.py + botK_freq_alllayers.json)
to K in {2,3,5}. Single IO pass, ~37s. Used to judge:
- whether L24 head ordering [h9,h10,h7,...] is stable across K (V4 mask validity);
- whether the marginal head added going from K=K* to K=K*+1 has a sharp freq
  drop (signal that further mask escalation is unjustified).

Output
------
exp/m1b2_phase2_v0/botK_freq_alllayers_multiK.json (per (K,L) entry)
+ stdout table per K.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


REPO = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
DUMP_DIR = REPO / "exp" / "m1b2_navtrain_full_alllayers"
OUT_PATH = REPO / "exp" / "m1b2_phase2_v0" / "botK_freq_alllayers_multiK.json"

LAYERS = (0, 4, 8, 12, 16, 20, 24)
KS = (2, 3, 4, 5)
N_HEADS = 16


def main() -> int:
    files = sorted(DUMP_DIR.glob("*.pt"))
    n = len(files)
    print(f"[scan] n_files = {n}")
    print(f"[scan] layers  = {LAYERS}")
    print(f"[scan] Ks      = {KS}")

    # y_botK[K][L]: (n, 16) bool
    y_botK = {K: {L: torch.zeros((n, N_HEADS), dtype=torch.bool) for L in LAYERS}
              for K in KS}
    n_ok = 0
    skipped = 0
    t0 = time.time()

    max_L = max(LAYERS)
    for idx, p in enumerate(files):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            skipped += 1
            continue
        attn = d.get("per_layer_vision_attn", None)
        if (attn is None or attn.dim() != 3 or attn.shape[0] <= max_L
                or attn.shape[1] != N_HEADS):
            skipped += 1
            continue
        # mean over vision tokens once per layer; reuse for all K
        for L in LAYERS:
            mean = attn[L].mean(dim=-1)  # (16,)
            # Pre-sort ascending once, then slice per K
            sorted_idx = torch.argsort(mean)
            for K in KS:
                bot_idx = sorted_idx[:K]
                y_botK[K][L][n_ok, bot_idx] = True
        n_ok += 1
        if (idx + 1) % 2000 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"[scan] {idx+1}/{n}  ({rate:.1f} files/s)  n_ok={n_ok}  skipped={skipped}")

    wall = time.time() - t0
    print(f"[scan] DONE  n_ok={n_ok}  skipped={skipped}  wall={wall:.1f}s")

    out: dict = {"n_ok": n_ok, "skipped": skipped, "Ks": list(KS),
                 "n_heads": N_HEADS, "layers": list(LAYERS),
                 "wall_seconds": round(wall, 1), "per_K": {}}

    for K in KS:
        print()
        print(f"=== K = {K} ===")
        print(f"{'L':>3} | {'top-K (by freq)':<28} | top-K freqs                    "
              f"| F1     EM     | mass(topK)/K")
        print("-" * 110)
        per_layer: dict = {}
        for L in LAYERS:
            y = y_botK[K][L][:n_ok]                                  # (n_ok, 16)
            head_freq = y.float().mean(dim=0)                        # (16,)
            const_topK = torch.topk(head_freq, k=K).indices          # (K,)
            pred = torch.zeros_like(y)
            pred[:, const_topK] = True
            tp = (pred & y).sum(dim=0).float()
            fp = (pred & ~y).sum(dim=0).float()
            fn = (~pred & y).sum(dim=0).float()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            macro_f1 = f1.mean().item()
            em = (pred == y).all(dim=1).float().mean().item()

            topK_freqs = head_freq[const_topK].tolist()
            mass = float(sum(topK_freqs))

            per_layer[L] = {
                "head_freq": [round(v, 4) for v in head_freq.tolist()],
                "const_topK_head_idxs": const_topK.tolist(),
                "const_topK_freqs": [round(v, 4) for v in topK_freqs],
                "const_macro_f1": round(macro_f1, 6),
                "const_exact_match": round(em, 6),
                "topK_mass": round(mass, 4),
            }

            topK_str = " ".join(f"h{h:<2}" for h in const_topK.tolist())
            freqs_str = " ".join(f"{v:.3f}" for v in topK_freqs)
            print(f"L{L:<2}  | {topK_str:<28} | {freqs_str:<30} "
                  f"| {macro_f1:.4f} {em:.4f} | {mass:.3f}/{float(K):.1f}")
        out["per_K"][K] = per_layer

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"[scan] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

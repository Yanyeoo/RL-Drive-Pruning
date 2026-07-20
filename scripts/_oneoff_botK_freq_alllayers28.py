"""
Full 28-layer bot-K head-frequency scan (K in {4,6,8}), same methodology as
_oneoff_botK_freq_alllayers.py but for ALL decoder layers 0..27 and multiple K.
Enables generic L{N}K{K} variants (fine-grained layer×prunability landscape,
cliff localization) without hand-picking heads.

Per scene: mean = attn[L].mean(-1); bot-K = K lowest-mean heads.
head_freq[L,K][h] = fraction of scenes h is in bot-K; const_topK = top-K by freq.

Output: exp/m1b2_phase2_v0/botK_freq_alllayers28.json
  { "n_ok":..., "per_layer": { "<L>": { "<K>": [sorted top-K head idxs], ... } } }
Run (CPU/IO, no GPU): <autovla-python> scripts/_oneoff_botK_freq_alllayers28.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import torch

REPO = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
DUMP_DIR = REPO / "exp" / "m1b2_navtrain_full_alllayers"
OUT = REPO / "exp" / "m1b2_phase2_v0" / "botK_freq_alllayers28.json"
LAYERS = list(range(28))
KS = (4, 6, 8)
N_HEADS = 16


def main() -> int:
    files = sorted(DUMP_DIR.glob("*.pt"))
    n = len(files)
    print(f"[scan28] n_files={n} layers=0..27 K={KS}", flush=True)
    # bottom-rank counts: for each layer, count[h] over scenes for each K
    # We store, per layer, the ascending-rank position tallies via bot-K membership.
    cnt = {L: {K: torch.zeros(N_HEADS) for K in KS} for L in LAYERS}
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
        if attn is None or attn.dim() != 3 or attn.shape[0] < 28 or attn.shape[1] != N_HEADS:
            skipped += 1
            continue
        means = attn.mean(dim=-1)  # (28,16)
        for L in LAYERS:
            order = torch.argsort(means[L])  # ascending
            for K in KS:
                cnt[L][K][order[:K]] += 1
        n_ok += 1
        if (idx + 1) % 2000 == 0:
            r = (idx + 1) / (time.time() - t0)
            print(f"[scan28] {idx+1}/{n} ({r:.1f} f/s) n_ok={n_ok} skipped={skipped}", flush=True)
    wall = time.time() - t0
    per_layer = {}
    for L in LAYERS:
        per_layer[str(L)] = {}
        for K in KS:
            freq = cnt[L][K] / max(n_ok, 1)
            topK = torch.topk(freq, k=K).indices.tolist()
            per_layer[str(L)][str(K)] = sorted(topK)
    out = {"n_ok": n_ok, "skipped": skipped, "n_heads": N_HEADS,
           "layers": LAYERS, "Ks": list(KS), "wall_seconds": round(wall, 1),
           "per_layer": per_layer}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"[scan28] DONE n_ok={n_ok} wall={wall:.1f}s -> {OUT}", flush=True)
    # sanity: print cliff-region layers
    for L in (0, 4, 8, 12, 16, 20, 22, 24, 25, 26, 27):
        print(f"  L{L}: K4={per_layer[str(L)]['4']} K6={per_layer[str(L)]['6']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

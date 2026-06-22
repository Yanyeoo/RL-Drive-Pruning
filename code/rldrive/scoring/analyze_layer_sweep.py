"""analyze_layer_sweep.py — pick L* from M1.a layer sweep outputs.

Loads `.pt` files from each layer dir, computes per-scene
vision-attention fraction = vision_attn.sum() (since vision_attn is already
the head-averaged attention sliced to vision_token_positions, summed across
those positions equals fraction of attention going to vision tokens out of
the full attention row 1.0). Aggregates mean ± std per layer and picks
L* = argmax.

Usage:
    python -m rldrive.scoring.analyze_layer_sweep \\
        --sweep-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644

Output:
  - prints per-layer table
  - writes <sweep_dir>/layer_sweep_summary.json with the same numbers
  - prints the chosen L*
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(
            "/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644"
        ),
    )
    p.add_argument(
        "--layers",
        type=str,
        default="0,4,8,12,16,20,24,27",
        help="comma-separated list of layer indices to analyze",
    )
    return p.parse_args()


def analyze_layer(layer_dir: Path) -> Dict:
    fracs: List[float] = []
    n_vision_set: List[int] = []
    bad: List[str] = []
    for pt in sorted(layer_dir.glob("*.pt")):
        try:
            d = torch.load(pt, map_location="cpu", weights_only=False)
            va = d["vision_attn"]
            if not isinstance(va, torch.Tensor):
                bad.append(pt.name)
                continue
            # vision_attn is the head-mean attention (q=last_instr) sliced to
            # vision_token_positions. vision_attn.sum() is the share of
            # attention mass going to vision tokens (in [0, 1]).
            frac = float(va.float().sum().item())
            fracs.append(frac)
            n_vision_set.append(int(va.numel()))
        except Exception as e:
            bad.append(f"{pt.name}:{type(e).__name__}")
    if not fracs:
        return {"n_scenes": 0, "mean": math.nan, "std": math.nan, "min": math.nan,
                "max": math.nan, "n_vision": None, "bad": bad}
    t = torch.tensor(fracs, dtype=torch.float64)
    return {
        "n_scenes": len(fracs),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if len(fracs) > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "n_vision": list(set(n_vision_set))[:5],
        "bad": bad,
    }


def main() -> int:
    args = parse_args()
    sweep_dir: Path = args.sweep_dir
    if not sweep_dir.is_dir():
        print(f"ERROR: sweep dir not found: {sweep_dir}")
        return 2

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"[analyze] sweep_dir = {sweep_dir}")
    print(f"[analyze] layers    = {layers}")
    print()
    print(f"{'layer':>5}  {'n':>4}  {'vision_frac_mean':>18}  {'std':>8}  "
          f"{'min':>8}  {'max':>8}  {'n_vision':>10}  bad")
    print("-" * 95)

    results: Dict[int, Dict] = {}
    for L in layers:
        layer_dir = sweep_dir / f"L{L:02d}"
        if not layer_dir.is_dir():
            print(f"L{L:02d}  MISSING DIR")
            continue
        r = analyze_layer(layer_dir)
        results[L] = r
        print(
            f"{L:>5}  {r['n_scenes']:>4}  {r['mean']:>18.4f}  {r['std']:>8.4f}  "
            f"{r['min']:>8.4f}  {r['max']:>8.4f}  {str(r['n_vision']):>10}  "
            f"{len(r['bad'])} bad"
        )

    valid = {L: r for L, r in results.items() if r["n_scenes"] > 0}
    if not valid:
        print("\nNo valid results.")
        return 3

    L_star = max(valid, key=lambda L: valid[L]["mean"])
    print()
    print(f"=> L* = layer {L_star} (vision_frac_mean = {valid[L_star]['mean']:.4f})")

    out_path = sweep_dir / "layer_sweep_summary.json"
    with out_path.open("w") as f:
        json.dump(
            {"sweep_dir": str(sweep_dir), "layers": layers,
             "results": {str(L): r for L, r in results.items()},
             "L_star": L_star,
             "L_star_vision_frac_mean": valid[L_star]["mean"]},
            f, indent=2,
        )
    print(f"\nWrote summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

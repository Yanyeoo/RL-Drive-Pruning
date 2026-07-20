#!/usr/bin/env python3
"""S2 headroom-gate oracle post-processing (no GPU).

Reads per-scene CSVs from results/raw/tokenprune_S2/ produced by
run_tokenprune_sweep.sh, computes the S2 gate metrics
(docs/specs/dynamic_headroom_gate_S2_spec.md §2-§3) and prints a verdict.

Arms expected (exp_name -> (selector, keep_ratio)):
  S2sub200_attnL12_r100 -> attn 1.0
  S2sub200_attnL12_r075 -> attn 0.75
  S2sub200_attnL12_r050 -> attn 0.5
  S2sub200_attnL12_r025 -> attn 0.25
  S2sub200_random_r050  -> random 0.5

PDMS = the `score` column. "pt" = PDMS * 100.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RAW = "/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/tokenprune_S2"

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "S2sub200"
if PREFIX == "COMBINED":
    PREFIXES = ["S2sub1500", "S2sub1500b"]   # first-1500 + 2nd-half = full shard0
else:
    PREFIXES = [PREFIX]

ARMS = {
    "attnL12_r100": ("attn", 1.00),
    "attnL12_r075": ("attn", 0.75),
    "attnL12_r050": ("attn", 0.50),
    "attnL12_r025": ("attn", 0.25),
    "random_r050":  ("random", 0.50),
}
R_GRID = [0.25, 0.50, 0.75, 1.00]     # attn arms + ref
EPS = [0.005, 0.01, 0.02]
PRIMARY_EPS = 0.01
PT = 100.0


def load_arm(exp):
    dfs = []
    for p in PREFIXES:
        path = os.path.join(RAW, f"{p}_{exp}.csv")
        if not os.path.isfile(path):
            continue
        d = pd.read_csv(path)
        d = d[d["token"] != "average"].copy()
        if "valid" in d:
            d = d[d["valid"] == True]  # noqa: E712
        dfs.append(d[["token", "score"]])
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("token")
    return df.rename(columns={"score": exp})


def main():
    frames = {}
    for exp in ARMS:
        f = load_arm(exp)
        n = 0 if f is None else len(f)
        print(f"[load] {exp:28s} valid_scenes={n}")
        if f is not None and n > 0:
            frames[exp] = f

    attn_exps = {r: e for e, (s, r) in ARMS.items() if s == "attn"}
    missing = [e for e in ARMS if e not in frames]
    if missing:
        print(f"[warn] arms missing/empty (not finished?): {missing}")

    # per-scene wide table on the intersection of scenes valid in ALL loaded arms
    merged = None
    for exp, f in frames.items():
        merged = f if merged is None else merged.merge(f, on="token", how="inner")
    if merged is None or len(merged) == 0:
        print("[abort] no common scenes yet.")
        return 1
    N = len(merged)
    print(f"\n[common valid scenes across all loaded arms] N={N}\n")

    # ---- per-arm mean PDMS ----
    print("=== per-arm mean PDMS (on common N) ===")
    arm_mean = {}
    for exp in frames:
        m = merged[exp].mean()
        arm_mean[exp] = m
        s, r = ARMS[exp]
        print(f"  {s:6s} r={r:<4} : PDMS={m:.6f} ({m*PT:.3f} pt)")

    # ---- Pareto (attn) vs r + r=0.5 drop vs r=1.0 ----
    print("\n=== Pareto (attn_L12): PDMS vs r ===")
    attn_avail = {r: e for r, e in attn_exps.items() if e in frames}
    for r in sorted(attn_avail):
        e = attn_avail[r]
        print(f"  r={r:<4}: {arm_mean[e]:.6f} ({arm_mean[e]*PT:.3f} pt)")

    drop_05 = gain_sel = None
    if 1.00 in attn_avail and 0.50 in attn_avail:
        p10 = arm_mean[attn_avail[1.00]]
        p05 = arm_mean[attn_avail[0.50]]
        drop_05 = p10 - p05
        print(f"\n  P-attn(r=0.5) drop vs r=1.0 : {drop_05:+.6f} ({drop_05*PT:+.3f} pt)")
    if 0.50 in attn_avail and "random_r050" in frames:
        pa = arm_mean[attn_avail[0.50]]
        pr = arm_mean["random_r050"]
        gain_sel = pa - pr
        print(f"  selector gain @r=0.5 (attn - random): {gain_sel:+.6f} ({gain_sel*PT:+.3f} pt)")

    # ---- per-scene oracle over attn r-grid (+ ref r=1.0) ----
    have_full_grid = all(r in attn_avail for r in R_GRID)
    if not have_full_grid:
        print(f"\n[oracle] need full attn r-grid {R_GRID}; have {sorted(attn_avail)}. "
              f"Skipping oracle until all attn arms finish.")
        return 0

    cols_by_r = {r: attn_avail[r] for r in R_GRID}
    M = merged[[cols_by_r[r] for r in R_GRID]].to_numpy()  # (N, 4) in R_GRID order
    rgrid = np.array(R_GRID)
    max_r = M.max(axis=1)                       # per-scene best PDMS
    oracle_ceiling = max_r.mean()               # ε=0 perfect selection
    best_fixed = max(arm_mean[cols_by_r[r]] for r in R_GRID)
    ceiling_gain = oracle_ceiling - best_fixed
    best_fixed_r = max(R_GRID, key=lambda r: arm_mean[cols_by_r[r]])

    print("\n=== per-scene oracle ===")
    print(f"  best fixed-r EPDMS : {best_fixed:.6f} (at r={best_fixed_r})")
    print(f"  oracle ceiling (ε=0): {oracle_ceiling:.6f} ({oracle_ceiling*PT:.3f} pt)")
    print(f"  ceiling gain over best fixed-r: {ceiling_gain:+.6f} ({ceiling_gain*PT:+.3f} pt)")

    print("\n=== r* histogram (min r within ε of per-scene max) ===")
    rstar_primary = None
    for eps in EPS:
        thresh = max_r - eps
        ge = M >= thresh[:, None]               # (N,4) which r are within ε
        # smallest r index that is within ε
        idx = ge.argmax(axis=1)                 # first True along R_GRID (ascending r)
        rstar = rgrid[idx]
        hist = {r: int((rstar == r).sum()) for r in R_GRID}
        frac_lt1 = float((rstar < 1.0).mean())
        nuniq = len(set(rstar.tolist()))
        tag = " (PRIMARY)" if eps == PRIMARY_EPS else ""
        print(f"  ε={eps:<5}: " + "  ".join(f"r={r}:{hist[r]:3d}" for r in R_GRID)
              + f"   frac(r*<1.0)={frac_lt1:.1%}  n_unique={nuniq}{tag}")
        if eps == PRIMARY_EPS:
            rstar_primary = rstar

    # ---- ε-oracle policy realized PDMS + mean compute (the iso-compute win) ----
    thr = max_r - PRIMARY_EPS
    ge = M >= thr[:, None]
    idxp = ge.argmax(axis=1)
    achieved = M[np.arange(len(M)), idxp]          # per-scene PDMS at chosen r*
    eps_oracle_pdms = achieved.mean()
    eps_oracle_keep = rgrid[idxp].mean()           # avg keep-ratio (compute proxy)
    pdms_r05 = arm_mean[cols_by_r[0.50]]
    print("\n=== ε=0.01 oracle policy (per-scene smallest r within ε) ===")
    print(f"  realized PDMS   : {eps_oracle_pdms:.6f} ({eps_oracle_pdms*PT:.3f} pt)")
    print(f"  mean keep-ratio : {eps_oracle_keep:.4f}  (avg compute; fixed iso-compute=0.5)")
    print(f"  vs r=1.0 ({arm_mean[cols_by_r[1.00]]*PT:.3f}pt): {(eps_oracle_pdms-arm_mean[cols_by_r[1.00]])*PT:+.3f} pt")
    print(f"  vs fixed r=0.5 ({pdms_r05*PT:.3f}pt): {(eps_oracle_pdms-pdms_r05)*PT:+.3f} pt "
          f"at {eps_oracle_keep:.2f} vs 0.50 keep")

    # ---- GATE verdict (§3) ----
    print("\n=== GATE verdict (spec §3) ===")
    DROP_BAR = 0.005    # 0.5 pt
    GAIN_BAR = 0.005    # 0.5 pt
    frac_lt1_p = float((rstar_primary < 1.0).mean())
    nuniq_p = len(set(rstar_primary.tolist()))
    cond_headroom = (drop_05 is not None) and (drop_05 <= DROP_BAR)
    cond_ceiling = ceiling_gain >= GAIN_BAR
    cond_variance = nuniq_p >= 2 and frac_lt1_p >= 0.20
    print(f"  cond1 headroom  P-attn(0.5) drop {drop_05*PT:+.3f}pt ≤ 0.5pt : {cond_headroom}")
    print(f"  cond2 ceiling   gain {ceiling_gain*PT:+.3f}pt ≥ 0.5pt        : {cond_ceiling}")
    print(f"  cond3 variance  n_unique(r*)={nuniq_p}≥2 & frac(r*<1)={frac_lt1_p:.1%}≥20% : {cond_variance}")

    if drop_05 is not None and drop_05 > 0.02:
        verdict = "FAIL -> stop (token pruning itself hurts PDMS, R-D-3)"
    elif cond_headroom and cond_ceiling and cond_variance:
        verdict = "PASS -> build S3"
    elif cond_headroom and not cond_variance:
        verdict = "PARTIAL -> fixed-ratio+better-selector; budget policy not justified"
    else:
        verdict = "INCONCLUSIVE/PARTIAL -> review numbers (see conditions above)"
    print(f"\n  >>> VERDICT: {verdict}\n")
    print(f"  NOTE: computed on N={N} scenes (prefix={PREFIX}); full navtest 4-shard is n≈11574.")
    print("        Expand scenes if borderline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

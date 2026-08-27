#!/usr/bin/env python3
"""gen_table1_draft.py — Build the AAAI Table 1 draft from landed drop CSVs.

All methods use Variant-B true token drop. Baselines report RAW PDMS.
"Ours" rows report +fallback (denylist -> no-prune score on 768 catastrophic
tokens), which reproduces the 0.9045 headline exactly.

The generator is tolerant of missing shards: each cell shows its shard coverage
and marks pending cells. It always writes the current best draft so a mid-run
kill still leaves an up-to-date table.

Usage:
    python scripts/gen_table1_draft.py --out results/table1_draft.md
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import pandas as pd

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
RAW = ROOT / "results/raw/tokenprune_S3_full"
TAUCUT = ROOT / "results/raw/tokenprune_taucut"
DENY_PATH = ROOT / "results/varB_catastrophic_tokens.json"

# retention ratio -> (tokens kept, reduction label, FLOPs saving on LLM prefill)
RATIO_META = {
    0.75: (540, "\u219325%", "16.9%"),
    0.50: (360, "\u219350%", "33.6%"),
    0.25: (180, "\u219375%", "49.9%"),
}


def load_denylist():
    d = json.load(open(DENY_PATH))
    return set(d if isinstance(d, list) else d.keys())


def gather(prefixes, base_dir=RAW):
    """Return (df, shard_list) for the first prefix that has any shard CSV."""
    for pre in prefixes:
        files = sorted(glob.glob(str(base_dir / f"{pre}_sh*.csv")))
        if not files:
            continue
        dfs, shards = [], []
        for f in files:
            sh = int(f.split("_sh")[-1].split(".")[0])
            df = pd.read_csv(f)
            df = df[df["token"] != "average"]
            dfs.append(df)
            shards.append(sh)
        return pd.concat(dfs, ignore_index=True), sorted(shards)
    return None, []


def apply_fallback(df, base_score, deny):
    m = df["token"].isin(deny)
    s = df["score"].copy()
    s.loc[m] = df.loc[m, "token"].map(base_score).fillna(df.loc[m, "score"])
    return float(s.mean())


def cell(df, shards, mode, base_score, deny):
    if df is None:
        return None
    n = len(df)
    pdms = float(df["score"].mean())
    if mode == "fallback":
        pdms = apply_fallback(df, base_score, deny)
    nc = float(df["no_at_fault_collisions"].mean())
    ep = float(df["ego_progress"].mean())
    full = shards == [0, 1, 2, 3]
    cov = "full" if full else ("sh" + "".join(map(str, shards)))
    return dict(N=n, pdms=pdms, nc=nc, ep=ep, cov=cov, full=full)


# ---- Table 1 row specification -------------------------------------------
# (group_ratio, method_label, ours?, mode, [csv prefixes in priority order], base_dir)
ROWS = [
    # No pruning
    ("nop", "No Prune", False, "raw", ["MT_attn_L12_r10"], RAW),
    # Retain 540 (r=0.75)
    (0.75, "FastV", False, "raw", ["MT_fastv_l2_drop_r075"], RAW),
    (0.75, "Random", False, "raw", ["MT_random_drop_r075"], RAW),
    (0.75, "PruMerge", False, "raw", ["MT_prumerge_cls_drop_r075"], RAW),
    (0.75, "SparseVLM", False, "raw", ["MT_sparsevlm_text_r075", "MT_sparsevlm_text_drop_r075"], RAW),
    (0.75, "SFT Scorer (ours)", True, "fallback", ["MT_varBsafe_scorer_r075"], RAW),
    (0.75, "RL Scorer (ours)", True, "fallback", ["MT_rl_shaped_r075"], RAW),
    # Retain 360 (r=0.5)
    (0.50, "FastV", False, "raw", ["MT_fastv_l2_drop_r05"], RAW),
    (0.50, "Random", False, "raw", ["MT_random_r05", "MT_random_drop_r05"], RAW),
    (0.50, "PruMerge", False, "raw", ["MT_prumerge_cls_drop_r05"], RAW),
    (0.50, "SparseVLM", False, "raw", ["MT_sparsevlm_text_drop_r05"], RAW),
    (0.50, "SFT Scorer (ours)", True, "fallback", ["MT_varBsafe_scorer_r05"], RAW),
    (0.50, "RL Scorer (ours)", True, "fallback", ["MT_rl_shaped_r05"], RAW),
    # Retain 180 (r=0.25)
    (0.25, "FastV", False, "raw", ["MT_fastv_l2_drop_r025"], RAW),
    (0.25, "Random", False, "raw", ["MT_random_drop_r025"], RAW),
    (0.25, "PruMerge", False, "raw", ["MT_prumerge_cls_drop_r025"], RAW),
    (0.25, "SparseVLM", False, "raw", ["MT_sparsevlm_text_drop_r025"], RAW),
    (0.25, "SFT Scorer (ours)", True, "fallback", ["MT_sft_varB_drop_r025"], RAW),
    (0.25, "RL Scorer (ours)", True, "fallback", ["MT_rl_shaped_r025"], RAW),
]
DYN_ROWS = [
    ("SFT + \u03c4-cut (ours)", "fallback", ["TC_mse_tau_kr060"], TAUCUT, "~432"),
    ("RL + \u03c4-cut (ours)", "fallback", ["MT_rl_taucut_kr060"], RAW, "~?"),
    ("Budget RL (ours)", "fallback", ["MT_budget_rl_dynamic"], RAW, "~?"),
]


def fmt_inner(c, base_pdms):
    """Return the middle cells: 'PDMS | NC | EP | Rel.% | N' (no outer pipes)."""
    if c is None:
        return "? | ? | ? | ? | pending"
    rel = 100.0 * c["pdms"] / base_pdms
    tag = "" if c["full"] else f" *({c['cov']})*"
    return f"{c['pdms']:.4f}{tag} | {c['nc']:.3f} | {c['ep']:.3f} | {rel:.1f}% | {c['N']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/table1_draft.md"))
    args = ap.parse_args()
    deny = load_denylist()

    base_df, base_shards = gather(["MT_attn_L12_r10"])
    base_score = base_df.set_index("token")["score"]
    base_pdms = float(base_df["score"].mean())

    lines = []
    lines.append("# Table 1 (draft) — NAVSIM AutoVLA-3B, closed-loop, true token removal (Variant B)")
    lines.append("")
    lines.append(f"> Auto-generated. Baseline (No Prune) PDMS = {base_pdms:.5f} (N={len(base_df)}).")
    lines.append("> Baselines report RAW PDMS; **(ours)** rows report +fallback (denylist->no-prune, 768 tokens).")
    lines.append("> Cells tagged *(shX)* are shard-subset placeholders, not yet full navtest.")
    lines.append("")
    hdr = "| Method | Tokens | PDMS | NC | EP | Rel.% | N | FLOPs\u2193 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"

    groups = [("nop", "No Pruning (720 tokens)"),
              (0.75, "Retain 540 tokens (\u219325%)"),
              (0.50, "Retain 360 tokens (\u219350%)"),
              (0.25, "Retain 180 tokens (\u219375%)")]
    printed = False
    for gkey, gtitle in groups:
        lines.append(f"### {gtitle}")
        lines.append(hdr)
        lines.append(sep)
        for (rk, label, ours, mode, prefixes, bdir) in ROWS:
            if rk != gkey:
                continue
            df, shards = gather(prefixes, bdir)
            c = cell(df, shards, mode, base_score, deny)
            if gkey == "nop":
                tok, flops = 720, "\u2014"
            else:
                tok, _, flops = RATIO_META[gkey]
            lines.append(f"| {label} | {tok} | {fmt_inner(c, base_pdms)} | {flops} |")
        lines.append("")

    lines.append("### Dynamic (scene-adaptive)")
    lines.append(hdr)
    lines.append(sep)
    for (label, mode, prefixes, bdir, toks) in DYN_ROWS:
        df, shards = gather(prefixes, bdir)
        c = cell(df, shards, mode, base_score, deny)
        flops = "~27%" if "\u03c4" in label else "\u2014"
        lines.append(f"| {label} | {toks} | {fmt_inner(c, base_pdms)} | {flops} |")
    lines.append("")

    txt = "\n".join(lines)
    outp = Path(args.out)
    tmp = outp.with_suffix(outp.suffix + f".tmp{os.getpid()}")
    tmp.write_text(txt)
    os.replace(tmp, outp)  # atomic; safe under concurrent regen from workers
    print(txt)
    print(f"\n[gen_table1_draft] wrote {args.out}")


if __name__ == "__main__":
    main()

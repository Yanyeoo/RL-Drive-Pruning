"""s3_aggregate_maintable.py — aggregate full-navtest main-table CSVs.

Reads results/raw/tokenprune_S3_full/MT_<sel>_r<rt>_sh<0..3>.csv, merges the 4
shards per (selector,ratio), reports mean PDMS, and the iso-compute r=0.5
comparison (scorer vs attn_L12 vs random vs r=1.0) on common tokens. Safe to run
anytime (uses whatever CSVs exist). Writes _aggregate.md next to the CSVs.
"""
import csv, glob, os, re
from collections import defaultdict

D = "/apdcephfs/private_shayladeng/tokenrl_autoVLA/results/raw/tokenprune_S3_full"


def load(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                d[row["token"]] = float(row["score"])
            except Exception:
                pass
    return d


def main():
    arms = defaultdict(dict)  # (sel,rt) -> token->score  (merged shards)
    shards = defaultdict(set)
    for p in sorted(glob.glob(f"{D}/MT_*.csv")):
        m = re.match(r"MT_(.+)_r(\d+)_sh(\d)\.csv", os.path.basename(p))
        if not m:
            continue
        sel, rt, sh = m.group(1), m.group(2), m.group(3)
        arms[(sel, rt)].update(load(p))
        shards[(sel, rt)].add(sh)

    lines = ["# full-navtest main-table aggregate", ""]
    lines.append("| selector | ratio | shards | N | mean PDMS |")
    lines.append("|---|---|---|---|---|")
    table = {}
    for (sel, rt), d in sorted(arms.items()):
        n = len(d)
        mean = sum(d.values()) / n if n else float("nan")
        table[(sel, rt)] = d
        lines.append(f"| {sel} | 0.{rt} | {len(shards[(sel,rt)])}/4 | {n} | {mean:.6f} |")
    # iso r=0.5 comparison on common tokens
    r05 = {k: v for k, v in table.items() if k[1] == "05"}
    ref10 = table.get(("attn_L12", "10")) or table.get(("scorer", "10"))
    lines += ["", "## iso-compute r=0.5 (common tokens across available r=0.5 arms + r=1.0)"]
    keys = list(r05.keys()) + ([("attn_L12", "10")] if ref10 else [])
    if r05:
        common = set.intersection(*[set(table[k]) for k in keys]) if keys else set()
        lines.append(f"common N = {len(common)}")
        for k in keys:
            mv = sum(table[k][t] for t in common) / len(common) if common else float("nan")
            lines.append(f"- {k[0]} r=0.{k[1]}: {mv:.6f}")
        if ("scorer", "05") in r05 and ref10 and common:
            sc = sum(table[('scorer','05')][t] for t in common)/len(common)
            r1 = sum(ref10[t] for t in common)/len(common)
            lines.append(f"- **scorer r=0.5 − r=1.0 = {100*(sc-r1):+.3f} pt** (claim1: >= -0.5pt = ~lossless)")
            if ("attn_L12","05") in r05:
                a = sum(table[('attn_L12','05')][t] for t in common)/len(common)
                lines.append(f"- **scorer − attn_L12 @r0.5 = {100*(sc-a):+.3f} pt** (claim3)")
            if ("random","05") in r05:
                rd = sum(table[('random','05')][t] for t in common)/len(common)
                lines.append(f"- **scorer − random @r0.5 = {100*(sc-rd):+.3f} pt** (claim3)")
    out = "\n".join(lines)
    print(out)
    open(f"{D}/_aggregate.md", "w").write(out + "\n")
    print(f"\n-> {D}/_aggregate.md")


if __name__ == "__main__":
    main()

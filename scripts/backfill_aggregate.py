#!/usr/bin/env python3
"""backfill_aggregate.py — Aggregate main-table backfill families to full navtest.

Raw PDMS only (no denylist, no fallback). For each family, concatenates
MT_<family>_sh0..3.csv, drops the 'average' row, reports N and mean score.
Only families with all four shards present are reported as COMPLETE.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--expected-scenes", type=int, default=11576)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for fam in args.families:
        shards = {}
        dfs = []
        complete = True
        for sh in range(4):
            p = result_dir / f"MT_{fam}_sh{sh}.csv"
            if not p.exists():
                complete = False
                shards[sh] = None
                continue
            df = pd.read_csv(p)
            df = df[df["token"] != "average"]
            shards[sh] = int(len(df))
            dfs.append(df)

        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            n = int(len(all_df))
            pdms = float(all_df["score"].mean())
        else:
            n, pdms = 0, float("nan")

        rec = {
            "family": fam,
            "complete_4shard": complete,
            "shard_rows": shards,
            "n_total": n,
            "raw_pdms": pdms,
            "meets_expected": (n == args.expected_scenes),
        }
        summary[fam] = rec

        if complete and dfs:
            merged = out_dir / f"MT_{fam}_full.csv"
            all_df.to_csv(merged, index=False)

        status = "COMPLETE" if complete else "INCOMPLETE"
        print(
            f"[{status}] {fam}: N={n} raw_PDMS="
            f"{pdms:.6f} shards={shards}"
        )

    (out_dir / "backfill_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir/'backfill_summary.json'}")


if __name__ == "__main__":
    main()

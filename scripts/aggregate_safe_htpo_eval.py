#!/usr/bin/env python3
"""Aggregate a no-denylist Safe-HTPO full-navtest dynamic evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

SUBMETRICS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
]


def load_rows(paths: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("token") == "average":
                    continue
                if not row.get("token"):
                    continue
                parsed: dict[str, float | str] = {"token": row["token"]}
                for key in ["score", *SUBMETRICS]:
                    parsed[key] = float(row[key])
                rows.append(parsed)
    return rows


def bootstrap_ci(values: np.ndarray, seed: int = 42, rounds: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(rounds, dtype=np.float64)
    n = len(values)
    for i in range(rounds):
        means[i] = values[rng.integers(0, n, size=n)].mean()
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def parse_keep_ratios(log_paths: list[Path]) -> list[float]:
    values: list[float] = []
    pattern = re.compile(r"\bkr=([0-9]+(?:\.[0-9]+)?)")
    for path in log_paths:
        if not path.exists():
            continue
        values.extend(float(x) for x in pattern.findall(path.read_text(errors="replace")))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-scenes", type=int, default=11576)
    args = parser.parse_args()

    csv_paths = sorted(args.result_dir.glob("*.csv"))
    if len(csv_paths) != 4:
        raise SystemExit(f"Expected four shard CSVs, found {len(csv_paths)} in {args.result_dir}")
    rows = load_rows(csv_paths)
    tokens = [str(row["token"]) for row in rows]
    if len(rows) != args.expected_scenes or len(set(tokens)) != len(tokens):
        raise SystemExit(
            f"Invalid coverage: rows={len(rows)}, unique_tokens={len(set(tokens))}, expected={args.expected_scenes}"
        )

    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    summary: dict[str, object] = {
        "run_id": args.run_id,
        "protocol": "Safe-HTPO dynamic budget, Variant-B true token drop, no denylist fallback",
        "n_scenes": len(rows),
        "pdms_mean": float(scores.mean()),
        "pdms_std": float(scores.std(ddof=1)),
        "pdms_p05": float(np.quantile(scores, 0.05)),
        "pdms_p50": float(np.quantile(scores, 0.50)),
        "pdms_ci95_bootstrap": bootstrap_ci(scores),
        "submetrics_mean": {
            key: float(np.mean([float(row[key]) for row in rows])) for key in SUBMETRICS
        },
        "csv_files": [str(path) for path in csv_paths],
    }

    ratios = parse_keep_ratios(sorted(args.log_dir.glob("*_dynamic_sh*.log")))
    if ratios:
        ratio_values = np.asarray(ratios, dtype=np.float64)
        summary["keep_ratio"] = {
            "n_logged": len(ratios),
            "mean": float(ratio_values.mean()),
            "std": float(ratio_values.std(ddof=1)),
            "p10": float(np.quantile(ratio_values, 0.10)),
            "p50": float(np.quantile(ratio_values, 0.50)),
            "p90": float(np.quantile(ratio_values, 0.90)),
            "mean_tokens_of_720": float(720 * ratio_values.mean()),
        }
    else:
        summary["keep_ratio"] = {"n_logged": 0, "warning": "No kr= entries found in eval logs"}

    output_json = args.result_dir / "safe_htpo_summary.json"
    output_md = args.result_dir / "safe_htpo_summary.md"
    output_json.write_text(json.dumps(summary, indent=2) + "\n")

    ratio = summary["keep_ratio"]
    lines = [
        "# Safe-HTPO Dynamic Evaluation Summary",
        "",
        f"- Run: `{args.run_id}`",
        "- Protocol: Variant-B true token drop; no denylist fallback.",
        f"- Coverage: {summary['n_scenes']} unique navtest scenes.",
        f"- PDMS: {summary['pdms_mean']:.5f} (95% bootstrap CI {summary['pdms_ci95_bootstrap'][0]:.5f}, {summary['pdms_ci95_bootstrap'][1]:.5f})",
        f"- PDMS P5: {summary['pdms_p05']:.5f}",
        "",
        "## Mean submetrics",
    ]
    lines.extend(f"- `{key}`: {value:.5f}" for key, value in summary["submetrics_mean"].items())
    lines.extend(["", "## Dynamic keep ratio"])
    if ratio["n_logged"]:
        lines.extend([
            f"- Mean/std: {ratio['mean']:.5f} / {ratio['std']:.5f}",
            f"- P10/P50/P90: {ratio['p10']:.5f} / {ratio['p50']:.5f} / {ratio['p90']:.5f}",
            f"- Mean retained visual tokens: {ratio['mean_tokens_of_720']:.1f} / 720",
        ])
    else:
        lines.append(f"- {ratio['warning']}")
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

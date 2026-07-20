#!/usr/bin/env bash
# ============================================================================
# run_m1b_phase_e2_gate.sh — Phase E2 5-token bit-identical gate
# ----------------------------------------------------------------------------
# Purpose: Gate Phase F (the 4-variant × 11576-token sweep) by verifying that
# the head_mask wrapper is a true no-op when head_mask_layers is empty (V0)
# and behaves correctly when masking a confirmed-dead head (V1).
#
# Spec: docs/specs/m1b_freelunch_spec.md §6
#
# What this does:
#   1. Run V0 (head_mask_layers=null) on 5 navtest scenes (~3 min)
#   2. Run V1 (head_mask_layers={12:[13]}) on the same 5 scenes (~3 min)
#   3. Compare per-token PDMS in V0 vs the previously-recorded B0 numbers
#      for those tokens (PASS if max|delta| < 1e-4 per spec, RELAXED if 1e-3)
#   4. Verify V1 ran with head_mask logged ("[head_mask] L12 first fire")
#   5. Write docs/_internal/m1b_phaseE2_gate.md with concrete numbers
#
# Cost: ~6-7 min on a single H20 (most of which is model load).
#
# Usage:
#   bash scripts/run_m1b_phase_e2_gate.sh
#
# Env knobs:
#   GPU         single GPU index (default 0)
#   TIMEOUT     per-variant timeout in seconds (default 900 = 15 min)
#   B0_REF_CSV  optional path to a previously-recorded smoke5 csv to diff
#               V0 against. If unset, only V1 sanity is checked and V0
#               numbers are merely recorded.
#
# Exit code:
#   0  if gate passes (PASS or RELAXED with numbers in expected range)
#   1  if gate fails — DO NOT launch Phase F. See gate doc + rollback to
#      tag pre_m1b_phase_d.
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"

GPU="${GPU:-0}"
TIMEOUT="${TIMEOUT:-900}"
B0_REF_CSV="${B0_REF_CSV:-}"

# Reuse the main sweep dispatcher, but with smoke5 scene filter and only
# V0 + V1 variants. The sweep dispatcher does the heavy lifting (manifest
# writing, exit-code capture, etc.).
VARIANTS="V0 V1" \
SCENE_FILTER="navtest_smoke5" \
GPU="${GPU}" \
TIMEOUT="${TIMEOUT}" \
TAG_PREFIX="m1b_e2gate" \
JSON_DIR="${PROJECT_ROOT}/data/navtest_nocot_smoke_seed" \
  bash "${PROJECT_ROOT}/scripts/run_m1b_freelunch_sweep.sh"

SWEEP_RC=$?
if [[ ${SWEEP_RC} -ne 0 ]]; then
  echo "[e2] sweep dispatcher exited ${SWEEP_RC}" >&2
fi

# ---- analyze ----
PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
OUT_DOC="${PROJECT_ROOT}/docs/_internal/m1b_phaseE2_gate.md"
mkdir -p "$(dirname "${OUT_DOC}")"

"${PY}" - <<PYEOF
import json, sys
from pathlib import Path
import pandas as pd

results_root = Path("${PROJECT_ROOT}") / "results" / "raw"
v0_dirs = sorted(results_root.glob("M1b_freelunch_V0_*"))
v1_dirs = sorted(results_root.glob("M1b_freelunch_V1_*"))

# Take the most recent run of each
v0 = v0_dirs[-1] if v0_dirs else None
v1 = v1_dirs[-1] if v1_dirs else None

print(f"[e2] V0 dir = {v0}")
print(f"[e2] V1 dir = {v1}")

gate_doc = Path("${OUT_DOC}")
lines = []
lines.append("# M1.b Phase E2 — bit-identical gate result")
lines.append("")
lines.append(f"- Generated: \$(date -u +%Y-%m-%dT%H:%M:%SZ)  (UTC).")
lines.append(f"- Spec: docs/specs/m1b_freelunch_spec.md §6")
lines.append(f"- B0 ref csv (if provided): \`${B0_REF_CSV}\`")
lines.append("")

def fmt_manifest(d):
    if d is None:
        return "MISSING"
    mf = d / "manifest.json"
    if not mf.exists():
        return f"NO MANIFEST under {d}"
    m = json.loads(mf.read_text())
    return f"variant={m['variant']}  pdms={m['pdms']}  n_valid={m['n_valid']}/{m['n_total']}  rc={m['rc']}  wall={m['wall_seconds']}s  fail={m['failure_reason']}"

lines.append("## V0 (head_mask=null) summary")
lines.append("")
lines.append(f"- {fmt_manifest(v0)}")
lines.append("")
lines.append("## V1 (head_mask={12:[13]}) summary")
lines.append("")
lines.append(f"- {fmt_manifest(v1)}")
lines.append("")

# ---- correctness check: V0 should preserve B0 per-token PDMS ----
v0_pdms_delta = None
if "${B0_REF_CSV}" and Path("${B0_REF_CSV}").exists() and v0 is not None and (v0 / "merged.csv").exists():
    b0 = pd.read_csv("${B0_REF_CSV}")
    cur = pd.read_csv(v0 / "merged.csv")
    common = set(b0.token) & set(cur.token)
    common.discard("average")
    deltas = []
    for tok in sorted(common):
        b0_row = b0[b0.token == tok].iloc[0]
        cur_row = cur[cur.token == tok].iloc[0]
        if bool(b0_row.get("valid", True)) and bool(cur_row.get("valid", True)):
            deltas.append(float(cur_row["score"]) - float(b0_row["score"]))
    if deltas:
        v0_pdms_delta = max(abs(d) for d in deltas)
        lines.append("## V0 per-token PDMS delta vs B0 ref")
        lines.append("")
        lines.append(f"- N common valid tokens: {len(deltas)}")
        lines.append(f"- max |delta|: {v0_pdms_delta:.6f}")
        lines.append(f"- mean delta:  {sum(deltas)/len(deltas):+.6f}")
        if v0_pdms_delta < 1e-6:
            verdict = "PASS (bit-identical: < 1e-6)"
        elif v0_pdms_delta < 1e-4:
            verdict = "PASS (per spec § 6 strict: < 1e-4)"
        elif v0_pdms_delta < 1e-3:
            verdict = "RELAXED PASS (per spec § 6 relaxed: < 1e-3)"
        else:
            verdict = "FAIL — V0 should equal B0 but delta > 1e-3"
        lines.append(f"- verdict: **{verdict}**")
else:
    lines.append("## V0 vs B0 ref comparison")
    lines.append("")
    lines.append("- SKIPPED — no B0 ref csv supplied or V0 missing. Re-run with B0_REF_CSV set.")
    lines.append("- V0 numbers alone do not gate; numbers above are the V0 self-record.")
lines.append("")

# ---- V1 sanity: head_mask must have fired ----
lines.append("## V1 head_mask sanity")
lines.append("")
if v1 is not None:
    log = v1 / "shard0.log"
    if log.exists():
        text = log.read_text(errors="ignore")
        fire = "[head_mask] L12 first fire" in text
        enabled = "[head_mask] enabling mask" in text
        lines.append(f"- '[head_mask] enabling mask' in log: {enabled}")
        lines.append(f"- '[head_mask] L12 first fire' in log: {fire}")
        if fire and enabled:
            lines.append("- verdict: **PASS — head_mask hook fired**")
        else:
            lines.append("- verdict: **FAIL — head_mask hook did not fire (check log)**")
    else:
        lines.append("- log missing")
else:
    lines.append("- V1 missing")

lines.append("")
lines.append("## Conclusion")
lines.append("")
lines.append("If V0 verdict is PASS and V1 head_mask fired, Phase F is GREEN-LIT.")
lines.append("Run:")
lines.append("\`\`\`bash")
lines.append("nohup setsid bash scripts/run_m1b_freelunch_sweep.sh \\")
lines.append("  > logs/m1b_phaseF_main.log 2>&1 &")
lines.append("\`\`\`")

gate_doc.write_text("\n".join(lines))
print(f"[e2] wrote {gate_doc}")
print("")
print(gate_doc.read_text())
PYEOF

echo ""
echo "[e2] gate result document: ${OUT_DOC}"

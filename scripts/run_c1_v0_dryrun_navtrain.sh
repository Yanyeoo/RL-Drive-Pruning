#!/usr/bin/env bash
# ============================================================================
# run_c1_v0_dryrun_navtrain.sh — C1: V0 baseline (no mask) on navtrain probe100
# ----------------------------------------------------------------------------
# Validates that:
#   1. navtrain_metric_cache (built by B1.0) is readable by run_pdm_score_cot.py
#   2. The wrapped agent (AutoVLAWithAttentionAgent) loads + runs on navtrain JSONs
#   3. PDMS comes out as a sane number (sanity, not a target)
#
# Pre-condition:
#   B1.0 (run_b1_0_navtrain_metric_caching.sh) must have completed
#   AND data/navtrain_metric_cache/ should contain ≈19,225 cache entries.
#
# This script forks run_m1b_freelunch_sweep.sh but:
#   - SCENE_FILTER = navtrain_probe100  (100 token, ~30 min on 1 GPU)
#   - VARIANTS     = V0                 (no head mask)
#   - METRIC_CACHE = data/navtrain_metric_cache  (NEW)
#   - JSON_DIR     = data/navtrain_nocot          (the 19,225 pretokenized JSONs)
#   - TIMEOUT      = 2700s (45 min)
#
# Spec: docs/_internal/m1b2_v4_spec_2026-06-25.md §C1 (link in journal)
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"

export VARIANTS="V0"
export SCENE_FILTER="navtrain_probe100"
export GPU="${GPU:-0}"
export TIMEOUT="${TIMEOUT:-2700}"
export TAG_PREFIX="c1_navtrain_v0_dryrun"
export JSON_DIR="${PROJECT_ROOT}/data/navtrain_nocot"
export METRIC_CACHE="${PROJECT_ROOT}/data/navtrain_metric_cache"
export RESULTS_ROOT="${PROJECT_ROOT}/results/raw"

# Pre-flight: confirm B1.0 produced cache
if [[ ! -d "${METRIC_CACHE}" ]]; then
  echo "[c1] FATAL: ${METRIC_CACHE} does not exist. Did B1.0 finish?" >&2
  exit 4
fi
# NB: B1.0 (run_metric_caching.py) writes uncompressed `metric_cache.pkl`,
# NOT `metric_cache.pkl.xz` — original draft of this pre-flight grepped for
# the wrong filename and false-warned even on a complete cache.
# Verified 2026-06-26: 19225 navtrain tokens land at depth 4.
N_CACHE=$(find "${METRIC_CACHE}" -maxdepth 4 -name "metric_cache.pkl" 2>/dev/null | wc -l)
echo "[c1] pre-flight: navtrain_metric_cache count = ${N_CACHE}"
if [[ ${N_CACHE} -lt 18000 ]]; then
  echo "[c1] WARNING: cache count ${N_CACHE} < 18000, B1.0 may be incomplete." >&2
  echo "[c1] Continue anyway? Press Ctrl-C in 5s to abort."
  sleep 5
fi

echo "[c1] launching V0 dryrun on navtrain_probe100 ..."
exec bash "${PROJECT_ROOT}/scripts/run_m1b_freelunch_sweep.sh"

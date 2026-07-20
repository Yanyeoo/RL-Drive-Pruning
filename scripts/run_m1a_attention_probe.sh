#!/usr/bin/env bash
# ============================================================================
# run_m1a_attention_probe.sh — M1.a attention layer-probe driver
# ----------------------------------------------------------------------------
# Wraps code/rldrive/scoring/run_attention_probe.py with the right PYTHONPATH
# and conda env. Mirrors scripts/run_autovla_navtest_dual_gpu.sh conventions
# so the env vars + checkpoints + codebook paths stay consistent.
#
# Usage:
#   bash scripts/run_m1a_attention_probe.sh \
#        --scene-filter navtest_smoke5 \
#        --save-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer14_smoke \
#        --layer-idx 14 \
#        --max-scenes 5
#
# Or for navtrain probe A (after chain_complete):
#   bash scripts/run_m1a_attention_probe.sh \
#        --scene-filter probe_A \
#        --json-dir /apdcephfs/.../data/navtrain_nocot \
#        --save-dir /apdcephfs/.../exp/m1a_layer14_navtrain \
#        --layer-idx 14 \
#        --max-scenes 100
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
AUTOVLA_ROOT="${PROJECT_ROOT}/code/third_party/AutoVLA"
NAVSIM_ROOT="${AUTOVLA_ROOT}/navsim"

PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
if [[ ! -x "${PY}" ]]; then
    echo "[m1a] FATAL: python not found at ${PY}" >&2
    exit 2
fi

# Pre-flight: aria2c is irrelevant here, but eager attention is. We don't run a
# check inside the bash layer; the python AutoVLAWithAttentionAgent constructor
# already raises if attn_impl != 'eager'.

# PYTHONPATH: rldrive code first, then navsim, then AutoVLA
export PYTHONPATH="${PROJECT_ROOT}/code:${NAVSIM_ROOT}:${AUTOVLA_ROOT}:${PYTHONPATH:-}"

# Forward all args verbatim to the runner
exec "${PY}" -m rldrive.scoring.run_attention_probe "$@"

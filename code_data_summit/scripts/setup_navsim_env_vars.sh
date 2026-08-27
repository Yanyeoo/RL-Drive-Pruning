#!/usr/bin/env bash
# ============================================================================
# setup_navsim_env_vars.sh — env vars required for AutoVLA → navsim evaluation
# ----------------------------------------------------------------------------
# Usage (must source, not bash):
#   source scripts/setup_navsim_env_vars.sh
#
# Differences from prior-work tokenrl/code/scripts/setup_navsim_env_vars.sh:
#   - NAVSIM_DEVKIT_ROOT points to AutoVLA's navsim fork (not prior tokenrl)
#   - NAVSIM_EXP_ROOT points to tokenrl_autoVLA/exp
#   - Data assets (sensor_blobs / maps) still come from tokenrl/data
#     (single source of truth; new repo doesn't re-download 116GB)
# ============================================================================

ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
PRIOR_ROOT=/apdcephfs/private_shayladeng/tokenrl

# AutoVLA's bundled navsim fork
export NAVSIM_DEVKIT_ROOT="$ROOT/code/third_party/AutoVLA/navsim"

# Experiment / cache output (new repo's own scope)
export NAVSIM_EXP_ROOT="$ROOT/exp"

# Map data (shared from prior-work)
export NUPLAN_MAPS_ROOT="$PRIOR_ROOT/data/maps/nuplan-maps-v1.0"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# Sensor + navsim_logs data root (shared from prior-work, 116GB sensor_blobs)
export OPENSCENE_DATA_ROOT="$PRIOR_ROOT/data/navsim_v2"

# H20 sm_90 + torch 2.4 + cu121 cuBLAS fp32 GEMM SIGFPE workaround
# (see docs/journal/2026-06-16_ma2_3_smoke_pass.md §2 E3)
export NVIDIA_TF32_OVERRIDE=0

mkdir -p "$NAVSIM_EXP_ROOT"

echo "[autovla navsim env vars]"
echo "  NAVSIM_DEVKIT_ROOT  = $NAVSIM_DEVKIT_ROOT"
echo "  NAVSIM_EXP_ROOT     = $NAVSIM_EXP_ROOT"
echo "  NUPLAN_MAPS_ROOT    = $NUPLAN_MAPS_ROOT"
echo "  NUPLAN_MAP_VERSION  = $NUPLAN_MAP_VERSION"
echo "  OPENSCENE_DATA_ROOT = $OPENSCENE_DATA_ROOT"
echo "  NVIDIA_TF32_OVERRIDE= $NVIDIA_TF32_OVERRIDE"

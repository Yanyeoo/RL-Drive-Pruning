#!/usr/bin/env bash
# ============================================================================
# env.example.sh — copy to env.sh and edit for YOUR machine, then `source env.sh`
#
# Every script in scripts/ currently hard-codes the original training node's
# absolute paths (/apdcephfs/private_shayladeng/...). Rather than rewriting all
# of them, define your own paths here and export RLDRIVE_* so the reproduction
# steps in docs/REPRODUCE.md can be followed verbatim.
#
#   cp env.example.sh env.sh   # env.sh is gitignored
#   $EDITOR env.sh
#   source env.sh
# ============================================================================

# --- 1. where you cloned THIS repository -----------------------------------
export RLDRIVE_ROOT="${RLDRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# --- 2. python interpreter of your conda env (see docs/REPRODUCE.md §1) ----
export RLDRIVE_PY="${RLDRIVE_PY:-$(command -v python)}"

# --- 3. NAVSIM / OpenScene data you downloaded yourself --------------------
# Must contain: navsim_logs/  sensor_blobs/  (see docs/REPRODUCE.md §2)
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HOME/data/navsim_v2}"

# nuPlan maps (downloaded separately, ~1.6GB)
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$HOME/data/maps/nuplan-maps-v1.0}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# --- 4. AutoVLA checkout (upstream clone + our overlay applied) ------------
export AUTOVLA_ROOT="${AUTOVLA_ROOT:-$RLDRIVE_ROOT/code/third_party/AutoVLA}"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA_ROOT/navsim"

# --- 5. frozen VLA backbone checkpoint (download from AutoVLA release) -----
export AUTOVLA_CKPT="${AUTOVLA_CKPT:-$RLDRIVE_ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt}"

# --- 6. experiment output ---------------------------------------------------
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$RLDRIVE_ROOT/exp}"
mkdir -p "$NAVSIM_EXP_ROOT"

# --- 7. required numerical workaround --------------------------------------
# H20 / sm_90 + torch 2.4 + cu121: fp32 cuBLAS GEMM can raise SIGFPE without this.
export NVIDIA_TF32_OVERRIDE=0

# --- 8. import paths -------------------------------------------------------
export PYTHONPATH="$RLDRIVE_ROOT/code:$NAVSIM_DEVKIT_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[rldrive env]"
echo "  RLDRIVE_ROOT        = $RLDRIVE_ROOT"
echo "  RLDRIVE_PY          = $RLDRIVE_PY"
echo "  OPENSCENE_DATA_ROOT = $OPENSCENE_DATA_ROOT"
echo "  NUPLAN_MAPS_ROOT    = $NUPLAN_MAPS_ROOT"
echo "  AUTOVLA_ROOT        = $AUTOVLA_ROOT"
echo "  AUTOVLA_CKPT        = $AUTOVLA_CKPT"
echo "  NAVSIM_EXP_ROOT     = $NAVSIM_EXP_ROOT"

# sanity warnings (non-fatal, so this file stays sourceable)
[[ -d "$OPENSCENE_DATA_ROOT/sensor_blobs" ]] || echo "  ⚠️  sensor_blobs not found under OPENSCENE_DATA_ROOT"
[[ -d "$NUPLAN_MAPS_ROOT" ]] || echo "  ⚠️  NUPLAN_MAPS_ROOT does not exist"
[[ -f "$AUTOVLA_CKPT" ]] || echo "  ⚠️  AUTOVLA_CKPT not found (needed for train/eval)"
[[ -f "$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml" ]] \
  || echo "  ⚠️  AutoVLA overlay not applied yet — see docs/REPRODUCE.md §3"

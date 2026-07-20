#!/usr/bin/env bash
# ============================================================================
# run_m1a_layer_sweep_4gpu.sh — 4-GPU parallel layer sweep for M1.a
# ----------------------------------------------------------------------------
# Sweeps a curated layer set across 100-scene navtest probe in parallel.
# Each GPU covers ~25 scenes via shard-stride=4, but all 4 GPUs run on
# DIFFERENT layers concurrently so we parallelize over (layer, gpu) pairs.
#
# Strategy: 8 layers × 100 scenes = 800 forward passes.
#   - Each GPU handles 2 layers (sequentially) × 100 scenes = 200 passes
#   - 1 pass ~3s on H20 -> ~10min/layer per GPU -> ~20min wall-clock total.
#   - 4 GPUs running independently, fully detached. Safe to lose this session.
#
# Output layout:
#   exp/m1a_layer_sweep_<TS>/L00/<token>.pt
#   exp/m1a_layer_sweep_<TS>/L04/<token>.pt
#   ...
#   exp/m1a_layer_sweep_<TS>/L27/<token>.pt
#   logs/m1a_sweep_<TS>_gpu<N>_L<LL>.log
#
# Layers chosen: {0, 4, 8, 12, 16, 20, 24, 27}.
# This skips a few layers (3, 7, ...) but covers Qwen2.5-VL-3B's 28 layers
# uniformly enough to find argmax(vision-attention fraction).
#
# Usage:
#   bash scripts/run_m1a_layer_sweep_4gpu.sh [TOKEN_LIST]
#
# If TOKEN_LIST is provided, must be a text file with one token per line.
# Otherwise defaults to first 100 scenes from data/navtest_nocot/ (sorted).
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
TS="$(date +%Y%m%d_%H%M)"
SWEEP_DIR="${PROJECT_ROOT}/exp/m1a_layer_sweep_${TS}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${SWEEP_DIR}" "${LOG_DIR}"

# Build/locate the 100-token list
TOKEN_LIST="${1:-${SWEEP_DIR}/tokens_100.txt}"
if [[ "$#" -lt 1 ]]; then
    # Auto-build deterministic 100-token list from navtest_nocot
    ls "${PROJECT_ROOT}/data/navtest_nocot/" | grep '\.json$' | sed 's/\.json$//' | sort | head -100 \
        > "${TOKEN_LIST}"
    echo "[sweep] auto-built token list with $(wc -l < ${TOKEN_LIST}) tokens at ${TOKEN_LIST}"
fi
N_TOKENS=$(wc -l < "${TOKEN_LIST}")
echo "[sweep] using TOKEN_LIST=${TOKEN_LIST} with ${N_TOKENS} tokens"
echo "[sweep] outputs -> ${SWEEP_DIR}/L<NN>/"

# Layer assignment: gpu 0 -> L00,L16; gpu 1 -> L04,L20; gpu 2 -> L08,L24; gpu 3 -> L12,L27
declare -A GPU_LAYERS
GPU_LAYERS[0]="0 16"
GPU_LAYERS[1]="4 20"
GPU_LAYERS[2]="8 24"
GPU_LAYERS[3]="12 27"

PIDS=()
for GPU in 0 1 2 3; do
    LAYERS_FOR_GPU="${GPU_LAYERS[$GPU]}"
    LAYER_LOG="${LOG_DIR}/m1a_sweep_${TS}_gpu${GPU}.log"
    # Sequentially run both layers on this GPU in one bg process.
    (
        echo "[gpu${GPU}] starting layers: ${LAYERS_FOR_GPU} at $(date '+%F %T')"
        for LAYER in ${LAYERS_FOR_GPU}; do
            LAYER_PAD=$(printf "%02d" "${LAYER}")
            SAVE="${SWEEP_DIR}/L${LAYER_PAD}"
            mkdir -p "${SAVE}"
            echo "[gpu${GPU}][L${LAYER_PAD}] start at $(date '+%F %T')"
            bash "${PROJECT_ROOT}/scripts/run_m1a_attention_probe.sh" \
                --scene-filter "navtest_100" \
                --save-dir "${SAVE}" \
                --layer-idx "${LAYER}" \
                --gpu "${GPU}" \
                --token-list "${TOKEN_LIST}" \
                --max-scenes 100 \
                2>&1
            EC=$?
            echo "[gpu${GPU}][L${LAYER_PAD}] done at $(date '+%F %T') exit=${EC}"
        done
        echo "[gpu${GPU}] all layers done at $(date '+%F %T')"
    ) > "${LAYER_LOG}" 2>&1 &
    PID=$!
    PIDS+=("${PID}")
    echo "[sweep] launched gpu${GPU} (layers ${LAYERS_FOR_GPU}) PID=${PID} log=${LAYER_LOG}"
done

# Status file for the next AI to find sweep status quickly
STATUS_FILE="${SWEEP_DIR}/SWEEP_STATUS.txt"
{
    echo "M1.a layer sweep — launched ${TS}"
    echo "PROJECT: ${PROJECT_ROOT}"
    echo "SWEEP_DIR: ${SWEEP_DIR}"
    echo "TOKEN_LIST: ${TOKEN_LIST} (${N_TOKENS} tokens)"
    echo ""
    echo "Per-GPU PIDs:"
    for GPU in 0 1 2 3; do
        echo "  gpu${GPU}: PID=${PIDS[$GPU]} layers=${GPU_LAYERS[$GPU]} log=${LOG_DIR}/m1a_sweep_${TS}_gpu${GPU}.log"
    done
    echo ""
    echo "Total: 8 layers × 100 scenes = 800 passes, ~20-30min wall-clock."
    echo ""
    echo "How to check progress:"
    echo "  pgrep -af run_attention_probe   # workers running?"
    echo "  for L in 00 04 08 12 16 20 24 27; do echo \"L\${L}: \$(ls ${SWEEP_DIR}/L\${L}/*.pt 2>/dev/null | wc -l) tensors\"; done"
    echo "  tail -5 ${LOG_DIR}/m1a_sweep_${TS}_gpu*.log"
} > "${STATUS_FILE}"
cat "${STATUS_FILE}"

echo ""
echo "[sweep] ALL 4 GPU workers detached. Check ${STATUS_FILE} or:"
echo "        tail -f ${LOG_DIR}/m1a_sweep_${TS}_gpu0.log"

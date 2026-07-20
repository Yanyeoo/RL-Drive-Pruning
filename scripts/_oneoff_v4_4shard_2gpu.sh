#!/usr/bin/env bash
# One-off driver: V4 navtest sweep on full 4 shards, 2 GPUs serial-x-2.
#   GPU 0: shard0 → shard1
#   GPU 1: shard2 → shard3
#
# Each shard call is a single invocation of run_m1b_freelunch_sweep.sh with
# VARIANTS="V4", a specific SCENE_FILTER and a fixed GPU id. Wall ~6500s
# per shard → total ~3.6h (two shards back-to-back per GPU).
#
# Designed to be launched twice in parallel (one per GPU lane) via nohup &.
# Usage:
#   bash _oneoff_v4_4shard_2gpu.sh 0 shard0 shard1   # lane A, GPU 0
#   bash _oneoff_v4_4shard_2gpu.sh 1 shard2 shard3   # lane B, GPU 1
#
# Created 2026-06-26 by builder agent for V4 spec §6 sweep, post-discovery
# that SCENE_FILTER=navtest_local_filtered does not exist in navsim hydra
# and the canonical comparison set is the 4-shard split used by V2/V3.
set -euo pipefail

GPU="${1:?usage: $0 <gpu> <shard_a> <shard_b>}"
SHARD_A="${2:?missing shard_a}"
SHARD_B="${3:?missing shard_b}"
TS_TAG="${4:-20260616_154858}"

ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
SWEEP="${ROOT}/scripts/run_m1b_freelunch_sweep.sh"
LOG_DIR="${ROOT}/logs/m1b_v4"
mkdir -p "${LOG_DIR}"

NOW=$(date +%Y%m%d_%H%M%S)
LANE_LOG="${LOG_DIR}/v4_lane_g${GPU}_${NOW}.log"

echo "[lane g${GPU}] start ${NOW}  shards=${SHARD_A},${SHARD_B}" | tee -a "${LANE_LOG}"

for SH in "${SHARD_A}" "${SHARD_B}"; do
  FULL_SCENE="navtest_local_filtered_${SH}_${TS_TAG}"
  echo "" | tee -a "${LANE_LOG}"
  echo "[lane g${GPU}] ===== ${SH}  scene=${FULL_SCENE}  ts=$(date +%H:%M:%S) =====" | tee -a "${LANE_LOG}"
  # Run sweep for V4 only on this shard / this GPU. Append to lane log.
  VARIANTS="V4" SCENE_FILTER="${FULL_SCENE}" GPU="${GPU}" TIMEOUT=14400 \
    bash "${SWEEP}" 2>&1 | tee -a "${LANE_LOG}"
  echo "[lane g${GPU}] ${SH} done at $(date +%H:%M:%S)" | tee -a "${LANE_LOG}"
done

echo "" | tee -a "${LANE_LOG}"
echo "[lane g${GPU}] LANE DONE at $(date +%H:%M:%S)" | tee -a "${LANE_LOG}"

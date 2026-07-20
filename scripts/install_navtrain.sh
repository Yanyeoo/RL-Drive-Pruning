#!/bin/bash
# ============================================================================
# install_navtrain.sh
#
# Run AFTER scripts/download_navtrain_robust.sh finishes successfully.
#
# Moves the staged navtrain dataset into the live OPENSCENE_DATA_ROOT layout
# expected by NAVSIM hydra configs (navtrain.yaml -> data_split: trainval):
#
#   ${OPENSCENE_DATA_ROOT}/navsim_logs/trainval/
#   ${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval/
#
# This step is intentionally separate from the downloader so a partial
# download never pollutes the live navtest tree.
#
# Safety:
#   - refuses to overwrite an existing non-empty trainval target
#   - uses rsync -a --remove-source-files then rmdir on success
#   - both staging and live root are on the same ceph mount -> rename-fast
# ============================================================================
set -euo pipefail

STAGING="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain"
LIVE_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2"

LIVE_LOGS="${LIVE_ROOT}/navsim_logs/trainval"
LIVE_BLOBS="${LIVE_ROOT}/sensor_blobs/trainval"

# Note: meta_datas tar extracts to `trainval_navsim_logs/trainval/*.pkl`
# (one extra level), so we move the inner `trainval/` directly to
# `${LIVE_LOGS}` (= ${OPENSCENE_DATA_ROOT}/navsim_logs/trainval).
STAGE_LOGS="${STAGING}/trainval_navsim_logs/trainval"
STAGE_BLOBS="${STAGING}/trainval_sensor_blobs/trainval"

log() { echo "[$(date '+%F %T')] $*"; }

# ------------------------------------------------------------------- preflight
[[ -d "${STAGE_LOGS}"  ]] || { echo "FATAL: missing ${STAGE_LOGS}";  exit 1; }
[[ -d "${STAGE_BLOBS}" ]] || { echo "FATAL: missing ${STAGE_BLOBS}"; exit 1; }

if [[ -d "${LIVE_LOGS}"  && -n "$(ls -A "${LIVE_LOGS}"  2>/dev/null)" ]]; then
    echo "FATAL: ${LIVE_LOGS} already exists and is not empty. Refusing to overwrite."
    exit 2
fi
if [[ -d "${LIVE_BLOBS}" && -n "$(ls -A "${LIVE_BLOBS}" 2>/dev/null)" ]]; then
    echo "FATAL: ${LIVE_BLOBS} already exists and is not empty. Refusing to overwrite."
    exit 2
fi

log "preflight ok"
log "staging logs:  $(du -sh ${STAGE_LOGS}  | cut -f1)"
log "staging blobs: $(du -sh ${STAGE_BLOBS} | cut -f1)"

# ------------------------------------------------------------------- move logs
log "=== moving navsim_logs/trainval ==="
mkdir -p "$(dirname "${LIVE_LOGS}")"
# same filesystem -> mv is atomic & instant
mv "${STAGE_LOGS}" "${LIVE_LOGS}"
log "navsim_logs/trainval installed: $(du -sh ${LIVE_LOGS} | cut -f1)"

# ----------------------------------------------------------------- move blobs
log "=== moving sensor_blobs/trainval ==="
mkdir -p "$(dirname "${LIVE_BLOBS}")"
mv "${STAGE_BLOBS}" "${LIVE_BLOBS}"
log "sensor_blobs/trainval installed: $(du -sh ${LIVE_BLOBS} | cut -f1)"

# ------------------------------------------------------------------- cleanup
log "=== cleanup staging artifacts ==="
rmdir "${STAGING}/trainval_sensor_blobs" 2>/dev/null || \
    log "WARN: ${STAGING}/trainval_sensor_blobs not empty, leaving alone"
# leave the ".installed" sentinels and staging dir itself for forensics
log "remaining in staging:"
ls -la "${STAGING}" 2>&1 | head -20 || true

log "=== ALL DONE ==="
log "Final layout:"
log "  ${LIVE_LOGS}"
log "  ${LIVE_BLOBS}"
log ""
log "Sanity check: list one log file and one sensor dir"
# NOTE: `ls "${X}" | head -3` triggers SIGPIPE (rc=141) under `set -euo pipefail`
# because head closes the pipe after 3 lines while ls is still writing.
# This produced a false-positive .chain_failed on 2026-06-23 03:12:18 even
# though all install work had already succeeded.
# See: docs/_internal/incident_2026-06-24_navtrain_chain_failed_false_positive.md
{ ls "${LIVE_LOGS}"  || true; } | head -n 3 || true
{ ls "${LIVE_BLOBS}" || true; } | head -n 3 || true

#!/bin/bash
# ============================================================================
# post_dl_chain.sh
#
# Watcher + chain executor.
# Polls for ${STAGING}/.download_complete; once it appears, runs the full
# post-download chain unattended:
#
#   1. install_navtrain.sh       (mv staged data into OPENSCENE_DATA_ROOT)
#   2. check_navtrain_sanity.py  (verify load + driving_command dist)
#   3. build_m02_splits.py       (probe_A / train_pool / val_pool)
#
# Stops on the first non-zero exit code; writes a clear status banner and
# leaves a final sentinel ${STAGING}/.chain_complete or .chain_failed.
#
# Run with nohup so the user can come back at 22:00 and see results.
# ============================================================================
set -uo pipefail   # NOT -e; we handle each step's exit code explicitly

REPO="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
STAGING="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain"
OPENSCENE_DATA_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2"
NAVSIM_DEVKIT_ROOT="${REPO}/code/third_party/AutoVLA/navsim"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/navsim/bin/python"

CHAIN_LOG="${REPO}/logs/post_dl_chain.log"
DL_DONE="${STAGING}/.download_complete"
CHAIN_OK="${STAGING}/.chain_complete"
CHAIN_FAIL="${STAGING}/.chain_failed"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${CHAIN_LOG}"; }
fail() {
    log "CHAIN FAILED at: $1"
    echo "FAIL $(date '+%F %T') step=$1" > "${CHAIN_FAIL}"
    exit 1
}

mkdir -p "$(dirname "${CHAIN_LOG}")"
log "==================== post_dl_chain.sh started ===================="
log "watching for: ${DL_DONE}"

# --- step 0: wait for download sentinel ------------------------------------
WAITED=0
while [[ ! -f "${DL_DONE}" ]]; do
    sleep 60
    WAITED=$((WAITED + 60))
    if (( WAITED % 600 == 0 )); then
        log "still waiting (${WAITED}s elapsed) ..."
        # also detect download process death without sentinel
        if ! pgrep -f download_navtrain_robust.sh > /dev/null; then
            log "WARN: download_navtrain_robust.sh no longer running but no sentinel yet"
            log "  last lines of download log:"
            tail -5 "${REPO}/logs/navtrain_download_aria2.log" 2>&1 | tee -a "${CHAIN_LOG}"
            log "  will keep polling for sentinel for another 10min then bail"
            sleep 600
            if [[ ! -f "${DL_DONE}" ]]; then
                fail "download_died_without_sentinel"
            fi
        fi
    fi
done
log "sentinel detected after ${WAITED}s"
log "sentinel content: $(cat "${DL_DONE}")"

# --- step 1: install_navtrain.sh -------------------------------------------
log "==================== STEP 1/3: install_navtrain.sh ===================="
bash "${REPO}/scripts/install_navtrain.sh" 2>&1 | tee -a "${CHAIN_LOG}"
rc=${PIPESTATUS[0]}
[[ $rc -eq 0 ]] || fail "install_navtrain.sh rc=$rc"
log "install_navtrain.sh OK"

# --- step 2: check_navtrain_sanity.py --------------------------------------
log "==================== STEP 2/3: check_navtrain_sanity.py ==============="
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT}" \
    PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}" \
    "${PY}" "${REPO}/scripts/check_navtrain_sanity.py" 2>&1 | tee -a "${CHAIN_LOG}"
rc=${PIPESTATUS[0]}
[[ $rc -eq 0 ]] || fail "check_navtrain_sanity.py rc=$rc"
log "check_navtrain_sanity.py OK"

# --- step 3: build_m02_splits.py -------------------------------------------
log "==================== STEP 3/3: build_m02_splits.py ===================="
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT}" \
    PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}" \
    "${PY}" "${REPO}/scripts/build_m02_splits.py" \
        --out-dir "${REPO}/data/splits" \
        --seed 0 2>&1 | tee -a "${CHAIN_LOG}"
rc=${PIPESTATUS[0]}
[[ $rc -eq 0 ]] || fail "build_m02_splits.py rc=$rc"
log "build_m02_splits.py OK"

# --- done ------------------------------------------------------------------
log "==================== ALL STEPS COMPLETED OK ===================="
echo "OK $(date '+%F %T')" > "${CHAIN_OK}"
log "wrote success sentinel: ${CHAIN_OK}"
log "M0 status: install + sanity + m02 splits all green"
log "outputs in: ${REPO}/data/splits/{probe_A.txt,train_pool.txt,val_pool.txt}"

exit 0

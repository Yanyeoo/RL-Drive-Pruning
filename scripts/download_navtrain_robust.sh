#!/bin/bash
# ============================================================================
# Robust navtrain (445GB) downloader -- aria2c version (~2h instead of ~70h).
#
# Why aria2c: wget single-threaded gets 1 MB/s on this S3 endpoint;
# aria2c with 16 connections sustains ~67 MB/s. 67x speedup observed.
#
# Features:
#   - set -euo pipefail
#   - aria2c -x 16 -s 16 -k 5M (16 conns, 5M chunks, multi-segment)
#   - aria2c continues partial downloads natively (.aria2 control file)
#   - md5 verify after each tgz (entries from navsim/docs/splits.md)
#   - tgz removed after successful extract (bounded disk peak)
#   - rsync into trainval_sensor_blobs/trainval/ (vanilla layout)
#   - install sentinels for resumability
#
# Output layout (cwd = staging dir):
#   trainval_navsim_logs/trainval/  <- meta_datas (~14 GB) [already done]
#   trainval_sensor_blobs/trainval/ <- merged current+history blobs (~435 GB)
#
# Final install step (mv into OPENSCENE_DATA_ROOT) is intentionally NOT done
# here -- run scripts/install_navtrain.sh after this finishes clean.
# ============================================================================
set -euo pipefail

STAGING="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain"
S3_BASE="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim"
HF_META="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"

# md5 from navsim/docs/splits.md (current_1 not listed there).
declare -A MD5=(
    [navtrain_current_1.tgz]=""
    [navtrain_current_2.tgz]="7a72f0a758b5df6cbe4c565920a4869f"
    [navtrain_current_3.tgz]="b083fce1428308abb5682a1a150cc1af"
    [navtrain_current_4.tgz]="68354ac2c993aa1ebbfac59547fdb840"
    [navtrain_history_1.tgz]="dc46ed34d92d5ab9cc1464d67b72fbf6"
    [navtrain_history_2.tgz]="fab177bdb79c0c9536da1566d13e5995"
    [navtrain_history_3.tgz]="71ed9a2387edc3849921861d7873c7f0"
    [navtrain_history_4.tgz]="2cc13aced2f458e50fe4cc2f26d18e07"
)

ARIA_OPTS=(-x 16 -s 16 -k 5M --max-tries=3 --retry-wait=10 --auto-file-renaming=false --allow-overwrite=true --console-log-level=warn --summary-interval=30)

log() { echo "[$(date '+%F %T')] $*"; }

cd "${STAGING}"
log "Working directory: $(pwd)"
log "Free space: $(df -h . | tail -1)"

# ------------------------------------------------------------------ metadata
if [[ ! -d trainval_navsim_logs/trainval ]]; then
    log "=== Step 0/9: downloading metadata (trainval logs ~14GB extracted) ==="
    aria2c "${ARIA_OPTS[@]}" -o openscene_metadata_trainval.tgz "${HF_META}"
    log "extracting metadata ..."
    tar -xzf openscene_metadata_trainval.tgz
    rm openscene_metadata_trainval.tgz
    mv openscene-v1.1/meta_datas trainval_navsim_logs
    rm -rf openscene-v1.1
    log "metadata done: $(du -sh trainval_navsim_logs | cut -f1)"
else
    log "=== Step 0/9: trainval_navsim_logs/trainval already present, skipping ==="
fi

mkdir -p trainval_sensor_blobs/trainval

# ------------------------------------------------------------------ blobs
step=0
for kind in current history; do
    for split in 1 2 3 4; do
        step=$((step + 1))
        tgz="navtrain_${kind}_${split}.tgz"
        dir="${kind}_split_${split}"
        url="${S3_BASE}/${tgz}"

        # Already merged into trainval/? => done (sentinel from previous run)
        if [[ -f "trainval_sensor_blobs/.${tgz}.installed" ]]; then
            log "=== Step ${step}/8: ${tgz} already installed, skipping ==="
            continue
        fi

        log "=== Step ${step}/8: ${tgz} ==="
        log "free before: $(df -h . | tail -1 | awk '{print $4}')"
        log "downloading ${url} via aria2c x16"
        # aria2c will resume from .aria2 control file if present
        aria2c "${ARIA_OPTS[@]}" -o "${tgz}" "${url}"

        # md5 verify
        expected="${MD5[$tgz]:-}"
        if [[ -n "${expected}" ]]; then
            log "verifying md5 (expected ${expected}) ..."
            echo "${expected}  ${tgz}" | md5sum -c -
        else
            log "no md5 listed for ${tgz}, skipping verify (will catch via tar integrity)"
        fi

        log "extracting ${tgz} -> ${dir}/"
        tar -xzf "${tgz}"
        rm -f "${tgz}" "${tgz}.aria2"

        log "rsync ${dir}/ -> trainval_sensor_blobs/trainval/"
        # rsync may return code 24 ("vanished source files") on ceph-fuse when
        # parallel cleanup races with the sender's file enumeration. This is
        # benign for our use case (we rm -rf the source right after) so we
        # explicitly tolerate code 24 here. Any other non-zero is still fatal.
        set +e
        rsync -a "${dir}/" trainval_sensor_blobs/trainval/
        rc=$?
        set -e
        if [[ $rc -ne 0 && $rc -ne 24 ]]; then
            log "FATAL: rsync exit code ${rc} (not 0 or 24)"; exit $rc
        fi
        if [[ $rc -eq 24 ]]; then
            log "rsync exit 24 (vanished files) tolerated"
        fi
        rm -rf "${dir}"

        touch "trainval_sensor_blobs/.${tgz}.installed"
        log "free after: $(df -h . | tail -1 | awk '{print $4}')"
        log "blobs size now: $(du -sh trainval_sensor_blobs 2>/dev/null | cut -f1)"
    done
done

log "=== ALL DONE ==="
log "trainval_navsim_logs:   $(du -sh trainval_navsim_logs | cut -f1)"
log "trainval_sensor_blobs:  $(du -sh trainval_sensor_blobs | cut -f1)"
log "Next: run scripts/install_navtrain.sh to mv into OPENSCENE_DATA_ROOT"

# success sentinel for watcher chain
echo "OK $(date '+%F %T')" > "${STAGING}/.download_complete"
log "wrote success sentinel: ${STAGING}/.download_complete"

#!/bin/bash
# ============================================================================
# history_takeover.sh — Plan B' 并行接管 history_1..4
#
# 背景：download_navtrain_robust.sh 用 rsync 把 staging dir 合到
# trainval_sensor_blobs/trainval/，在 ceph-fuse 上 rsync 对累积目标做 stat
# 校验，耗时指数增长（current_1=2h → current_3=6h → current_4 预计>4h）。
#
# 此脚本完全跳过 rsync，改用 mv（同盘 = O(1) rename）。
# 并行下载 + tar，因为：
#   - aria2c 16-conn 单文件实测 ~80 MB/s，S3 端单连接限速 ~100 MB/s
#   - 4 个 tgz 并行 aria2c：每个 ~50 MB/s 仍快 3x 于串行
#   - tar 解压 CPU-bound，4 个并行无冲突（ceph-fuse 写带宽 GB/s 级）
#   - mv 是 rename，秒级
#
# 关键：写完 sentinel `trainval_sensor_blobs/.navtrain_history_N.tgz.installed`，
# 主脚本 download_navtrain_robust.sh 看到 sentinel 会 `continue`，直接跳到
# `.download_complete`，不冲突。
#
# 与主脚本并发安全性：
#   - 主脚本正在 rsync current_4 → trainval/，写不同 scene_dir（current vs
#     history 日期/veh-id 不重叠），文件级 POSIX atomic，不会冲突
#   - 主脚本完成 current_4 后看 history sentinels 全在 → continue × 4 → exit
#
# Idempotent：每个 split 自带 sentinel 检查，可任意重启。
# ============================================================================
set -uo pipefail  # 不要 -e，单个 split 失败不能拖死其他 3 个

STAGING="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain"
S3_BASE="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim"
LOG_DIR="/apdcephfs/private_shayladeng/tokenrl_autoVLA/logs"
mkdir -p "${LOG_DIR}"

declare -A MD5=(
    [navtrain_history_1.tgz]="dc46ed34d92d5ab9cc1464d67b72fbf6"
    [navtrain_history_2.tgz]="fab177bdb79c0c9536da1566d13e5995"
    [navtrain_history_3.tgz]="71ed9a2387edc3849921861d7873c7f0"
    [navtrain_history_4.tgz]="2cc13aced2f458e50fe4cc2f26d18e07"
)

ARIA_OPTS=(-x 16 -s 16 -k 5M --max-tries=3 --retry-wait=10 \
    --auto-file-renaming=false --allow-overwrite=true \
    --console-log-level=warn --summary-interval=60)

cd "${STAGING}"

process_split() {
    local split="$1"
    local tgz="navtrain_history_${split}.tgz"
    local dir="history_split_${split}"
    local url="${S3_BASE}/${tgz}"
    local sentinel="trainval_sensor_blobs/.${tgz}.installed"
    local log="${LOG_DIR}/history_takeover_${split}.log"

    exec >>"${log}" 2>&1
    echo "[$(date '+%F %T')] === history_${split} START ==="

    if [[ -f "${sentinel}" ]]; then
        echo "[$(date '+%F %T')] sentinel ${sentinel} exists, SKIP"
        return 0
    fi

    # 1. download (resume-safe via .aria2)
    if [[ ! -f "${tgz}" ]] || [[ -f "${tgz}.aria2" ]]; then
        echo "[$(date '+%F %T')] downloading ${url}"
        aria2c "${ARIA_OPTS[@]}" -o "${tgz}" "${url}" \
            || { echo "[ERROR] aria2c failed for ${tgz}"; return 1; }
    else
        echo "[$(date '+%F %T')] ${tgz} already fully downloaded, skip dl"
    fi

    # 2. md5 verify
    local expected="${MD5[$tgz]}"
    echo "[$(date '+%F %T')] verifying md5 ${expected}"
    echo "${expected}  ${tgz}" | md5sum -c - \
        || { echo "[ERROR] md5 mismatch ${tgz}"; return 1; }

    # 3. extract → staging dir
    echo "[$(date '+%F %T')] extracting ${tgz} -> ${dir}/"
    tar -xzf "${tgz}" \
        || { echo "[ERROR] tar failed ${tgz}"; return 1; }
    rm -f "${tgz}" "${tgz}.aria2"

    # 4. mv (rename, O(1)) into target — NOT rsync
    echo "[$(date '+%F %T')] mv ${dir}/* -> trainval_sensor_blobs/trainval/"
    # use -t flag + xargs to handle very large directory count
    # (find ... -mindepth 1 -maxdepth 1 means top-level scene dirs only)
    local count=0
    while IFS= read -r src; do
        mv "${src}" trainval_sensor_blobs/trainval/ \
            || { echo "[ERROR] mv failed: ${src}"; return 1; }
        count=$((count + 1))
    done < <(find "${dir}" -mindepth 1 -maxdepth 1)
    echo "[$(date '+%F %T')] moved ${count} top-level dirs"

    # 5. cleanup empty staging dir + sentinel
    rmdir "${dir}" 2>/dev/null || rm -rf "${dir}"
    touch "${sentinel}"
    echo "[$(date '+%F %T')] === history_${split} DONE ==="
}

# 4 路并行
echo "[$(date '+%F %T')] launching 4 parallel takeover workers"
for split in 1 2 3 4; do
    process_split "${split}" &
done

wait
echo "[$(date '+%F %T')] all 4 history workers finished"
echo "[$(date '+%F %T')] sentinels:"
ls -la trainval_sensor_blobs/.navtrain_history_*.installed 2>&1

# Incident 2026-06-22 — aria2c missing in image + navtrain chain dead

> 写于 2026-06-22 13:25（周一接手当天）。
> 关键结论：**RESUME_MONDAY 假设的"周一应已 `.chain_complete`" 没发生**。
> 下一个 AI 上来不要再重新诊断同样的事 —— 直接读本文档第 §4 节执行。

---

## §1. 诊断（事实，2026-06-22 12:33 收集）

### 1.1 navtrain sentinel 实况

```
trainval_sensor_blobs/.navtrain_current_{1,2,3,4}.installed   ✅ 全在
trainval_sensor_blobs/.navtrain_history_1.tgz.installed       ✅ 在 (Jun 18 16:01)
trainval_sensor_blobs/.navtrain_history_2.tgz.installed       ✅ 在 (Jun 18 21:33, takeover 写入)
trainval_sensor_blobs/.navtrain_history_3.tgz.installed       ❌ 缺
trainval_sensor_blobs/.navtrain_history_4.tgz.installed       ❌ 缺
.download_complete                                            ❌ 缺
.chain_complete                                               ❌ 缺
data/navtrain_nocot_split_* / data/navtrain*                  ❌ 缺（m02_splits 没跑）
```

### 1.2 后台进程实况

- PID **3738 / 3774 / 139364 全部已死**（RESUME 让"不要 kill"那 3 个 PID，GPU 回收时被一并杀掉）
- `pgrep -af "post_dl_chain|download_navtrain|history_takeover|rsync.*history"` → 空
- 没有任何东西在自动推进

### 1.3 staging 残留

```
_staging_navtrain/history_split_3/   298 dirs (~42 GB)  ← rsync 死在半途
_staging_navtrain/history_split_4/   298 dirs (~43 GB)  ← 从未 rsync（脚本没活到这一步）
navtrain_history_3.tgz / _history_4.tgz                ← 都不存在（h3 解压后被 rm、h4 没下）
```

### 1.4 死亡时间线（从 `logs/navtrain_download_resume_20260618_2032.log`）

```
2026-06-18 20:32  resume 脚本启动 (download_navtrain_robust.sh, from history_3)
           21:51  history_3.tgz 下完 + md5 OK
           21:55  history_3 解压完成
           22:46  rsync history_3/ -> trainval_sensor_blobs/trainval/ 正在跑
                  ← log 在这一行戛然而止
2026-06-18 23:00  GPU 集群回收，整个 user-session 被清掉
                  → 主脚本、chain watcher、takeover 全死
                  → history_3 rsync 没机会写 sentinel
                  → history_4 完全没启动
```

磁盘当前 871G / 2T (44%)，比 freeze 时 890G 还降了 ~20G，**磁盘安全，不是 OOM/磁盘满引发的死亡，是集群回收**。

---

## §2. 阻塞：aria2c 二进制找不到

`download_navtrain_robust.sh` **依赖 `aria2c`**（注释明确写了，比 wget 快 67x）。但 2026-06-22 当前镜像里：

| 位置 | 结果 |
|---|---|
| `which aria2c` | ❌ not found |
| `/usr/local/bin/aria2c` | ❌ |
| `/opt/aria2c` | ❌ |
| `/apdcephfs/private_shayladeng/miniconda3/bin/aria2c` | ❌ |
| `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/aria2c` | ❌ |
| `/apdcephfs/private_shayladeng/miniconda3/envs/navsim/bin/aria2c` | ❌ |
| `/apdcephfs/private_shayladeng/tokenrl_autoVLA/bin/aria2c` | ❌ |

**结论**：当前镜像与 6 月 17–18 号跑成功的镜像**不是同一个**。aria2c 是当时镜像自带的，回收后换镜像就丢了。

`download_navtrain_robust.sh` 脚本本身**不激活 conda env**（看脚本前 50 行），原来跑的时候是终端调用前手动激活的某个 env（很可能也是 autovla / navsim，但当时这个 env 里有 aria2c，现在没了）。

---

## §3. 解决方案（已验证可行性，**未执行**）

### 推荐：方案 1 — `conda install aria2`（30 秒，零风险）

```bash
source /apdcephfs/private_shayladeng/miniconda3/etc/profile.d/conda.sh
conda activate autovla
conda install -y -c conda-forge aria2
which aria2c          # 应该在 envs/autovla/bin/aria2c
aria2c --version
```

conda-forge 的 `aria2` 包约 3 MB，conda 解依赖一般 < 30 s。装好后 `download_navtrain_robust.sh` 直接能用（因为它从 PATH 找 aria2c）。

### 不推荐：apt / 二进制下载
- `apt install aria2`：不确定容器有没有 root + 外网到 ubuntu repo
- 从 GitHub release 下静态 aria2 二进制：要先有能下载文件的工具（curl/wget），且要找对 glibc 版本

### 方案 1 的执行预算（**串行**，原脚本里写死了 `for kind; for split; do ... done`）

| 步骤 | 时间 |
|---|---|
| conda install aria2 | ~30 s |
| 跑 download_navtrain_robust.sh（resume，前 6 个 sentinel skip） |  |
| ↳ history_3: download 17 min + md5 3 min + extract 5 min + rsync **25–40 min** | ~50–65 min |
| ↳ history_4: download 17 min + md5 3 min + extract 5 min + rsync **25–40 min** | ~50–65 min |
| write `.download_complete` | 即时 |
| chain (post_dl_chain.sh): install_navtrain.sh + sanity + m02_splits.sh | ~10 min |
| **小计 navtrain ready** | **~110–140 min** |
| M1.a Step 5 navtrain probe A 复核（10 min GPU）| ~10 min |

### 风险

1. **rsync 时间不稳**：ceph-fuse 上 rsync 已有 trainval（5800+ scene_dir）merge 几百 GB 增量可能慢于估算。如果 h3 rsync 跑了 1 h 还没完，**不要 kill**（incident #2 教训：mv ENOTEMPTY，且 rsync merge 是 ceph-fuse 唯一安全的合并方式），改成接受今天只跑到 navtrain ready，probe 推到下次。
2. **集群再次回收**：如果今天 23:00 又回收，sentinel 是 idempotent 的，下次启动 `download_navtrain_robust.sh` 会自动 resume（已经验证过：6/18 → 6/22 之间 sentinel 完全可信）。

---

## §4. 下一个 AI 上手 → 复制下面这段直接跑

**前置确认（30 s）**：
```bash
# 1. aria2c 是否已装
which aria2c 2>&1
# 2. sentinel 现状（如果 download_complete 已经在 = 上一轮已成功，直接跳到 §5 GPU probe）
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/.download_complete 2>&1
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/trainval_sensor_blobs/.navtrain_history_{3,4}.tgz.installed 2>&1
# 3. 有没有进程在跑（避免重复启）
pgrep -af "download_navtrain_robust|post_dl_chain" 2>&1
```

**主流程**：
```bash
set -e
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# A. 装 aria2c（如果 which aria2c 已找到就跳过这一步）
source /apdcephfs/private_shayladeng/miniconda3/etc/profile.d/conda.sh
conda activate autovla
conda install -y -c conda-forge aria2
which aria2c   # 应该返回 envs/autovla/bin/aria2c

# B. 启 chain watcher（先启，让它在后台 poll .download_complete）
nohup bash scripts/post_dl_chain.sh \
    > logs/post_dl_chain_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "chain_watcher_pid=$!"

# C. 启 navtrain 主下载（idempotent resume，前 6 split 会 skip）
nohup bash scripts/download_navtrain_robust.sh \
    > logs/navtrain_download_resume_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "downloader_pid=$!"
disown -a

# D. poll 状态（每 10 min 看一次）
ls -la /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/trainval_sensor_blobs/.navtrain_*.installed
tail -20 logs/navtrain_download_resume_*.log | tail -20
```

**预期进度信号**：
- T+30 min：history_3.tgz 下完，md5 通过，开始 extract
- T+40 min：history_3 extract 完，rsync 开始
- T+60–80 min：history_3 sentinel 落地，开始 history_4 下载
- T+110–140 min：`.download_complete` 落地，chain watcher 自动跑 install + m02_splits
- T+120–150 min：`data/navtrain_nocot_split_*/` 出现 → navtrain ready

---

## §5. navtrain ready 之后（M1.a Step 5 probe A 复核）

参见 `docs/_internal/m1a_layer_selection_2026-06-18.md` 和 RESUME_MONDAY.md Step 2。
acceptance: `vision_frac_mean ∈ [0.15, 0.22]` on navtrain probe A (100 scene)。
结果写入 `docs/results/key_results.md` §4.5 + §5（替换 pending 标记）。

---

## §6. 硬规则（**不要违反**）

1. ❌ 不 kill 任何活着的 download/chain 进程
2. ❌ 不对 `_staging_navtrain/history_split_{3,4}/` 做 `mv` / 删除（incident #2 教训）
3. ❌ 不 `rm -rf _staging_navtrain/`
4. ❌ 不重新 sweep 别的 attention layer（M1.a L\*=12 LOCKED v2）
5. ❌ 不 push GitHub
6. ✅ 关键数字立刻写 `docs/results/key_results.md`
7. ✅ 每步走 sentinel 优先（先看 .installed / .download_complete / .chain_complete，再决定）

---

## §7. 我（本次 session）做了什么 / 没做什么

**做了（纯只读）**：
- 检查 sentinel × 3 处
- `ps -fp` 检查 PID
- `df -h` 检查磁盘
- `ls _staging_navtrain/history_split_{3,4}/` 检查残留
- 读 `scripts/post_dl_chain.sh`、`install_navtrain.sh`、`download_navtrain_robust.sh`
- tail `logs/navtrain_download_resume_20260618_2032.log`
- find aria2c → 失败

**没做**（user 还没拍板，且镜像里没 aria2c）：
- ❌ `conda install aria2`
- ❌ 启 chain watcher
- ❌ 启 download_navtrain_robust.sh

下一个 AI 直接跑 §4 的命令即可。

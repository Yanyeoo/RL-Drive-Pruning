# 🌅 明早第一件事 (RESUME navtrain 下载)

> 写于 2026-06-16 20:45。22:00 机器回收，下载没跑完。
> 明早一上来就 **完全无脑跑这一份**。

---

## TL;DR — 三条命令重启全流程

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# 1. 重启下载（自动断点续传）
nohup bash scripts/download_navtrain_robust.sh \
  > logs/navtrain_download_aria2.log 2>&1 &
echo "DOWNLOAD_PID=$!"

# 2. 重启 chain watcher（下载完自动跑 install + sanity + m02_splits）
nohup bash scripts/post_dl_chain.sh \
  > logs/post_dl_chain.log 2>&1 &
echo "CHAIN_PID=$!"

# 3. verify 起来了
sleep 5 && pgrep -af "download_navtrain|post_dl_chain"
```

跑完这三条就可以放着不管了。预计**白天 4-5h 全 chain 完**（明早起 → 中午前 done）。

---

## 为什么能放心 resume

### download_navtrain_robust.sh 的幂等机制（已 verify ✅）

| 已完成的步骤 | resume 时的行为 |
|---|---|
| `trainval_navsim_logs/trainval/` 已存在 | step 0 整段 skip ✅ |
| `.navtrain_current_1.tgz.installed` sentinel | 整个 split 1 skip ✅ |
| 半下载的 `.tgz` + `.aria2` 控制文件 | aria2c 自动从断点续 ✅ |
| 半解压的 `current_split_N/` 目录 | tar 重跑（OK，会覆盖）|
| 半 rsync 的 `trainval_sensor_blobs/trainval/` | rsync 幂等，重跑只复制缺失文件 ✅ |

### 当前回收前的状态（20:45）

- ✅ tgz #1 下完（54GB → 已解压删除）
- 🔄 split_1 rsync 进行中 → trainval_sensor_blobs/trainval/
- ❌ tgz #1 的 `.installed` sentinel 还没写（rsync 没完）
- ❌ 还有 7 个 tgz（current_2/3/4 + history_1/2/3/4）

**所以明天 resume 时**：split_1 会**重跑一遍 rsync**（5min），然后开始下 tgz #2。损失约 5min，可接受。

---

## 监控（明早起来 + 中午前各 cat 一次）

```bash
# 整体进度
tail -30 /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/navtrain_download_aria2.log

# chain 状态
tail -20 /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/post_dl_chain.log

# 看到这个文件 = 全 chain 跑完
ls -la /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/.chain_complete
```

---

## chain 跑完后下一步（M0.2 → M1）

看到 `.chain_complete` 出现后：

1. 把 navtrain 数字写进 `docs/results/key_results.md` §M0.2（按 SOP）
2. 检查 `data/splits/{probe_A.txt,train_pool.txt,val_pool.txt}` 三个文件存在
3. 进 M1.a：attention hook 改造 `code/.../autovla.py:vlm.generate()` (line ~527)
   - 设计在 `docs/_internal/m1a_attention_hook_design.md`（如果还没写就先写）

---

## 如果 resume 出问题（debug 入口）

```bash
# 看 staging 目录现状
ls -la /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/
ls -la /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/trainval_sensor_blobs/.*.installed 2>/dev/null

# 看磁盘
df -h /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/

# 看是不是有遗留 lock
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/_staging_navtrain/*.aria2 2>/dev/null
# 有的话不用动，aria2c 会自动用它续传
```

**已知坑：无**。脚本设计干净。

---

## 速查

- 下载脚本：`scripts/download_navtrain_robust.sh`
- chain 脚本：`scripts/post_dl_chain.sh`
- 关键数字 SOP：`docs/results/key_results.md` + `docs/results/README.md`
- 22h handoff 详细背景：`docs/_internal/handoff_2026-06-16_22h.md`

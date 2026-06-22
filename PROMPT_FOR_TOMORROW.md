# 给下一个 AI 的开场白（直接复制发给它）

> 写于 2026-06-18 16:10。上一个 AI 17:00 被回收，无记忆传递。

---

## 复制下面这一整段发给新 AI：

```
你好，接手 RL-Drive-Pruning 项目。前一个 AI session 在 17:00 被回收，无记忆传递。
GPU 从 4×H20 降到 2×H20。

【第一件事 — 强制】
不要做任何动作。先按顺序读完这 3 个文件：

1. /apdcephfs/private_shayladeng/tokenrl_autoVLA/RESUME_TOMORROW.md
2. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/handoff_2026-06-18_session_death.md
3. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/incident_2026-06-18_false_stall_diagnosis.md

读完用一段话告诉我：
- navtrain 现在在什么状态？（哪些已完成、哪些还在跑、哪些进程不能动）
- 用户已经做了什么决定？（关于 M1.a 在 navtest 还是 navtrain 上跑）
- 你接下来第一个具体动作是什么？（必须是 Step 0 sanity，不能直接跑 smoke）

【硬规则】
1. 不要重启 navtrain 下载脚本或任何后台 rsync
2. 不要对 _staging_navtrain/ 做 tar/mv/rsync 操作（前任踩过 2 次）
3. 不要 push GitHub（没授权）
4. 关键数字（PDMS/L*/sanity）必须当场写 docs/results/key_results.md
5. 不确定就停下来问，不要自己改

【工作目录】
/apdcephfs/private_shayladeng/tokenrl_autoVLA

【当前 milestone】
- M0 baseline ✅ B0 navtest PDMS=89.83 locked
- M0.2 navtrain 数据 🔄 后台 rsync 中（不阻塞）
- M1.a attention probe 🎯 你的任务，已确认在 navtest 上跑

开始读文档。
```

---

## 为什么这么写

- **第一段就给"不要做什么"**：上一个 AI 反复踩同一类坑（看到怪状态就改），新 AI 必须先冷静读文档
- **强制汇报 3 个具体问题**：避免 AI 假装看完就开干，3 个问题答不上来 = 没读
- **GPU 降级单独提**：避免 AI 复用 4 卡分片脚本崩
- **硬规则浓缩成 5 条**：每条都对应今天踩的具体坑
- **第一个动作钉死是 Step 0 sanity**：避免 AI 直接 launch smoke 然后炸

---

## 如果新 AI 不听话

如果它跳过文档直接开始动手，立刻打断：

> 停。你没按规则读文档。把 RESUME_TOMORROW.md 和 handoff_2026-06-18_session_death.md 读完，
> 然后回答我那 3 个问题，再开始干。

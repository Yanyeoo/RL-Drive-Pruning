# RESUME_MONDAY — 周一接手 (2026-06-22 起)

> 写于 2026-06-18 19:30。今天 23:00 GPU 回收，下周一恢复。
> M1.a 已**完全交付**（navtest, n=500, L\*=12 locked）。
> 周一主任务：M1.a Step 5 复核 + M1.b 启动。

---

## ⚠️ BEFORE YOU DO ANYTHING — 2026-06-22 13:25 更新

下面 §0 这段是 2026-06-22 周一当天接手后发现的，**必须先读、覆盖下文 §M0.2 的乐观假设**：

### §0. navtrain chain 实际**没跑完**，且当前镜像里没 aria2c

- ❌ RESUME 原文 §M0.2 的"PID 3738 / 3774 / 139364 后台 rsync 中"已**全部死亡**（6/18 23:00 GPU 回收带走）
- ❌ `.chain_complete` / `.download_complete` 都不存在
- ❌ `history_3.tgz.installed` / `history_4.tgz.installed` sentinel 都不存在（rsync 死在半途）
- ❌ 当前镜像里 `aria2c` 二进制找不到（6/17-18 用的镜像换了，二进制丢了）
- ✅ 前 6 个 sentinel（current_1..4 + history_1, 2）完好，可 resume
- ✅ 磁盘 871G / 2T (44%) 安全

**下一步看这里**：`docs/_internal/incident_2026-06-22_aria2c_missing_chain_dead.md` §4 节有可直接复制粘贴的 5 行命令（conda install aria2 + 启 chain watcher + 启 downloader resume），预算 ~110–140 min 到 navtrain ready，再 +10 min GPU probe。

**已淘汰的指令**（不要再相信下文 §M0.2 这两条）：
- ~~"这 3 个 PID 不要 kill"~~ ← 它们已经自然死亡，pgrep 找不到
- ~~"预计周一应该已 `.chain_complete`"~~ ← 实际没发生

---

## 状态快照（2026-06-18 19:30 freeze）

### M0 baseline
- ✅ B0 navtest PDMS = **89.83** locked （`docs/results/key_results.md §3`）

### M0.2 navtrain 数据下载
- ✅ current_1..4：4 split 已 install 完成，sentinel 全在
- 🔄 history_1..4：后台 rsync 进行中（PID 3738 主脚本 + PID 139364 user takeover）
  - 这 3 个 PID **不要 kill**，慢但收敛
  - history_3 / history_4 还在等主脚本顺序处理
  - **预计周一应该已 `.chain_complete`**（5 天 + 周末，绰绰有余）
  - 如果周一还没 `.chain_complete`：检查 `_staging_navtrain/.../trainval_sensor_blobs/.navtrain_*.installed` sentinel；不要碰 staging dir
- chain：post_dl_chain.sh（PID 3774）poll 中，齐了自动触发 install_navtrain → m02_splits

### M1.a attention layer probing
- ✅ **L\* = 12 LOCKED v2** on navtest（n=500）
  - L12 vision_frac_mean = **0.1861**
  - L27 vision_frac_mean = 0.1805（被淘汰）
  - 三重支持：n=500 数值领先 + 下游 15 层 flop 收益 + fine sweep 孤立尖峰结构
  - 决策文档：`docs/_internal/m1a_layer_selection_2026-06-18.md`
  - 数据：`exp/m1a_layer_sweep_20260618_1644/`（14 layer × n=100 + L12/L27 × n=400 extra = 2200 forward passes）
- 🎯 **唯一 pending**：navtrain probe A 10-min 复核（M1.a Step 5）

### M1.b — 还没动
- 设计：在 L12 上学一个 per-scene pruning policy，剪 vision token，PDMS 不掉
- 周一 navtrain `.chain_complete` 后立即可启

---

## 复制下面这一整段发给新 AI：

```
你好，接手 RL-Drive-Pruning 项目。上一个 AI session 在 2026-06-18 23:00 GPU 回收时收尾，无记忆传递。

【第一件事 — 强制】
不要做任何动作。先按顺序读完这些文件：

1. /apdcephfs/private_shayladeng/tokenrl_autoVLA/RESUME_MONDAY.md      ← 本文件
2. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/results/key_results.md          ← 看 §3 (B0) + §4 (M1.a L*=12)
3. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/m1a_layer_selection_2026-06-18.md  ← L*=12 决策全文
4. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/handoff_2026-06-18_session_death.md
5. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/incident_2026-06-18_false_stall_diagnosis.md

读完用一段话告诉我：
- navtrain 现在状态？(.chain_complete 到了没？current_*/history_* sentinel 全不全？)
- M1.a 状态？(L*=12 锁了吗？navtest n=500 的关键数字是什么？)
- 你接下来第一个具体动作是什么？

【硬规则】
1. 不要重启 navtrain 下载脚本或任何后台 rsync
2. 不要对 _staging_navtrain/ 做 tar/mv/rsync 操作（前任踩过 2 次）
3. 不要 push GitHub（没授权）
4. 关键数字（PDMS/L*/sanity）必须当场写 docs/results/key_results.md
5. 不确定就停下来问，不要自己改
6. M1.a 的 L*=12 是 LOCKED 决策，不要重新 sweep 别的层（除非 probe A 复核失败）

【工作目录】
/apdcephfs/private_shayladeng/tokenrl_autoVLA

【当前 milestone】
- M0 baseline ✅ B0 navtest PDMS=89.83 locked
- M1.a attention probe ✅ L*=12 LOCKED on navtest n=500，唯一 pending = navtrain probe A 10-min 复核
- M0.2 navtrain 数据 🔄 周末后台 rsync，理应已完成；先确认 .chain_complete sentinel
- M1.b RL pruning policy 🎯 你的主任务（前提：navtrain ready + probe A 复核 OK）

【今天周一推荐顺序】
Step 1 (5 min)  确认 navtrain 状态：ls _staging_navtrain/.../*.installed + ls _staging_navtrain/.../.chain_complete
Step 2 (10 min) M1.a Step 5 — navtrain probe A 复核 L*=12（1 卡，100 scene，与 navtest 数字对比）
                跑法：bash scripts/run_m1a_attention_probe.sh --scene-filter navtrain_probeA \
                      --save-dir exp/m1a_navtrain_probeA/L12 --layer-idx 12 --gpu 0 \
                      --token-list <从 navtrain split 选 100 token> --max-scenes 100
                acceptance: vision_frac_mean ∈ [0.15, 0.22]（与 navtest 0.1861 在量级一致）
Step 3 (剩余时间)  M1.b kickoff — 写 spec → smoke → 训练
                需要：L12 attention 作为 input feature 接入 RL policy

开始读文档。
```

---

## 工程速查（写给新 AI 看）

### M1.a 关键产物路径

```
exp/m1a_layer_sweep_20260618_1644/
├── tokens_100.txt                  # coarse 用的 100 token
├── tokens_500.txt                  # n=500 sanity 全集
├── tokens_100_to_499.txt           # 100–499 增量 token
├── L00..L27/                       # 14 layer × 100 .pt（coarse + fine）
├── L12_500_extra/                  # L12 增量 400 .pt
└── L27_500_extra/                  # L27 增量 400 .pt（已淘汰，留作存档）
```

### 重新算 vision_frac

```bash
PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code \
  /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
  -m rldrive.scoring.analyze_layer_sweep \
  --sweep-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644 \
  --layers 0,4,8,10,11,12,13,14,16,20,24,25,26,27
```

数字应当与 `m1a_layer_selection_2026-06-18.md §4.2` 完全一致（L12=0.1789, L27=0.1804 on n=100 only；用 `L12_500_extra` merge 后 L12=0.1861 on n=500）。

### probe A on navtrain（Step 5 模板）

navtrain split 名字按 `m02_splits.sh` 的输出确定（多半是 `data/navtrain_nocot_split_*/`）。从中选 100 个 token（lexical 前 100 即可），跑：

```bash
bash scripts/run_m1a_attention_probe.sh \
    --scene-filter navtrain_probeA \
    --save-dir exp/m1a_navtrain_probeA/L12 \
    --layer-idx 12 --gpu 0 \
    --token-list <navtrain_tokens_100.txt> \
    --max-scenes 100
```

→ 读 100 .pt 算 vision_frac_mean。
- 若 ∈ [0.15, 0.22] → 一致，**M1.a 完全交付，进 M1.b**
- 若 < 0.10 或 > 0.30 → 报告 user，触发 escalate（可能要在 navtrain 上重做 14-layer sweep）

### 不能动的进程（截至 2026-06-18 19:30）

```
PID 3738    主下载脚本，rsync history_1
PID 3774    post_dl_chain.sh，poll .download_complete
PID 139364…  user takeover rsync history_2
```

周一恢复 GPU 时，这些 PID 应该都已自然结束（chain 跑完）。如果还在跑，**仍然不要 kill**。

---

## 为什么这么写

- **首段先列状态快照**：让接手 AI 立刻知道"M1.a 已交付，不要重做"
- **L\*=12 三处提**（快照、prompt、speedlookup）：避免 AI 看到 n=100 时的 L27=0.1804 又起念头切回 L27
- **probe A acceptance 区间写死**：避免新 AI 拿 navtrain probe A 0.17 这种"看起来差不多"的数当 fail
- **Step 1 钉死成 navtrain status check**：避免直接跑 GPU job 后才发现数据没到
- **强调"M1.a 决策 LOCKED 不要重做"**：今天踩过的双峰陷阱不要再来一遍

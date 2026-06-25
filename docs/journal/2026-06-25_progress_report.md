# 2026-06-25 (周四) 全日进度报告 — M1.b₂ Stage 3 落地

**窗口**：10:00 → 24:00（GPU 4× H20 单机）
**实际有效使用**：14:31 → 17:47（计算）+ ~30 min 落档（17:47 → 18:00）
**窗口利用率**：~3.5 h / 14 h ≈ **25%**（剩余 10.5 h 富余）

---

## 0. TL;DR — 今日做了什么 / 拿到了什么

**一句话**：把昨晚 Stage 2 pretokenize 出来的 19,225 个 navtrain token，今天用 4× H20 全部跑出 per-layer × per-head vision-attention 张量，落盘 24 GB，**0 OOM**，**0 死锁**，**8 个数据噪声 assert（attention 已保存，无伤）**，3h16m 收工。

**核心产出**：`exp/m1b2_navtrain_full_alllayers/*.pt` — 19,225 个 `(28, 16, 720)` fp32 张量，是 M1.b₂ Phase 2（learned head-gating policy）的训练输入。

**与昨日（06-24）的关系**：
- 昨日 Stage 1：验码 multilayer attention pipeline（D0 smoke 100 token）✅
- 昨日 Stage 2：navtrain pretokenize 19,225 JSON ✅
- 今日 Stage 3：上面两者合体跑 full-scale ✅ ← **本报告主题**

---

## 1. 时间线（按事件，不是流水）

| 时刻 | 事件 | 备注 |
|---|---|---|
| 10:00 | 窗口开放，AI onboard | 读 `NEXT_AI_HANDOFF_2026-06-25.md` + `PATHS.md` + 昨晚 journal，10 min 上下文 |
| ~10:30 | 与用户对齐方案 | 选「先 dryrun → 4-GPU shard 全量」（方案 1） |
| 14:18 | Step A dryrun 启动（GPU 0, 20 token） | 验证 multilayer pipeline + 跑通 navtrain_avail19k 这个新 scene_filter |
| 14:24 | Dryrun model load 完成 | 130.7 s checkpoint 冷启动 |
| 14:27 | Dryrun PASS | 20/20 OK, 2.88 s/scene, shape `(28,16,720)`, 30 GB GPU peak |
| 14:31 | Step B 4-GPU 全量启动 | `--shard-stride 4 --shard-index k` (k=0..3) |
| 14:44 | 4 shard 全部进入 forward 阶段 | model load + dataset init ~13 min (4 procs 并发) |
| 15:15 (T+30m) | 第一次健康 check | 16% 进度, 2.40 s/scene, 无错 |
| 15:45 (T+60m) | 32% 进度 | 2.36 s/scene, **0 err** |
| 16:16 (T+90m) | 50% 进度 | 2.31 s/scene, **2 err**（首发 trajectory assert, shard 1） |
| 16:52 (T+140m) | 72% 进度 | 2.27 s/scene, **3 err** 累计 |
| 17:21 (T+170m) | 87% 进度 | 2.25 s/scene, 6 err |
| **17:47 (T+196m)** | **4 shard 全部 DONE** | **19,217 OK, 8 err, ~3h16m wall** |
| 17:50 | Acceptance + 落档完成 | journal + key_results §7 + RESUME 顶部 + denylist |
| 17:51 | 全部 GPU 释放 | 0 后台 job 残留 |

---

## 2. 干了哪些工作（按动作分类）

### 2.1 计算工作（GPU）

| 阶段 | 命令骨架 | 资源 | 时长 |
|---|---|---|---|
| **Dryrun** | `run_m1a_attention_probe.sh ... --max-scenes 20 --gpu 0 --all-layers --num-layers 28` | 1× H20 | 8 min（含 130 s 冷启动） |
| **Full run** | `for k in 0..3: ... --shard-stride 4 --shard-index $k --gpu $k` | 4× H20 并发 | 3h16m wall（每 shard ~181 min） |

每 30 min 一次 health watch（GPU 显存、log tail、`.pt` 数、错误扫描），共 7 次 check。

### 2.2 验收工作（CPU）

- File census：`ls *.pt | wc -l` → 19,225（精确等于 token-list）
- Shape 抽检：random 5 + last 3 文件全部 `(28, 16, 720)`, `multi_layer=True`, `num_layers=28`
- Error 归因：8 个 err 全部归因到 `Trajectory poses and sampling have unequal number of poses`（navsim 数据噪声）；定位到 attention `.pt` 在 forward hook 内保存（assert 之前），所以 8 个 err token 的 `.pt` 也已存盘。

### 2.3 落档工作（文档）

| 文件 | 类型 | 大小 |
|---|---|---|
| `docs/journal/2026-06-25_m1b2_stage3_done.md` | 新建 — Stage 3 完整 journal（9 章节, 11 KB） | NEW |
| `docs/results/key_results.md` §7 | 新增 — Stage 3 章节（7 个子表） | +250 行 |
| `docs/results/key_results.md` Quick-ref 表 | 加 M1.b₂ Stage 3 一行 | EDIT |
| `docs/results/key_results.md` Changelog | 加 2026-06-25 17:47 一行 | EDIT |
| `RESUME_MONDAY.md` 顶部 | 覆盖昨晚的 stub，写今日完结状态 | EDIT |
| `exp/m1b2_navtrain_full_alllayers/_stage3_trajectory_err_tokens.txt` | 新建 — 8-token denylist | NEW |
| `docs/journal/2026-06-25_progress_report.md` | 新建 — 本报告 | NEW |

---

## 3. 效果（量化）

### 3.1 数据产出

| 指标 | 数值 |
|---|---|
| **总 `.pt` 文件数** | **19,225 / 19,225 = 100%** |
| 每文件 shape | `per_layer_vision_attn = (28, 16, 720)` |
| 每文件大小 | 1.30 MB avg |
| 张量元数据 | `multi_layer=True`, `average_heads=False`, `layer_idxs[28]`, `vision_token_positions(720,)`, `last_instr_idx`, `prompt_len`, `vision_blocks[3]` |
| 总磁盘 | **24 GB**（pre-flight 预算 25 GB，吻合） |
| 磁盘余量 | `/apdcephfs` 还有 1.2 TB free |

### 3.2 性能（与预期对比）

| 指标 | 实测 | 预算/基线 | 评价 |
|---|---:|---:|---|
| s/scene 平均 | **2.25** | 2.55 (Stage 1 单进程) | ✅ **快 11%**（dataloader IO 并行有红利） |
| Wall（4 GPU 并行） | **3h16m** | < 14h 窗口 | ✅ 25% 利用率 |
| Wall（单 GPU 等效） | ~12 h | — | 单 GPU 跑不完 14h 窗口需要 sharding |
| GPU 显存峰值 | **30.9 GB** / 98 GB | < 50 GB 红线 | ✅ <31% 利用 |
| Shard wall 不均衡 | **3.4 min**（max−min） | — | ✅ stride sharding 自然均衡 |
| Error 率 | **0.042%**（8/19225） | < 0.1% | ✅ |

### 3.3 错误清单（8 个 trajectory-assert）

机制：attention 在 forward hook 内保存，trajectory-pose 验证发生在 forward 之后。所以 8 个 err token 的 attention `.pt` **也已保存**。

```
8d2dd1aea23a5183  67e64fb0e9245ccc  7c7cc0871be859d9  ca8281be07935921
e97dbf85c52d56b9  eecdb97c332f550d  f69d0668f4b8595e  fcad5dfb8da65554
```

下游策略：
- M1.b₂ Phase 2 训练 → 用全 19,225（attention 没问题）
- 未来 navtrain PDMS 评估 → 减掉这 8 个，用 19,217（与 M0/M1.a 安全集一致）

### 3.4 Acceptance gates（全 PASS）

| Gate | Target | Observed | Verdict |
|---|---|---:|:---:|
| `|.pt|` ≈ \|token-list\| | 19,225 ± 10 | 19,225 | ✅ exact |
| Shape `(28, 16, 720)` (抽检) | all pass | 8/8 pass | ✅ |
| `multi_layer=True`, `num_layers=28` | all pass | 8/8 pass | ✅ |
| s/scene avg | ≤ 4.0 | 2.25 | ✅ |
| GPU peak / GPU | < 50 GB | 30.9 GB | ✅ |
| Disk | < 40 GB | 24 GB | ✅ |
| Wall < 14h | <14h | 3h16m | ✅ |
| Error rate | < 0.1% | 0.042% | ✅ |

**8/8 全 PASS**。

---

## 4. 解锁的下一步（按价值排序）

| # | 任务 | 是否吃 GPU | 估时 | 价值 |
|---|---|---|---|---|
| 1 | **M1.b₂ Phase 2 design doc** | ❌ | 1–2 h | **高** — Phase 2 训练的前置依赖 |
| 2 | Per-scene rank-variance 分析（在新的 19,225 张量上） | 轻（CPU 即可） | ~30 min | 中 — 给 Phase 2 提供先验分布 |
| 3 | M1.b₂ Phase 2 训练原型 | 是 | ~2–6 GPU-h | 高 — 真正落地 learned head-gating |
| 4 | navtrain free-lunch sweep（V0/V1/V2/V3 镜像 M1.b₁） | 是 | ~12 GPU-h | 中 — 验证 free-lunch 在 navtrain 是否仍成立 |

今日窗口剩余 ~6 h GPU，可以加塞 #2 + 部分 #3，但建议 **#1 优先**（不烧 GPU 额度，且阻塞后面）。

---

## 5. 风险 / 待办（明确传递给下一个 AI）

- **8 个 trajectory-assert token**：denylist 已落档（`_stage3_trajectory_err_tokens.txt`），下游一定要读。
- **Stage 3 dryrun output `exp/m1b2_stage3_dryrun/` 保留**：作 shape sanity reference；体积 25 MB，可后续清理。
- **scene_filter `navtrain_avail19k`**：当前是 placeholder，被 `--token-list` 短路（参见 `run_attention_probe.py` L159–166）。如未来不用 `--token-list`，需补 filter 实现。
- **Stage 1 multilayer 代码改动从未 merge 到 main 工作流**：本次仍在 dev branch；如要长期复用，需评审一次。

---

## 6. 关键文件清单（一键 onboarding）

| 文件 | 用途 |
|---|---|
| `exp/m1b2_navtrain_full_alllayers/*.pt` | **核心产出** — 19,225 张量 |
| `exp/m1b2_navtrain_full_alllayers/_stage3_trajectory_err_tokens.txt` | 8-token denylist |
| `logs/m1b2_full/shard_{0,1,2,3}.log` | 4 个 shard 完整运行 log |
| `logs/m1b2_full/dryrun.log` | Step A dryrun log |
| `docs/journal/2026-06-25_m1b2_stage3_done.md` | Stage 3 完整 journal（9 章节） |
| `docs/journal/2026-06-25_progress_report.md` | **本报告** — 今日全景 |
| `docs/results/key_results.md` §7 | Stage 3 永久记录（项目级 source-of-truth） |
| `RESUME_MONDAY.md` 顶部 | 下次 AI 入场必读 |

---

## 7. 一句话总结给下个 AI

「**Stage 3 跑完了**，输入 `(N=19225, L=28, H=16, V=720)` 的 per-head vision attention 全在 `exp/m1b2_navtrain_full_alllayers/*.pt`，下一步去写 M1.b₂ Phase 2 design doc，**不需要 GPU**。」

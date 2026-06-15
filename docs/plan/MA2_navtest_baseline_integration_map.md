# MA2 — AutoVLA → NAVSIM navtest 接入地图

> **Created**: 2026-06-15 17:20
> **Status**: 只读探索结论快照（事实层）
> **Purpose**: 把"AutoVLA 跑 navtest EPDMS"这件事拆到可执行粒度，给 `implementation_plan.md` M0.3 / M0.4 提供事实依据
> **Companions**:
> - 子里程碑（可执行清单）：`implementation_plan.md` § M0.3 / M0.4 展开（MA2.1–MA2.5）
> - 测评基准切换背景：`docs/journal/2026-06-15_benchmark_switch_to_navtest.md`
> - 风险登记：`docs/plan/risks.md`

---

## TL;DR

**AutoVLA 自带的 navsim fork 已经把 navtest 评估入口接好 90%。**
目标 split 默认就是 navtest；agent、配置、主入口、ckpt、Qwen base、数据、地图，**全部本地齐备**。
我们要做的不是"接入"，而是 **填路径 + 准备 2 项预处理 + 包一层双卡 dispatch**。

---

## 1. 资产清单（本地实况）

### 1.1 代码资产（AutoVLA 自带）

| 资产 | 路径 | 备注 |
|---|---|---|
| 评估主入口 | `code/third_party/AutoVLA/navsim/navsim/planning/script/run_pdm_score_cot.py` | CoT 版（不是普通 `run_pdm_score.py`） |
| 启动脚本模板 | `code/third_party/AutoVLA/navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh` | `TRAIN_TEST_SPLIT` 默认 `navtest` |
| CV submission 模板 | `code/third_party/AutoVLA/navsim/scripts/submission/run_cv_create_submission_pickle.sh` | 默认也是 navtest |
| Agent 实现 | `code/third_party/AutoVLA/navsim/navsim/agents/autovla_agent.py` | 507 行，AbstractAgent 子类 |
| Agent hydra 配置 | `code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/agent/autovla_agent.yaml` | 指 ckpt + tokenizer |
| 评测 hydra 配置 | `code/third_party/AutoVLA/navsim/navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml` | 用 `OPENSCENE_DATA_ROOT` 拼路径 |
| 训练配置（推断结构用） | `code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` | |

### 1.2 数据 / 模型资产

| 资产 | 路径 | 大小 | 状态 |
|---|---|---|---|
| AutoVLA ckpt | `tokenrl_autoVLA/models/AutoVLA/AutoVLA_PDMS_89.ckpt` | 16 GB | ✅ |
| Qwen2.5-VL-3B base | `tokenrl_autoVLA/models/Qwen2.5-VL-3B-Instruct/` | 7.1 GB | ✅ |
| navtest sensor_blobs | `tokenrl/data/navsim_v2/sensor_blobs/test/` | 116 GB | ✅ |
| navtest navsim_logs | `tokenrl/data/navsim_v2/navsim_logs/test/` | 982 MB | ✅ |
| nuplan-maps-v1.0 | `tokenrl/data/maps/nuplan-maps-v1.0/` | — | ✅ |

> ⚠️ `tokenrl_autoVLA/models/Qwen2.5-VL-3B-Instruct` ≠ AutoVLA agent yaml 默认的相对路径 `./Qwen2.5-VL-3B-Instruct` —— **MA2.3 必须改成绝对路径或建软链**

---

## 2. 评估主流程拆解

`run_pdm_score_cot.py` 的实际调用链：

```
main()
  └─ hydra 装配 cfg (train_test_split=navtest)
  └─ SceneLoader(
        sensor_blobs_path  = $OPENSCENE_DATA_ROOT/sensor_blobs/test,
        navsim_log_path    = $OPENSCENE_DATA_ROOT/navsim_logs/test,
        scene_filter       = navtest 的 token list)
  └─ MetricCacheLoader(metric_cache_path)          ← ⚠️ 必须先 build
  └─ worker_map 把 token 列表切片，每 worker:
       └─ run_pdm_score(scene_loader, agent, metric_cache):
            ├─ agent = instantiate(autovla_agent) + agent.initialize() → load ckpt
            ├─ for each token:
            │    ├─ load metric_cache[token]      (.xz pickle)
            │    ├─ load json_data_path/{token}.json   ← ⚠️ 必须先 preprocess
            │    ├─ agent_input = scene_loader.get_agent_input_from_token(token)
            │    ├─ trajectory = agent.compute_trajectory(agent_input)
            │    │       └─ AutoVLA.predict(features):
            │    │             ├─ get_prompt() → 8 路 image + CoT prompt
            │    │             ├─ vlm.generate() → token ids
            │    │             └─ action_tokenizer.decode_token_ids_to_trajectory()
            │    │                                    → poses [T, 3] (x, y, heading)
            │    └─ pdm_score(trajectory, metric_cache[token]) → EPDMS 子项
            └─ 汇总 csv（per-token）
       merge 各 worker csv → 最终 EPDMS
```

---

## 3. AutoVLA Agent 形态

- **AbstractAgent 子类**：`autovla_agent.py`，关键 method：
  - `get_sensor_config()` → 8 路相机 enabled，**LiDAR 关闭**（`use_lidar=False`）
  - `initialize()` → 加载 ckpt + tokenizer
  - `get_trajectory_sampling()` → `time_horizon=5s, interval_length=0.5s` → **10 个 future pose**（与 navtest 默认一致）
  - `compute_trajectory(agent_input) → Trajectory`
- **输入**：单帧 multi-view（cam_f0/l0/l1/l2/r0/r1/r2/b0）+ 历史 ego state + nav cmd + CoT 文本提示
- **输出**：`Trajectory(poses=[10, 3], sampling=TrajectorySampling(...))` + 副产物 `cot_results` 字符串
- **单帧 forward**：batch=1，单卡，预计 **几百 ms ~ 1 s**（含 VLM generate，待 MA2.4 smoke 实测）

---

## 4. 必须解决的 4 个预制项

### 4.1 metric_cache（navtest）—— ❌ 缺

- `prior work tokenrl/exp/` 下只有 `metric_cache_navhard*` 和 `metric_cache_warmup`，**没有 navtest 的**
- 必须跑：
  ```
  navsim/scripts/evaluation/run_metric_caching.sh \
      train_test_split=navtest
  ```
- 估时 30 min ~ 2 h，跑一次终身复用
- 输出大小估计几 GB（按 navhard 比例外推）

### 4.2 navtest_nocot JSON —— ❌ 缺

- `run_pdm_score_cot.py` line 82 直接 `open(json_data_path/{token}.json)`
- ⚠️ **evaluation 时不再从 scene 反推 instruction**，而是吃**预生成 json**（与训练侧 dataset 一致）
- 必须跑：`tools/preprocessing/nocot_sample_generation.py`（具体入口 MA2.1 摸清）
- AutoVLA 启动脚本中默认路径名是 `navtest_nocot`（不带 CoT 标签）

### 4.3 环境变量 —— 启动脚本要求

启动 `run_pdm_score_cot.py` 必须 export：

| 变量 | 应指向 | 备注 |
|---|---|---|
| `NAVSIM_DEVKIT_ROOT` | `code/third_party/AutoVLA/navsim/navsim` | devkit 源码 |
| `OPENSCENE_DATA_ROOT` | `tokenrl/data/navsim_v2` | **关键**：默认 yaml 用 `$OPENSCENE_DATA_ROOT/{sensor_blobs,navsim_logs}/${train_test_split.data_split}` 自动拼，无需改 yaml |
| `NUPLAN_MAPS_ROOT` | `tokenrl/data/maps/nuplan-maps-v1.0` | |
| `NAVSIM_EXP_ROOT` | `tokenrl_autoVLA/exp/` | 评测结果 csv 落地 |

> 优点：**路径全部由 env var 拼**，我们不动 hydra yaml，干净。

### 4.4 双 H20 并行 dispatch —— 自己写

- AutoVLA 自带 sh 只 `CUDA_VISIBLE_DEVICES=0` 单卡
- 双卡参考 prior work pattern：`tokenrl/code/scripts/run_oracle_navhard_dual_gpu.sh`
  - 按 token list hash 一切二
  - 两条 `run_pdm_score_cot.py` 并行，各占一张卡
  - 最后 merge per-worker csv
- 是约 50 行 shell 的活，**不动 AutoVLA repo 内部**

---

## 5. 已知坑 / 风险

| # | 坑 | 影响 | 缓解 |
|---|---|---|---|
| C1 | AutoVLA 模板 sh 的 `CONFIG_PATH="$./config/..."` 多了个 `$` 字面量 | 启动失败 | 抄出来时去掉 |
| C2 | `run_pdm_score_cot.py` 用 nuplan 自带 `worker_map`，可能依赖 ray/dask | worker 起不来 | 退化为 `worker=single_machine_thread_pool` |
| C3 | `compute_trajectory` 每帧 sensor IO + VLM generate，单帧可能 2–5 s | navtest ~12k token，双卡 8–24 h | MA2.4 先 smoke 5 scene 实测，外推总耗时 |
| ~~C4~~ | ~~`lora_conf.use_lora=false` —— ckpt `AutoVLA_PDMS_89` 不确定是 base 还是 lora-merged~~ | ~~加载 shape 不匹配~~ | **✅ 2026-06-15 17:30 已解除**：inspect 显示 Lightning ckpt，825 张 fp32 tensor 全在 `autovla.vlm.*` 前缀下，**无 lora_A/B、无 base_layer**，参数量 4.07 B 对应 Qwen2.5-VL-3B full-FT/merged，标准加载即可。详见 `scripts/inspect_autovla_ckpt.py` |
| C5 | `pretrained_model_path: ./Qwen2.5-VL-3B-Instruct` 是相对路径 | 找不到 base | 改绝对路径或在工作目录 ln -s |
| C6 | `requires_scene=False` 但主入口仍走 `SceneLoader`+sensor_config | 间接要求 sensor_blobs 齐 | 本地已齐 |
| C7 | navtest 12k token，CoT 输出可能含 wrong-format → trajectory decode 失败 | 部分 token EPDMS 缺失 | autovla_agent 内部已有 fallback，需 MA2.5 抽检比例 |

---

## 6. 子里程碑映射（→ implementation_plan.md M0）

事实层（本文档）→ 可执行层（implementation_plan.md M0 § MA2.1–MA2.5）：

| 子里程碑 | 解决 | 工作量 | 关键输出 |
|---|---|---|---|
| MA2.1 | § 4.2 navtest_nocot preprocessing | 半天写 + 1–3 h 跑 | `data/navtest_nocot/*.json` |
| MA2.2 | § 4.1 navtest metric_cache | 0.5 h 写 + 0.5–2 h 跑 | `exp/metric_cache_navtest/` |
| MA2.3 | § 4.3 env + § 4.4 dual-gpu dispatch | 半天 | `scripts/run_autovla_navtest_dual_gpu.sh` |
| MA2.4 | smoke run + § 5 C3 验证（~~C4 已解除~~） | 半天 + 1 h 跑 | smoke csv + 单帧时延数 |
| MA2.5 | 全量双卡 navtest | 8–24 h 挂机 | **B0 EPDMS 数（M0 验收）** |

---

## 7. 接入完成的退出标准

- [ ] 拿到 navtest EPDMS 数值（M0 验收主指标），落地 `results/baseline_r1.0_navtest.pkl`
- [ ] 落地单帧时延、kept token 数、CoT 失败率，写入 `docs/journal/MA2_b0_navtest.md`
- [ ] 数值满足止损线：EPDMS > prior-work ceiling 0.41（否则触发骨干重评估）
- [ ] dispatcher 可复用：后续 MA3+ 改了模型只换 ckpt 路径就能跑

---

## 8. 文档关系

```
MA2_navtest_baseline_integration_map.md   ← 本文档（事实快照）
        │ feeds
        ▼
implementation_plan.md § M0.3/M0.4         ← 可执行清单（MA2.1–MA2.5）
        │ refers
        ▼
risks.md / journal/MA2_*.md                ← 实施时落地
```

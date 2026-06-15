# 2026-06-15 — MA2 接入地图落盘

> 17:00–17:25 一次"只读探索 AutoVLA → NAVSIM navtest 接入"专项的产出归档。

## 起因

昨天 design freeze 后，今天评测集切到 navtest（见 `2026-06-15_benchmark_switch_to_navtest.md`），
但 implementation_plan.md 的 M0.3 / M0.4 还停留在"AutoVLA 接入 NAVSIM submission pipeline + EPDMS 单元测试"这种抽象描述，
没有事实层支撑（路径、文件、坑、子里程碑都没拆）。
本次专项专门补这一段。

## 做了什么

**只读探索 AutoVLA repo**，没有改一行代码。摸清楚以下事实：

1. AutoVLA 自带 `navsim/` fork，里面 **agent、主入口、配置、启动模板已经做完 90%**
   - 评估主入口：`run_pdm_score_cot.py`（CoT 版本，不是普通 run_pdm_score）
   - Agent：`navsim/agents/autovla_agent.py`（507 行，AbstractAgent 子类）
   - 启动模板：`scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh`，默认 `TRAIN_TEST_SPLIT=navtest`
2. 关键 ckpt 在本地：
   - `tokenrl_autoVLA/models/AutoVLA/AutoVLA_PDMS_89.ckpt` (16G)
   - `tokenrl_autoVLA/models/Qwen2.5-VL-3B-Instruct/` (7.1G)
   - 早先错以为没有，是因为顶层 `/models` 目录是空的（已澄清）
3. navtest 数据 / nuplan-maps 全齐（`tokenrl/data/navsim_v2/{sensor_blobs,navsim_logs}/test` + `tokenrl/data/maps/nuplan-maps-v1.0`）
4. **2 项预处理缺**：
   - navtest_nocot JSON（`run_pdm_score_cot.py` line 82 直接吃预生成 json）
   - navtest metric_cache（`tokenrl/exp/` 只有 navhard 的）
5. **AutoVLA 模板 sh 有 typo**：`CONFIG_PATH="$./config/..."` 多了 `$`
6. **路径机制干净**：default_evaluation.yaml 用 `OPENSCENE_DATA_ROOT` 拼路径，
   只要 env var 指对，无需改 yaml

## 文档落盘

1. **新建** `docs/plan/MA2_navtest_baseline_integration_map.md`（事实快照层）
   - 资产清单、流程拆解、4 项预制项、7 条已知坑
2. **改写** `docs/plan/implementation_plan.md` § M0.3
   - 把抽象 M0.3 展开为 MA2.1–MA2.5 五个子里程碑（命名延续 deprecated milestones.md 的 MA2 标签）
   - M0.4 修订：只保留 navtrain baseline（navtest 部分挪进 MA2.5），并明确"M0.4 与 M1.b 应合并跑省 1× navtrain 算力"
   - M0 验收 / 产物表同步更新
3. **在** `docs/plan/milestones.md` MA2 段加交叉引用指针（不展开内容，保持 deprecated 性质）

## MA2.1–MA2.5 子里程碑全图

| 子里程碑 | 任务 | 工作量 | 关键输出 |
|---|---|---|---|
| MA2.1 | navtest_nocot JSON preprocessing | 半天 + 1–3 h | `data/navtest_nocot/*.json` |
| MA2.2 | navtest metric_cache build | 0.5 h + 0.5–2 h | `exp/metric_cache_navtest/` |
| MA2.3 | 双 H20 dispatcher + 环境变量 | 半天 | `scripts/run_autovla_navtest_dual_gpu.sh` |
| MA2.4 | Smoke run（5 scene） | 半天 + 1 h | 单帧时延 / CoT 失败率实测 |
| MA2.5 | navtest 全量（B0 数字） | 8–24 h 挂机 | `results/baseline_r1.0_navtest.pkl` |

## 一次回滚自纠

写 implementation_plan.md M0.4 时一开始把 navtrain baseline 也合并进 MA2.5，
意识到 navtrain baseline 是 M1/M2 RL/oracle 的 advantage 来源，不是评测产物，立刻回滚。
教训：M0.4 (navtrain) 和 MA2.5 (navtest) 是两份独立的 baseline，名字像但目的不同。

## What's next

接下来要么开 MA2.1（摸 nocot_sample_generation.py 输入输出，但仍只读），
要么开 MA2.3（拆 prior work dual_gpu 脚本，画可复用骨架），
要么直接动手 MA2.1 写 preprocessing 脚本（开始改码）。
等用户裁决。

## Open questions

- Q1（C4 风险）：`AutoVLA_PDMS_89.ckpt` 是 base 还是 lora-merged？需要查 HF 模型卡或 inspect ckpt 内 key prefix
- Q2：navtest token 总数到底是多少？我引用了 "~12k" 这个数（来自直觉），MA2.1 摸数据时务必确认精确数字
- Q3：单帧时延的"几百 ms ~ 1 s"是 prior work 经验外推，AutoVLA + CoT 实际可能更慢，**MA2.4 必须实测**

---

## 17:30 — 续：用户授权"无问自答"模式，开始解 Q1

模式切换：用户授权我此后不再询问，每步自问自选自行推进，只在他主动问进度时汇报。

### Q1 已解决（C4 风险解除）

- 工具：`scripts/inspect_autovla_ckpt.py`（保留为工具，未来每份新 ckpt 都该过一遍）
- 跑在 `envs/autovla` python 上
- 结果：
  - top-level keys = `['pytorch-lightning_version', 'state_dict']` —— **Lightning ckpt**
  - 825 张 tensor，**全部 fp32，全部前缀 `autovla.vlm.*`**
  - **无 lora_A/lora_B，无 base_layer** —— 不是 PEFT 状态
  - 参数总量 **4.07 B**（对应 Qwen2.5-VL-3B）
  - 体积 16 GB ≈ 4.07 B × 4 B fp32，能对上
- **结论**：C4 风险解除（GREEN）。标准 Lightning load_from_checkpoint 即可，无需 lora merge。

### 顺带锤定的事实

- backbone 实际是 **Qwen2.5-VL-3B-Instruct**（之前接入地图已经写对，但口头讨论曾混用 7B；今天确认归 3B）
- ckpt 加载路径要走 **Lightning**（`autovla.vlm.*` 前缀），不能直接 `AutoModel.from_pretrained`；MA2.3 dispatcher 写法要注意
- `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtest.yaml` 已经存在且填了绝对路径 —— **MA2.1 半成**

### 同步的文档修订

- 接入地图 §5 风险表 C4 行 → 划掉 + 加 GREEN 注
- 接入地图 §子里程碑映射表 MA2.4 行 → 移除 C4 验证项
- `implementation_plan.md` MA2.4 段 → 同步移除 C4，留 C3 + C7

### 现状盘点（17:35）

| 事项 | 状态 |
|---|---|
| Q1 / C4 ckpt 风险 | ✅ 解除 |
| MA2.1 navtest_nocot config | ✅ 已就位（绝对路径） |
| MA2.1 实跑产物 `data/navtest_nocot/` | ❌ 空 |
| MA2.2 `exp/metric_cache_navtest/` | ❌ 空 |
| MA2.3 dispatcher | ❌ 待写 |

### 下一步自选

按 Q1 → MA2.1 摸 `nocot_sample_generation.py` → MA2.1 写启动脚本 → 实跑 → MA2.2。下一步：**读 `tools/preprocessing/nocot_sample_generation.py`**。

---

## 18:50 — MA2.1 dry-run 跑通；途中 5 个 fix 落锤

### dry-run 路径

写了 `scripts/run_ma2_1_navtest_nocot_preprocessing.sh`，跑五次才通。每次踩坑都修一处，最终 180s 出 225 个 json。

### Fix 1 — PYTHONPATH

`tools/preprocessing/nocot_sample_generation.py` 第 11 行 `from dataset_utils.preprocessing....`，从 repo root 解析。upstream `run_nuplan_preprocessing.sh` 没设 PYTHONPATH（推测他们的 conda env 自带 hack 或 `python -m`），我们这里必须显式：

```bash
cd ${AUTOVLA_ROOT}
export PYTHONPATH="${AUTOVLA_ROOT}:${PYTHONPATH:-}"
```

### Fix 2 — `NUPLAN_MAPS_ROOT` 多一层

错值：`${MAPS_ROOT}/`
正值：`${MAPS_ROOT}/nuplan-maps-v1.0`

nuplan 的 `get_maps_db` 在 `<NUPLAN_MAPS_ROOT>/nuplan-maps-v1.0.json` 找 metadata，需要该 json 直接在 root 下。该 json 实际位于 `…/maps/nuplan-maps-v1.0/nuplan-maps-v1.0.json`。

### Fix 3 — `sensor_blobs/test` 套娃

Tencent 存储结构：
```
sensor_blobs/test/openscene-v1.1/sensor_blobs/test/<log_name>/CAM_*/...
```
而 navsim 直接拿 `sensor_blobs_path + "<log>/CAM_F0/<hash>.jpg"` 拼，所以 `sensor_blobs_path` 必须指到最内层 `test/`。

dataset yaml 已改：
```yaml
sensor_blobs_path: ${OPENSCENE_DATA_ROOT}/sensor_blobs/test/openscene-v1.1/sensor_blobs/test
```

### Fix 4 — 4 个缺图 log

navsim_logs/test 有 147 pkl，sensor_blobs 只有 143 log 目录，差 4：
- `2021.06.03.13.55.17_veh-35_02419_02561`
- `2021.06.03.17.06.58_veh-35_03860_03992`
- `2021.09.29.14.44.26_veh-28_00337_00504`
- `2021.10.06.07.26.10_veh-52_01245_02064`

这 4 个全在 `navtest.yaml#log_names` 白名单内（共 136 个）。处理：复制一份 `navtest_local_filtered.yaml`，`sed -i /…/d` 删 4 行。dataset yaml `scene_filter` 改指新文件。

### Fix 5 — navtest 实际 token 数

scene_filter 报 **Extracted 11596 scenarios**（去掉 4 个缺图 log 后）。之前直觉 "~12k" 验证为正确。这个数会进 MA2.3 dispatcher 的耗时估算（双卡每卡 5798 token，单帧 1 s 是 1.6 h，单帧 0.5 s 是 50 min）。

### dry-run 性能数

180 s / 8 worker：
- Extracted 11596 scenarios
- Processing 218/11596 in 180s ≈ 1.21 it/s（progress bar）
- 落盘 225 json
- 估算 8 worker 全量 ≈ **1.9 h**，32 worker 乐观 **~30 min**（待实测）

### 现状盘点（18:55）

| 事项 | 状态 |
|---|---|
| MA2.1 启动脚本 | ✅ `scripts/run_ma2_1_navtest_nocot_preprocessing.sh`（5 个 fix 已 inline） |
| MA2.1 navtest_local_filtered.yaml | ✅ AutoVLA repo 内（upstream-only 副本，无侵入） |
| MA2.1 数据 | ⏳ dry-run 225/11596，待全量 |
| MA2.2 metric_cache | ❌ |
| MA2.3 dispatcher | ❌ |

### 下一步自选

(c) 落 journal → (a) 32 worker 挂全量。后台跑约 30-90 min。期间起 MA2.2 / MA2.3 探路并行。

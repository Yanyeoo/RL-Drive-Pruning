# PATHS.md — 项目级关键路径地图

> 维护：每次重大产出后更新（新建数据集 / 新代码入口 / 新 ckpt）。
> 写于 2026-06-24 21:50，下次维护建议：每周或每个 milestone 完成后。

---

## 0. Workspace root

```
WS = /apdcephfs/private_shayladeng/tokenrl_autoVLA
```

下面所有 "相对路径" 都从 `WS` 起。绝对路径中以 `WS/` 标记。

---

## 1. 数据（共享只读 + 项目可写）

### 1.1 共享只读（在 `tokenrl/` 不在 `tokenrl_autoVLA/`，注意！）

| 用途 | 路径 |
|---|---|
| navsim_v2 数据根 | `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/` |
| ├─ trainval pickle | `.../navsim_v2/navsim_logs/trainval/` (1310 个 `.pkl`，每个一个 log) |
| ├─ trainval sensor blobs | `.../navsim_v2/sensor_blobs/trainval/` (1192 个 log 目录) |
| ├─ test pickle | `.../navsim_v2/navsim_logs/test/` |
| ├─ test sensor blobs | `.../navsim_v2/sensor_blobs/test/` (**100% 完整**) |
| ├─ navhard two_stage | `.../navsim_v2/navhard_two_stage/` |
| └─ warmup two_stage | `.../navsim_v2/warmup_two_stage/` |
| maps | `/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0/` |

⚠️ **navtrain partial-download 状况**（2026-06-24 21:20 实测）：
- `sensor_blobs/trainval/` 有 1192 个 log 目录全在；
- 每个 log 内 `CAM_F0..CAM_B0` 8 个 cam 子目录都在，**jpg 同步下载**（要么 8 cam 全有此 frame，要么 8 cam 全没有）；
- 每个 log 实际下载 jpg 数从 9 到几百不等，平均 ~150 jpg / ~625 scene = **~24%/log**；
- navtrain.yaml 的 103,288 trigger token 中，**只 19,225（18.6%）满足 15-frame（±4 历史 + +10 未来）完整 window**。
- 详细诊断：`docs/journal/2026-06-24_m1b2_stage1_2_full_journey.md` §2.3。

### 1.2 项目可写

| 路径 | 内容 | 创建时间 |
|---|---|---|
| `data/navtest_nocot/` | navtest pretokenized json（M1.a 用） | 早期 |
| `data/navtrain_nocot_probe100/` | 100 个 navtrain probe token，cherry-picked 完整 | M1.a setup |
| `data/navtrain_nocot/` | **19,225 navtrain pretokenized json**（100 symlink + 19,125 real） | **2026-06-24 21:41**（本次） |
| `data/navtest_metric_cache/` | navtest metric cache | 早期 |
| `data/splits/` | split 定义 | 早期 |

---

## 2. 模型 / Checkpoint

| 路径 | 内容 |
|---|---|
| `models/Qwen2.5-VL-3B-Instruct/` | 基础 VLM，bf16，2 个 safetensors shard ~6 GB |
| `ckpt/AutoVLA/` | AutoVLA 论文公开 ckpt（baseline 推理用） |

**模型架构关键参数**（Qwen2.5-VL-3B）：
- `num_hidden_layers = 28`
- `num_attention_heads = 16` ⚠️ **不是 24**（2026-06-24 实测修正）
- vision tokens / image = 240（3 cam × 240 = 720 total）

---

## 3. 代码

### 3.1 Top-level
| 目录 | 内容 |
|---|---|
| `code/rldrive/` | 自研 RL / scoring 入口（agents, configs, scoring） |
| `code/third_party/AutoVLA/` | AutoVLA 上游 fork |
| `code/configs/` | 顶层 yaml configs |
| `scripts/` | shell + python helper scripts |
| `tools/` | 杂项 python 工具（**新增于 6-24**） |

### 3.2 关键入口

| 入口 | 路径 |
|---|---|
| **multilayer attention 抽取** | `code/rldrive/scoring/run_attention_probe.py` |
| └─ wrapper（设环境变量、env 激活） | `scripts/run_m1a_attention_probe.sh` |
| **nocot tokenize** | `code/third_party/AutoVLA/tools/preprocessing/nocot_sample_generation.py` |
| dataset config 目录 | `code/third_party/AutoVLA/config/dataset/` |
| scene_filter yaml 目录 | `code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/` |
| AutoVLA 主模型 | `code/third_party/AutoVLA/models/autovla_with_attention.py` |
| Attention capture hook | `code/third_party/AutoVLA/models/attention_capture.py` |

### 3.3 关键 scene_filter yaml
| 文件 | token 数 | 用途 |
|---|---|---|
| `.../scene_filter/navtest.yaml` | 12,146 | navtest split |
| `.../scene_filter/navtrain.yaml` | 103,288 | navtrain split（**论文宣称的全量，但 jpg 不全**） |
| `.../scene_filter/navtrain_probeA.yaml` | 100 | M1.a probe set，cherry-picked |
| `.../scene_filter/navtrain_avail19k.yaml` | **19,225** | **navtrain ∩ 15-frame jpg 完整**（2026-06-24 新建） |

### 3.4 关键 dataset config
| 文件 | scene_filter | 用途 |
|---|---|---|
| `code/.../config/dataset/qwen2.5-vl-3B-navtest.yaml` | navtest | M1.a |
| `code/.../config/dataset/qwen2.5-vl-3B-navtrain_probe100.yaml` | navtrain_probeA | M1.a probe |
| `code/.../config/dataset/qwen2.5-vl-3B-navtrain_full.yaml` | **navtrain_avail19k** | **M1.b₂ stage 2/3**（2026-06-24 新建） |

---

## 4. 实验产出（`exp/`）

| 目录 / 文件 | 内容 |
|---|---|
| `exp/m1a_navtrain_probeA_setup/` | M1.a probe setup（token list, scene filter 源数据） |
| ├─ `tokens_100.txt` | 100 个 probe token |
| ├─ `navtrain_window_clean_tokens.txt` | **19,225 完整 window token（上一个 AI 在 6/24 19:22 已生成）** |
| └─ `navtrain_window_report.json` | 扫描统计 |
| `exp/m1a_navtrain_probeA_L12/` | M1.a L=12 perhead attention 抽取（probe A 结果） |
| `exp/m1a_layer_sweep_20260618_1644/` | M1.a layer sweep 完整结果 |
| `exp/m1a_perhead_L12/` | M1.a perhead 详细分析 |
| `exp/m1b2_d0_smoke_probe100_alllayers/` | **M1.b₂ Stage 1：100 个 .pt，shape (28,16,720)**（2026-06-24 21:08） |
| `exp/m1b2_navtrain_full_window_tokens.txt` | 19,225 token list（与 navtrain_window_clean_tokens.txt **同名单**，重复产出） |
| `exp/m1b2_navtrain_available_tokens.txt` | 152,495（sensor_blobs 实际 jpg 有的 trigger 候选） |
| `exp/m1b2_navtrain_available_intersect.txt` | 103,288（navtrain.yaml ∩ available_tokens） |
| `exp/m1b_phaseF_full_*` | 一系列 6/23-6/24 phaseF run |

### 4.1 Stage 1 输出张量 schema（`exp/m1b2_d0_smoke_probe100_alllayers/<token>.pt`）

```python
{
    "per_layer_vision_attn": Tensor(28, 16, 720),  # 28 layers × 16 heads × 720 vision tokens
    "layer_idxs": [0, 1, ..., 27],                  # 全 28 层 index
    "multi_layer": True,
    "average_heads": False,
    "vision_blocks": [(108,349), (372,613), (636,877)],  # 3 摄像头 block 在 720 序列中的位置
    "token": "...",
    ...
}
```

体积 1.30 MB / 文件（28 × 16 × 720 × 4 bytes ≈ 1.29 MB）。

---

## 5. Conda env / Python

| 项 | 值 |
|---|---|
| autovla env | `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/` |
| python | `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python` (Python 3.9.23) |
| 激活方式 | `source /apdcephfs/private_shayladeng/miniconda3/bin/activate autovla` |

**关键环境变量**（运行 nocot / scoring 前必设）：
```bash
export NUPLAN_MAPS_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2"
export NAVSIM_EXP_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp"
export NUPLAN_EXP_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp"
export PYTHONPATH="/apdcephfs/private_shayladeng/tokenrl_autoVLA/code/third_party/AutoVLA:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
```

或直接 source: `bash scripts/setup_navsim_env_vars.sh`。

---

## 6. Logs

| 目录 | 用途 |
|---|---|
| `logs/m1b2_d0/` | Stage 1 + Stage 2 (6-24)  |
| `logs/scan_navtrain_window_20260624_192206.log` | 6/24 19:22 上一个 AI 扫窗口的 log |
| `logs/m1b_*/`, `logs/m1a_*/`, ... | 各 phase 历史 log |

---

## 7. Docs 索引

| 文档 | 用途 |
|---|---|
| `RESUME_MONDAY.md`（项目根） | 长期周末 / 周一接手指南 |
| `docs/PATHS.md`（**本文件**） | 项目级路径地图 |
| `docs/results/key_results.md` | 论文表 / 实验主结果汇总 |
| `docs/_internal/handoff_*.md` | 跨 session AI handoff（按日期） |
| `docs/_internal/plan_*.md` | 执行计划 |
| `docs/_internal/incident_*.md` | 事故 / 数据问题诊断 |
| `docs/_internal/m1b_kickoff_plan.md` | M1.b D0/E2/F 阶段总规划 |
| `docs/_internal/risk_navtrain_data_missing.md` | navtrain 数据缺失风险登记 |
| `docs/journal/` | 按日 journal |
| `docs/_internal/NEXT_AI_HANDOFF_2026-06-25.md` | **下一次 session 接手（6-25 H20 4 卡）** |

---

## 8. 已知"重复劳动"陷阱（避免下一个 AI 再造轮子）

1. **窗口扫描脚本**有 3 版（**功能一致**）：
   - `scripts/scan_navtrain_missing_images.py`（上一个 AI 6/24 19:19）
   - `scripts/scan_navtrain_window.py`（上一个 AI 6/24 19:21）
   - `tools/scan_navtrain_full_window.py`（我 6/24 21:18）

   **结果对得上**：3 个产出 19,225 token list 完全 diff=0。下次先看 `exp/m1a_navtrain_probeA_setup/navtrain_window_clean_tokens.txt`。

2. **probe100 cherry-pick**：`data/navtrain_nocot_probe100/` 是手工挑的完整 token（M1.a setup AI 挑的），代表性 ≠ navtrain 平均，仅用于 D0 smoke/验码。

3. **scene_filter yaml token 格式**：navtrain.yaml 用 `  - 'xxxxx'`（**带 quote**），不是 `- xxxxx`。正则匹配要写 `^\s*-\s*'([a-f0-9]+)'`。

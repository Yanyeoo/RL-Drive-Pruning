# 2026-06-15 — MA2.2 metric_cache 全量 / MA2.3 GRPO smoke 调试

> 19:00 – 21:50 一段。MA2.2 metric_cache 全量跑通，MA2.3 GRPO smoke 进入 training_step 后在 `generate_sample` 内的 cuBLAS GEMM 上挂掉，今天主要在收敛根因。

---

## 1. MA2.2 — navtest metric_cache 全量

**入口**: `scripts/run_ma2_2_navtest_metric_cache.sh`（dry-run 5 scene + full 11596 一键切换）

**结果（11596 / 11596 全部成功）**:
- 耗时：约 1.5 h（基于 `logs/ma2_2_full.log` 时间戳 19:14 启动 → 20:37 完成）
- 落盘：`data/navtest_metric_cache/`（路径写进 `qwen2.5-vl-3B-navtest.yaml` 的 `metric_cache_path`）
- 元数据：`data/navtest_metric_cache/metadata/navtest_metric_cache_metadata_node_0.csv`
- 覆盖率：preflight 验证 navtest_nocot 全部 11596 token 都能命中 metric_cache（key 集合相等）

至此 MA2.1 (navtest_nocot 11596 json) + MA2.2 (metric_cache 11596 lzma) 全 GREEN，MA2.3 进入 smoke。

---

## 2. MA2.3 GRPO smoke — bug story

### 2.1 启动配置

- 入口：`tools/run_rft.py --config training/qwen2.5-vl-3B-navtest-grpo-nocot`
- yaml：`config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml`
- 单卡（`devices=1`），smoke 设 `max_steps=2`，`per_device_batch_size=1`，`num_generations=2`，`max_length=2048`，LoRA r=8 / alpha=16

### 2.2 症状

进入 `training_step` 后，`generate_sample` 调 `model.vlm.generate(...)`，进程被 **SIGFPE（floating-point exception）** 杀掉。Faulthandler 栈定位到：

```
peft/tuners/lora/layer.py:757  -- LoRA Linear forward
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:956   k_proj
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1208  attention block
```

cuBLAS 在 fp32 GEMM 时崩，不抛 Python 异常，而是 signal kill。

### 2.3 排查路径（按时间顺序）

| 步骤 | 假设 | 验证手段 | 结论 |
|---|---|---|---|
| H1 | bf16/fp16 在 Hopper sm_90 上 cuBLAS bug | 改 `AutoVLA.__init__` 内 `torch_dtype=bfloat16` → 可配置 `training.dtype`，默认 fp32 | bf16 / fp16 复现 SIGFPE；fp32 单独跑 generate 通（5–6 s / 1 video × 8 frames）|
| H2 | Lightning Trainer 偷偷开了 autocast 把 fp32 cast 回 bf16 | 在 generate_sample 加 `torch.is_autocast_enabled()` 打印；并用 `torch.amp.autocast(enabled=False)` 强制关 | autocast = False，且 `q_proj.base_layer.weight = fp32`；**还是 SIGFPE** |
| H3 | training mode 下 attention 走不同路径 | minimal repro 改 `.train()` + 真实视频 shape (4784, 1176) | minimal repro 仍 PASS |
| H4 | 真实 dataloader 的 batch shape 才会触发 cuBLAS bug | 在 Trainer.fit 路径里把 `model_inputs` pickle 到 `/tmp/autovla_fpe_repro.pkl`，离线脚本用同一 batch 复跑 | **离线脚本仍 SIGFPE** — 完全脱离 Trainer 也复现 |

**结论**：与 Lightning / DDP / autocast 完全无关。是 cuBLAS 在 H20 (sm_90, torch 2.4 + cu12.1) 对**该 batch 的某个 specific shape (input_ids=(1,941), pixel_values_videos=(2880,1176), video_grid_thw=(3,3))** 的 fp32 GEMM 触发的底层 bug。

minimal repro 没复现，是因为我用 1 个视频 8 帧 → patch shape (4784, 1176)；真实 batch 是 **3 个视频**（front_left / front / front_right 三摄）→ patch shape (2880, 1176)，**video_grid_thw=(3,3)** 才是触发条件。

### 2.4 当前状态（21:50）

- 已落定的 minimal repro：`/tmp/autovla_fpe_repro.pkl`（gitignored, 临时；smoke 重新跑一次会再生）
- 已落定的代码改动（保留在 `models/autovla.py`）：
  - `training.dtype` 可配置（fp32 / bf16 / fp16），默认 fp32
  - `generate_sample` 内显式 `torch.amp.autocast(device_type='cuda', enabled=False)` 兜底
- **未解**：cuBLAS sm_90 fp32 bug 的最终绕过方案。下一个 AI 接手时需要继续从以下任一方向打：
  - **(a) attn_implementation 切 eager**（最后一个未跑完的实验；理论上绕开 cuBLAS GEMM 走 ref kernel）
  - **(b) 升级 torch 2.5+ / cu12.4+**（可能在新版 cuBLAS 已修复）
  - **(c) 临时绕过**：禁用 TF32 (`torch.backends.cuda.matmul.allow_tf32 = False`) 或换 H800/A100 卡
  - **(d) 直接把 `generate` 改为 fp32 + 小 chunk 分段调用**

实测 (a) / (b) / (c) 之前不要再做 minimal repro 的 ablation，**直接对真实 batch 重放**才有意义（用 `/tmp/autovla_fpe_repro.pkl`，脚本在 `docs/_internal/ma2_3_smoke_runbook.md` 旁边可以复刻）。

---

## 3. 工程副产物

| 路径 | 内容 |
|---|---|
| `code/third_party/AutoVLA/models/autovla.py` | dtype 可配置 + 强制关 autocast（保留）|
| `code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml` | smoke 训练配置（保留）|
| `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtest.yaml` | navtest dataset（保留）|
| `code/third_party/AutoVLA/dataset_utils/preprocessing/nuplan_dataset.py` | 支持显式 `navsim_log_path` / `sensor_blobs_path`（保留）|
| `code/third_party/AutoVLA/tools/run_rft.py` | 改为接受我们 yaml 的相对路径 / PYTHONPATH 兼容（保留）|
| `code/third_party/AutoVLA/tools/preprocessing/nocot_sample_generation.py` | MA2.1 dry-run + 全量 兼容修复（保留）|
| `code/third_party/AutoVLA/runs/lightning_logs/` | smoke 历史 tfevents（gitignored，可清）|

**关键事实**：AutoVLA 仓库的 `origin` 是 `ucla-mobility/AutoVLA`（上游，无写权限），所以这些改动**不能 push 进 AutoVLA**。落地方式：

- patch 形式归档到父仓库 `docs/_internal/patches/`（gitignored，仅本地）：
  - `autovla_repo_tracked.patch` — 4 个 tracked 文件的 diff
  - `autovla_repo_untracked.tar.gz` — 4 个新增 yaml 的 tarball
- 父仓库 (`Yanyeoo/RL-Drive-Pruning`) 只 push docs / 顶层 README / 启动脚本，**不 push AutoVLA 子目录**

---

## 4. 现状盘点（21:50）

| 子里程碑 | 状态 |
|---|---|
| MA2.1 navtest_nocot JSON ×11596 | ✅ GREEN |
| MA2.2 metric_cache ×11596 | ✅ GREEN |
| MA2.3 GRPO smoke `training_step` 入口 | ❌ SIGFPE @ cuBLAS GEMM（根因已锁定，待绕过）|
| MA2.4 双卡 dispatcher | ⏸ 阻塞于 MA2.3 |
| MA2.5 navtest 全量 baseline EPDMS | ⏸ 阻塞于 MA2.3 |

---

## 5. 下一步（无人值守 / 给下一个 AI）

按优先级：

1. **跑 attn_implementation='eager' 实验**（已设计好，~10 分钟），写进 journal 续段
2. 如 eager 通 → smoke 切 eager + 重启 → 看 1 个 training_step 是否完整跑出 loss
3. 如 eager 也挂 → 装 torch 2.5 + cu12.4（环境名 `autovla_t25`），重跑
4. smoke 通了之后再开 MA2.4 dispatcher

*written 2026-06-15 21:55*

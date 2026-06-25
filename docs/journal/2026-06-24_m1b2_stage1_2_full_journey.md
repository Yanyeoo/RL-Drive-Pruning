# 2026-06-24 (周三) M1.b₂ Stage 1 + Stage 2 完整流水账

> 时间窗：21:00 启 Stage 1 → 21:41 Stage 2 跑完 → 23:00 GPU 回收
> 上一手：今晚 20:30–20:43 我已改完 multilayer attention 代码（未验码），handoff 写在
> `docs/_internal/handoff_2026-06-24_2043_multilayer_attention_done_navtrain_json_blocked.md`
> 用户授权后启 `plan_2026-06-24_2055_path3_execution.md` 的 Path 3。

---

## 0. TL;DR

- **Stage 1（multilayer attention D0 验码 + smoke）**：✅ 100/100 OK，2.55 s/scene，全 28 层 × 16 head × 720 vision tokens，shape `(28, 16, 720)`。
- **Stage 2（navtrain tokenize）**：第一次起 21:11 fail（FileNotFoundError），第二次起 21:23 ✅ **19,225/19,225 完成（19 min）**。
- 关键修正：Qwen2.5-VL-3B `num_attention_heads = 16`（不是 24）。
- 关键发现：sensor_blobs/trainval **只有 ~18.6% trigger token 满足 15-frame 完整 window**。

---

## 1. Stage 1：multilayer attention D0 验码

### 1.1 启动（21:00）
- Dry check 全过：wrapper 透传 OK，`--all-layers` flag 存在，probe100 json=100，token_list=100，save_dir 空，GPU 0 free 97 GB。
- 命令（PID=38782）：
  ```
  nohup bash scripts/run_m1a_attention_probe.sh \
    --scene-filter navtrain_probeA \
    --json-dir /apdcephfs/.../data/navtrain_nocot_probe100 \
    --token-list /apdcephfs/.../exp/m1a_navtrain_probeA_setup/tokens_100.txt \
    --save-dir /apdcephfs/.../exp/m1b2_d0_smoke_probe100_alllayers \
    --all-layers --num-layers 28 --gpu 0 --max-scenes 100 \
    > logs/m1b2_d0/stage1.log 2>&1 &
  ```

### 1.2 第一次踩坑：log 30 s 0 字节（误判）
- 30 s 检查：log 0 字节、GPU 0 MiB、RSS 2 MB → 怀疑命令没接到 stdin。
- 实际：Python 进程正常起来了（PID 38782，200+ thread），`/proc/PID/stack` 显示在 fuse `vfs_read` sleep ——**model checkpoint 冷启动从 cephfs 读 ~6 GB safetensors**，正常需 2–4 min。
- 教训：M1.a 时也是这种 90 s 后才出第一行 log；不要 sleep 30 s 就慌。

### 1.3 收敛（21:09，4.2 min 跑完）
| 指标 | 实测 | 状态 |
|---|---|---|
| 文件数 | 100/100 | ✅ |
| s/scene | 2.55（M1.a 单层 2.5 s，multilayer overhead ≈ 0%）| ✅ |
| MISSING / err | 0 / 0 | ✅ |
| 显存峰值 | 30.9 GB | ✅ |
| `per_layer_vision_attn.shape` | **(28, 16, 720)** | ✅ |
| `layer_idxs` | `[0..27]` 全 28 层 | ✅ |
| `multi_layer` flag | True | ✅ |
| `average_heads` | False | ✅ |
| `vision_blocks` | `[(108,349),(372,613),(636,877)]` 3 个摄像头 block | ✅ |
| .pt 体积 | 1.30 MB（28 × 16 × 720 × 4 bytes ≈ 1.29 MB，对得上）| ✅ |

### 1.4 ⚠️ 一个修正
- 我（和 m1b_kickoff_plan）原以为 Qwen2.5-VL-3B `num_attention_heads = 24`。**实际是 16**。
- 文档（key_results / future plans）里写 head 数时**用 16**。
- 不影响代码（capture 时 `tensor.shape` 直接拿）。

---

## 2. Stage 2：navtrain tokenize — 三层踩坑日记

### 2.1 准备（21:09–21:11）
- 写 `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtrain_full.yaml`（拷自 probe100 yaml，scene_filter 改成 navtrain.yaml）。
- 起 100 个 symlink：`data/navtrain_nocot_probe100/*.json → data/navtrain_nocot/*.json`，让 `--pre_generated_dir` 跳过这 100 个（避免重抽）。
- 验 navtrain.yaml token list = **103,288**（论文规模对上）。

### 2.2 第一次起 stage 2（21:11，PID=42726）
- 命令：`tools/preprocessing/nocot_sample_generation.py --config dataset/qwen2.5-vl-3B-navtrain_full --output_dir .../navtrain_nocot --num_workers 64 --pre_generated_dir .../navtrain_nocot`
- 5 min 后 ❌ 失败：
  ```
  FileNotFoundError: /apdcephfs/.../sensor_blobs/trainval/
    2021.06.14.13.27.42_veh-35_03142_03404/CAM_F0/b1ffaa96ce4f53d2.jpg
  ```

### 2.3 三层诊断（21:14–21:21）

#### 第一层假设：log dir 整个没下
查：sensor_blobs/trainval 有 **1192 个 log 目录** ✅，目标 log 也在；目标 log 下 CAM_F0/CAM_B0... 8 个 cam 子目录都在。**否决**。

#### 第二层假设：cam jpg 全没下
查：CAM_F0 有 91 个 jpg，但**没有** `b1ffaa96ce4f53d2.jpg`。`grep -c b1ffaa` 输出 0。**部分下载**。

#### 第三层假设：partial download is everywhere
5 个采样 log 全部 50/50 missing → 看似很严重。但**进一步验证发现 jpg basename ≠ scene token**：
- 写 Python 检查 scene `cams[CAM_F0][data_path]` 的 basename；
- 抽样 log `2021.06.14.14.25.15_veh-26_03964_04278`：625 scene 在 pickle，CAM_F0 只 213 jpg，**ratio 34%**。
- 8 个 cam 子目录 jpg 数完全同步（要么全下要么全没下）。

#### 关键 insight：M1.a probe100 怎么过的？
查 navtest 状况：navtest **100% 完整**（scene count = jpg count），所以 M1.a baseline 89.83 跑通。
查 probe100 token：5/5 sample 的 ±4 history + +10 future + 当前帧 **15 frame 全部在磁盘上** → probe100 setup 时（更早的 AI）**手工 cherry-pick 了完整 token**。

#### 真相
navtrain trigger frame 的 jpg 都在（103,288 全在），**但 ±history/future 邻居 jpg 部分缺失**。Scene loader 加载时要读 4+10+1=15 frame × 8 cam = 120 jpg/scene，**一个 cam 一个 frame 缺就 crash**。

### 2.4 解决方案（21:21–21:23）
**思路**：预扫 103,288 trigger token，只保留 ±4 历史 + +10 未来 + 当前帧的 CAM_F0 jpg 全在磁盘的（8 个 cam 同步下，扫 1 个就够）。

工具：`tools/scan_navtrain_full_window.py`（新建，32 worker 并行扫 1192 log pickle）。

结果：
```
navtrain.yaml triggers:        103,288
satisfy 15-frame full window:    19,225 (18.6%)
```

造 scene_filter `navtrain_avail19k.yaml`：拷 navtrain.yaml schema（has_route, num_history_frames=4...），`tokens:` block 换成 19,225。

dataset yaml 切换：`qwen2.5-vl-3B-navtrain_full.yaml` 指向 `navtrain_avail19k.yaml`。

### 2.5 第二次起 stage 2（21:23，PID=64808）
- 命令同 §2.2（唯一变化：scene_filter）。
- 21:24 看到 `Extracted 19225 scenarios` ✅ + 64 worker fork。
- 21:33（10 min 内）：9,287 real json，**29 token/s**，预测 ETA 5.7 min。
- 21:39（16 min）：17,798/19,225 (92%)，速率 19–46 it/s（partition cache 命中后飙到 40+）。
- 21:41 ✅ **19,225/19,225 完成**，进程正常退出，输出 `"All preprocessing data without CoT results have been saved"`。

### 2.6 Acceptance 结果（21:43）
| 指标 | 实测 | 预期 | 状态 |
|---|---|---|---|
| total json | 19,225 | 19,225 | ✅ |
| real new files | 19,125 | 19,225 - 100 skip | ✅ |
| symlinks 留存 | 100 | 100（probe100 alias） | ✅ |
| scene_filter token | 19,225 | 19,225 | ✅ |
| schema 字段 | token, dataset_name, cot_output, velocity, acceleration, instruction, gt_trajectory, his_trajectory, 8 cam_paths | 全 | ✅ |
| disk usage | 122 MB | — | ✅ |
| 错误 / MISSING | 0 / 0 | 0 | ✅ |

---

## 3. 产出物清单

### 3.1 数据
- `data/navtrain_nocot/` — 19,225 个 pretokenized json（19,125 real + 100 symlink → probe100），122 MB。
- `exp/m1b2_d0_smoke_probe100_alllayers/` — 100 个 attention .pt，shape (28,16,720)，total ~130 MB。

### 3.2 索引文件
- `exp/m1b2_navtrain_available_tokens.txt` — 152,495 个 token（sensor_blobs/trainval 实际有 CAM_F0 jpg 的 trigger 候选，含 navval 部分）。
- `exp/m1b2_navtrain_available_intersect.txt` — 103,288（与 navtrain.yaml 交集；trigger frame 都在但 history/future 未必）。
- `exp/m1b2_navtrain_full_window_tokens.txt` — **19,225（真正可用，15-frame 完整）**。

### 3.3 新建 / 修改文件
| 路径 | 类型 | 说明 |
|---|---|---|
| `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtrain_full.yaml` | new | Stage 2 dataset config，指向 avail19k scene_filter |
| `code/third_party/AutoVLA/navsim/.../scene_filter/navtrain_avail19k.yaml` | new | 19,225 token 白名单，schema 同 navtrain.yaml |
| `tools/scan_available_tokens.py` | new | 第一遍扫（看 CAM_F0 jpg 总集） |
| `tools/scan_navtrain_full_window.py` | new | 第二遍扫（验 15-frame 完整 window） |

### 3.4 Log
- `logs/m1b2_d0/stage1.log` — Stage 1 (4.2 min)。
- `logs/m1b2_d0/stage2_attempt1_FAILED.log` — Stage 2 第一次失败。
- `logs/m1b2_d0/stage2.log` — Stage 2 第二次成功（19 min）。

---

## 4. 仍未做的事 / 给明天的话

详见 `docs/_internal/NEXT_AI_HANDOFF_2026-06-25.md`（同时间提交）。

简要：
- ✅ Stage 1 (multilayer attention 代码验过 + D0 PASS) 不需要重做。
- ✅ Stage 2 (19,225 navtrain pretokenized json) 不需要重做，直接用 `data/navtrain_nocot/`。
- ⏳ Stage 3：**用 19,225 token 跑 4 GPU multilayer attention 抽取**（论文级 statistic 用），ETA 19225 × 2.55s / (4 GPU × 0.85 scale) ≈ **2.4 h** wall。明天 H20 4 卡 10–24h 窗口绰绰有余。
- ⏳ 论文 Section 4 head selection / layer selection statistic — 待 Stage 3 完成后。

---

## 5. 我学到的 / 留给未来 AI 的注意点

1. **不要在 fuse fs sleep 30 s 就慌**。Qwen2.5-VL-3B 冷启动 model load 在 cephfs/fuse 上 90–120 s 是正常的。
2. **navtest 全完整，navtrain 部分下载**。后者只 18.6% trigger 满足 15-frame window。预扫脚本在 `tools/scan_navtrain_full_window.py`，可复用。
3. **Qwen2.5-VL-3B heads = 16，不是 24**。多处文档要订正。
4. **probe100 是手工 cherry-picked**（M1.a setup 时上一个 AI 挑的完整 token），不能代表 navtrain 平均难度。但 stage 1 multilayer 验码用它没问题，因为只验 shape/数值/速率。
5. **`--pre_generated_dir` 是从 output_dir 自己读 token 名做 skip set 的**（`nocot_sample_generation.py` line 91–99）。symlink 也会被识别为已存在 → 完美避免重抽 100 个 probe token。

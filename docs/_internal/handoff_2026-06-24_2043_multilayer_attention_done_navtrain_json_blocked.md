# Handoff 2026-06-24 20:43 — M1.b₂ multi-layer attention 代码已 ready，但 navtrain pretokenized json **不存在**，阻塞全量抽取

> 这是 2026-06-24 周三晚上的 session handoff。今晚 GPU 窗口 ~20:30–22:50（~2h），代码改完了，但发现今晚要跑的 D0 stress test 数据前提**不成立**，跟用户报告后用户让我备份并写 handoff。下一个 AI 接手时请先读本文件。

---

## 1. TL;DR （30 秒读完）

- **代码已改完且零 lint**：multi-layer (28-layer) per-head attention capture pipeline 全套就位
  - `attention_capture.py`：新增 `patch_attention_capture_multilayer()`，保留老的 `patch_attention_capture()` 不动
  - `autovla_with_attention.py`：新增 ctor 参数 `attention_layer_idxs: Optional[List[int]] = None`，`_save_attention` 走多层分支存 stacked tensor `(L=28, H=24, N_vision)`
  - `run_attention_probe.py`：新增 `--all-layers` flag（隐含 per-head + multilayer），自动设 `layer_idxs=list(range(num_layers))`
  - 3 个文件 backup 在 `docs/_internal/backups_20260624_2043/`
- **代码**还**没跑过任何 forward pass 验证**（没做 D0 smoke）。先做 smoke。
- **阻塞**：navtrain pretokenized json 全量**不存在**。当前 `data/` 只有 `navtrain_nocot_probe100/`（100 个 json，今天 M1.a probe A 用的）。
- M1.b₂ 设计的"navtrain 全量 attention 抽取"需要 103,288 个 token 的 json，但 tokenize pipeline 还没在 navtrain 全量上跑过。

---

## 2. 今晚发生了什么

### 用户给的任务（路径 C）
今晚 4 GPU 跑 M1.b₂ 全 28 层 per-head attention 抽取，目标：拿到一个完整的 (L, H, scene) 张量数据集，供 M1.c LambdaRank head-selection scorer 训练。

### 我做的事

1. **读 + 确认改动点**（3 个文件）：`attention_capture.py` / `autovla_with_attention.py` / `run_attention_probe.py`
2. **改代码**（最小侵入，保留 M1.a 单层老路径完全不动）：
   - `patch_attention_capture_multilayer(vlm, layer_idxs, prompt_index, bucket)`
     - 每层独立 one-shot flag，每层 pre-fill (q_len>1) 那一次抓 (num_heads, q_len, q_len) → 切 vis-row → 存 cpu float32
     - 返回前把 attn_weights 设 None（继续防止 HF generate 累积 7GB Path A 那个 bug）
     - finally{} 保证每层 forward 恢复
   - `AutoVLAWithAttentionAgent.__init__` 加 `attention_layer_idxs`，非空时 OVERRIDES 老的 `attention_layer_idx` + 强制 `attention_average_heads=False`
   - `_save_attention` 多层分支：`torch.stack([per_layer[L] for L in layer_idxs], dim=0)` → 存 dict 含 `per_layer_vision_attn (L,H,N_vision)` + `layer_idxs` + `multi_layer=True`
   - `run_attention_probe.py` 加 `--all-layers` + `--num-layers 28`
3. **lint 全过**（3 个文件零警告）
4. **准备 D0 时发现数据空洞**：`data/navtrain_nocot/` 不存在；只有 `navtrain_nocot_probe100/` 100 个 json
5. **停手报告用户**

### 用户回复
让我备份 + 写本文件提醒下一个 AI，**没有让我决定路 1/2/3**。所以现在状态是：代码 ready，**等用户下次决策才能继续**。

---

## 3. 关键文件 diff 摘要（够你看懂在哪改了）

### `code/rldrive/scoring/attention_capture.py`
末尾 `__all__` 之前插入 `@contextmanager patch_attention_capture_multilayer(...)`（~80 行），再把 `patch_attention_capture_multilayer` 加进 `__all__`。原 `patch_attention_capture` **一行没动**。

### `code/rldrive/agents/autovla_with_attention.py`
3 处改动：
- import 加 `patch_attention_capture_multilayer`
- `__init__` 加 `attention_layer_idxs: Optional[List[int]] = None` 参数 + `self._attn_layer_idxs` 状态
- `compute_trajectory` 里 patch 调用按 `self._attn_layer_idxs is not None` 分支
- `_save_attention` 加 multilayer 分支（在原 single-layer 分支之前 early-return）

### `code/rldrive/scoring/run_attention_probe.py`
- 加 `--all-layers` / `--num-layers` argparse
- agent 实例化改成 `agent_kwargs = dict(...)` + 条件注入 `attention_layer_idxs`

详细 unified diff 可用：
```bash
diff -u docs/_internal/backups_20260618_xxxx/attention_capture.py code/rldrive/scoring/attention_capture.py
```
（M1.a 时代 backup 路径见 `docs/_internal/m1a_*.md` 索引；今晚的 backup 在 `docs/_internal/backups_20260624_2043/` —— 这是"改完后的最新副本"，要拿"改之前"的需要从 git 取）

---

## 4. ⚠️ 阻塞详情：navtrain pretokenized json 不存在

### 当前 `data/` 实际内容（2026-06-24 20:30 实测）
```
data/
  navtrain_nocot_probe100/   ← 100 个 json （M1.a navtrain probe A 用的）
  navtest_nocot/             ← navtest 全量已有 11,576 scene
  navsim_v2_trainval/        ← 原始 navsim pickle（chain pipeline 已完）
```

### 为什么不存在
`.chain_complete` 标记的是 **SceneLoader 能 build 出 103,288 个 navtrain scene 对象**（原始 pickle 解析完）这件事，**不是** "103,288 个 pretokenized json 已落盘"。两件事之间差一个 **batch tokenization pass**：每个 token 需要 build prompt + load 三摄像头 video → encode → save json，~2–4 s/token CPU bound。

probe100 是 M1.a 时手动小规模跑的，全量 navtrain tokenize pipeline **没启过**。

### 估算
- 单进程：103k token × 2.5s ≈ **71h**
- 4 进程并行：~18h
- 8 进程并行：~9h

---

## 5. 三条候选路径（用户还没拍板）

下面这 3 条是 20:35 我给用户的建议，**用户没回答选哪条**。下一个 AI 接手时请向用户确认。

### 路径 1：今晚直接跑 navtrain tokenize（CPU bound）
- 4 GPU 闲置（除非 video encoder 在 GPU 上跑——要查代码）
- 110 min × 4 process ≈ 440 process-min ≈ **预期产出 ~13k 新 json**
- 之后下一个 GPU 窗口再跑 M1.b₂ multilayer attention 抽取（~10h on 4 GPU 抽 13k）

### 路径 2（用户原意 C 的最小可行变种）：probe100 那 100 个 token 用 multi-layer 重抽
- 已有 json，立刻能跑，~4 min on 1 GPU
- 验证 multi-layer hook 代码正确性（D0 真正目的）
- 拿到 100 token × 28 层 × 24 head 的样本 → 可以做 M1.c v0 pipeline 验通
- 但 100 个 sample 训不出 LambdaRank scorer（太少）
- **优点**：本晚最值得做的事，至少把今晚改的代码验通

### 路径 3：路 2 + 路 1 混合（我推荐）
- Stage 1 (20:50–21:00, 10 min)：probe100 100 个 multi-layer 重抽 → D0 验码 + 拿样本 .pt + 验显存
- Stage 2 (21:00–22:50, 110 min)：起 4 GPU/CPU 跑 navtrain tokenize，预期产出 ~13k 新 json
- Stage 3（下次 GPU 窗口）：multilayer attention 抽取这 13k

---

## 6. 给下一个 AI 的硬规则

1. **不要随便启动 4 GPU 跑 M1.b₂ 全量抽取**——前提（navtrain json）不在。
2. **不要假设 D0 已经跑过**——今晚 D0 还**没执行**，改完代码就因为 json 缺失停了。
3. **不要回滚今晚的代码改动**——3 个文件改动都正确、零 lint，只是没验码。
4. **优先做的事**：跟用户确认走路径 1/2/3 中哪条；如果用户授权，先做路 2 的 D0 smoke（probe100 100 token multi-layer 重抽 4 min）来验码。
5. **D0 acceptance**（路 2 跑完后检查）：
   - .pt 体积：每个 ~1.9 MB（28×24×720×4B）
   - 显存峰值：< 12 GB（单 GPU prefill 时多一层 attn_weights tensor ~85 MB，但 patched layer **串行**执行不并发）
   - s/token：跟 M1.a probe A 单层 ~2.5 s/token 相比，预期 28 层 ~3.0–3.5 s/token（hook 本身不重算，HF 重算 attn_weights 才重）
   - 错误率：0/100 MISSING.json
6. **代码 backup 路径**：`docs/_internal/backups_20260624_2043/`（这是改完后的最新副本）。
7. **预期 .pt 字段**（multi-layer 模式下）：
   ```python
   {
     "per_layer_vision_attn": Tensor(L=28, H=24, N_vision≈720),  # cpu float32
     "layer_idxs": [0,1,...,27],
     "vision_token_positions": ...,
     "last_instr_idx": int,
     "vision_blocks": ...,
     "captured_q_len": int,
     "prompt_len": int,
     "average_heads": False,
     "multi_layer": True,
   }
   ```

---

## 7. 跑路 2 D0 smoke 的具体命令（拿走即用）

需要先看 `run_m1a_attention_probe.sh` 怎么 wrap `run_attention_probe.py`，加 `--all-layers` 透传。或者直接：

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
PYTHONPATH=code CUDA_VISIBLE_DEVICES=0 \
  /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
  -m rldrive.scoring.run_attention_probe \
  --scene-filter navtrain_probeA \
  --token-list <100 个 token 文件，跟 M1.a probe A 一样> \
  --save-dir exp/m1b2_d0_smoke_probe100_alllayers \
  --all-layers --num-layers 28 \
  --gpu 0 --max-scenes 100 \
  --checkpoint <跟 M1.a 一致> \
  --sensor-data <跟 M1.a 一致> \
  --codebook <跟 M1.a 一致> \
  --config <跟 M1.a 一致>
```

⚠️ checkpoint/sensor-data/codebook/config 4 个参数请直接从 `scripts/run_m1a_attention_probe.sh` 复用（我没列具体值因为脚本是 source of truth）。

跑完检查：
```bash
ls exp/m1b2_d0_smoke_probe100_alllayers/*.pt | wc -l       # 期望 100
ls exp/m1b2_d0_smoke_probe100_alllayers/*.MISSING.json     # 期望空
python -c "import torch; d=torch.load('exp/m1b2_d0_smoke_probe100_alllayers/$(ls exp/m1b2_d0_smoke_probe100_alllayers/*.pt|head -1)'); print(d['per_layer_vision_attn'].shape, d.keys())"
# 期望: torch.Size([28, 24, ~720]) dict_keys([... 'multi_layer': True ...])
du -sh exp/m1b2_d0_smoke_probe100_alllayers/                # 期望 ~190 MB (100 * 1.9 MB)
```

---

## 8. 当前 GPU/进程状态（20:43 freeze）

- 没有任何后台 job 在跑（GPU 全 idle）
- 没有 `.in_progress` / `.lock` 文件被我创建
- `data/` 没动过任何文件
- 唯一 IO 改动：
  - 3 个 `.py` 文件修改（见 §3）
  - 1 个 backup 目录创建：`docs/_internal/backups_20260624_2043/`
  - 本 handoff 文件
  - `RESUME_MONDAY.md` 顶部追加 §0.5 区块（见下）

---

## 9. M1.b₂ 整体设计回顾（避免下个 AI 重新发明）

> 这部分是我对 M1.b₂ 工作流的理解，从今晚对话推断。如果跟用户已存在的 spec 不一致，**以用户 spec 为准**。

- **目标**：训练一个 per-scene head-selection scorer（M1.c），输入 = navtrain 全量 (scene, L, H) 的 attention pattern，输出 = 每个 scene 该激活哪些 head（mask）。LambdaRank loss。
- **M1.b₂ 是数据生成阶段**：navtrain 103k token × 28 层 × 24 head × ~720 vision token = 大概 40 GB pt 数据（per-token 1.9 MB × 103k ≈ 196 GB——这数字偏大，可能要切 split）。**注意磁盘**。
- **M1.c 是训练阶段**：拿这些 .pt 作 input → 输出 per-scene head mask。
- M1.b₁（Level-0）已经做过：固定 mask L24 dead head，跑 4×4 sweep （V0/V1/V2/V3）navtest n=11576，结论 V1 dominant（PDMS 89.62, vs V0 89.83；详见 RESUME_MONDAY.md §M1.b 区段——但那是 6/18 写的，准确性待核）。

---

## 10. 联系上下文（给下一个 AI 的 onboarding）

- 用户正在做的项目：**RL-Drive-Pruning**（autoVLA 推理时 attention head pruning，保 PDMS 不掉）
- 当前最大 milestone：M1.b₂ 全 navtrain attention 抽取 → M1.c per-scene head policy 训练
- 你接手时**首先读**：
  1. 本文件（你正在读）
  2. `RESUME_MONDAY.md`（特别是新加的 §0.5 区块和原 §M1.a/§M1.b 区块）
  3. `docs/results/key_results.md` §3 (B0=89.83) + §4 (M1.a L*=12 locked)
  4. `docs/_internal/m1a_layer_selection_2026-06-18.md`
- **不要做**：
  - 不要重启 navtrain 数据下载（chain pipeline 已完）
  - 不要 push GitHub（没授权）
  - 不要假设 D0 跑过
  - 不要在没跟用户确认路径 1/2/3 之前直接启 4 GPU job
- **可以做**：
  - 跑 §7 的 D0 smoke 命令（前提：用户授权或路径 2/3 被选）
  - 检查 `scripts/run_m1a_attention_probe.sh` 怎么 wrap 来确认 §7 命令的 4 个 path 参数
  - 查 video encoder 跑 CPU 还是 GPU（决定路径 1 要不要占 GPU）

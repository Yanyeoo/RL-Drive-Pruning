# 🚨 RESUME — 2026-06-18 17:00 之后

> 写于 2026-06-18 16:58。前任 AI 17:00 被回收，2 卡 H20。
> **强制：先把这个文件读完再做任何事**。

---

## 🎉 前任在死前做完了

1. ✅ M1.a smoke 5/5 OK (16:42)
2. ✅ M1.a 4-GPU layer sweep 800/800 OK (16:44–16:56, ~12min)
3. ✅ 数字写进 `docs/results/key_results.md` §M1.a
4. ✅ 详细 journal: `docs/journal/m1a_layer_sweep_navtest_2026-06-18.md`

## 🎯 关键发现

**双峰 attention 模式**：

```
layer:  0    4    8    12   16   20   24   27
frac:  5.4  3.0  10.3 17.9 2.6  9.5  4.6  18.0
                       ▲                    ▲
                    peak 1               peak 2
```

L\* = 27 (vision_frac=0.1804) 但 L12 (0.1789) 几乎并列。**还不能锁 L\***，需要 fine sweep。

---

## 你的下一步（按优先级）

### Step 1: 读 journal（5min）

`docs/journal/m1a_layer_sweep_navtest_2026-06-18.md` — 完整结果 + 解读 + 推荐方案。

重点理解：**双峰是真的，不是噪声**（per-layer SE ≈ 0.06%，gap 远大于 SE）。

### Step 2: Fine sweep（~20min on 2-GPU）

zoom in 两个 peak：

```bash
# 编辑 scripts/run_m1a_layer_sweep_4gpu.sh
# 把 GPU_LAYERS 从 4-GPU 改为 2-GPU 配置：
#   GPU_LAYERS[0]="10 11 12 13 14"   # zoom on peak 1
#   GPU_LAYERS[1]="25 26 27"         # zoom on peak 2 (L27 已有，重跑做 sanity)
# 删掉 GPU_LAYERS[2]/[3] 的 loop（或加 if [ -n "${GPU_LAYERS[$GPU]:-}" ]; then ... fi）
# 然后跑：
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
bash scripts/run_m1a_layer_sweep_4gpu.sh \
    exp/m1a_layer_sweep_20260618_1644/tokens_100.txt
```

或者更简单：直接用 single-layer 模式串行跑 7 个新 layer：

```bash
SWEEP=/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644
TOKENS=${SWEEP}/tokens_100.txt
for L in 10 11 13 14 25 26; do
    LP=$(printf "%02d" $L)
    [ -d ${SWEEP}/L${LP} ] && continue   # skip if exists
    bash scripts/run_m1a_attention_probe.sh \
        --scene-filter navtest_100 \
        --save-dir ${SWEEP}/L${LP} \
        --layer-idx $L \
        --gpu $((L % 2)) \
        --token-list $TOKENS \
        --max-scenes 100 \
        2>&1 | tee logs/m1a_fine_L${LP}.log
done
```

### Step 3: Re-analyze（30s）

```bash
PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code \
    /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
    -m rldrive.scoring.analyze_layer_sweep \
    --sweep-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644 \
    --layers 0,4,8,10,11,12,13,14,16,20,24,25,26,27
```

### Step 4: Lock L\*

按 journal 末尾的"Recommended"建议：**如果 L12 ≈ L27 仍然成立，prefer L12**（pruning 在早层省更多 flop）。

写决策追加到 journal + 更新 `key_results.md` §M1.a（headline 数字）。

### Step 5（可选）: navtrain probe A 复核

只在 `.chain_complete` 出现后做（明早或更晚）。

---

## navtrain 后台状态（不阻塞，**别动**）

| PID | 干啥 |
|---|---|
| 3738 | `download_navtrain_robust.sh` 主脚本 history_1 rsync 中（13:17 起） |
| 3774 | `post_dl_chain.sh` chain watcher |
| 139364... | 用户授权的 history_2 takeover rsync（14:55 起）|

不要 kill 任何一个。详见 `docs/_internal/handoff_2026-06-18_session_death.md`。

---

## 完整文档导航

按这顺序读：

1. **本文件** ← 你在看
2. `docs/journal/m1a_layer_sweep_navtest_2026-06-18.md` — **M1.a 主结果**
3. `docs/_internal/handoff_2026-06-18_session_death.md` — 完整交接
4. `docs/_internal/incident_2026-06-18_false_stall_diagnosis.md` — 前任踩坑记
5. `docs/_internal/decision_proposal_2026-06-17_m1a_on_navtest.md` — navtest pivot 已确认
6. `docs/_internal/m1a_prereqs.md` — V2/V3/V4 sanity 详解

代码（已完整可用）：
- `code/rldrive/scoring/run_attention_probe.py` ✅ scene loop 已写完
- `code/rldrive/scoring/analyze_layer_sweep.py` ✅ analysis 工具已写好
- `code/rldrive/agents/autovla_with_attention.py` — wrapper agent
- `code/rldrive/scoring/attention_capture.py` — patch + save logic
- `scripts/run_m1a_attention_probe.sh` — bash 包装
- `scripts/run_m1a_layer_sweep_4gpu.sh` — 4 卡 sweep launcher（fine sweep 改 GPU_LAYERS 即可）

---

## 硬规则（不要破）

1. **不要重启 navtrain 下载脚本或后台 rsync**
2. **不要对 `_staging_navtrain/` 做 tar/mv/rsync**（前任踩过 2 次）
3. **不要 push GitHub**（没授权）
4. **关键数字必须当场写 `docs/results/key_results.md`**（SOP 强制）
5. **不确定就停下来问**

---

## 环境速查

| 项 | 路径 |
|---|---|
| project root | `/apdcephfs/private_shayladeng/tokenrl_autoVLA` |
| autovla conda | `/apdcephfs/private_shayladeng/miniconda3/envs/autovla` |
| ckpt | `/apdcephfs/private_shayladeng/ckpt/AutoVLA_PDMS_89.ckpt` |
| navtest json | `data/navtest_nocot/` |
| smoke seed | `data/navtest_nocot_smoke_seed/` |
| **当前 sweep** | `exp/m1a_layer_sweep_20260618_1644/` |
| OPENSCENE_DATA_ROOT | `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2` |

---

## Milestone 进度

- M0 ✅ B0 navtest PDMS=89.83 locked
- M0.2 🔄 navtrain rsync 后台跑
- **M1.a coarse sweep ✅ 完成 (n=100, 8 layers, 双峰发现)**
- **M1.a fine sweep 🎯 你的工作（zoom L10-14 + L25-27）**
- M1.a L\* lock ⏳
- M1.b ⏳ 等 M1.a + navtrain done

---

## 当前活动 GPU job（16:44 起，4 卡 detach）

```
SWEEP_DIR=/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644
TOKEN_LIST=${SWEEP_DIR}/tokens_100.txt   (前 100 个 navtest_nocot tokens, 字典序)
```

8 个 layer，每张 GPU 串行跑 2 个 layer，4 卡并行：

| GPU | layers | 落盘目录 |
|---|---|---|
| 0 | L00 → L16 | `${SWEEP_DIR}/L00/`, `L16/` |
| 1 | L04 → L20 | `${SWEEP_DIR}/L04/`, `L20/` |
| 2 | L08 → L24 | `${SWEEP_DIR}/L08/`, `L24/` |
| 3 | L12 → L27 | `${SWEEP_DIR}/L12/`, `L27/` |

时间预算：每 layer ~6min（model load 已 cached → ~20s + 100×3.5s = ~6min）。每 GPU 跑 2 layer ≈ 12-13min。**16:44 启动 → 估计 16:57 全完**。

### 怎么查进度（你接手第一件事）

```bash
SWEEP=/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644

# 1. workers 还在跑吗？
pgrep -af "run_attention_probe"

# 2. 每个 layer 落盘几个 tensor（应该最终都 = 100）
for L in 00 04 08 12 16 20 24 27; do
    n=$(ls $SWEEP/L${L}/*.pt 2>/dev/null | wc -l)
    echo "L${L}: ${n}/100 tensors"
done

# 3. 每个 GPU 的最新日志
tail -5 /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/m1a_sweep_20260618_1644_gpu*.log

# 4. status 文件（launcher 自动生成）
cat $SWEEP/SWEEP_STATUS.txt
```

### 期望最终状态

8 个目录每个有 100 个 `.pt` 文件 = 800 个 attention tensor。每个 tensor 存 `(N_vision,)` 头平均 attention 向量（~1KB）。

### 如果某个 layer 没跑完（有 .pt 数 < 100）

最可能原因：worker 被 OOM 或 driver 错误。查对应 gpu log。**不要重启**，直接报告 user 看缺哪些层、再决定补。

---

## 接下来你的工作

### Step A: verify sweep 全完（5min）

按上面"查进度"的步骤，确认 8×100=800 个 .pt 都在。如果差太多 → 报告 user。

### Step B: 写 L\* analysis（30-60min）

对每个 layer 算"vision-attention fraction" = `sum(attn[vision_tokens]) / sum(attn[all_kv])`（每 scene 算一个值，再 100-scene 平均）。

参考 `code/rldrive/scoring/attention_capture.py` 看 .pt 的字段（应该有 `attention`, `vision_token_positions`, `prompt_index` 等）。

写一个简单脚本：
```python
# code/rldrive/scoring/analyze_layer_sweep.py
import torch
from pathlib import Path

SWEEP = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644")
LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]
results = {}
for L in LAYERS:
    layer_dir = SWEEP / f"L{L:02d}"
    fracs = []
    for pt in layer_dir.glob("*.pt"):
        d = torch.load(pt, map_location="cpu")
        # adapt to actual keys; check one .pt first
        attn = d["attention"]                        # shape e.g. (N_vision,) head-mean
        # if attention is normalized over ALL kv keys, then vision_frac = attn.sum()
        # if it's already vision-only normalized, you need access to non-vision attn too
        # Inspect first .pt manually before deciding the formula
        fracs.append(float(attn.sum()))
    results[L] = sum(fracs) / len(fracs) if fracs else float("nan")
    print(f"L{L:02d}: vision_frac_mean = {results[L]:.4f} (n={len(fracs)})")
print("L* =", max(results, key=results.get))
```

### Step C: 写 decision doc

`docs/_internal/m1a_layer_selection_2026-06-18.md`：
- sweep 配置（8 layers × 100 scenes）
- 每 layer 平均 vision-attn fraction
- 选定的 L\*
- 简单 sanity（layer 27 应该接近 1，layer 0 应该接近 0/random，中间层应该有 peak）

按 SOP 把 L\* + score 写进 `docs/results/key_results.md` §M1.a。

### Step D（可选，低优先级）：navtrain probe A 复核

只在 `.chain_complete` 出现后做。100-scene 同样跑一遍，10min 出结果。L\* 漂移 ≤2 层算通过。

---

## navtrain 后台状态（不阻塞 M1.a，**别动**）

| PID | 干啥 |
|---|---|
| 3738 | `download_navtrain_robust.sh` 主脚本 history_1 rsync 中（13:17 起） |
| 3774 | `post_dl_chain.sh` chain watcher（poll `.download_complete`）|
| 139364... | 用户授权的 history_2 takeover rsync（14:55 起）|

不要 kill 任何一个。详见 `docs/_internal/handoff_2026-06-18_session_death.md`。

---

## 完整文档导航

按这顺序读：

1. **本文件** ← 你在看
2. `docs/_internal/handoff_2026-06-18_session_death.md` — 完整交接 + 三个会咬你的坑
3. `docs/_internal/incident_2026-06-18_false_stall_diagnosis.md` — 前任踩坑记
4. `docs/_internal/decision_proposal_2026-06-17_m1a_on_navtest.md` — 用户已确认 (a) navtest pivot
5. `docs/_internal/m1a_prereqs.md` — V2/V3/V4 sanity 详解 + cost 估算
6. `docs/_internal/m1_attention_hook_design.md` — attention hook 设计

代码：
- `code/rldrive/scoring/run_attention_probe.py` — 主 runner（**已经写完 scene loop**）
- `code/rldrive/agents/autovla_with_attention.py` — wrapper agent（V2/V3/V4 assert 内置）
- `code/rldrive/scoring/attention_capture.py` — patch + save logic
- `scripts/run_m1a_attention_probe.sh` — bash 包装
- `scripts/run_m1a_layer_sweep_4gpu.sh` — 4 卡 sweep launcher（**就是当前在跑的那个**）

---

## 硬规则（不要破）

1. **不要重启 navtrain 下载脚本或后台 rsync**（健康，自己会收敛）
2. **不要在 sweep 还在跑时 launch 新 GPU job**（会撞内存）
3. **不要对 `_staging_navtrain/` 做 tar/mv/rsync 操作**（前任踩过 2 次）
4. **不要 push GitHub**（没授权）
5. **关键数字（L\*/vision_frac）必须当场写 `docs/results/key_results.md`**（SOP 强制）
6. **不确定就停下来问**，特别是 sweep 出现异常时

---

## smoke 已通过的证据

```
/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_smoke_L14/
  00c489998dd4555c.pt  018690bcb255590d.pt  020ba7462c6f52b3.pt
  02d9591fc6de53c8.pt  02ecdf935f895a86.pt
```

5/5 OK, 0 err, avg 3.5s/scene, V2/V3/V4 q_len assert 全过。

---

## 环境速查

| 项 | 路径 |
|---|---|
| project root | `/apdcephfs/private_shayladeng/tokenrl_autoVLA` |
| autovla conda | `/apdcephfs/private_shayladeng/miniconda3/envs/autovla` |
| ckpt | `/apdcephfs/private_shayladeng/ckpt/AutoVLA_PDMS_89.ckpt` |
| navtest json | `/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtest_nocot/` |
| smoke seed | `/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtest_nocot_smoke_seed/` |
| 当前 sweep dir | `/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644` |
| OPENSCENE_DATA_ROOT | `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2` |

---

## Milestone 进度

- M0 (baseline) ✅ B0 navtest PDMS=89.83 locked
- M0.2 (navtrain ingest) 🔄 后台 rsync，**不阻塞**
- M1.a smoke ✅ 5/5 (16:42)
- **M1.a layer sweep 🔄 4 卡 detach，等结果（你的工作）**
- M1.a L\* selection ⏳ 你接手
- M1.b ⏳ 等 M1.a + navtrain done

---

下一个 AI: 16:55 左右开始按 Step A 查 sweep 进度。如果 17:00 GPU 缩到 2 卡时 sweep 没完，剩下的 layer 会因为 GPU 消失而失败 —— 那就只用已完成的 layers 选 L\*（应该够：8 layer 里至少 4 个该完成了，能看到 trend）。

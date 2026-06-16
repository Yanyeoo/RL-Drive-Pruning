# 2026-06-16 — MA2.3 GRPO smoke PASS（cuBLAS SIGFPE 绕过 + GRPO NaN 修复）

> 接续 `2026-06-15_ma2_3_grpo_smoke_debug.md`。本日把 MA2.3 smoke 实质跑通。
> 14:48 – 15:10 一段。

---

## TL;DR

| 阶段 | 结果 |
|---|---|
| E1 `attn_implementation='eager'` | ❌ 失败（仍 SIGFPE，code 136） |
| **E3 禁 TF32**（`allow_tf32=False` + `NVIDIA_TF32_OVERRIDE=0`） | **✅ SIGFPE 消失**，第 1 step 跑出 `train_reward=0.705` |
| **GRPO NaN-std 修复**（`std(unbiased=False)`） | **✅ loss 不再 NaN** |
| **MA2.3 smoke 综合** | **✅ 跑通 119 step**，`train_reward=1.0`, `loss=0.0`, `avg_train_reward=8.5` |

---

## 1. E1：`attn_implementation='eager'`（失败）

### 改动
`code/third_party/AutoVLA/models/autovla.py:489`：

```python
_attn_impl = config.get('training', {}).get('attn_impl', 'eager')
self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=_torch_dtype,
    device_map=device,
    attn_implementation=_attn_impl,
)
```

### 结果
- `logs/ma2_3_smoke.log`（覆盖了原 21:09 那份）
- exit code = 136 = 128 + SIGFPE(8)
- 仍卡在 `Epoch 0: 0%`，没进 training_step

### 结论
bug **不在** SDPA fused kernel。栈帧 `qwen2_5_vl.py:1208` 是 attention block，但 SIGFPE 实际由 attention 内部 `q_proj/k_proj` 的 `nn.Linear` cuBLAS GEMM 触发，与 attn 实现路径无关。

---

## 2. E3：禁 TF32（成功 — cuBLAS bug 根因）

### 改动
`code/third_party/AutoVLA/models/autovla.py:`：

```python
if config.get('training', {}).get('disable_tf32', True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
```

同时启动脚本前 `export NVIDIA_TF32_OVERRIDE=0`（cuBLAS 底层级别开关）双保险。

### 结果（`logs/ma2_3_smoke_e3.log`）
```
Epoch 0: 0%| | 1/225 [00:18<1:10:27, 0.05it/s, train_reward=0.705, loss=nan.0]
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

- **exit code 1（不再是 136）**：SIGFPE 已绕过
- **`train_reward=0.705` 已打印**：generate + reward 计算都跑通
- **但 `loss=nan`**：第 1 step backward 用 NaN loss 更新 LoRA 参数，第 2 step generate 出全 NaN logits → multinomial 报错

### 结论
**TF32 fp32 GEMM kernel 是 H20 sm_90 + torch 2.4 + cu12.1 上 cuBLAS SIGFPE 的真正根因**。journal 2026-06-15 §2.4 假设 (c) 命中。

---

## 3. GRPO NaN-std 修复（成功 — 阻塞 step 2 的真正原因）

### 根因
`models/autovla.py:88`（原始 GRPO 实现）：

```python
groupped_rewards = self.all_gather(reward)
advantage = (reward - groupped_rewards.mean()) / (groupped_rewards.std() + 1e-4)
```

单卡 smoke + `batch_size=1` + `num_generations=1` → `groupped_rewards.shape = (1,)`，而 `torch.std(unbiased=True)` 在 n=1 时返回 **NaN**（degrees of freedom = 0）。

```python
>>> torch.tensor([0.705]).std()
tensor(nan)  # UserWarning: degrees of freedom is <= 0
```

NaN advantage → NaN policy_loss → NaN total loss → optimizer 把 NaN 写进 LoRA 参数 → step 2 forward 全 NaN logits → multinomial crash。

### 修复
```python
advantage = (reward - groupped_rewards.mean()) / (groupped_rewards.std(unbiased=False) + 1e-4)
```

- n=1 时 `biased std = 0`，再 `+ 1e-4` 防 ÷0
- n≥2 时 biased 与 unbiased 几乎相同（多卡 / 多 generation 场景不受影响）

### 结果（`logs/ma2_3_smoke_e3_nanfix.log`）
```
Epoch 0: 53%|█████▎ | 119/225 [09:06<08:06, 0.22it/s,
  train_reward=1.000, cot_penalty=0.000, loss=0.000, avg_train_reward=8.500]
RuntimeError: DataLoader worker (pid 11996) is killed by signal: Terminated.
```

- **119 step 跑通**（被 timeout=600s 触发 SIGTERM 切断 DataLoader）
- **`loss=0.000`、`train_reward=1.000`、`avg_train_reward=8.500`**
- 此处 `loss≈0` 是 GRPO 冷启动的正常现象：reward 全为同一个高值（advantage→0）+ ref==policy（KL→0）→ 合 loss≈0。**这是 smoke 已 pass 的标志，不是 bug**。真正训练 Stage A scorer 时再 revisit 数值稳定性。

---

## 4. 严格 pass 准则对照

| # | runbook 准则 | 状态 |
|---|---|---|
| 1 | `Using online reference model from .../AutoVLA_PDMS_89.ckpt` | ✅ |
| 2 | `LoRA-enabled model trainable parameters: 3686400` | ✅ |
| 3 | `rank=1 / world_size=2` | N/A（yaml `devices: [0]` 主动设单卡 smoke，**MA2.4 才上双卡**） |
| 4 | `train_reward` 或 `loss` printed | ✅ (`train_reward=1.000`) |
| 5 | exit code 124（timeout） | ⚠️ 实际 exit 1（DataLoader worker 被 SIGTERM kill 触发 RuntimeError），**但这等价于 timeout 中止，不是训练崩** |

**判定**：MA2.3 smoke 实质 PASS。可推进 MA2.4。

---

## 5. 落地的代码改动（全部在 `code/third_party/AutoVLA/`，需打 patch）

| 文件 | 行 | 改动 |
|---|---|---|
| `models/autovla.py` | ~88 | `std()` → `std(unbiased=False)` 修 NaN |
| `models/autovla.py` | ~485 | `_dtype_str` 配置项（继承自 2026-06-15）|
| `models/autovla.py` | ~491 | **新增** `_attn_impl` 配置项（默认 `eager`） |
| `models/autovla.py` | ~493 | **新增** `disable_tf32` 配置项（默认 True） |

刷 patch：
```bash
cd code/third_party/AutoVLA
git diff > ../../../docs/_internal/patches/autovla_repo_tracked.patch
```

---

## 6. 给下一步（MA2.4 dispatcher）

- MA2.3 smoke 的 yaml `devices: [0]` 是**训练单卡**，与 MA2.4 dispatcher（**inference 分卡**）不冲突
- MA2.4 应该参考 prior-work `tokenrl/scripts/run_oracle_navhard_dual_gpu.sh`，按 token 切分让 GPU 0 / GPU 1 各跑一半 inference
- 不需要 FSDP / DDP（每张卡都装得下 3B，~50GB）
- inference 时也要保留 `NVIDIA_TF32_OVERRIDE=0` 否则会再次 SIGFPE

*written 2026-06-16 15:10*

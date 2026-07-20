# MA2.3 → MA2.4 接力 handoff（2026-06-15 → next-AI）

> 接续 `2026-06-15_ma2_3_grpo_smoke_debug.md`。如果你是接手的 AI / 协作者，请先读完那篇 journal 再回来读本文件。

---

## 立刻先做的 3 件事

```bash
# 1. 读对外文档（GitHub 上能看到的）
cat README.md
cat docs/plan/design_decisions.md
cat docs/plan/implementation_plan.md

# 2. 读本地 _internal（不在 GitHub）
cat docs/_internal/handoff.md
cat docs/_internal/ma2_3_smoke_runbook.md

# 3. 读最新 journal（关键调试上下文）
cat docs/journal/2026-06-15_ma2_3_grpo_smoke_debug.md
```

---

## 第一项任务：跑通 MA2.3 GRPO smoke

### 你已知的事实
- MA2.1（11596 navtest_nocot json）、MA2.2（11596 metric_cache）都已 GREEN。
- MA2.3 smoke 启动到 `training_step` 第一步，挂在 `model.vlm.generate(...)` 内 cuBLAS GEMM 上，SIGFPE。
- 根因**不是** dtype（已用 fp32 + 关 autocast 排除）。
- 根因**不是** Lightning（已用离线 pickle 重放排除）。
- 根因**是** cuBLAS 在 H20 sm_90 + torch 2.4 + cu12.1 上对某种**真实 3-video batch** 的 fp32 GEMM 触发底层 bug。

### 你要做的实验（按优先级）

| # | 实验 | 工作量 | 预期 |
|---|---|---|---|
| **E1** | 把 `attn_implementation` 改成 `'eager'`，复跑 smoke | ~10 min | 多半通过（绕开 cuBLAS GEMM 走 ref kernel）|
| E2 | 升级到 torch 2.5 + cu12.4 的独立 env (`autovla_t25`)，复跑 | ~30 min | 大概率通过（cuBLAS 已修）|
| E3 | 禁 TF32 (`torch.backends.cuda.matmul.allow_tf32 = False`) 复跑 | ~5 min | 可能通过 |
| E4 | 换 A100 / H800 卡跑（如果有） | 视调度 | 必通 |

### E1 入口（建议从这里开始）

修 `code/third_party/AutoVLA/models/autovla.py` 的 `AutoVLA.__init__`：

```python
self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=_torch_dtype,
    device_map=device,
    attn_implementation='eager',   # ← 新增这一行
)
```

然后跑：

```bash
bash docs/_internal/scripts_internal/run_ma2_3_navtest_grpo_smoke.sh
# (这个脚本在 _internal 里，因为含绝对路径)
```

观察 `logs/ma2_3_smoke.log` 里的：
- `Using online reference model from .../AutoVLA_PDMS_89.ckpt`
- `LoRA-enabled model trainable parameters: 3686400`
- 是否 SIGFPE / 是否打印 `train_reward` / `loss`

通过的 pass 准则在 `docs/_internal/ma2_3_smoke_runbook.md` 末尾。

---

## 第二项任务：smoke 通了之后

按 `docs/plan/implementation_plan.md` § M0.3 推进 MA2.4 / MA2.5：

- **MA2.4**：写双 H20 dispatcher (`scripts/run_autovla_navtest_dual_gpu.sh`)；参考 prior-work 内部仓库的 `run_oracle_navhard_dual_gpu.sh`
- **MA2.5**：navtest 全量 baseline（11596 token），出 B0 EPDMS

---

## 工作环境（绝对路径，**不要 commit**）

| 资产 | 位置 |
|---|---|
| 本仓库根 | `/apdcephfs/private_shayladeng/tokenrl_autoVLA/` |
| AutoVLA 上游 clone | `<本仓库根>/code/third_party/AutoVLA/`（**gitignored**）|
| conda env | `/apdcephfs/private_shayladeng/miniconda3/envs/autovla/` |
| Qwen2.5-VL-3B | `<本仓库根>/models/Qwen2.5-VL-3B-Instruct/` |
| AutoVLA SFT ckpt | `<本仓库根>/models/AutoVLA/AutoVLA_PDMS_89.ckpt` |
| navtest_nocot json | `<本仓库根>/data/navtest_nocot/` |
| metric_cache | `<本仓库根>/data/navtest_metric_cache/` |
| NAVSIM 数据根 (`OPENSCENE_DATA_ROOT`) | `/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/` |
| nuplan maps (`NUPLAN_MAPS_ROOT`) | `/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0/` |
| prior-work 老仓库（参考） | `/apdcephfs/private_shayladeng/tokenrl/` |

---

## 重要的"不要"

1. **不要** `git add code/third_party/`（AutoVLA 上游仓库，16GB）。已 gitignore 兜底。
2. **不要** `git add scripts/` 或 `docs/_internal/`。已 gitignore 兜底。
3. **不要**在 push 的文件（README / docs/plan / docs/journal）里写 `/apdcephfs/...` 绝对路径。脱敏规则见 `docs/_internal/handoff.md` §2。
4. **不要**对 `code/third_party/AutoVLA/` 做大改后忘了刷新 patch：
   ```bash
   cd code/third_party/AutoVLA
   git diff > ../../../docs/_internal/patches/autovla_repo_tracked.patch
   ```
5. **不要**重跑 minimal repro 来 ablation cuBLAS bug。**用真实 dump pickle 重放**才有意义（看 journal §2.3 的 H4 行）。

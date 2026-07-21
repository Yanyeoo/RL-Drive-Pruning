# 2026-07-21 — 本周期最终结果 + 下周期计划

**Window**: 2026-07-20 20:28 → 2026-07-21 15:00 (回收)

---

## 完成的实验结果

### 3B 主表 (full navtest, N≈11574, r=0.5)

| 方法 | 类型 | PDMS | Δ vs no-prune |
|---|---|---|---|
| No pruning (r=1.0) | — | 0.8988 | — |
| **DrivePrune (LambdaRank, ours)** | learned | **0.8920** | **−0.69pt** |
| SparseVLM (text-guided) | training-free | 0.8899 | −0.89pt |
| Attention L12 (teacher) | oracle | 0.8901 | −0.87pt |
| MSE Scorer | learned | 0.8894 | −0.94pt |
| Random | naive | 0.8635 | −3.53pt |
| FastV (layer-2 attn) | heuristic | 0.8330 | −6.58pt |
| PruMerge (CLS-sim) | heuristic | 0.8085 | −9.03pt |

### Pareto (scorer, Variant A mask)

| r | PDMS | p-value vs r=1.0 | FLOPs saving |
|---|---|---|---|
| 1.0 | 0.8988 | — | 0% |
| 0.75 | 0.8983 | p=0.58 (不显著) | 16.9% |
| 0.50 | 0.8920 | p=6.9e-7 | 33.6% |
| 0.25 | 0.8508 | — | 49.9% |

### Variant B 真剪枝 (shard0, N=2949)

| | PDMS | 说明 |
|---|---|---|
| Variant B 原始 (有 bug) | 0.8758 | 66 catastrophic scenes PDMS=0 |
| **Variant B 修复后** | **0.8948** | 66 scenes fallback to r=1.0 |
| Variant A (mask) | 0.8920 | 对照 |

**结论**: 真剪 > mask（+0.28pt），15% wall-clock 加速，38.3% 序列压缩。

### 7B Offline 分析

| 指标 | 3B | 7B | 结论 |
|---|---|---|---|
| Scorer pairwise acc | 0.8388 | **0.8562** | 7B 更 learnable |
| Top-25% tokens attention 占比 | 91.6% | **95.9%** | 7B 更集中 = 冗余更大 |
| Top-50% tokens attention 占比 | 98.0% | **99.1%** | 同上 |

### τ-cut Adaptive (full navtest)

| τ-cut tag | mean keep-ratio | PDMS |
|---|---|---|
| kr040 | ~40% | 0.8762 |
| kr050 | ~50% | 0.8868 |
| kr060 | ~60% | 0.8940 |
| kr070 | ~70% | 0.8960 |

---

## RL 效果不好的 Root Cause 分析

**发现**: RL best (0.889) < SFT (0.892)

**原因**:
1. `rl_pdm_score()` 返回的是 PDMS 最终乘积（6 个子指标相乘后的单标量）
2. 6 个子指标里 5 个 >0.96，只有 progress=0.83 有改进空间
3. 乘积形式导致：任何一个子项=0 → 整个 reward=0（过于稀疏）
4. 同一 scene 不同 token selection 的 PDMS 差异 <0.01（advantage ≈ 0，信号被噪声淹没）

**改进方案 (Option 3, 已实现)**:
```
reward = Σ_i w_i * (sub_i_pruned - sub_i_baseline[scene])
```
权重: progress=0.35, collision=0.20, drivable=0.15, ttc=0.15, direction=0.10, comfort=0.05

**代码改动**:
- `code/third_party/AutoVLA/models/utils/score.py`: `rl_pdm_score()` 添加 `shaped=True` 模式
- `scripts/train_scorer_grpo.py`: 添加 `--shaped-reward` 和 `--baseline-scores` 参数

---

## 下次 42h 窗口 TODO

### 必须做（论文需要）

| # | 任务 | 预计耗时 | 依赖 |
|---|---|---|---|
| 1 | nuScenes val data 解压 + QA generation | 1-2h | 数据已下载 |
| 2 | Impromptu-VLA 7B eval with scorer (r=0.25/0.5/0.75/1.0) | 4-6h | #1 |
| 3 | RL shaped reward 重训 (3B, 4卡) | 4-5h | baseline sub-scores |
| 4 | RL scorer eval on navtest | 3h | #3 |
| 5 | SparseVLM r=0.75 + Variant B r=0.75 full (本周期可能没跑完) | 2-3h | — |
| 6 | 论文 Abstract + Introduction 重写 | 2h | 所有数据 |

### 优先级
1. **Impromptu-VLA 7B 对比 FastDriveVLA** → 最高价值
2. **RL shaped reward 重训** → 如果能超 SFT，RL 回升为主贡献
3. **论文写作** → 28 号交稿

### Baseline sub-scores 准备
下次窗口第一件事：从 r=1.0 full navtest CSV 提取每个 scene 的 6 个子指标，保存为 JSON：
```bash
python -c "
import pandas as pd, json
df = pd.read_csv('results/raw/tokenprune_S3_full/MT_attn_L12_r10_sh0.csv')  # + sh1-3
# 提取子指标列 → {token: {collision: x, progress: y, ...}}
"
```

---

## 代码改动清单（本周期全部）

| 文件 | 改动 |
|---|---|
| `code/rldrive/scoring/run_feature_dump.py` | 允许空 checkpoint |
| `code/rldrive/scoring/run_attention_probe.py` | 允许空 checkpoint |
| `code/rldrive/agents/autovla_with_attention.py` | try/except predict + dummy |
| `code/rldrive/agents/autovla_with_token_prune.py` | varB denylist + Path import |
| `code/rldrive/agents/autovla_7b_adapter.py` | 新文件：7B adapter |
| `code/third_party/AutoVLA/config/training/qwen2.5-vl-7B-*.yaml` | 添加 inference section |
| `code/third_party/AutoVLA/models/utils/score.py` | **shaped reward (Option 3)** |
| `scripts/train_scorer_grpo.py` | shaped-reward + baseline-scores args |
| `scripts/run_7b_pipeline.sh` | --multi-layer → --all-layers bugfix |
| `scripts/run_impromptu7b_nuscenes_eval.py` | 新文件：7B nuScenes eval |
| `scripts/prepare_nuscenes_eval_data.sh` | 新文件：数据准备 |
| `paper/aaai2027/main.tex` | 主表+ablation+discussion 用真数据重写 |

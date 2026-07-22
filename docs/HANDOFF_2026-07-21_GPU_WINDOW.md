# GPU 窗口执行手册 (2026-07-21)

**资源**: 5× H20 (97GB each), ~21h  
**目标**: 跑完论文所需的最后几个实验  
**环境**: `conda activate navsim`  
**项目根目录**: `/apdcephfs/private_shayladeng/tokenrl_autoVLA`

---

## 项目一句话介绍

我们在做 **自动驾驶 VLA 的 vision token 动态剪枝**。在 AutoVLA (3B, Qwen2.5-VL) 上，训了一个 0.6M 参数的 MLP scorer 决定哪些 vision token 重要。现在要用 RL (shaped driving reward) 微调这个 scorer 使其超越 SFT baseline。

---

## 优先级排序

| P | 任务 | GPU | 预计时间 | 产出 |
|---|------|-----|---------|------|
| **P0** | 动态证据离线分析 (CPU only) | 0 | 3min | 论文 figure |
| **P1** | RL shaped reward 训练 | 4卡 | 4-5h | RL scorer ckpt |
| **P2** | RL scorer eval on navtest | 4卡 | 3h | PDMS 数字 |
| **P3** | Impromptu-VLA 7B nuScenes eval | 4卡 | 4-6h | 泛化性数据 |

**P1+P2 是最高优先级**（论文 RL 主贡献数据）。P3 是 bonus。

---

## P0: 动态证据 (CPU, 3 分钟)

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
conda activate navsim
python scripts/analyze_taucut_dynamic.py
```

**输出**:
- `results/analysis/taucut_dynamic_stats.json`
- `results/analysis/taucut_dynamic_histogram.png`
- `results/analysis/taucut_dynamic_scatter.png`

**验证**: stats.json 中 `std_keep_ratio > 0.05` 且 `difficulty_analysis` 显示难场景 keep ratio > 简单场景。

---

## P1: RL Shaped Reward 训练

### 前置: 准备 baseline sub-scores JSON

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
conda activate navsim

# 从 r=1.0 full navtest CSV 提取每个 scene 的 6 个子指标
python3 -c "
import pandas as pd, json
from pathlib import Path

root = Path('results/raw/tokenprune_S3_full')
rows = {}
for sh in range(4):
    csv = root / f'MT_attn_L12_r10_sh{sh}.csv'
    if not csv.exists():
        print(f'WARN: {csv} not found'); continue
    df = pd.read_csv(csv)
    df = df[df['token'] != 'average']
    for _, r in df.iterrows():
        if r.get('valid', True):
            rows[r['token']] = {
                'collision': float(r['no_at_fault_collisions']),
                'drivable': float(r['drivable_area_compliance']),
                'progress': float(r['ego_progress']),
                'ttc': float(r['time_to_collision_within_bound']),
                'comfort': float(r['comfort']),
                'direction': float(r['driving_direction_compliance']),
            }
print(f'Extracted {len(rows)} scenes')
with open('results/baseline_sub_scores.json', 'w') as f:
    json.dump(rows, f)
print('Saved to results/baseline_sub_scores.json')
"
```

**验证**: `wc -l results/baseline_sub_scores.json` 应有 ~11575 个 entries。

### 启动 RL 训练

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
conda activate navsim

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_scorer_grpo.py \
    --scorer-ckpt ckpt/s3_token_scorer \
    --out-dir ckpt/s3_token_scorer_rl_shaped \
    --keep-ratio 0.5 \
    --num-epochs 3 \
    --group-size 8 \
    --lr 3e-5 \
    --kl-beta 0.01 \
    --shaped-reward \
    --baseline-scores results/baseline_sub_scores.json \
    --seed 42
```

**注意**:
- `--scorer-ckpt ckpt/s3_token_scorer` 是 LambdaRank SFT ckpt（不是 MSE 的）
- `--shaped-reward` 启用 Option 1+3 合并 reward
- `--baseline-scores` 指向刚生成的 JSON
- 预计 4-5 小时，会自动保存 best ckpt

**监控**: 看 stdout 中的 `mean_reward` 是否逐 epoch 上升。

---

## P2: RL Scorer Eval on Navtest

RL 训练完成后，评估：

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
conda activate navsim
source scripts/setup_navsim_env_vars.sh

# 4-shard 并行评测
for SH in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SH python scripts/run_tokenprune_eval.py \
        --scorer-ckpt ckpt/s3_token_scorer_rl_shaped \
        --keep-ratio 0.5 \
        --shard-id $SH --num-shards 4 \
        --out-csv results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh${SH}.csv \
        --selector scorer &
done
wait
echo "All 4 shards done"
```

**如果没有 `run_tokenprune_eval.py`**，用现有的 sweep 脚本：
```bash
# 检查可用的评测脚本
ls scripts/run_tokenprune*.sh scripts/run_tokenprune*.py
```

**验证**: 合并 4 shard CSV，计算 mean PDMS。如果 > 0.8920 (SFT baseline)，则 **RL 成功超越 SFT**。

```bash
python3 -c "
import pandas as pd
from pathlib import Path
dfs = []
for sh in range(4):
    p = Path(f'results/raw/tokenprune_S3_full/MT_rl_shaped_r05_sh{sh}.csv')
    if p.exists():
        df = pd.read_csv(p)
        df = df[df['token']!='average']
        dfs.append(df)
if dfs:
    all_df = pd.concat(dfs)
    print(f'N={len(all_df)}, mean PDMS={all_df[\"score\"].mean():.6f}')
    print(f'SFT baseline: 0.8920 → RL gain: {all_df[\"score\"].mean() - 0.8920:.4f} pt')
"
```

---

## P3: Impromptu-VLA 7B nuScenes Eval (Bonus)

这个更复杂，取决于之前的下载和准备是否完成。

```bash
# 检查 Impromptu-VLA 是否已下载
ls models/Impromptu-VLA-7B*/config.json 2>/dev/null

# 检查 nuScenes val data
ls data/nuscenes_val/ 2>/dev/null
```

如果都就绪，参考 `scripts/run_impromptu7b_nuscenes_eval.py`。如果没有就跳过，论文用 offline 7B 分析 (已有数据) 足够。

---

## 关键路径和文件

| 用途 | 路径 |
|------|------|
| SFT scorer (LambdaRank) | `ckpt/s3_token_scorer/` |
| MSE scorer (for τ-cut) | `ckpt/s3_token_scorer_mse/` |
| Feature dump (4000 scenes) | `data/s3_scorer/features/*.pt` |
| Navtest r=1.0 baseline CSV | `results/raw/tokenprune_S3_full/MT_attn_L12_r10_sh*.csv` |
| τ-cut 结果 | `results/raw/tokenprune_taucut/TC_mse_tau_kr*.csv` |
| RL 训练脚本 | `scripts/train_scorer_grpo.py` |
| 论文决策记录 | `paper/aaai2027/story_decisions.md` |
| 最终结果汇总 | `docs/journal/2026-07-21_final_results.md` |
| 环境变量 | `scripts/setup_navsim_env_vars.sh` |

---

## 关键数字 (已确定，不要覆盖)

| 指标 | 值 | 来源 |
|------|-----|------|
| No-prune baseline | 0.8988 | full navtest |
| SFT scorer r=0.5 | 0.8920 (−0.69pt) | full navtest |
| Attention L12 r=0.5 | 0.8901 (−0.87pt) | full navtest |
| Random r=0.5 | 0.8635 (−3.52pt) | full navtest |
| FastV r=0.5 | 0.8330 (−6.58pt) | full navtest |
| τ-cut kr060 | 0.8940 (−0.48pt) | full navtest |
| Variant B | 0.8948 (−0.40pt, 1.15×) | shard0 |
| r=0.75 (free-lunch) | 0.8983 (p=0.58) | full navtest |

**目标**: RL shaped reward scorer r=0.5 PDMS > 0.8920

---

## 止损

| 情况 | 动作 |
|------|------|
| RL 训 3 epoch 后 reward 没涨 | 检查 baseline_sub_scores.json 是否正确加载；尝试调 lr 到 1e-5 |
| RL eval 仍 < SFT (0.8920) | 论文保留当前写法（RL as negative insight + reward design ablation）|
| OOM | 减 group_size 到 4；或单卡跑 |
| 脚本找不到 module | `export PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code:$PYTHONPATH` |

---

*生成时间: 2026-07-21 17:08*

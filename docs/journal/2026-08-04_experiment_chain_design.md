# 2026-08-04 无人值守实验链设计

> 写于 19:30，训练 step 37/300，kr 稳定。
> 目标：训练完成后自动执行全量 eval + 消融实验，占满到明天 10:00。
> 约束：4×H20，无人接管。

## 时间线与资源分配

```
NOW  19:30  v4 训练 step 37/300 (kr 0.51-0.55, 稳定)
     00:20  v4 训练完成 (~300 steps, ~5.8h)

GPU 分配策略（4 GPU）：
BLOCK1  00:20-04:00  (3.7h)
  GPU0: v4 ckpt FINAL  eval shard0
  GPU1: v4 ckpt FINAL  eval shard1  
  GPU2: v4 ckpt FINAL  eval shard2
  GPU3: v4 ckpt FINAL  eval shard3
  → 产出 v4 全量 PDMS

BLOCK2  04:00-07:30  (3.5h)
  GPU0: v4 ckpt BEST   eval shard0+shard1 (串行, ~1.8h+1.8h 不够！)
  
  实际方案：消融实验用 shard0-only quick eval（~50min each）
  GPU0: v4 ckpt_best eval shard0 (~50min)
  GPU1: v4 prune_variant=drop eval shard0 (~50min) 
  GPU2: SFT r=0.355 full eval shard0+1+2+3 (已有部分CSV，只补缺)
  GPU3: 汇总脚本运行

  04:00-04:50  v4_best shard0 eval (GPU0)
               v4_drop  shard0 eval (GPU1)
  04:50-05:40  v4 kr analysis (GPU0 idle → 汇总)
               
  实际更优方案：因为只有 4 GPU 且 eval 是串行的（max_workers=1），
  每个 shard eval ~50min。可以做多个 shard0-only eval 快速对比。

BLOCK3  07:30-10:00  (2.5h)
  汇总所有结果 + 生成报告
```

## 实际可行方案（4 GPU, eval 单线程）

每个 shard eval ~50min（2949 scenes, max_workers=1）。

时间线重算：
```
00:20-04:00  (3.7h)
  GPU0: v4_FINAL shard0 (0:20-1:10) → shard1 (1:10-2:00) → shard2 (2:00-2:50) → 汇总
  GPU1: v4_FINAL shard1 (0:20-1:10) → shard2 (1:10-2:00) → shard3 (2:00-2:50)
  GPU2: v4_FINAL shard2 (0:20-1:10) → shard3 (1:10-2:00) → 汇总
  GPU3: v4_FINAL shard3 (0:20-1:10) → 空闲 → 汇总
  → 所有 4 shard 在 1:10 完成（并行）
  → 1:10 开始汇总 + 额外实验

01:10-04:00  (2.8h)
  GPU0: v4 ckpt_best eval shard0 (1:10-2:00)
  GPU1: v4 prune_variant=drop eval shard0 (1:10-2:00)
  GPU2: SFT r=0.355 eval shard0 (1:10-2:00) [已有全量 0.85575，补 shard0 确认]
  GPU3: 汇总脚本 + kr 分布分析
  → 2:00 全部完成

02:00-04:00  (2h) — 如果 v4 PDMS 不够好，跑额外实验
  但此时我们无法知道结果...所以需要预设条件分支

04:00-10:00  (6h)
  GPU0-3: 全量 eval 任何新的 variant（如果有的话）
  或: 汇总 + 报告生成
```

## 决策：链式脚本设计

链 1（主链）：训练 → v4 eval → 消融 eval → 汇总
链 2（条件链）：如果 v4 PDMS > 0.89，跳过额外训练；如果 < 0.88，自动启动 v4_round2

关键：eval 结果决定后续动作，但无人值守。用 bash 条件判断实现。

### 链结构

```
train_v4 (进行中)
  ↓
wait_train + aggregate
  ↓
eval_v4_full (4 GPU parallel, 4 shard)
  ↓
aggregate_v4 → PDMS
  ↓
if PDMS > 0.89:
    eval_v4_best + eval_v4_drop + eval_sft_baselines
    → aggregate_all → report
else:
    train_v4_round2 (lower efficiency_beta=0.02, 1 epoch, ~5.8h)
    → eval_v4r2_full → aggregate → report
```

时间检查：
- 如果 PDMS > 0.89（好情况）：eval 链 1:10-4:00 完成，剩余 6h 做额外分析
- 如果 PDMS < 0.88（坏情况）：train round2 从 ~1:30 开始，~5.8h → 7:20 完成，eval ~3.7h → 11:00（超出回收时间！）

**结论：不能做 round2 训练，时间不够。如果 v4 不够好，只能靠消融实验提供分析素材。**

### 最终方案：单链，无分支

所有实验预设好，顺序执行：
1. v4 训练（进行中）
2. v4 FINAL ckpt 全量 eval（4 shard 并行）
3. v4 ckpt_best shard0 eval
4. v4 prune_variant=drop shard0 eval  
5. SFT baselines 全量 eval（补缺）
6. 汇总报告

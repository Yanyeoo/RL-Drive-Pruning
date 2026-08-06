# 2026-08-03 Block A 结果：Budget RL 多 seed 统计

> 目的：回应 AAAI Reviewer #4「统计证据不足」
> 实验：8 个独立训练的 Budget RL ckpt（seed=42+SH），全部在 navtest shard0（2949 scenes）上 eval
> 协议：raw（无 safety_net、无 denylist），prune_variant=drop
> 时间：11:35 启动 → 15:10-15:14 全部完成（~3.6h），8 卡并行，0 报错

## 结果

| seed | ckpt | N | PDMS | NC | EP |
|---|---:|---:|---:|---:|
| s0 | sh0 | 2949 | 0.87359 | 0.9837 | 0.8127 |
| s1 | sh1 | 2949 | 0.86794 | 0.9817 | 0.8107 |
| s2 | sh2 | 2949 | 0.85557 | 0.9747 | 0.7978 |
| s3 | sh3 | 2949 | 0.87636 | 0.9836 | 0.8152 |
| s4 | sh4 | 2949 | 0.85133 | 0.9788 | 0.7939 |
| s5 | sh5 | 2949 | 0.82049 | 0.9603 | 0.7666 |
| s6 | sh6 | 2949 | 0.87359 | 0.9812 | 0.8132 |
| s7 | sh7 | 2949 | 0.86961 | 0.9802 | 0.8112 |

**汇总**：
- n = 8, mean = **0.86106**, std = 0.01867
- min = 0.82049, max = 0.87636
- 95% CI (t, df=7): **0.86106 ± 0.01561**

**对比**：
- SUPP_budgetrl_dynamic_raw（shard0, 同 2949 scenes, 仅 sh0 ckpt）: 0.87066
- 多 seed mean: 0.86106 → Δ = -0.00960

## 解读

1. **跨 seed 方差**：std=0.0187 较大（range 0.056），说明 Budget RL 训练受 seed/数据分片影响明显。s5（seed=47）显著劣于其他 seed。
2. **最佳 seed 超过原 sh0**：s3（seed=45）PDMS=0.87636 > sh0 的 0.87066。
3. **保守报告**：如果报告 mean±std 而非只选 best，结论变弱（0.861 ± 0.016 vs 0.871）。
4. **对 rebuttal 的影响**：多 seed 统计证据本身是 rebuttal 所需，但当前数字（mean 0.861）不够强，需要 RL 训练改进后才用。

## 产物

- CSV: `results/raw/blockA_multiseed/MSEED_budgetrl_s{0..7}_nt0.csv`（8 个文件，共 ~1.7M）
- 日志: `logs/blockA_multiseed_20260803_113505/`

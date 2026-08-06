# 2026-08-03 主线决策：将本周期任务从「补充实验」改为「重新训练 Budget RL」

> 用户明确：`0.9107` 确认是假的，主线任务是训练 RL 达到 SOTA。
> 时间窗口：H20×8 今晚 24:00 回收。

## 决策

### 当前状态

- **Block A（多 seed eval）**：已启动于 11:35，8 卡并行，预计 15:15-15:30 完成。已有 0.9107 不成立的多项证据，多 seed 结果对 rebuttal 仍有价值但不 urgent。
- **GPU**：8 卡满载（Block A 占用），无遗留进程。

### 决策：Block A 跑完后再开始 RL 训练（方案 B）

理由：
1. Block A 已投入 8 卡模型加载（17.7G/卡），kill 浪费 ~30min 加载成本
2. Block A ETA 15:15-15:30，剩余 8.5h 足够跑 1-2 epoch Budget RL
3. 历史训练曲线：900 steps, 3 epochs, ~17h total。1 epoch ≈ 300 steps ≈ 5-6h。
4. 原训练从 SFT scorer 初始化（raw PDMS=0.8707），训练结束后 best ckpt 的 PDMS 约 0.8486（变差了！）
   → 这说明当前 REINFORCE 方案可能已经到瓶颈，继续原方案未必有效
   → 需要策略改动

### 关键发现：原 RL 训练反而降低了 PDMS

```
SFT scorer (Stage-1 only) raw:        PDMS ≈ 0.89199 (r=0.5)
SFT scorer r=0.355 raw:               PDMS ≈ 0.85575
Budget RL dynamic raw (best ckpt):    PDMS = 0.84865  ← 比 SFT 更差！
```
**Stage-2 Budget RL 训练没有提升 PDMS，反而降低了。** 这与 supplement 宣称的 0.9107 完全相反。

### 本周期 RL 训练策略

由于原 REINFORCE + Gaussian budget 方案已经收敛到比 SFT 更差的点，继续用同样超参再训大概率不会突破。
需要至少做一个关键改动。

**策略 A（最小改动，今晚可跑完）：改 reward 权重 + 增大 efficiency_beta**

当前 reward：
- driving_scale=2.0, efficiency_beta=0.15
- 训练曲线 kr 从 0.55 降到 0.30（efficiency bonus 推得太狠，损害了 PDMS）

改进方向：降低 efficiency_beta（减少过度剪枝的激励），增大 driving_scale（让 reward 更关注质量）

预计：1 epoch 可完成，PDMS 是否能超过 SFT 的 0.892 不确定。

**策略 B（ICLR 路线，今晚可完成框架但训练不完）：Per-token counterfactual credit assignment**

改动 reward 分解方式，让每个 token 有独立 advantage。代码量约 200-300 行改动。
今晚可能只能完成 smoke test（1-2 epoch），全量训练需要后续 GPU 窗口。

### 决策：策略 A 立即启动，策略 B 代码并行开发

- **Block A 结束后（15:30）**：启动 8 GPU Budget RL 训练，超参：efficiency_beta=0.05, driving_scale=3.0, 1 epoch
- **Block A 跑期间的 3.5h**：在后台实现 Per-token REINFORCE 代码改动
- 策略 A 结果今晚出来，作为 baseline；策略 B 代码 ready，下次 GPU 窗口直接跑

---

## 本周期剩余时间线

```
11:35-15:30  Block A (多 seed eval) ← 已启动
15:30-21:00  RL train 策略A 1 epoch (5.5h)
21:00-24:00  eval 策略A on navtest (3.6h 全量)
             如果策略A > SFT 0.892: 成功！继续跑更多 epoch
             如果策略A < SFT 0.892: 证明需要策略B，下次窗口直接上
             + 离线分析：修正 supplement + 产周期报告
```

## Reverse 指令

如果用户认为应该立即 kill Block A 改跑 RL：
```bash
kill $(cat logs/blockA_multiseed_20260803_113505/pids.txt) 8930 32346
# 然后启动 RL 训练（策略 A 或策略 B）
```

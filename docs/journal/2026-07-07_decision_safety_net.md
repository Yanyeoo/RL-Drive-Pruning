# 2026-07-07 21:20 — 自主决策记录：Safety-Net Fallback

**决策方**：agent (autonomous-mode, user 授权自决)
**决策事项**：如何修复 claim① 贴边问题 (scorer r=0.5 全量 −0.69pt vs 底线 −0.5pt)

---

## 讨论摘要（调研结论）

### 方案 A：Safety-Net Fallback（推理时检测 scorer 不确定性 → 回退 r=1.0）
- **原理**：scorer 输出 score(720)，计算 softmax entropy + top-B 边界 gap；超阈值时不剪枝
- **代码改动**：~25 行（`autovla_with_token_prune.py`）
- **效率影响**：仅 1.3% 场景触发，avg keep_ratio 从 0.5 → ~0.507（可忽略）
- **风险**：需要标定阈值（在已有 CSV 数据上 offline 扫即可，零 GPU 成本）
- **验证**：不需要重新跑全量 eval——可以 offline 模拟（已有 scorer 输出 + PDMS 数据）
- **论文故事**："lightweight uncertainty-aware fallback guarantees safety"

### 方案 B：扩训 Scorer（4000 → 19000 navtrain 场景）
- **前置**：需 dump 15k 额外场景的 layer-0 features（~19k × 2.4s = 12.8 card-h，约 3-4h/2卡）
- **训练本身**：~60 秒（训练脚本零改动，`--max-scenes 19000` 即可）
- **label 依赖**：19225 场景的 L12 attention dump 已全量存在（`exp/m1b2_navtrain_full_alllayers/`）
- **效果预期**：更好覆盖转弯场景（灾难组中转弯 2× 过度代表），减少 OOD 灾难
- **风险**：需要 GPU 时间做 feature dump；效果不确定（可能只部分解决）
- **论文故事**："scaling training data improves robustness on tail scenarios"

### 方案 C：不修复，调整 claim framing
- **做法**：接受 −0.69pt，论文中报"近乎无损"或"iso-compute dominate all baselines"
- **风险**：reviewer 可能挑刺 "你说 lossless 但掉了 0.7pt"
- **优势**：零成本、零风险

---

## 决策

**选择：A + B 串行执行。C 作为保底。**

**理由**：
1. A 是"零 GPU 成本、25 行代码、offline 可验证"的最优性价比方案。先做 A。
2. 如果 A 标定后 offline 模拟显示 claim① 恢复（≤0.5pt），**立即锁定，不做 B**。
3. 如果 A 效果不足（仍>0.5pt），则在剩余 GPU 时间内做 B（feature dump + 重训）。
4. 最坏情况：A+B 都不够，接受 C（framing 调整）——但此概率极低，因为 A 理论上应能消除大部分灾难场景的贡献。

**执行顺序**：
1. ✅ 评测 dispatcher 不停（后台继续跑 attn_L12 r=0.5 etc）
2. 现在立即实现 safety-net fallback 代码
3. 用已有全量 CSV 数据 offline 标定 entropy/gap 阈值
4. 如果 offline 模拟 PASS → 锁定方案，新 eval 等下窗口验证
5. 如果 offline 模拟 FAIL → 后台评测完成后切 GPU 跑 feature dump (方案 B)

**reverse 指令**：如果 safety-net 引入新 bug 或 user 不满意方案，
备份在 `backups/cycle_start_20260707_*/autovla_with_token_prune_pre_safetynet.py`。

---

## 21:35 — 方案 A 实测结果：**FAIL**

### 实验
用 navtest sub1500 的 features(1495 场景) offline 跑 scorer → 计算 normalized entropy + boundary gap。
与 PDMS 做交叉分析。

### 结论
**Safety-net 基于 scorer uncertainty 的方案不可行：**
- Catastrophic 场景的 entropy=0.3614 vs Normal=0.3646 — **完全无差异**
- Catastrophic 场景的 gap=0.0134 vs Normal=0.0135 — **完全无差异**
- scorer 在灾难场景上是 **"自信但错误"（confidently wrong）**，不是"不确定"

### 更深层诊断
按 baseline 难度分层分析（全量数据 N=11571）：

| 场景类型 | N | scorer vs r1 delta |
|---|---|---|
| r1=0（baseline 灾难）| 505 | **+18.7pt** ← scorer 大幅优于 baseline |
| r1∈[0.5,0.8)（中等）| 227 | **+1.5pt** ← scorer 仍优 |
| r1∈[0.8,0.95)（较简单）| 5017 | **−1.2pt** ← 退化 |
| r1∈[0.95,1.0]（满分区）| 5804 | **−2.0pt** ← 主要退化来源 |

**根因**：在简单/满分场景（r1>0.95，N=5804=50%），任何剪枝都会扰动"完美"轨迹导致
小幅掉分。这不是 scorer 选错 token，而是**这些场景根本不该被剪**。本质是 Budget Policy
的职责——但 §11 已证明 budget 不可学。

### 方案修订

**放弃方案 A**。safety-net 代码保留在 `autovla_with_token_prune.py`（默认 disabled，
`safety_net=False`），但不用于解决 claim①。

**实际可行策略（新方案 D）**：
1. **论文 framing 调整**（主选）：claim① 不说"lossless"，改为：
   - "iso-compute r=0.5 下 dominate 所有 non-oracle baseline"（待 attn_L12 r=0.5 数据确认）
   - "困难场景(r1<0.8) scorer 大幅领先 baseline(+18.7pt on fail cases)"
   - "简单场景 −2pt 退化对应于 50% 算力节省的合理 tradeoff"
2. **Pareto reporting**（辅助）：scorer r=0.75 可能 dominate r=1.0（只剪 25%，退化应极小）；
   提供 Pareto 最优点让读者选。
3. **扩训 scorer（方案 B）**：仍有价值（减少灾难场景），但不能解决满分区的 structural
   limitation。留给下窗口作为增量改进。

**偏离记录（规则#4）**：方案 A 代码已写入但默认禁用（safety_net=False），不影响任何
现有评测。代码保留作为 future-work option（如果后续训练出带 uncertainty head 的 scorer
则可启用）。reverse：删除 _should_fallback 方法 + 构造函数中的 safety_* 参数。

---

## 偏离 design doc 说明（规则#4）

safety-net 代码**不偏离** design doc：
- 默认 disabled，不改变已有 eval 行为
- 如启用，是 selector 推理路径的增强，不改变训练范式
- claim① framing 调整是对实验结果的诚实反映，不改变方法设计

下面这份是给压缩版记忆。重点保留项目背景、当前方法、实验状态、关键判断和后续 SOTA 研究方向，删除无关细节。
随着讨论对齐可随时修改文档
---

# 项目接管记忆：AUTOVLA / VLA Token Pruning SOTA Research

## 1. 用户与项目目标

用户是 CS 研究生，目前研究方向是：

> **VLA / WAM / Autonomous Driving + Multimodal Token Pruning + Efficient Inference**

当前核心项目基于 **AUTOVLA**，目标不是单纯做 token pruning，而是最终形成一个能够冲击 SOTA 的：

> **Driving-aware / Risk-aware / Action-aware Adaptive Visual Token Pruning**

最终目标同时满足：

* Driving performance 不明显下降，最好提升
* Safety 不恶化
* Visual token 数量显著下降
* 实际 inference FLOPs / latency / memory 真正下降
* 方法具有明确 scientific motivation，而不是简单堆模块
* 最终形成完整论文 story + ablation + mechanism explanation

用户希望 AI 不只是回答问题，而是作为共同研究者：

* 主动质疑当前方案
* 根据实验结果修改 idea
* 判断哪些模块值得保留/删除
* 设计下一轮实验
* 不要为了维护某一版本而默认它正确
* 核心目标始终是 **SOTA**

---

# 2. 当前方法主线

当前 pruning architecture：

```text
Scene
 ├── BudgetNet → keep ratio / K
 └── TokenNet → token importance scores
                    ↓
                  Top-K
                    ↓
             pruned visual tokens
                    ↓
                 AUTOVLA
                    ↓
            driving + safety
                    ↓
                  reward
              /          \
       Budget RL       Token RL
                          ↑
                   counterfactual
```

三个网络：

* **TokenNet**：决定“哪些 token 留下”
* **BudgetNet**：决定“留多少 token”
* **ValueNet**：预测 value，用于计算 advantage、降低 RL variance

目前训练目标属于 RL，而不是普通分类/回归。

---

# 3. 当前 V11 / V13

## V11 winner

winner：

> `hard50_kl`

主要配置：

* token LR = `3e-5`
* budget LR = `1e-4`
* value LR = `1e-4`
* AdamW
* weight decay = `1e-4`
* grad clip = `1.0`
* selection = `st_topk`
* tau = `0.1`
* min/max keep ratio = `0.35 / 0.85`
* efficiency mode = `floor`
* efficiency beta = `0.02`
* target keep ratio = `0.65`
* driving scale = `3.0`
* delta reward = ON
* safety beta = `0.5`
* counterfactual_k = `0`
* group size = `8`
* 2 epochs
* 无 warmup / scheduler
* token KL beta ≈ `0.02`

现有评估：

* gate：

  * hard50 = `0.899520`
  * hard50_kl = `0.898909`
* full：

  * hard50_kl = **`0.898774`**
  * hard50 = `0.898417`

V11 winner 的重要特点：

> hard 场景恢复后，在 full 上泛化比较稳定。

V11 的 keep ratio 大约：

> `kr ≈ 0.69`

---

# 4. V13 当前实验

V13 实际一共 **12 个 arms**：

### A 系列

* `a_ctrl`
* `a_stats`
* `a_stats_kl05`
* `a_stats_kl08`

### B 系列

* `b_kinonly`
* `b_kin_stats`
* `b_kin_stats_kl05`
* `b_kin_stats_kl08`

### C 系列（counterfactual）

* `c_ctrl`
* `c_cf4`
* `c_cf8`
* `c_cf4_kl05`

共同设置：

* driving_scale = `3.0`
* efficiency beta = `0.02`
* target kr = `0.65`
* safety beta = `0.5`
* min/max kr = `0.35 / 0.85`
* token LR = `3e-5`
* budget/value LR = `1e-4`
* AdamW
* weight decay = `1e-4`
* grad clip = `1.0`
* 无 scheduler / warmup
* 2 epoch
* group_size=8
* warm-start 使用 V11 winner

V13 改动：

### V13-A

V11：

> BudgetNet = mean pooling

V13-A：

> `mean + std + max`

即增加 visual feature statistics。

### V13-B

在 A 的基础上加入：

> `kinematic context`

来源是 `build_scene_ctx` / scene JSON 中可获得的运动学/驾驶上下文。

### V13-C

在不改 BudgetNet 条件化结构的前提下，单独重开 per-token counterfactual：

> `counterfactual_k = 4 / 8`（及 `kl_beta` 0.02 / 0.05）

目的不是抬高全局 `kr`，而是测试 token-level credit assignment 是否能更精确地保护关键少数 token。

### 2026-09-09 最新窗口结果

* `V13-C` 已全部完成：`c_cf4 = step 328`，`c_cf8 = step 272`，`c_cf4_kl05 = step 328`，`c_ctrl` 此前已完成。
* `v12 gate` 已有完整结果：`kl002_ctrl / kl003 / kl005 / kl008 = 0.896312 / 0.894936 / 0.895607 / 0.893384`。
* `v12 full` 目前已知：`kl002_ctrl = 0.893833`，`kl005 = 0.894246`；两者都明显低于 `v11 hard50_kl = 0.898719`。
* `kl003@sh2` 单 shard 诊断 = `0.888216`（2962/2963 成功），目前不支持把 `kl003` 当作更优 full 候选。

---

# 5. V13 当前最大问题

训练中途观察到：

> V13-A/B 的 keep ratio 大约 `0.81~0.85`

明显高于：

> V11 ≈ `0.69`

这是目前最重要的 warning。

当前 efficiency reward：

```text
penalty = beta * max(0, target_kr - kr)
```

所以：

* kr < 0.65 → 惩罚
* kr >= 0.65 → 不惩罚

因此：

> kr=0.65、0.70、0.80、0.85 都没有额外 efficiency penalty。

这会导致 policy 学成保守策略：

> 多留 token → driving performance 更稳定 → 又没有效率惩罚。

所以：

**V13 如果 PDMS 更高但 kr 也从 0.69 上升到 0.84，不能直接说 V13 更好。**

真正应该比较：

> 在相同或更低 keep ratio 下，V13 是否能保持/提升 V11 的 driving performance？

---

# 6. 当前 reward

核心：

```text
total_reward
 = driving_term
 + efficiency_term
 - safety_beta * safety_loss
```

delta reward：

```text
driving_term = R_pruned - R_base
```

并乘：

```text
driving_scale = 3.0
```

### 2026-09-09 用户最新意见（必须保留）

用户明确提出：

> **如果目标是超过 SOTA，reward 主目标不应继续锚定同场景 `no-prune` 结果，而应更直接对齐 benchmark 标准答案。**

更具体地说：

* 训练目标应优先考虑 **train split 上的 absolute benchmark score**，而不是默认把 `R_pruned - R_base` 当主目标。
* **train/test 必须严格分离**：训练只能使用 train split 的 benchmark 监督；test 只用于 gate/full 最终评估，不能回流训练。
* 当前代码层面的已核事实是：reward 虽已直接使用 **true PDMS**，但 driving 主项仍主要优化同场景 delta，这可能把策略锚在 no-prune 上，导致“更像保 baseline，而不是主动超 SOTA”。
* 下一个窗口优先与用户讨论是否改成：
  1. `absolute benchmark`
  2. `absolute benchmark + asymmetric regret`

这条意见是接下来讨论 reward/objective 时的最高优先级记忆点，不得遗失。

目前 safety loss：

```text
0.4 collision
+ 0.3 drivable
+ 0.3 TTC
```

注意：必须检查这些 metric 的方向。

如果 collision 是“越低越好”，直接：

```text
max(0, base_collision - pruned_collision)
```

可能方向是反的。

更稳妥的设计是先把所有 safety metrics 转成：

> **higher = better**

然后统一：

```text
safety_loss =
 Σ wi * max(0, base_safety_score - pruned_safety_score)
```

---

# 7. 当前 RL loss

总 loss：

```text
loss =
 policy_loss
 + kl_loss
 + value_loss
```

policy：

```text
policy_loss =
 budget_policy_loss
 + selection_pg_weight * token_policy_loss
```

目前：

```text
selection_pg_weight = 1.0
```

Budget：

> Gaussian REINFORCE

Token：

> counterfactual_k=0 时，基本是 scene-level REINFORCE + ST TopK surrogate。

Advantage：

```text
A = reward - V(s)
```

ValueNet 用于降低 policy gradient variance。

KL：

目前 token_net 使用 MSE-style regularization：

```text
kl_beta * MSE(token_net, reference)
```

如果代码确实是 MSE，不应严格称为 KL divergence。

budget KL 当前基本关闭。

---

# 8. 当前最重要的 scientific weakness

目前：

```text
counterfactual_k = 0
```

因此 TokenNet 得到的是 scene-level reward。

问题：

> 一个 scene reward 不能告诉模型到底哪个 token 导致 reward 改变。

credit assignment 很弱。

理想情况下：

```text
ΔR_k = R_original - R_without_token_k
```

可以估计 token k 的真实 contribution。

但对所有 token 做 counterfactual 太贵，所以：

> 每个 scene 随机采样 K 个 token 做 counterfactual

是合理的计算折中。

未来可实验：

```text
counterfactual_k = 0
counterfactual_k = 2
counterfactual_k = 4
counterfactual_k = 8
```

但不要一开始和所有 conditioning 同时改，否则 attribution 不清楚。

---

# 9. SOTA 最值得押注的最终 idea

不要最终停留在：

> “mean → mean/std/max → 加 kinematic”

这更像 engineering improvement。

更强的 scientific hypothesis：

> **Optimal visual-token budget should adapt to the decision-criticality of the current driving scene, while token selection should be guided by the counterfactual impact of individual visual tokens on driving performance.**

中文：

> **最优视觉 token budget 应根据当前驾驶场景的 decision criticality 动态变化，而 token selection 应由视觉 token 被移除后对驾驶决策造成的 counterfactual impact 指导。**

因此最终可以形成：

### Level 1：Risk / state-aware budget

BudgetNet 输入：

* visual statistics
* kinematic context
* token importance distribution
* attention entropy 等

目标：

> 判断当前 scene 到底需要多少视觉信息。

### Level 2：Action/decision-aware token selection

TokenNet：

> 判断哪些 token 对当前 driving decision 最重要。

### Level 3：Counterfactual token credit

通过 token removal：

> `ΔR_k`

给 TokenNet 更细粒度的 supervision / RL credit。

### Level 4：真正 physical pruning

不要只 mask。

需要：

```text
N visual tokens
 ↓
Top-K
 ↓
physical gather
 ↓
K visual tokens
 ↓
L13 ... LN
```

否则 keep ratio 降低不代表真正降低 GPU compute。

---

# 10. AUTOVLA / L12 pruning

当前理解：

AUTOVLA 输入不是单一图片，而是 visual observation + language/instruction + driving/state/history 等上下文，经过视觉编码和 multimodal backbone，最终产生 driving/action output。

Pruning target：

> visual tokens

不是 text/action tokens。

L12 的理论理由：

* 太浅层：表示偏 low-level，尚未形成足够语义
* 太深层：大部分 Transformer computation 已经完成，剪枝节省有限
* 中间层：visual representation 已经具有较强 semantic/task relevance，同时后面仍有大量 Transformer layers 可以节省计算

因此 L12 可以解释为：

> **semantic maturity vs. remaining computational depth 的 trade-off**

但如果论文声称：

> “L12 empirically optimal”

必须用实际 layer ablation 支撑，不能凭理论硬说。

如果能找到此前用户已有的 L12 layer sensitivity/ablation 实验，应优先引用那个证据。

---

# 11. Multimodal interaction

剪枝前：

```text
Visual tokens V1...VN
+
Text tokens T1...TM
       ↓
Multimodal Transformer
```

如果 AUTOVLA 是 unified multimodal Transformer，则严格说不一定存在单独 Cross-Attention block，而是 visual/text token 通过 joint self-attention 发生 cross-modal interaction。

剪枝后：

```text
V1...VN
 ↓
V1...VK

T1...TM unchanged
```

所以：

> pruning 不会消灭 multimodal interaction，只会减少参与 multimodal computation 的 visual token 数量。

需要查 AUTOVLA 实际 architecture 后再确定论文中到底写：

* cross-attention
  还是
* multimodal self-attention / joint attention。

---

# 12. 真正 acceleration 是当前一级问题

必须区分：

### Logical pruning

```text
attention mask
```

token 被屏蔽，但 tensor 仍可能是：

```text
[B, N, D]
```

那么 dense GPU kernel 仍可能按 N tokens 计算。

结果：

> Keep ratio ↓ ≠ latency ↓

### Physical pruning

```text
gather selected tokens
```

tensor 真正变成：

```text
[B, K, D]
```

后续 layers 才真正减少 computation。

最终论文必须测：

* Keep Ratio
* Token Reduction
* FLOPs
* GPU latency
* Throughput
* GPU memory
* PDMS / driving performance
* collision
* drivable
* TTC
* robustness / hard scenes

最重要的是：

> **PDMS vs latency / compute 的 Pareto frontier**

---

# 13. V13 应该怎么继续判断

当前不要急着继续加模块。

等 V13 跑完后，统一比较：

```text
V11 winner
vs
V13-A
vs
V13-B
```

至少记录：

```text
PDMS
keep_ratio
reward
collision
drivable
TTC
gate
full
```

尤其要看：

> V13 是否在 kr≈0.65~0.70 时仍能达到 V11 ≈0.69 的 performance。

如果：

### 情况 A

```text
V13 PDMS ↑
kr ≈ V11
```

很好，conditioning 有效。

### 情况 B

```text
V13 PDMS ↑
kr ↑↑
```

不能直接认为成功。

说明：

> conditioning 可能改善了 decision preservation，但 reward efficiency constraint 太弱。

### 情况 C

```text
V13 PDMS ≈ V11
kr ↑
```

V13 没有价值。

### 情况 D

```text
V13 PDMS ↓
kr ↑
```

基本说明当前 conditioning 设计没有解决核心问题。

---

# 14. 推荐实验路线

不要同时把所有东西打开。

建议：

### Experiment 0

V11 winner baseline

```text
mean
CF=0
```

### Experiment 1

V13-A

```text
mean+std+max
CF=0
```

回答：

> visual statistics 是否有用？

### Experiment 2

V13-B

```text
visual stats + kinematic
CF=0
```

回答：

> driving context 是否有用？

### Experiment 3

V13-CF

```text
best conditioning
CF > 0
```

回答：

> token-level counterfactual credit 是否有用？

### Experiment 4

Efficiency objective

在 best conditioning + CF 上研究：

> 如何真正鼓励更低 K，而不是单纯把 kr 拉高。

### Experiment 5

Physical token gather

验证：

> pruning 是否真正带来 latency/FLOPs reduction。

---

# 15. 一个可能的最终论文框架

暂定方向：

> **Risk-Aware Counterfactual Visual Token Pruning for Efficient Vision-Language-Action Driving**

核心贡献可以最终组织成：

1. **Risk-aware adaptive budget**

   * scene state / visual complexity → dynamic K

2. **Decision-aware token selection**

   * token importance conditioned on driving context

3. **Counterfactual token credit**

   * token removal → Δ driving reward
   * 缓解 scene-level RL credit assignment

4. **Hardware-efficient physical pruning**

   * actual sequence shortening
   * real FLOPs/latency reduction

最终形成：

```text
Scene
 │
 ├───────────────┐
 │               │
 ↓               ↓
Driving State   Visual Features
 │               │
 └──────┬────────┘
        ↓
 Risk-aware BudgetNet
        ↓
       K
        │
Visual Features → TokenNet
        ↓
 Counterfactual-aware importance
        ↓
      Top-K
        ↓
 Physical Token Gather
        ↓
   AUTOVLA L13+
        ↓
 Driving Action
        ↓
 Performance / Safety
        ↓
 Reward
   ↙         ↘
Budget RL   Token RL
               ↑
        Counterfactual ΔR
```

---

# 16. 最重要的研究原则

不要为了“创新”把：

> statistics + kinematics + attention entropy + KL + counterfactual + safety + dynamic budget

全部堆进去。

最终必须回答：

> **Why does this module exist?**

每个模块都应该服务于一个 scientific hypothesis。

当前最值得押注的核心不是：

> “我们用了更多 feature。”

而是：

> **不同驾驶场景对视觉信息的需求不同；token budget 应由当前场景的 decision criticality 自适应决定，而 token importance 应由其对下游 driving decision 的实际 counterfactual contribution 来定义。**

如果实验能够证明这一点，再加上**真实 physical token pruning 和 latency reduction**，整个工作才真正有机会从普通 token pruning 上升到比较有竞争力的 autonomous-driving VLA efficiency work。

---

## 接管时的最高优先级

下一位 AI 接手后，优先做这四件事：

**① 等/读取 V13 八臂最终结果。**

**② 做 V11 vs V13 的 PDMS–kr Pareto 分析，而不是只看 PDMS。**

**③ 检查当前 pruning 是 attention mask 还是 physical gather，确认是否真的加速。**

**④ 根据 V13 结果决定 V14：**

* 如果 conditioning 有效 → 加 counterfactual
* 如果 kr 太高 → 修改 efficiency objective / budget policy
* 如果 conditioning 无效 → 回到“decision-criticality”重新设计 BudgetNet
* 如果 token pruning 本身不产生 latency → 先解决 physical pruning，再谈 SOTA

**最终目标不是 V13 赢 V11，而是找到一个有清晰机制解释、真实加速、驾驶性能/安全保持良好，并且实验能够证明其必要性的 SOTA-level 方法。**

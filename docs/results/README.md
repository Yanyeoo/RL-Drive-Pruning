# `docs/results/` — Single source of truth for numbers

## Why this folder exists

之前所有关键数字（B0 PDMS、子指标、与 paper 对比……）都散落在：
- `docs/journal/MA2_b0_navtest.md`（详细但深埋）
- 对话历史（一压缩就丢）
- `exp/.../merged.csv`（原始数据，每次都得现算）

结果就是：**用户问"我们 baseline 多少？"时，AI 要重新 grep + 算 + 搜
paper 才能答**。这是 doc 的失败。

`docs/results/key_results.md` 是这个问题的解药 —— **一处汇总所有头条数字**。

---

## SOP — 什么时候必须改 `key_results.md`

### 必须改（强制）

1. **任何 milestone 跑出第一个数字** → 立刻在 §0 表加一行 + 在下面新建 §N
2. **rerun / rerun 后数字漂了 > 0.5%** → 改对应行 + changelog 记录漂移原因
3. **跟外部 reference (paper / leaderboard) 比较** → 必须把 Δ 算出来写进表
4. **环境/infra 状态变化**（cuBLAS bug 修了/数据集到位/换卡）→ 改 §3

### 不要改（防过拟合 doc）

1. 调试中临时数字（rerun 一次就变的）→ 写在 journal，**不要**进 key_results
2. 子任务的中间数字（每个 shard 跑了多少 token）→ journal
3. 未跑完的 partial run → journal，跑完再总结到 key_results

---

## 写法约束

1. **Headline number 要在第一行表里就出现** — 不要让人 scroll
2. **每个数字必须能反查 artifact** — csv 路径 / 命令 / journal 链接
3. **对比 reference 必须有 Δ 和 judgment**（✅ 复现 / ⚠ 略差 / ❌ 不一致）
4. **空表头先写好**（reserved sections）— 后续 milestone 填空，不要重组结构
5. **Changelog 必填** — 任何改动加一行 `| YYYY-MM-DD HH:MM | what changed |`

---

## 与其他 doc 的关系

```
docs/results/key_results.md    ← 数字（你现在这里）
docs/journal/<milestone>.md    ← 怎么跑出来的、有啥坑
docs/plan/implementation_plan.md ← 计划（应该跑啥）
docs/_internal/handoff*.md     ← 给下一个 AI 的快速上下文
```

**新 AI / 新协作者的阅读顺序**：
1. `README.md`（项目是啥）
2. **`docs/results/key_results.md`**（现状跑到哪、效果如何）← 强烈先读
3. `docs/plan/implementation_plan.md`（接下来要干啥）
4. 对应 journal（细节）

---

## 维护者 checklist

跑完一次 milestone 实验后：

- [ ] csv / log / artifact 收齐，路径记好
- [ ] 在 journal 写当天的事（free-form OK）
- [ ] **在 key_results.md 加一行/一节，含**:
  - [ ] headline number
  - [ ] vs reference 比较（Δ + judgment）
  - [ ] sub-metrics breakdown（如适用）
  - [ ] artifact 路径
  - [ ] changelog 记一笔
- [ ] 对外文档需要更新吗？（README / implementation_plan）

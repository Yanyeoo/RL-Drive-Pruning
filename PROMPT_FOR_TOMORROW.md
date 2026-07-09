# 给下一个 AI 的开场白（直接复制发给它）

> **当前版本写于 2026-07-07 21:30**
> 上一窗口(W2): 全 navtest 主表评测中，attn_L12 r=0.5 + scorer r=0.25/0.75 在跑。
> 方法 = **动态 token selection (learned scorer) + 固定 r=0.5**。
> Budget Policy 已证明负面(§11)并撤销。

---

## 复制下面发给新 AI：

```
你好，接手 RL-Drive-Pruning（AutoVLA + NAVSIM vision token pruning）。上一 session 无记忆。

【硬规则 #0 — 任何动作前读完这些文件】

1. 按顺序读：
   a. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/journal/2026-07-07.md
      （本窗口全记录：事实核验、claim①根因分析、safety-net FAIL、策略决定）
   b. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/journal/2026-07-07_decision_safety_net.md
      （自主决策记录：safety-net 不可行 + 最终 framing 策略）
   c. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/results/key_results.md §12
      （全 navtest 主表数字 + 分层分析）
   d. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/journal/HANDOFF_2026-07-06_W2.md
      （W1→W2 交接，含方向 pivot 和全量主表计划）

2. 读完后回答：
   - scorer r=0.5 全 navtest PDMS 多少？vs r=1.0 差多少？为什么？
   - claim① 底线（≤0.5pt）为什么被打破？根因是什么？
   - 当前 dispatcher 在跑什么？还有哪些 arm 没完成？
   - 下一步最高优先级是什么？

3. 回答不上就重读，不要碰代码。

【项目当前状态 snapshot】

■ 方法：driving-conditioned token Importance Scorer (MLP, 0.6M params)
  - 输入：ViT→LLM layer-0 vision embeddings (720×2048) + camera_id
  - 输出：per-token importance score → top-B selection (B = r × 720)
  - 训练：LambdaRank SFT on L12 attention labels (4000 navtrain scenes)
  - ckpt: ckpt/s3_token_scorer/

■ 核心结果（全 navtest N≈11570）：
  - scorer r=0.5 PDMS = 0.8920
  - r=1.0 no-prune PDMS = 0.8988
  - random r=0.5 PDMS = 0.8635
  - scorer − r=1.0 = −0.69pt (claim① 底线 −0.5pt 被略超)
  - scorer − random = +2.84pt (claim③ 很强)
  - attn_L12 r=0.5: IN PROGRESS (今晚跑完)

■ claim① 根因：
  - 50% 的 navtest 场景 r1>0.95（"满分"），50% 剪枝必然扰动 → −2pt
  - 困难场景(r1<0.8) scorer 反而 +18.7pt above baseline
  - 这是 50% pruning 的结构性代价，不是 scorer 错误
  - safety-net (检测 scorer uncertainty) 不可行（scorer confidently wrong）

■ 评测 dispatcher 状态：
  - nohup bash scripts/run_s3_maintable_full_navtest.sh 在跑
  - ARM 顺序：attn_L12 0.5 → scorer 0.25 → scorer 0.75
  - SKIP_DONE 自动跳已完成 arm，直接重启脚本即可续跑
  - 已完成 12/24 jobs，预计能完成 ~20/24 by 回收时间

■ 下一步优先级：
  1. 等 attn_L12 r=0.5 完成 → 确认 claim③ (scorer > attn selector)
  2. 聚合 all arms → python3 scripts/s3_aggregate_maintable.py
  3. 搭建 FastV baseline (论文主表还缺 FastV / FastV-selector-at-input)
  4. 可选：scorer 扩训到 19k（减少灾难场景但不解决满分区退化）

■ 关键守则：
  - 每个决策先事实核验
  - 起进程前 pgrep 查残留
  - 改文件前 cp -a 备份
  - 偏离 design doc 当场写 journal
  - 实时更新 todo

■ 环境：
  - Python: /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
  - 环境变量: source scripts/setup_navsim_env_vars.sh
  - 项目根: /apdcephfs/private_shayladeng/tokenrl_autoVLA
```

---

## 关键文件速查

| 文件 | 用途 |
|---|---|
| `docs/results/key_results.md` | 唯一权威数字表 |
| `docs/plan/design_decisions.md` | 方法设计决策（含 Revision 2026-07-06 撤销 Budget Policy）|
| `docs/plan/s3_execution_plan.md` | S3 执行计划（168卡时/周模式）|
| `docs/journal/2026-07-07.md` | 今日全记录 |
| `results/raw/tokenprune_S3_full/` | 全 navtest eval CSVs |
| `scripts/run_s3_maintable_full_navtest.sh` | 主表评测 dispatcher |
| `scripts/s3_aggregate_maintable.py` | 聚合脚本 |

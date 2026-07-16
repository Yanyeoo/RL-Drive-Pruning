# 给下一个 AI 的开场白（直接复制发给它）

> **当前版本写于 2026-07-16 15:20**
> 本窗口(W7) 完成：Scorer GRPO 代码实现+验证、Variant B 分析。
> 下一窗口核心：42h 8卡集中出齐所有论文实验数据。

---

## 复制下面发给新 AI：

```
你好，接手 RL-Drive-Pruning（AutoVLA + NAVSIM vision token pruning）。上一 session 无记忆。

【硬规则 #0 — 任何动作前读完这些文件】

1. 按顺序读：
   a. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/journal/2026-07-16.md
   b. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/journal/2026-07-15.md
   c. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/plan/aaai_proposal_for_advisor.md
   d. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/results/key_results.md §12

2. 读完后回答：
   - Scorer GRPO 是什么？代码在哪？pilot 结果如何？
   - 42h 窗口的 win conditions 是什么？
   - Variant B 的问题是什么？如何修？
   - 当前论文最大的 gap 是什么？

3. 回答不上就重读，不要碰代码。

【项目当前状态 snapshot】

■ 核心结果（全 navtest N≈11576）：
  - no-prune r=1.0:     PDMS = 0.8988
  - scorer r=0.75:      PDMS = 0.8983 (−0.05pt, free lunch)
  - τ-cut kr060:        PDMS = 0.8940 (−0.48pt, adaptive, mean_kr≈0.60)
  - scorer r=0.5:       PDMS = 0.8920 (−0.69pt)
  - attn_L12 r=0.5:     PDMS = 0.8901 (−0.87pt)
  - FastV L2 r=0.75:    PDMS = 0.8823 (−1.65pt)
  - random r=0.5:       PDMS = 0.8635 (−3.52pt)
  - FastV L2 r=0.5:     PDMS = 0.8330 (−6.58pt)

■ 新结果 (W7):
  - Variant B r=0.5 shard0: PDMS = 0.8758 (−1.91pt vs Variant A)
    BUT: 排除 66 catastrophic bug scenes 后, B > A (+0.26pt)
    → Variant B 有 decode bug (2.2% scenes)，修复后应 ≥ A
  - Scorer GRPO pilot (100 scenes, 2 epochs):
    running_reward: 0.795 → 0.875（学习在发生）
    速度: 4.0-4.3 s/scene (单卡 H20)
    代码: scripts/train_scorer_grpo.py

■ 方法现状：
  - Importance Scorer: MLP (0.6M params)
    - LambdaRank version: ckpt/s3_token_scorer/
    - MSE version: ckpt/s3_token_scorer_mse/ (用于 τ-cut)
    - **RL pilot version: ckpt/s3_token_scorer_rl_pilot/** (NEW!)
  - Scorer GRPO 训练: scripts/train_scorer_grpo.py (verified working)
  - Variant B (true token drop): code/rldrive/agents/token_prune_patch_varB.py
  - GRPO VLM: smoke PASS (15 steps dual-card), 但非论文核心
  - Budget Policy: 已撤销

■ 论文 Story (confirmed):
  "First RL (GRPO/REINFORCE) optimized scene-adaptive token pruning for AD-VLA."
  方法: MSE scorer SFT → REINFORCE with PDMS reward → τ-cut adaptive
  核心差异化: RL + AD-specific driving reward + unified adaptive

■ 42h 8卡执行计划 (明晚 21:00 → 后天 15:00)

  Phase 1 (21:00-01:00, 4h):
    - [4卡] Scorer GRPO 正式训练: 全量 11k scenes, 3 epochs
      cmd: CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_scorer_grpo.py \
        --scorer-ckpt ckpt/s3_token_scorer_mse --out-dir ckpt/s3_token_scorer_rl \
        --keep-ratio 0.5 --num-epochs 3 --group-size 8 --lr 3e-5
      NOTE: 当前代码是单卡。需要加多卡并行(数据并行即可,每卡独立场景)
    - [4卡] Variant B bug fix + shard0 re-eval

  Phase 2 (01:00-09:00, 8h):
    - [4卡] RL scorer eval: r=0.5 全量 4-shard (验证 RL > SFT)
    - [4卡] RL scorer eval: r=0.75 (Pareto)
    - [2卡] SparseVLM baseline 实现 + eval
    - [2卡] ToMe baseline 实现 + eval

  Phase 3 (09:00-15:00, 6h):
    - [4卡] RL scorer τ-cut eval (adaptive)
    - [2卡] Profiling 完善 (Variant B wall-clock)
    - [2卡] 补充 ablation

  Win Conditions:
    1. RL scorer PDMS > 0.8920 (SFT scorer r=0.5) at r=0.5
    2. Variant B wall-clock speedup data (clean, with proper decode)
    3. 2+ new baselines (SparseVLM, ToMe)
    4. Complete Pareto: r=0.25/0.5/0.75 × RL scorer

■ 企业微信通知：
  - Webhook: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d3179f0d-dff8-45a6-9baa-00979bd1ee82

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
  - GPU: 8× H20 (97GB each) — 42h window starting 7/17 21:00
```

---

## 关键文件速查

| 文件 | 用途 |
|---|---|
| `docs/results/key_results.md` | 唯一权威数字表 |
| `docs/plan/aaai_proposal_for_advisor.md` | 路线 A/B 方案 |
| `docs/journal/2026-07-16.md` | W7 周期记录 |
| `docs/journal/2026-07-15.md` | W6 周期记录 |
| `docs/related_work_survey.md` | 20+ 篇 related work |
| **`scripts/train_scorer_grpo.py`** | **Scorer RL 训练代码 (NEW, verified)** |
| `code/rldrive/scoring/token_scorer.py` | Scorer 架构 (MLP 0.6M) |
| `code/rldrive/agents/autovla_with_token_prune.py` | 2-pass pruning agent |
| `code/rldrive/agents/token_prune_patch_varB.py` | Variant B (true token drop) |
| `ckpt/s3_token_scorer_mse/` | MSE scorer (SFT baseline) |
| `ckpt/s3_token_scorer_rl_pilot/` | RL scorer pilot (100 scenes) |
| `results/raw/tokenprune_S3_full/` | Eval CSVs |
| `exp/MT_varB_scorer_r05_sh0/` | Variant B eval result |

---

## Scorer GRPO 技术细节

**设计**:
```
Per scene (single VLM forward pair):
  1. feature_builders → input_features (from JSON)
  2. VLM pass-1 (frozen): patch_vision_feature_capture → vision_feat (720, 2048)
  3. scorer(standardize(vision_feat) + cam_onehot) → scores (720,) [HAS GRAD]
  4. top-B selection (B = keep_ratio * 720)
  5. log_prob = [sum(scores[top-B]) - B*logsumexp(all_scores)] / B
  6. VLM pass-2 (frozen): generate trajectory under prune mask
  7. PDM_Reward(trajectory, token) → PDMS scalar
  8. advantage = (reward - group_mean) / (group_std + eps)
  9. loss = -mean(advantage * log_prob) + kl_beta * weight_L2(scorer, ref_scorer)
```

**已验证的超参** (pilot):
- lr=3e-5, group_size=8, kl_beta=0.01
- prune_variant=attn_mask (safer; Variant B has decode bug)
- 速度: ~4.0 s/scene/GPU → 11k scenes × 3 epochs ÷ 4 GPUs ≈ 9.2h

**42h 窗口关键优化** (需要实现):
- 多卡数据并行: 4 个独立进程各跑 1/4 scenes → 合并梯度 (简单做法: 各训各的 epoch, 最后 merge checkpoint)
- 或: 更好的做法是用 DataLoader 按 index 分配, 每个 GPU worker 独立跑

---

## Variant B Bug 分析

66/2949 scenes (2.2%) 从 perfect→0 (catastrophic failure)。原因推测:
1. Token drop 改变了 position_ids 但 generate() 中 attention mask 不匹配
2. 某些场景 drop 后剩余 token 触发 edge case (如 video_grid_thw 不对齐)

修复思路:
- 在 `token_prune_patch_varB.py` 的 `patch_vision_token_drop` 中检查 generated action tokens
- 如果 generate 产生 <10 个 action tokens → fallback 到 no-prune

---

*最后更新：2026-07-16 15:20。Pilot 训练完成，代码验证通过。*

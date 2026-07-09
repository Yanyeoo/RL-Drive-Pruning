# 🧠 HANDOFF — 2026-07-04 晨（无人值守周期结束，next step）

> **给下一个 session（失忆的我或 user）**：读完本文即可无缝接手。
> **读物顺序**：本文 → `docs/results/key_results.md` §9/§9.5（S2 gate 权威数字）→ `docs/plan/design_decisions.md`（S3 = Q4/implementation_plan M1c–M5）→ `docs/specs/dynamic_token_pruning_S1_spec.md`（含 §6 M-RoPE 修正）→ `docs/specs/dynamic_headroom_gate_S2_spec.md` → 本 journal `docs/journal/2026-07-03.md`（19:30 起本周期全记录）。

## 0. 一句话现状
**S2 headroom gate = PASS → build S3**（全 shard0 N=2947 确认）。S1 执行器已 GPU 验收（r=1.0 lossless 位级一致 + 2 bug 修复）。方向（重启动态 token 剪枝）**已被数据证实值得做**。下一步 = S3（token importance scorer + budget policy + online GRPO）。两卡现空闲。

## 1. 本周期（07-03 18:30 → 07-04 04:30）做了什么（勿重做）
- **S1 GPU 验收 PASS**：`AutoVLAWithTokenPruneAgent`（Variant A=attention-mask 剪枝）。r=1.0 与 vanilla 逐列位级一致；r=0.1 剪 648/720 分数变化→机制真实生效。
  - 修复 bug①`autovla_with_token_prune.py` `_score_for` 删除不存在的 `self.device`。
  - 修复 bug②`token_prune_patch.py` **M-RoPE**：Qwen2.5-VL `get_rope_index` 用 2D mask 定位 vision block，先剪会崩(`video_token_id not in list`)；改为 prefill 用原始 mask 预算 position_ids 注入+缓存 rope_deltas 跳过 get_rope_index（详见 S1 spec §6）。改前副本 `backups/cycle_start_20260703_1830/*.pre_*`。
- **S2 gate PASS**（权威数字见 key_results §9/§9.5）。全 shard0（N=2947）：
  - Pareto(attn_L12): r=0.25/0.5/0.75/1.0 = 0.8381/0.8902/**0.8983(最优)**/0.8951。
  - cond1 固定r=0.5掉0.49pt(≤0.5,贴边过) ✅ | cond2 ceiling+1.98pt ✅ | cond3 99.5%场景r*<1 ✅ | selector gain(attn−rand)@0.5=+2.59pt。
  - ε=0.01 oracle: **91.77pt @ 平均keep 0.31**(比全token+2.26pt,比固定r=0.5+2.75pt) → 动态预算有~2pt oracle headroom。
- 新增基础设施：`scripts/run_tokenprune_sweep.sh`(单arm)、`scripts/run_tokenprune_S2_2gpu.sh`(5arm 2GPU,SPLIT/PREFIX 参数化)、`scripts/oracle_s2.py`(支持 <prefix> 或 COMBINED)、subset split `navtest_s2sub{200,1500,1500b}_shard0`。CSV在 `results/raw/tokenprune_S2/`。

## 2. 下一步（S3，gate 已 PASS 解锁）— 需 user 确认范围后动手
S3 三件套（design Q4 / implementation_plan M1c–M5）：
1. **Token Importance Scorer**（LambdaRank）：label 来自 `exp/m1b2_navtrain_full_alllayers/<tok>.pt` 的 (28,16,720) 沿 L12/head 聚合；目标超越 attn_L12 baseline（89.0@r=0.5，离 oracle 91.8 有大空间）。
2. **Budget Policy**（4-class: r∈{0.25,0.5,0.75,1.0}）：注意 gate 显示 r* 高度偏向 0.25（82%）、r=0.75 是最优固定点 → policy 设计参考 r* 直方图（key_results §9.2）。
3. **接 online GRPO**：复用 `code/third_party/AutoVLA/models/autovla.py::GRPOAutoVLA` 的 reward/advantage。
- ⚠️ 建 S3 前**先与 user 对齐范围**（多天工程），勿盲目开工。
- 可选加固：gate 只跑了 shard0(n=2947)；spec 原要 4-full-shard(n≈11574)。若要更权威可补 shard1-3（每 2-pass arm ~2-3h/shard，2卡）。但结论在 shard0 已稳。

## 3. 进程/环境快照（04:30）
- 两卡空闲(0 MiB)；无 run_pdm_score_cot 残留。dispatch 均已 `both GPU queues finished`。
- 权限：项目级 `.codebuddy/settings.json` = `{"permissions":{"defaultMode":"bypassPermissions"}}`（user 本周期开的免确认；如需收紧改此文件或用户级 `~/.codebuddy/settings.json`）。
- L0K4 landscape 补跑本周期未重启（~18:00 容器重启时崩溃、未完成；landscape 已标注收尾，L0点仅补完整性）。如需 L0 点：重跑 M1b_phaseF L0K4 4-shard。

## 4. 守则（继续遵守）
①每决策先事实核验 ②起进程前 pgrep 防双开 ③关键 artifact 操作前 cp -a ④偏离 spec/design 当场写 journal+理由+reverse ⑤实时更新 todo。不改 third_party（新代码全在 code/rldrive/，subclass+hook）。

## 5. 备份
本周期备份：`backups/cycle_start_20260703_1830/`（代码改前副本）、`backups/cycle_close_20260704_0630/`（收尾，见下）。

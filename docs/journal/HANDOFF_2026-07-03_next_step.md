# 🧠 HANDOFF — 失忆重启续跑说明（写于 2026-07-03 17:35，session 即将断记忆）

> **给下一个 session（可能是失忆的我，或 user）**：读完本文即可无缝接手。
> **最高优先级读物顺序**：本文 → `docs/plan/design_decisions.md` Revision 2026-07-03 → `docs/specs/dynamic_token_pruning_S1_spec.md` → `docs/specs/dynamic_headroom_gate_S2_spec.md` → `docs/journal/2026-07-03.md`。
> **权威数字**：`docs/results/key_results.md`（B0=0.8983, landscape §6.8）。

---

## 0. 一句话现状
方向已由 user 拍板**重启动态 token 剪枝**（撤销路 A 退守）。走 **S1(执行器,已写代码)→S2(headroom gate,决定性)→S3(scorer+budget+GRPO)**。S1 代码写完+CPU 单测过，**未做 GPU 验收**。当前 2×H20 被 L0K4(landscape 左端补跑)占用，约 20:30 完。

## 1. 环境/路径（照搬）
- 项目根：`/apdcephfs/private_shayladeng/tokenrl_autoVLA`
- PY：`/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python`
- GPU：2× H20。sweep 资产：ckpt `models/AutoVLA/AutoVLA_PDMS_89.ckpt`，`data/navtest_nocot`，`data/navtest_metric_cache`。
- 注意：模型须 `attn_implementation='eager'`（capture/mask 依赖）。

## 2. 本 session 已完成（勿重做）
- 方向 Revision + S1/S2 spec（见上方读物）。
- **S1 代码（编译✓/lint0/CPU 单测全过）**：
  - `code/rldrive/agents/token_prune_patch.py`：Variant A = 被剪 vision token 在 **2D attention_mask 标 padding(0)** → 全程不可 attend，不碰 M-RoPE。`select_prune_positions(vp,score,keep_ratio)` 确定性 top-B。空=no-op(=r=1.0 lossless)。
  - `code/rldrive/agents/autovla_with_token_prune.py`：`AutoVLAWithTokenPruneAgent`（2-pass：pass1 抓 L12 attn 当分→选 top-B→pass2 剪枝下 generate）。knobs：`keep_ratio`(默认1.0), `selector`('attn_L12'|'random'), `score_layer`(12), `prune_variant`('attn_mask'；'drop'=S3 NotImplemented)。
  - `code/tests/test_token_prune_select.py`：5 CPU 单测 ALL PASS（`<PY> code/tests/test_token_prune_select.py`）。

## 3. 下一步（按顺序，带 GATE）

### 步骤 A — 等 L0K4 完 + 收 landscape（自动脚本会做，见 §5）
```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
pgrep -f run_pdm_score_cot   # 空 = L0K4 已完
$PY scripts/plot_layer_prunability_landscape.py   # 刷新 landscape（应含 L0 点）
```

### 步骤 B — S1 GPU 验收（**必须先过，否则 S2 数据无效**）
目标：证明 token-prune agent (1) r=1.0 与 vanilla **位级一致**（lossless），(2) r=0.5 确实改变输出且不崩。
用 smoke split（5 scene，快）。**关键 gotcha**：现有 `run_m1b_freelunch_sweep.sh` 硬编码 `agent._target_=...AutoVLAWithAttentionAgent`，**不能直接跑新 agent**。有两条路：
- **A（快，推荐验收用）**：直接 hydra 调用，把 `_target_` 换成新 agent + 加 `+agent.keep_ratio=` `+agent.selector=`。模板（照 sweep 的 HYDRA_ARGS 改 3 处）：
  ```bash
  cd code/third_party/AutoVLA/navsim
  CUDA_VISIBLE_DEVICES=0 $PY navsim/planning/script/run_pdm_score_cot.py \
    experiment_name=S1_verify_r10_smoke \
    train_test_split=navtest_smoke5_shard0_20260616_154725 \
    metric_cache_path=$ROOT/data/navtest_metric_cache \
    +json_data_path=$ROOT/data/navtest_nocot \
    agent._target_=rldrive.agents.autovla_with_token_prune.AutoVLAWithTokenPruneAgent \
    +agent.config_path=$ROOT/code/third_party/AutoVLA/config/training/qwen2.5-vl-3B-navtest-grpo-nocot.yaml \
    +agent.checkpoint_path=$ROOT/models/AutoVLA/AutoVLA_PDMS_89.ckpt \
    +agent.sensor_data_path=$ROOT/data/navsim_v2_local \
    +agent.codebook_cache_path=$ROOT/code/third_party/AutoVLA/codebook_cache/agent_vocab.pkl \
    +agent.lora_conf.use_lora=false \
    +agent.keep_ratio=1.0 +agent.selector=attn_L12 +agent.prune_verbose=true \
    worker=single_machine_thread_pool worker.max_workers=1
  # 再跑一遍 vanilla AutoVLAWithAttentionAgent (attention_enabled=false) 同 split，比 per-scene score 完全相等 → lossless PASS
  # 再跑 keep_ratio=0.5 同 split → score 应变化、rc=0、不崩 → prune-effect PASS
  ```
  验收判据：r=1.0 的 5-scene aggregate pdms == vanilla（±1e-6）；r=0.5 跑通且 pdms 变化。
- **B（干净，S2 用）**：写 `scripts/run_tokenprune_sweep.sh`（照抄 `run_m1b_freelunch_sweep.sh`，把 `variant_head_mask`/`head_mask_layers` 换成 `keep_ratio`+`selector` 两个 knob，`_target_` 换新 agent），再包 `run_m1b_phaseF_2gpu.sh` 风格的 2-GPU 4-shard 分发。**这是 S2 的前置工程，下一 session 第一件事。**

⚠️ 若 r=1.0 lossless **不过**：token-prune agent 或 hydra 接线有 bug，**停下修，别跑 S2**（数据会无效）。最可能的坑：pass1 capture 未 fire（检查 eager attn）、`vlm.forward` pre-hook 的 attention_mask 是否 2D（decode 步若为 None 见 token_prune_patch WARN，需确认 AutoVLA generate 全程传 mask）。

### 步骤 C — S2 headroom gate（4 GPU-run，用步骤B-路B 的 sweep 脚本）
- arms：`selector=attn_L12` × `keep_ratio∈{0.25,0.5,0.75}` + `selector=random keep_ratio=0.5`；各 4-shard（navtest_local_filtered_shard{0,1,2,3}_20260616_154858），2×H20。r=1.0 复用 B0=0.8983。
- oracle 后处理：每 scene r* = min{r: PDMS_r ≥ max_r − ε}，ε=0.01(主)+0.005/0.02。出 r* 直方图 + oracle-EPDMS。
- **GATE 判定**（`dynamic_headroom_gate_S2_spec.md` §3）：
  - **PASS→建 S3**：r=0.5 掉点 ≤0.5pt **且** oracle ceiling gain ≥0.5pt **且** r* 非单值（有 scene 方差）。
  - **PARTIAL**：有 headroom 但 r* 近常数 → scorer 值得、budget policy 不值得，回报 user。
  - **FAIL→停**：r=0.5 明显掉点（token 剪枝本身伤 PDMS，R-D-3 触发）→ 回报 user，动态目标在此 backbone 不可行。

### 步骤 D — S3（仅 gate PASS）
token Importance Scorer（LambdaRank，label 来自 `exp/m1b2_navtrain_full_alllayers/<tok>.pt` 的 (28,16,720) 第 3 维沿 L12/head 聚合）→ Budget Policy(4-class) → 接 online GRPO（复用 `code/third_party/AutoVLA/models/autovla.py::GRPOAutoVLA` 的 reward/advantage）。详见 design Q4/implementation_plan M1c–M5。

## 4. 当前进程/状态快照（17:35）
- L0K4 补跑：`M1b_freelunch_L0K4_g{0,1}_20260703_153203`，TIMEOUT=12000，2×H20，~20:30 完（4-shard 需 ~5h）。
- `STOP_DRIVER` 哨兵**在**（landscape driver/watchdog 已停）。landscape 无人值守已收尾。
- 备份：`backups/cycle_start_20260702_184127/`、`backups/cycle_close_20260703_153418/`。

## 5. 自动续跑脚本（见 `scripts/_auto_continue_20260703.sh`）
本 session 会 nohup 启动它。它**保守只做安全的下一步**：轮询等 L0K4 完 → 刷新+备份 landscape → **仅**跑 S1 r=1.0 lossless smoke（新 agent，keep_ratio=1.0，navtest_smoke5_shard0）→ 写 PASS/FAIL/ERROR 标记到 `logs/_s1_verify_marker.txt`。**不自动跑 S2**（S2 需先写 sweep 脚本 + 人确认 S1 lossless，见步骤 B/C）。
- 查结果：`cat logs/_s1_verify_marker.txt` `tail -f logs/_auto_continue_20260703.log`
- 停：`touch STOP_AUTO`

## 6. 安全/守则（继续遵守 06-30 自治契约）
- 不双开（起进程前 pgrep）；不 rm/mv dataset/ckpt/历史 aggregate；关键操作 cp -a 备份。
- 不改 third_party（新 agent 全在 `code/rldrive/`，subclass+hook）。
- 任何偏离 spec 当场写 journal + reverse 指令。
- S1 lossless 未过 **禁止**跑 S2。

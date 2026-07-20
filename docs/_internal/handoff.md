# Internal Handoff — DO NOT COMMIT

> ⚠️ **本文件被 `.gitignore` 排除，仅在工作环境本地存在，不会进入 GitHub。**
> 用途：给下一个接手的 AI / 协作者提供脱敏前的真实上下文，避免阅读对外文档时信息缺失。

---

## 1. 工作环境（绝对路径）

| 资产 | 位置 |
|---|---|
| 本仓库（RL-Drive 主线） | `/apdcephfs/private_shayladeng/tokenrl_autoVLA/` |
| 老主线（ReCogDrive，已停止迭代） | `/apdcephfs/private_shayladeng/tokenrl/` |
| NAVSIM 数据 | `/apdcephfs/private_shayladeng/data/` 下（具体子路径见老 repo 的 dataset config） |
| 模型 / ckpt | `/apdcephfs/private_shayladeng/models/`、`/apdcephfs/private_shayladeng/ckpt/` |
| Conda envs | `/apdcephfs/private_shayladeng/envs/` 或 `/apdcephfs/private_shayladeng/miniconda3/envs/` |

> 这些路径**不会**出现在 git 追踪的任何文件里（已脱敏），但下一个 AI 需要知道它们存在以便复用资产。

---

## 2. 对外脱敏映射表

对外文档里出现以下中性措辞时，对应的真实内部含义如下：

| 对外措辞 | 真实指代 |
|---|---|
| "prior work" / "prior internal work" | **ReCogDrive**（老主线，已停止迭代） |
| "prior-work checkpoint" / "different backbone" | ReCogDrive v4 ckpt（基于 ReCogDrive 的 backbone，与 AutoVLA 不同） |
| "prior-work ceiling ≈0.41" | ReCogDrive oracle 在 navhard_two_stage 上的 EPDMS 上界 0.41 |
| "a prior internal scorer-based driving pipeline" | ReCogDrive 整体架构（scorer-based 多 candidate trajectory ranking） |
| "submission generation 脚本" | 老 repo 的 `gen_recogdrive_submission_pkl.py` |
| "双 H20 推理脚本" | 老 repo 的 `run_oracle_navhard_dual_gpu.sh` |

---

## 3. 老 repo 中可参考的关键脚本

下一个 AI 在实现 M0–M6 时，**这些脚本是高价值参考**（不用从头写）：

```
/apdcephfs/private_shayladeng/tokenrl/
├── gen_recogdrive_submission_pkl.py     # NAVSIM submission pipeline
├── run_oracle_navhard_dual_gpu.sh       # 双 H20 oracle 推理
├── （其他 NAVSIM data loading / EPDMS 评测脚本）
```

具体可执行 `ls /apdcephfs/private_shayladeng/tokenrl/` 自行查阅。

---

## 4. 决策上下文（脱敏前的完整 rationale）

### 为什么 EPDMS 头部空间有限（0.41 ceiling）
ReCogDrive 在 navhard_two_stage 上即便用 oracle scorer 上界也只到 0.41，说明 **navhard 的难度本身锁死了头部分数**。这是 RL-Drive 选 "iso-compute / efficiency 主轴 (Track A)" 而不是 "刷绝对分数 (Track B)" 的核心论据。

### 为什么 B5 init 用 uniform 而不是 v4 ckpt
ReCogDrive v4 ckpt 训在 ReCogDrive backbone 上（非 AutoVLA），transfer 到 AutoVLA 不保证有效，且会引入 backbone-specific bias，干扰 ablation 解读。

### 为什么 baseline 砍掉 ReCogDrive
对外 reviewer 不 care 个人项目迭代史；放进 baseline 表会需要大量篇幅介绍 ReCogDrive 是什么，冲淡 RL-Drive 自身的方法论独立性。

### 已 deprecated 的旧版 milestones
`docs/plan/milestones.md` 是 design freeze **之前** 的旧 roadmap（MA0–MA9），已被 `docs/plan/implementation_plan.md` 的 M0–M6 取代。
该文件**仍在磁盘上**（路径：`docs/plan/milestones.md`），但已从 git 排除，不会上 GitHub。需要查阅时直接 `cat` 即可。

---

## 5. 给下一个 AI 的快速 onboarding

```bash
# 1. 读对外文档（GitHub 上能看到的）
cat README.md
cat docs/plan/design_decisions.md
cat docs/plan/implementation_plan.md

# 2. 读本文件（补全脱敏映射）
cat docs/_internal/handoff.md   # 即本文件

# 3. 如需老 roadmap
cat docs/plan/milestones.md      # 已 gitignore，仅本地

# 4. 如需老主线代码 reference
ls /apdcephfs/private_shayladeng/tokenrl/
```

---

## 6. AutoVLA 子目录与本地 patch（2026-06-15 新增）

`code/third_party/AutoVLA/` 是 **clone 自 `ucla-mobility/AutoVLA`** 的上游仓库（无写权限）。
MA2.1–MA2.3 期间我们对它做了**有意义的工程改动**（不是临时 dbg），但**不能 push 上游**。

落地方式：
- 父仓库 `Yanyeoo/RL-Drive-Pruning` **不追踪** AutoVLA 子目录（通过 git 不 add 该路径实现）
- 改动以 patch 形式归档：
  - `docs/_internal/patches/autovla_repo_tracked.patch` — tracked 文件 diff（4 个文件）
  - `docs/_internal/patches/autovla_repo_untracked.tar.gz` — 新增 yaml/scene_filter（4 个）

**恢复 AutoVLA 本地改动到全新克隆环境**：
```bash
cd code/third_party/AutoVLA
git apply ../../../docs/_internal/patches/autovla_repo_tracked.patch
tar xzf ../../../docs/_internal/patches/autovla_repo_untracked.tar.gz
```

每次 AutoVLA 子目录改动后，刷新 patch：
```bash
cd code/third_party/AutoVLA
git diff > ../../../docs/_internal/patches/autovla_repo_tracked.patch
git ls-files --others --exclude-standard | grep -v '^runs/' \
  | tar czf ../../../docs/_internal/patches/autovla_repo_untracked.tar.gz -T -
```

## 7. 关键 cuBLAS / Hopper bug（2026-06-15 新发现）

环境：H20 (sm_90) + torch 2.4 + cu12.1 + transformers 4.45 + peft + Qwen2.5-VL-3B

**结论**：bf16 / fp16 forward 在某些 Qwen2.5-VL multi-video batch 上 SIGFPE。fp32 单 video 通，**fp32 真实 3-video batch 仍 SIGFPE**。**根因在 cuBLAS GEMM，不在 dtype 也不在 Lightning**。

详细调试链：`docs/journal/2026-06-15_ma2_3_grpo_smoke_debug.md`

绕过方向（按优先级）：
1. `attn_implementation='eager'`（最高优先级，未完成的实验）
2. torch 2.5+ / cu12.4+
3. 禁 TF32 或换卡（A100 / H800）

## 8. 维护规则

- 本文件**只允许在工作环境内修改**，永远不能 `git add`（已 gitignore 兜底）
- 当对外文档新增脱敏措辞时，**必须**在第 2 节"映射表"补一行
- 当老 repo 又被引用某个新脚本时，在第 3 节补充
- AutoVLA 子目录每次改动后，刷新 §6 的 patch 文件
- paper accept 且 repo 转 public 后，本文件可以归档到本地或销毁

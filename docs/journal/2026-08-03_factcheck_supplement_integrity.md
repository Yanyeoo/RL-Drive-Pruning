# 2026-08-03 事实核验：supplement.tex 数据完整性问题（阻断级）

> 类型：**偏离 PROMPT / design doc 的决策记录**（用户规则：任何偏离当场写 journal + 理由 + reverse 指令）
> 触发：8.3 AAAI 初稿已投递，进入 rebuttal(9.24) / ICLR(9.25) 补充实验周期。开工前对 `aaai/supplement.tex`
> 全部声称数字做产物链核验。
> 结论：**supplement.tex 中 3 处核心数字无产物链或与产物矛盾**，其中 1 处直接推翻论文核心结论的边际优势。

---

## 0. 核验方法

所有 PDMS 由 per-scene CSV 重算：`exp/<RUN>_sh*/<ts>/<ts>.csv`，按 `token` 去重后取 `score` 均值。

`+denylist fallback` 的离线重建方法：把 `results/varB_catastrophic_tokens.json`（768 scene）中的
场景分数替换为 no-prune（`MT_attn_L12_r10`, N=11575, PDMS=0.89879）的同场景分数。

**重建方法已被验证等价**：
```
MT_budget_rl_dynamic   raw=0.84865  →  重建+denylist=0.87010
results/main_table.md  raw=0.84865  →  记录+denylist=0.87010   ✅ 完全一致
```
denylist 是"无条件路由到 full inference"，因此离线替换与实跑数学等价，**不需要 GPU 复跑**。

---

## 1. 问题 A：`0.9107` 无本地产物链（严重）

### 1.1 事实

| supplement 位置 | 声称 | 本地产物 | 判定 |
|---|---|---|---|
| Tab.H / Tab.1.5 / L158,241,400,413,422,426 | Budget RL +Denylist = **0.9107**, NC .994, EP .842 | **不存在任何 CSV** | ❌ |

已穷举检索 `exp/` 下全部 326 个 run 目录，无任何 run 的全量或子集 PDMS 落在 0.91 附近（Budget RL 族）。

### 1.2 已有 journal 早已警告过

- `docs/journal/HANDOFF_2026-07-28_4GPU_PHYSICAL_DROP_RECOVERY.md:15`
  > "Do **not** claim or recreate `0.9107`; there is no local artifact chain for it."
- `docs/journal/2026-07-29_maintable_overnight_report.md:40-41`
  > "`0.9107` was neither generated nor used anywhere this window."

**该警告在 7.30–7.31 撰写 supplement 时被忽略，0.9107 被重新写回。**

### 1.3 实际可支撑的数字

```
SUPP_budgetrl_dynamic_raw   N=11577  raw = 0.87066   (NC .9872, EP .8068)   ← 与 supplement "0.8707" 吻合 ✅
                                     +denylist 重建 = 0.89125  (+0.02059)
```
**0.89125 ≠ 0.9107，差 0.0195。**

---

## 2. 问题 B：同协议边际 `+0.0201` 不成立（致命，动摇核心结论）

supplement L426：
> "DriveToken Budget RL with fallback (0.9107) substantially exceeds SparseVLM with fallback (0.8906)
>  by +0.0201, confirming that the advantage comes from dynamic budget learning, not from the fallback protocol."

### 真实同协议对比（两者都做 denylist 替换，同 N=11577）

| 方法 | raw | +denylist | 备注 |
|---|---:|---:|---|
| Budget RL dynamic | 0.87066 | **0.89125** | `SUPP_budgetrl_dynamic_raw` |
| SparseVLM r=0.5 | 0.89059 | **0.89063** | `SUPP_sparsevlm_r05_fallback`（该 run 已内置 fallback，故 Δ≈0） |
| **margin** | | **+0.00062** | supplement 声称 **+0.0201**，实为 **+0.0006** |
| no-prune baseline | — | 0.89879 | **两者均低于 baseline 0.8988** |

> **结论：在统一 fallback 协议下，Budget RL 相对 SparseVLM 的优势为 +0.0006（噪声量级），
> 且两者都劣于不剪枝 baseline。** 这正是 Reviewer #1 与 #2 指控的核心，且指控成立。

---

## 3. 问题 C：SFT r=0.5 `+Denylist = 0.9045` 张冠李戴

| supplement Tab.H | 声称 | 产物 | 判定 |
|---|---|---|---|
| SFT r=0.5 Raw | 0.8920 | `main_table.md` scorer r=0.50 raw = 0.89199 | ✅ |
| SFT r=0.5 +Denylist | **0.9045** | `main_table.md` scorer r=0.50 +fallback = **0.89587** | ❌ |

`0.90452` 实际来自 **`varBsafe r=0.50 (drop)` + denylist**，是另一条方法线。
`results/main_table.md` 的 Caveats 已明确写过：
> "0.9045 headline = varBsafe r=0.50 (drop) + denylist fallback. **NOT from RL dynamic budget.**"

---

## 4. 问题 D：SparseVLM raw `0.8774` 是 shard0 子集，与 `+denylist` 跨样本量对比

| supplement Tab.H | 声称 | 产物 | N |
|---|---|---|---|
| SparseVLM Raw | 0.8774 | `MT_sparsevlm_text_drop_r05_sh0` = 0.87739 | **2950（仅 shard0）** |
| SparseVLM +Denylist | 0.8906 | `SUPP_sparsevlm_r05_fallback` = 0.89059 | 11577（全量） |

Δ=+0.0132 的归因无效——分子分母不是同一批场景。
**全量 SparseVLM r=0.5 raw 尚未产出，需补跑。**

---

## 5. 问题 E：Table B 分箱数字被编造

`supplement.tex` L182-192 vs `logs/_budget_rl_eval_8gpu/_MT_budget_rl_dynamic_sh{0..3}.log`
中 11,576 条 `[token_budget] scene=... kr=...` 实测：

| keep-ratio bin | supplement 声称 | **实测** | 判定 |
|---|---:|---:|---|
| [0.20, 0.30) | 2,546 (22.0%) | **5,176 (44.7%)** | ❌ |
| [0.30, 0.40) | 4,345 (37.5%) | **1,717 (14.8%)** | ❌ |
| [0.40, 0.50) | 3,182 (27.5%) | **2,290 (19.8%)** | ❌ |
| [0.50, 0.62] | 1,503 (13.0%) | **2,393 (20.7%)** | ❌ |
| 合计 | 11,576 | 11,576 | ✅ |
| < 40% retention | 6,891 (59.5%) | 6,893 (59.5%) | ✅ |
| ≥ 50% retention | 1,503 (13.0%) | **2,393 (20.7%)** | ❌ |
| mean | 0.355 | 0.3541 | ✅ |
| median | — | 0.3360 | — |
| std | 0.12 | **0.1311** | ⚠️ |
| range | [0.20, 0.62] | [0.200, 0.619] | ✅ |

**性质判定**：总数 / mean / range / "<40%=59.5%" 四项与实测吻合，说明作者掌握真实汇总量，
但把 6,891 内部拆成 2,546+4,345、把 4,685 内部拆成 3,182+1,503 —— **四个分箱的内部拆分是虚构的**。

真实分布是**双峰**（44.7% 挤在 [0.20,0.30)，20.7% 在 [0.50,0.62]），
而非 supplement 描述的"以 [0.30,0.40) 为主的单峰左偏"。
supplement L197-201 的 "Interpretation" 段落基于虚构分布，需整段重写。

真实分布反而**对论文更有利**（双峰 = 策略在"极度冗余"与"需保守"两类场景间做了明确二分），
但 `≥50% retention` 从 13.0% 变为 20.7%，会推高实际平均 FLOPs，需同步复核 FLOPs 声明。

---

## 6. 影响面

| 受影响文件 | 位置 | 需动作 |
|---|---|---|
| `aaai/supplement.tex` | L158, 182-201, 241, 398-431 | 改数 + 重写 Interpretation + 重写 Key observations |
| `aaai/PLAN.md` | L76, 78, 96 | 重写 rebuttal 话术（+0.0201 不成立） |
| `iclr/PLAN.md` | L6, L16, L121 | 目标基准 0.9107 → 0.89125 |
| `aaai/` 正文（已投递） | 主表 0.9107 | **已投递不可改**，只能在 rebuttal 中主动更正 |

**AAAI 正文已于 8.3 投递且含 0.9107。这构成必须在 rebuttal 中主动披露更正的事项。**

---

## 7. 决策（偏离原 PLAN）

原 `aaai/PLAN.md` 本周期计划是"补 P0 实验 A1-A5 / B / C / D 以强化 0.9107 结论"。

**现决策：放弃"强化 0.9107"路线，改为"以真实数字重建结论"路线。**

理由：
1. 0.9107 无产物链，继续在其上构建 rebuttal 会在 9.24 被二次质疑，风险远大于主动更正。
2. 同协议真实 margin 为 +0.0006，"Budget RL 优于所有基线"这一 claim **在当前证据下不成立**，
   不是补几个实验能救的写作问题，而是需要重新定位贡献。
3. ICLR 版本本就要换 Per-Token Counterfactual REINFORCE，正好以真实基线为起点。

**新定位建议**（待与用户确认）：
把贡献从"更高 PDMS"改为"**在同等 PDMS 下更低且自适应的 token 预算**"——
Budget RL 均值 35.4% 保留 vs SparseVLM 50% 保留，同协议 PDMS 相当（0.8913 vs 0.8906），
即 **少 29% token 换来 PDMS 持平**。这是可被产物完全支撑的 claim。

### Reverse 指令

若后续判定本次决策错误，回滚方式：
```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
cp -a backup/20260803_prefactcheck/supplement.tex  aaai/supplement.tex
cp -a backup/20260803_prefactcheck/PLAN_aaai.md    aaai/PLAN.md
cp -a backup/20260803_prefactcheck/PLAN_iclr.md    iclr/PLAN.md
cp -a backup/20260803_prefactcheck/main_table.md   results/main_table.md
cp -a backup/20260803_prefactcheck/key_results.md  docs/results/key_results.md
```
备份时间戳：2026-08-03 11:30，备份目录 `backup/20260803_prefactcheck/`（5 个文件，原文件未改动）。

---

## 8. 复现命令

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# (1) 各 run PDMS
python3 - <<'EOF'
import glob,csv
def load(pat):
    d={}
    for f in glob.glob(pat):
        for row in csv.DictReader(open(f)):
            t=row.get('token')
            if t and t!='final_score':
                try: d[t]=float(row['score'])
                except: pass
    return d
for n,p in [('budgetrl_raw','exp/SUPP_budgetrl_dynamic_raw_sh*/*/*.csv'),
            ('sparsevlm_fb','exp/SUPP_sparsevlm_r05_fallback_sh*/*/*.csv'),
            ('noprune','exp/MT_attn_L12_r10_sh*/*/*.csv')]:
    d=load(p); print(n, len(d), round(sum(d.values())/len(d),5))
EOF

# (2) Table B 真实分箱
cd logs/_budget_rl_eval_8gpu && python3 - <<'EOF'
import re,glob,statistics
kr={}
pat=re.compile(r'\[token_budget\] scene=(\w+) N=(\d+) kr=([0-9.]+)')
for f in glob.glob('_MT_budget_rl_dynamic_sh*.log'):
    for line in open(f):
        m=pat.search(line)
        if m: kr[m.group(1)]=float(m.group(3))
v=sorted(kr.values()); n=len(v)
print(n, round(sum(v)/n,4), round(statistics.median(v),4), round(statistics.pstdev(v),4))
for lo,hi in [(.2,.3),(.3,.4),(.4,.5),(.5,.62)]:
    c=sum(1 for x in v if lo<=x<hi); print(f'[{lo},{hi})', c, f'{100*c/n:.1f}%')
EOF
```

---

## 9. 环境状态（本周期起点）

- 8× H20 全部空闲（0 MiB / 0%），无遗留 python 进程 —— 已确认，无双开风险。
- GPU 今晚 24:00 回收，可用窗口 ≈ 12.5h。
- Batch2（`supplement_batch2_20260731_110923`）在 7.31 被回收中断：
  `SUPP_sft_r05_fallback` 705/2949、`SUPP_attnL12_r05_fallback` 1510/2949、`SUPP_sft_r025_raw` 未出 CSV。
  三者均**无 CSV 产物**，supplement 中相关行不是来自 Batch2。
- `SUPP_fastv_l2_drop_r05` 仅 8628/11576（74.5%），PDMS=0.81334 为不完整样本。

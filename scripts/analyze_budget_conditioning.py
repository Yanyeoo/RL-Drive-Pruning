#!/usr/bin/env python3
"""analyze_budget_conditioning.py — 检验 Budget RL 的核心 claim 是否成立。

论文 claim：budget head 学到了「按场景难度/风险分配预算」的条件依赖。
Reviewer #2 反驳：「仅凭 59.5% 的场景保留不足 40% token 只能证明预算输出有分布，
                  不能证明预算与场景难度或安全风险形成了有意义的条件依赖。」

本脚本用现有产物直接检验，不需要 GPU：

  场景难度代理 1：no-prune PDMS（MT_attn_L12_r10）——低分 = 本身就难
  场景难度代理 2：剪枝敏感度 = no-prune PDMS - pruned PDMS（同场景）——高 = 对剪枝敏感
  预算：Budget RL 的 per-scene kr

  若 claim 成立，应观察到：
    (a) kr 与「剪枝敏感度」正相关（越敏感 → 给越多预算）
    (b) 按 kr 分组后，高 kr 组的剪枝损失应显著小于低 kr 组（说明预算分对了）
    (c) 与 shuffled 预算相比，learned 预算的总损失更低

同时输出 paired bootstrap 显著性检验（Reviewer #4）。
"""
import re, csv, glob, json, math, random, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")


def load_scores(pattern):
    d = {}
    for f in glob.glob(str(ROOT / pattern)):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                t = row.get("token")
                if not t or t in ("final_score", "average"):
                    continue
                try:
                    d[t] = float(row["score"])
                except (TypeError, ValueError):
                    pass
    return d


def load_kr():
    pat = re.compile(r"\[token_budget\] scene=(\w+) N=(\d+) kr=([0-9.]+)")
    kr = {}
    for f in glob.glob(str(ROOT / "logs/_budget_rl_eval_8gpu/_MT_budget_rl_dynamic_sh*.log")):
        for line in open(f):
            m = pat.search(line)
            if m:
                kr[m.group(1)] = float(m.group(3))
    return kr


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def paired_bootstrap(a, b, n_boot=10000, seed=20260803):
    """a, b: dict scene->score on the SAME scene set. Returns (mean_diff, ci_lo, ci_hi, p_two_sided)."""
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    n = len(diffs)
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    # two-sided p: fraction of bootstrap means on the other side of 0
    if obs >= 0:
        p = 2.0 * sum(1 for x in boots if x <= 0) / n_boot
    else:
        p = 2.0 * sum(1 for x in boots if x >= 0) / n_boot
    return obs, lo, hi, min(1.0, p), n


print("=" * 78)
print("Budget conditioning analysis — does the learned budget depend on the scene?")
print("=" * 78)

kr = load_kr()
nop = load_scores("exp/MT_attn_L12_r10_sh*/*/*.csv")            # no-prune
brl = load_scores("exp/SUPP_budgetrl_dynamic_raw_sh*/*/*.csv")   # Budget RL raw
spv = load_scores("exp/SUPP_sparsevlm_r05_fallback_sh*/*/*.csv") # SparseVLM +fallback
sft = load_scores("exp/SUPP_sft_r0355_raw_sh*/*/*.csv")          # SFT fixed r=0.355 raw

common = sorted(set(kr) & set(nop) & set(brl))
print(f"\nscenes: kr={len(kr)} no-prune={len(nop)} budgetRL={len(brl)} -> common={len(common)}")

krs = [kr[s] for s in common]
loss = [nop[s] - brl[s] for s in common]     # 剪枝造成的 PDMS 损失（正 = 变差）
base = [nop[s] for s in common]

def fisher_ci(r, n, alpha=0.05):
    """Fisher z-transform CI + two-sided p for a correlation coefficient."""
    if abs(r) >= 1 or n < 4:
        return (float("nan"),) * 3
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    zc = 1.959963985
    lo, hi = math.tanh(z - zc * se), math.tanh(z + zc * se)
    zstat = abs(z) / se
    # two-sided normal tail
    p = math.erfc(zstat / math.sqrt(2))
    return lo, hi, p


print("\n--- (a) 相关性（含 Fisher-z 95%CI 与 p 值）---")
for label, ys in [("prune_loss", loss), ("no_prune_PDMS", base)]:
    for nm, fn in [("pearson", pearson), ("spearman", spearman)]:
        r = fn(krs, ys)
        lo_, hi_, p = fisher_ci(r, len(krs))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  corr(kr, {label:14s}) {nm:9s} r={r:+.4f} 95%CI=[{lo_:+.4f},{hi_:+.4f}] "
              f"p={p:.3e} {sig}")
print("  claim 检验: corr(kr, no_prune_PDMS) 显著为负 => budget head 确实给「本身更难的场景」更多预算")
print("            corr(kr, prune_loss) ~ 0      => 但预算与「对剪枝的敏感度」几乎无关")

print("\n--- (b) 按 kr 分组的剪枝损失 ---")
bins = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.62)]
print(f"  {'kr bin':14s} {'n':>6s} {'no-prune':>9s} {'budgetRL':>9s} {'loss':>9s}")
for lo_, hi_ in bins:
    idx = [i for i, v in enumerate(krs) if lo_ <= v < hi_]
    if not idx:
        continue
    b = sum(base[i] for i in idx) / len(idx)
    p = sum(brl[common[i]] for i in idx) / len(idx)
    print(f"  [{lo_:.2f},{hi_:.2f})   {len(idx):6d} {b:9.4f} {p:9.4f} {b - p:+9.4f}")
print("  期望（若 claim 成立）: 各组 loss 应大致持平（预算分对了 → 难场景多给预算后损失被抹平）")
print("  反例信号: 低 kr 组 loss 明显更大 → 预算给少了，说明没分对")

print("\n--- (c) paired bootstrap 显著性 (Reviewer #4) ---")
pairs = [
    ("BudgetRL_raw  vs  SFT_fixed_r0.355_raw", brl, sft),
    ("BudgetRL_raw  vs  no_prune",             brl, nop),
    ("SparseVLM_fb  vs  no_prune",             spv, nop),
    ("BudgetRL_raw  vs  SparseVLM_fb",         brl, spv),
]
for name, a, b in pairs:
    if not a or not b:
        print(f"  {name:42s} SKIP (missing artifact)")
        continue
    obs, lo_, hi_, p, n = paired_bootstrap(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name:42s} n={n:6d} Δ={obs:+.5f} 95%CI=[{lo_:+.5f},{hi_:+.5f}] p={p:.4f} {sig}")

# 同协议（denylist 替换）后的对比
deny = set(json.loads((ROOT / "results/varB_catastrophic_tokens.json").read_text()))
def with_deny(d):
    out = dict(d)
    for t in list(out):
        if t in deny and t in nop:
            out[t] = nop[t]
    return out
brl_d, spv_d = with_deny(brl), with_deny(spv)
obs, lo_, hi_, p, n = paired_bootstrap(brl_d, spv_d)
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
print(f"  {'BudgetRL+deny vs  SparseVLM+deny (同协议)':42s} n={n:6d} Δ={obs:+.5f} "
      f"95%CI=[{lo_:+.5f},{hi_:+.5f}] p={p:.4f} {sig}")

print("\n--- (d) 汇总 PDMS ---")
for nm, d in [("no_prune", nop), ("BudgetRL_raw", brl), ("BudgetRL+deny", brl_d),
              ("SparseVLM_fb", spv), ("SparseVLM+deny", spv_d), ("SFT_r0.355_raw", sft)]:
    if d:
        print(f"  {nm:18s} N={len(d):6d} PDMS={sum(d.values())/len(d):.5f}")
print("=" * 78)

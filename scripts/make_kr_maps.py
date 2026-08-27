#!/usr/bin/env python3
"""make_kr_maps.py — 生成 matched-compute 对照所需的 per-scene keep_ratio 映射表。

目的（回应 AAAI Reviewer #2「缺少 matched-compute 的动态预算对照」）：

构造三个只在「预算来源」上不同、其余完全相同的配置，token ranking 全部来自同一个
Budget RL scorer（ckpt sh0），因此差分可以干净地归因：

  (1) const  : 所有场景 keep_ratio ≡ mean(learned)          → 无预算方差、无场景关联
  (2) shuffle: learned 预算的一个随机置换                    → 有预算方差、无场景关联
  (3) learned: 模型自己预测的 per-scene 预算（已有产物）      → 有预算方差、有场景关联

  Δ(2)-(1) = 「预算有方差」本身的贡献（可正可负）
  Δ(3)-(2) = 「预算与场景匹配」的贡献 ← 这正是论文 claim 的核心

三者平均 keep_ratio 完全相同（置换不改变边际分布），因此平均 FLOPs 严格 matched。

数据来源：logs/_budget_rl_eval_8gpu/_MT_budget_rl_dynamic_sh{0..3}.log
          （prune_verbose=true 的 Budget RL sh0 全 navtest run，11,576 scenes）
"""
import re
import json
import glob
import random
import statistics
from pathlib import Path

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
LOGDIR = ROOT / "logs/_budget_rl_eval_8gpu"
OUTDIR = ROOT / "results/kr_maps"
OUTDIR.mkdir(parents=True, exist_ok=True)
SEED = 20260803

pat = re.compile(r"\[token_budget\] scene=(\w+) N=(\d+) kr=([0-9.]+)")
kr = {}
for f in sorted(glob.glob(str(LOGDIR / "_MT_budget_rl_dynamic_sh*.log"))):
    for line in open(f):
        m = pat.search(line)
        if m:
            kr[m.group(1)] = float(m.group(3))

scenes = sorted(kr)                      # 排序保证可复现
vals = [kr[s] for s in scenes]
n = len(scenes)
mean_kr = sum(vals) / n
assert n == 11576, f"expected 11576 scenes, got {n}"

# (1) const map
const_map = {s: round(mean_kr, 6) for s in scenes}

# (2) shuffled map — 置换 values，保持 keys
rng = random.Random(SEED)
perm = vals[:]
rng.shuffle(perm)
shuf_map = {s: v for s, v in zip(scenes, perm)}

# 校验：边际分布严格一致
assert sorted(shuf_map.values()) == sorted(vals), "permutation changed the multiset!"
assert abs(sum(shuf_map.values()) / n - mean_kr) < 1e-12
n_fixed = sum(1 for s in scenes if abs(shuf_map[s] - kr[s]) < 1e-12)

(OUTDIR / "kr_map_const.json").write_text(json.dumps(const_map))
(OUTDIR / "kr_map_shuffled.json").write_text(json.dumps(shuf_map))

meta = {
    "source_logs": "logs/_budget_rl_eval_8gpu/_MT_budget_rl_dynamic_sh{0..3}.log",
    "source_ckpt": "ckpt/s3_token_scorer_budget_rl_20260722_155943_sh0/ckpt_best",
    "n_scenes": n,
    "mean_keep_ratio": mean_kr,
    "median_keep_ratio": statistics.median(vals),
    "std_keep_ratio": statistics.pstdev(vals),
    "min": min(vals), "max": max(vals),
    "shuffle_seed": SEED,
    "n_scenes_unchanged_by_shuffle": n_fixed,
    "purpose": "matched-compute control for AAAI Reviewer #2",
}
(OUTDIR / "kr_maps_meta.json").write_text(json.dumps(meta, indent=2))

print(f"n_scenes            = {n}")
print(f"mean keep_ratio     = {mean_kr:.6f}")
print(f"median              = {statistics.median(vals):.4f}")
print(f"std                 = {statistics.pstdev(vals):.4f}")
print(f"range               = [{min(vals):.3f}, {max(vals):.3f}]")
print(f"shuffle seed        = {SEED}")
print(f"scenes unchanged    = {n_fixed} ({100*n_fixed/n:.2f}%)  <- should be tiny")
print(f"marginal dist check = PASS (identical multiset)")
print(f"\nwrote:\n  {OUTDIR/'kr_map_const.json'}\n  {OUTDIR/'kr_map_shuffled.json'}\n  {OUTDIR/'kr_maps_meta.json'}")

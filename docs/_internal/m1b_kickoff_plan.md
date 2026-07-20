# M1.b Kickoff Plan — Full navtrain Attention Extraction at L=12

**Status**: 设计完成，未启动（2026-06-24 23:00 GPU 回收窗口装不下）。
**Owner**: 下个 AI session（≥4 GPU、≥20h 连续窗口可启）。
**Goal**: 在 navtrain 全 103,288 token 上抽 L=12 attention 张量，作为 M1.c learned per-scene pruning policy 的数据池。
**Prereqs**: ✅ M1.a Step 5 PASS (vfm=0.1693, n=100) · ✅ navtrain `.chain_complete` · ✅ 单卡 pipeline 0 err (已 n=100 实证)。

---

## 1. 单元成本预算（基于今晚 100-token 实测）

| 量 | 实测值 | 来源 |
|---|---:|---|
| 单 token wall-clock @ 1×H20 | **2.16 s/scene** | `2026-06-24_m1a_step5_navtrain_probeA_pass.md` |
| 单 token .pt 体积 | **~10.3 KB** | `ls -la exp/m1a_navtrain_probeA_L12/*.pt` |
| 单 token err 率 | **0 / 100** | probeA run summary |

外推 103,288 token：

| | 1×H20 | 4×H20 (理想并行) |
|---|---:|---:|
| Wall-clock | **62.0 h** | **15.5 h** |
| 含 model load × N_workers | +1.9 min/worker | +7.6 min total |
| 输出 .pt 总量 | 1.04 GB | 同 |
| nocot 输入 JSON | 已存在 `data/navtrain_nocot_*/` （M0.2 产物，先 verify）| — |

→ **4 GPU × ~16 h** 是合理预算。**不要**在 <18 h 的窗口启动。

---

## 2. D0 干跑（**第一步，必做**）

在启动全量前，跑一个 **500-token / 单 GPU / ~18 min** 的干跑，目的：

1. **复测速度** —— 100 → 500 是否仍 2.16 s/scene（warm-up 摊薄后的真稳态）
2. **复测 err 率** —— err 是否仍 0（小样本不暴露的边界 scene 在 500 量级开始浮现）
3. **复测 .pt 体积分布** —— 是否存在异常大的样本
4. **触摸 IO** —— 验证 sensor blob fs 在持续 18 min 单 GPU 拉取下不抖动（白天 CephFS 偶尔 stall）

### 2.1 D0 命令

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# 1) 从 navtrain.yaml 抽 500 token（避开 probeA 已用的前 100）
PYTHONPATH=code:code/third_party/AutoVLA/navsim:code/third_party/AutoVLA \
  /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python - <<'PY'
import yaml, pathlib
src = pathlib.Path("code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
toks = yaml.safe_load(src.read_text())["tokens"]
used = set(open("exp/m1a_navtrain_probeA_setup/tokens_100.txt").read().split())
fresh = [t for t in toks if t not in used][:500]
out = pathlib.Path("exp/m1b_D0_setup")
out.mkdir(parents=True, exist_ok=True)
(out / "tokens_500.txt").write_text("\n".join(fresh) + "\n")
print("written", len(fresh), "to", out / "tokens_500.txt")
PY

# 2) 同样的方式生成 scene_filter yaml（参照 navtrain_probe100.yaml 模板）
# ... （Step 5 已有等价脚本，复用 scripts/build_navtrain_probe_yaml.sh 或同名 helper）

# 3) 跑 probe（复用 M1.a 的 driver）
mkdir -p exp/m1b_D0_L12 logs
nohup bash scripts/run_m1a_attention_probe.sh \
    --scene-filter navtrain_D0_500 \
    --save-dir exp/m1b_D0_L12 \
    --layer-idx 12 \
    --gpu 0 \
    --max-scenes 500 \
    > logs/m1b_D0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "D0 PID=$!"
```

### 2.2 D0 acceptance（D0 完成后 **必须** 全 PASS 才能启 M1.b 全量）

| 检查 | 阈值 | 行动 if fail |
|---|---|---|
| `ls exp/m1b_D0_L12/*.pt | wc -l` | == 500 | 查 err，决定是否需要 token blacklist |
| `err` 数（从 log grep） | ≤ 5 (1%) | > 5 → 报 user，不要启全量 |
| 总 wall-clock | < 22 min | 若 >> 22 min → 重估 4-GPU 预算 |
| 平均 .pt 体积 | 10 ± 2 KB | 若爆增 → 检查是否存了不该存的张量 |
| 磁盘 free | `df -h /apdcephfs` 仍 > 100 G | 不够则先清 |

D0 全 PASS → 写一行 `RESUME_MONDAY.md`，进 §3。

---

## 3. M1.b 全量 4-GPU 切片方案

### 3.1 切片策略：**token list 等分**（不用 yaml）

不要走 navtest sweep 那条 4-shard yaml 路线（那个 race-fix 用 `_g${GPU}` 后缀不适合 attention 抽取，会让 N=4 个 dir 各存一份 manifest，aggregation 复杂）。这次用更简单的 **token-list 4 等分 + 共享 save-dir**：

```
tokens_103288.txt  →  4 等分
  shard0: token[      0 : 25822]   → GPU 0
  shard1: token[  25822 : 51644]   → GPU 1
  shard2: token[  51644 : 77466]   → GPU 2
  shard3: token[  77466 : 103288]  → GPU 3

  all 4 workers 写入同一个 save-dir: exp/m1b_full_L12/*.pt
  （token 名做 dir entry → 天然无冲突，不需要 race-fix）
```

每个 worker 一个 log：`logs/m1b_full_g{0..3}_<TS>.log`。

### 3.2 dispatcher 草稿（**新写，不要复用 phaseF**）

```bash
# scripts/run_m1b_full_4gpu.sh （还未写，下个 AI 写）
# - 接收 GPUS="0 1 2 3"
# - 自动 split token list 成 4 shard
# - 起 4 个 nohup setsid worker（每个跑 run_m1a_attention_probe.sh --gpu $G --token-list shard$G.txt）
# - 写 manifest.json {git_head, ts, n_total, shards: [...]}
# - 不需要 aggregate.json（M1.b 产物是 .pt 不是 PDMS 标量）
```

关键约定：
- **`--max-scenes -1` 或省略** → probe 跑完整 shard
- **resume 语义**：worker 启动前 `ls save-dir/*.pt`，已存在的 token 自动 skip（probe runner 已支持，见 `run_attention_probe.py`）
- **timeout**：单 worker 不设 timeout（25K token × 2.16 s ≈ 15.5 h），用 dispatcher-level 22h watchdog 兜底
- **rc 收集**：worker exit → dispatcher 累计 rc，4 都为 0 → 整体 PASS

### 3.3 watchdog

```bash
# scripts/watch_m1b_full.sh （新写）
# 每 10 min：
#   - count .pt in save-dir
#   - tail last err lines per worker log
#   - eta = (103288 - count) * 2.16 s / 4
# 退出：count == 103288 → DONE
#       某 worker process 死且 count < target → PARTIAL_FAIL
#       wall > 22h → TIMEOUT
```

### 3.4 完工 acceptance

| | 目标 |
|---|---|
| `ls exp/m1b_full_L12/*.pt | wc -l` | **103,288** ± 100（少量 SceneLoader 边界 scene 可接受） |
| err 累计 | ≤ 1% (≤ 1033) |
| 磁盘 | ≤ 1.5 G |
| manifest.json | git head + per-shard rc + total wall |

写入 `docs/results/key_results.md` §M1.b₂ 一行（新增 section），journal 一份。

---

## 4. ⚠️ 不要做的事

1. ❌ 不要复用 `scripts/run_m1b_freelunch_sweep.sh` —— 那是 PDMS sweep 的，跟 attention 抽取 IO pattern 完全不同
2. ❌ 不要在 < 18 h 的 GPU 窗口启动 —— 跑半截被回收 = save-dir 半成品污染，下次 resume 也是麻烦
3. ❌ 不要把 .pt 存到 `results/` —— attention 张量是中间产物，归属 `exp/m1b_full_L12/`
4. ❌ 不要改 `run_attention_probe.py` 的输出格式 —— M1.a / M1.b₀ 已有 14 layer × 100-500 个 .pt 依赖该格式
5. ❌ 不要在 D0 之前直接启全量 —— 2.16 s/scene 是 n=100 估计，500 才是稳态测量

---

## 5. 后续 (M1.c) 的衔接

M1.b 产物 → M1.c：
- 输入：`exp/m1b_full_L12/{token}.pt` × 103,288
- 每 .pt 包含：`q_attn (1, 16, 941, 941)` 截到 vision block + scene meta
- M1.c：训一个 per-scene token-relevance regressor (input=visual feature, target=vision_frac in {0,1} mask)
- 不需要再跑 nocot inference（M1.b 已经把 forward 走过一遍）

→ 这就是为什么 M1.b 必须**一次跑完 103K**，不能再分批：每次 model load 1.9 min × N 批 浪费严重。

---

## 6. 接手 checklist

下个 AI 进入时按顺序做：

1. 读本文件 + `RESUME_MONDAY.md` 顶部 2026-06-24 节
2. 跑 D0（§2），约 22 min
3. D0 acceptance 全过 → 写 `scripts/run_m1b_full_4gpu.sh` + `scripts/watch_m1b_full.sh`（草稿在 §3.2/3.3）
4. dry-run / bash -n 通过
5. 确认 ≥ 18 h GPU 窗口
6. 启动全量 + watchdog
7. 完工：count check + manifest + 写 key_results.md §M1.b₂ + journal

预算：D0 22 min + 写脚本 ~1h + 全量 16h + 收尾 30min ≈ 18h 总窗口。

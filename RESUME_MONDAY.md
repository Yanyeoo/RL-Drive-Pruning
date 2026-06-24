# RESUME_MONDAY — 周一接手 (2026-06-22 起)

> 写于 2026-06-18 19:30。今天 23:00 GPU 回收，下周一恢复。
> M1.a 已**完全交付**（navtest, n=500, L\*=12 locked）。
> 周一主任务：M1.a Step 5 复核 + M1.b 启动。

---

## 🛑 2026-06-24 20:25 — 23:00 GPU 回收前状态（**START HERE 下一个 AI**） 🛑

**时间窗口现实**：现在 20:25，距 23:00 GPU 回收 **2h35min**。M1.b 全量 navtrain attention 抽取
（103,288 token × 2.16 s/scene）单卡需 62 h，4 卡需 ~16 h —— **2h35 装不下任何有意义的 M1.b 进度**。

**决策**：今晚 **不启** M1.b。把这 2h35 用于交付文档，让下个 AI（≥18h 窗口）能直接照跑。

**今天交付清单**（截至 20:25 已完成）：

| | 状态 | 路径 |
|---|---|---|
| navtrain UNBLOCKED（`.chain_failed` 假阳性 fix） | ✅ 17:30 | incident_2026-06-24_navtrain_chain_failed_false_positive.md |
| M1.a Step 5 navtrain probe A PASS（vfm=0.1693, n=100） | ✅ 20:10 | journal/2026-06-24_m1a_step5_navtrain_probeA_pass.md |
| `key_results.md` §3 navtrain status / §4.5 probe A 数据表 / changelog 三处更新 | ✅ 20:25 | docs/results/key_results.md |
| **M1.b kickoff plan**（D0 干跑 + 4-GPU 切片 + acceptance） | ✅ 20:25 | **docs/\_internal/m1b\_kickoff\_plan.md** ← 下个 AI 必读 |
| RESUME_MONDAY.md 顶部状态节（本节） | ✅ 20:25 | 本文件 |

**M1.b 状态**：⏳ 未启动。设计完整、命令模板齐全。下个 AI 进入后**第一步读 m1b_kickoff_plan.md**，然后：

1. **D0 干跑**（500 token / 1 GPU / 22 min）—— 验 2.16 s/scene 稳态、err 率、.pt 体积、IO（必做，不可跳）
2. D0 全 PASS → 写 `scripts/run_m1b_full_4gpu.sh` + `scripts/watch_m1b_full.sh`（plan §3.2/3.3 有草稿）
3. 确认 ≥ 18 h GPU 窗口 → 启全量
4. 完工 → 写 `key_results.md` §M1.b₂ 新章 + journal

**核心数字（外推用）**：
- 单 token wall-clock = **2.16 s @ 1×H20**
- 单 token .pt size = **~10.3 KB**（n=100 实测，1.1 MB / 100 token）
- 全量 103,288 token 预算：**4 GPU × ~16 h，磁盘 ~1.04 GB**
- err 率：n=100 上 0/100，需 D0 在 n=500 复核

**今晚 23:00 前剩余动作**：仅做"不烧 GPU 的收尾"——
- 可能：本节再润色 / `scripts/run_m1b_full_4gpu.sh` 的脚本草稿先写出来（下次 AI 直接 ready）
- 不要：启 GPU job、改 yaml、push GitHub

**下个 AI 一进来读这 4 个文件**（按顺序）：
1. 本文件本节（你正在读的）
2. `docs/_internal/m1b_kickoff_plan.md`
3. `docs/results/key_results.md` §4.5 + §6
4. `docs/journal/2026-06-24_m1a_step5_navtrain_probeA_pass.md`（若需理解 probe A 细节）

---

## 🟢🟢🟢 2026-06-24 20:10 — M1.a Step 5 navtrain probe A **PASS** 🟢🟢🟢

**L\*=12 在 navtrain 也成立**。M1.a 完整交付（navtest + navtrain 双验）。

| | navtest (already locked) | navtrain probe A (NEW) |
|---|---|---|
| N | 500 | 100 |
| L | 12 | 12 |
| vision_frac_mean | (locked, 详见 key_results) | **0.1693** |
| std | — | 0.0527 |
| min / max | — | 0.0705 / 0.3783 |
| acceptance [0.15, 0.22] | PASS | **PASS** ✅ |

落地物：
- 100 token list: `exp/m1a_navtrain_probeA_setup/tokens_100.txt`
- nocot 数据: `data/navtrain_nocot_probe100/*.json` (100 files)
- probe outputs: `exp/m1a_navtrain_probeA_L12/*.pt` (100 files)
- summary: `exp/m1a_navtrain_probeA_L12/probeA_summary.json`
- yaml: `code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_probe100.yaml`

**下一步立即可推**：**M1.b — 全 navtrain attention 抽取**（L=12, 全 103,288 token），为 M1.c 数据池。

### ⚠️ 给下个 AI 的扫描坑（**不要重复犯**）

为了这次 probe，我先写了一个 "扫缺图" 脚本 (`scripts/scan_navtrain_missing_images.py` + `scan_navtrain_window.py`)，
对每个 navtrain target token 检查 [-4,+10] 窗口内 8 cam × 14 frame 共 112 张 jpg 是否全在磁盘。
结果报 **81% unusable**，吓得我准备改 MA2.x 全部跑 19K 子集。

**这是错的**——和 incident §3 描述的"`build_all_sensors()` smoke test 必然失败"是**同一个反模式**。
- navtrain 真实 sensor_config 在 14 帧窗口里**只用 1-2 个 key-frame 的 cam**（每 cam 每 scene 共 9 张 jpg 是设计）
- 全量 navtrain 103,288 token 都是可用的（incident §2.2 已用标准 SceneLoader 证过 diff=0）
- D4 实测 100 token 全成功（ok=100, skip=0, err=0）已经反向证明

产物 `navtrain_window_clean_tokens.txt` / `navtrain_window_report.json` **保留作 forensic，但不要再当作 MA2.x 的 token 池来用**。

---

## 🟢🟢 2026-06-24 17:30 — navtrain UNBLOCKED 🟢🟢

**给下一个 AI 的最重要一句话**：`.chain_failed` 是假的。**navtrain 数据完整可用，立刻可以做 M0.4 和 M1.a Step 5**。

证据 + 处理：

- 数据落点 `$OPENSCENE_DATA_ROOT = /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/`
  - `navsim_logs/trainval/`  = **1310 .pkl, 14 G** ✅
  - `sensor_blobs/trainval/` = **1192 scene dir, 443 G** ✅（navsim 文档 spec 445 G）
- 8 个 `navtrain_*.tgz` **md5 全过**（见 download log 2026-06-23 03:00:05）
- navsim 标准 `SceneLoader` + `navtrain.yaml` filter 实测：**built=103288 == declared=103288, diff=0** ✅
- `.chain_failed` 已翻 → `.chain_failed.false_positive`，写入 `.chain_complete`（含 provenance）
- `scripts/install_navtrain.sh:81-82` 已 patch（`ls | head -3` SIGPIPE → 加 `|| true` 兜底）

**根因**：`install_navtrain.sh:81` 末尾 `ls "${LIVE_LOGS}" | head -3` 在 `set -euo pipefail` 下触发 SIGPIPE → rc=141 → `.chain_failed`。此时所有 install 实质工作已成功。全文档见：

- `docs/_internal/incident_2026-06-24_navtrain_chain_failed_false_positive.md`（完整 timeline + 给下个 AI 的避坑指南）
- `docs/journal/2026-06-24_chain_failed_was_false_positive.md`（短总结）

**坑预警 ⚠️**：不要拿 `SensorConfig.build_all_sensors()` + `get_scene_from_token` 做 smoke test —— 一定会 `FileNotFoundError` on cam jpg，**这是 navtrain 稀疏 key-frame 设计**（每个 scene 每 cam 只有 9 张图，不是 1:1 with log frame），不是数据缺失。用 `build_no_sensors()` 验存在性，用训练 yaml 里的 `sensor_config` 做端到端。详见 incident doc §3。

**下一步（按优先级）**：

1. **M0.4** navtrain r=1.0 baseline EPDMS（本来今天就该跑，被假 sentinel 卡 1.5 天）
2. **M1.a Step 5** navtrain probe A，验 L\*=12 在 navtrain 也成立（acceptance `vision_frac_mean ∈ [0.15, 0.22]`，模板在本文件下方"probe A on navtrain"小节）
3. → M1.b 全 navtrain attention 抽取 → M1.c → M2 → M5

V4 isolation 不在主线上，先放着。

---

## ⚠️⚠️ START HERE — 2026-06-23 19:17 最新交接（Phase F full navtest sweep） ⚠️⚠️

**当前情况（更新于 19:17）**：

- **GPU 已 idle**（0/1 都 0 MiB used）。所有 sweep 进程已 kill，无残留。
- **17:14 INCIDENT (timeout)**：单 GPU V0 sweep 因 `TIMEOUT=5400s` 太短被 `timeout(1)` 强杀，manifest `rc=124 pdms=null`，cell 数据丢失。17:22 kill PG=109061 止损，失败 dir 归档 `results/raw/_failed_timeout/`。
- **19:11 INCIDENT (race)**：首次 2 GPU dispatcher 启动后立刻发现 worker0 (V0:s0) 与 worker1 (V0:s1) 解析出**同一个** out dir `M1b_freelunch_V0_20260623_191145`。根因：`run_m1b_freelunch_sweep.sh` 用本地 `date +%Y%m%d_%H%M%S` 拼 dir 名（line 101→102/109），**两 worker 同秒启动 → 同 TS → 同 dir → manifest/csv 互相覆盖**，会导致 16 cell 数据互踩。19:13 立刻 SIGKILL 全部 10 个残留进程，空 dir 归档 `results/raw/_failed_race/M1b_freelunch_V0_20260623_191145_RACED`（里面只有 0-byte 的 shard0.log，无数据损失）。
- **19:15 RACE FIX**：改 `scripts/run_m1b_freelunch_sweep.sh` line 102/109，**EXP_NAME 和 VARIANT_DIR 都加入 `_g${GPU}` 标识**：
  - `EXP_NAME="${TAG_PREFIX}_${VARIANT}_g${GPU}_${TS}"`
  - `VARIANT_DIR=".../M1b_freelunch_${VARIANT}_g${GPU}_${TS}"`
  - dispatcher `is_done()` 的 glob `M1b_freelunch_${V}_*` **仍能匹配**（`_g0_...` 是 `_*` 的子集）
  - 完工 aggregation 脚本读 `manifest.scene_filter` 区分 shard，**不依赖 dir 名**
  - bash -n 通过
- **所有失败/陈旧 dir 已归档**：
  - `results/raw/_failed_timeout/` ≥ 8 dir（17:14 timeout 事故）
  - `results/raw/_failed_race/` 1 dir（19:11 race 事故，空）
  - `results/raw/M1b_freelunch_*` 当前**空**，重跑全部 16 cell 不会有冲突。
- 目标：full navtest (4 shard × ~2894 token ≈ 11576 token) × 4 variant (V0/V1/V2/V3) 的 PDMS。M1.b free-lunch 假设 + Pareto front 的**论文表数据**。
- **关键命名约定（race-fix 后）**：所有 cell 的 dir 名格式 = `M1b_freelunch_<V>_g<G>_<TS>`，例如 `M1b_freelunch_V0_g0_20260623_192000` （V0 on GPU0）。shard 仍**只能靠 `manifest.json:scene_filter`** 区分（因为 dispatcher 把 4 shard 分到 2 GPU，同 GPU 上的 shard 是串行而非通过 dir 名编码）。
- **Timeout 陷阱**：dispatcher 默认 `TIMEOUT=8100`（见 `scripts/run_m1b_phaseF_2gpu.sh:36`），**不要再 export TIMEOUT 覆盖**。

**你（下一个 AI）必须做的步骤**：

### 1. Sanity check — 确认 GPU idle、RAW 干净

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
pgrep -af "run_m1b_freelunch_sweep|run_pdm_score_cot" || echo "clean"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ls -d results/raw/M1b_freelunch_* 2>/dev/null || echo "RAW empty (good)"
ls results/raw/_failed_timeout/ | wc -l   # 应该 >= 8
```

### 2. 等 2 GPU 切换到位 — 确认 nvidia-smi 看到 2 卡且都 idle

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
# 应该看到 GPU 0 和 GPU 1，memory.used 都 < 1 GiB
```

### 3. 启动 2-GPU 全 sweep dispatcher

dispatcher 脚本：`scripts/run_m1b_phaseF_2gpu.sh`（已写好、dry-run 通过）。

特性：
- 自动 SKIP 已有 (V, S) 的 done dir（V0 shard0 若 step 1 拿到 manifest 就会自动跳）
- 16 job 均分到 2 GPU
- TIMEOUT 8100s/job（~2.25h，比观察到的 108min 留 25% 余量）
- 输出落 `results/raw/M1b_freelunch_<V>_<TS>/aggregate.json + manifest.json`
- dispatcher 自己的 log：`logs/m1b_phaseF_2gpu_<TS>.log`
- 两个 worker log：`logs/m1b_phaseF_2gpu_gpu{0,1}_<TS>.log`

启动命令：

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
TS=$(date +%Y%m%d_%H%M%S)
nohup setsid env GPUS="0 1" \
  bash scripts/run_m1b_phaseF_2gpu.sh \
  > logs/m1b_phaseF_2gpu_boot_${TS}.log 2>&1 < /dev/null &
DISPATCHER_PID=$!
disown 2>/dev/null
echo "${DISPATCHER_PID}" > /tmp/phaseF_2gpu_pid.path
ps -o sid= -p ${DISPATCHER_PID} > /tmp/phaseF_2gpu_sid.path
echo "dispatcher pid=${DISPATCHER_PID} sid=$(cat /tmp/phaseF_2gpu_sid.path)"
# 3 分钟后确认 worker 起来了
sleep 180
tail -50 logs/m1b_phaseF_2gpu_${TS}.log
ls logs/m1b_phaseF_2gpu_gpu{0,1}_${TS}.log 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### 4. 起 watchdog（每 10 min 自动 dump 矩阵）

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
nohup setsid bash scripts/watch_m1b_phaseF_2gpu.sh \
    > logs/m1b_phaseF_2gpu_watch.boot.log 2>&1 < /dev/null &
disown 2>/dev/null
sleep 12
echo "=== first snapshot ==="
cat logs/m1b_phaseF_2gpu_watch.log
```

watchdog 自动停止条件：
- 16 cells 全 done → exit 0
- dispatcher 死了但有 cell 没 done → exit 1（PARTIAL）
- 22h 超时（兜底）→ exit 2

### 5. 监视点（每隔 ~1h check 一下）

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
echo "=== latest watch snapshot ==="
tail -30 logs/m1b_phaseF_2gpu_watch.log
echo "=== current matrix ==="
# 也可以手动跑一次 watchdog 的 dump（不阻塞）：
bash -c 'source scripts/watch_m1b_phaseF_2gpu.sh > /dev/null 2>&1 &
WPID=$!; sleep 8; kill ${WPID} 2>/dev/null; wait 2>/dev/null
tail -30 logs/m1b_phaseF_2gpu_watch.log'
```

### 6. 完工后 — aggregate 4-variant × 4-shard 数字 → 论文表

预计 **次日 ~08:00–10:00** 全部完工（16h × 2GPU）。watchdog DONE 后：

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
# 列出 16 个 cell 的 PDMS
python3 - <<'PY'
import json, glob
from collections import defaultdict
rows = defaultdict(dict)
for f in glob.glob("results/raw/M1b_freelunch_*/manifest.json"):
    m = json.load(open(f))
    a = json.load(open(f.replace("manifest", "aggregate")))
    sf = m.get("scene_filter", "")
    if "navtest_local_filtered_shard" not in sf:
        continue
    s = sf.split("shard")[1].split("_")[0]
    v = m.get("variant")
    if m.get("rc") == 0 and a.get("pdms") is not None:
        rows[v][f"s{s}"] = (a["pdms"], a.get("n_valid"))
print(f"{'var':<4} {'s0':>15} {'s1':>15} {'s2':>15} {'s3':>15}")
for v in sorted(rows):
    cells = [f"{rows[v].get(f's{i}',('-','-'))[0]:.2f}({rows[v].get(f's{i}',('-','-'))[1]})" if rows[v].get(f's{i}') else '-' for i in range(4)]
    print(f"{v:<4} " + " ".join(f"{c:>15}" for c in cells))
PY
# 然后写入 docs/results/key_results.md 的 M1.b §
```

### 故障预案

- **某个 cell 卡死**（log 长时间不动）：单 cell timeout 8100s 自动 kill 该 job，sweep 继续下一个；若是整个 worker 卡 → `kill -9 -- -<sid>` 后用 `SKIP_DONE=1` 重启 dispatcher，跳已 done 跑剩下的
- **重启 dispatcher 怎么续跑**：直接重新跑 step 3 的命令，SKIP_DONE 默认开，已 done 的会被 manifest+aggregate 检测自动跳
- **GPU OOM / CUDA error**：通常出现在 model load 阶段。看 shard0.log 里 traceback。若反复在同一 variant 出，可能是 V2/V3 mask 太激进 → 报告 user
- **磁盘满**：每个 cell 产出 ~10 MB。16 cell ≈ 160 MB，安全。但 model load tmpfile 可能占用，注意 `/tmp` 用量

### 一定不要做的事

1. ❌ 不要改 `scripts/run_m1b_freelunch_sweep.sh`（inner runner，改了会破坏 dispatcher 的 dir 假设）
2. ❌ 不要在 dispatcher 跑的时候删 `results/raw/M1b_freelunch_*` —— 是热数据
3. ❌ 不要 `kill -9 dispatcher pid` —— 用 SID kill 整个 process group：`kill -9 -- -$(cat /tmp/phaseF_2gpu_sid.path)`
4. ❌ 不要在结果出来前 push GitHub
5. ❌ 不要自己加新 variant V4/V5 —— 当前 V0/V1/V2/V3 已锁

### 结果记录到这里 ⬇️ （**完工后请填**）

```
[Phase F full navtest sweep, 4 var × 4 shard, n≈11574 / variant]

INCIDENT LOG (2026-06-23, before this run):
  15:44 单 GPU sweep 启动 V0 s0 (TIMEOUT=5400s 来自外部 env, 太短)
  17:14 V0 s0 在 2273/2949 (77%) 被 timeout(1) SIGKILL → manifest rc=124 pdms=null
        sweep 接着启动 V1 s0, 也注定 timeout
  17:22 人工 kill 整个 sweep PG (SID=109061), 失败 dir 归档到 results/raw/_failed_timeout/
  17:25 GPU idle, RAW 干净, 等待 18:00 切 2-GPU 重跑
  19:11 race: 两 worker 同 TS → 同 dir, 13 秒后 kill, 归档 _failed_race/
  19:15 race fix: EXP_NAME/VARIANT_DIR 加 _g${GPU}，bash -n 通过
  fix:  dispatcher 默认 TIMEOUT=8100s (脚本第36行), 启动时别再 export TIMEOUT 覆盖

launched: 2026-06-23 19:18:20 (TS=20260623_191820, 2 GPU)
finished: 2026-06-24 09:53:03 (dispatcher rc_agg=0; watchdog DONE 09:58:18)
wall:     ~14h 35min for 16 cells (~109 min/cell, 8 cells/GPU 串行)
git HEAD: f084f26

PDMS matrix (cell = pdms × n_valid):
        s0                s1                s2                s3            all (n_valid-weighted)
V0      0.8958 (2949)     0.9003 (2796)     0.8938 (2962)     0.9042 (2867)   0.8985   (N=11574)
V1      0.8960 (2948)     0.9020 (2796)     0.8921 (2963)     0.9026 (2867)   0.8981   (N=11574)
V2      0.8621 (2949)     0.8520 (2796)     0.8421 (2962)     0.8619 (2866)   0.8545   (N=11573)
V3      0.8630 (2947)     0.8498 (2796)     0.8372 (2962)     0.8649 (2867)   0.8537   (N=11572)

Free-lunch check (V0 vs V1):  Δ = V1 - V0 = -0.0004  (|Δ| ≤ 0.005 → free-lunch CONFIRMED)
                              即 mask L12:{h13} 单头 → noise-level 影响，验证 M1.b₀ dead-head 假设

Pareto front (V0/V1/V2/V3):   0.8985 / 0.8981 / 0.8545 / 0.8537
                              - V0→V1:  -0.04 pp, 1 head masked  (0.39% KV saved)  → free-lunch
                              - V1→V2:  -4.36 pp, +3 heads L27   (1.56% KV total)  → cliff at L27 mask
                              - V2→V3:  -0.08 pp, +11 heads L24  (5.86% KV total)  → L24 mask is cheap
                              结论：L27:{h0,h8,h9} 不是 "free"，per-token g_mean 看着小但
                                    对 trajectory 决策仍重要 → V1 是当前最优 Pareto 点。
                                    L24 11-head mask 几乎免费（V2→V3 持平），后续 Level-2
                                    learned policy 可以默认 mask L24 全部 dead head。

Notes / anomalies:
  - 16/16 cell rc=0, 无 OOM, 无 retry
  - n_valid 浮动 (2866–2949 vs nominal shard size 2894) 来自 navtest 中本身就有少量
    invalid scene (无 ego trajectory)，每 variant 总和 11572–11574 / 理论 11576 (差 2–4)
  - 各 variant n_valid 在每 shard 上完全一致 (s0=2947–2949, s1=2796, s2=2962–2963, s3=2866–2867)
    → 验证 invalid scene 与 mask 无关，是 dataset 固有
  - canonical dir 见 RAW listing (V0_g{0,1}_191820 + 4 个 V0_g{0,1}_21:xxxx + 8 个 V1/V2/V3)
```

---

## ⚠️ BEFORE YOU DO ANYTHING — 2026-06-22 13:25 更新

下面 §0 这段是 2026-06-22 周一当天接手后发现的，**必须先读、覆盖下文 §M0.2 的乐观假设**：

### §0. navtrain chain 实际**没跑完**，且当前镜像里没 aria2c

- ❌ RESUME 原文 §M0.2 的"PID 3738 / 3774 / 139364 后台 rsync 中"已**全部死亡**（6/18 23:00 GPU 回收带走）
- ❌ `.chain_complete` / `.download_complete` 都不存在
- ❌ `history_3.tgz.installed` / `history_4.tgz.installed` sentinel 都不存在（rsync 死在半途）
- ❌ 当前镜像里 `aria2c` 二进制找不到（6/17-18 用的镜像换了，二进制丢了）
- ✅ 前 6 个 sentinel（current_1..4 + history_1, 2）完好，可 resume
- ✅ 磁盘 871G / 2T (44%) 安全

**下一步看这里**：`docs/_internal/incident_2026-06-22_aria2c_missing_chain_dead.md` §4 节有可直接复制粘贴的 5 行命令（conda install aria2 + 启 chain watcher + 启 downloader resume），预算 ~110–140 min 到 navtrain ready，再 +10 min GPU probe。

**已淘汰的指令**（不要再相信下文 §M0.2 这两条）：
- ~~"这 3 个 PID 不要 kill"~~ ← 它们已经自然死亡，pgrep 找不到
- ~~"预计周一应该已 `.chain_complete`"~~ ← 实际没发生

---

## 状态快照（2026-06-18 19:30 freeze）

### M0 baseline
- ✅ B0 navtest PDMS = **89.83** locked （`docs/results/key_results.md §3`）

### M0.2 navtrain 数据下载
- ✅ current_1..4：4 split 已 install 完成，sentinel 全在
- 🔄 history_1..4：后台 rsync 进行中（PID 3738 主脚本 + PID 139364 user takeover）
  - 这 3 个 PID **不要 kill**，慢但收敛
  - history_3 / history_4 还在等主脚本顺序处理
  - **预计周一应该已 `.chain_complete`**（5 天 + 周末，绰绰有余）
  - 如果周一还没 `.chain_complete`：检查 `_staging_navtrain/.../trainval_sensor_blobs/.navtrain_*.installed` sentinel；不要碰 staging dir
- chain：post_dl_chain.sh（PID 3774）poll 中，齐了自动触发 install_navtrain → m02_splits

### M1.a attention layer probing
- ✅ **L\* = 12 LOCKED v2** on navtest（n=500）
  - L12 vision_frac_mean = **0.1861**
  - L27 vision_frac_mean = 0.1805（被淘汰）
  - 三重支持：n=500 数值领先 + 下游 15 层 flop 收益 + fine sweep 孤立尖峰结构
  - 决策文档：`docs/_internal/m1a_layer_selection_2026-06-18.md`
  - 数据：`exp/m1a_layer_sweep_20260618_1644/`（14 layer × n=100 + L12/L27 × n=400 extra = 2200 forward passes）
- 🎯 **唯一 pending**：navtrain probe A 10-min 复核（M1.a Step 5）

### M1.b — 还没动
- 设计：在 L12 上学一个 per-scene pruning policy，剪 vision token，PDMS 不掉
- 周一 navtrain `.chain_complete` 后立即可启

---

## 复制下面这一整段发给新 AI：

```
你好，接手 RL-Drive-Pruning 项目。上一个 AI session 在 2026-06-18 23:00 GPU 回收时收尾，无记忆传递。

【第一件事 — 强制】
不要做任何动作。先按顺序读完这些文件：

1. /apdcephfs/private_shayladeng/tokenrl_autoVLA/RESUME_MONDAY.md      ← 本文件
2. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/results/key_results.md          ← 看 §3 (B0) + §4 (M1.a L*=12)
3. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/m1a_layer_selection_2026-06-18.md  ← L*=12 决策全文
4. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/handoff_2026-06-18_session_death.md
5. /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/incident_2026-06-18_false_stall_diagnosis.md

读完用一段话告诉我：
- navtrain 现在状态？(.chain_complete 到了没？current_*/history_* sentinel 全不全？)
- M1.a 状态？(L*=12 锁了吗？navtest n=500 的关键数字是什么？)
- 你接下来第一个具体动作是什么？

【硬规则】
1. 不要重启 navtrain 下载脚本或任何后台 rsync
2. 不要对 _staging_navtrain/ 做 tar/mv/rsync 操作（前任踩过 2 次）
3. 不要 push GitHub（没授权）
4. 关键数字（PDMS/L*/sanity）必须当场写 docs/results/key_results.md
5. 不确定就停下来问，不要自己改
6. M1.a 的 L*=12 是 LOCKED 决策，不要重新 sweep 别的层（除非 probe A 复核失败）

【工作目录】
/apdcephfs/private_shayladeng/tokenrl_autoVLA

【当前 milestone】
- M0 baseline ✅ B0 navtest PDMS=89.83 locked
- M1.a attention probe ✅ L*=12 LOCKED on navtest n=500，唯一 pending = navtrain probe A 10-min 复核
- M0.2 navtrain 数据 🔄 周末后台 rsync，理应已完成；先确认 .chain_complete sentinel
- M1.b RL pruning policy 🎯 你的主任务（前提：navtrain ready + probe A 复核 OK）

【今天周一推荐顺序】
Step 1 (5 min)  确认 navtrain 状态：ls _staging_navtrain/.../*.installed + ls _staging_navtrain/.../.chain_complete
Step 2 (10 min) M1.a Step 5 — navtrain probe A 复核 L*=12（1 卡，100 scene，与 navtest 数字对比）
                跑法：bash scripts/run_m1a_attention_probe.sh --scene-filter navtrain_probeA \
                      --save-dir exp/m1a_navtrain_probeA/L12 --layer-idx 12 --gpu 0 \
                      --token-list <从 navtrain split 选 100 token> --max-scenes 100
                acceptance: vision_frac_mean ∈ [0.15, 0.22]（与 navtest 0.1861 在量级一致）
Step 3 (剩余时间)  M1.b kickoff — 写 spec → smoke → 训练
                需要：L12 attention 作为 input feature 接入 RL policy

开始读文档。
```

---

## 工程速查（写给新 AI 看）

### M1.a 关键产物路径

```
exp/m1a_layer_sweep_20260618_1644/
├── tokens_100.txt                ian# coarse 用的 100 token
├── tokens_500.txt                  # n=500 sanity 全集
├── tokens_100_to_499.txt           # 100–499 增量 token
├── L00..L27/                       # 14 layer × 100 .pt（coarse + fine）
├── L12_500_extra/                  # L12 增量 400 .pt
└── L27_500_extra/                  # L27 增量 400 .pt（已淘汰，留作存档）
```

### 重新算 vision_frac

```bash
PYTHONPATH=/apdcephfs/private_shayladeng/tokenrl_autoVLA/code \
  /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
  -m rldrive.scoring.analyze_layer_sweep \
  --sweep-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_layer_sweep_20260618_1644 \
  --layers 0,4,8,10,11,12,13,14,16,20,24,25,26,27
```

数字应当与 `m1a_layer_selection_2026-06-18.md §4.2` 完全一致（L12=0.1789, L27=0.1804 on n=100 only；用 `L12_500_extra` merge 后 L12=0.1861 on n=500）。

### probe A on navtrain（Step 5 模板）

navtrain split 名字按 `m02_splits.sh` 的输出确定（多半是 `data/navtrain_nocot_split_*/`）。从中选 100 个 token（lexical 前 100 即可），跑：

```bash
bash scripts/run_m1a_attention_probe.sh \
    --scene-filter navtrain_probeA \
    --save-dir exp/m1a_navtrain_probeA/L12 \
    --layer-idx 12 --gpu 0 \
    --token-list <navtrain_tokens_100.txt> \
    --max-scenes 100
```

→ 读 100 .pt 算 vision_frac_mean。
- 若 ∈ [0.15, 0.22] → 一致，**M1.a 完全交付，进 M1.b**
- 若 < 0.10 或 > 0.30 → 报告 user，触发 escalate（可能要在 navtrain 上重做 14-layer sweep）

### 不能动的进程（截至 2026-06-18 19:30）

```
PID 3738    主下载脚本，rsync history_1
PID 3774    post_dl_chain.sh，poll .download_complete
PID 139364…  user takeover rsync history_2
```

周一恢复 GPU 时，这些 PID 应该都已自然结束（chain 跑完）。如果还在跑，**仍然不要 kill**。

---

## 为什么这么写

- **首段先列状态快照**：让接手 AI 立刻知道"M1.a 已交付，不要重做"
- **L\*=12 三处提**（快照、prompt、speedlookup）：避免 AI 看到 n=100 时的 L27=0.1804 又起念头切回 L27
- **probe A acceptance 区间写死**：避免新 AI 拿 navtrain probe A 0.17 这种"看起来差不多"的数当 fail
- **Step 1 钉死成 navtrain status check**：避免直接跑 GPU job 后才发现数据没到
- **强调"M1.a 决策 LOCKED 不要重做"**：今天踩过的双峰陷阱不要再来一遍

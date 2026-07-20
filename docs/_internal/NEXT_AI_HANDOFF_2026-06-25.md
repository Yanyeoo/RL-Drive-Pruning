# NEXT_AI_HANDOFF — 2026-06-25 周四接手

> 写于 2026-06-24 21:55 | 上一手：6-24 这个 session（M1.b₂ stage 1+2 全部交付）
> 下一手 GPU 窗口：**2026-06-25 10:00 → 2026-06-26 00:00（H20 单机 4 卡，14h，备注"短时"= 当作有效 ~10h）**

---

## 0. 第一件事：读这 4 个文档（按顺序，每篇 5 min）

1. **本文件**（NEXT_AI_HANDOFF_2026-06-25.md）
2. **`docs/PATHS.md`** — 项目级路径地图（全新建，所有数据/代码/产出在哪）
3. **`docs/journal/2026-06-24_m1b2_stage1_2_full_journey.md`** — 昨晚 M1.b₂ stage 1+2 完整流水账
4. **`docs/_internal/plan_2026-06-24_2055_path3_execution.md`** — 原计划（stage 1+2 已 ✅，stage 3 待你做）

读完上面 4 个 = 完全 onboard。**不要**回 RESUME_MONDAY.md 找历史（那已 2 天前；以这 4 个为准）。

---

## 1. 一句话状态

> **M1.b₂ stage 1（D0 multilayer attention 验码）+ stage 2（19,225 token navtrain pretokenize）全部 ✅ 完成。
> 你今天的任务：用 19,225 token 跑 4-GPU multilayer attention 全量抽取 → 论文级 head/layer selection statistic。**

---

## 2. 当前可信成果（不要重做！）

| 产出 | 路径 | 数量 / 大小 |
|---|---|---|
| Stage 1 attention .pt（probe100 验码） | `exp/m1b2_d0_smoke_probe100_alllayers/*.pt` | 100 个，各 1.30 MB |
| Stage 2 navtrain pretokenized json | `data/navtrain_nocot/*.json` | 19,225 个（19,125 real + 100 symlink），122 MB |
| 19,225 完整 window token 名单 | `exp/m1a_navtrain_probeA_setup/navtrain_window_clean_tokens.txt` | 19,225 行（与 `exp/m1b2_navtrain_full_window_tokens.txt` 完全一致）|
| scene_filter yaml | `code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_avail19k.yaml` | 19,225 token |
| dataset yaml | `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtrain_full.yaml` | 指向 navtrain_avail19k |

### 张量 schema（Stage 1 .pt）
```python
{
  "per_layer_vision_attn": Tensor(28, 16, 720),  # ⚠️ heads=16, 不是 24
  "layer_idxs": [0..27],
  "multi_layer": True, "average_heads": False,
  "vision_blocks": [(108,349),(372,613),(636,877)],
  "token": "...",
}
```

---

## 3. 你今天的目标（推荐）

### 3.1 主任务：Stage 3 — 19,225 token × 全 28 层 multilayer attention 抽取（4 GPU）

**预算估算**：
- Stage 1 实测 2.55 s/scene 单 GPU；
- 4 GPU shard 0.85× scale → 19,225 × 2.55 / (4 × 0.85) ≈ **2.4 h wall**；
- 输出体积：19,225 × 1.30 MB ≈ **25 GB**（注意磁盘）；
- 留 1 h cold-start + acceptance + 缓冲 → **3.5 h 即结束**。**14h 窗口绰绰有余**。

**怎么 4 GPU shard**：`scripts/run_m1a_attention_probe.sh` 已支持 `--gpu N --max-scenes M`。看 plan §3.2/3.3 / m1b_kickoff_plan.md 里有 `run_m1b_full_4gpu.sh` 草稿。模式：
- 把 19,225 token 切 4 份（~4806 each）；
- 每份起 1 个 process 占一张 GPU；
- 共享 save-dir；
- 用 `tokens_100.txt` 同样格式喂 `--token-list`。

**输入命令模板**（参考 stage 1 命令）：
```bash
# shard k ∈ {0,1,2,3}, gpu k:
nohup bash scripts/run_m1a_attention_probe.sh \
  --scene-filter navtrain_avail19k \      # 注意：用新 yaml
  --json-dir /apdcephfs/.../data/navtrain_nocot \
  --token-list /apdcephfs/.../exp/m1b2_full_shards/shard_${k}.txt \
  --save-dir /apdcephfs/.../exp/m1b2_navtrain_full_alllayers \
  --all-layers --num-layers 28 --gpu ${k} \
  > logs/m1b2_full/shard_${k}.log 2>&1 &
```

shard 切分脚本：5 行 python，`split exp/m1a_navtrain_probeA_setup/navtrain_window_clean_tokens.txt` into 4。

**Acceptance 阈值**（同 stage 1）：
- 文件数 = 19,225（允许 0–10 个 MISSING.json，逐个查 root cause）
- s/scene 平均 ≤ 4.0
- 每 .pt shape (28, 16, 720)
- 显存 < 50 GB / GPU
- 0 Python error

### 3.2 次任务（如果时间还剩 ≥ 2 h）

跑论文 Section 4 的 head selection / layer selection statistic：
- 对每 head (l, h) 算 `top-k vision token attention sum` over 19,225 scene；
- rank by mean → identify "salient heads"；
- compare with M1.a probe A 的 100 token vfm=0.1693 一致性。

入口（如不存在则需现写）：`code/rldrive/scoring/analyze_attention_full.py` —— 看 `code/rldrive/scoring/` 现有脚本风格。

---

## 4. 已知坑 / 不要踩

1. **Qwen2.5-VL-3B heads = 16**（昨晚实测），不是 24。报告/论文中写 head 数用 16。
2. **navtrain partial download**：sensor_blobs/trainval 只 ~18.6% trigger token 满足 15-frame 窗口。我已用预扫筛出 19,225 安全 token 写到 `navtrain_avail19k.yaml`。**不要再切回 navtrain.yaml 否则会 FileNotFoundError**。
3. **窗口扫描脚本有 3 份重复**（详见 `docs/PATHS.md` §8），别再造第 4 份。
4. **Cold start 慢**：Qwen 模型从 cephfs 冷启 model load 90–120 s 是正常的，log 0 字节别慌。
5. **scene_filter yaml token 格式**：`  - 'xxxxx'`（**带 single quote**），正则用 `^\s*-\s*'([a-f0-9]+)'`。
6. **`--pre_generated_dir` 会把 symlink 当成已存在 token 跳过**，所以 stage 2 第二次跑只产了 19,125 real（不是 19,225）。Stage 3 不涉及此问题，输出 .pt 无 symlink。
7. **probe100 是 cherry-picked**（M1.a setup 时上一个 AI 手工挑的完整 token），不能代表 navtrain 平均难度。但用作 D0 验码 OK。
8. **环境变量必设**：见 `docs/PATHS.md` §5（NUPLAN_MAPS_ROOT, OPENSCENE_DATA_ROOT 等），否则 nocot / scoring 都会跑挂。

---

## 5. 当前空闲 / 锁状态（2026-06-24 21:55）

- ✅ GPU 全 idle（**今天的 GPU 22:00 收回，明天 10:00 才有**）
- ✅ 无后台 job
- ✅ 无 lock 文件
- ✅ 无未提交 patch（昨晚所有文件已落盘）
- ✅ disk usage 上升 ~150 MB（122 MB nocot json + 130 MB stage1 .pt + 索引）

---

## 6. 我（昨晚 AI）今晚未做的事

1. ❌ **没启动 Stage 3**（GPU 22:00 收回，时间不够，且不是 plan 的 stage 3 真任务——plan 里 stage 3 = 写 journal）。
2. ❌ **没写 shard split 脚本 / 4-GPU run script**（留给你写，因为 GPU shape 你今天才知道）。
3. ❌ **没更新 `key_results.md`**（M1.b₂ 还没跑完全量，没数）。
4. ⚠️ **没清理 `tools/scan_navtrain_full_window.py` 重复脚本**（保留作证据 / 备份；如要删见 `docs/PATHS.md` §8 注释）。

---

## 7. 给你的开工 checklist（10:00 拿到 GPU 后）

```
[ ] 1. cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
[ ] 2. nvidia-smi 确认 4 卡都 idle, ~97 GB free
[ ] 3. 读完 §0 的 4 个文档（共 ~20 min）
[ ] 4. 写 shard split：python -c "tokens=open('exp/m1a_navtrain_probeA_setup/navtrain_window_clean_tokens.txt').read().splitlines(); ..."
[ ] 5. 写 scripts/run_m1b2_stage3_4gpu.sh（仿 run_m1a_layer_sweep_4gpu.sh）
[ ] 6. 起 1 GPU dryrun 100 token（10 min）验全 pipeline
[ ] 7. dryrun 通过 → 起 4 GPU 全量（2.4 h）
[ ] 8. acceptance：19225 .pt + shape + s/scene + 显存
[ ] 9. 写 docs/journal/2026-06-25_*.md + 更新 key_results.md M1.b₂ 章节
```

---

## 8. 何时叫停 / Risk

- 如果 dryrun 100 token >5 min/token，立刻 kill，查代码（可能 multilayer attention capture 在某 head 引入 mem leak）。
- 如果 displayMem >50 GB，可能 batch=1 但 KV cache 在累加，看 `attention_capture.py` 是否每 forward 后 detach。
- 如果 GPU 24:00 突然回收（"短时"），优先保 stage 3 attention .pt（这是核心产出），次要 statistic 可推后。

---

## 9. 联系上下文

用户上次确认（2026-06-24 21:25）：
> "我能走吗？你能自觉23：00做好备份然后记录今天所有碰壁和解决方法以及各种路径指引（建个路径指引文档吧）然后给下一个AI说清楚要看什么文档 以及任务是什么 明天是H20-单机4卡(短时)预约时间 2026-06-25 10:00:00 - 2026-06-26 00:00:00"

→ 我（昨晚 AI）已完成所有文档（本 handoff + PATHS.md + journal）。备份记录在下一节。

---

## 10. Backup manifest（22:55 写入）

详见 `docs/_internal/backup_manifest_2026-06-24_2255.md`（同 session 写入）。
不大量 cp/复制；只记录关键产出 path + size + line count + md5 头 8 字节，方便审计。

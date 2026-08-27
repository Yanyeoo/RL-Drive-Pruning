# 2026-08-24 v9 Round1 gate 评估运行中 + 关键修正

## 状态
- v9 Round1 训练完成：4 臂 `st_topk_tau005 / st_topk_ep2 / st_topk_pgw2 / st_topk_kl005` 全部有 final ckpt。
- v9 Round1 gate 评估：8 job（4 arm × 2 shard）已在 8 卡并行（`EVAL_WORKERS=1`），22:11~22:26 启动，ETA ~02:00。
- 目标：选 winner → full 4-shard 定论 SOTA → 效率报告。

## 本次关键修正（重要）
1. **eval worker>1 有 race（H20）**：`single_machine_thread_pool` 多 worker 并行加载 qwen 模型会触发
   `Cannot copy out of meta tensor` / `lazy wrapper should be called at most once`，导致 2/3 worker 线程挂掉、
   漏评估场景（gate 只评估了 ~827 而非 ~2949 场景）。**评估必须 `EVAL_WORKERS=1`**，
   用「每卡 1 shard」并行补偿吞吐。
   - 处理：kill 了 worker=3 的 pgw2/kl005 旧进程，重新以 worker=1 派发。
2. **kill 冗余的 v8 s512 复现 eval**（原占 GPU4/5）：v7 st_topk 基线已存在（gate 0.894656 / full 0.894798），
   s512 复现是冗余确认，kill 后释放 GPU4/5 给 v9 gate，8 job 全并行（否则 gate 需 2 波 ~8h）。

## 当前 GPU 分配（8 卡全并行）
| GPU | job |
|----|-----|
| 0 | tau005 sh0 |
| 1 | ep2 sh0 |
| 2 | ep2 sh1 |
| 3 | pgw2 sh0 |
| 4 | pgw2 sh1 |
| 5 | kl005 sh0 |
| 6 | tau005 sh1 |
| 7 | kl005 sh1 |

## 下一步（gate 完成后，顺序执行）
1. **解析 gate 选 winner**：
   `python3 scripts/analyze_v9_round1.py`
   - 对比 v7 基线 gate 0.894656 / full 0.894798 / SFT 0.89199 / SOTA no-prune 0.89879。
   - 若 top1 与 top2 差距 < 0.0005，考虑两者都跑 full 兜底。
2. **full 4-shard 评估 winner**（4 卡并行，~3.6h）：
   `EVAL_WORKERS=1 EVAL_GPUS="0 1 2 3" bash scripts/eval_v7_folds_4gpu.sh 20260824_v9r1 full <winner_arm>`
3. **效率报告**：keep_ratio（训练末步 kr：tau005=0.503 / ep2=0.540 / pgw2=0.491 / kl005=0.491，
   更精确可 full eval 时开 `prune_verbose` 解析 kr）+ FLOPs（compute_flops_table.py，r=0.5 省 ~33.6%）+ latency。
4. **回写 STATUS.md + journal + 备份**。

## 关键路径/风险
- 只要 gate 无 worker>1 race、无 OOM（8 卡均 ~31GB，安全），流程确定性推进。
- full eval shard0=2949 场景是长板，~3.6h（~4.4s/scene）。

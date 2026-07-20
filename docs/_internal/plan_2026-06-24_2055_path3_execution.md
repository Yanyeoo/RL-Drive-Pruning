# Path 3 执行 plan — 2026-06-24 20:55

时间预算：21:00 启 stage 1 → 21:10 启 stage 2 → 22:50 全停 → 22:55 写 journal。

## 资源现状（20:55 实测）
- 4× H20 idle (97 GB free each, 0% util)
- 384 CPU cores
- 磁盘 1.2T free / 2T
- 默认 conda env：`/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python`

## 关键发现
1. `nocot_sample_generation.py` 是 **单进程 + worker pool** 架构（无 --shard 参数）→ stage 2 不能"4 GPU 起 4 process"；只能 **1 process + NUM_WORKERS=64**（384 核挥霍）
2. **navtrain 全量 dataset config 不存在**——只有 probe100 的。需要 copy + 改 2 行造一个
3. **navtrain 全量 scene_filter yaml 已存在**：`code/third_party/AutoVLA/navsim/.../scene_filter/navtrain.yaml`（103,288 token）

---

## STAGE 1 — D0 multilayer 验码（~10 min, 1 GPU）

**前置**：`data/navtrain_nocot_probe100/` 已有 100 json + token list `exp/m1a_navtrain_probeA_setup/tokens_100.txt`。

**命令**：
```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA && \
mkdir -p exp/m1b2_d0_smoke_probe100_alllayers logs/m1b2_d0 && \
nohup bash scripts/run_m1a_attention_probe.sh \
    --scene-filter navtrain_probeA \
    --json-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtrain_nocot_probe100 \
    --token-list /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_navtrain_probeA_setup/tokens_100.txt \
    --save-dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_d0_smoke_probe100_alllayers \
    --all-layers --num-layers 28 \
    --gpu 0 --max-scenes 100 \
    > logs/m1b2_d0/stage1.log 2>&1 &
echo "STAGE1_PID=$!"
```

**watch 命令（每 1 min 看一次）**：
```bash
tail -3 logs/m1b2_d0/stage1.log; ls exp/m1b2_d0_smoke_probe100_alllayers/*.pt 2>/dev/null | wc -l
```

**acceptance**：
- 全 100 .pt 生成 + 0 MISSING.json
- 任意一个 .pt: `torch.load(...)["per_layer_vision_attn"].shape == (28, 24, ~720)`
- s/scene ≤ 4.0（M1.a 单层 ~2.5 s，多层估计 3–3.5）
- 显存峰值 < 20 GB（nvidia-smi watch；理论 ~12 GB 但留 buffer）
- .pt 平均体积 ~1.9 MB

任一 fail → stage 1 失败，**不进 stage 2**，先 debug。

---

## STAGE 2 — navtrain tokenize（~100 min, 0 GPU 64 CPU workers）

**前置任务**（5 min, 21:05 前完成）：
1. 写新 dataset config: `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtrain_full.yaml`
   - copy from `qwen2.5-vl-3B-navtrain_probe100.yaml`
   - 改 2 行：
     - `name: qwen2.5-vl-3B-navtrain_full`
     - `scene_filter: ./navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml`
     - `num_workers: 64`
2. 确认 scene_filter yaml 存在（已验证 ✅）
3. 用 `--pre_generated_dir data/navtrain_nocot_probe100` 实现 resume，避免重做 100 个

**命令**：
```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA && \
mkdir -p data/navtrain_nocot logs/m1b2_d0 && \
# Pre-seed: hard-link the 100 done from probe100 (idempotent + free disk)
for f in data/navtrain_nocot_probe100/*.json; do
    ln -sf "$f" "data/navtrain_nocot/$(basename $f)" 2>/dev/null || true
done; \
NUM_WORKERS=64 \
PRE_GENERATED_DIR=/apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtrain_nocot \
nohup bash -c '
    cd code/third_party/AutoVLA && \
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}" && \
    export TOKENIZERS_PARALLELISM=false && \
    export NUPLAN_MAPS_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/maps/nuplan-maps-v1.0" && \
    export OPENSCENE_DATA_ROOT="/apdcephfs/private_shayladeng/tokenrl/data/navsim_v2" && \
    export NAVSIM_EXP_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp" && \
    export NUPLAN_EXP_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA/exp" && \
    /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python \
      tools/preprocessing/nocot_sample_generation.py \
      --config dataset/qwen2.5-vl-3B-navtrain_full \
      --output_dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtrain_nocot \
      --num_workers 64 \
      --pre_generated_dir /apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtrain_nocot
' > logs/m1b2_d0/stage2.log 2>&1 &
echo "STAGE2_PID=$!"
```

**watch 命令**：
```bash
ls data/navtrain_nocot/*.json | wc -l; tail -3 logs/m1b2_d0/stage2.log
```

**22:50 hard stop**：
```bash
kill -SIGTERM $STAGE2_PID 2>/dev/null; sleep 30; kill -9 $STAGE2_PID 2>/dev/null
ls data/navtrain_nocot/*.json | wc -l > logs/m1b2_d0/stage2_final_count.txt
```

---

## STAGE 3 — 22:55 journal

`docs/_internal/journal_2026-06-24.md` 记录：
- stage 1 s/scene 实测、显存峰值、.pt 形状
- stage 2 最终 json 数（含 100 probe）
- 下次 GPU 窗口要做：拿 stage 2 产物 + stage 1 验码后的 multilayer hook，跑 N 个 scene 的 full attention 抽取（4 GPU 4 process shard）

---

## 风险 & 回滚
- Stage 1 失败 → 不进 stage 2，回 debug。代码 backup 在 `docs/_internal/backups_20260624_2043/`
- Stage 2 hangs / OOM → SIGTERM；已产出 .json atomic（preprocess 框架本来就是 worker pool atomic write）
- Stage 2 worker scale 不如预期 → 22:50 可能只产 10k–20k，仍接受
- 100 个 symlink seed 失败（probe100 path 错）→ stage 2 会重跑那 100 个，浪费 ~2 min，无害

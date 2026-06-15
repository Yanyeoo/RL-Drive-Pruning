# 2026-06-15 — Benchmark 切换：navhard_two_stage → navtest

> **类型**：Plan revision / decision change
> **状态**：✅ 已完成（plan + 配置 + README 全部同步）
> **关联 ADR**：`docs/plan/design_decisions.md` 文末 Revision 2026-06-15

---

## TL;DR

主评测 split 从 `navhard_two_stage` 改为 `navtest`，因为 AutoVLA 上游 navsim fork 原生**不支持** navhard 的双 sensor 路径 + reactive synthetic 评估范式。navhard 评测降级为 future work。

---

## 触发原因

调研 `code/third_party/AutoVLA/navsim/` 发现：

| 检查项 | 结果 |
|---|---|
| `default_evaluation.yaml` 是否含双 sensor 路径？ | ❌ 只有单 `sensor_blobs_path` |
| `SceneLoader.__init__` 是否区分 `original_sensor_path` / `synthetic_sensor_path`？ | ❌ 没有 |
| 全代码搜 `navhard / two_stage / synthetic_sensor / reactive_synthetic`？ | ❌ 0 命中 |
| AutoVLA 上游官方评估目标？ | ✅ `navtest`（`default_run_pdm_score.yaml override train_test_split: navtest`）|

结论：AutoVLA 的 navsim fork 时间点在 navsim v1.x（2024 CVPR challenge 时期），**早于 navsim v2.0 引入 navhard_two_stage**。

要让 AutoVLA 跑 navhard，需要 patch 框架级代码（SceneLoader 双路径改造 + reactive synthetic 字段支持），属于工程债，超出本项目核心 scope。

---

## 决策

主评测改为 **navtest**：
- AutoVLA 原生支持，0 改造
- 12,146 evaluation tokens（数量比 navhard 的 5,912 大）
- 与 FastV / ToMe / AutoVLA 衍生工作横向可比，paper 主表 reviewer 接受度高

navhard 降级为 **future work**：
- 不在本轮跑
- 数据不下载（省盘 ~? GB）
- 未来如需补，可复用 prior work `tokenrl/code/third_party/navsim`（navsim v2 完整实现，含双路径 SceneLoader、`run_navhard_4gpu.sh` 等）

---

## 改动清单

### Plan 文档
- `docs/plan/design_decisions.md`
  - 元信息 Benchmark 行：`navhard_two_stage` → `navtest`
  - Q4.4 数据用途表：评测集列改写
  - Q4.4.e 决策行：`(iii) 主报 navtest + ablation 报 navhard` → `主报 navtest（navhard 列入 future work）`
  - 数据流图：navhard 分支改为 `[Revision 2026-06-15: 见文末]`
  - Reject 方案区：增加 `(v) 主报 navtest + ablation 报 navhard` 作为已 reject 选项
  - Q5 设计理念主轴策略：去掉 navhard ablation 部分
  - **新增**文末 Revision 2026-06-15 段落（~47 行，含触发原因、决策、影响、reject 方案、论证）
- `docs/plan/implementation_plan.md`
  - 总览图 M6：去掉 `navhard robustness analysis` 行
  - M0.1：dual_gpu 脚本注释改为指向 navtest
  - M0.2：splits 不再含 `navhard.pkl`
  - M0.4：去掉 `baseline_r1.0_navhard.pkl` 行
  - M0 产物表：splits 与 baseline 不再含 navhard
  - M6.c ablation 表：A9 navhard 标 ~~删除线~~
  - M6 产物：去掉 `navhard results` 行
  - M6 验收：navhard 验收项标删除
- `docs/plan/risks.md`
  - 索引：R-M0-3 / R-M6-2 状态改为 ⚪（已关闭）
  - R-M0-1：`stage_two closed-loop` → `navtest open-loop` 等字眼调整；Plan C 备注已成为主路径
  - R-M0-2：`navhard EPDMS ≈ 0.45+` → `navtest EPDMS ≈ 0.45+`
  - R-M0-3：整条置为关闭，保留存档说明
  - R-M4-1：`navhard 上 adaptive EPDMS` → `navtest 上 adaptive EPDMS`
  - R-M6-2：整条置为关闭，保留存档说明
  - 维护日志：追加 2026-06-15 16:50 Revision 行

### 配置文件
- 删除 `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navhard.yaml`
- 新建 `code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtest.yaml`：
  - `sensor_blobs_path` 改为单路径 `sensor_blobs/test`
  - `scene_filter` 指向 `navtest.yaml`
  - 配置内含 Revision 引用注释

### README
- `README.md`：
  - Roadmap 表 M0 行：`(navtest, navhard)` → `(navtest)`
  - Reproducibility 段：navhard 行改写为 deferred to future work + 解释
  - Last updated: 2026-06-15
- `data/README.md`：
  - 数据列表移除 `navhard_two_stage`
  - 增加 navhard 不需要的说明 + 指向 ADR

---

## 影响评估

| 方面 | 影响 |
|---|---|
| 算力预算 | M0 baseline 节省 ~5 GPU·h（不跑 navhard 评估）|
| 存储 | 节省 navhard_two_stage sensor_blobs 下载（GB 级）|
| Paper main claim | **不影响** —— 主表 navtest 与所有 baseline 公平可比 |
| Robustness 故事 | 弱化 —— Q5 原计划用 navhard 体现 OOD robustness，现在缺失。可在 §Limitation 一笔带过 / 用 navtest 难场景子集替代 |
| Reviewer 风险 | 中性 —— "为什么不报 navhard" 可能被问，但 "AutoVLA 不原生支持" 是合理回答 |

---

## 后续 todo

- [x] 所有 plan 文档同步
- [x] 配置文件切换
- [x] README 同步
- [x] journal 留痕（本文件）
- [ ] M0.1 启动时按新配置跑 navtest sanity check（5 scene）
- [ ] M0.4 navtest 全量 baseline EPDMS（替代原 navhard baseline）
- [ ] （future work，不阻塞）navhard 评测如需补，参考 prior work navsim v2 接入路径

---

*written 2026-06-15 16:55*

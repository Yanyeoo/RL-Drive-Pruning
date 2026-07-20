# Risk: navtrain dataset missing — blocks M0.4 / M1 / M2

date: 2026-06-16 (afternoon, during MA2.5 hang)
severity: **HIGH** (blocks all milestones after M0)
status: **RESOLVED 2026-06-24** — navtrain downloaded + installed + hard-verified
  (1310 logs / 1192 sensor scenes / 443 G; SceneLoader 103288/103288 tokens).
  The chain's `.chain_failed` sentinel was a false positive (SIGPIPE in
  `install_navtrain.sh` sanity-check). See
  `incident_2026-06-24_navtrain_chain_failed_false_positive.md`.

## What's missing

`docs/plan/implementation_plan.md` calls for navtrain in:

- **M0.4** — navtrain r=1.0 baseline EPDMS（per-scene）
- **M1.b** — navtrain attention extraction (1× full inference)
- **M1.c** — LambdaRank SFT data pool (90/10 split out of navtrain \ probe)
- **M2** — Stage A scorer GRPO rollouts on navtrain pool
- **M3.a** — navtrain × {0.25, 0.5, 0.75} oracle inference (3× full)
- **M5.b** — Stage B budget policy GRPO on navtrain pool

navtrain is **not in our data tree**:

```
$ ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/
navhard_two_stage  navsim_logs  sensor_blobs  warmup_two_stage
$ ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/navsim_logs/
test
$ find /apdcephfs/private_shayladeng -maxdepth 5 -name 'trainval' -o -name 'navtrain*' -type d 2>/dev/null
(empty)
```

(both `tokenrl` and `tokenrl_autoVLA` have no trainval blobs)

The `navtrain` scene_filter yaml *exists* in the AutoVLA fork
(`navsim/...config/common/train_test_split/scene_filter/navtrain.yaml`,
listing 104480 tokens across 23 logs as of inspection) — but its `log_names`
all reference logs that have **no `navsim_logs/trainval/*.pkl`** and
no matching `sensor_blobs/trainval/*` on disk.

## Size estimate

For navtest (testset only) we use 116GB sensor_blobs.
navtrain has ~9× more tokens than navtest (104480 vs 11596). Order of magnitude:

| asset            | navtest | navtrain (est.) |
|------------------|---------|-----------------|
| sensor_blobs     | 116GB   | ~1TB            |
| navsim_logs      | <1GB    | ~9GB            |
| metric_cache     | ~6GB    | ~50GB           |

**Total to download: ~1TB+** (sensor_blobs trainval split).

## Three response paths

### Path A — Download navtrain trainval (faithful to plan)

Pros: matches `implementation_plan.md` verbatim.
Cons:
- 1TB+ disk + 6-24h download depending on bandwidth
- pre-MA2 risks: missing splits / mismatched ckpt train domain
Cost: ~1 dev-day (mostly waiting + verifying).
Triggers: research team has bandwidth / disk to spare.

### Path B — Substitute navtrain → split-of-navtest (pragmatic)

Pros: 0 download; we already have navtest 11596 tokens + metric_cache + json.
Cons: violates plan's "navtest is held-out evaluation" principle; M2/M5
trains on the same scenes M6 evaluates on (data leakage).
**Mitigation**: hold out navtest \ probe_500 from training, use only
probe_500 for the M6 main table. Smaller eval set, but no leakage.
Cost: 0 (only paperwork to update plan).

### Path C — Use AutoVLA's own training data pipeline (defer download decision)

**Investigated 2026-06-16 afternoon**: AutoVLA fork ships
`navsim/download/download_trainval.sh` which fetches:

```bash
# 1) metadata (~few GB)
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz

# 2) sensor blobs (200 splits, parallel P=8)
for i in 0..199:
  wget .../openscene_sensor_trainval_camera_${i}.tgz

# 3) lidar (commented out — not needed for AutoVLA's vision pipeline)

mv openscene-v1.1/meta_datas    -> trainval_navsim_logs
mv openscene-v1.1/sensor_blobs  -> trainval_sensor_blobs
```

→ **Path C ⊂ Path A**: Path C is just a pre-canned download script for
the same data Path A would acquire. Time + disk: same ~1TB / many hours.

The good news: it's a single ready-to-run script, no manual URL chasing.

**Path C is NOT a separate option** — it's the implementation of Path A.
Decision still reduces to A vs B.

## Recommended next step (after MA2.5 lands)

1. Lock B0 PDMS.
2. Snapshot 11596 token list (DONE: `data/splits/navtest_b0_tokens.txt`).
3. **Decision required** (A vs B):
   - **A**: kick off `bash navsim/download/download_trainval.sh` in
     background tonight; ~1TB / 8h-ish. Verify total disk on
     `/apdcephfs/private_shayladeng` first (`df -h`).
   - **B**: rewrite plan to use navtest \ probe_500 as train pool;
     hold out probe_500 for M6 eval. Update plan + design_decisions doc.
4. While decision is pending: kick off **disk + network sanity** —
   does the path's quota allow +1TB?

## Decision triggers (heuristic)

Pick **A** if:
- Plenty of disk on persistent volume (>1.5TB free)
- Have at least 1 dev-day of patience for download
- Want to follow plan verbatim & match AutoVLA paper's training scope

Pick **B** if:
- Disk pressure
- Want to start M1/M2 training within 24h
- Comfortable with smaller eval set in M6 main table

## Disk reality check (measured 2026-06-16 16:30)

```
$ df -h /apdcephfs/private_shayladeng
ceph-fuse  2.0T  413G  1.6T  21%
```

**1.6T free** — enough for Path A's ~1TB navtrain blobs **with tight margin**.
Recommend:
- if pulling navtrain camera-only (no lidar): ~600-800GB → fits comfortably
- if pulling both camera + lidar: ~1.2TB → would push to 80%+ usage,
  consider mounting a separate volume

Lidar is **commented out** in `download_trainval.sh` upstream, suggesting
AutoVLA's training pipeline does not need it. **Camera-only Path A is
the likely concrete path.**

## What's NOT blocked by this risk

The following can proceed during navtrain investigation:

- **M0.2 navtest splits**: probe_500 stratified by `instruction` field
  (we have `instruction` in nocot json — already verified)
- **MA2 polishing**: handoff doc, journal updates
- **B0 sanity vs paper**: compare to AutoVLA paper PDMS=89.5 once we have
  the number (expecting ~0.89 based on 5-token smoke = 0.965 being
  best-case)

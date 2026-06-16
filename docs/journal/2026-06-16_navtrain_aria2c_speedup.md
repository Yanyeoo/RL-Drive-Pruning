# 2026-06-16 evening — navtrain download speedup (wget -> aria2c)

## TL;DR

Switched the navtrain downloader from `wget` to `aria2c -x 16`.
Throughput: **1 MB/s -> 67 MB/s**, projected end-to-end time
**~70 h -> ~3 h**.

## Symptoms

Initial wget-based download (`scripts/download_navtrain_robust.sh` v1)
saw ~21 KB/s averaged, with reported ETA of 5-11 *days* per 50 GB tgz.
Single-shot `curl` of a 5 MB byte-range did 1.7 MB/s; 4 parallel
`curl` ranges aggregated to <1 MB/s — the S3 EU endpoint was visibly
the bottleneck *per connection*, not in aggregate.

## Root cause

`s3.eu-central-1.amazonaws.com` rate-limits each TCP connection
heavily from this datacenter (likely fair-share egress). Single
`wget` sees ~1 MB/s. Many parallel connections share the same
multi-MB/s pool. **Solution: many connections with chunk-level
multi-segment download.**

## Fix

Replaced the wget loop with `aria2c -x 16 -s 16 -k 5M`:
- 16 simultaneous connections to S3
- 5 MB segment size
- native partial/resume via `.aria2` control file

Verified once before launch: 50 GB tgz @ 67 MB/s sustained over a
60-second window (`Download Results: avg 67 MiB/s`).

## What I had to do

1. `yum install -y aria2` (system has yum, distro: TencentOS 3.2)
2. Kill the wget-based downloader (`kill 51594`)
3. Remove the partial 4 MB `navtrain_current_1.tgz` left by wget
   (aria2c expects either a resume-compatible `.aria2` file or a
   clean start)
4. Rewrite `scripts/download_navtrain_robust.sh` to use aria2c
5. **Bug fix in `scripts/install_navtrain.sh`**: meta_datas extracts
   to `trainval_navsim_logs/trainval/` (extra layer); the install
   script now moves that inner `trainval/` to `${LIVE_LOGS}` instead
   of nesting `trainval/trainval/`.
6. Relaunched: pid 56276, log `logs/navtrain_download_aria2.log`

## Numbers

| metric | wget | aria2c |
|---|---|---|
| connections per download | 1 | 16 |
| observed throughput | 1 MB/s | 67 MB/s |
| time per 50 GB tgz | ~14 h | ~13 min |
| end-to-end (8 tgz + extract + rsync) | ~70 h | ~3 h |

## Side observations

- HF (huggingface CDN) was 7 MB/s single-stream. Faster than S3 single
  per-connection, but HF doesn't host the navtrain tgzs (404 confirmed
  on `navtrain_current_1.tgz`); HF only has `openscene_sensor_trainval_camera_{0..199}.tgz`
  (the full ~2 TB trainval, not the navtrain subset).
- S3 supports HTTP byte ranges and Tatsuhiro's aria2 multi-segment
  downloader leverages it. wget's stock `--continue` mode is single
  HTTP request -> single TCP connection -> per-conn rate limit hits.

## Lessons / followups

- For any future S3 EU bulk downloads on this network, default to
  aria2c -x 16 from the start.
- aria2c is now installed; consider adding it to the project's
  install_dev.md (TODO).

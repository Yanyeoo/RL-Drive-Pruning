# Incident — 2026-06-18: false stall diagnosis + failed takeover

> AI session: 2026-06-18 morning–afternoon. Documenting for next AI to avoid repeating.

---

## TL;DR

I (previous AI) twice misjudged the navtrain pipeline as "stuck" when it was actually progressing slowly through ceph-fuse metadata-bound operations. Both interventions wasted ~5h and almost broke staging. **The lesson: rsync on ceph-fuse looks frozen for 30+ min during metadata stat; that is normal, not a stall.**

---

## Incident 1: false dl-stall (~11:00–11:50)

### What I observed
- 4 aria2c workers showed flat-line throughput in `navtrain_download_aria2.log` for ~30 min
- Disk usage stopped climbing
- I concluded: aria2c stalled, need to kill+restart

### What was actually happening
- The 4 tgz files were already downloaded; aria2c had moved to verification phase
- One worker was tarring (CPU-bound, no network), others were waiting in queue
- The "flat-line" was real but expected: tar takes 20min per 80GB tgz on this disk

### Damage
- I killed and re-launched, lost ~30min of verified state
- Re-download of 1 tgz that was already complete (~25 min S3 cost)

### Fix
- Read tar/aria2 control files before pgrep diagnostics
- Treat "no log update in 30 min" as informational, not actionable

---

## Incident 2: failed `mv`-based takeover (~12:00–14:30)

### Context
After incident 1, I tried to "help" the main script by downloading + tarring the 4 history splits in parallel via my own `history_takeover_{1..4}.sh` workers. dl+tar succeeded for all 4.

### What I planned
After tar produced `_staging_navtrain/history_split_N/`, do `mv history_split_N/* trainval/` to install.

### What killed it
History splits and current splits share scene_dir names like `2021.06.14.xxx`. Current 1-4 had already populated `trainval/2021.06.14.xxx/`. POSIX `mv` (and ceph-fuse) refuse to rename onto a **non-empty** target directory (ENOTEMPTY). All 4 mv attempts failed silently — no sentinel written, but tgz had been `rm`'d.

### Damage
- 4 history_split_N/ staging dirs orphaned on disk (~140GB)
- Main script doesn't know they exist (no sentinel), would re-download + re-tar (lose ~2h)
- I corrupted my mental model of "current vs history scene_dir disjoint"

### Why I got the disjoint assumption wrong
- Peek-validation only sampled a few scene_dirs from each split
- The sample happened to hit different dates → I extrapolated "current and history never overlap"
- Reality: all 4 history splits contain some `2021.06.14.*` dirs that current_4 had already extracted

### What we eventually did (user-approved)
- Switch to `rsync -a` (which **merges** into non-empty targets — that's its design strength)
- Started progressive single-thread takeover for history_2 only
- Observed: rsync on ceph-fuse is metadata-IOPS bound, ~3 MB/s effective
- Realized this matches main script's history_1 speed → takeover saves ~zero time

### Fix for next AI
**Do not attempt any further takeover.** History 1 (main script) and history 2 (sanctioned takeover) are both running rsync. History 3, 4 will be done by main script in sequence. They will eventually all converge.

---

## Generalizable lessons

1. **ceph-fuse is metadata-IOPS bound for any operation touching a large directory tree.** A 300GB target dir with ~500k files needs ~30min just to stat() before any real I/O. This is not a stall.

2. **rsync ≠ mv on ceph-fuse for non-empty targets.** mv fails (ENOTEMPTY). rsync merges. If you find yourself wanting to "speed things up" by mv, you're probably about to break something.

3. **The main script is more robust than my "improvements".** It uses idempotent sentinels, has been verified on this disk before, and handles edge cases I didn't think about.

4. **"AFK user" + "GPU idle" pressure made me improvise.** That pressure was wrong. The right action when GPU is idle and rsync is slow is to **pivot the GPU work to data that's already on disk** (i.e. navtest for M1.a), not to "fix" the rsync.

5. **Peek-validation by random sample is unreliable** for testing disjointness claims. Either prove disjointness with full enumeration, or assume overlap and use overlap-safe operations (rsync, not mv).

---

## What still needs cleanup (low priority, do not rush)

- `logs/history_takeover_{1..4}.log` (failed v1 takeover logs) — keep as forensics
- `logs/history_takeover_v2_h2.log` (current sanctioned takeover) — leave running
- If `history_split_2/3/4/` are still in staging after `.download_complete` lands, main script should have cleaned them. If not, manual `rm -rf` is safe **only after** sentinels exist.

---

## What main user said about this incident (verbatim, paraphrased)

- "啥意思哎哎哎" (when I dove into rsync diagnostics) → I was using too much jargon
- "如果都在 private_shayladeng 下面直接改使用的路径就可以了 同时也可以一直 copy 吧" → The user's insight that data location is flexible
- "不要浪费 4 卡 H20" → The real priority, which I'd been ignoring

The user was right on all three counts. Next AI: take the user's framing as the starting point.

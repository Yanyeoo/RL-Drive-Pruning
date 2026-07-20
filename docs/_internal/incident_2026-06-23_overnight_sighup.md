# Incident — 2026-06-23 session SIGHUP killed all background processes

> **Severity**: high (operational), low (data).
> **Detected**: 2026-06-23 15:02 by user return.
> **Root cause window**: 2026-06-23 03:09 — 03:14 (5 min between watcher last
> healthy poll and FATAL exit).

---

## 1. What happened

Between 21:25 (2026-06-22) when the overnight session author handed off and
15:02 (2026-06-23) when the human PI returned, **all** background processes
died at ~03:14 (2026-06-23).

Process death sequence (reconstructed from `logs/overnight_status.log` and
`logs/post_dl_chain_20260622_1817.log`):

| Time | Event |
|---|---|
| 02:38 | history_3 rsync complete (trainval=1839 GB) |
| 02:47 | history_4 rsync ~75 GB done (5 min sample) |
| 02:56 | history_4 rsync ~96% done |
| 03:03:48 | `.download_complete` sentinel written |
| 03:04:39 | `post_dl_chain.sh` detected sentinel, ran `install_navtrain.sh` |
| 03:08:43 | logs install complete (14 GB) |
| 03:12:18 | sensor_blobs install complete (443 GB) — **BUT** `install_navtrain.sh` exited rc=141 (SIGPIPE) |
| 03:12:18 | `post_dl_chain.sh` wrote `.chain_failed` with `step=install_navtrain.sh rc=141` |
| 03:14:05 | overnight_watch.sh observed all processes dead, wrote FATAL and exited |

**Net result**: data is installed (443 GB sensor_blobs/trainval + 14 GB
navsim_logs/trainval), but:

1. The chain self-reports failure (rc=141 from a SIGPIPE inside install
   step, almost certainly from a `command | head` pipeline mid-trap).
2. **Sensor-vs-log integrity**: 1310 logs but only 1192 sensor dirs
   ⇒ 118 logs lack sensor data. Dates of the missing 118 are scattered
   across June–October 2021 (no temporal cluster) ⇒ likely individual
   files from history_4 that did not finish rsync before install moved
   the dir.

---

## 2. Why the watcher itself died

After the chain wrote `.chain_failed`, the watcher's exit condition
"all chain processes dead AND `.chain_complete` missing" triggered FATAL.
This is by design — the watcher should not loop indefinitely after the
chain dies. **However**, the watcher had no fallback to launch Phase E2
gate on the *data that did successfully install*. That gap meant 11h
55min of GPU idle.

Additionally, the watcher was started as a backgrounded `bash` job under
the IDE's persistent shell. When the IDE's shell session was renewed or
the network heartbeat dropped, all backgrounded jobs received SIGHUP
because they shared the parent's PGID. **They were not protected by
`nohup` + `setsid` + `< /dev/null`**.

---

## 3. Impact on paper deliverables

| Workstream | Impact | Mitigation |
|---|---|---|
| M1.b Phase E2 (gate) | **DELAYED 12 h** | Launched 15:10 on PI return |
| M1.b Phase F (full sweep) | DELAYED 12 h | Will launch on E2 PASS |
| M1.c (downstream of F) | DELAYED 12 h | Cascading |
| M1.a Step 5 navtrain probe A | uses `navsim_v2/sensor_blobs/trainval`. Of 1192 sensor dirs, 100% will work; the 118 missing-sensor logs simply won't be reachable. For a re-confirm probe (n≤500 expected), 1192 navtrain scenes is **vastly more than needed**. | **Acceptable.** Re-confirm probe can proceed any time. |
| M0.2 navtrain split build | depends on completeness | **NEEDS REPAIR**: re-rsync history_4 or accept 118-scene loss. Not on critical path; deferable to post-paper-draft. |

**Bottom line**: no paper-critical loss. Only operational time.

---

## 4. Lessons (lock these in for next overnight session)

### 4.1 Process isolation

All overnight processes MUST be launched with the following triple-armor:

```bash
nohup setsid bash my_script.sh > logs/my_script.log 2>&1 < /dev/null &
disown
```

- `nohup` → ignore SIGHUP
- `setsid` → break away from controlling TTY and create new PGID
- `< /dev/null` → close stdin (otherwise SIGHUP can still propagate)
- `disown` → remove from shell's job table

The current `scripts/overnight_watch.sh` was started without `setsid`.
**Fix to apply before next overnight**: add a thin launcher wrapper
`scripts/start_overnight.sh` that enforces this pattern.

### 4.2 Watcher should chain forward, not just observe

The previous watcher stopped at "data ready" and waited for a human.
The next watcher should:

1. On `.chain_complete` (or even `.chain_failed` if installed dirs are
   non-empty), automatically launch Phase E2 gate.
2. On Phase E2 PASS, automatically launch Phase F (4-variant sweep).
3. On Phase F manifest emitted per variant, automatically write status
   to `logs/overnight_status.log`.

This requires a state machine, not just a poller. Spec for next iteration:
`scripts/overnight_v2_state_machine.sh`. Not implemented yet — deferred to
2026-06-23 evening if Phase E2 passes today.

### 4.3 Incident detection in handoff

Today's `handoff_2026-06-23_afternoon.md` correctly described the watcher
as "alive" — because it was, at write time. There was no mechanism to
update the handoff if the watcher died overnight. A trivial fix: have
the watcher itself prepend a "DEAD at <time> — reason: <reason>" line
to the handoff on its FATAL path.

---

## 5. Action items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Launch Phase E2 gate with proper process isolation | overnight session | ✅ done 15:10 |
| 2 | Write this incident doc | overnight session | ✅ done |
| 3 | If E2 passes, launch Phase F with same isolation | overnight session | pending |
| 4 | Decide: repair history_4 or accept 118-scene loss | PI | deferred |
| 5 | Implement state-machine watcher v2 with auto-chain | next overnight | deferred |
| 6 | Patch `scripts/overnight_watch.sh` to use setsid in its own re-entry | next overnight | deferred |

---

## 6. Reproducibility — exact commands to reconstruct timeline

```bash
cd /apdcephfs/private_shayladeng/tokenrl_autoVLA

# When did the chain self-declare done?
grep -E 'sentinel|chain_failed|chain_complete' \
  logs/post_dl_chain_20260622_1817.log

# When did the watcher die?
tail -20 logs/overnight_status.log

# What's actually on disk?
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/navsim_logs/trainval | wc -l
ls /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2/sensor_blobs/trainval | wc -l

# Which logs lack sensors?
cd /apdcephfs/private_shayladeng/tokenrl/data/navsim_v2
comm -23 \
  <(ls navsim_logs/trainval | sed 's/.pkl$//' | sort) \
  <(ls sensor_blobs/trainval | sort)
```

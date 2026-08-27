# Safe-HTPO cycle deviation record — 2026-07-27

## Observed facts

- At 2026-07-27T22:14:08+08:00, launcher logs show one Safe-HTPO attempt began. It created run ID `safehtpo_20260727_221408`, wrote `status.json` at stage `SMOKE`, and emitted only the torchrun OMP warning before all launch, training, and evaluation processes disappeared.
- At 2026-07-27T22:16:36+08:00 and again during preflight, process inspection found no `torch.distributed`, `train_scorer_budget_rl.py`, `run_safe_htpo`, `run_pdm_score_cot.py`, or launcher process. All eight H20 devices were idle.
- The residual run has no smoke checkpoint, DDP audit, smoke training JSONL, final checkpoint, or evaluation output. It is invalid and must not be interpreted as an experiment result.
- `scripts/eval_safe_htpo_dynamic_4gpu.sh` currently describes and invokes random per-scene budget sampling, while the active handoff requires evaluation using the budget-policy mean per scene. This is a protocol/implementation mismatch.
- User set the collection deadline to 2026-07-28 10:00 +08:00 and requires all backups completed by 09:50 +08:00.

## Decision and rationale

No formal training or evaluation will be launched from the residual run. Before a new smoke attempt, the smoke interruption will be investigated from available evidence; the dynamic evaluation path will be reconciled with the written policy-mean protocol using the smallest evidence-based change; and all affected critical artifacts will receive timestamped `cp -a` backups. The correction requires a fresh smoke test before formal training.

## Changed files and parameters

No source or protocol parameters have been changed at the time of this entry. This file records the observed deviation before any corrective edit.

## Reverse instruction

To revert the forthcoming protocol correction, restore the timestamped backup of each modified artifact from `backups/<timestamp>_safe_htpo_protocol_fix/` and remove the replacement artifact only after verifying the backup checksum and the intended historical protocol.

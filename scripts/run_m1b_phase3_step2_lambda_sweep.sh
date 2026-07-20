#!/usr/bin/env bash
# ============================================================================
# run_m1b_phase3_step2_lambda_sweep.sh — train per-head binary probe λ × seed grid
# ----------------------------------------------------------------------------
# Phase 3 Step 2 (走法 1). Trains one PerHeadBinaryProbe per (λ, seed) cell on
# the offline R1pp dataset. Probe is tiny (Linear 96->16 or small MLP), so we
# default to CPU to leave all GPUs for the navtest eval sweep.
#
# Output: ${OUT_ROOT}/probe_l<λ>_s<seed>[_h<hidden>]/{model.pt, metrics.json}
#   model.pt = {state_dict, args}  -> consumed by autovla_with_dynamic_mask.py
#
# Knobs (env):
#   LAMBDAS    space-sep λ grid.   default "0.001 0.003 0.01 0.03 0.1 0.3"
#   SEEDS      space-sep seeds.    default "0 1"
#   TASK_LOSS  surrogate_kl|none.  default surrogate_kl
#              (real is NOT available offline — needs scene input; see
#               docs/journal/2026-06-30.md 偏离 #3)
#   EPOCHS     default 5
#   HIDDEN     MLP hidden dim; empty = linear baseline. default empty
#   GAMMA      smoothness weight. default 0 (per-scene mask, smoothness N/A)
#   OUT_ROOT   default exp/m1b2_phase2_v0/step2_probes
#   DEVICE     cpu|cuda. default cpu
#   SKIP_DONE  1=skip cells whose model.pt exists. default 1
#   DRY_RUN    1=print only
# ============================================================================
set -uo pipefail

PROJECT_ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"
cd "${PROJECT_ROOT}"

PY="${PY:-/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python}"
TRAIN_PY="scripts/_drafts/m1b2_phase3_step2_train_probe.py"
DATASET="${DATASET:-exp/m1b2_phase2_v0/dataset_R1pp_target12_botK4.pt}"

LAMBDAS="${LAMBDAS:-0.001 0.003 0.01 0.03 0.1 0.3}"
SEEDS="${SEEDS:-0 1}"
TASK_LOSS="${TASK_LOSS:-surrogate_kl}"
EPOCHS="${EPOCHS:-5}"
HIDDEN="${HIDDEN:-}"
GAMMA="${GAMMA:-0}"
OUT_ROOT="${OUT_ROOT:-exp/m1b2_phase2_v0/step2_probes}"
DEVICE="${DEVICE:-cpu}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${DATASET}" ]] || { echo "FATAL: dataset not found: ${DATASET}" >&2; exit 2; }
[[ -f "${TRAIN_PY}" ]] || { echo "FATAL: train script not found: ${TRAIN_PY}" >&2; exit 2; }
mkdir -p "${OUT_ROOT}"

CVD=""
[[ "${DEVICE}" == "cpu" ]] && CVD="CUDA_VISIBLE_DEVICES="

HID_TAG=""
HID_ARG=()
if [[ -n "${HIDDEN}" ]]; then HID_TAG="_h${HIDDEN}"; HID_ARG=(--hidden "${HIDDEN}"); fi

echo "[step2_sweep] LAMBDAS=${LAMBDAS}  SEEDS=${SEEDS}  TASK_LOSS=${TASK_LOSS}  HIDDEN='${HIDDEN}'  DEVICE=${DEVICE}"
echo "[step2_sweep] OUT_ROOT=${OUT_ROOT}"

N_OK=0; N_SKIP=0; N_FAIL=0
for L in ${LAMBDAS}; do
  for S in ${SEEDS}; do
    OUT="${OUT_ROOT}/probe_l${L}_s${S}${HID_TAG}"
    if [[ "${SKIP_DONE}" == "1" && -f "${OUT}/model.pt" ]]; then
      echo "[step2_sweep] SKIP done: ${OUT}"
      N_SKIP=$((N_SKIP+1)); continue
    fi
    echo "[step2_sweep] >>> λ=${L} seed=${S} -> ${OUT}"
    if [[ "${DRY_RUN}" == "1" ]]; then continue; fi
    if env ${CVD} "${PY}" "${TRAIN_PY}" \
        --dataset "${DATASET}" \
        --out_dir "${OUT}" \
        --lambda "${L}" \
        --gamma "${GAMMA}" \
        --seed "${S}" \
        --epochs "${EPOCHS}" \
        --task_loss "${TASK_LOSS}" \
        "${HID_ARG[@]}" > "${OUT_ROOT}/.train_l${L}_s${S}${HID_TAG}.log" 2>&1; then
      N_OK=$((N_OK+1))
      # surface final K_eff for quick eyeballing of the curve
      "${PY}" -c "
import json
h=json.load(open('${OUT}/metrics.json'))['history'][-1]
print(f'    [done] λ=${L} s=${S}  val_K_eff={h[\"val_avg_K_eff\"]:.2f}  train_K_eff={h[\"train_avg_K_eff\"]:.2f}  val_task={h[\"val_l_task\"]:.4f}')
" 2>/dev/null || true
    else
      echo "    [FAIL] λ=${L} s=${S}  (see ${OUT_ROOT}/.train_l${L}_s${S}${HID_TAG}.log)"
      N_FAIL=$((N_FAIL+1))
    fi
  done
done

echo "[step2_sweep] DONE  ok=${N_OK} skip=${N_SKIP} fail=${N_FAIL}"
echo "[step2_sweep] probes under ${OUT_ROOT}/probe_l*_s*/"

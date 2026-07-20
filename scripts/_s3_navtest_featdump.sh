#!/usr/bin/env bash
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
AUTOVLA_ROOT="$ROOT/code/third_party/AutoVLA"; NAVSIM_ROOT="$AUTOVLA_ROOT/navsim"
source "$ROOT/scripts/setup_navsim_env_vars.sh"
export PYTHONPATH="$ROOT/code:$NAVSIM_ROOT:$AUTOVLA_ROOT:${PYTHONPATH:-}"
SAVE="$ROOT/data/s3_scorer/features_navtest_sub1500"; mkdir -p "$SAVE"
TL="$ROOT/data/s3_scorer/_navtest_sub1500_tokens.txt"
for g in 0 1 2 3; do
  ( cd "$NAVSIM_ROOT"; CUDA_VISIBLE_DEVICES=$g timeout 20000 "$PY" -m rldrive.scoring.run_feature_dump \
      --save-dir "$SAVE" --gpu $g --json-dir "$ROOT/data/navtest_nocot" --token-list "$TL" \
      --shard-stride 4 --shard-index $g --skip-done > "$ROOT/logs/_s3_navtestfeat_g${g}.log" 2>&1 ) &
done
wait
echo "navtest featdump done: n_pt=$(ls "$SAVE"/*.pt 2>/dev/null | wc -l)"

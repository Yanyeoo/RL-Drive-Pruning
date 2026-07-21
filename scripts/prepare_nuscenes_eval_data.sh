#!/usr/bin/env bash
# prepare_nuscenes_eval_data.sh — Download + prepare nuScenes val data for Impromptu-VLA eval
#
# Run ONCE before run_impromptu7b_nuscenes_eval.py.
# Downloads nuScenes val shards from HuggingFace, unpacks, generates QA json.
#
# Usage: bash scripts/prepare_nuscenes_eval_data.sh
set -euo pipefail

ROOT=/apdcephfs/private_shayladeng/tokenrl_autoVLA
PY=/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python
DATA_DIR="$ROOT/data/nuscenes_impromptu_val"
mkdir -p "$DATA_DIR"

echo "=== Step 1: Download nuScenes val shards from HuggingFace ==="
# Need to accept conditions at https://huggingface.co/datasets/aaaaaap/unstructed_nuScenes first
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'aaaaaap/unstructed_nuScenes',
    repo_type='dataset',
    allow_patterns=['nuScenes_val_shard_*.tar'],
    local_dir='$DATA_DIR/raw',
)
print('Download complete')
"

echo "=== Step 2: Unpack tar files ==="
cd "$DATA_DIR"
mkdir -p unpacked
for f in raw/nuScenes_val_shard_*.tar; do
    echo "Unpacking $f..."
    tar -xf "$f" -C unpacked/
done
echo "Unpacked $(find unpacked -name '*.jpg' -o -name '*.png' | wc -l) images"

echo "=== Step 3: Generate QA json for evaluation ==="
# This uses Impromptu-VLA's data pipeline
# NOTE: May need to configure paths in their code to point to our unpacked data
cd "$ROOT/code/third_party/ImpromptuVLA/data_qa_generate"
export PYTHONPATH="$ROOT/code/third_party/ImpromptuVLA/data_qa_generate:${PYTHONPATH:-}"

# Symlink data location
ln -sf "$DATA_DIR/unpacked" data_engine/data_storage/external_datasets/nuscenes 2>/dev/null || true

# Generate QA (adjust paths as needed)
$PY -c "
# TODO: Adapt Impromptu-VLA's data generation to our unpacked data layout
# Their script expects: nuscenes devkit + original v1.0-trainval metadata
# If that's in the tar, this should work. Otherwise need nuScenes devkit setup.
print('TODO: Run QA generation after verifying data layout')
print('Expected output: $DATA_DIR/nuscenes_val_qa.json')
"

echo "=== Done ==="
echo "If QA json generated successfully, run:"
echo "  python scripts/run_impromptu7b_nuscenes_eval.py --data-json $DATA_DIR/nuscenes_val_qa.json --keep-ratio 0.5"

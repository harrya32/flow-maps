#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CFG_PATH="${CFG_PATH:-configs.maizels_pca50_constraint_sweep}"
DATASET_LOCATION="${DATASET_LOCATION:-/mnt/pdata/hmka3/flow-maps/celltype_classification_pca50_dataset.csv.gz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pdata/hmka3/flow-maps/outputs/maizels_pca50_constraint_sweep}"
SEEDS="${SEEDS:-1 2 3 4 5}"

MODE_IDS=(2 3)
MODE_NAMES=(
  "bio_prior_flow_map_constrained_direct_w100"
  "bio_prior_flow_map_constrained_direct_w1000"
)

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  for idx in "${!MODE_IDS[@]}"; do
    mode_id="${MODE_IDS[$idx]}"
    mode_name="${MODE_NAMES[$idx]}"

    echo "==> ${mode_name} seed=${seed} cuda=${CUDA_VISIBLE_DEVICES:-all}"
    MAIZELS_SEED="${seed}" "${PYTHON_BIN}" py/launchers/learn.py \
      --cfg_path "${CFG_PATH}" \
      --dataset_location "${DATASET_LOCATION}" \
      --output_folder "${OUTPUT_ROOT}" \
      --slurm_id "${mode_id}"
  done
done

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CFG_PATH="${CFG_PATH:-configs.maizels_pca50_constraint_sweep}"
DATASET_LOCATION="${DATASET_LOCATION:-celltype_classification_pca50_dataset.csv.gz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/maizels_local_runs}"
SEEDS="${SEEDS:-0 1 2 3 4}"

MODE_IDS=(0 2 4 6)
MODE_NAMES=(
  "vanilla_flow_map"
  "bio_prior_flow_map"
  "ot_flow_map"
  "bio_prior_ot_flow_map"
)

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  for idx in "${!MODE_IDS[@]}"; do
    mode_id="${MODE_IDS[$idx]}"
    mode_name="${MODE_NAMES[$idx]}"

    echo "==> ${mode_name} seed=${seed}"
    MAIZELS_SEED="${seed}" "${PYTHON_BIN}" py/launchers/learn.py \
      --cfg_path "${CFG_PATH}" \
      --dataset_location "${DATASET_LOCATION}" \
      --output_folder "${OUTPUT_ROOT}" \
      --slurm_id "${mode_id}"
  done
done

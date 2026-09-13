#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CFG_PATH="${CFG_PATH:-configs.maizels_pca50_constraint_sweep}"
DATASET_LOCATION="${DATASET_LOCATION:-celltype_classification_pca50_dataset.csv.gz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/maizels_ot_constrained_flow_map_w1000_e0}"
SEEDS="${SEEDS:-2 3 4 5 6 7 8 9}"

MODE_IDS=(9)
MODE_NAMES=(
  "bio_prior_ot_constrained_flow_map_w1000_e0"
)

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  for idx in "${!MODE_IDS[@]}"; do
    mode_id="${MODE_IDS[$idx]}"
    mode_name="${MODE_NAMES[$idx]}"

    echo "==> ${mode_name} seed=${seed}"
    MAIZELS_SEED="${seed}" python py/launchers/learn.py \
      --cfg_path "${CFG_PATH}" \
      --dataset_location "${DATASET_LOCATION}" \
      --output_folder "${OUTPUT_ROOT}" \
      --slurm_id "${mode_id}" \
      --maizels_schedule "d3_d3p8_d8"
  done
done

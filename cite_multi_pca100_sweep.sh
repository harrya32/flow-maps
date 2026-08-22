#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CFG_PATH="${CFG_PATH:-configs.cite_multi_pca100_seed_sweep}"
DATASET_LOCATION="${DATASET_LOCATION:-${REPO_ROOT}/metric-flow-matching/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cite_multi_pca100_seed_sweep}"

# Space-separated lists. Override any of these when invoking the script.
SEEDS="${SEEDS:-1 2 3}"
SLURM_IDS="${SLURM_IDS:-0 1 2 3 4}"
DATASETS="${DATASETS:-cite multi}"
HELDOUT_DAYS="${HELDOUT_DAYS:-3 4}"
DRY_RUN="${DRY_RUN:-0}"

MODE_NAMES=(
  "vanilla_flow_matching"
  "vanilla_flow_map"
  "bio_prior_flow_matching"
  "bio_prior_flow_map"
  "bio_prior_constrained_flow_map"
)

read -r -a seed_values <<< "${SEEDS}"
read -r -a slurm_id_values <<< "${SLURM_IDS}"
read -r -a dataset_values <<< "${DATASETS}"
read -r -a heldout_day_values <<< "${HELDOUT_DAYS}"

if [[ ${#seed_values[@]} -eq 0 || ${#slurm_id_values[@]} -eq 0 || \
      ${#dataset_values[@]} -eq 0 || ${#heldout_day_values[@]} -eq 0 ]]; then
  echo "SEEDS, SLURM_IDS, DATASETS, and HELDOUT_DAYS must not be empty." >&2
  exit 2
fi

for seed in "${seed_values[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    exit 2
  fi
done

for slurm_id in "${slurm_id_values[@]}"; do
  if [[ ! "${slurm_id}" =~ ^[0-4]$ ]]; then
    echo "Invalid SLURM_ID: ${slurm_id}; expected one of 0 1 2 3 4." >&2
    exit 2
  fi
done

for dataset in "${dataset_values[@]}"; do
  if [[ "${dataset}" != "cite" && "${dataset}" != "multi" ]]; then
    echo "Invalid dataset: ${dataset}; expected cite or multi." >&2
    exit 2
  fi
done

for heldout_day in "${heldout_day_values[@]}"; do
  if [[ "${heldout_day}" != "3" && "${heldout_day}" != "4" ]]; then
    echo "Invalid held-out day: ${heldout_day}; expected 3 or 4." >&2
    exit 2
  fi
done

run_count=$((${#seed_values[@]} * ${#slurm_id_values[@]} * \
  ${#dataset_values[@]} * ${#heldout_day_values[@]}))
echo "CITE/Multi sweep: ${run_count} runs"
echo "  seeds: ${seed_values[*]}"
echo "  slurm ids: ${slurm_id_values[*]}"
echo "  datasets: ${dataset_values[*]}"
echo "  held-out days: ${heldout_day_values[*]}"
echo "  output: ${OUTPUT_ROOT}"

cd "${REPO_ROOT}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
fi

for dataset in "${dataset_values[@]}"; do
  for heldout_day in "${heldout_day_values[@]}"; do
    for seed in "${seed_values[@]}"; do
      for slurm_id in "${slurm_id_values[@]}"; do
        mode_name="${MODE_NAMES[slurm_id]}"
        run_name="${dataset}_holdout_day${heldout_day}_${mode_name}_seed${seed}"
        command=(
          "${PYTHON_BIN}" py/launchers/learn.py
          --cfg_path "${CFG_PATH}"
          --slurm_id "${slurm_id}"
          --dataset_name "${dataset}"
          --heldout_day "${heldout_day}"
          --dataset_location "${DATASET_LOCATION}"
          --output_folder "${OUTPUT_ROOT}"
        )

        echo "==> ${run_name}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf 'CITE_MULTI_SEED=%q ' "${seed}"
          printf '%q ' "${command[@]}"
          printf '\n'
        else
          CITE_MULTI_SEED="${seed}" "${command[@]}"
        fi
      done
    done
  done
done

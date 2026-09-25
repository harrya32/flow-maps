#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CFG_PATH="${CFG_PATH:-configs.cite_multi_pca100_seed_sweep}"
DATASET_LOCATION="${DATASET_LOCATION:-${CITE_MULTI_DATA_DIR:-${HOME}/Desktop/flow-maps-data}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cite_multi_pca100_seed_sweep}"

# Space-separated lists. Override any of these when invoking the script.
SEEDS="${SEEDS:-1 2 3}"
SLURM_IDS="${SLURM_IDS:-0 1 2 3 4 5 6 7 8}"
DATASETS="${DATASETS:-cite multi}"
HELDOUT_DAYS="${HELDOUT_DAYS:-3 4}"
TIME_MODES="${TIME_MODES:-equal_time}"
LEARNING_RATE="${LEARNING_RATE:-}"
DRY_RUN="${DRY_RUN:-0}"

MODE_NAMES=(
  "vanilla_flow_matching"
  "vanilla_flow_map"
  "bio_prior_flow_matching"
  "bio_prior_flow_map"
  "bio_prior_constrained_flow_map"
  "bio_prior_ot_flow_map"
  "bio_prior_ot_constrained_flow_map"
  "ot_flow_map"
  "bio_prior_ot_constrained_flow_matching"
  "ot_flow_matching"
)

read -r -a seed_values <<< "${SEEDS}"
read -r -a slurm_id_values <<< "${SLURM_IDS}"
read -r -a dataset_values <<< "${DATASETS}"
read -r -a heldout_day_values <<< "${HELDOUT_DAYS}"
read -r -a time_mode_values <<< "${TIME_MODES}"

if [[ ${#seed_values[@]} -eq 0 || ${#slurm_id_values[@]} -eq 0 || \
      ${#dataset_values[@]} -eq 0 || ${#heldout_day_values[@]} -eq 0 || \
      ${#time_mode_values[@]} -eq 0 ]]; then
  echo "SEEDS, SLURM_IDS, DATASETS, HELDOUT_DAYS, and TIME_MODES must not be empty." >&2
  exit 2
fi

for seed in "${seed_values[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    exit 2
  fi
done

for slurm_id in "${slurm_id_values[@]}"; do
  if [[ ! "${slurm_id}" =~ ^[0-9]$ ]]; then
    echo "Invalid SLURM_ID: ${slurm_id}; expected one of 0 1 2 3 4 5 6 7 8 9." >&2
    exit 2
  fi
done

if [[ -n "${LEARNING_RATE}" ]] && ! [[ "${LEARNING_RATE}" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "Invalid LEARNING_RATE: ${LEARNING_RATE}; expected a positive number." >&2
  exit 2
fi

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

for time_mode in "${time_mode_values[@]}"; do
  if [[ "${time_mode}" != "equal_time" && "${time_mode}" != "real_time" ]]; then
    echo "Invalid time mode: ${time_mode}; expected equal_time or real_time." >&2
    exit 2
  fi
done

run_count=$((${#seed_values[@]} * ${#slurm_id_values[@]} * \
  ${#dataset_values[@]} * ${#heldout_day_values[@]} * ${#time_mode_values[@]}))
echo "CITE/Multi sweep: ${run_count} runs"
echo "  seeds: ${seed_values[*]}"
echo "  slurm ids: ${slurm_id_values[*]}"
echo "  datasets: ${dataset_values[*]}"
echo "  held-out days: ${heldout_day_values[*]}"
echo "  time modes: ${time_mode_values[*]}"
echo "  learning rate: ${LEARNING_RATE:-config default}"
echo "  output: ${OUTPUT_ROOT}"

cd "${REPO_ROOT}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
fi

for dataset in "${dataset_values[@]}"; do
  for time_mode in "${time_mode_values[@]}"; do
    for heldout_day in "${heldout_day_values[@]}"; do
      for seed in "${seed_values[@]}"; do
        for slurm_id in "${slurm_id_values[@]}"; do
          mode_name="${MODE_NAMES[slurm_id]}"
          run_name="${dataset}_${time_mode}_holdout_day${heldout_day}_${mode_name}_seed${seed}"
          command=(
            .venv-flowmaps-metal/bin/python py/launchers/learn.py
            --cfg_path "${CFG_PATH}"
            --slurm_id "${slurm_id}"
            --dataset_name "${dataset}"
            --heldout_day "${heldout_day}"
            --cite_multi_time_mode "${time_mode}"
            --dataset_location "${DATASET_LOCATION}"
            --output_folder "${OUTPUT_ROOT}"
          )
          if [[ -n "${LEARNING_RATE}" ]]; then
            command+=(--learning_rate "${LEARNING_RATE}")
          fi

          echo "==> ${run_name}"
          if [[ "${DRY_RUN}" == "1" ]]; then
            printf 'CITE_MULTI_SEED=%q ' "${seed}"
            printf 'ENABLE_PJRT_COMPATIBILITY=1 '
            printf 'JAX_PLATFORMS=METAL,cpu '
            printf '%q ' "${command[@]}"
            printf '\n'
          else
            CITE_MULTI_SEED="${seed}" \
              ENABLE_PJRT_COMPATIBILITY=1 \
              JAX_PLATFORMS=METAL,cpu \
              "${command[@]}"
          fi
        done
      done
    done
  done
done

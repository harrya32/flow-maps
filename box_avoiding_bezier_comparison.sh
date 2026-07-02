#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/box_avoiding_bezier_comparison}"
METRICS_CSV="${METRICS_CSV:-${OUTPUT_ROOT}/box_avoiding_bezier_metrics.csv}"
SEEDS="${SEEDS:-42 43 44 45 46}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RESET_CSV="${RESET_CSV:-1}"

CFG_PATH="${CFG_PATH:-configs.box_avoiding_bezier_comparison}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-500}"
FINAL_CKPT_INDEX="${FINAL_CKPT_INDEX:-$((TOTAL_STEPS / SAVE_FREQ))}"

EVAL_N="${EVAL_N:-4096}"
EVAL_SEED="${EVAL_SEED:-12345}"
EMA_FAC="${EMA_FAC:-0.9999}"
WASSERSTEIN_PROJECTIONS="${WASSERSTEIN_PROJECTIONS:-256}"
MARGINAL_TIMES="${MARGINAL_TIMES:-0.25,0.5,0.75}"
PATH_POINTS="${PATH_POINTS:-401}"
EULER_STEPS="${EULER_STEPS:-200}"
FLOWMAP_STEPS="${FLOWMAP_STEPS:-1,2,5,10,25}"

MODE_IDS=(0 1 2 3)
MODE_NAMES=(
  "vanilla-flow-matching"
  "vanilla-flow-map"
  "constrained-flow-matching"
  "constrained-flow-map"
)

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${RESET_CSV}" == "1" ]]; then
  rm -f "${METRICS_CSV}"
fi

for seed in ${SEEDS}; do
  for idx in "${!MODE_IDS[@]}"; do
    mode_id="${MODE_IDS[$idx]}"
    mode_name="${MODE_NAMES[$idx]}"
    run_name="${mode_name}-${seed}"

    echo "==> ${run_name}"

    if [[ "${RUN_TRAIN}" == "1" ]]; then
      BOX_BEZIER_SEED="${seed}" "${PYTHON_BIN}" py/launchers/learn.py \
        --cfg_path "${CFG_PATH}" \
        --slurm_id "${mode_id}" \
        --dataset_location "" \
        --output_folder "${OUTPUT_ROOT}"
    fi

    if [[ "${RUN_EVAL}" == "1" ]]; then
      checkpoint="${OUTPUT_ROOT}/${run_name}_${FINAL_CKPT_INDEX}.pkl"
      if [[ ! -f "${checkpoint}" ]]; then
        checkpoint="$(ls -t "${OUTPUT_ROOT}/${run_name}_"*.pkl 2>/dev/null | head -n 1 || true)"
      fi
      if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        echo "Could not find checkpoint for ${run_name} in ${OUTPUT_ROOT}" >&2
        exit 1
      fi

      BOX_BEZIER_SEED="${seed}" "${PYTHON_BIN}" py/launchers/eval_box_avoiding_bezier.py \
        --cfg_path "${CFG_PATH}" \
        --slurm_id "${mode_id}" \
        --checkpoint "${checkpoint}" \
        --out_csv "${METRICS_CSV}" \
        --mode_name "${mode_name}" \
        --training_seed "${seed}" \
        --eval_seed "${EVAL_SEED}" \
        --ema_fac "${EMA_FAC}" \
        --n_eval "${EVAL_N}" \
        --wasserstein_projections "${WASSERSTEIN_PROJECTIONS}" \
        --marginal_times "${MARGINAL_TIMES}" \
        --path_points "${PATH_POINTS}" \
        --euler_steps "${EULER_STEPS}" \
        --flowmap_steps "${FLOWMAP_STEPS}" \
        --rescale_cache_dir "${OUTPUT_ROOT}"
    fi
  done
done

echo "Wrote metrics to ${METRICS_CSV}"

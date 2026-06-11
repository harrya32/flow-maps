#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/pdata/hmka3/flow-maps"
DATASET_DIR="/mnt/pdata/hmka3/flow-maps/datasets"
OUT_ROOT="/mnt/pdata/hmka3/flow-maps/outputs"
PYTHON_BIN="python"

TRAIN_DIR_FLOWMAP="${OUT_ROOT}/schiebinger_pca2_diag075"
TRAIN_DIR_FM="${OUT_ROOT}/schiebinger_pca2_diag100"

EVAL_DIR_FLOWMAP="${OUT_ROOT}/schiebinger_pca2_diag075_eval"
EVAL_DIR_FM="${OUT_ROOT}/schiebinger_pca2_diag100_eval"

latest_ckpt() {
  local dir="$1"
  local ckpt
  ckpt="$(ls -1t "${dir}"/*.pkl 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "No checkpoint found in: ${dir}" >&2
    exit 1
  fi
  echo "${ckpt}"
}

cd "${REPO_ROOT}"

CKPT_FLOWMAP="$(latest_ckpt "${TRAIN_DIR_FLOWMAP}")"
CKPT_FM="$(latest_ckpt "${TRAIN_DIR_FM}")"

echo "Flow-map checkpoint: ${CKPT_FLOWMAP}"
echo "Flow-matching checkpoint: ${CKPT_FM}"

# Eval run 1: flow map (diag_fraction=0.75 training run)
${PYTHON_BIN} py/launchers/eval_schiebinger_heldout.py \
  --cfg_path configs.schiebinger_lsd \
  --slurm_id 0 \
  --dataset_location "${DATASET_DIR}" \
  --checkpoint "${CKPT_FLOWMAP}" \
  --ema_fac 0.999 \
  --heldout_max_times 0 \
  --points_per_time 2000 \
  --seed 0 \
  --out_dir "${EVAL_DIR_FLOWMAP}"

# Eval run 2: FM baseline (diag_fraction=1.0 training run)
${PYTHON_BIN} py/launchers/eval_schiebinger_heldout.py \
  --cfg_path configs.schiebinger_lsd \
  --slurm_id 0 \
  --dataset_location "${DATASET_DIR}" \
  --checkpoint "${CKPT_FM}" \
  --ema_fac 0.999 \
  --heldout_max_times 0 \
  --points_per_time 2000 \
  --seed 0 \
  --out_dir "${EVAL_DIR_FM}"

# Optional quick comparison (mean over held-out times)
python - <<PY
import csv
from pathlib import Path

files = {
    "flow_map_diag075": Path("${EVAL_DIR_FLOWMAP}/schiebinger_heldout_metrics.csv"),
    "flow_matching_diag1": Path("${EVAL_DIR_FM}/schiebinger_heldout_metrics.csv"),
}

for name, fp in files.items():
    with fp.open() as f:
        rows = list(csv.DictReader(f))
    mean_mse = sum(float(r["mean_mse"]) for r in rows) / len(rows)
    cov_fro = sum(float(r["cov_fro_error"]) for r in rows) / len(rows)
    print(f"{name:20s} n_times={len(rows):2d} mean(mean_mse)={mean_mse:.6f} mean(cov_fro)={cov_fro:.6f}")
PY
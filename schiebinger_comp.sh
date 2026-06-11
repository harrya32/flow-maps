#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/pdata/hmka3/flow-maps"
DATASET_DIR="/mnt/pdata/hmka3/flow-maps/datasets"
OUT_ROOT="/mnt/pdata/hmka3/flow-maps/outputs"
PYTHON_BIN="python"   # or full path to your env python

cd "${REPO_ROOT}"

# Run 1: flow-map training (self-distillation on), diag_fraction=0.75
${PYTHON_BIN} py/launchers/learn.py \
  --cfg_path configs.schiebinger_lsd \
  --slurm_id 0 \
  --dataset_location "${DATASET_DIR}" \
  --output_folder "${OUT_ROOT}/schiebinger_pca2_diag075"

# Run 2: flow-matching baseline (diagonal-only), diag_fraction=1.0
TMP_CFG="/tmp/schiebinger_fm_diag1.py"
cat > "${TMP_CFG}" <<'PY'
from configs.schiebinger_lsd import get_config as _base_get_config

def get_config(slurm_id, dataset_location="", output_folder=""):
    cfg = _base_get_config(slurm_id, dataset_location, output_folder)
    cfg.optimization.diag_fraction = 1.0
    cfg.logging.wandb_name = "schiebinger_pca2_fm_diag1"
    cfg.logging.output_name = cfg.logging.wandb_name
    return cfg
PY

export PYTHONPATH="/tmp:${PYTHONPATH:-}"

${PYTHON_BIN} py/launchers/learn.py \
  --cfg_path schiebinger_fm_diag1 \
  --slurm_id 0 \
  --dataset_location "${DATASET_DIR}" \
  --output_folder "${OUT_ROOT}/schiebinger_pca2_diag100"

rm -f "${TMP_CFG}"
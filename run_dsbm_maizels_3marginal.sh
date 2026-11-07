#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWMAPS_PYTHON="${FLOWMAPS_PYTHON:-python}"

cd "$FLOWMAPS_ROOT"
"$FLOWMAPS_PYTHON" -m dsbm_baseline.train \
  --config_path "$FLOWMAPS_ROOT/dsbm_baseline/configs/maizels_3marginal.yaml" \
  --working_dir "$FLOWMAPS_ROOT/outputs/dsbm_maizels_pca50" \
  "$@"


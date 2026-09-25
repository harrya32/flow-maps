#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWMAPS_PYTHON="${FLOWMAPS_PYTHON:-python}"

cd "$FLOWMAPS_ROOT"
"$FLOWMAPS_PYTHON" -m branchsbm_baseline.train \
  --config_path "$FLOWMAPS_ROOT/branchsbm_baseline/configs/maizels_3marginal.yaml" \
  --working_dir "$FLOWMAPS_ROOT/outputs/branchsbm_maizels_pca50" \
  "$@"


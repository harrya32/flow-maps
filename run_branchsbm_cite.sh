#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWMAPS_PYTHON="${FLOWMAPS_PYTHON:-python}"

cd "$FLOWMAPS_ROOT"
"$FLOWMAPS_PYTHON" -m branchsbm_baseline.train \
  --config_path "$FLOWMAPS_ROOT/branchsbm_baseline/configs/cite.yaml" \
  --working_dir "$FLOWMAPS_ROOT/outputs/branchsbm_cite_pca100" \
  "$@"


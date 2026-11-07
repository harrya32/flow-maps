#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWMAPS_PYTHON="${FLOWMAPS_PYTHON:-python}"
cd "$FLOWMAPS_ROOT"
"$FLOWMAPS_PYTHON" -m cellmnn_baseline.train \
  --dataset_name cite \
  --working_dir "$FLOWMAPS_ROOT/outputs/cellmnn_cite_pca100" \
  "$@"


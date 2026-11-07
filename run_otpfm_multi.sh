#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWMAPS_PYTHON="${FLOWMAPS_PYTHON:-python}"

cd "$FLOWMAPS_ROOT"
"$FLOWMAPS_PYTHON" -m otpfm_baseline.train \
  --dataset_name multi \
  --working_dir "$FLOWMAPS_ROOT/outputs/otpfm_multi_pca100" \
  "$@"

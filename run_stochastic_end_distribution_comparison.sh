#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${FLOWMAPS_PYTHON:-python}"

exec "${PYTHON_BIN}" \
  "${FLOWMAPS_ROOT}/scripts/run_stochastic_end_distribution_comparison.py" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$FLOWMAPS_ROOT/run_cellmnn_cite.sh" "$@"
bash "$FLOWMAPS_ROOT/run_cellmnn_multi.sh" "$@"
bash "$FLOWMAPS_ROOT/run_cellmnn_maizels.sh" "$@"


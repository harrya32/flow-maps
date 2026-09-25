#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$FLOWMAPS_ROOT/run_branchsbm_maizels_3marginal.sh" "$@"
"$FLOWMAPS_ROOT/run_branchsbm_cite.sh" "$@"
"$FLOWMAPS_ROOT/run_branchsbm_multi.sh" "$@"


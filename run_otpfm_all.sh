#!/usr/bin/env bash
set -euo pipefail

FLOWMAPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$FLOWMAPS_ROOT/run_otpfm_cite.sh" "$@"
bash "$FLOWMAPS_ROOT/run_otpfm_multi.sh" "$@"
bash "$FLOWMAPS_ROOT/run_otpfm_maizels.sh" "$@"

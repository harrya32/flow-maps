"""OTP-FM adapters for the shared single-cell benchmarks."""

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "OTP-FM"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
for root in (
    REPO_ROOT / "py",
    REPO_ROOT / "metric-flow-matching",
    UPSTREAM_ROOT,
    UPSTREAM_ROOT / "src",
):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

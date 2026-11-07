"""Cell-MNN adapters for the shared single-cell benchmarks."""

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "cell-mnn"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".cache" / "numba"))
# PyTorch's MPS backend does not implement every linear-algebra/MMD operation.
# Its CPU fallback preserves autograd for these operations on Macs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for root in (REPO_ROOT / "py", REPO_ROOT / "metric-flow-matching", UPSTREAM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

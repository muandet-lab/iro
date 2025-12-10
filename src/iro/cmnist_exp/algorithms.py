"""Thin proxy to the shared IRO algorithms implementation.

Keeping this file ensures backwards-compatibility for scripts/notebooks that
do `import algorithms` from the CMNIST_spectral folder while delegating the
actual logic to the toolbox code under `iro.utility.algorithms`.
"""
from pathlib import Path
import sys

# Ensure local toolbox path (repo/src) is available
SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iro.utility.algorithms import *  # noqa: F401,F403

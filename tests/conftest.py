"""Make the repo root and studies/ importable without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "studies")):
    if p not in sys.path:
        sys.path.insert(0, p)

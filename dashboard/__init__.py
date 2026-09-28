"""Streamlit live view (NOT a locked file). Reads ONLY from a DuckDB file
(see dashboard/seed_demo_db.py, which writes it; this package's own modules
never generate or fit anything at render time).

Adds src/ (this repo's own locked modules) and the pinned plant-ingest
submodule's src/ (the cleared plant demo's OEE report code) to sys.path, so
both can be imported unmodified - the same pattern src/ingest.py,
runner/__init__.py and monitoring/__init__.py already use.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
_PLANT_SRC = REPO_ROOT / "vendor/plant-shift-oee-report/src"

for _p in (_SRC, _PLANT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

"""Model-fit + threshold-selection runner (NOT a locked file).

This package is the "model runner" the frozen protocol and Astra's lock-3
sign-off both require to be built and reviewed before any locked-seed
evaluation happens (vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md,
vault:10-projects/freelance/launch/twin/experiment-lock-status.md "Next step").

It never edits src/generator.py, src/features.py, src/evaluator.py,
src/ingest.py, src/twin_config.py, config.yaml or any experiment-lock*.json -
it only imports and calls them. Every entry point refuses to run at all
unless lock_guard.verify_lock() passes: experiment-lock.json's own sha256
equals the sha Astra signed off, and every file hash it records still
matches the file on disk.

Adds src/ to sys.path so it can import the locked modules the same way
tests/conftest.py does, without copying or editing them.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

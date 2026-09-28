"""Rule + model alert log (NOT a locked file; frozen modules are only
imported, never edited, never fitted on the locked train/validation/test
split table).

This is an OPERATIONAL monitoring demo, separate from the frozen protocol's
locked evaluation:

  * monitoring.rules - a causal, expanding per-sensor z-score rule computed
    from a machine's own observed history to date (never future samples).
  * monitoring.model_monitor - an IsolationForest fitted once on a fixed
    causal warm-up slice of a machine's own history ("operational
    baseline"), then scored forward, using config.yaml's frozen
    isolation_forest params (contamination taken as-is, never tuned here).
  * monitoring.alert_log - writes both alert streams to
    `monitoring_alert_log` in the same DuckDB file src/ingest.py writes
    sensor readings to, for the dashboard to read.

No number this module produces is a locked-evaluation result: it scores
whatever demonstration history is currently in the database (development
seeds or hand-fed messages in this milestone), using a fit/threshold
procedure the frozen protocol never specifies (a live per-machine
operational baseline, not the protocol's train/validation/test split). See
README "What this demo does not claim".

Adds src/ to sys.path so it can import the locked modules (features,
twin_config) the same way tests/conftest.py and runner/ do.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

"""dashboard.seed_demo_db: exercised only on the DEVELOPMENT table (seeds
901-906) with a short demo window, never the locked table. This is the
slowest test in the suite (it runs the real generator -> ingest ->
features -> monitoring pipeline, just over a short window instead of the
full multi-day dev dataset)."""
from pathlib import Path

import generator

DEV_SEEDS = {901, 902, 903, 904, 905, 906}
LOCKED_SEEDS = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
               201, 202, 203, 204, 301, 302, 303, 304, 305, 306}


def test_seed_writes_a_reconciled_dev_seed_only_demo_db(tmp_path):
    from dashboard.seed_demo_db import seed

    db_path = tmp_path / "twin_demo.duckdb"
    summary = seed(db_path, window_hours=0.75, model_warmup_windows=30)

    assert db_path.is_file()
    rec = summary["sensor"]["reconciliation"]
    assert rec["reconciled"] is True
    assert rec["handled"] == sum(summary["sensor"]["messages_by_stratum"].values())

    ds = generator.generate("dev")
    assert summary["sensor"]["dataset_hash"] == generator.dataset_hash(ds)
    seeds = {s for sp in ds.splits.values() for s in sp.seeds}
    assert seeds == DEV_SEEDS
    assert not (seeds & LOCKED_SEEDS)

    assert summary["plant_oee"]["accepted"] > 0
    assert set(summary["monitoring"]) == {"validation/FILLER", "validation/CAPPER", "test/LABELLER"}
    for stratum in summary["monitoring"].values():
        assert stratum["rows_written"] == stratum["rule_alerts"] + stratum["model_alerts"]


def test_seed_is_safe_to_rerun_onto_the_same_file(tmp_path):
    from dashboard.seed_demo_db import seed

    db_path = tmp_path / "twin_demo.duckdb"
    seed(db_path, window_hours=0.75, model_warmup_windows=30)
    # --force replaces the file cleanly rather than appending onto stale data
    summary = seed(db_path, force=True, window_hours=0.75, model_warmup_windows=30)
    assert summary["sensor"]["reconciliation"]["reconciled"] is True

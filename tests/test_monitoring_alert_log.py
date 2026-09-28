"""monitoring.alert_log: hand-built fixtures, an in-memory DuckDB."""
import numpy as np
import pytest


def _frame_with_spike(cfg, n=80, spike_at=70):
    import features

    rng = np.random.default_rng(4)
    cols = features.feature_columns(cfg)
    X = rng.standard_normal((n, len(cols)))
    X[spike_at] += 60.0
    # also make vibration_rms__mean_5m spike for the rule
    X[:, cols.index("vibration_rms__mean_5m")] = 10.0 + 0.1 * rng.standard_normal(n)
    X[spike_at, cols.index("vibration_rms__mean_5m")] = 500.0
    columns = {c: X[:, i] for i, c in enumerate(cols)}
    elig = np.ones(n, dtype=bool)
    return features.FeatureFrame(np.arange(n, dtype=np.int64) * 10, columns,
                                 np.full(n, "", dtype=object), elig, {"split": "dev"})


@pytest.fixture
def con():
    import duckdb
    c = duckdb.connect(":memory:")
    yield c
    c.close()


def test_init_schema_creates_the_table(con):
    from monitoring.alert_log import init_schema

    init_schema(con)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "monitoring_alert_log" in tables


def test_run_monitoring_writes_rule_and_model_alerts(con, cfg):
    from monitoring.alert_log import recent_alerts, run_monitoring

    frame = _frame_with_spike(cfg)
    summary = run_monitoring(con, "LINE_A.FILLER", frame, cfg, rule_min_history=30,
                             model_warmup_windows=50)
    assert summary["rule_alerts"] >= 1
    assert summary["model_note"] is None
    assert summary["rows_written"] == summary["rule_alerts"] + summary["model_alerts"]

    rows = recent_alerts(con, machine="LINE_A.FILLER")
    assert len(rows) == summary["rows_written"]
    assert {r["source"] for r in rows} <= {"rule", "model"}
    assert all(r["machine"] == "LINE_A.FILLER" for r in rows)


def test_run_monitoring_is_idempotent_on_rerun(con, cfg):
    from monitoring.alert_log import recent_alerts, run_monitoring

    frame = _frame_with_spike(cfg)
    run_monitoring(con, "LINE_A.FILLER", frame, cfg, rule_min_history=30, model_warmup_windows=50)
    first_count = con.execute("SELECT count(*) FROM monitoring_alert_log").fetchone()[0]
    run_monitoring(con, "LINE_A.FILLER", frame, cfg, rule_min_history=30, model_warmup_windows=50)
    second_count = con.execute("SELECT count(*) FROM monitoring_alert_log").fetchone()[0]
    assert first_count == second_count > 0


def test_run_monitoring_notes_insufficient_baseline_instead_of_crashing(con, cfg):
    from monitoring.alert_log import run_monitoring

    frame = _frame_with_spike(cfg, n=20, spike_at=15)
    summary = run_monitoring(con, "LINE_A.CAPPER", frame, cfg, rule_min_history=5,
                             model_warmup_windows=180)
    assert summary["model_alerts"] == 0
    assert summary["model_note"] is not None
    assert "eligible windows" in summary["model_note"]


def test_recent_alerts_filters_by_machine(con, cfg):
    from monitoring.alert_log import recent_alerts, run_monitoring

    frame = _frame_with_spike(cfg)
    run_monitoring(con, "LINE_A.FILLER", frame, cfg, rule_min_history=30, model_warmup_windows=50)
    run_monitoring(con, "LINE_A.CAPPER", frame, cfg, rule_min_history=30, model_warmup_windows=50)
    filler_rows = recent_alerts(con, machine="LINE_A.FILLER")
    all_rows = recent_alerts(con, machine=None, limit=1000)
    assert all(r["machine"] == "LINE_A.FILLER" for r in filler_rows)
    assert len(all_rows) >= len(filler_rows)

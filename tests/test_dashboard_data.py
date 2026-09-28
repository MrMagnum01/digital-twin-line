"""dashboard.data: hand-built DuckDB fixtures only (no generator, no real
seeds, no plant simulation) - fast, isolated tests of the pure query and
schematic-SVG functions."""
from datetime import datetime, timezone

import duckdb
import pytest

from dashboard import data as dd


@pytest.fixture
def con(cfg):
    import ingest
    from ingester import init_db as vendor_init_db   # vendor src on sys.path via dashboard/__init__.py

    from monitoring.alert_log import init_schema as monitoring_init_schema

    c = duckdb.connect(":memory:")
    ingest.init_db(c)
    vendor_init_db(c)
    monitoring_init_schema(c)
    yield c
    c.close()


def _insert_reading(con, tag, ts_epoch, **sensors):
    cols = ["vibration_rms", "bearing_temp_c", "motor_current_a", "line_pressure_bar"]
    vals = [sensors.get(c) for c in cols]
    con.execute(
        "INSERT INTO sensor_readings (log_id, machine, ts_epoch, " + ", ".join(cols) + ") "
        "VALUES (nextval('sensor_log_id_seq'), ?, ?, " + ", ".join("?" * len(cols)) + ")",
        [tag, ts_epoch] + vals)
    con.execute(
        "INSERT INTO sensor_ingest_log (received_at, category, tag, ts_epoch, raw_payload) "
        "VALUES (now(), 'accepted', ?, ?, '{}')", [tag, ts_epoch])


def test_machines_lists_the_configured_assets_in_order(cfg):
    ms = dd.machines(cfg)
    assert [m["asset"] for m in ms] == list(cfg["line"]["assets"])
    assert ms[0]["tag"] == "LINE_A.FILLER"


def test_latest_readings_is_oldest_first_and_bounded(con):
    tag = "LINE_A.FILLER"
    for i in range(5):
        _insert_reading(con, tag, 1000 + i * 10, vibration_rms=1.0 + i)
    rows = dd.latest_readings(con, tag, limit=3)
    assert [r["ts_epoch"] for r in rows] == [1020, 1030, 1040]
    assert rows[0]["vibration_rms"] == pytest.approx(3.0)
    assert rows[0]["ts"].tzinfo is not None


def test_ingest_summary_counts_and_watermark(con):
    tag = "LINE_A.FILLER"
    for i in range(3):
        _insert_reading(con, tag, 2000 + i * 10, vibration_rms=1.0)
    summary = dd.ingest_summary(con, tag)
    assert summary["by_category"]["accepted"] == 3
    assert summary["accepted_rows"] == 3
    assert summary["last_accepted_ts"] == datetime.fromtimestamp(2020, tz=timezone.utc)


def test_latest_state_returns_none_without_evidence(con):
    assert dd.latest_state(con, "LINE_A.FILLER") is None


def test_latest_state_returns_the_most_recent_state_event(con):
    tag = "LINE_A.CAPPER"
    con.execute(
        "INSERT INTO state_events (log_id, machine, ts, state, reason_code, seq, is_late) "
        "VALUES (1, ?, ?, 'RUN', NULL, 1, false)", [tag, datetime(2024, 1, 1, 0, 0)])
    con.execute(
        "INSERT INTO state_events (log_id, machine, ts, state, reason_code, seq, is_late) "
        "VALUES (2, ?, ?, 'DOWN', 'JAM', 2, false)", [tag, datetime(2024, 1, 1, 1, 0)])
    st = dd.latest_state(con, tag)
    assert st == {"state": "DOWN", "reason_code": "JAM", "ts": datetime(2024, 1, 1, 1, 0)}


def test_alert_summary_counts_by_source(con):
    from monitoring.alert_log import _insert

    now = datetime(2024, 1, 1)
    _insert(con, [(now, "LINE_A.FILLER", 0, 10, "rule", "vibration_rms", 3.5, "detail"),
                 (now, "LINE_A.FILLER", 10, 20, "model", "", 0.9, "detail2"),
                 (now, "LINE_A.CAPPER", 0, 10, "rule", "bearing_temp_c", 3.1, "detail3")])
    all_summary = dd.alert_summary(con)
    assert all_summary["counts"] == {"rule": 2, "model": 1}
    filler_summary = dd.alert_summary(con, tag="LINE_A.FILLER")
    assert filler_summary["counts"] == {"rule": 1, "model": 1}


def test_schematic_svg_is_valid_and_escapes_untrusted_strings(cfg):
    states = {
        "FILLER": {"state": "RUN", "reason_code": None, "ts": None},
        "CAPPER": {"state": "DOWN", "reason_code": "<script>alert(1)</script>", "ts": None},
        "LABELLER": None,
    }
    svg = dd.schematic_svg(cfg, states)
    assert svg.startswith("<svg")
    assert svg.count("<rect") >= 3   # 3 machine boxes (+ surface background rect)
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg
    assert "no state evidence" in svg   # LABELLER, unknown state


def test_schematic_svg_names_state_as_text_not_only_colour(cfg):
    states = {a: None for a in cfg["line"]["assets"]}
    states["FILLER"] = {"state": "RUN", "reason_code": None, "ts": None}
    svg = dd.schematic_svg(cfg, states)
    assert ">RUN<" in svg

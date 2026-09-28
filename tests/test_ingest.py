"""SENSOR ingest regression tests, unit level (no broker, no container), in
the style of plant-shift-oee-report/tests/test_ingester_validation.py: every
payload class is categorised, nothing is silently dropped, and category
totals reconcile to the number of messages handled."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pytest

import generator as g
import ingest

REPO = Path(__file__).resolve().parent.parent
PINNED = "7e378facb410b529d8e2d72d329daf2eee515161"
SENSORS = ("vibration_rms", "bearing_temp_c", "motor_current_a", "line_pressure_bar")


@pytest.fixture
def ing():
    return ingest.SensorIngester(con=duckdb.connect(":memory:"))


def msg(ts="2023-09-04T00:00:00+00:00", tag="LINE_A.FILLER", **kw):
    base = {"ts": ts, "tag": tag, "type": "SENSOR", "vibration_rms": 2.0,
            "bearing_temp_c": 45.0, "motor_current_a": 11.5, "line_pressure_bar": 4.0}
    base.update(kw)
    return json.dumps(base)


def cats(i):
    return dict(i.con.execute("SELECT category, count(*) FROM sensor_ingest_log GROUP BY 1").fetchall())


def _total(i):
    return i.con.execute("SELECT count(*) FROM sensor_ingest_log").fetchone()[0]


# ------------------------------------------------------- pinned reuse
def test_reused_functions_come_from_pinned_submodule_unmodified():
    sub = REPO / "vendor/plant-shift-oee-report"
    head = subprocess.run(["git", "-C", str(sub), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == PINNED
    status = subprocess.run(["git", "-C", str(sub), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True, check=True).stdout
    assert status == "", f"pinned submodule has local modifications:\n{status}"
    for fn in (ingest._parse_ts, ingest._content_hash):
        assert fn.__module__ == "ingester"
        assert Path(fn.__code__.co_filename).resolve() == (sub / "src/ingester.py").resolve()
    src = (REPO / "src/ingest.py").read_text()
    assert "def _parse_ts" not in src and "def _content_hash" not in src


# ------------------------------------------------------------ grammar
def test_accepted(ing):
    assert ing.handle_raw(msg()) == "accepted"
    row = ing.con.execute("SELECT machine, ts_epoch, vibration_rms FROM sensor_readings").fetchone()
    assert row == ("LINE_A.FILLER", 1693785600, 2.0)


def test_null_sensor_is_accepted_as_missing(ing):
    assert ing.handle_raw(msg(bearing_temp_c=None)) == "accepted"
    assert ing.con.execute("SELECT bearing_temp_c FROM sensor_readings").fetchone()[0] is None


def test_invalid_json_is_bad_payload_with_raw_preserved(ing):
    assert ing.handle_raw("{not json") == "bad_payload"
    assert ing.con.execute("SELECT category, raw_payload FROM sensor_ingest_log").fetchone() == \
        ("bad_payload", "{not json")


BAD = [
    msg(ts="not-a-ts"),
    msg(ts="2023-09-04T00:00:00"),                       # naive
    msg(ts="2023-09-04T00:00:00.500000+00:00"),          # off the 1 Hz grid
    msg(ts=1693785600),                                  # wrong type
    msg(type="STATE"),
    msg(tag=5),
    msg(tag=""),
    msg(vibration_rms="2.0"),
    msg(vibration_rms=True),
    msg(vibration_rms=[2.0]),
    msg(vibration_rms={"v": 2}),
    msg(vibration_rms=-0.1),                             # below range
    msg(bearing_temp_c=900.0),                           # above range
    msg(vibration_rms=None, bearing_temp_c=None, motor_current_a=None, line_pressure_bar=None),
    msg(extra_field=1),
    json.dumps({"ts": "2023-09-04T00:00:00+00:00", "tag": "LINE_A.FILLER", "type": "SENSOR",
                "vibration_rms": 1.0, "bearing_temp_c": 1.0, "motor_current_a": 1.0}),   # missing key
    '{"ts": "2023-09-04T00:00:00+00:00", "tag": "LINE_A.FILLER", "type": "SENSOR", '
    '"vibration_rms": NaN, "bearing_temp_c": 1, "motor_current_a": 1, "line_pressure_bar": 1}',
    '{"ts": "2023-09-04T00:00:00+00:00", "tag": "LINE_A.FILLER", "type": "SENSOR", '
    '"vibration_rms": Infinity, "bearing_temp_c": 1, "motor_current_a": 1, "line_pressure_bar": 1}',
    '{"ts": "2023-09-04T00:00:00+00:00", "ts": "2023-09-04T00:00:01+00:00", "tag": "LINE_A.FILLER", '
    '"type": "SENSOR", "vibration_rms": 1, "bearing_temp_c": 1, "motor_current_a": 1, '
    '"line_pressure_bar": 1}',                           # duplicate key ("duplicate header")
    "[]",
    "",
    "null",
]


@pytest.mark.parametrize("raw", BAD)
def test_grammar_violations_are_bad_payload(ing, raw):
    assert ing.handle_raw(raw) == "bad_payload"
    assert _total(ing) == 1
    assert ing.con.execute("SELECT count(*) FROM sensor_readings").fetchone()[0] == 0


def test_non_utf8_bytes_are_bad_payload(ing):
    assert ing.handle_raw(b"\xff\xfe{") == "bad_payload"
    assert _total(ing) == 1


def test_unknown_tag_quarantined(ing):
    assert ing.handle_raw(msg(tag="LINE_A.SEALER")) == "unknown_tag"
    assert ing.con.execute("SELECT count(*) FROM sensor_readings").fetchone()[0] == 0
    assert "LINE_A.SEALER" in ing.con.execute("SELECT raw_payload FROM sensor_ingest_log").fetchone()[0]


def test_duplicate_and_conflict(ing):
    m = msg()
    assert ing.handle_raw(m) == "accepted"
    # same content, different formatting -> still a duplicate (content hash)
    assert ing.handle_raw(json.dumps(json.loads(m), indent=2, sort_keys=True)) == "duplicate"
    assert ing.handle_raw(msg(vibration_rms=9.9)) == "reading_conflict"
    assert ing.con.execute("SELECT vibration_rms FROM sensor_readings").fetchall() == [(2.0,)]
    assert cats(ing) == {"accepted": 1, "duplicate": 1, "reading_conflict": 1}


def test_out_of_order_rejected_and_logged(ing):
    assert ing.handle_raw(msg(ts="2023-09-04T00:00:05+00:00")) == "accepted"
    assert ing.handle_raw(msg(ts="2023-09-04T00:00:03+00:00")) == "out_of_order_rejected"
    assert ing.handle_raw(msg(ts="2023-09-04T00:00:03+00:00")) == "duplicate"
    assert ing.handle_raw(msg(ts="2023-09-04T00:00:06+00:00")) == "accepted"
    assert ing.con.execute("SELECT count(*) FROM sensor_readings").fetchone()[0] == 2
    # per-tag watermark: another machine's earlier ts is fine
    assert ing.handle_raw(msg(ts="2023-09-04T00:00:01+00:00", tag="LINE_A.CAPPER")) == "accepted"


def test_known_total_reconciliation(ing):
    payloads = [msg(ts=f"2023-09-04T00:00:{s:02d}+00:00") for s in range(0, 20, 2)]
    payloads += [payloads[3], "{bad", msg(tag="X.Y"), msg(ts="2023-09-04T00:00:03+00:00"),
                 msg(ts="2023-09-04T00:00:18+00:00", vibration_rms=1.0)]
    for p in payloads:
        ing.handle_raw(p)
    rec = ing.reconciliation()
    assert rec["reconciled"]
    assert rec["handled"] == rec["logged_total"] == len(payloads)
    assert sum(rec["by_category"].values()) == len(payloads)
    assert rec["by_category"] == {"accepted": 10, "duplicate": 1, "reading_conflict": 1,
                                  "out_of_order_rejected": 1, "unknown_tag": 1, "bad_payload": 1}


# ------------------------------------------------------------ sources
def test_missing_and_zero_byte_sources_fail_categorised(tmp_path, ing):
    with pytest.raises(ingest.IngestSourceError) as e1:
        ingest.ingest_jsonl(tmp_path / "nope.jsonl", ing)
    assert e1.value.category == "missing_source"
    (tmp_path / "empty.jsonl").write_bytes(b"")
    with pytest.raises(ingest.IngestSourceError) as e2:
        ingest.ingest_jsonl(tmp_path / "empty.jsonl", ing)
    assert e2.value.category == "zero_byte_source"
    assert _total(ing) == 0


def test_jsonl_every_line_accounted(tmp_path, ing):
    lines = [msg(ts="2023-09-04T00:00:00+00:00"), "", "{bad", msg(ts="2023-09-04T00:00:01+00:00")]
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(lines) + "\n")
    rec = ingest.ingest_jsonl(p, ing)
    assert rec["source_lines"] == 4 and rec["source_reconciled"] and rec["reconciled"]
    assert rec["by_category"]["accepted"] == 2 and rec["by_category"]["bad_payload"] == 2


# ------------------------------------------------------ durability
class FailingCon:
    """Proxy that fails the fact INSERT on demand (write-failure injection)."""

    def __init__(self, con):
        self._con, self.fail = con, False

    def execute(self, sql, *a):
        if self.fail and sql.lstrip().startswith("INSERT INTO sensor_readings"):
            raise RuntimeError("injected fact-insert failure")
        return self._con.execute(sql, *a)


def test_write_failure_rolls_back_log_and_state():
    con = FailingCon(duckdb.connect(":memory:"))
    i = ingest.SensorIngester(con=con)
    assert i.handle_raw(msg(ts="2023-09-04T00:00:00+00:00")) == "accepted"
    i.commit()
    assert i.handle_raw(msg(ts="2023-09-04T00:00:01+00:00")) == "accepted"   # uncommitted
    con.fail = True
    with pytest.raises(RuntimeError):
        i.handle_raw(msg(ts="2023-09-04T00:00:02+00:00"))
    con.fail = False
    # batch tail rolled back exactly like a crash; no log row without its fact row
    assert i.con.execute("SELECT count(*) FROM sensor_ingest_log").fetchone()[0] == 1
    assert i.con.execute("SELECT count(*) FROM sensor_readings").fetchone()[0] == 1
    # in-memory state rebuilt from the DB: retries are accepted, not phantom duplicates
    assert i.handle_raw(msg(ts="2023-09-04T00:00:01+00:00")) == "accepted"
    assert i.handle_raw(msg(ts="2023-09-04T00:00:02+00:00")) == "accepted"
    assert i.reconciliation()["reconciled"]


def test_restart_replay_is_not_double_counted(tmp_path):
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    i = ingest.SensorIngester(con=con)
    ms = [msg(ts=f"2023-09-04T00:00:{s:02d}+00:00") for s in range(5)]
    for m in ms:
        assert i.handle_raw(m) == "accepted"
    i.commit()
    con.close()
    con2 = duckdb.connect(db)
    i2 = ingest.SensorIngester(con=con2)
    assert [i2.handle_raw(m) for m in ms] == ["duplicate"] * 5
    assert i2.handle_raw(msg(ts="2023-09-04T00:00:02+00:00", vibration_rms=7.0)) == "reading_conflict"
    assert con2.execute("SELECT count(*) FROM sensor_readings").fetchone()[0] == 5
    assert i2.reconciliation()["reconciled"]
    con2.close()


def test_crash_mid_batch_leaves_no_orphan_rows(tmp_path):
    db = tmp_path / "crash.duckdb"
    script = textwrap.dedent(f"""
        import os, sys, json
        sys.path.insert(0, {str(REPO / 'src')!r})
        import duckdb, ingest
        i = ingest.SensorIngester(con=duckdb.connect({str(db)!r}))
        for s in range(ingest.COMMIT_BATCH_SIZE + 37):
            m = {{"ts": "2023-09-04T%02d:%02d:%02d+00:00" % (s // 3600, (s // 60) % 60, s % 60),
                 "tag": "LINE_A.FILLER", "type": "SENSOR", "vibration_rms": 1.0,
                 "bearing_temp_c": 40.0, "motor_current_a": 10.0, "line_pressure_bar": 3.0}}
            i.handle_raw(json.dumps(m))
        os._exit(0)
    """)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    con = duckdb.connect(str(db))
    logs = con.execute("SELECT count(*) FROM sensor_ingest_log").fetchone()[0]
    facts = con.execute("SELECT count(*) FROM sensor_readings").fetchone()[0]
    assert logs == facts == ingest.COMMIT_BATCH_SIZE      # only the committed batch survived
    con.close()


# ------------------------------------------------- integration (dev seed)
def test_generator_messages_roundtrip_through_ingest(dev_ds):
    """Dev-seed slice around a planted dropout: generator -> SENSOR messages
    -> ingest -> declared grid is value-identical to the raw channel, and the
    dropout remains visible as missing grid seconds."""
    ev = next(e for e in dev_ds.labels.events if e.fault_type == "dropout")
    r = dev_ds.readings[(ev.split, ev.asset)]
    tag = f"LINE_A.{ev.asset}"
    lo, hi = ev.onset - 300, ev.end + 300
    msgs = g.to_messages(r, tag, lo, hi)
    i = ingest.SensorIngester(con=duckdb.connect(":memory:"))
    for m in msgs:
        assert i.handle_raw(m) == "accepted"
    rec = i.reconciliation()
    assert rec["reconciled"] and rec["handled"] == len(msgs) < hi - lo
    back = ingest.readings_grid(i.con, tag, lo, hi)
    for s in SENSORS:
        assert np.array_equal(back.values[s], r.values[s][lo - r.t0:hi - r.t0], equal_nan=True)
    assert np.isnan(back.values["vibration_rms"][ev.onset - lo:ev.end - lo]).all()

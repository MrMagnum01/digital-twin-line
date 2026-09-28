"""Validated intake of the twin's SENSOR messages into DuckDB.

REUSE BY PINNED REFERENCE (see README "Reuse of the cleared plant ingest"):
  `_parse_ts` and `_content_hash` are imported unmodified from
  vendor/plant-shift-oee-report/src/ingester.py, a git submodule pinned at
  7e378facb410b529d8e2d72d329daf2eee515161 (the CLEARED plant demo). They
  are schema-agnostic (ISO8601 tz-aware timestamp parsing; canonical-JSON
  sha256 content identity). Nothing from the submodule is copied or edited.
  Everything else below - the SENSOR grammar, the schema, categories,
  reconciliation and transaction handling - is NEW code for the new sensor
  schema, written to the same validation contract as the plant ingester
  (and common-rules.md). Integrity guarantees are NOT inherited merely by
  importing; they are tested in tests/test_ingest.py.

Grammar (one JSON object per message, exactly these keys):
  {"ts": "<ISO8601, tz-aware, whole second>", "tag": "LINE_A.<MACHINE>",
   "type": "SENSOR", "vibration_rms": num|null, "bearing_temp_c": num|null,
   "motor_current_a": num|null, "line_pressure_bar": num|null}
  num = JSON int/float (not bool), finite, inside config sensors.ingest_range.
  At least one sensor must be non-null. Duplicate JSON keys, NaN/Infinity
  literals, extra or missing keys are bad_payload.

Categories (each handled message gets exactly one ingest-log row):
  accepted              valid, ts newer than the tag's newest accepted ts
  duplicate             same (tag, ts) already seen with identical content
  reading_conflict      same (tag, ts) already seen with DIFFERENT content;
                        logged only, the original fact row is untouched
  out_of_order_rejected valid, new (tag, ts), but ts <= newest accepted ts for
                        the tag (no late-acceptance window: a 1 Hz grid
                        sample that arrives after a newer one is logged and
                        quarantined, never silently merged)
  unknown_tag           tag is not one of the configured machines
  bad_payload           invalid JSON / grammar / type / range violation

Known-total reconciliation: sum over categories == messages handled.

Durability: the log row and its fact row are written in one DuckDB
transaction and committed together (batches of COMMIT_BATCH_SIZE, never
splitting a message). On a write failure the transaction is rolled back and
all in-memory state (seen keys, watermarks, stats) is rebuilt from what is
durably committed. A fresh SensorIngester on an existing DB restores state
from DuckDB, so replaying committed messages is classified duplicate, not
double-counted.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

from twin_config import load_config

_PLANT_SRC = Path(__file__).resolve().parent.parent / "vendor/plant-shift-oee-report/src"
if not (_PLANT_SRC / "ingester.py").is_file():
    raise ImportError(
        f"pinned plant ingest not found at {_PLANT_SRC}; run `git submodule update --init`")
sys.path.insert(0, str(_PLANT_SRC))
# Do not write __pycache__ into the read-only pinned submodule.
_prev_dwb = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    from ingester import _content_hash, _parse_ts  # noqa: E402  (pinned 7e378fa, unmodified)
finally:
    sys.dont_write_bytecode = _prev_dwb

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
COMMIT_BATCH_SIZE = 500
MSG_TYPE = "SENSOR"
CATEGORIES = ("accepted", "duplicate", "reading_conflict", "out_of_order_rejected",
              "unknown_tag", "bad_payload")
_SEEN_CATEGORIES = ("accepted", "out_of_order_rejected")


class IngestSourceError(Exception):
    """A source-level failure (not a message): missing or zero-byte input."""

    def __init__(self, category: str, detail: str):
        super().__init__(f"{category}: {detail}")
        self.category = category


class _DuplicateKey(ValueError):
    pass


def _no_dup_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(f"duplicate key {k!r}")
        out[k] = v
    return out


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def _parse_json(raw: str):
    return json.loads(raw, object_pairs_hook=_no_dup_keys, parse_constant=_reject_constant)


def _num_ok(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def validate_grammar(msg, cfg: dict | None = None) -> tuple[bool, str]:
    """Strict SENSOR grammar. Never raises."""
    cfg = cfg or load_config()
    sensors = cfg["sensors"]["order"]
    ranges = cfg["sensors"]["ingest_range"]
    if not isinstance(msg, dict):
        return False, "payload is not a JSON object"
    expected = {"ts", "tag", "type", *sensors}
    for k in sorted(expected - set(msg)):
        return False, f"missing field {k}"
    for k in sorted(set(msg) - expected, key=str):
        return False, f"unexpected field {k!r}"
    if not isinstance(msg["type"], str) or msg["type"] != MSG_TYPE:
        return False, f"type must be {MSG_TYPE!r}"
    if not isinstance(msg["tag"], str) or not msg["tag"]:
        return False, "tag must be a non-empty string"
    ts = _parse_ts(msg["ts"])
    if ts is None:
        return False, "ts is not a valid timezone-aware ISO8601 timestamp"
    if ts.microsecond != 0:
        return False, "ts is not on the 1 Hz grid (fractional second)"
    present = 0
    for name in sensors:
        v = msg[name]
        if v is None:
            continue
        if not _num_ok(v):
            return False, f"{name} must be a finite number or null"
        lo, hi = ranges[name]
        if not (lo <= float(v) <= hi):
            return False, f"{name}={v} outside ingest range [{lo}, {hi}]"
        present += 1
    if present == 0:
        return False, "all sensor values are null"
    return True, ""


def _safe_fields(msg) -> tuple[str | None, str | None]:
    if not isinstance(msg, dict):
        return None, None
    tag = msg.get("tag") if isinstance(msg.get("tag"), str) else None
    ts_raw = msg.get("ts") if isinstance(msg.get("ts"), str) else None
    return tag, ts_raw


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    for stmt in SCHEMA_PATH.read_text().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)


@dataclass
class SensorIngester:
    con: duckdb.DuckDBPyConnection
    cfg: dict = field(default_factory=load_config)
    seen: dict = field(default_factory=dict)        # (tag, ts_epoch) -> content_hash
    watermark: dict = field(default_factory=dict)   # tag -> max accepted ts_epoch
    stats: dict = field(default_factory=dict)
    handled: int = 0
    _uncommitted: int = 0
    _in_txn: bool = False

    def __post_init__(self):
        self.sensors = list(self.cfg["sensors"]["order"])
        self.known_tags = {a["tag"] for a in self.cfg["line"]["assets"].values()}
        init_db(self.con)
        self._restore_state()

    # -- state ---------------------------------------------------------------
    def _restore_state(self) -> None:
        rows = self.con.execute(
            "SELECT tag, ts_epoch, content_hash FROM sensor_ingest_log "
            "WHERE category IN ('accepted','out_of_order_rejected')").fetchall()
        self.seen = {(t, e): h for t, e, h in rows}
        self.watermark = dict(self.con.execute(
            "SELECT machine, max(ts_epoch) FROM sensor_readings GROUP BY machine").fetchall())
        self.stats = dict(self.con.execute(
            "SELECT category, count(*) FROM sensor_ingest_log GROUP BY category").fetchall())
        self.handled = sum(self.stats.values())

    def _begin_if_needed(self) -> None:
        if not self._in_txn:
            self.con.execute("BEGIN TRANSACTION")
            self._in_txn = True

    def commit(self) -> None:
        if self._in_txn:
            self.con.execute("COMMIT")
            self._in_txn = False
            self._uncommitted = 0

    def _log(self, category, tag, ts_raw, ts_epoch, raw, reason, content_hash=None) -> int:
        self._begin_if_needed()
        return self.con.execute(
            """INSERT INTO sensor_ingest_log
               (received_at, category, tag, ts_raw, ts_epoch, raw_payload, reason, content_hash)
               VALUES (now(), ?, ?, ?, ?, ?, ?, ?) RETURNING log_id""",
            [category, tag, ts_raw, ts_epoch, raw, reason, content_hash]).fetchone()[0]

    def _record(self, category: str) -> str:
        self.stats[category] = self.stats.get(category, 0) + 1
        self.handled += 1
        self._uncommitted += 1
        if self._uncommitted >= COMMIT_BATCH_SIZE:
            self.commit()
        return category

    def _recover(self) -> None:
        try:
            self.con.execute("ROLLBACK")
        except Exception:
            pass
        self._in_txn = False
        self._uncommitted = 0
        self._restore_state()

    # -- intake ----------------------------------------------------------------
    def handle_raw(self, raw) -> str:
        """Classify and store one raw message. Never raises for bad input."""
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                raw_s = raw.decode("utf-8", errors="backslashreplace")
                self._log("bad_payload", None, None, None, raw_s, "not valid UTF-8")
                return self._record("bad_payload")
        if not isinstance(raw, str):
            self._log("bad_payload", None, None, None, repr(raw), "payload is not text")
            return self._record("bad_payload")
        try:
            msg = _parse_json(raw)
        except _DuplicateKey as exc:
            self._log("bad_payload", None, None, None, raw, str(exc))
            return self._record("bad_payload")
        except (ValueError, TypeError):
            self._log("bad_payload", None, None, None, raw, "invalid JSON")
            return self._record("bad_payload")

        tag, ts_raw = _safe_fields(msg)
        ok, reason = validate_grammar(msg, self.cfg)
        if not ok:
            self._log("bad_payload", tag, ts_raw, None, raw, reason)
            return self._record("bad_payload")
        ts_epoch = int(_parse_ts(ts_raw).timestamp())
        if tag not in self.known_tags:
            self._log("unknown_tag", tag, ts_raw, ts_epoch, raw,
                      f"tag {tag!r} not in configured machine registry")
            return self._record("unknown_tag")

        h = _content_hash(msg)
        key = (tag, ts_epoch)
        try:
            if key in self.seen:
                if self.seen[key] == h:
                    self._log("duplicate", tag, ts_raw, ts_epoch, raw,
                              "same (tag, ts) already ingested with identical content", h)
                    return self._record("duplicate")
                self._log("reading_conflict", tag, ts_raw, ts_epoch, raw,
                          "same (tag, ts) already ingested with different content", h)
                return self._record("reading_conflict")
            wm = self.watermark.get(tag)
            if wm is not None and ts_epoch <= wm:
                self.seen[key] = h
                self._log("out_of_order_rejected", tag, ts_raw, ts_epoch, raw,
                          f"ts {ts_epoch} not after newest accepted ts {wm} for tag", h)
                return self._record("out_of_order_rejected")
            self.seen[key] = h
            self.watermark[tag] = ts_epoch
            log_id = self._log("accepted", tag, ts_raw, ts_epoch, raw, None, h)
            self.con.execute(
                "INSERT INTO sensor_readings (log_id, machine, ts_epoch, "
                + ", ".join(self.sensors) + ") VALUES (?, ?, ?, " + ", ".join("?" * len(self.sensors)) + ")",
                [log_id, tag, ts_epoch] + [None if msg[s] is None else float(msg[s]) for s in self.sensors])
            return self._record("accepted")
        except Exception:
            self._recover()
            raise

    def reconciliation(self) -> dict:
        """Known-total check: every handled message has exactly one outcome."""
        self.commit()
        logged = dict(self.con.execute(
            "SELECT category, count(*) FROM sensor_ingest_log GROUP BY category").fetchall())
        total = sum(logged.values())
        facts = self.con.execute("SELECT count(*) FROM sensor_readings").fetchone()[0]
        return {
            "handled": self.handled,
            "logged_total": total,
            "by_category": {c: logged.get(c, 0) for c in CATEGORIES},
            "fact_rows": facts,
            "reconciled": total == self.handled and facts == logged.get("accepted", 0)
                          and set(logged) <= set(CATEGORIES),
        }


def ingest_jsonl(path: str | Path, ing: SensorIngester) -> dict:
    """Ingest a JSON-lines source. A missing or zero-byte source is a
    categorised IngestSourceError (never an empty success). Every line -
    including a blank or malformed one - is one handled message."""
    p = Path(path)
    if not p.exists():
        raise IngestSourceError("missing_source", str(p))
    if not p.is_file():
        raise IngestSourceError("missing_source", f"{p} is not a regular file")
    data = p.read_bytes()
    if len(data) == 0:
        raise IngestSourceError("zero_byte_source", str(p))
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]          # trailing newline terminator, not a record
    before = ing.handled
    for line in lines:
        ing.handle_raw(line)
    rec = ing.reconciliation()
    rec["source_lines"] = len(lines)
    rec["source_reconciled"] = ing.handled - before == len(lines)
    return rec


def readings_grid(con: duckdb.DuckDBPyConnection, tag: str, start: int, end: int,
                  cfg: dict | None = None):
    """Rebuild the 1 Hz declared grid [start, end) for one machine from the
    fact table. Grid seconds with no accepted row are NaN (missing), so a
    dropout stays visible to features.py's missingness accounting."""
    from generator import SensorReadings

    cfg = cfg or load_config()
    sensors = cfg["sensors"]["order"]
    n = end - start
    vals = {s: np.full(n, np.nan) for s in sensors}
    rows = con.execute(
        "SELECT ts_epoch, " + ", ".join(sensors) + " FROM sensor_readings "
        "WHERE machine = ? AND ts_epoch >= ? AND ts_epoch < ? ORDER BY ts_epoch",
        [tag, start, end]).fetchall()
    for row in rows:
        i = row[0] - start
        for j, s in enumerate(sensors):
            if row[1 + j] is not None:
                vals[s][i] = row[1 + j]
    return SensorReadings(start, vals)

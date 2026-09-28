"""Writes the rule + model alert streams into monitoring_alert_log
(monitoring/schema.sql), in the same DuckDB file src/ingest.py writes
sensor readings to. See monitoring/__init__.py for what this is and is not."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from monitoring import SCHEMA_PATH
from monitoring.model_monitor import InsufficientBaselineError, fit_baseline, score_frame
from monitoring.rules import expanding_zscore_alerts

_COLUMNS = ("alert_id", "generated_at", "machine", "window_start", "window_end",
           "source", "sensor", "score", "detail")


def init_schema(con) -> None:
    for stmt in SCHEMA_PATH.read_text().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)


def _insert(con, rows: list[tuple]) -> None:
    if not rows:
        return
    con.executemany(
        "INSERT INTO monitoring_alert_log "
        "(generated_at, machine, window_start, window_end, source, sensor, score, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
        rows)


def run_monitoring(con, machine: str, frame, cfg: dict, *, rule_k: float = 3.0,
                   rule_min_history: int = 30, model_warmup_windows: int = 180) -> dict:
    """Score one asset's FeatureFrame with both the rule and the model
    monitor and append every alerting window to monitoring_alert_log.
    Idempotent: rerunning over already-scored history writes no duplicate
    rows (UNIQUE(machine, window_start, source, sensor) + ON CONFLICT DO
    NOTHING)."""
    init_schema(con)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    bs = cfg["features"]["window_s"]
    sensors = cfg["sensors"]["order"]

    rule_alert, rule_sensor, rule_detail, rule_z = expanding_zscore_alerts(
        frame, sensors, k=rule_k, min_history=rule_min_history)
    rows = [(now, machine, int(ws), int(ws) + bs, "rule", str(rule_sensor[i]), float(rule_z[i]),
            str(rule_detail[i]))
           for i, ws in zip(np.flatnonzero(rule_alert), frame.window_start[rule_alert])]

    model_alert_count = 0
    model_note = None
    try:
        baseline = fit_baseline(frame, cfg, warmup_windows=model_warmup_windows)
        model_alert, scores = score_frame(frame, baseline, cfg)
        rows += [(now, machine, int(ws), int(ws) + bs, "model", "", float(scores[i]),
                 f"isolation_forest score={scores[i]:.3f} > threshold={baseline['threshold']:.3f} "
                 f"(q{baseline['threshold_quantile']} of a {baseline['warmup_windows']}-window baseline)")
                for i, ws in zip(np.flatnonzero(model_alert), frame.window_start[model_alert])]
        model_alert_count = int(model_alert.sum())
    except InsufficientBaselineError as exc:
        model_note = str(exc)

    _insert(con, rows)
    return {"machine": machine, "rule_alerts": int(rule_alert.sum()),
           "model_alerts": model_alert_count, "model_note": model_note,
           "rows_written": len(rows)}


def recent_alerts(con, machine: str | None = None, limit: int = 200) -> list[dict]:
    q = ("SELECT alert_id, generated_at, machine, window_start, window_end, "
        "source, sensor, score, detail FROM monitoring_alert_log")
    params: list = []
    if machine is not None:
        q += " WHERE machine = ?"
        params.append(machine)
    q += " ORDER BY window_start DESC LIMIT ?"
    params.append(limit)
    rows = con.execute(q, params).fetchall()
    return [dict(zip(_COLUMNS, r)) for r in rows]

"""Pure data-access functions for the dashboard (NOT a locked file; reads
only, no fitting, no generation). Kept free of any Streamlit import so it
can be unit-tested directly.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

from dashboard import palette

_SENSOR_COLUMNS = ("vibration_rms", "bearing_temp_c", "motor_current_a", "line_pressure_bar")


def machines(cfg: dict) -> list[dict]:
    """[{asset, tag}] in config.yaml's fixed asset order."""
    return [{"asset": a, "tag": meta["tag"]} for a, meta in cfg["line"]["assets"].items()]


def latest_readings(con, tag: str, limit: int = 500) -> list[dict]:
    """Most recent `limit` accepted sensor_readings rows for one machine tag,
    oldest first (so a line chart draws left-to-right in time order)."""
    rows = con.execute(
        "SELECT ts_epoch, vibration_rms, bearing_temp_c, motor_current_a, line_pressure_bar "
        "FROM sensor_readings WHERE machine = ? ORDER BY ts_epoch DESC LIMIT ?",
        [tag, limit]).fetchall()
    rows = list(reversed(rows))
    return [{"ts_epoch": r[0], "ts": datetime.fromtimestamp(r[0], tz=timezone.utc),
            "vibration_rms": r[1], "bearing_temp_c": r[2], "motor_current_a": r[3],
            "line_pressure_bar": r[4]} for r in rows]


def ingest_summary(con, tag: str) -> dict:
    """Known-total reconciliation view for one tag: category counts and the
    accepted watermark, straight from sensor_ingest_log/sensor_readings."""
    rows = con.execute(
        "SELECT category, count(*) FROM sensor_ingest_log WHERE tag = ? GROUP BY category",
        [tag]).fetchall()
    by_category = dict(rows)
    n_readings = con.execute(
        "SELECT count(*) FROM sensor_readings WHERE machine = ?", [tag]).fetchone()[0]
    last_ts = con.execute(
        "SELECT max(ts_epoch) FROM sensor_readings WHERE machine = ?", [tag]).fetchone()[0]
    return {"by_category": by_category, "accepted_rows": n_readings,
           "last_accepted_ts": (datetime.fromtimestamp(last_ts, tz=timezone.utc)
                                if last_ts is not None else None)}


def latest_state(con, tag: str) -> dict | None:
    """The plant demo's own most recent STATE event for one machine (from
    the cleared plant ingest's state_events table), or None if it has no
    STATE evidence yet. Never invents a default state."""
    row = con.execute(
        "SELECT state, reason_code, ts FROM state_events WHERE machine = ? "
        "ORDER BY ts DESC LIMIT 1", [tag]).fetchone()
    if row is None:
        return None
    return {"state": row[0], "reason_code": row[1], "ts": row[2]}


def alert_summary(con, tag: str | None = None) -> dict:
    from monitoring.alert_log import recent_alerts

    rows = recent_alerts(con, machine=tag, limit=1000)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return {"counts": counts, "rows": rows}


def oee_report(db_path: str) -> dict:
    """The cleared plant demo's own OEE numbers, computed by ITS OWN
    unmodified report.build_report_data() against the STATE/COUNT/ALARM
    tables seeded by dashboard/seed_demo_db.py from its own deterministic
    synthetic day. Not a locked-evaluation result; see report.OEE_METHOD_NOTE
    and README "What this demo does not claim"."""
    import report

    return report.build_report_data(db_path)


def schematic_svg(cfg: dict, states: dict[str, dict | None], width: int = 720, height: int = 200) -> str:
    """A phone-readable line schematic: FILLER -> CAPPER -> LABELLER boxes
    coloured by each machine's latest known STATE, with the state name and
    (for DOWN) its reason code always rendered as text - colour is never
    the only signal. `states` maps asset name -> latest_state() result (or
    None). Every interpolated string is HTML-escaped before it reaches the
    SVG markup."""
    assets = list(cfg["line"]["assets"])
    n = len(assets)
    box_w, box_h = 160, 90
    gap = (width - n * box_w) / max(n + 1, 1)
    y = (height - box_h) / 2
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'role="img" aria-label="Line A schematic">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{palette.SURFACE_LIGHT}"/>')

    centers = []
    for i, asset in enumerate(assets):
        x = gap + i * (box_w + gap)
        centers.append((x, x + box_w))
        st = states.get(asset)
        if st is None:
            color = palette.STATE_UNKNOWN_COLOR
            label = "no state evidence"
        else:
            role = palette.STATE_STATUS.get(st["state"], "warning")
            color = palette.STATUS[role]
            label = st["state"] if st["state"] != "DOWN" or not st.get("reason_code") \
                else f'DOWN ({st["reason_code"]})'
        safe_asset = html.escape(str(asset))
        safe_label = html.escape(str(label))
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="10" '
                     f'fill="{color}" stroke="{palette.BASELINE}" stroke-width="1"/>')
        parts.append(f'<text x="{x + box_w / 2}" y="{y + box_h / 2 - 6}" text-anchor="middle" '
                     f'font-family="system-ui, sans-serif" font-size="16" font-weight="600" '
                     f'fill="#ffffff">{safe_asset}</text>')
        parts.append(f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 16}" text-anchor="middle" '
                     f'font-family="system-ui, sans-serif" font-size="12" '
                     f'fill="#ffffff">{safe_label}</text>')

    for (_, x1_end), (x2_start, _) in zip(centers, centers[1:]):
        my = y + box_h / 2
        parts.append(f'<line x1="{x1_end}" y1="{my}" x2="{x2_start - 14}" y2="{my}" '
                     f'stroke="{palette.TEXT_MUTED}" stroke-width="3" marker-end="url(#arrow)"/>')

    parts.insert(1, (
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        f'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{palette.TEXT_MUTED}"/></marker></defs>'))
    parts.append('</svg>')
    return "".join(parts)

"""Streamlit live view (NOT a locked file). Reads ONLY from the DuckDB file
dashboard/seed_demo_db.py writes - this script never generates, fits or
ingests anything itself.

Run: `streamlit run dashboard/app.py` (after `python3 dashboard/seed_demo_db.py`).
"""
from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from dashboard import REPO_ROOT  # noqa: F401  (import first: wires sys.path)
from dashboard import data as dd
from dashboard import palette
from twin_config import load_config

ROLE_LINE = ("Synthetic portfolio demonstration, implemented with AI coding agents; "
            "independent review pending. No client data or client work.")
CLAIM_NOTE = ("This dashboard renders whatever demo data is in the DuckDB file below "
             "(development seeds only in this milestone). It makes no industrial "
             "reliability or performance claim; see README \"What this demo does not claim\".")

st.set_page_config(page_title="Digital twin line - LINE_A", layout="wide")


@st.cache_resource(show_spinner=False)
def _connection(db_path: str, _mtime: float):
    return duckdb.connect(db_path, read_only=True)


def get_connection(db_path: str):
    """Reopens the cached read-only connection if the file has changed
    since it was opened (e.g. a rerun of seed_demo_db.py)."""
    mtime = Path(db_path).stat().st_mtime
    return _connection(db_path, mtime)


def sensor_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["ts"], value_vars=list(palette.SENSOR_COLOR),
                   var_name="sensor", value_name="value")
    color_scale = alt.Scale(domain=list(palette.SENSOR_COLOR), range=list(palette.SENSOR_COLOR.values()))
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("ts:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("sensor:N", scale=color_scale, legend=alt.Legend(title="Sensor")),
            tooltip=[alt.Tooltip("ts:T", title="Time"), "sensor:N",
                    alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=220)
        .configure_axis(gridColor=palette.GRIDLINE, domainColor=palette.BASELINE,
                        labelColor=palette.TEXT_SECONDARY, tickColor=palette.BASELINE)
        .configure_view(strokeWidth=0)
    )


def oee_bar_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(color=palette.SENSOR_COLOR["vibration_rms"], cornerRadiusEnd=4)
        .encode(
            x=alt.X("oee:Q", title="OEE", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", title=None, sort=None),
            tooltip=["label:N", alt.Tooltip("oee:Q", format=".1%")],
        )
        .properties(height=28 * max(len(df), 1) + 20)
        .configure_axis(gridColor=palette.GRIDLINE, domainColor=palette.BASELINE,
                        labelColor=palette.TEXT_SECONDARY)
        .configure_view(strokeWidth=0)
    )


def render(db_path: str) -> None:
    st.title("Digital twin line - LINE_A")
    st.caption(ROLE_LINE)

    if not Path(db_path).is_file():
        st.error(f"No DuckDB file at {db_path!r}. Run `python3 dashboard/seed_demo_db.py` first.")
        return

    cfg = load_config()
    con = get_connection(db_path)

    states = {}
    for m in dd.machines(cfg):
        states[m["asset"]] = dd.latest_state(con, m["tag"])
    st.markdown("### Line schematic")
    st.components.v1.html(dd.schematic_svg(cfg, states), height=220)

    st.markdown("### Per-machine sensors")
    machine_names = [m["asset"] for m in dd.machines(cfg)]
    tabs = st.tabs(machine_names)
    for tab, m in zip(tabs, dd.machines(cfg)):
        with tab:
            rows = dd.latest_readings(con, m["tag"], limit=1000)
            if not rows:
                st.info("No sensor readings ingested yet for this machine.")
                continue
            summary = dd.ingest_summary(con, m["tag"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Accepted readings", summary["accepted_rows"])
            c1.caption(f"ingest categories: {summary['by_category']}")
            last_ts = summary["last_accepted_ts"]
            c2.metric("Last reading (UTC)", last_ts.strftime("%Y-%m-%d %H:%M:%S") if last_ts else "-")
            alerts = dd.alert_summary(con, m["tag"])
            c3.metric("Alerts (rule / model)", f"{alerts['counts'].get('rule', 0)} / "
                                               f"{alerts['counts'].get('model', 0)}")
            st.altair_chart(sensor_chart(rows), use_container_width=True)
            with st.expander("Raw readings table"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Alerts")
    all_alerts = dd.alert_summary(con, tag=None)["rows"]
    if all_alerts:
        st.dataframe(pd.DataFrame(all_alerts), use_container_width=True, height=260)
    else:
        st.info("No alerts logged yet.")

    st.markdown("### OEE (from the cleared plant ingest)")
    st.caption("Independently reviewed pipeline (vendor/plant-shift-oee-report), unmodified, scored "
              "on its own deterministic synthetic day - not the twin's own sensor data.")
    try:
        oee = dd.oee_report(db_path)
    except Exception as exc:   # noqa: BLE001  (surfaced to the operator, not swallowed)
        st.warning(f"OEE report unavailable: {exc}")
    else:
        st.caption(oee.get("method_note", ""))
        rows = []
        for machine, shifts in oee.get("machines", {}).items():
            for shift, s in shifts.items():
                if s.get("complete") and s.get("oee") is not None:
                    rows.append({"label": f"{machine} / {shift}", "oee": s["oee"]})
        if rows:
            st.altair_chart(oee_bar_chart(rows), use_container_width=True)
        incomplete = oee.get("completeness", {}).get("incomplete", [])
        if incomplete:
            st.warning(f"{len(incomplete)} machine/shift window(s) withheld as incomplete "
                      "(insufficient telemetry coverage) - see report detail below.")
        with st.expander("Full OEE report JSON"):
            st.json(oee)

    st.caption(CLAIM_NOTE)


if __name__ == "__main__":
    render(os.environ.get("TWIN_DB_PATH", "out/twin_demo.duckdb"))

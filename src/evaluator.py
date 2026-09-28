"""Pure scoring code (LOCK). Frozen by config.yaml `scoring:`.

The evaluator fits nothing and chooses no threshold. It scores whatever raw
thresholded alert windows it is given against the generator's label channel.
It is exercised in this milestone only on hand-built fixtures.

Inputs per stratum (one split x one asset), StratumInput:
  window_start   int64 epoch s of every declared 10 s window of the split
  eligible       bool per window (from features.py; warm-up / DQ excluded)
  dq_reason      str per window ('' if eligible)
  alert          bool per window: RAW thresholded prediction BEFORE debounce;
                 alert on an ineligible window is refused (no score exists)
  planned_stops  [(start, end)] non-operating intervals (label channel)
  normal_changes [(start, end)] unseen-normal-change intervals (label channel)

Input validation (frozen): window_start must be a non-empty, strictly
increasing, integer-dtype grid with a constant step of exactly
features.window_s - no missing, duplicate or misaligned steps - and (when
scored through score_all) must exactly span its declared split's [start,
end) bounds, endpoint to endpoint. eligible/alert must be bool-dtype arrays
of the same length as window_start; dq_reason, when given, likewise.
Malformed input is refused (InputError), never coerced into an apparently
valid score. point_level coverage = eligible windows / len(window_start);
once the grid is validated complete, this is coverage over the split's
actual declared grid, never 1.0 on a broken one.

Event level (per stratum; see config for exact definitions):
  * alert time of a window = its end; episodes = runs of alert windows with
    gap (next window start - previous window end) <= debounce_gap_s; never
    across assets or splits (each stratum is debounced on its own).
  * one-to-one matching: events by (onset, event_id); each takes the
    earliest UNMATCHED episode whose start is in [onset, end + tail].
    Unmatched events = FN; unmatched episodes = FP (including repeats inside
    an already-matched event). Pre-onset episodes are never credited.
  * F1 = 2TP/(2TP+FP+FN); precision = TP/(TP+FP); recall = TP/(TP+FN);
    any zero denominator -> None (reported as N/A, never 0).
Point level: a window is positive for a channel if it intersects any
[onset, end) of that channel; confusion over ELIGIBLE windows, with excluded
windows (and their truth) reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twin_config import load_config


class SplitOverlapError(ValueError):
    pass


class InputError(ValueError):
    pass


class PoolingError(ValueError):
    pass


@dataclass
class Event:
    event_id: str
    split: str
    asset: str
    fault_type: str
    channel: str          # "equipment" | "data"
    onset: int
    end: int              # exclusive label end


@dataclass
class StratumInput:
    split: str
    asset: str
    window_start: np.ndarray
    eligible: np.ndarray
    alert: np.ndarray
    dq_reason: np.ndarray | None = None
    planned_stops: list = field(default_factory=list)
    normal_changes: list = field(default_factory=list)


def ratio(num, den):
    return None if den == 0 else num / den


def _cfg(cfg):
    return cfg or load_config()


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------
def validate_splits(split_ranges: dict) -> None:
    """Refuse (raise) overlapping or inverted split ranges rather than score."""
    items = sorted(split_ranges.items(), key=lambda kv: kv[1][0])
    for n, (s, e) in items:
        if e <= s:
            raise SplitOverlapError(f"split {n} is empty or inverted")
    for (n1, (s1, e1)), (n2, (s2, e2)) in zip(items, items[1:]):
        if s2 < e1:
            raise SplitOverlapError(f"splits {n1} and {n2} overlap: refusing to score")


# ---------------------------------------------------------------------------
# Debounce and matching
# ---------------------------------------------------------------------------
def debounce(window_start: np.ndarray, alert: np.ndarray, cfg: dict | None = None) -> list[dict]:
    """Episodes for ONE stratum. Returns [{episode_id, start, end, first_window,
    n_windows}] where start = end time of the first alert window."""
    cfg = _cfg(cfg)
    bs = cfg["features"]["window_s"]
    gap_max = cfg["scoring"]["debounce_gap_s"]
    ws = np.asarray(window_start)[np.asarray(alert, dtype=bool)]
    if len(ws) and np.any(np.diff(ws) <= 0):
        raise InputError("window_start must be strictly increasing")
    eps: list[dict] = []
    for a in ws:
        a = int(a)
        if eps and (a - eps[-1]["end"]) <= gap_max:
            eps[-1]["end"] = a + bs
            eps[-1]["n_windows"] += 1
        else:
            eps.append({"episode_id": len(eps), "first_window": a, "start": a + bs,
                        "end": a + bs, "n_windows": 1})
    return eps


def match_events(events: list[Event], episodes: list[dict], cfg: dict | None = None) -> list[dict]:
    """One-to-one matching for ONE stratum. Returns the complete match table."""
    cfg = _cfg(cfg)
    tail = cfg["scoring"]["scoring_tail_s"]
    order = sorted(events, key=lambda e: (e.onset, e.event_id))
    used: set[int] = set()
    table = []
    for ev in order:
        hit = None
        for ep in episodes:                       # episodes are in start order
            if ep["episode_id"] in used:
                continue
            if ev.onset <= ep["start"] <= ev.end + tail:
                hit = ep
                break
        overlaps = sorted(o.event_id for o in events if o.event_id != ev.event_id
                          and o.onset <= ev.end + tail and ev.onset <= o.end + tail)
        row = {"event_id": ev.event_id, "fault_type": ev.fault_type, "channel": ev.channel,
               "onset": ev.onset, "end": ev.end, "overlapping_event_ids": overlaps,
               "matched_episode_id": None, "matched_alert_start": None, "delay_s": None}
        if hit is not None:
            used.add(hit["episode_id"])
            row.update(matched_episode_id=hit["episode_id"], matched_alert_start=hit["start"],
                       delay_s=hit["start"] - ev.onset)
        table.append(row)
    return table


def _in_intervals(t: np.ndarray, intervals) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for s, e in intervals:
        m |= (t >= s) & (t < e)
    return m


def _quantiles(values, cfg):
    sc = cfg["scoring"]
    if not values:
        return {f"p{int(q * 100)}": None for q in sc["delay_quantiles"]}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(q * 100)}": float(np.quantile(arr, q, method=sc["delay_quantile_method"]))
            for q in sc["delay_quantiles"]}


def _prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


# ---------------------------------------------------------------------------
# Stratum scoring
# ---------------------------------------------------------------------------
def _validate_grid(ws: np.ndarray, bs: int) -> None:
    if ws.ndim != 1:
        raise InputError("window_start must be a 1-D array")
    if not np.issubdtype(ws.dtype, np.integer):
        raise InputError(f"window_start must be an integer dtype, got {ws.dtype}")
    if len(ws) == 0:
        raise InputError("stratum has no declared windows: refusing to score an empty grid")
    expected = ws[0] + bs * np.arange(len(ws), dtype=np.int64)
    if not np.array_equal(ws, expected):
        raise InputError("window_start is not a complete, unique, strictly increasing grid "
                          f"with a constant step of {bs}s: refusing to coerce a broken grid")


def score_stratum(inp: StratumInput, events: list[Event], cfg: dict | None = None) -> dict:
    cfg = _cfg(cfg)
    bs = cfg["features"]["window_s"]
    tail = cfg["scoring"]["scoring_tail_s"]
    ws = np.asarray(inp.window_start)
    _validate_grid(ws, bs)
    ws = ws.astype(np.int64)
    elig_raw = np.asarray(inp.eligible)
    alert_raw = np.asarray(inp.alert)
    if elig_raw.dtype != np.bool_:
        raise InputError(f"eligible must be a bool-dtype array, got {elig_raw.dtype}")
    if alert_raw.dtype != np.bool_:
        raise InputError(f"alert must be a bool-dtype array, got {alert_raw.dtype}")
    elig, alert = elig_raw, alert_raw
    if elig.ndim != 1 or alert.ndim != 1:
        raise InputError("eligible and alert must be exact 1-D arrays")
    if not (len(ws) == len(elig) == len(alert)):
        raise InputError("window_start/eligible/alert lengths differ")
    if np.any(alert & ~elig):
        raise InputError("alert on an ineligible window: no score exists there")
    if inp.dq_reason is not None:
        _dq = np.asarray(inp.dq_reason, dtype=object)
        if _dq.ndim != 1:
            raise InputError("dq_reason must be an exact 1-D array")
        if len(_dq) != len(ws):
            raise InputError("dq_reason length differs from window_start")
    reason = (np.asarray(inp.dq_reason, dtype=object) if inp.dq_reason is not None
              else np.where(elig, "", "excluded").astype(object))
    evs = [e for e in events if e.split == inp.split and e.asset == inp.asset]
    if len(ws):
        lo, hi = int(ws[0]), int(ws[-1]) + bs
        for e in evs:
            if e.onset < lo or e.end + tail > hi:
                raise InputError(f"event {e.event_id} (+ tail) is not inside its split")
    ids = [e.event_id for e in evs]
    if len(set(ids)) != len(ids):
        raise InputError("duplicate event IDs")

    stop = _in_intervals(ws, inp.planned_stops)
    change = _in_intervals(ws, inp.normal_changes)
    oper = ~stop

    # ---- event level ----
    eps = debounce(ws, alert, cfg)
    table = match_events(evs, eps, cfg)
    matched_eps = {r["matched_episode_id"] for r in table if r["matched_episode_id"] is not None}
    idx = {int(a): i for i, a in enumerate(ws)}
    for ep in eps:
        i = idx[ep["first_window"]]
        ep["in_planned_stop"] = bool(stop[i])
        ep["in_normal_change"] = bool(change[i])
        ep["matched"] = ep["episode_id"] in matched_eps
    tp = len(matched_eps)
    fn = sum(1 for r in table if r["matched_episode_id"] is None)
    fp = len(eps) - tp
    op_hours = int((elig & oper).sum()) * bs / 3600.0
    stop_hours = int((elig & stop).sum()) * bs / 3600.0
    change_hours = int((elig & change).sum()) * bs / 3600.0
    fp_op = sum(1 for ep in eps if not ep["matched"] and not ep["in_planned_stop"])
    fp_stop = sum(1 for ep in eps if not ep["matched"] and ep["in_planned_stop"])
    fp_change = sum(1 for ep in eps if not ep["matched"] and ep["in_normal_change"])
    delays = [r["delay_s"] for r in table if r["delay_s"] is not None]

    by_type: dict = {}
    for r in table:
        d = by_type.setdefault(r["fault_type"], {"events": 0, "tp": 0})
        d["events"] += 1
        d["tp"] += r["matched_episode_id"] is not None
    for d in by_type.values():
        d["fn"] = d["events"] - d["tp"]
        d["recall"] = ratio(d["tp"], d["events"])
    by_channel: dict = {}
    for ch in cfg["scoring"]["point_channels"]:
        rows = [r for r in table if r["channel"] == ch]
        k = sum(1 for r in rows if r["matched_episode_id"] is not None)
        by_channel[ch] = {"events": len(rows), "tp": k, "fn": len(rows) - k,
                          "recall": ratio(k, len(rows))}

    event_level = {
        **_prf(tp, fp, fn),
        "events": len(evs), "episodes": len(eps),
        "precision_kind": "generic alert precision (no predicted fault type)",
        "recall_by_fault_type": by_type, "recall_by_channel": by_channel,
        "false_alerts_operating": fp_op,
        "operating_hours_observed_eligible": op_hours,
        "false_alerts_per_operating_hour": ratio(fp_op, op_hours),
        "planned_stop": {"unmatched_episodes": fp_stop,
                         "episodes_starting": sum(1 for ep in eps if ep["in_planned_stop"]),
                         "observed_hours": stop_hours,
                         "false_alerts_per_observed_hour": ratio(fp_stop, stop_hours)},
        "unseen_normal_change": {"unmatched_episodes": fp_change,
                                 "alert_active_seconds": int((alert & change).sum()) * bs,
                                 "observed_hours": change_hours,
                                 "false_alerts_per_observed_hour": ratio(fp_change, change_hours)},
        "alert_active_fraction_operating": ratio(int((alert & elig & oper).sum()),
                                                 int((elig & oper).sum())),
        "delay": {"matched_events": len(delays), "events": len(evs),
                  "miss_rate": ratio(fn, len(evs)), **_quantiles(delays, cfg),
                  "quantile_method": cfg["scoring"]["delay_quantile_method"]},
        "match_table": table,
        "episodes_table": eps,
    }

    # ---- point level ----
    point: dict = {}
    for ch in cfg["scoring"]["point_channels"]:
        truth = np.zeros(len(ws), dtype=bool)
        for e in evs:
            if e.channel == ch:
                truth |= (ws < e.end) & (ws + bs > e.onset)
        t, p = truth[elig], alert[elig]
        ptp, pfp = int((t & p).sum()), int((~t & p).sum())
        pfn, ptn = int((t & ~p).sum()), int((~t & ~p).sum())
        excl = {}
        for r in sorted(set(reason[~elig].tolist())):
            m = (~elig) & (reason == r)
            excl[r] = {"windows": int(m.sum()), "positive_windows": int((m & truth).sum())}
        point[ch] = {"tp": ptp, "fp": pfp, "fn": pfn, "tn": ptn,
                     "precision": ratio(ptp, ptp + pfp), "recall": ratio(ptp, ptp + pfn),
                     "f1": ratio(2 * ptp, 2 * ptp + pfp + pfn),
                     "eligible_windows": int(elig.sum()),
                     "eligible_positive_windows": int(t.sum()),
                     "excluded_windows": int((~elig).sum()),
                     "excluded_positive_windows": int((truth & ~elig).sum()),
                     "excluded_by_reason": excl,
                     "coverage": ratio(int(elig.sum()), len(ws))}
    return {"split": inp.split, "asset": inp.asset, "windows_total": int(len(ws)),
            "event_level": event_level, "point_level": point}


def score_all(strata: list[StratumInput], events: list[Event], split_ranges: dict,
              cfg: dict | None = None) -> dict:
    """Score every stratum separately. Refuses overlapping splits, windows
    outside their split, events not covered by any stratum, and duplicate
    strata. Never pools strata."""
    cfg = _cfg(cfg)
    bs = cfg["features"]["window_s"]
    validate_splits(split_ranges)
    seen = set()
    for s in strata:
        if s.split not in split_ranges:
            raise InputError(f"unknown split {s.split}")
        if (s.split, s.asset) in seen:
            raise InputError(f"duplicate stratum {s.split}/{s.asset}")
        seen.add((s.split, s.asset))
        lo, hi = split_ranges[s.split]
        ws = np.asarray(s.window_start)
        if len(ws) == 0 or ws.min() != lo or ws.max() + bs != hi:
            raise InputError(f"stratum {s.split}/{s.asset} window_start does not exactly span "
                             f"its declared split bounds [{lo}, {hi})")
    for e in events:
        if (e.split, e.asset) not in seen:
            raise InputError(f"event {e.event_id} has no stratum")
    return {f"{s.split}/{s.asset}": score_stratum(s, events, cfg) for s in strata}


def pool_counts(results: list[dict], test_split: str = "test") -> dict:
    """Sum event-level counts across strata (validation selection only).
    Refuses to pool any test stratum: seen/unseen test assets are reported
    separately, never as one asset-generalisation number."""
    if any(r["split"] == test_split for r in results):
        raise PoolingError("test strata are never pooled")
    tp = sum(r["event_level"]["tp"] for r in results)
    fp = sum(r["event_level"]["fp"] for r in results)
    fn = sum(r["event_level"]["fn"] for r in results)
    fa = sum(r["event_level"]["false_alerts_operating"] for r in results)
    hrs = sum(r["event_level"]["operating_hours_observed_eligible"] for r in results)
    eps = sum(r["event_level"]["episodes"] for r in results)
    return {**_prf(tp, fp, fn), "false_alerts_operating": fa, "operating_hours": hrs,
            "false_alerts_per_operating_hour": ratio(fa, hrs), "total_alerts": eps}


# ---------------------------------------------------------------------------
# Plumbing from the generator's label channel (scoring context only)
# ---------------------------------------------------------------------------
def events_from_labels(labels) -> list[Event]:
    return [Event(e.event_id, e.split, e.asset, e.fault_type, e.channel, e.onset, e.end)
            for e in labels.events]


def stratum_input(frame, alert: np.ndarray, labels, split: str, asset: str) -> StratumInput:
    """Build a StratumInput from a features.FeatureFrame (eligibility, dq
    reasons), a raw alert vector over ALL of the frame's windows, and the
    label channel's scoring context (planned stops, normal changes)."""
    stops = [(p["start"], p["end"]) for p in labels.planned_stops
             if p["split"] == split and p["asset"] == asset]
    changes = [(c["start"], c["end"]) for c in labels.normal_changes
               if c["split"] == split and c["asset"] == asset]
    # Do not coerce: a float/NaN alert vector must reach score-time validation as-is and be refused there
    # (Astra lock-r2: [NaN,0,0] must never become [True,False,False]).
    return StratumInput(split, asset, frame.window_start, frame.eligible, np.asarray(alert),
                        dq_reason=frame.dq_reason, planned_stops=stops, normal_changes=changes)


# ---------------------------------------------------------------------------
# Baselines as scoring inputs (not models)
# ---------------------------------------------------------------------------
def no_alert(inp_eligible: np.ndarray) -> np.ndarray:
    return np.zeros(len(inp_eligible), dtype=bool)


def always_alert(inp_eligible: np.ndarray) -> np.ndarray:
    """Constant detector: alerts on every eligible window (it has no
    operating-state input, so it also alerts in planned stops)."""
    return np.asarray(inp_eligible, dtype=bool).copy()


# ---------------------------------------------------------------------------
# Candidate selection rule (exercised ONLY on hand-built fixtures here)
# ---------------------------------------------------------------------------
def select_candidate(candidates: list[dict], cfg: dict | None = None) -> dict:
    """candidates: [{candidate_id, strictness, tp, fp, fn,
    false_alerts_per_operating_hour, total_alerts}] - validation counts.
    A 'no_alert' candidate is always added if absent. Rule: max event F1
    among candidates within the false-alert cap; ties -> fewer false alerts
    (fp) -> fewer total alerts -> stricter (higher strictness). Returns
    {selected, table} with every candidate retained."""
    cfg = _cfg(cfg)
    cap = cfg["scoring"]["false_alert_cap_per_operating_hour"]
    cands = [dict(c) for c in candidates]
    if cfg["scoring"]["no_alert_always_candidate"] and not any(c["candidate_id"] == "no_alert" for c in cands):
        fn = max((c["tp"] + c["fn"] for c in cands), default=0)
        cands.append({"candidate_id": "no_alert", "strictness": float("inf"), "tp": 0, "fp": 0,
                      "fn": fn, "false_alerts_per_operating_hour": 0.0, "total_alerts": 0})
    for c in cands:
        c["f1"] = ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])
        fa = c["false_alerts_per_operating_hour"]
        c["within_cap"] = fa is not None and fa <= cap
    ok = [c for c in cands if c["within_cap"]]
    if not ok:
        return {"selected": None, "table": cands, "note": "no candidate meets the false-alert cap"}
    best = min(ok, key=lambda c: (-(c["f1"] if c["f1"] is not None else -1.0), c["fp"],
                                  c["total_alerts"], -c["strictness"]))
    return {"selected": best["candidate_id"], "table": cands}

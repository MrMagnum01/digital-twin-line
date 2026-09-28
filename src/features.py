"""Causal feature pipeline (LOCK). Frozen by config.yaml `features:`.

Per (split, asset) the input is a SensorReadings object from the RAW channel
only - exactly the split's 1 Hz declared grid, NaN for a missing sample.
Nothing from the label channel (events, schedule, seeds, asset identity) is
accepted or produced as a feature column; assert_no_leak() enforces this.

Window convention: the output grid has one row per 10 s window k covering
grid seconds [a_k, a_k + 10). Its features are emitted at e_k = a_k + 10 and
use only raw 1 Hz samples with split_start <= t < e_k. The 5-minute and
30-minute rolling summaries at e_k cover e_k - W <= t < e_k, clipped at the
split start. Every call starts from empty state, so nothing carries across a
split boundary; readings that do not exactly span the split are refused.

Per sensor and rolling window W in {5m, 30m}: mean, population std (ddof=0)
and OLS slope (value per second) over the finite samples in the window,
emitted only when finite samples >= min_observations[W] (else NaN and the
window is ineligible: insufficient_observations). missing_frac for the 10 s
window and each W = 1 - finite samples / declared grid seconds in the
window. +/-inf in a rolling window makes it ineligible (non_finite_input).
No imputation. The first 30 minutes of every split are ineligible (warm_up).
Ineligible windows are never "healthy": they carry a labelled dq_reason and
are reported separately by excluded_report().

Current-window health (frozen): the rolling 5m/30m statistics can stay
finite from OLDER samples while the window's OWN 10 s block is currently
blind, so eligibility also depends on this window's own per-sensor sample
count, not only the rolling counts - a live sensor dropout can never be
reported healthy on stale history. Per window: total loss (every sensor has
zero finite samples in this 10 s block) -> ineligible, dq_reason
"all_sensors_missing". Partial loss (at least one, but not all, sensors have
zero finite samples in this 10 s block) -> ineligible, dq_reason
"partial_sensor_missing" ("missing sensor data must never render healthy
green" applies whether one sensor or all are currently blind). dq_reason
priority when more than one condition holds for the same window (highest
wins): warm_up > all_sensors_missing > partial_sensor_missing >
non_finite_input > insufficient_observations.

Computation is exact-per-window: 10 s block statistics (count, mean, centred
second moments of value and time, co-moment) are combined over the window's
blocks with the parallel (Chan et al.) update. Each window's value depends
only on its own blocks, so perturbing a later sample cannot change an
earlier output (tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from generator import SensorReadings
from twin_config import load_config

_CHUNK = 8192


class LeakError(AssertionError):
    pass


class TrainOnlyError(ValueError):
    pass


class SplitSpanError(ValueError):
    pass


@dataclass
class FeatureFrame:
    window_start: np.ndarray            # int64 epoch s, a_k
    columns: dict                       # feature name -> float64 array (allow-listed)
    dq_reason: np.ndarray               # '' if eligible, else first reason
    eligible: np.ndarray                # bool
    meta: dict = field(default_factory=dict)   # non-feature bookkeeping (never a column)


def feature_columns(cfg: dict | None = None) -> list[str]:
    """The exact allow-list of output feature columns, in fixed order."""
    cfg = cfg or load_config()
    f = cfg["features"]
    cols = []
    for s in cfg["sensors"]["order"]:
        for w in f["rolling_windows_s"]:
            for stat in f["stats"]:
                cols.append(f"{s}__{stat}_{w}")
        for w in f["missing_fraction"]["windows"]:
            cols.append(f"{s}__missing_frac_{w}")
    return cols


def assert_no_leak(columns, cfg: dict | None = None) -> None:
    """Raise LeakError unless `columns` is exactly the allow-list and no name
    contains a denied (label/asset/seed/schedule/...) substring."""
    cfg = cfg or load_config()
    cols = list(columns)
    deny = [d.lower() for d in cfg["features"]["deny_substrings"]]
    for c in cols:
        low = str(c).lower()
        for d in deny:
            if d in low:
                raise LeakError(f"feature column {c!r} contains denied token {d!r}")
    allowed = feature_columns(cfg)
    if cols != allowed:
        extra = sorted(set(cols) - set(allowed))
        missing = sorted(set(allowed) - set(cols))
        raise LeakError(f"feature columns differ from allow-list (extra={extra}, missing={missing})")


def _block_stats(x: np.ndarray, bs: int):
    xb = x.reshape(-1, bs)
    valid = np.isfinite(xb)
    ninf = np.isinf(xb).sum(axis=1).astype(np.float64)
    n = valid.sum(axis=1).astype(np.float64)
    safe_n = np.where(n > 0, n, 1.0)
    xs = np.where(valid, xb, 0.0)
    xbar = xs.sum(axis=1) / safe_n
    dx = np.where(valid, xb - xbar[:, None], 0.0)
    u = np.arange(bs, dtype=np.float64)
    ubar = np.where(valid, u, 0.0).sum(axis=1) / safe_n
    du = np.where(valid, u[None, :] - ubar[:, None], 0.0)
    m2 = (dx * dx).sum(axis=1)
    mtt = (du * du).sum(axis=1)
    mtx = (du * dx).sum(axis=1)
    return n, xbar, ubar, m2, mtt, mtx, ninf


def _rolling(stats, k_blocks: int, bs: int):
    n, xbar, ubar, m2, mtt, mtx, ninf = stats
    pad = k_blocks - 1
    P = [np.concatenate([np.zeros(pad), a]) for a in (n, xbar, ubar, m2, mtt, mtx, ninf)]
    V = [sliding_window_view(a, k_blocks) for a in P]
    rows = len(n)
    offs = (bs * np.arange(k_blocks, dtype=np.float64))[None, :]
    out = {k: np.empty(rows) for k in ("N", "mean", "M2", "Stt", "Stx", "ninf")}
    with np.errstate(invalid="ignore", divide="ignore"):
        for r0 in range(0, rows, _CHUNK):
            r1 = min(rows, r0 + _CHUNK)
            vn, vx, vu, vm2, vtt, vtx, vinf = (v[r0:r1] for v in V)
            N = vn.sum(axis=1)
            tb = offs + vu
            mean = (vn * vx).sum(axis=1) / N
            tbar = (vn * tb).sum(axis=1) / N
            ddx = vx - mean[:, None]
            ddt = tb - tbar[:, None]
            out["N"][r0:r1] = N
            out["mean"][r0:r1] = mean
            out["M2"][r0:r1] = vm2.sum(axis=1) + (vn * ddx * ddx).sum(axis=1)
            out["Stt"][r0:r1] = vtt.sum(axis=1) + (vn * ddt * ddt).sum(axis=1)
            out["Stx"][r0:r1] = vtx.sum(axis=1) + (vn * ddt * ddx).sum(axis=1)
            out["ninf"][r0:r1] = vinf.sum(axis=1)
    return out


def compute_features(readings: SensorReadings, split_start: int, split_end: int, *,
                     split_name: str | None = None, cfg: dict | None = None) -> FeatureFrame:
    cfg = cfg or load_config()
    if not isinstance(readings, SensorReadings):
        raise TypeError("compute_features accepts only raw-channel SensorReadings")
    f = cfg["features"]
    sensors = cfg["sensors"]["order"]
    bs = f["window_s"]
    if set(readings.values) != set(sensors):
        raise ValueError(f"readings must carry exactly the sensors {sensors}")
    if readings.t0 != split_start or readings.n != split_end - split_start:
        raise SplitSpanError("readings must span exactly [split_start, split_end) - no carry across splits")
    if (split_end - split_start) % bs:
        raise SplitSpanError("split length must be a whole number of 10 s windows")
    n_win = (split_end - split_start) // bs
    window_start = split_start + bs * np.arange(n_win, dtype=np.int64)
    blocks_seen = np.arange(1, n_win + 1, dtype=np.float64)   # blocks available since split start

    cols: dict = {}
    insufficient = np.zeros(n_win, dtype=bool)
    nonfinite = np.zeros(n_win, dtype=bool)
    cur_sensor_missing = np.zeros(n_win, dtype=np.int64)   # sensors with 0 finite samples THIS 10s block
    finite_total = 0
    for s in sensors:
        x = np.asarray(readings.values[s], dtype=np.float64)
        st = _block_stats(x, bs)
        finite_total += int(st[0].sum())
        cur_sensor_missing += (st[0] == 0)
        rolled = {}
        for w, W in f["rolling_windows_s"].items():
            K = W // bs
            r = _rolling(st, K, bs)
            rolled[w] = (r, K)
            ok = (r["N"] >= f["min_observations"][w]) & (r["Stt"] > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(ok, r["mean"], np.nan)
                std = np.where(ok, np.sqrt(np.maximum(r["M2"], 0.0) / r["N"]), np.nan)
                slope = np.where(ok, r["Stx"] / r["Stt"], np.nan)
            cols[f"{s}__mean_{w}"] = mean
            cols[f"{s}__std_{w}"] = std
            cols[f"{s}__slope_{w}"] = slope
            insufficient |= ~ok
            nonfinite |= r["ninf"] > 0
        for w in f["missing_fraction"]["windows"]:
            if w == f"{bs}s":
                cols[f"{s}__missing_frac_{w}"] = 1.0 - st[0] / bs
            else:
                r, K = rolled[w]
                expected = np.minimum(K, blocks_seen) * bs
                cols[f"{s}__missing_frac_{w}"] = 1.0 - r["N"] / expected
    ordered = {c: cols[c] for c in feature_columns(cfg)}
    assert_no_leak(ordered.keys(), cfg)

    warm_end = split_start + cfg["generator"]["warmup_minutes"] * 60
    n_sensors = len(sensors)
    all_sensors_missing = cur_sensor_missing == n_sensors
    partial_sensor_missing = (cur_sensor_missing > 0) & ~all_sensors_missing
    reason = np.full(n_win, "", dtype=object)
    reason[insufficient] = "insufficient_observations"
    reason[nonfinite] = "non_finite_input"
    reason[partial_sensor_missing] = "partial_sensor_missing"
    reason[all_sensors_missing] = "all_sensors_missing"
    reason[window_start < warm_end] = "warm_up"
    eligible = reason == ""
    meta = {"split": split_name, "split_start": split_start, "split_end": split_end,
            "warmup_end": warm_end, "grid_samples": len(sensors) * (split_end - split_start),
            "finite_samples": finite_total}
    return FeatureFrame(window_start, ordered, reason, eligible, meta)


def features_for_dataset(ds, cfg: dict | None = None) -> dict:
    """Features for every (split, asset) of a generated dataset. Reads ONLY
    ds.readings (raw channel) and each split's [start, end) bounds; the label
    channel ds.labels is never touched. One independent call per split, so
    feature state resets at every split boundary."""
    cfg = cfg or load_config()
    out = {}
    for (split, asset), r in sorted(ds.readings.items()):
        sp = ds.splits[split]
        out[(split, asset)] = compute_features(r, sp.start, sp.end, split_name=split, cfg=cfg)
    return out


def excluded_report(frame: FeatureFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    bs = cfg["features"]["window_s"]
    out = {"windows_total": int(len(frame.window_start)),
           "windows_eligible": int(frame.eligible.sum()), "excluded": {}}
    for r in cfg["features"]["dq_reasons"]:
        n = int((frame.dq_reason == r).sum())
        out["excluded"][r] = {"windows": n, "seconds": n * bs}
    out["observed_sample_fraction"] = (frame.meta["finite_samples"] / frame.meta["grid_samples"]
                                       if frame.meta.get("grid_samples") else None)
    return out


def to_matrix(frame: FeatureFrame, columns: list[str] | None = None) -> np.ndarray:
    """Eligible rows only, allow-listed columns only."""
    assert_no_leak(frame.columns.keys())
    cols = columns if columns is not None else list(frame.columns)
    for c in cols:
        if c not in frame.columns:
            raise LeakError(f"{c!r} is not an allow-listed feature column")
    return np.column_stack([frame.columns[c][frame.eligible] for c in cols]) if cols else \
        np.empty((int(frame.eligible.sum()), 0))


@dataclass
class FittedScaler:
    scaler: object
    columns: list          # kept columns, in order
    dropped: list          # zero-variance columns dropped by the frozen rule
    n_fit_rows: int


def fit_scaler(train_frames: list[FeatureFrame], cfg: dict | None = None) -> FittedScaler:
    """Fit the pinned scaler on TRAIN eligible windows only. Refuses any
    frame not tagged split='train'. Applies the frozen zero-variance rule."""
    from sklearn.preprocessing import RobustScaler

    cfg = cfg or load_config()
    sc = cfg["features"]["scaler"]
    if sc["name"] != "sklearn.preprocessing.RobustScaler":
        raise ValueError("config names a scaler this code does not implement")
    if not train_frames:
        raise TrainOnlyError("no train frames")
    for fr in train_frames:
        if fr.meta.get("split") != "train":
            raise TrainOnlyError(f"scaler may be fitted on train only, got split={fr.meta.get('split')!r}")
    cols = feature_columns(cfg)
    X = np.vstack([to_matrix(fr, cols) for fr in train_frames])
    if X.shape[0] == 0:
        raise TrainOnlyError("no eligible train windows")
    q = np.percentile(X, [25.0, 75.0], axis=0, method="linear")
    iqr = q[1] - q[0]
    eps = cfg["features"]["zero_variance"]["epsilon"]
    keep = [c for c, v in zip(cols, iqr) if v > eps]
    dropped = [c for c, v in zip(cols, iqr) if not v > eps]
    p = dict(sc["params"])
    p["quantile_range"] = tuple(p["quantile_range"])
    scaler = RobustScaler(**p).fit(X[:, [cols.index(c) for c in keep]])
    return FittedScaler(scaler, keep, dropped, int(X.shape[0]))


def transform(fs: FittedScaler, frame: FeatureFrame) -> np.ndarray:
    return fs.scaler.transform(to_matrix(frame, fs.columns))

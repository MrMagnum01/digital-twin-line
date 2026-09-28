"""Operational IsolationForest baseline (NOT a locked file; NOT the frozen
protocol's train/validation/test selection - see monitoring/__init__.py).

Fits a scaler + IsolationForest ONCE on a fixed warm-up slice (the first
`warmup_windows` eligible windows, in time order) of one asset's own
feature history, then scores every eligible window against that fixed
baseline - including the warm-up slice itself, whose own scores are
NON-causal (in-sample/retrospective): each of those windows' fitted score
depends on later warm-up samples used to fit the very model scoring it, so
it is not a claim about what the baseline would have alerted on at the
time. `score_frame` marks those windows non-causal and never alerts on
them (Astra freeze-review MUST-FIX 5, 2026-09-28); their scores are still
returned for visibility, just excluded from causal alert/performance
claims. IsolationForest's parameters, including contamination, are taken
as-is from config.yaml `models.isolation_forest.params` and never tuned
here; the alert threshold is the strictest train-score quantile in
config.yaml's frozen candidate grid, not a validation-selected one (there
is no validation split in a live operational baseline).
"""
from __future__ import annotations

import numpy as np


class InsufficientBaselineError(RuntimeError):
    pass


class NonFiniteBaselineError(RuntimeError):
    """Raised when a fitted baseline's own train scores/threshold, or an
    eligible window's scored value, comes out non-finite. Mirrors
    runner.models.NonFiniteEligibleInputError: a failed measurement must
    raise or be reported unknown, never silently become alert=False (Astra
    freeze-review r2 group 2, 2026-09-28 - "monitor scores must raise
    everywhere, never turn into False")."""


def fit_baseline(frame, cfg: dict, warmup_windows: int = 180) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    from features import FeatureFrame, feature_columns, to_matrix

    eligible = np.asarray(frame.eligible, dtype=bool)
    elig_idx = np.flatnonzero(eligible)
    if len(elig_idx) < warmup_windows:
        raise InsufficientBaselineError(
            f"only {len(elig_idx)} eligible windows available for {frame.meta.get('split')!r}, "
            f"need at least {warmup_windows} for an operational baseline")
    warm_mask = np.zeros(len(eligible), dtype=bool)
    warm_mask[elig_idx[:warmup_windows]] = True
    warm_frame = FeatureFrame(frame.window_start, frame.columns, frame.dq_reason, warm_mask, frame.meta)

    cols = feature_columns(cfg)
    X = to_matrix(warm_frame, cols)
    sc = cfg["features"]["scaler"]
    q = np.percentile(X, [25.0, 75.0], axis=0, method="linear")
    iqr = q[1] - q[0]
    eps = cfg["features"]["zero_variance"]["epsilon"]
    keep = [c for c, v in zip(cols, iqr) if v > eps]
    params = dict(sc["params"])
    params["quantile_range"] = tuple(params["quantile_range"])
    X_keep = X[:, [cols.index(c) for c in keep]]
    scaler = RobustScaler(**params).fit(X_keep)

    if_params = dict(cfg["models"]["isolation_forest"]["params"])
    model = IsolationForest(**if_params)
    Xs = scaler.transform(X_keep)
    model.fit(Xs)
    train_scores = -model.score_samples(Xs)
    if not np.all(np.isfinite(train_scores)):
        raise NonFiniteBaselineError(
            "operational baseline train anomaly scores contain non-finite values: refusing to "
            "derive a threshold from them")

    tc = cfg["models"]["isolation_forest"]["threshold_candidates"]
    thr_q = tc["quantiles"][-1]     # strictest candidate: fewest baseline false alerts
    threshold = float(np.quantile(train_scores, thr_q, method=tc["quantile_method"]))
    if not np.isfinite(threshold):
        raise NonFiniteBaselineError(
            f"operational baseline threshold={threshold!r} is non-finite: refusing to score "
            "against an undefined threshold rather than silently returning no-alert")
    fit_window_starts = frozenset(int(x) for x in frame.window_start[warm_mask])
    fit_boundary = int(frame.window_start[warm_mask].max())
    return {"scaler": scaler, "keep_columns": keep, "model": model, "threshold": threshold,
           "threshold_quantile": thr_q, "warmup_windows": warmup_windows,
           "fit_window_starts": fit_window_starts, "fit_boundary": fit_boundary}


def score_frame(frame, baseline: dict, cfg: dict):
    """Score every eligible window of `frame` against a fixed fitted
    baseline (fit_baseline()'s return dict). Returns (alert, scores,
    causal): bool, float and bool arrays the length of `frame`; scores is
    NaN where ineligible. `causal` is False for ineligible windows AND for
    every window at or before `fit_boundary` (the end of the baseline's own
    fitting interval) - not merely the exact warm-up window_starts
    (fit_window_starts). Excluding only the exact fit set is insufficient
    for scoring EARLIER history: a timestamp before the fitting period but
    not itself one of the fit windows (e.g. a gap, or a separate, earlier
    frame passed to this same fixed baseline) would otherwise be marked
    causal=True, even though the baseline did not exist yet at that time
    (Astra freeze-review r2 group 5, 2026-09-28). Those earlier/in-sample
    scores are still returned for visibility, just excluded from causal
    alert/performance claims. An eligible window whose score comes out
    non-finite raises rather than silently scoring as no-alert/healthy
    (mirrors runner.models' eligible-input contract; r2 group 2)."""
    from features import to_matrix

    eligible = np.asarray(frame.eligible, dtype=bool)
    n = len(eligible)
    scores = np.full(n, np.nan)
    if eligible.any():
        X = to_matrix(frame, baseline["keep_columns"])
        Xs = baseline["scaler"].transform(X)
        scores[eligible] = -baseline["model"].score_samples(Xs)
    bad = eligible & ~np.isfinite(scores)
    if bad.any():
        raise NonFiniteBaselineError(
            f"operational baseline anomaly score is non-finite on {int(bad.sum())} eligible "
            "window(s): an eligible window must never carry a non-finite score")
    fit_boundary = baseline.get("fit_boundary")
    if fit_boundary is None:
        fit_starts = baseline["fit_window_starts"]
        after_fit = np.array([int(ws) not in fit_starts for ws in frame.window_start], dtype=bool)
    else:
        after_fit = np.asarray(frame.window_start, dtype=np.int64) > fit_boundary
    causal = eligible & after_fit
    if not np.isfinite(baseline["threshold"]):
        raise NonFiniteBaselineError(
            f"operational baseline threshold={baseline['threshold']!r} is non-finite: refusing "
            "to score against an undefined threshold rather than silently returning no-alert")
    with np.errstate(invalid="ignore"):
        beyond = scores > baseline["threshold"]   # NaN comparisons only remain on ineligible rows
    alert = beyond & causal
    return alert, scores, causal

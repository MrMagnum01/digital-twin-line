"""Operational IsolationForest baseline (NOT a locked file; NOT the frozen
protocol's train/validation/test selection - see monitoring/__init__.py).

Fits a scaler + IsolationForest ONCE on a fixed, causal warm-up slice (the
first `warmup_windows` eligible windows, in time order) of one asset's own
feature history, then scores every eligible window - including the warm-up
slice itself - against that fixed baseline. IsolationForest's parameters,
including contamination, are taken as-is from config.yaml
`models.isolation_forest.params` and never tuned here; the alert threshold
is the strictest train-score quantile in config.yaml's frozen candidate
grid, not a validation-selected one (there is no validation split in a live
operational baseline).
"""
from __future__ import annotations

import numpy as np


class InsufficientBaselineError(RuntimeError):
    pass


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

    tc = cfg["models"]["isolation_forest"]["threshold_candidates"]
    thr_q = tc["quantiles"][-1]     # strictest candidate: fewest baseline false alerts
    threshold = float(np.quantile(train_scores, thr_q, method=tc["quantile_method"]))
    return {"scaler": scaler, "keep_columns": keep, "model": model, "threshold": threshold,
           "threshold_quantile": thr_q, "warmup_windows": warmup_windows}


def score_frame(frame, baseline: dict, cfg: dict):
    """Score every eligible window of `frame` against a fixed fitted
    baseline (fit_baseline()'s return dict). Returns (alert, scores): bool
    and float arrays the length of `frame`; scores is NaN where ineligible."""
    from features import to_matrix

    eligible = np.asarray(frame.eligible, dtype=bool)
    n = len(eligible)
    scores = np.full(n, np.nan)
    if eligible.any():
        X = to_matrix(frame, baseline["keep_columns"])
        Xs = baseline["scaler"].transform(X)
        scores[eligible] = -baseline["model"].score_samples(Xs)
    with np.errstate(invalid="ignore"):
        beyond = scores > baseline["threshold"]
    alert = np.where(np.isfinite(scores), beyond, False) & eligible
    return alert, scores

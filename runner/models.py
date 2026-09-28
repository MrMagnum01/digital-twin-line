"""Model fitting and threshold-candidate construction (NOT a locked file).

Implements the frozen protocol's two selection-eligible models exactly as
config.yaml `models:` and `scoring:` freeze them:

  * static_threshold - per-sensor mean +/- k*sigma on {sensor}__mean_5m,
    train mean/std, k from the frozen k_grid.
  * isolation_forest - scikit-learn IsolationForest with every parameter
    (including contamination) taken as-is from config.yaml, never tuned
    here; candidate thresholds are quantiles of the TRAIN anomaly-score
    distribution, also taken from config.yaml.

This module fits nothing on validation or test and chooses no threshold
itself - it only builds candidate alert vectors. Threshold *selection* is
runner.pipeline calling evaluator.select_candidate on validation-only
pooled counts, per the frozen tie-break rule.
"""
from __future__ import annotations

import numpy as np


class NonFiniteEligibleInputError(RuntimeError):
    """Raised when a window the pipeline marked eligible carries a
    non-finite feature value, train statistic, threshold or model score.
    features.py's own eligibility flag covers a window's OWN 10 s block;
    it does not guarantee every derived rolling (5m/30m) aggregate used
    here is finite. Silently treating that as 'no alert' would report a
    healthy negative for a measurement that actually failed (Astra
    freeze-review MUST-FIX 2, 2026-09-28)."""


def train_mean_std(train_frames: list, sensors: list[str]) -> dict:
    """Population mean/std (ddof=0) of {sensor}__mean_5m over TRAIN eligible
    windows, pooled across every given train FeatureFrame - the static
    baseline's own frozen fit_on rule (train eligible windows only, no
    schedule/label filtering)."""
    out = {}
    for s in sensors:
        col = f"{s}__mean_5m"
        vals = np.concatenate([fr.columns[col][fr.eligible] for fr in train_frames])
        mean, std = float(vals.mean()), float(vals.std(ddof=0))
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise NonFiniteEligibleInputError(
                f"train mean/std for {s} is non-finite (mean={mean}, std={std}): refusing to "
                "build a threshold model on a corrupted train statistic")
        out[s] = (mean, std)
    return out


def static_threshold_alert(frame, sensor_mean_std: dict, k: float, sensors: list[str]) -> np.ndarray:
    """Raw alert vector (bool, full frame length, False on ineligible
    windows) for one k: alerts where ANY sensor's mean_5m lies outside
    [train_mean - k*train_std, train_mean + k*train_std]. An ELIGIBLE
    window whose mean_5m is non-finite raises rather than silently
    scoring as no-alert/healthy."""
    n = len(frame.eligible)
    eligible = np.asarray(frame.eligible, dtype=bool)
    out_of_band = np.zeros(n, dtype=bool)
    if not np.isfinite(k):
        raise NonFiniteEligibleInputError(
            f"static-threshold k={k!r} is non-finite: refusing to compare eligible windows "
            "against an undefined band rather than silently returning no-alert")
    for s in sensors:
        mean, std = sensor_mean_std[s]
        lo, hi = mean - k * std, mean + k * std
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise NonFiniteEligibleInputError(
                f"{s} static-threshold band is non-finite (lo={lo}, hi={hi}): refusing to "
                "compare eligible windows against it - a NaN/inf bound must never silently "
                "compare False on both sides")
        col = np.asarray(frame.columns[f"{s}__mean_5m"], dtype=np.float64)
        bad = eligible & ~np.isfinite(col)
        if bad.any():
            raise NonFiniteEligibleInputError(
                f"{s}__mean_5m is non-finite on {int(bad.sum())} eligible window(s): an eligible "
                "window must never carry a non-finite feature value")
        beyond = (col < lo) | (col > hi)
        out_of_band |= beyond & eligible
    return out_of_band & eligible


def fit_isolation_forest(X_train: np.ndarray, cfg: dict):
    """Fit IsolationForest with exactly the frozen params (contamination
    fixed at the config value, not tuned)."""
    from sklearn.ensemble import IsolationForest

    params = dict(cfg["models"]["isolation_forest"]["params"])
    model = IsolationForest(**params)
    model.fit(X_train)
    return model


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """The frozen anomaly score: -score_samples (higher = more anomalous)."""
    return -model.score_samples(X)


def isolation_forest_thresholds(train_scores: np.ndarray, cfg: dict) -> dict:
    """{quantile: threshold} from TRAIN anomaly-score quantiles, per the
    frozen candidate grid. Higher threshold is stricter (fewer alerts)."""
    if not np.all(np.isfinite(train_scores)):
        raise NonFiniteEligibleInputError(
            "train anomaly scores contain non-finite values: refusing to derive threshold "
            "candidates from them")
    spec = cfg["models"]["isolation_forest"]["threshold_candidates"]
    return {q: float(np.quantile(train_scores, q, method=spec["quantile_method"]))
            for q in spec["quantiles"]}


def isolation_forest_alert(frame, fitted_scaler, model, threshold: float) -> np.ndarray:
    """Raw alert vector for one threshold: score the frame's eligible rows
    through the train-fitted scaler, alert where the anomaly score exceeds
    the threshold. False everywhere ineligible (no score exists there). An
    ELIGIBLE window whose anomaly score comes out non-finite raises rather
    than silently scoring as no-alert/healthy."""
    from features import transform

    if not np.isfinite(threshold):
        raise NonFiniteEligibleInputError(
            f"isolation-forest threshold={threshold!r} is non-finite: refusing to compare "
            "eligible scores against an undefined threshold rather than silently returning "
            "no-alert")
    n = len(frame.eligible)
    eligible = np.asarray(frame.eligible, dtype=bool)
    scores = np.full(n, np.nan)
    if eligible.any():
        scores[eligible] = anomaly_scores(model, transform(fitted_scaler, frame))
    bad = eligible & ~np.isfinite(scores)
    if bad.any():
        raise NonFiniteEligibleInputError(
            f"isolation-forest anomaly score is non-finite on {int(bad.sum())} eligible "
            "window(s): an eligible window must never carry a non-finite score")
    beyond = scores > threshold
    return beyond & eligible

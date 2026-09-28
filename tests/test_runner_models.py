"""runner.models: hand-built fixtures only (no generator, no real seeds)."""
import copy

import numpy as np
import pytest

from runner import models as model_lib


def _frame(vals, eligible):
    import features
    n = len(vals)
    ws = np.arange(0, n * 10, 10, dtype=np.int64)
    cols = {"vibration_rms__mean_5m": np.asarray(vals, dtype=np.float64)}
    elig = np.asarray(eligible, dtype=bool)
    dq = np.where(elig, "", "excluded").astype(object)
    return features.FeatureFrame(ws, cols, dq, elig, {})


def test_static_threshold_alert_flags_only_out_of_band_eligible_windows():
    vals = [10.0, 10.0, 100.0, 10.0, -100.0]
    eligible = [True, True, True, False, True]
    frame = _frame(vals, eligible)
    mean_std = {"vibration_rms": (10.0, 1.0)}
    alert = model_lib.static_threshold_alert(frame, mean_std, k=2.0, sensors=["vibration_rms"])
    assert alert.dtype == np.bool_
    assert list(alert) == [False, False, True, False, True]


def test_static_threshold_alert_never_alerts_on_ineligible_nan():
    vals = [np.nan, 100.0]
    eligible = [False, False]
    frame = _frame(vals, eligible)
    mean_std = {"vibration_rms": (10.0, 1.0)}
    alert = model_lib.static_threshold_alert(frame, mean_std, k=2.0, sensors=["vibration_rms"])
    assert list(alert) == [False, False]


# --- Astra freeze-review MUST-FIX 2 (2026-09-28): an eligible window with a
# non-finite feature value/score must raise, not silently become a "no
# alert" healthy negative. ---

def test_static_threshold_alert_raises_on_eligible_nan():
    vals = [np.nan, 10.0]
    eligible = [True, True]
    frame = _frame(vals, eligible)
    mean_std = {"vibration_rms": (10.0, 1.0)}
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.static_threshold_alert(frame, mean_std, k=2.0, sensors=["vibration_rms"])


def test_static_threshold_alert_raises_on_eligible_inf():
    vals = [np.inf, 10.0]
    eligible = [True, True]
    frame = _frame(vals, eligible)
    mean_std = {"vibration_rms": (10.0, 1.0)}
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.static_threshold_alert(frame, mean_std, k=2.0, sensors=["vibration_rms"])


# --- Astra freeze-review r2 group 2 (2026-09-28): a NaN/inf threshold
# itself - not just a non-finite input/score - must raise, never silently
# compare False on both sides and return no-alert on an eligible, finite
# input. ---

def test_static_threshold_alert_raises_on_non_finite_k_with_finite_eligible_input():
    vals = [10.0, 100.0]
    eligible = [True, True]
    frame = _frame(vals, eligible)
    mean_std = {"vibration_rms": (10.0, 1.0)}
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.static_threshold_alert(frame, mean_std, k=float("nan"), sensors=["vibration_rms"])


def test_isolation_forest_alert_raises_on_non_finite_threshold_with_finite_eligible_score(
        cfg, monkeypatch):
    tiny = _tiny_if_cfg(cfg)
    rng = np.random.default_rng(7)
    import features

    cols = list(features.feature_columns(cfg))
    n_fit = 32
    X_train = rng.standard_normal((n_fit, len(cols)))
    train_frame = features.FeatureFrame(
        np.arange(n_fit, dtype=np.int64) * 10,
        {c: X_train[:, i] for i, c in enumerate(cols)},
        np.full(n_fit, "", dtype=object), np.ones(n_fit, dtype=bool), {"split": "train"})
    fs = features.fit_scaler([train_frame], tiny)
    X = np.column_stack([X_train[:, cols.index(c)] for c in fs.columns])
    model = model_lib.fit_isolation_forest(X, tiny)

    n_test = 3
    test_vals = rng.standard_normal((n_test, len(cols)))
    eligible = np.array([True, True, True])
    test_frame = features.FeatureFrame(
        np.arange(n_test, dtype=np.int64) * 10,
        {c: test_vals[:, i] for i, c in enumerate(cols)},
        np.full(n_test, "", dtype=object), eligible, {"split": "test"})

    monkeypatch.setattr(model_lib, "anomaly_scores", lambda model, X: np.array([1.0, 2.0, 3.0]))
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.isolation_forest_alert(test_frame, fs, model, threshold=float("nan"))


def test_train_mean_std_raises_on_non_finite_statistic():
    f1 = _frame([np.inf, np.inf], [True, True])
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.train_mean_std([f1], ["vibration_rms"])


def test_isolation_forest_thresholds_raises_on_non_finite_train_scores(cfg):
    tiny = _tiny_if_cfg(cfg)
    scores = np.array([1.0, np.nan, 2.0])
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.isolation_forest_thresholds(scores, tiny)


def test_train_mean_std_pools_across_frames_and_respects_eligibility():
    f1 = _frame([1.0, 2.0, 100.0], [True, True, False])   # 100.0 excluded (ineligible)
    f2 = _frame([3.0, 4.0], [True, True])
    out = model_lib.train_mean_std([f1, f2], ["vibration_rms"])
    pooled = np.array([1.0, 2.0, 3.0, 4.0])
    mean, std = out["vibration_rms"]
    assert mean == pytest.approx(pooled.mean())
    assert std == pytest.approx(pooled.std(ddof=0))


def _tiny_if_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["models"]["isolation_forest"]["params"] = {
        "n_estimators": 10, "max_samples": 8, "max_features": 1.0,
        "bootstrap": False, "contamination": 0.1, "random_state": 7,
    }
    c["models"]["isolation_forest"]["threshold_candidates"] = {
        "source": "train anomaly-score quantiles", "quantiles": [0.5, 0.9],
        "quantile_method": "linear", "plus_no_alert": True,
        "strictness": "higher threshold is stricter",
    }
    return c


def test_fit_isolation_forest_uses_config_params_verbatim(cfg):
    tiny = _tiny_if_cfg(cfg)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 3))
    model = model_lib.fit_isolation_forest(X, tiny)
    params = model.get_params()
    assert params["contamination"] == 0.1
    assert params["n_estimators"] == 10
    assert params["max_samples"] == 8
    assert params["bootstrap"] is False
    assert params["random_state"] == 7


def test_anomaly_scores_are_deterministic_given_the_same_params(cfg):
    tiny = _tiny_if_cfg(cfg)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 3))
    scores_a = model_lib.anomaly_scores(model_lib.fit_isolation_forest(X, tiny), X)
    scores_b = model_lib.anomaly_scores(model_lib.fit_isolation_forest(X, tiny), X)
    np.testing.assert_array_equal(scores_a, scores_b)


def test_isolation_forest_thresholds_are_a_monotone_quantile_grid(cfg):
    tiny = _tiny_if_cfg(cfg)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 3))
    scores = model_lib.anomaly_scores(model_lib.fit_isolation_forest(X, tiny), X)
    thr = model_lib.isolation_forest_thresholds(scores, tiny)
    assert set(thr) == {0.5, 0.9}
    assert thr[0.9] >= thr[0.5]


def test_isolation_forest_alert_never_alerts_on_ineligible_windows(cfg):
    import features

    tiny = _tiny_if_cfg(cfg)
    rng = np.random.default_rng(2)
    cols = list(features.feature_columns(cfg))
    n_fit = 64
    X_train = rng.standard_normal((n_fit, len(cols)))
    train_frame = features.FeatureFrame(
        np.arange(n_fit, dtype=np.int64) * 10,
        {c: X_train[:, i] for i, c in enumerate(cols)},
        np.full(n_fit, "", dtype=object), np.ones(n_fit, dtype=bool), {"split": "train"})
    fs = features.fit_scaler([train_frame], tiny)
    X = np.column_stack([X_train[:, cols.index(c)] for c in fs.columns])
    model = model_lib.fit_isolation_forest(X, tiny)
    scores = model_lib.anomaly_scores(model, X)
    thr = float(np.quantile(scores, 0.5))

    n_test = 5
    test_vals = rng.standard_normal((n_test, len(cols)))
    eligible = np.array([True, True, False, True, False])
    test_frame = features.FeatureFrame(
        np.arange(n_test, dtype=np.int64) * 10,
        {c: test_vals[:, i] for i, c in enumerate(cols)},
        np.where(eligible, "", "excluded").astype(object), eligible, {"split": "test"})
    alert = model_lib.isolation_forest_alert(test_frame, fs, model, thr)
    assert alert.dtype == np.bool_
    assert not alert[~eligible].any()


def test_isolation_forest_alert_raises_on_non_finite_score(cfg, monkeypatch):
    import features

    tiny = _tiny_if_cfg(cfg)
    rng = np.random.default_rng(6)
    cols = list(features.feature_columns(cfg))
    n_fit = 32
    X_train = rng.standard_normal((n_fit, len(cols)))
    train_frame = features.FeatureFrame(
        np.arange(n_fit, dtype=np.int64) * 10,
        {c: X_train[:, i] for i, c in enumerate(cols)},
        np.full(n_fit, "", dtype=object), np.ones(n_fit, dtype=bool), {"split": "train"})
    fs = features.fit_scaler([train_frame], tiny)
    X = np.column_stack([X_train[:, cols.index(c)] for c in fs.columns])
    model = model_lib.fit_isolation_forest(X, tiny)

    n_test = 3
    test_vals = rng.standard_normal((n_test, len(cols)))
    eligible = np.array([True, True, True])
    test_frame = features.FeatureFrame(
        np.arange(n_test, dtype=np.int64) * 10,
        {c: test_vals[:, i] for i, c in enumerate(cols)},
        np.full(n_test, "", dtype=object), eligible, {"split": "test"})

    monkeypatch.setattr(model_lib, "anomaly_scores",
                        lambda model, X: np.array([1.0, np.nan, 2.0]))
    with pytest.raises(model_lib.NonFiniteEligibleInputError):
        model_lib.isolation_forest_alert(test_frame, fs, model, threshold=0.5)

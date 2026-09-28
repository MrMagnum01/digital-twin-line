"""monitoring.model_monitor: hand-built fixtures only."""
from unittest.mock import patch

import numpy as np
import pytest

from monitoring.model_monitor import (InsufficientBaselineError, MissingFitBoundaryError,
                                     NonFiniteBaselineError, fit_baseline, score_frame)


def _frame(cfg, X, eligible, split="dev"):
    import features
    cols = features.feature_columns(cfg)
    n = X.shape[0]
    columns = {c: X[:, i] for i, c in enumerate(cols)}
    elig = np.asarray(eligible, dtype=bool)
    dq = np.where(elig, "", "excluded").astype(object)
    return features.FeatureFrame(np.arange(n, dtype=np.int64) * 10, columns, dq, elig,
                                 {"split": split})


def test_fit_baseline_refuses_too_few_eligible_windows(cfg):
    rng = np.random.default_rng(0)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((10, n_cols))
    frame = _frame(cfg, X, [True] * 10)
    with pytest.raises(InsufficientBaselineError):
        fit_baseline(frame, cfg, warmup_windows=180)


def test_fit_baseline_and_score_are_deterministic(cfg):
    rng = np.random.default_rng(1)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((60, n_cols))
    frame = _frame(cfg, X, [True] * 60)
    b1 = fit_baseline(frame, cfg, warmup_windows=50)
    b2 = fit_baseline(frame, cfg, warmup_windows=50)
    alert1, scores1, causal1 = score_frame(frame, b1, cfg)
    alert2, scores2, causal2 = score_frame(frame, b2, cfg)
    np.testing.assert_array_equal(alert1, alert2)
    np.testing.assert_allclose(scores1, scores2)
    np.testing.assert_array_equal(causal1, causal2)


def test_score_frame_never_alerts_on_ineligible_windows(cfg):
    rng = np.random.default_rng(2)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    eligible = [True] * 80
    eligible[70] = False
    frame = _frame(cfg, X, eligible)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)
    alert, scores, causal = score_frame(frame, baseline, cfg)
    assert not alert[70]
    assert np.isnan(scores[70])
    assert not causal[70]


def test_a_clear_outlier_scores_higher_than_baseline_noise(cfg):
    rng = np.random.default_rng(3)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    X[75] += 50.0     # a gross outlier well outside the baseline's own range
    frame = _frame(cfg, X, [True] * 80)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)
    _, scores, causal = score_frame(frame, baseline, cfg)
    assert scores[75] > np.nanmedian(scores[:50])
    assert causal[75]


# --- Astra freeze-review MUST-FIX 5 (2026-09-28): the baseline's own
# fitting interval must not be scored/alerted as if it were a causal,
# out-of-sample judgement. ---

def test_fit_baseline_records_the_exact_fitting_windows(cfg):
    rng = np.random.default_rng(5)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((60, n_cols))
    eligible = [True] * 60
    eligible[5] = False
    frame = _frame(cfg, X, eligible)
    baseline = fit_baseline(frame, cfg, warmup_windows=10)
    elig = np.asarray(eligible, dtype=bool)
    expected = frame.window_start[elig][:10]
    assert baseline["fit_window_starts"] == frozenset(int(x) for x in expected)


def test_score_frame_never_alerts_within_its_own_fitting_interval(cfg):
    rng = np.random.default_rng(4)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    X[10] += 50.0     # a gross outlier INSIDE the warm-up/fitting slice
    frame = _frame(cfg, X, [True] * 80)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)
    alert, scores, causal = score_frame(frame, baseline, cfg)
    assert not causal[10]
    assert not alert[10]                          # never alerts on its own fitting interval
    assert scores[10] > np.nanmedian(scores[:50])  # score is still visible, just not causal
    assert causal[60]                              # windows after the fitting interval are causal


# --- Astra freeze-review r2 group 2 (2026-09-28): a non-finite eligible
# model score must raise, never silently become alert=False. ---

def test_score_frame_raises_on_non_finite_eligible_score(cfg, monkeypatch):
    rng = np.random.default_rng(8)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    frame = _frame(cfg, X, [True] * 80)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)

    class _BoomModel:
        def score_samples(self, Xs):
            out = np.ones(Xs.shape[0])
            out[5] = np.nan
            return out

    monkeypatch.setitem(baseline, "model", _BoomModel())
    with pytest.raises(NonFiniteBaselineError):
        score_frame(frame, baseline, cfg)


def test_score_frame_raises_on_non_finite_baseline_threshold(cfg):
    rng = np.random.default_rng(9)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    frame = _frame(cfg, X, [True] * 80)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)
    baseline["threshold"] = float("nan")
    with pytest.raises(NonFiniteBaselineError):
        score_frame(frame, baseline, cfg)


# --- Astra freeze-review r2 group 5 (2026-09-28): excluding only the exact
# fit_window_starts is insufficient - a timestamp before the fitting
# period (earlier history, scored against a baseline fit later) but not
# itself one of the fit windows must still be non-causal. ---

def test_score_frame_marks_earlier_history_non_causal_using_the_fit_boundary(cfg):
    rng = np.random.default_rng(10)
    n_cols = len(__import__("features").feature_columns(cfg))
    X_fit = rng.standard_normal((80, n_cols))
    fit_frame = _frame(cfg, X_fit, [True] * 80)
    baseline = fit_baseline(fit_frame, cfg, warmup_windows=50)
    assert baseline["fit_boundary"] == int(fit_frame.window_start[49])

    # An entirely separate, EARLIER frame: none of its window_start values
    # are members of fit_window_starts (they are disjoint, negative
    # timestamps), so the old set-membership check would have marked every
    # one of these windows causal=True even though the baseline did not
    # exist yet at that time.
    import features

    n_hist = 20
    X_hist = rng.standard_normal((n_hist, n_cols))
    cols = features.feature_columns(cfg)
    columns = {c: X_hist[:, i] for i, c in enumerate(cols)}
    window_start = (np.arange(n_hist, dtype=np.int64) - n_hist) * 10  # strictly negative
    assert not (set(int(w) for w in window_start) & baseline["fit_window_starts"])
    history_frame = features.FeatureFrame(
        window_start, columns, np.full(n_hist, "", dtype=object),
        np.ones(n_hist, dtype=bool), {"split": "dev"})

    _, _, causal = score_frame(history_frame, baseline, cfg)
    assert not causal.any()


# --- Astra freeze-review r3 group 5 (2026-09-28): a missing fit_boundary
# must RAISE, with no fallback to the fit_window_starts membership test -
# Astra's exact probe fixture (fit_window_starts={10, 20}, an earlier
# timestamp of 1) previously came back causal=True through that fallback. ---

def test_score_frame_raises_when_fit_boundary_is_missing():
    from types import SimpleNamespace

    frame = SimpleNamespace(eligible=np.array([True]), window_start=np.array([1]))
    baseline = dict(
        keep_columns=[], scaler=SimpleNamespace(transform=lambda x: x),
        model=SimpleNamespace(score_samples=lambda x: np.array([-1.0])), threshold=0.5,
        fit_window_starts={10, 20})
    with patch("features.to_matrix", return_value=np.array([[0.0]])):
        with pytest.raises(MissingFitBoundaryError):
            score_frame(frame, baseline, {})

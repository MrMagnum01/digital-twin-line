"""monitoring.model_monitor: hand-built fixtures only."""
import numpy as np
import pytest

from monitoring.model_monitor import InsufficientBaselineError, fit_baseline, score_frame


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

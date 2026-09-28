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
    alert1, scores1 = score_frame(frame, b1, cfg)
    alert2, scores2 = score_frame(frame, b2, cfg)
    np.testing.assert_array_equal(alert1, alert2)
    np.testing.assert_allclose(scores1, scores2)


def test_score_frame_never_alerts_on_ineligible_windows(cfg):
    rng = np.random.default_rng(2)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    eligible = [True] * 80
    eligible[70] = False
    frame = _frame(cfg, X, eligible)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)
    alert, scores = score_frame(frame, baseline, cfg)
    assert not alert[70]
    assert np.isnan(scores[70])


def test_a_clear_outlier_scores_higher_than_baseline_noise(cfg):
    rng = np.random.default_rng(3)
    n_cols = len(__import__("features").feature_columns(cfg))
    X = rng.standard_normal((80, n_cols))
    X[75] += 50.0     # a gross outlier well outside the baseline's own range
    frame = _frame(cfg, X, [True] * 80)
    baseline = fit_baseline(frame, cfg, warmup_windows=50)
    _, scores = score_frame(frame, baseline, cfg)
    assert scores[75] > np.nanmedian(scores[:50])

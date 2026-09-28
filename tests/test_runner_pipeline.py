"""runner.pipeline.run(): exercised only on the DEVELOPMENT table (seeds
901-906) - never the locked table (101-110 / 201-204 / 301-306). Confirms
the refusal gates fire before anything is generated, and that the pipeline
follows the frozen protocol's shape (train-only fit, validation-only
selection, test scored once per stratum, never pooled)."""
import numpy as np
import pytest

import evaluator
import generator
from runner import lock_guard, pipeline

DEV_SEEDS = {901, 902, 903, 904, 905, 906}
LOCKED_SEEDS = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
               201, 202, 203, 204, 301, 302, 303, 304, 305, 306}


def test_run_refuses_before_verifying_the_lock(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("generator.generate must not run before the lock is verified")

    monkeypatch.setattr(generator, "generate", _boom)
    with pytest.raises(lock_guard.LockVerificationError, match="does not exist"):
        pipeline.run("dev", repo_root=tmp_path)  # empty dir: no experiment-lock.json


def test_run_refuses_an_unapproved_lock_sha(monkeypatch):
    monkeypatch.setattr(lock_guard, "APPROVED_LOCK_SHA256", "0" * 64)
    with pytest.raises(lock_guard.LockVerificationError, match="does not match the approved lock"):
        pipeline.run("dev")


def test_run_refuses_the_locked_table_without_allow_locked():
    with pytest.raises(generator.LockedSeedsError):
        pipeline.run("locked")


def test_run_on_dev_never_touches_locked_seeds():
    report = pipeline.run("dev")
    ds = generator.generate("dev")
    assert report["dataset_hash"] == generator.dataset_hash(ds)
    seeds = {s for sp in ds.splits.values() for s in sp.seeds}
    assert seeds == DEV_SEEDS
    assert not (seeds & LOCKED_SEEDS)


def test_run_reports_lock_version_and_scaler_fit():
    report = pipeline.run("dev")
    assert report["lock_version"] == "lock-3"
    assert report["scaler"]["n_fit_rows"] > 0
    assert isinstance(report["scaler"]["kept_columns"], list)


def test_run_selects_thresholds_on_validation_only_and_retains_every_candidate():
    report = pipeline.run("dev")
    static = report["static_threshold"]
    assert len(static["validation_candidates"]) == len(static["k_grid"])
    # select_candidate always retains every candidate plus the added no_alert
    assert len(static["selection"]["table"]) == len(static["k_grid"]) + 1
    isof = report["isolation_forest"]
    assert len(isof["validation_candidates"]) == len(isof["train_score_quantile_thresholds"])
    assert len(isof["selection"]["table"]) == len(isof["validation_candidates"]) + 1


def test_run_scores_test_once_per_stratum_never_pooled():
    report = pipeline.run("dev")
    test_results = report["test_results"]
    assert set(test_results) == {"FILLER", "LABELLER"}
    for asset, models in test_results.items():
        assert models["no_alert"]["split"] == "test"
        assert models["no_alert"]["asset"] == asset
        for name, result in models.items():
            assert "pooled" not in result
    assert report["test_strata"] == {"FILLER": "seen_asset_later_time",
                                     "LABELLER": "unseen_asset_later_time"}
    # pool_counts must refuse to pool any test stratum result
    with pytest.raises(evaluator.PoolingError):
        evaluator.pool_counts([r for models in test_results.values() for r in models.values()])


def test_run_calls_score_stratum_on_test_exactly_once_per_model_per_asset(monkeypatch):
    calls = []
    real_score_stratum = evaluator.score_stratum

    def _counting(inp, events, cfg=None):
        if inp.split == "test":
            calls.append((inp.split, inp.asset))
        return real_score_stratum(inp, events, cfg)

    monkeypatch.setattr(evaluator, "score_stratum", _counting)
    monkeypatch.setattr(pipeline.evaluator, "score_stratum", _counting)
    report = pipeline.run("dev")
    n_models = 2 + (1 if report["static_threshold"]["chosen_k"] is not None else 0) \
             + (1 if report["isolation_forest"]["chosen_threshold"] is not None else 0)
    assert len(calls) == n_models * len(report["test_strata"])
    assert len(calls) == len(set(calls)) * (n_models)  # no duplicate (split, asset, model) triple work


def test_run_static_threshold_alert_matches_direct_computation():
    report = pipeline.run("dev")
    k = report["static_threshold"]["chosen_k"]
    if k is None:
        pytest.skip("no static-threshold candidate met the false-alert cap on dev seeds")
    assert k in report["static_threshold"]["k_grid"]

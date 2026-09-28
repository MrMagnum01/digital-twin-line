"""runner.pipeline.run(): exercised only on the DEVELOPMENT table (seeds
901-906) - never the locked table (101-110 / 201-204 / 301-306). Confirms
the refusal gates fire before anything is generated, and that the pipeline
follows the frozen protocol's shape (train-only fit, validation-only
selection, test scored once per stratum, never pooled)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import evaluator
import generator
from runner import lock_guard, pipeline

DEV_SEEDS = {901, 902, 903, 904, 905, 906}
LOCKED_SEEDS = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
               201, 202, 203, 204, 301, 302, 303, 304, 305, 306}

REPO = Path(__file__).resolve().parent.parent


def _copy_repo_subset(tmp_path: Path) -> Path:
    """A byte-for-byte valid copy of exactly the files experiment-lock.json
    hashes, unmodified, plus the lock itself: hash verification passes
    against it, but it is NOT where this process's already-imported
    generator/features/evaluator/twin_config modules physically live."""
    lock = json.loads((REPO / "experiment-lock.json").read_bytes())
    (tmp_path / "experiment-lock.json").write_bytes((REPO / "experiment-lock.json").read_bytes())
    for section in ("sha256", "supporting_sha256"):
        for rel in lock[section]:
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, dst)
    return tmp_path


def test_run_refuses_before_verifying_the_lock(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("generator.generate must not run before the lock is verified")

    monkeypatch.setattr(generator, "generate", _boom)
    with pytest.raises(lock_guard.LockVerificationError, match="does not exist"):
        pipeline.run("dev", repo_root=tmp_path)  # empty dir: no experiment-lock.json


# --- Astra freeze-review MUST-FIX 1 (2026-09-28): repo_root must not be
# able to verify one directory's bytes while the process actually executes
# same-named modules imported from elsewhere. ---

def test_run_refuses_a_repo_root_whose_hashes_pass_but_whose_imports_are_elsewhere(
        tmp_path, monkeypatch):
    monkeypatch.setattr(lock_guard, "_submodule_head",
                        lambda root, path: lock_guard.PINNED_SUBMODULE_COMMIT)
    monkeypatch.setattr(lock_guard, "_submodule_dirty", lambda root, path: False)
    _copy_repo_subset(tmp_path)
    # tmp_path's copies hash-match the lock exactly; the real generator/
    # features/evaluator/twin_config modules this process already imported
    # still live under the real repo, not tmp_path.
    with pytest.raises(lock_guard.LockVerificationError, match="imported from"):
        pipeline.run("dev", repo_root=tmp_path)


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
    # A model whose selection is "no_alert" gets its test row for free (it
    # reuses the already-scored no_alert baseline result, not a fresh
    # score_stratum call - see test_run_selecting_no_alert_keeps_the_model_
    # test_row below), so only a numeric selection costs its own call.
    static_selected = report["static_threshold"]["selection"]["selected"]
    if_selected = report["isolation_forest"]["selection"]["selected"]
    n_models = 2
    if static_selected is not None and static_selected != "no_alert":
        n_models += 1
    if if_selected is not None and if_selected != "no_alert":
        n_models += 1
    assert len(calls) == n_models * len(report["test_strata"])
    assert len(calls) == len(set(calls)) * (n_models)  # no duplicate (split, asset, model) triple work


# --- Astra freeze-review MUST-FIX 3 (2026-09-28): a selected no_alert
# candidate is a valid model policy, never a dropped test row. ---

def test_run_selecting_no_alert_keeps_the_model_test_row(monkeypatch):
    def _force_no_alert(candidates, cfg=None):
        return {"selected": "no_alert", "table": candidates}

    monkeypatch.setattr(pipeline.evaluator, "select_candidate", _force_no_alert)
    report = pipeline.run("dev")
    assert report["static_threshold"]["selection"]["selected"] == "no_alert"
    assert report["static_threshold"]["chosen_k"] is None
    assert report["isolation_forest"]["selection"]["selected"] == "no_alert"
    assert report["isolation_forest"]["chosen_threshold"] is None
    assert report["test_results"]
    for asset, models in report["test_results"].items():
        assert "static_threshold" in models, f"{asset}: static_threshold row dropped"
        assert "isolation_forest" in models, f"{asset}: isolation_forest row dropped"
        assert models["static_threshold"] == models["no_alert"]
        assert models["isolation_forest"] == models["no_alert"]


def test_run_a_genuine_no_candidate_within_cap_still_omits_the_row(monkeypatch):
    """Contrast case: selected=None (no candidate met the false-alert cap)
    is a real absence, distinct from a selected "no_alert" policy - that
    row is correctly left out, and the report says why."""
    def _force_none(candidates, cfg=None):
        return {"selected": None, "table": candidates, "note": "no candidate meets the cap"}

    monkeypatch.setattr(pipeline.evaluator, "select_candidate", _force_none)
    report = pipeline.run("dev")
    assert report["static_threshold"]["selection"]["selected"] is None
    assert report["static_threshold"]["chosen_k"] is None
    for models in report["test_results"].values():
        assert "static_threshold" not in models
        assert "isolation_forest" not in models


# --- Astra freeze-review r2 group 1 (2026-09-28): the literal probe -
# mutate twin_config.load_config()'s cached dict's contamination to 0.4,
# then call pipeline.run(); the value actually used for generation/fitting
# must still be the locked 0.01, never the mutated 0.4. ---

def test_run_is_unaffected_by_a_mutated_twin_config_cache():
    import twin_config

    cached = twin_config.load_config()
    original = cached["models"]["isolation_forest"]["params"]["contamination"]
    cached["models"]["isolation_forest"]["params"]["contamination"] = 0.4
    try:
        report = pipeline.run("dev")
        assert report["isolation_forest"]["params"]["contamination"] == original == 0.01
    finally:
        cached["models"]["isolation_forest"]["params"]["contamination"] = original


# --- Astra freeze-review r2 group 4 (2026-09-28): select() must expose a
# persistence boundary BEFORE test scoring - a SelectionResult with no test
# data in it at all, distinct from run()'s combined convenience wrapper. ---

def test_select_returns_before_any_test_scoring_score_test_completes_it():
    sel = pipeline.select("dev")
    assert not hasattr(sel, "test_results")
    report = pipeline.score_test(sel)
    assert report["test_results"]
    assert report == pipeline.run("dev")


def test_run_static_threshold_alert_matches_direct_computation():
    report = pipeline.run("dev")
    k = report["static_threshold"]["chosen_k"]
    if k is None:
        pytest.skip("no static-threshold candidate met the false-alert cap on dev seeds")
    assert k in report["static_threshold"]["k_grid"]


# --- Astra freeze-review r3 group 4 (2026-09-28): pipeline.run() must
# never call select() then score_test() for a locked evaluation without
# giving the persisted selection a chance to reach durable storage first -
# it must refuse outright rather than repeat the old dev-convenience
# shortcut for allow_locked=True. Astra's probe patches select()/score_test
# /cli._write_exclusive and calls pipeline.run('dev'), observing that the
# dev convenience path never touches cli's exclusive-write machinery; the
# same shortcut applied to a locked call is the actual bug. ---

def test_run_refuses_allow_locked_never_calling_select_or_score_test(monkeypatch):
    from unittest.mock import patch

    from runner import cli

    sentinel = object()
    with patch.object(pipeline, "select", return_value=sentinel) as select_mock, \
         patch.object(pipeline, "score_test", return_value={}) as score_mock, \
         patch.object(cli, "_write_exclusive") as write_mock:
        with pytest.raises(RuntimeError, match="refuses allow_locked=True"):
            pipeline.run("locked", allow_locked=True)
        assert select_mock.call_count == 0
        assert score_mock.call_count == 0
        assert write_mock.call_count == 0
    # dev remains the unrestricted convenience path.
    report = pipeline.run("dev")
    assert report["table"] == "dev"


# --- Astra freeze-review r3 group 4 (2026-09-28): score_test() must accept
# only the one persisted, immutable SelectionResult select() itself
# produced - never a hand-built stand-in merely carrying table='locked'.
# Astra's exact probe fixture and assertion (inverted: the old bug let this
# score twice with equal results; the fix refuses it outright, on the very
# first call, because it is not a real SelectionResult at all). ---

def test_score_test_refuses_a_fixture_that_only_carries_locked_table_identity():
    from types import SimpleNamespace

    sel = SimpleNamespace(
        test_assets=[], table="locked", lock={}, dataset_hash="fixture", scaler_report={},
        static_threshold_report={}, isolation_forest_report={}, validation_baselines={},
        test_strata={}, honesty_note="fixture")
    with pytest.raises(TypeError, match="SelectionResult"):
        pipeline.score_test(sel)


def test_score_test_refuses_a_mutated_selection_result():
    sel = pipeline.select("dev")
    sel.chosen_k = (sel.chosen_k or 0) + 1.0  # mutate a field score_test() actually uses
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


# --- Astra freeze-review r4 group 4 (2026-09-28): "the fitted model and
# scoring state are outside the selection hash ... Replacing sel.if_model
# leaves selection_id unchanged; the replacement model reaches
# isolation_forest_alert under the original identity. sensor_mean_std,
# scaler and frames likewise are outside that identity." Adapted from
# Astra's probe (hand-built fixture / mocked scoring, no locked generation
# or subprocess): here exercised against a real select() selection so the
# fitted model/scaler genuinely differ, not two identically-empty stand-ins. ---

def test_selection_identity_changes_when_the_fitted_model_is_replaced():
    from sklearn.ensemble import IsolationForest

    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.if_model = IsolationForest(n_estimators=3, random_state=0).fit(np.zeros((5, 1)))
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


def test_selection_identity_changes_when_the_scaler_is_replaced():
    import features as feat

    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.fs = feat.FittedScaler(scaler=object(), columns=sel.fs.columns,
                               dropped=sel.fs.dropped, n_fit_rows=sel.fs.n_fit_rows)
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


def test_selection_identity_changes_when_sensor_mean_std_is_replaced():
    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.sensor_mean_std = {"replaced": (0.0, 1.0)}
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


def test_selection_identity_changes_when_a_frame_is_replaced():
    from types import SimpleNamespace

    sel = pipeline.select("dev")
    original_id = sel.selection_id
    key = next(iter(sel.frames))
    sel.frames[key] = SimpleNamespace(eligible=np.array([True]))
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


# --- Astra freeze-review r5 group 4 (2026-09-28): "the selection identity
# must bind EVERY value consumed at scoring, namely labels, events, the
# effective cfg and the sensor set, in addition to the model, scaler,
# sensor_mean_std and frames ... Astra's probe changes each of those four
# without changing selection_id, and replacement truth or events reach
# scoring." One regression test per changed input, using the exact
# replacement values from Astra's adjacent probe
# (40-sessions/2026-09-28-astra-twin-runner-r5-probes.py) - each must now be
# refused, both at the identity-hash level and at score_test(). ---

def test_selection_identity_changes_when_labels_are_replaced():
    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.labels = ["replacement truth"]
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


def test_selection_identity_changes_when_events_are_replaced():
    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.events = ["replacement fault episodes"]
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


def test_selection_identity_changes_when_cfg_is_replaced():
    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.cfg = {"evaluation": "changed"}
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)


def test_selection_identity_changes_when_sensors_are_replaced():
    sel = pipeline.select("dev")
    original_id = sel.selection_id
    sel.sensors = ["other sensor"]
    assert pipeline._selection_identity(sel) != original_id
    with pytest.raises(pipeline.SelectionIntegrityError):
        pipeline.score_test(sel)

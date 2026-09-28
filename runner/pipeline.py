"""Orchestrates the frozen protocol's model-fit + threshold-selection run
(NOT a locked file; imports and calls the locked ones without editing them).

run() is the single entry point:
  1. lock_guard.require_lock() - refuses to do anything unless the approved
     experiment-lock.json still matches every file it hashes.
  2. generator.generate(table_name, allow_locked=allow_locked, ...) - refuses
     the locked seed table unless the caller explicitly passes
     allow_locked=True. Nothing in this milestone does; tests here use only
     the development table (dev seeds 901-906) or hand-built fixtures.
  3. features.fit_scaler() on TRAIN frames only (train-only, per protocol).
  4. Static-threshold and IsolationForest candidates are built and scored on
     VALIDATION only, pooled across the validation assets, and a threshold
     is chosen per model by evaluator.select_candidate() (max event F1
     within the false-alert cap; ties -> fewer false alerts -> fewer total
     alerts -> stricter). IsolationForest's contamination is taken as-is
     from config.yaml and never tuned; only its decision-function threshold
     is chosen here.
  5. TEST is scored exactly once, per stratum (never pooled - test strata
     are always reported separately, matching evaluator.pool_counts'
     refusal to pool any test stratum), using the model+threshold chosen in
     step 4 - no threshold search on test.
"""
from __future__ import annotations

import numpy as np

from runner import lock_guard, models as model_lib

import evaluator
import features as feat
import generator
from twin_config import load_config


def _cand_summary(pooled: dict) -> dict:
    return {
        "tp": pooled["tp"], "fp": pooled["fp"], "fn": pooled["fn"],
        "false_alerts_per_operating_hour": pooled["false_alerts_per_operating_hour"],
        "total_alerts": pooled["total_alerts"],
    }


def _score_asset(frame, alert, labels, split, asset, events, cfg) -> dict:
    inp = evaluator.stratum_input(frame, alert, labels, split, asset)
    return evaluator.score_stratum(inp, events, cfg)


def run(table_name: str = "dev", *, allow_locked: bool = False,
        cfg: dict | None = None, repo_root=None) -> dict:
    lock = lock_guard.require_lock(repo_root)
    cfg = cfg or load_config()
    sensors = cfg["sensors"]["order"]

    ds = generator.generate(table_name, allow_locked=allow_locked, cfg=cfg)
    frames = feat.features_for_dataset(ds, cfg)
    events = evaluator.events_from_labels(ds.labels)

    train_assets = ds.splits["train"].assets
    val_assets = ds.splits["validation"].assets
    test_assets = ds.splits["test"].assets
    train_frames = [frames[("train", a)] for a in train_assets]

    fs = feat.fit_scaler(train_frames, cfg)
    sensor_mean_std = model_lib.train_mean_std(train_frames, sensors)

    X_train = np.vstack([feat.to_matrix(fr, fs.columns) for fr in train_frames])
    if_model = model_lib.fit_isolation_forest(X_train, cfg)
    train_scores = model_lib.anomaly_scores(if_model, X_train)
    if_thresholds = model_lib.isolation_forest_thresholds(train_scores, cfg)

    def _validation_pooled(alert_fn) -> dict:
        results = [_score_asset(frames[("validation", a)], alert_fn(frames[("validation", a)]),
                                ds.labels, "validation", a, events, cfg)
                   for a in val_assets]
        return evaluator.pool_counts(results)

    k_grid = cfg["models"]["static_threshold"]["k_grid"]
    static_candidates = []
    for k in k_grid:
        pooled = _validation_pooled(
            lambda fr, k=k: model_lib.static_threshold_alert(fr, sensor_mean_std, k, sensors))
        static_candidates.append({"candidate_id": f"static_k{k}", "strictness": float(k),
                                  **_cand_summary(pooled)})
    static_selection = evaluator.select_candidate(static_candidates, cfg)
    k_by_candidate = {f"static_k{k}": k for k in k_grid}
    chosen_k = k_by_candidate.get(static_selection["selected"])

    if_candidates = []
    for q, thr in if_thresholds.items():
        pooled = _validation_pooled(
            lambda fr, thr=thr: model_lib.isolation_forest_alert(fr, fs, if_model, thr))
        if_candidates.append({"candidate_id": f"if_q{q}", "strictness": float(thr),
                              **_cand_summary(pooled)})
    if_selection = evaluator.select_candidate(if_candidates, cfg)
    thr_by_candidate = {f"if_q{q}": thr for q, thr in if_thresholds.items()}
    chosen_if_threshold = thr_by_candidate.get(if_selection["selected"])

    validation_baselines = {
        "no_alert": _validation_pooled(lambda fr: evaluator.no_alert(fr.eligible)),
        "always_alert": _validation_pooled(lambda fr: evaluator.always_alert(fr.eligible)),
    }

    test_results: dict = {}
    for asset in test_assets:
        frame = frames[("test", asset)]
        stratum = {
            "no_alert": _score_asset(frame, evaluator.no_alert(frame.eligible),
                                     ds.labels, "test", asset, events, cfg),
            "always_alert": _score_asset(frame, evaluator.always_alert(frame.eligible),
                                         ds.labels, "test", asset, events, cfg),
        }
        if chosen_k is not None:
            alert = model_lib.static_threshold_alert(frame, sensor_mean_std, chosen_k, sensors)
            stratum["static_threshold"] = _score_asset(frame, alert, ds.labels, "test", asset, events, cfg)
        if chosen_if_threshold is not None:
            alert = model_lib.isolation_forest_alert(frame, fs, if_model, chosen_if_threshold)
            stratum["isolation_forest"] = _score_asset(frame, alert, ds.labels, "test", asset, events, cfg)
        test_results[asset] = stratum

    unseen = set(cfg["unseen_assets"])
    test_strata = {a: ("unseen_asset_later_time" if a in unseen else "seen_asset_later_time")
                  for a in test_assets}

    return {
        "table": table_name,
        "lock_version": lock.get("lock_version"),
        "dataset_hash": generator.dataset_hash(ds),
        "scaler": {"kept_columns": fs.columns, "dropped_zero_variance": fs.dropped,
                  "n_fit_rows": fs.n_fit_rows},
        "static_threshold": {
            "train_mean_std": sensor_mean_std, "k_grid": k_grid,
            "validation_candidates": static_candidates, "selection": static_selection,
            "chosen_k": chosen_k,
        },
        "isolation_forest": {
            "params": cfg["models"]["isolation_forest"]["params"],
            "train_score_quantile_thresholds": if_thresholds,
            "validation_candidates": if_candidates, "selection": if_selection,
            "chosen_threshold": chosen_if_threshold,
        },
        "validation_baselines": validation_baselines,
        "test_strata": test_strata,
        "test_results": test_results,
        "honesty_note": ("Test scored exactly once per stratum, after threshold selection on "
                         "validation only. Test strata are never pooled (evaluator.pool_counts "
                         "refuses to pool any test stratum): report seen-asset (FILLER) and "
                         "unseen-asset (LABELLER) results separately. IsolationForest need not "
                         "beat the static threshold; a negative result is reported as-is."),
    }

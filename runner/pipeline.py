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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runner import lock_guard, models as model_lib

import evaluator
import features as feat
import generator
import twin_config


@dataclass
class SelectionResult:
    """Everything select() computed up to (never including) test scoring:
    the full report shape minus test_results, plus the internal state
    score_test() needs to continue without repeating generation/fitting.
    Exists so a caller (runner.cli) can persist the validation-only
    selection BEFORE any test-split score_stratum call is made - not after
    pipeline.run() has already scored test and merely hand back a slice of
    its result (Astra freeze-review r2 group 4, 2026-09-28)."""

    table: str
    lock: dict
    dataset_hash: str
    scaler_report: dict
    static_threshold_report: dict
    isolation_forest_report: dict
    validation_baselines: dict
    test_strata: dict
    honesty_note: str
    cfg: object
    sensors: list
    frames: dict
    labels: object
    events: object
    test_assets: list
    sensor_mean_std: dict
    fs: object
    if_model: object
    static_selected: str | None
    chosen_k: float | None
    if_selected: str | None
    chosen_if_threshold: float | None


def _cand_summary(pooled: dict) -> dict:
    return {
        "tp": pooled["tp"], "fp": pooled["fp"], "fn": pooled["fn"],
        "false_alerts_per_operating_hour": pooled["false_alerts_per_operating_hour"],
        "total_alerts": pooled["total_alerts"],
    }


def _score_asset(frame, alert, labels, split, asset, events, cfg) -> dict:
    inp = evaluator.stratum_input(frame, alert, labels, split, asset)
    return evaluator.score_stratum(inp, events, cfg)


def select(table_name: str = "dev", *, allow_locked: bool = False, repo_root=None) -> SelectionResult:
    """Everything through validation-only threshold selection - stops
    BEFORE any test-split score_stratum call. No `cfg` parameter: a
    caller-supplied cfg dict could disagree with the one require_lock()
    just verified on disk (Astra freeze-review MUST-FIX 1, 2026-09-28 - a
    probe reached generation with an unverified contamination=0.4 this
    way). The runtime config is always loaded fresh from the exact bytes
    require_lock() just re-hashed, deep-frozen so no caller-held reference
    (e.g. twin_config.load_config()'s own process-wide cached dict) can
    mutate the value this run actually executes against - re-verified
    against the lock at this point of use, not only when require_lock()
    ran (r2 group 1, 2026-09-28)."""
    root = Path(repo_root).resolve() if repo_root is not None else lock_guard.REPO_ROOT
    lock = lock_guard.require_lock(root)
    lock_guard.require_import_paths(root, {
        "src/generator.py": generator,
        "src/features.py": feat,
        "src/evaluator.py": evaluator,
        "src/twin_config.py": twin_config,
    })
    if allow_locked:
        lock_guard.require_clean_worktree(root)
        # Experiment-wide, keyed by lock sha, outside any --run-dir: a
        # second locked evaluation of the SAME experiment-lock identity is
        # refused regardless of --run-dir or caller (r2 group 4).
        lock_guard.reserve_locked_evaluation(root)
    cfg = lock_guard.load_verified_config(root, lock)
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
    static_selected = static_selection["selected"]
    k_by_candidate = {f"static_k{k}": k for k in k_grid}
    # A selected candidate_id of "no_alert" is evaluator.select_candidate's
    # own valid choice (best F1 among within-cap candidates), never "no
    # candidate met the cap" (selected is None then) - the two must not
    # collapse to the same chosen_k=None (Astra freeze-review MUST-FIX 3,
    # 2026-09-28: that dropped the model's whole test row).
    chosen_k = k_by_candidate.get(static_selected) if static_selected != "no_alert" else None

    if_candidates = []
    for q, thr in if_thresholds.items():
        pooled = _validation_pooled(
            lambda fr, thr=thr: model_lib.isolation_forest_alert(fr, fs, if_model, thr))
        if_candidates.append({"candidate_id": f"if_q{q}", "strictness": float(thr),
                              **_cand_summary(pooled)})
    if_selection = evaluator.select_candidate(if_candidates, cfg)
    if_selected = if_selection["selected"]
    thr_by_candidate = {f"if_q{q}": thr for q, thr in if_thresholds.items()}
    chosen_if_threshold = thr_by_candidate.get(if_selected) if if_selected != "no_alert" else None

    validation_baselines = {
        "no_alert": _validation_pooled(lambda fr: evaluator.no_alert(fr.eligible)),
        "always_alert": _validation_pooled(lambda fr: evaluator.always_alert(fr.eligible)),
    }

    unseen = set(cfg["unseen_assets"])
    test_strata = {a: ("unseen_asset_later_time" if a in unseen else "seen_asset_later_time")
                  for a in test_assets}

    return SelectionResult(
        table=table_name,
        lock=lock,
        dataset_hash=generator.dataset_hash(ds),
        scaler_report={"kept_columns": fs.columns, "dropped_zero_variance": fs.dropped,
                      "n_fit_rows": fs.n_fit_rows},
        static_threshold_report={
            "train_mean_std": sensor_mean_std, "k_grid": k_grid,
            "validation_candidates": static_candidates, "selection": static_selection,
            "chosen_k": chosen_k,
        },
        isolation_forest_report={
            # dict(...): a plain, JSON-serializable copy of the frozen cfg
            # mapping - not the MappingProxyType itself (json.dumps cannot
            # serialize that natively).
            "params": dict(cfg["models"]["isolation_forest"]["params"]),
            "train_score_quantile_thresholds": if_thresholds,
            "validation_candidates": if_candidates, "selection": if_selection,
            "chosen_threshold": chosen_if_threshold,
        },
        validation_baselines=validation_baselines,
        test_strata=test_strata,
        honesty_note=("Test scored exactly once per stratum, after threshold selection on "
                      "validation only. Test strata are never pooled (evaluator.pool_counts "
                      "refuses to pool any test stratum): report seen-asset (FILLER) and "
                      "unseen-asset (LABELLER) results separately. IsolationForest need not "
                      "beat the static threshold; a negative result is reported as-is."),
        cfg=cfg,
        sensors=sensors,
        frames=frames,
        labels=ds.labels,
        events=events,
        test_assets=test_assets,
        sensor_mean_std=sensor_mean_std,
        fs=fs,
        if_model=if_model,
        static_selected=static_selected,
        chosen_k=chosen_k,
        if_selected=if_selected,
        chosen_if_threshold=chosen_if_threshold,
    )


def score_test(sel: SelectionResult) -> dict:
    """Score TEST exactly once per stratum, using the model+thresholds
    select() already chose on validation - no threshold search on test.
    Split out from select() so a caller can persist the validation-only
    selection to durable storage BEFORE this runs (Astra freeze-review r2
    group 4, 2026-09-28)."""
    test_results: dict = {}
    for asset in sel.test_assets:
        frame = sel.frames[("test", asset)]
        stratum = {
            "no_alert": _score_asset(frame, evaluator.no_alert(frame.eligible),
                                     sel.labels, "test", asset, sel.events, sel.cfg),
            "always_alert": _score_asset(frame, evaluator.always_alert(frame.eligible),
                                         sel.labels, "test", asset, sel.events, sel.cfg),
        }
        # A "no_alert" selection is a real, protocol-valid policy for that
        # model - it must still get its own test row (identical to the
        # no_alert baseline's own numbers, since that is exactly what was
        # selected), not be silently omitted as if the model had none.
        if sel.static_selected == "no_alert":
            stratum["static_threshold"] = stratum["no_alert"]
        elif sel.chosen_k is not None:
            alert = model_lib.static_threshold_alert(frame, sel.sensor_mean_std, sel.chosen_k, sel.sensors)
            stratum["static_threshold"] = _score_asset(frame, alert, sel.labels, "test", asset,
                                                       sel.events, sel.cfg)
        if sel.if_selected == "no_alert":
            stratum["isolation_forest"] = stratum["no_alert"]
        elif sel.chosen_if_threshold is not None:
            alert = model_lib.isolation_forest_alert(frame, sel.fs, sel.if_model, sel.chosen_if_threshold)
            stratum["isolation_forest"] = _score_asset(frame, alert, sel.labels, "test", asset,
                                                       sel.events, sel.cfg)
        test_results[asset] = stratum

    return {
        "table": sel.table,
        "lock_version": sel.lock.get("lock_version"),
        "dataset_hash": sel.dataset_hash,
        "scaler": sel.scaler_report,
        "static_threshold": sel.static_threshold_report,
        "isolation_forest": sel.isolation_forest_report,
        "validation_baselines": sel.validation_baselines,
        "test_strata": sel.test_strata,
        "test_results": test_results,
        "honesty_note": sel.honesty_note,
    }


def run(table_name: str = "dev", *, allow_locked: bool = False, repo_root=None) -> dict:
    """Convenience wrapper: select() then score_test() in one call, for
    dev/fixture iteration and any caller that does not need the
    selection-before-test persistence boundary select()/score_test() exist
    to provide."""
    return score_test(select(table_name, allow_locked=allow_locked, repo_root=repo_root))

"""Features: causality, split reset, leak guard, missingness, train-only
scaling. Hand-built inputs and development seeds only."""
import copy

import numpy as np
import pytest

import features as F
import generator as g

T0 = 1_700_006_400          # arbitrary UTC midnight used by hand-built fixtures
DAY = 86400


def _readings(n=DAY, fill=None, seed=0):
    rng = np.random.default_rng(seed)
    vals = {s: 10.0 + rng.standard_normal(n) for s in
            ("vibration_rms", "bearing_temp_c", "motor_current_a", "line_pressure_bar")}
    if fill:
        fill(vals)
    return g.SensorReadings(T0, vals)


def _frame(r, split="train"):
    return F.compute_features(r, r.t0, r.t0 + r.n, split_name=split)


# ------------------------------------------------------------- leak guard
def test_output_columns_are_exactly_the_allow_list(cfg):
    fr = _frame(_readings(3600))
    F.assert_no_leak(fr.columns.keys())
    assert list(fr.columns) == F.feature_columns(cfg)
    assert len(fr.columns) == 4 * (2 * 3 + 3)


@pytest.mark.parametrize("bad", ["asset_id", "event_id", "fault_type", "label", "seed",
                                 "split", "operating_state", "planned_stop", "onset_s",
                                 "Machine", "regime", "is_changeover"])
def test_deny_list_rejects_label_like_columns(bad):
    cols = F.feature_columns() + [bad]
    with pytest.raises(F.LeakError):
        F.assert_no_leak(cols)


def test_non_allow_listed_column_rejected():
    with pytest.raises(F.LeakError):
        F.assert_no_leak(F.feature_columns() + ["vibration_rms__max_5m"])


def test_label_channel_mutation_does_not_change_features(dev_ds):
    """Perturb only the label channel of a whole dev dataset; every feature
    value of every (split, asset) is unchanged."""
    before = F.features_for_dataset(dev_ds)
    mutated = copy.copy(dev_ds)
    mutated.labels = copy.deepcopy(dev_ds.labels)
    for e in mutated.labels.events:
        e.onset += 3600
        e.end += 3600
        e.fault_type = "none"
        e.channel = "equipment" if e.channel == "data" else "data"
        e.event_id = "X" + e.event_id
    mutated.labels.planned_stops.clear()
    mutated.labels.regime_segments.clear()
    mutated.labels.normal_changes.append({"start": 0, "end": 1})
    after = F.features_for_dataset(mutated)
    assert before.keys() == after.keys()
    for key in before:
        F.assert_no_leak(after[key].columns.keys())
        for c in before[key].columns:
            assert np.array_equal(before[key].columns[c], after[key].columns[c], equal_nan=True)
        assert np.array_equal(before[key].dq_reason, after[key].dq_reason)


def test_features_refuse_non_raw_inputs(dev_ds):
    with pytest.raises(TypeError):
        F.compute_features(dev_ds.labels, 0, 10)
    with pytest.raises(TypeError):
        F.compute_features({"vibration_rms": np.zeros(10)}, 0, 10)


# ------------------------------------------------------------- causality
def test_future_sample_perturbation_leaves_earlier_features_unchanged():
    r = _readings()
    base = _frame(r)
    cut = T0 + 40_000          # perturb every sample at t >= cut
    vals2 = {k: v.copy() for k, v in r.values.items()}
    for k in vals2:
        vals2[k][cut - T0:] += 50.0
        vals2[k][cut - T0 + 100: cut - T0 + 700] = np.nan
    pert = _frame(g.SensorReadings(T0, vals2))
    ends = base.window_start + 10
    early = ends <= cut
    later = ends > cut
    for c in base.columns:
        assert np.array_equal(base.columns[c][early], pert.columns[c][early], equal_nan=True), c
    assert np.array_equal(base.dq_reason[early], pert.dq_reason[early])
    assert not np.allclose(base.columns["vibration_rms__mean_5m"][later],
                           pert.columns["vibration_rms__mean_5m"][later], equal_nan=True)


def test_values_match_brute_force_causal_definition(cfg):
    def holes(v):
        for k in v:
            v[k][5000:5400] = np.nan
            v[k][::97] = np.nan
    r = _readings(n=7200, fill=holes, seed=3)
    fr = _frame(r)
    x = r.values["motor_current_a"]
    t = np.arange(len(x))
    for k in (185, 300, 529, 540, 719):
        e = (k + 1) * 10
        for w, W in (("5m", 300), ("30m", 1800)):
            lo = max(0, e - W)
            xs, ts = x[lo:e], t[lo:e]
            m = np.isfinite(xs)
            xs, ts = xs[m], ts[m]
            got_mean = fr.columns[f"motor_current_a__mean_{w}"][k]
            if len(xs) < cfg["features"]["min_observations"][w]:
                assert np.isnan(got_mean)
                continue
            assert got_mean == pytest.approx(xs.mean(), rel=1e-12)
            assert fr.columns[f"motor_current_a__std_{w}"][k] == pytest.approx(xs.std(), rel=1e-9)
            slope = np.polyfit(ts.astype(float), xs, 1)[0]
            assert fr.columns[f"motor_current_a__slope_{w}"][k] == pytest.approx(slope, rel=1e-7, abs=1e-12)
            exp_n = e - lo
            assert fr.columns[f"motor_current_a__missing_frac_{w}"][k] == pytest.approx(1 - len(xs) / exp_n)


def test_linear_signal_slope():
    def lin(v):
        for k in v:
            v[k][:] = 5.0 + 0.01 * np.arange(len(v[k]))
    fr = _frame(_readings(n=3600, fill=lin))
    k = 300
    assert fr.columns["line_pressure_bar__slope_5m"][k] == pytest.approx(0.01)
    assert fr.columns["line_pressure_bar__slope_30m"][k] == pytest.approx(0.01)


# ------------------------------------------------------ split reset / warm-up
def test_split_span_enforced_no_carry():
    r = _readings(n=7200)
    with pytest.raises(F.SplitSpanError):
        F.compute_features(r, T0, T0 + 3600)          # readings extend past the split
    with pytest.raises(F.SplitSpanError):
        F.compute_features(r, T0 - 3600, T0 + 7200)   # would need data before the split


def test_state_resets_at_split_boundary(dev_ds):
    """Test-split FILLER features equal those computed from the test split's
    readings alone, and the first windows use only the split's own samples."""
    frames = F.features_for_dataset(dev_ds)
    sp = dev_ds.splits["test"]
    r = dev_ds.readings[("test", "FILLER")]
    alone = F.compute_features(g.SensorReadings(r.t0, {k: v.copy() for k, v in r.values.items()}),
                               sp.start, sp.end, split_name="test")
    for c in alone.columns:
        assert np.array_equal(frames[("test", "FILLER")].columns[c], alone.columns[c], equal_nan=True)
    # window 0 of the split: 30 min summary covers 10 s of this split only
    x = r.values["bearing_temp_c"][:10]
    x = x[np.isfinite(x)]
    assert alone.columns["bearing_temp_c__missing_frac_30m"][0] == pytest.approx(1 - len(x) / 10)
    assert alone.dq_reason[0] == "warm_up"


def test_warmup_excluded_and_reported(cfg):
    fr = _frame(_readings(n=7200))
    n_warm = cfg["generator"]["warmup_minutes"] * 60 // 10
    assert (fr.dq_reason[:n_warm] == "warm_up").all()
    assert not fr.eligible[:n_warm].any()
    assert fr.eligible[n_warm:].all()
    rep = F.excluded_report(fr)
    assert rep["excluded"]["warm_up"] == {"windows": n_warm, "seconds": n_warm * 10}


# ------------------------------------------------------------- missingness
def test_complete_dropout_stays_in_denominator():
    a, b = 20_000, 20_000 + 900   # 15-minute whole-asset dropout

    def drop(v):
        for k in v:
            v[k][a:b] = np.nan
    fr = _frame(_readings(fill=drop))
    assert len(fr.window_start) == DAY // 10          # no window vanishes
    k = np.arange(a // 10, b // 10)
    for s in ("vibration_rms", "line_pressure_bar"):
        assert (fr.columns[f"{s}__missing_frac_10s"][k] == 1.0).all()
    # every window of a COMPLETE (all-sensor) dropout is ineligible - never
    # healthy on stale rolling history - for its whole duration, not only
    # once the rolling window itself runs dry (Astra MUST-FIX 1)
    assert not fr.eligible[k].any()
    assert (fr.dq_reason[k] == "all_sensors_missing").all()
    rep = F.excluded_report(fr)
    assert rep["excluded"]["all_sensors_missing"]["windows"] == len(k)
    assert rep["observed_sample_fraction"] < 1.0
    # missing values are never imputed to a "healthy" number
    assert np.isnan(fr.columns["vibration_rms__mean_5m"][k[-1]])


# --------------------------- current-window health (Astra MUST-FIX 1) -----
# Wired in from vault:40-sessions/2026-09-28-astra-twin-lock-probes.py: a
# complete current-window dropout must never be reported healthy on stale
# rolling history.
def test_complete_all_sensor_dropout_is_ineligible(cfg):
    x = {s: np.ones(3600) for s in cfg["sensors"]["order"]}
    for a in x.values():
        a[2400:2520] = np.nan
    fr = F.compute_features(g.SensorReadings(0, x), 0, 3600, split_name="dev")
    i = (fr.window_start >= 2400) & (fr.window_start < 2520)
    assert int(i.sum()) == 12
    assert not fr.eligible[i].any()
    assert set(fr.dq_reason[i].tolist()) == {"all_sensors_missing"}


def test_partial_sensor_dropout_is_ineligible(cfg):
    x = {s: np.ones(3600) for s in cfg["sensors"]["order"]}
    sensors = cfg["sensors"]["order"]
    x[sensors[0]][2400:2520] = np.nan          # only one of four sensors goes dark
    fr = F.compute_features(g.SensorReadings(0, x), 0, 3600, split_name="dev")
    i = (fr.window_start >= 2400) & (fr.window_start < 2520)
    assert int(i.sum()) == 12
    assert not fr.eligible[i].any()
    assert set(fr.dq_reason[i].tolist()) == {"partial_sensor_missing"}


def test_non_finite_input_is_labelled():
    def inf(v):
        v["bearing_temp_c"][30_000] = np.inf
    fr = _frame(_readings(fill=inf))
    k = 30_000 // 10
    assert fr.dq_reason[k] == "non_finite_input"
    assert fr.dq_reason[k + 179] == "non_finite_input"      # still inside the 30 min window
    assert fr.dq_reason[k + 180] == ""


# ------------------------------------------------------------- scaler
def _manual_frame(split, shift=0.0, n=400, seed=1):
    cols = F.feature_columns()
    rng = np.random.default_rng(seed)
    data = {c: rng.standard_normal(n) + shift for c in cols}
    data[cols[1]] = np.full(n, 3.0)          # zero-variance column in this fixture
    return F.FeatureFrame(np.arange(n) * 10, data, np.full(n, "", dtype=object),
                          np.ones(n, dtype=bool), {"split": split})


def test_scaler_is_fitted_on_train_only():
    tr = _manual_frame("train")
    va = _manual_frame("validation", shift=100.0, seed=2)
    fs = F.fit_scaler([tr])
    with pytest.raises(F.TrainOnlyError):
        F.fit_scaler([tr, va])
    with pytest.raises(F.TrainOnlyError):
        F.fit_scaler([va])
    kept = fs.columns
    X = np.column_stack([tr.columns[c] for c in kept])
    assert np.allclose(fs.scaler.center_, np.median(X, axis=0))
    # validation data is transformed with train statistics, not re-centred
    z = F.transform(fs, va)
    assert np.median(z) > 50
    # frozen zero-variance rule: drop
    assert fs.dropped == [F.feature_columns()[1]]
    assert F.feature_columns()[1] not in kept
    assert type(fs.scaler).__name__ == "RobustScaler"


def test_scaler_uses_eligible_rows_only():
    tr = _manual_frame("train")
    tr.eligible[:200] = False
    for c in tr.columns:
        tr.columns[c][:200] = 1e6
    fs = F.fit_scaler([tr])
    assert np.all(np.abs(fs.scaler.center_) < 10)

"""monitoring.rules: hand-built fixtures only."""
import numpy as np

from monitoring.rules import expanding_zscore_alerts


def _frame(vibration, eligible):
    import features
    n = len(vibration)
    ws = np.arange(0, n * 10, 10, dtype=np.int64)
    cols = {"vibration_rms__mean_5m": np.asarray(vibration, dtype=np.float64)}
    elig = np.asarray(eligible, dtype=bool)
    dq = np.where(elig, "", "excluded").astype(object)
    return features.FeatureFrame(ws, cols, dq, elig, {})


def _baseline_then_spike(n_baseline=60, spike_at=80, spike_value=500.0, n_total=100):
    rng = np.random.default_rng(0)
    vals = 10.0 + 0.1 * rng.standard_normal(n_total)
    vals[spike_at] = spike_value
    eligible = [True] * n_total
    return vals, eligible


def test_no_alert_before_min_history_is_reached():
    vals, eligible = _baseline_then_spike()
    frame = _frame(vals, eligible)
    alert, sensor, detail, z = expanding_zscore_alerts(frame, ["vibration_rms"], k=3.0, min_history=30)
    assert not alert[:30].any()


def test_alerts_on_a_clear_spike_after_the_baseline():
    vals, eligible = _baseline_then_spike()
    frame = _frame(vals, eligible)
    alert, sensor, detail, z = expanding_zscore_alerts(frame, ["vibration_rms"], k=3.0, min_history=30)
    assert alert[80]
    assert sensor[80] == "vibration_rms"
    assert "vibration_rms" in detail[80]
    assert not alert[31:79].any()   # quiet baseline noise never trips k=3


def test_ineligible_windows_never_alert():
    vals, eligible = _baseline_then_spike()
    eligible = list(eligible)
    eligible[80] = False           # the spike window itself is ineligible
    frame = _frame(vals, eligible)
    alert, _, _, _ = expanding_zscore_alerts(frame, ["vibration_rms"], k=3.0, min_history=30)
    assert not alert[80]


def test_causal_a_later_perturbation_never_changes_an_earlier_decision():
    vals, eligible = _baseline_then_spike()
    frame_a = _frame(vals, eligible)
    alert_a, _, _, z_a = expanding_zscore_alerts(frame_a, ["vibration_rms"], k=3.0, min_history=30)

    vals_b = list(vals)
    vals_b[95] = 9999.0            # perturb a sample AFTER the spike at 80
    frame_b = _frame(vals_b, eligible)
    alert_b, _, _, z_b = expanding_zscore_alerts(frame_b, ["vibration_rms"], k=3.0, min_history=30)

    np.testing.assert_array_equal(alert_a[:95], alert_b[:95])
    np.testing.assert_allclose(z_a[:95], z_b[:95], equal_nan=True)


def test_worst_sensor_is_reported_when_several_trip_at_once():
    import features

    n = 100
    rng = np.random.default_rng(1)
    vib = 10.0 + 0.1 * rng.standard_normal(n)
    temp = 48.0 + 0.2 * rng.standard_normal(n)
    vib[90] = 200.0   # big |z|
    temp[90] = 49.0   # small |z|
    ws = np.arange(0, n * 10, 10, dtype=np.int64)
    cols = {"vibration_rms__mean_5m": vib, "bearing_temp_c__mean_5m": temp}
    elig = np.ones(n, dtype=bool)
    frame = features.FeatureFrame(ws, cols, np.full(n, "", dtype=object), elig, {})
    alert, sensor, detail, _ = expanding_zscore_alerts(frame, ["vibration_rms", "bearing_temp_c"],
                                                       k=3.0, min_history=30)
    assert alert[90]
    assert sensor[90] == "vibration_rms"
    assert "vibration_rms" in detail[90]


def test_sensor_and_detail_are_empty_where_there_is_no_alert():
    vals, eligible = _baseline_then_spike()
    frame = _frame(vals, eligible)
    alert, sensor, detail, _ = expanding_zscore_alerts(frame, ["vibration_rms"], k=3.0, min_history=30)
    quiet = ~alert
    assert all(sensor[quiet] == "")
    assert all(detail[quiet] == "")

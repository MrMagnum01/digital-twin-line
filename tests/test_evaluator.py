"""Evaluator correctness on HAND-BUILT fixtures only (no model, no generated
scores). Covers every adversarial fixture listed in the binding amendments'
"Release claims and checks" that belongs to scoring."""
import numpy as np
import pytest

import evaluator as E

S = 1_000_000_000 - (1_000_000_000 % 86400)   # a UTC midnight
HOURS = 6
N = HOURS * 360


def stratum(alert_ranges=(), split="validation", asset="FILLER", eligible=None,
            planned_stops=(), normal_changes=(), reasons=None):
    ws = S + 10 * np.arange(N, dtype=np.int64)
    elig = np.ones(N, dtype=bool) if eligible is None else eligible
    alert = np.zeros(N, dtype=bool)
    for a, b in alert_ranges:            # seconds from S, window starts in [a, b)
        alert |= (ws >= S + a) & (ws < S + b)
    return E.StratumInput(split, asset, ws, elig, alert & elig,
                          dq_reason=reasons, planned_stops=[(S + a, S + b) for a, b in planned_stops],
                          normal_changes=[(S + a, S + b) for a, b in normal_changes])


def ev(eid, onset, end, ftype="bearing_wear", channel="equipment", split="validation", asset="FILLER"):
    return E.Event(eid, split, asset, ftype, channel, S + onset, S + end)


def el(res):
    return res["event_level"]


# -------------------------------------------------------------- debounce
def test_debounce_gap_rule():
    ws = S + 10 * np.arange(20)
    a = np.zeros(20, dtype=bool)
    a[[0, 7]] = True          # window 0 ends S+10, window 7 starts S+70: gap 60 -> merge
    assert len(E.debounce(ws, a)) == 1
    a[:] = False
    a[[0, 8]] = True          # gap 70 -> two episodes
    eps = E.debounce(ws, a)
    assert len(eps) == 2 and eps[0]["start"] == S + 10 and eps[1]["start"] == S + 90


# ------------------------------------------------- amendment fixtures
def test_one_alert_spanning_two_faults_yields_at_most_one_tp():
    events = [ev("E1", 3600, 4200), ev("E2", 3700, 4300, ftype="overheating")]
    res = E.score_stratum(stratum([(3600, 4400)]), events)
    assert el(res)["tp"] == 1 and el(res)["fn"] == 1 and el(res)["fp"] == 0
    rows = {r["event_id"]: r for r in el(res)["match_table"]}
    assert rows["E1"]["matched_episode_id"] == 0 and rows["E2"]["matched_episode_id"] is None
    assert rows["E1"]["overlapping_event_ids"] == ["E2"]


def test_repeated_alerts_within_a_matched_event_are_fp():
    events = [ev("E1", 3600, 7200)]
    res = E.score_stratum(stratum([(3700, 3720), (4000, 4020), (5000, 5020)]), events)
    assert el(res)["tp"] == 1 and el(res)["fp"] == 2 and el(res)["fn"] == 0
    assert el(res)["precision"] == pytest.approx(1 / 3)


def test_always_alert_cannot_get_perfect_recall_by_episode_reuse():
    events = [ev("E1", 3600, 4200), ev("E2", 9000, 9600), ev("E3", 15000, 15600)]
    inp = stratum()
    inp.alert = E.always_alert(inp.eligible)
    res = E.score_stratum(inp, events)
    assert el(res)["episodes"] == 1
    assert el(res)["tp"] <= 1
    assert el(res)["recall"] < 1.0
    # the continuous episode starts before every onset: pre-onset never credited
    assert el(res)["tp"] == 0 and el(res)["fn"] == 3 and el(res)["fp"] == 1
    assert el(res)["alert_active_fraction_operating"] == 1.0


def test_always_alert_with_gaps_still_one_tp_per_episode():
    events = [ev("E1", 3600, 4200), ev("E2", 3650, 4250), ev("E3", 3700, 4300)]
    elig = np.ones(N, dtype=bool)
    elig[(np.arange(N) >= 355) & (np.arange(N) < 362)] = False   # gap 3550..3620 -> new episode
    inp = stratum(eligible=elig)
    inp.alert = E.always_alert(inp.eligible)
    res = E.score_stratum(inp, events)
    assert el(res)["tp"] == 1 and el(res)["fn"] == 2


def test_no_event_no_alert_strata_report_na():
    res = E.score_stratum(stratum(), [])
    e = el(res)
    assert e["precision"] is None and e["recall"] is None and e["f1"] is None
    assert e["delay"]["p50"] is None and e["delay"]["p90"] is None and e["delay"]["miss_rate"] is None
    assert e["recall_by_channel"]["equipment"]["recall"] is None
    for ch in ("equipment", "data"):
        p = res["point_level"][ch]
        assert p["precision"] is None and p["recall"] is None and p["f1"] is None


def test_no_alert_with_events_is_f1_zero_not_na():
    events = [ev("E1", 3600, 4200)]
    inp = stratum()
    inp.alert = E.no_alert(inp.eligible)
    e = el(E.score_stratum(inp, events))
    assert e["f1"] == 0 and e["recall"] == 0 and e["precision"] is None
    assert e["false_alerts_per_operating_hour"] == 0
    assert e["delay"]["matched_events"] == 0 and e["delay"]["miss_rate"] == 1.0
    assert e["delay"]["p50"] is None


def test_alerts_with_no_events_have_na_recall_and_zero_precision():
    e = el(E.score_stratum(stratum([(3600, 3610)]), []))
    assert e["recall"] is None and e["precision"] == 0 and e["f1"] == 0 and e["fp"] == 1


def test_split_overlap_is_refused():
    with pytest.raises(E.SplitOverlapError):
        E.validate_splits({"train": (0, 100), "validation": (50, 200)})
    inp = stratum()
    with pytest.raises(E.SplitOverlapError):
        E.score_all([inp], [], {"validation": (S, S + N * 10), "test": (S + 100, S + 10 ** 6)})


def test_dropout_stays_in_coverage_accounting():
    """Dropout windows are ineligible (no score) but stay in the denominators:
    counted as excluded positives for the data channel, never as TN."""
    elig = np.ones(N, dtype=bool)
    k = (np.arange(N) >= 360) & (np.arange(N) < 450)       # S+3600 .. S+4500
    elig[k] = False
    reasons = np.where(elig, "", "insufficient_observations").astype(object)
    events = [ev("D1", 3600, 4500, ftype="dropout", channel="data")]
    res = E.score_stratum(stratum(eligible=elig, reasons=reasons), events)
    p = res["point_level"]["data"]
    assert res["windows_total"] == N
    assert p["excluded_windows"] == 90 and p["excluded_positive_windows"] == 90
    assert p["excluded_by_reason"]["insufficient_observations"] == {"windows": 90, "positive_windows": 90}
    assert p["tn"] + p["fp"] + p["fn"] + p["tp"] == N - 90
    assert p["coverage"] == pytest.approx((N - 90) / N)
    assert el(res)["fn"] == 1 and el(res)["recall_by_channel"]["data"]["events"] == 1


def test_alert_on_ineligible_window_is_refused():
    elig = np.ones(N, dtype=bool)
    elig[5] = False
    inp = stratum(eligible=elig)
    inp.alert = np.ones(N, dtype=bool)
    with pytest.raises(E.InputError):
        E.score_stratum(inp, [])


# ------------------------------------------------------------- matching
def test_pre_onset_alarm_not_credited():
    events = [ev("E1", 3600, 4200)]
    e = el(E.score_stratum(stratum([(3000, 3100)]), events))
    assert e["tp"] == 0 and e["fn"] == 1 and e["fp"] == 1


def test_match_window_includes_tail_and_excludes_beyond():
    events = [ev("E1", 3600, 4200)]
    # alert window starting 4790 ends at 4800 = end + 600: matched (inclusive)
    assert el(E.score_stratum(stratum([(4790, 4800)]), events))["tp"] == 1
    assert el(E.score_stratum(stratum([(4800, 4810)]), events))["tp"] == 0


def test_carrying_alert_through_event_is_not_a_new_start():
    events = [ev("E1", 3600, 4200)]
    e = el(E.score_stratum(stratum([(3000, 5000)]), events))
    assert e["tp"] == 0 and e["episodes"] == 1


def test_matching_order_is_onset_then_event_id():
    events = [ev("B", 3600, 4200), ev("A", 3600, 4200)]
    e = el(E.score_stratum(stratum([(3700, 3710), (4000, 4010)]), events))
    rows = {r["event_id"]: r for r in e["match_table"]}
    assert rows["A"]["matched_alert_start"] == S + 3710
    assert rows["B"]["matched_alert_start"] == S + 4010
    assert e["tp"] == 2


def test_delay_matched_only_with_documented_quantiles():
    events = [ev(f"E{i}", 1800 + i * 3000, 2400 + i * 3000) for i in range(5)]
    alerts = [(1800 + i * 3000 + d, 1810 + i * 3000 + d) for i, d in enumerate([0, 60, 120, 300])]
    e = el(E.score_stratum(stratum(alerts), events))
    assert e["tp"] == 4 and e["fn"] == 1
    delays = sorted(r["delay_s"] for r in e["match_table"] if r["delay_s"] is not None)
    assert delays == [10, 70, 130, 310]
    assert e["delay"]["p50"] == pytest.approx(np.quantile(delays, 0.5, method="linear"))
    assert e["delay"]["p90"] == pytest.approx(np.quantile(delays, 0.9, method="linear"))
    assert e["delay"]["miss_rate"] == pytest.approx(0.2)
    assert e["delay"]["matched_events"] == 4 and e["delay"]["events"] == 5


def test_recall_by_type_and_channel_generic_precision():
    events = [ev("E1", 3600, 4200), ev("D1", 9000, 9600, ftype="stuck_sensor", channel="data")]
    e = el(E.score_stratum(stratum([(3700, 3710)]), events))
    assert e["recall_by_fault_type"]["bearing_wear"]["recall"] == 1.0
    assert e["recall_by_fault_type"]["stuck_sensor"]["recall"] == 0.0
    assert e["recall_by_channel"] == {
        "equipment": {"events": 1, "tp": 1, "fn": 0, "recall": 1.0},
        "data": {"events": 1, "tp": 0, "fn": 1, "recall": 0.0}}
    assert "generic" in e["precision_kind"]


# ------------------------------------------------------------ point level
def test_point_level_no_point_adjustment_and_channels_separate():
    events = [ev("E1", 3600, 4200), ev("D1", 9000, 9600, ftype="dropout", channel="data")]
    res = E.score_stratum(stratum([(3700, 3710)]), events)
    pe = res["point_level"]["equipment"]
    assert el(res)["tp"] == 1
    assert pe["tp"] == 1 and pe["fn"] == 59       # one hit does not mark the interval
    assert pe["recall"] == pytest.approx(1 / 60)
    pd = res["point_level"]["data"]
    assert pd["tp"] == 0 and pd["fn"] == 60 and pd["fp"] == 1


def test_point_truth_uses_intersection_with_window():
    events = [ev("E1", 3605, 3606)]                # one second inside window [3600, 3610)
    res = E.score_stratum(stratum([(3600, 3610)]), events)
    assert res["point_level"]["equipment"]["tp"] == 1
    assert res["point_level"]["equipment"]["eligible_positive_windows"] == 1


def test_point_predictions_are_raw_windows_before_debounce():
    res = E.score_stratum(stratum([(3600, 3610), (3640, 3650)]), [])
    assert el(res)["episodes"] == 1
    assert res["point_level"]["equipment"]["fp"] == 2


# --------------------------------------------- operating hours / stops
def test_planned_stop_false_alarms_reported_separately():
    stops = [(7200, 10800)]
    res = E.score_stratum(stratum([(8000, 8010), (12000, 12010)], planned_stops=stops), [])
    e = el(res)
    assert e["false_alerts_operating"] == 1
    assert e["operating_hours_observed_eligible"] == pytest.approx(HOURS - 1)
    assert e["false_alerts_per_operating_hour"] == pytest.approx(1 / (HOURS - 1))
    assert e["planned_stop"]["unmatched_episodes"] == 1
    assert e["planned_stop"]["observed_hours"] == pytest.approx(1.0)
    assert e["planned_stop"]["false_alerts_per_observed_hour"] == pytest.approx(1.0)
    assert e["fp"] == 2                  # nothing hidden from the FP count


def test_unseen_normal_change_false_alerts():
    res = E.score_stratum(stratum([(15000, 15100)], split="test", normal_changes=[(14400, 21600)]), [])
    u = el(res)["unseen_normal_change"]
    assert u["unmatched_episodes"] == 1 and u["alert_active_seconds"] == 100
    assert u["observed_hours"] == pytest.approx(2.0)


# ------------------------------------------------- strata / pooling
def test_event_outside_split_is_refused():
    with pytest.raises(E.InputError):
        E.score_stratum(stratum(), [ev("E1", N * 10 - 300, N * 10 - 100)])   # tail beyond split


def test_score_all_rejects_windows_outside_split_and_orphan_events():
    inp = stratum()
    with pytest.raises(E.InputError):
        E.score_all([inp], [], {"validation": (S + 10, S + N * 10)})
    with pytest.raises(E.InputError):
        E.score_all([inp], [ev("E1", 3600, 4200, asset="CAPPER")], {"validation": (S, S + N * 10)})


def test_test_strata_are_never_pooled():
    rng = {"test": (S, S + N * 10)}
    res = E.score_all([stratum(split="test", asset="FILLER"), stratum(split="test", asset="LABELLER")],
                      [], rng)
    assert set(res) == {"test/FILLER", "test/LABELLER"}
    with pytest.raises(E.PoolingError):
        E.pool_counts(list(res.values()))
    val = E.score_all([stratum(asset="FILLER", alert_ranges=[(100, 110)]), stratum(asset="CAPPER")],
                      [], {"validation": (S, S + N * 10)})
    pooled = E.pool_counts(list(val.values()))
    assert pooled["fp"] == 1 and pooled["operating_hours"] == pytest.approx(2 * HOURS)


# ------------------------------------------- selection rule (fixture only)
def test_select_candidate_rule_on_hand_built_counts():
    cands = [
        {"candidate_id": "k2", "strictness": 2, "tp": 8, "fp": 30, "fn": 2,
         "false_alerts_per_operating_hour": 1.5, "total_alerts": 38},          # over cap
        {"candidate_id": "k3", "strictness": 3, "tp": 6, "fp": 4, "fn": 4,
         "false_alerts_per_operating_hour": 0.4, "total_alerts": 10},
        {"candidate_id": "k4", "strictness": 4, "tp": 6, "fp": 4, "fn": 4,
         "false_alerts_per_operating_hour": 0.4, "total_alerts": 10},          # tie -> stricter
        {"candidate_id": "k5", "strictness": 5, "tp": 5, "fp": 5, "fn": 5,
         "false_alerts_per_operating_hour": 0.5, "total_alerts": 10},
    ]
    out = E.select_candidate(cands)
    assert out["selected"] == "k4"
    ids = [c["candidate_id"] for c in out["table"]]
    assert "no_alert" in ids and len(ids) == 5           # all candidates retained
    no_alert = next(c for c in out["table"] if c["candidate_id"] == "no_alert")
    assert no_alert["f1"] == 0 and no_alert["within_cap"]


def test_select_candidate_tiebreak_fewer_false_alerts_then_fewer_alerts():
    base = {"false_alerts_per_operating_hour": 0.2}
    cands = [
        {"candidate_id": "a", "strictness": 1, "tp": 2, "fp": 2, "fn": 2, "total_alerts": 4, **base},
        {"candidate_id": "b", "strictness": 0, "tp": 3, "fp": 1, "fn": 3, "total_alerts": 4, **base},
    ]
    # F1(a) = 4/8 = 0.5; F1(b) = 6/10 = 0.6 -> b
    assert E.select_candidate(cands)["selected"] == "b"
    cands = [
        {"candidate_id": "a", "strictness": 1, "tp": 2, "fp": 2, "fn": 2, "total_alerts": 5, **base},
        {"candidate_id": "b", "strictness": 0, "tp": 2, "fp": 2, "fn": 2, "total_alerts": 4, **base},
    ]
    assert E.select_candidate(cands)["selected"] == "b"   # same F1 and FP -> fewer alerts


def test_select_candidate_no_alert_wins_when_everything_breaks_cap():
    cands = [{"candidate_id": "k1", "strictness": 1, "tp": 9, "fp": 90, "fn": 1,
              "false_alerts_per_operating_hour": 5.0, "total_alerts": 99}]
    assert E.select_candidate(cands)["selected"] == "no_alert"


# ------------------------------------ dev-seed wiring (structure only)
def test_dev_seed_wiring_with_trivial_baselines(dev_ds):
    """Plumbing check on DEVELOPMENT seeds: features -> evaluator with the two
    trivial baselines. Asserts structure/accounting only; no model, no
    threshold, no locked seed."""
    import features as F
    frames = F.features_for_dataset(dev_ds)
    events = E.events_from_labels(dev_ds.labels)
    ranges = {n: (sp.start, sp.end) for n, sp in dev_ds.splits.items()}
    for make in (E.no_alert, E.always_alert):
        strata = [E.stratum_input(fr, make(fr.eligible), dev_ds.labels, s, a)
                  for (s, a), fr in frames.items()]
        res = E.score_all(strata, events, ranges)
        assert set(res) == {f"{s}/{a}" for s, a in frames}
        for key, r in res.items():
            s, a = key.split("/")
            n_ev = sum(1 for e in events if e.split == s and e.asset == a)
            e = r["event_level"]
            assert e["tp"] + e["fn"] == n_ev == len(e["match_table"])
            assert e["tp"] <= e["episodes"]
            for ch in ("equipment", "data"):
                p = r["point_level"][ch]
                assert p["tp"] + p["fp"] + p["fn"] + p["tn"] + p["excluded_windows"] == r["windows_total"]
                assert p["excluded_by_reason"]["warm_up"]["windows"] == 180
            if make is E.no_alert:
                assert e["episodes"] == 0 and (e["f1"] == 0 if n_ev else e["f1"] is None)
            else:
                # one continuous episode per eligible run: it cannot be reused
                assert e["tp"] <= e["episodes"] and e["alert_active_fraction_operating"] == 1.0

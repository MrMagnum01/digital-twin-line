"""Generator: determinism, seed derivation, split separation, fault rules.
Development seeds only (901-906); the locked table is only ever checked for
its date mapping and for being refused."""
import copy
import dataclasses
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import generator as g

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- determinism
def test_same_dev_seeds_identical_within_process(dev_ds):
    again = g.generate("dev")
    assert g.dataset_hash(again) == g.dataset_hash(dev_ds)
    for key, r in dev_ds.readings.items():
        r2 = again.readings[key]
        assert r2.t0 == r.t0
        for s in r.values:
            assert np.array_equal(r.values[s], r2.values[s], equal_nan=True)
    assert g.labels_to_json(again.labels) == g.labels_to_json(dev_ds.labels)


def test_same_dev_seeds_identical_across_processes(dev_ds):
    hashes = []
    for hashseed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, str(REPO / "src/generator.py"), "--hash-only"],
                             cwd=REPO, env=env, capture_output=True, text=True, check=True)
        hashes.append(out.stdout.strip())
    assert hashes[0] == hashes[1] == g.dataset_hash(dev_ds)


def test_seed_derivation_is_splitmix64_not_hash():
    # First SplitMix64 output for state 0 (published reference value).
    assert g.mix64(0) == 0xE220A8397B1DCDAF
    a = g.derive_seed(901, 1, 1)
    assert a == g.derive_seed(901, 1, 1)
    assert len({g.derive_seed(901, 1, s) for s in range(1, 16)}) == 15
    assert g.derive_seed(901, 1, 1) != g.derive_seed(901, 2, 1)
    assert g.derive_seed(901, 1, 1) != g.derive_seed(902, 1, 1)
    import ast
    for mod in ("generator.py", "features.py", "evaluator.py", "ingest.py"):
        tree = ast.parse((REPO / "src" / mod).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "hash"]
        assert not calls, f"builtin hash() used in {mod}"


def test_distinct_dates_have_distinct_noise(dev_ds):
    r = dev_ds.readings[("train", "FILLER")]
    d = 86400
    v = r.values["vibration_rms"]
    assert not np.array_equal(v[:d], v[d:2 * d], equal_nan=True)


# ------------------------------------------------------------- locked guard
def test_locked_table_is_refused():
    with pytest.raises(g.LockedSeedsError):
        g.generate("locked")


def test_any_locked_seed_is_refused(cfg):
    t = copy.deepcopy(cfg["split_table_dev"])
    t["splits"][2]["seeds"] = [905, 301]
    with pytest.raises(g.LockedSeedsError):
        g.generate(table=t)


# ------------------------------------------------------------ split layout
def test_locked_table_date_mapping_structure(cfg):
    """Structure only (date mapping); nothing is generated."""
    specs = g.build_split_specs(cfg["split_table_locked"], cfg)
    dates = [d for sp in specs.values() for d in sp.dates]
    assert len(dates) == len(set(dates)) == 20
    ds = [date.fromisoformat(x) for x in dates]
    assert all((b - a).days == 1 for a, b in zip(ds, ds[1:]))
    assert [len(specs[n].dates) for n in ("train", "validation", "test")] == [10, 4, 6]
    assert specs["train"].seeds == list(range(101, 111))
    assert specs["validation"].seeds == list(range(201, 205))
    assert specs["test"].seeds == list(range(301, 307))
    assert specs["train"].end == specs["validation"].start
    assert specs["validation"].end == specs["test"].start
    assert set(specs["train"].assets) == set(specs["validation"].assets) == {"FILLER", "CAPPER"}
    assert set(specs["test"].assets) == {"LABELLER", "FILLER"}


def test_dev_split_separation_time_and_asset(dev_ds, cfg):
    sp = dev_ds.splits
    tr, va, te = (set(sp[n].dates) for n in ("train", "validation", "test"))
    assert not (tr & va) and not (va & te) and not (tr & te)
    assert sp["train"].end <= sp["validation"].start and sp["validation"].end <= sp["test"].start
    for n in ("train", "validation"):
        assert "LABELLER" not in sp[n].assets
        assert all(k[1] != "LABELLER" for k in dev_ds.readings if k[0] == n)
    assert {k[1] for k in dev_ds.readings if k[0] == "test"} == {"LABELLER", "FILLER"}
    strata = cfg["split_table_dev"]["test_strata"]
    assert strata["FILLER"] == "seen_asset_later_time"
    assert strata["LABELLER"] == "unseen_asset_later_time"
    for (split, _), r in dev_ds.readings.items():
        assert r.t0 == sp[split].start and r.n == sp[split].end - sp[split].start


@pytest.mark.parametrize("mutate,exc", [
    (lambda t: t["splits"][0]["assets"].append("LABELLER"), g.SplitTableError),
    (lambda t: t["splits"][1]["assets"].__setitem__(0, "LABELLER"), g.SplitTableError),
    (lambda t: t["splits"][1]["seeds"].append(901), g.SplitTableError),
    (lambda t: t["splits"].reverse(), g.SplitTableError),
    (lambda t: t["splits"][2]["assets"].append("PALLETISER"), g.SplitTableError),
])
def test_bad_split_tables_are_refused(cfg, mutate, exc):
    t = copy.deepcopy(cfg["split_table_dev"])
    mutate(t)
    with pytest.raises(exc):
        g.build_split_specs(t, cfg)


def test_overlapping_split_ranges_refused():
    with pytest.raises(g.SplitTableError):
        g.check_splits_disjoint({"train": (0, 100), "validation": (90, 200)})


# ------------------------------------------------------------- fault rules
def _state_at(cfg, t, split_start):
    sod = (t - split_start) % 86400
    for seg in cfg["generator"]["regime_schedule"]:
        if seg["start"] <= sod < seg["end"]:
            return seg["state"]


def test_events_obey_placement_rules(dev_ds, cfg):
    rules = cfg["generator"]["fault_state_rules"]
    tail = cfg["generator"]["scoring_tail_minutes"] * 60
    ids = [e.event_id for e in dev_ds.labels.events]
    assert len(ids) == len(set(ids))
    for e in dev_ds.labels.events:
        sp = dev_ds.splits[e.split]
        assert e.tail_end == e.end + tail
        assert sp.warmup_end <= e.onset and e.tail_end <= sp.end
        allowed = rules[f"{e.channel}_allowed_states"]
        # every second of [onset, end + tail) is in an allowed state, same day
        for t in range(e.onset, e.tail_end, 30):
            assert _state_at(cfg, t, sp.start) in allowed, (e.event_id, t)
        assert (e.onset - sp.start) // 86400 == (e.tail_end - 1 - sp.start) // 86400
        assert "PLANNED_STOP" not in allowed
    for (split, asset) in dev_ds.readings:
        evs = sorted((e for e in dev_ds.labels.events if e.split == split and e.asset == asset),
                     key=lambda e: e.onset)
        for a, b in zip(evs, evs[1:]):
            assert a.tail_end <= b.onset, "same-asset events (with tails) must not overlap"


def test_exact_quotas(dev_ds, cfg):
    for name, sp in dev_ds.splits.items():
        n_days, n_assets = len(sp.seeds), len(sp.assets)
        for ftype, spec in cfg["generator"]["faults"].items():
            evs = [e for e in dev_ds.labels.events if e.split == name and e.fault_type == ftype]
            rule, val = spec["prevalence"]["rule"], spec["prevalence"]["value"]
            if rule == "per_asset_day_fraction":
                for a in sp.assets:
                    assert sum(e.asset == a for e in evs) == g.round_half_up(val * n_days)
            else:
                assert len(evs) == g.round_half_up(n_days * n_assets / val)
            assert all(e.channel == spec["channel"] for e in evs)


def test_prevalence_report(dev_ds):
    rep = g.prevalence_report(dev_ds)
    assert rep["quota_mode"] == "exact_quota"
    for name, s in rep["splits"].items():
        assert set(s["faults"]) == {"bearing_wear", "overheating", "pressure_spikes",
                                    "stuck_sensor", "dropout"}
        assert abs(s["missing_jitter_rate_realised"] - 0.005) < 0.001


def test_planted_data_faults_visible_in_raw_channel(dev_ds, cfg):
    for e in dev_ds.labels.events:
        r = dev_ds.readings[(e.split, e.asset)]
        a, b = e.onset - r.t0, e.end - r.t0
        if e.fault_type == "dropout":
            for s in cfg["sensors"]["order"]:
                assert np.isnan(r.values[s][a:b]).all()
        elif e.fault_type == "stuck_sensor":
            seg = r.values[e.sensor][a:b]
            seg = seg[~np.isnan(seg)]
            assert np.all(seg == e.params["held_value"])


def test_normal_change_is_test_only_and_labelled_normal(dev_ds):
    ch = dev_ds.labels.normal_changes
    assert ch and all(c["split"] == "test" and c["label"] == "NORMAL" for c in ch)
    assert {c["asset"] for c in ch} == {"LABELLER", "FILLER"}
    assert not any(e.fault_type == "product_changeover" for e in dev_ds.labels.events)


def test_every_split_has_all_normal_regimes(dev_ds):
    for name in dev_ds.splits:
        states = {s["state"] for s in dev_ds.labels.regime_segments if s["split"] == name}
        assert {"STARTUP", "STEADY_80", "SPEED_CHANGE", "STEADY_100", "PLANNED_STOP"} <= states


def test_raw_channel_carries_no_labels(dev_ds):
    fields = {f.name for f in dataclasses.fields(g.SensorReadings)}
    assert fields == {"t0", "values"}
    r = next(iter(dev_ds.readings.values()))
    assert set(r.values) == {"vibration_rms", "bearing_temp_c", "motor_current_a", "line_pressure_bar"}

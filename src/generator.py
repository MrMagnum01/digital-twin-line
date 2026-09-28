"""Deterministic synthetic sensor generator for the digital-twin line (LOCK).

Implements the frozen protocol plus the binding amendments
(vault:10-projects/freelance/launch/twin/frozen-ml-protocol.md). Every number
comes from config.yaml; nothing here is a tunable default.

Outputs are split into two channels that never share an object:

  * RAW channel   - SensorReadings: a 1 Hz grid of the four sensors per
                    (split, asset), NaN where a sample is missing. This is
                    the only thing features.py / ingest.py consume.
  * LABEL channel - LabelChannel: fault events (IDs, type, equipment/data
                    channel, onset/end), unseen-normal-change intervals,
                    regime schedule, planned stops, warm-up, and the jitter
                    truth. Consumed only by the evaluator.

Seeds: each split's date seeds map to consecutive UTC dates in table order.
Per-asset / per-stream seeds are derived with a documented SplitMix64 fold
(see config.yaml generator.seed_derivation) - never Python's hash().

LOCKED seeds (101-110 / 201-204 / 301-306) are refused unless the caller
passes allow_locked=True. Nothing in this milestone does; tests and the CLI
use only the development table (split_table_dev).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from twin_config import load_config

M64 = (1 << 64) - 1


class GeneratorError(RuntimeError):
    pass


class LockedSeedsError(GeneratorError):
    pass


class SplitTableError(GeneratorError):
    pass


# ---------------------------------------------------------------------------
# Stable seed derivation (documented in config.yaml and README)
# ---------------------------------------------------------------------------
def mix64(x: int) -> int:
    """SplitMix64 finaliser. Pure integer arithmetic: identical across
    processes, platforms and Python versions."""
    x = (int(x) + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def derive_seed(date_seed: int, asset_code: int, stream_code: int) -> int:
    return mix64(mix64(mix64(date_seed) ^ asset_code) ^ stream_code)


def split_seed(split_seeds: list[int], stream_code: int) -> int:
    acc = 0
    for s in split_seeds:
        acc = mix64(acc ^ int(s))
    return mix64(acc ^ stream_code)


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorReadings:
    """RAW channel only: timestamps are t0 + arange(n) (1 Hz grid, epoch
    seconds UTC); values[sensor] is float64 with NaN = missing sample.
    Deliberately carries no asset, split, seed or label information."""

    t0: int
    values: dict

    @property
    def n(self) -> int:
        return len(next(iter(self.values.values())))

    def timestamps(self) -> np.ndarray:
        return self.t0 + np.arange(self.n, dtype=np.int64)


@dataclass
class SplitSpec:
    name: str
    seeds: list
    dates: list          # ISO date strings, table order
    assets: list
    start: int           # epoch s, inclusive
    end: int             # epoch s, exclusive
    warmup_end: int


@dataclass
class FaultEvent:
    event_id: str
    split: str
    asset: str
    date: str
    fault_type: str
    code: str
    channel: str         # "equipment" | "data"
    sensor: str
    onset: int           # epoch s, inclusive
    end: int             # epoch s, exclusive (label end)
    tail_end: int        # end + scoring tail
    params: dict = field(default_factory=dict)


@dataclass
class LabelChannel:
    """LABEL channel. Never passed to features.py."""

    events: list
    normal_changes: list
    planned_stops: list      # {split, asset, start, end}
    regime_segments: list    # {split, asset, state, start, end}
    warmup: list             # {split, start, end}
    jitter_truth: dict       # "split/asset" -> {"jitter_missing": n, "samples": n}


@dataclass
class GeneratedDataset:
    table_name: str
    splits: dict             # name -> SplitSpec
    readings: dict           # (split, asset) -> SensorReadings
    labels: LabelChannel


# ---------------------------------------------------------------------------
# Split table
# ---------------------------------------------------------------------------
def _locked_seeds(cfg: dict) -> set[int]:
    return {s for sp in cfg["split_table_locked"]["splits"] for s in sp["seeds"]}


def build_split_specs(table: dict, cfg: dict | None = None) -> dict:
    """Map seeds to consecutive UTC dates in table order and validate the
    table structurally. Raises SplitTableError on overlap/duplication/asset
    leakage instead of producing overlapping splits."""
    cfg = cfg or load_config()
    gen = cfg["generator"]
    day_s = gen["day_seconds"]
    warm = gen["warmup_minutes"] * 60
    day1 = date.fromisoformat(table["day1_utc"])
    names = [sp["name"] for sp in table["splits"]]
    if names != ["train", "validation", "test"]:
        raise SplitTableError(f"splits must be train, validation, test in that order, got {names}")
    all_seeds = [s for sp in table["splits"] for s in sp["seeds"]]
    if len(set(all_seeds)) != len(all_seeds):
        raise SplitTableError("a seed appears more than once in the split table")
    known_assets = set(cfg["line"]["assets"])
    unseen = set(cfg["unseen_assets"])
    specs: dict = {}
    idx = 0
    for sp in table["splits"]:
        if not sp["seeds"]:
            raise SplitTableError(f"split {sp['name']} has no seeds")
        for a in sp["assets"]:
            if a not in known_assets:
                raise SplitTableError(f"unknown asset {a!r}")
        if len(set(sp["assets"])) != len(sp["assets"]):
            raise SplitTableError(f"duplicate asset in split {sp['name']}")
        if sp["name"] in ("train", "validation") and unseen & set(sp["assets"]):
            raise SplitTableError(f"unseen asset(s) {sorted(unseen & set(sp['assets']))} in {sp['name']}")
        dates = [(day1 + timedelta(days=idx + i)).isoformat() for i in range(len(sp["seeds"]))]
        start = _epoch(day1 + timedelta(days=idx))
        end = start + len(sp["seeds"]) * day_s
        specs[sp["name"]] = SplitSpec(sp["name"], list(sp["seeds"]), dates, list(sp["assets"]),
                                      start, end, start + warm)
        idx += len(sp["seeds"])
    test_assets = set(specs["test"].assets)
    if set(table["test_strata"]) != test_assets:
        raise SplitTableError("test_strata must name exactly the test assets")
    check_splits_disjoint({k: (v.start, v.end) for k, v in specs.items()})
    return specs


def check_splits_disjoint(ranges: dict) -> None:
    """Raise SplitTableError if any two [start, end) split ranges overlap."""
    items = sorted(ranges.items(), key=lambda kv: kv[1][0])
    for (n1, (s1, e1)), (n2, (s2, e2)) in zip(items, items[1:]):
        if s2 < e1:
            raise SplitTableError(f"splits {n1} and {n2} overlap in time")
    for n, (s, e) in ranges.items():
        if e <= s:
            raise SplitTableError(f"split {n} is empty or inverted")


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# Regime schedule
# ---------------------------------------------------------------------------
def _state_names(cfg: dict) -> list[str]:
    names: list[str] = []
    for seg in cfg["generator"]["regime_schedule"]:
        if seg["state"] not in names:
            names.append(seg["state"])
    return names


def _day_profile(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(speed fraction s, state code) for one day at 1 Hz."""
    gen = cfg["generator"]
    n = gen["day_seconds"]
    names = _state_names(cfg)
    s = np.full(n, np.nan)
    st = np.full(n, -1, dtype=np.int16)
    for seg in gen["regime_schedule"]:
        a, b = seg["start"], seg["end"]
        k = np.arange(b - a, dtype=np.float64)
        s[a:b] = seg["s_from"] + (seg["s_to"] - seg["s_from"]) * k / (b - a)
        st[a:b] = names.index(seg["state"])
    if np.isnan(s).any() or (st < 0).any():
        raise GeneratorError("regime_schedule does not tile the day")
    return s, st


def _allowed_blocks(cfg: dict, allowed: list[str]) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs (seconds of day) of allowed states."""
    blocks: list[list[int]] = []
    for seg in cfg["generator"]["regime_schedule"]:
        if seg["state"] in allowed:
            if blocks and blocks[-1][1] == seg["start"]:
                blocks[-1][1] = seg["end"]
            else:
                blocks.append([seg["start"], seg["end"]])
    return [(a, b) for a, b in blocks]


# ---------------------------------------------------------------------------
# Fault schedule
# ---------------------------------------------------------------------------
def _durations(spec: dict, rng: np.random.Generator) -> int:
    d = spec["duration_s"]
    if d["dist"] == "fixed":
        return int(d["value"])
    if d["dist"] == "uniform_int_inclusive":
        return int(rng.integers(d["min"], d["max"] + 1))
    raise GeneratorError(f"unknown duration dist {d['dist']}")


def plan_faults(split: SplitSpec, cfg: dict) -> list[FaultEvent]:
    gen = cfg["generator"]
    day_s = gen["day_seconds"]
    tail = gen["scoring_tail_minutes"] * 60
    streams = gen["seed_derivation"]["streams"]
    asset_codes = {a: cfg["line"]["assets"][a]["code"] for a in split.assets}
    assets_sorted = sorted(split.assets, key=lambda a: asset_codes[a])
    n_days = len(split.seeds)
    rng_q = _rng(split_seed(split.seeds, streams["fault_quota"]))
    rules = gen["fault_state_rules"]
    blocks = {
        "equipment": _allowed_blocks(cfg, rules["equipment_allowed_states"]),
        "data": _allowed_blocks(cfg, rules["data_allowed_states"]),
    }

    # 1) exact quotas: which asset-days get one event of each type
    chosen: dict[str, list[tuple[int, str]]] = {}
    for ftype, spec in gen["faults"].items():
        rule, val = spec["prevalence"]["rule"], spec["prevalence"]["value"]
        picks: list[tuple[int, str]] = []
        if rule == "per_asset_day_fraction":
            for a in assets_sorted:
                k = round_half_up(val * n_days)
                days = sorted(int(d) for d in rng_q.choice(n_days, size=k, replace=False))
                picks += [(d, a) for d in days]
        elif rule == "one_per_n_asset_days":
            pool = [(d, a) for d in range(n_days) for a in assets_sorted]
            k = round_half_up(len(pool) / val)
            idx = sorted(int(i) for i in rng_q.choice(len(pool), size=k, replace=False))
            picks = [pool[i] for i in idx]
        else:
            raise GeneratorError(f"unknown prevalence rule {rule}")
        chosen[ftype] = sorted(picks, key=lambda da: (da[0], asset_codes[da[1]]))

    # 2) durations, onsets and parameters, in fixed processing order
    events: list[FaultEvent] = []
    occupied: dict[str, list[tuple[int, int]]] = {a: [] for a in split.assets}
    sensors = cfg["sensors"]["order"]
    for ftype, spec in gen["faults"].items():
        for d, a in chosen[ftype]:
            date_seed = split.seeds[d]
            rng = _rng(derive_seed(date_seed, asset_codes[a], streams[f"fault_{ftype}"]))
            dur = _durations(spec, rng)
            span = dur + tail
            day0 = split.start + d * day_s
            cands = []
            for bs, be in blocks[spec["channel"]]:
                lo = max(day0 + bs, split.warmup_end)
                hi = day0 + be - span          # inclusive last onset
                if hi >= lo:
                    cands.append(np.arange(lo, hi + 1, dtype=np.int64))
            c = np.concatenate(cands) if cands else np.empty(0, dtype=np.int64)
            ok = np.ones(len(c), dtype=bool)
            for o, oe in occupied[a]:
                ok &= (c + span <= o) | (c >= oe)
            c = c[ok]
            if len(c) == 0:
                raise GeneratorError(f"no eligible onset for {ftype} on {split.dates[d]} {a}")
            onset = int(c[int(rng.integers(len(c)))])
            params: dict = {}
            sensor = spec["sensor"]
            if ftype == "pressure_spikes":
                sev = spec["severity"]
                n_sp = int(rng.integers(sev["n_min"], sev["n_max"] + 1))
                offs = sorted(int(x) for x in rng.choice(dur, size=n_sp, replace=False))
                params = {"n_spikes": n_sp, "spike_offsets_s": offs}
            elif ftype == "stuck_sensor":
                sensor = sensors[int(rng.integers(len(sensors)))]
            occupied[a].append((onset, onset + span))
            ymd = split.dates[d].replace("-", "")
            n_same = 1 + sum(1 for e in events if e.asset == a and e.date == split.dates[d]
                             and e.fault_type == ftype)
            eid = gen["event_id_format"].format(date=ymd, asset=a, code=spec["code"], n=n_same)
            events.append(FaultEvent(eid, split.name, a, split.dates[d], ftype, spec["code"],
                                     spec["channel"], sensor, onset, onset + dur,
                                     onset + dur + tail, params))
    ids = [e.event_id for e in events]
    if len(set(ids)) != len(ids):
        raise GeneratorError("event IDs are not unique")
    return sorted(events, key=lambda e: (e.onset, e.event_id))


# ---------------------------------------------------------------------------
# Signal synthesis
# ---------------------------------------------------------------------------
def _ar1(eps: np.ndarray, z0: float, phi: float) -> np.ndarray:
    b = [math.sqrt(1.0 - phi * phi)]
    return lfilter(b, [1.0, -phi], eps, zi=[phi * z0])[0]


def _synthesize(split: SplitSpec, asset: str, events: list[FaultEvent], cfg: dict):
    gen = cfg["generator"]
    day_s = gen["day_seconds"]
    sensors = cfg["sensors"]["order"]
    streams = gen["seed_derivation"]["streams"]
    code = cfg["line"]["assets"][asset]["code"]
    nom = gen["assets_nominal"][asset]
    sig = gen["assets_sigma"][asset]
    resp = gen["speed_response"]
    n_days = len(split.seeds)
    n = n_days * day_s

    s_day, st_day = _day_profile(cfg)
    s = np.tile(s_day, n_days)

    # unseen normal change (test only), label channel records it separately
    unc = gen["unseen_normal_change"]
    current_mult = np.ones(n)
    changes = []
    if split.name in unc["splits"]:
        for d in range(n_days):
            a0, b0 = d * day_s + unc["start_s"], d * day_s + unc["end_s"]
            current_mult[a0:b0] = 1.0 + unc["relative_shift"]
            changes.append({"change_id": f"NC-{split.dates[d].replace('-', '')}-{asset}",
                            "split": split.name, "asset": asset, "kind": unc["kind"],
                            "label": unc["label"], "sensor": unc["sensor"],
                            "relative_shift": unc["relative_shift"],
                            "start": split.start + a0, "end": split.start + b0})

    # noise innovations: per date, from that date's noise seed
    eps = np.empty((len(sensors), n))
    z0 = None
    for d, ds in enumerate(split.seeds):
        rng = _rng(derive_seed(ds, code, streams["noise"]))
        if d == 0:
            z0 = rng.standard_normal(len(sensors))
        eps[:, d * day_s:(d + 1) * day_s] = rng.standard_normal((len(sensors), day_s))
    vals: dict[str, np.ndarray] = {}
    for i, name in enumerate(sensors):
        y = _ar1(eps[i], float(z0[i]), gen["noise"]["ar1_phi"][name])
        r = resp[name]
        if name == "bearing_temp_c":
            amb = r["ambient_c"]
            target = amb + (nom[name] - amb) * s
            a = math.exp(-1.0 / r["lag_tau_s"])
            level = lfilter([1.0 - a], [1.0, -a], target, zi=[a * target[0]])[0]
            scale = np.full(n, sig[name])
        else:
            f = r["floor"] + (1.0 - r["floor"]) * s
            level = nom[name] * f
            scale = sig[name] * f if r["noise_scales_with_level"] else np.full(n, sig[name])
            if name == unc["sensor"]:
                level = level * current_mult
        vals[name] = level + scale * y

    # equipment faults (additive)
    for e in events:
        a0, b0 = e.onset - split.start, e.end - split.start
        dur = b0 - a0
        sev = gen["faults"][e.fault_type]["severity"]
        if e.fault_type == "bearing_wear":
            ramp = sev["from"] + (sev["to"] - sev["from"]) * np.arange(dur) / (dur - 1)
            vals[e.sensor][a0:b0] += sig[e.sensor] * ramp
        elif e.fault_type == "overheating":
            ramp = sev["from"] + (sev["to"] - sev["from"]) * np.arange(dur) / (dur - 1)
            vals[e.sensor][a0:b0] += ramp
        elif e.fault_type == "pressure_spikes":
            for off in e.params["spike_offsets_s"]:
                vals[e.sensor][a0 + off:a0 + off + sev["width_s"]] += sev["height_sigma"] * sig[e.sensor]
    for name in cfg["sensors"]["non_negative"]:
        np.maximum(vals[name], 0.0, out=vals[name])

    # data faults
    for e in events:
        a0, b0 = e.onset - split.start, e.end - split.start
        if e.fault_type == "stuck_sensor":
            held = float(vals[e.sensor][a0])
            e.params["held_value"] = held
            vals[e.sensor][a0:b0] = held
        elif e.fault_type == "dropout":
            for name in sensors:
                vals[name][a0:b0] = np.nan
    dropped = np.zeros(n, dtype=bool)
    for e in events:
        if e.fault_type == "dropout":
            dropped[e.onset - split.start:e.end - split.start] = True

    # normal missingness jitter, drawn for every grid sample
    rate = gen["missing_jitter"]["rate"]
    jit_total = 0
    jit_effective = 0
    for d, ds in enumerate(split.seeds):
        rng = _rng(derive_seed(ds, code, streams["jitter"]))
        mask = rng.random((len(sensors), day_s)) < rate
        sl = slice(d * day_s, (d + 1) * day_s)
        jit_total += int(mask.sum())
        jit_effective += int((mask & ~dropped[sl]).sum())
        for i, name in enumerate(sensors):
            vals[name][sl][mask[i]] = np.nan

    st = np.tile(st_day, n_days)
    return (SensorReadings(split.start, vals), changes, st,
            {"jitter_missing_drawn": jit_total, "jitter_missing_outside_dropout": jit_effective,
             "sensor_samples": len(sensors) * n})


def _segments(split: SplitSpec, asset: str, cfg: dict) -> list[dict]:
    out = []
    for d in range(len(split.seeds)):
        base = split.start + d * cfg["generator"]["day_seconds"]
        for seg in cfg["generator"]["regime_schedule"]:
            out.append({"split": split.name, "asset": asset, "state": seg["state"],
                        "start": base + seg["start"], "end": base + seg["end"]})
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate(table_name: str = "dev", *, allow_locked: bool = False,
             cfg: dict | None = None, table: dict | None = None) -> GeneratedDataset:
    """Generate a full split table. table_name 'dev' uses split_table_dev.
    The locked table (or any table containing a locked seed) is refused
    unless allow_locked=True - which nothing in this LOCK milestone passes."""
    cfg = cfg or load_config()
    if table is None:
        key = {"dev": "split_table_dev", "locked": "split_table_locked"}.get(table_name)
        if key is None:
            raise GeneratorError(f"unknown table {table_name!r}")
        table = cfg[key]
    seeds = {s for sp in table["splits"] for s in sp["seeds"]}
    if (seeds & _locked_seeds(cfg)) and not allow_locked:
        raise LockedSeedsError(
            "refusing to generate on locked seeds: experiment freeze sign-off is pending "
            "(this milestone is a lock only; use the development table)")
    specs = build_split_specs(table, cfg)
    readings: dict = {}
    events: list = []
    changes: list = []
    stops: list = []
    segments: list = []
    jitter: dict = {}
    non_op = set(cfg["generator"]["non_operating_states"])
    for name, sp in specs.items():
        split_events = plan_faults(sp, cfg)
        for asset in sp.assets:
            ev = [e for e in split_events if e.asset == asset]
            r, ch, _st, jt = _synthesize(sp, asset, ev, cfg)
            readings[(name, asset)] = r
            changes += ch
            jitter[f"{name}/{asset}"] = jt
            segs = _segments(sp, asset, cfg)
            segments += segs
            stops += [{k: g[k] for k in ("split", "asset", "start", "end")}
                      for g in segs if g["state"] in non_op]
        events += split_events
    warm = [{"split": n, "start": sp.start, "end": sp.warmup_end} for n, sp in specs.items()]
    labels = LabelChannel(sorted(events, key=lambda e: (e.split, e.asset, e.onset, e.event_id)),
                          changes, stops, segments, warm, jitter)
    return GeneratedDataset(table_name, specs, readings, labels)


def prevalence_report(ds: GeneratedDataset, cfg: dict | None = None) -> dict:
    """Realised prevalence per split and fault type, next to the declared
    quota rule, plus realised missingness jitter."""
    cfg = cfg or load_config()
    out: dict = {"table": ds.table_name, "quota_mode": "exact_quota", "splits": {}}
    for name, sp in ds.splits.items():
        n_days = len(sp.seeds)
        n_ad = n_days * len(sp.assets)
        split_ev = [e for e in ds.labels.events if e.split == name]
        per_type = {}
        for ftype, spec in cfg["generator"]["faults"].items():
            evs = [e for e in split_ev if e.fault_type == ftype]
            asset_days = {(e.asset, e.date) for e in evs}
            per_type[ftype] = {
                "channel": spec["channel"],
                "rule": spec["prevalence"]["rule"],
                "rule_value": spec["prevalence"]["value"],
                "events": len(evs),
                "events_per_asset_day": len(evs) / n_ad,
                "fraction_of_asset_days_with_event": len(asset_days) / n_ad,
                "by_asset": {a: sum(1 for e in evs if e.asset == a) for a in sp.assets},
            }
        jit = {k: v for k, v in ds.labels.jitter_truth.items() if k.startswith(name + "/")}
        drawn = sum(v["jitter_missing_drawn"] for v in jit.values())
        total = sum(v["sensor_samples"] for v in jit.values())
        fault_s = {ch: sum(e.end - e.onset for e in split_ev if e.channel == ch)
                   for ch in ("equipment", "data")}
        out["splits"][name] = {
            "days": n_days, "assets": sp.assets, "asset_days": n_ad,
            "faults": per_type,
            "missing_jitter_rate_declared": cfg["generator"]["missing_jitter"]["rate"],
            "missing_jitter_rate_realised": drawn / total if total else None,
            "fault_seconds_fraction": {ch: v / (n_ad * cfg["generator"]["day_seconds"])
                                       for ch, v in fault_s.items()},
        }
    return out


def labels_to_json(labels: LabelChannel) -> dict:
    return {
        "events": [asdict(e) for e in labels.events],
        "normal_changes": labels.normal_changes,
        "planned_stops": labels.planned_stops,
        "regime_segments": labels.regime_segments,
        "warmup": labels.warmup,
        "jitter_truth": labels.jitter_truth,
    }


def dataset_hash(ds: GeneratedDataset) -> str:
    """sha256 over the raw channel (exact float64 bytes, sorted keys) and the
    canonical JSON of the label channel."""
    h = hashlib.sha256()
    for key in sorted(ds.readings):
        r = ds.readings[key]
        h.update(f"{key[0]}/{key[1]}/{r.t0}/{r.n}".encode())
        for name in sorted(r.values):
            h.update(name.encode())
            h.update(np.ascontiguousarray(r.values[name], dtype="<f8").tobytes())
    h.update(json.dumps(labels_to_json(ds.labels), sort_keys=True).encode())
    return h.hexdigest()


def _iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()


def to_messages(readings: SensorReadings, tag: str, start: int | None = None,
                stop: int | None = None, cfg: dict | None = None) -> list[str]:
    """Render the raw channel as SENSOR messages for src/ingest.py: one JSON
    message per grid second where at least one sensor is present (a whole-
    asset dropout second emits nothing - the comms link is down), null for
    an individually missing sensor."""
    cfg = cfg or load_config()
    sensors = cfg["sensors"]["order"]
    lo = 0 if start is None else start - readings.t0
    hi = readings.n if stop is None else stop - readings.t0
    out = []
    for i in range(lo, hi):
        row = {name: readings.values[name][i] for name in sensors}
        if all(math.isnan(v) for v in row.values()):
            continue
        msg = {"ts": _iso(readings.t0 + i), "tag": tag, "type": "SENSOR"}
        msg.update({k: (None if math.isnan(v) else float(v)) for k, v in row.items()})
        out.append(json.dumps(msg))
    return out


def write_outputs(ds: GeneratedDataset, out_dir: Path) -> None:
    """Raw channel -> readings/*.npz, label channel -> labels.json (separate
    files). Development use only; out/ is git-ignored."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readings").mkdir(exist_ok=True)
    for (split, asset), r in ds.readings.items():
        np.savez(out_dir / "readings" / f"{split}_{asset}.npz", t0=np.int64(r.t0),
                 **{k: v for k, v in r.values.items()})
    (out_dir / "labels.json").write_text(json.dumps(labels_to_json(ds.labels), indent=1, sort_keys=True))
    (out_dir / "prevalence.json").write_text(json.dumps(prevalence_report(ds), indent=1, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the DEVELOPMENT split table (dev seeds only).")
    p.add_argument("--out", default="out/dev")
    p.add_argument("--hash-only", action="store_true", help="print the dataset hash and exit")
    args = p.parse_args()
    ds = generate("dev")
    if args.hash_only:
        print(dataset_hash(ds))
        return
    write_outputs(ds, Path(args.out))
    print(json.dumps({"table": "dev", "dataset_sha256": dataset_hash(ds)}))


if __name__ == "__main__":
    main()

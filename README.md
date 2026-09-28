# Digital-twin line: predictive-maintenance experiment lock (synthetic demo)

**Role:** Synthetic portfolio demonstration, implemented with AI coding agents. No client data or client work.

**Independent review:** the company's separate AI reviewer cleared the experiment freeze (lock-3), the runner at commit c654237, and the integrity of the one retained locked evaluation. That scope is bounded: synthetic data only, and no performance or real-world claim. Later commits are not covered by that review.

## Result of the one locked evaluation (measured, synthetic, 9 events per stratum)

> Under a frozen synthetic evaluation, Isolation Forest did not improve on the always-alert baseline. The static rule detected more events on the seen asset, but at substantially more false alerts. On the unseen asset, the methods had the same event-level score, with only dropout events matched. Isolation Forest was active for about 99.9% of eligible operating time. These nine-event-per-stratum results demonstrate the evaluation workflow, not a useful anomaly detector or real-world performance.

The run is one-shot and reserved, and will not be rerun or tuned. The retained artifacts (pre_run, selection and evaluation) and their hashes are held in the company record.

> **History.** The experiment was frozen (lock-3) and signed off before any evaluation existed. All development used separate **development** seeds (901-906) and hand-built fixtures. The locked seeds (101-110 / 201-204 / 301-306) were used exactly once, by the reviewed runner, for the evaluation above.

This repo freezes the machinery for a synthetic predictive-maintenance
experiment on a fictional packaging line, before any evaluation trace exists.
It follows `vault:10-projects/freelance/launch/twin/frozen-ml-protocol.md`,
including its binding amendments. The deliverables are:

- a deterministic sensor generator,
- a validated ingest,
- a causal feature pipeline,
- a pure scoring evaluator,
- one frozen config,
- `experiment-lock.json`, which holds the protocol hash, the source hashes and a complete dependency lock with hashes.

Any measured number that comes out of this machinery later will hold **on this synthetic set only; no industrial reliability claim**.

## What's in the box

| Path | What it is |
|---|---|
| `config.yaml` | The **single frozen source of truth** for every number: seeds, the seed-derivation formula, regimes, fault shapes and quotas, jitter, warm-up, tail, features, scaler, scoring definitions and model parameters. The modules read their numbers from it; none hardcodes a duplicate. |
| `src/generator.py` | Deterministic 1 Hz generator. It has two separate output channels: RAW sensor readings and LABELS. |
| `src/ingest.py`, `src/schema.sql` | Validated SENSOR-message intake into DuckDB. It reuses two functions from the pinned plant demo (see below). |
| `src/features.py` | Causal 10 s-window features, the data-quality outcomes, the leak guard and the train-only RobustScaler. |
| `src/evaluator.py` | Event-level and point-level scoring, the baselines as inputs, and the selection rule. It contains no model code. |
| `src/twin_config.py` | Config loader. |
| `scripts/make_lock.py` | Writes `experiment-lock.json` from the committed bytes. It reads files only. |
| `vendor/plant-shift-oee-report` | Git **submodule**, pinned at `7e378facb410b529d8e2d72d329daf2eee515161`. Read-only. |
| `requirements.txt` / `requirements.lock` | Direct pins, and the complete transitive lock with sha256 hashes. |
| `runner/` | The model-fit + threshold-selection runner (this milestone). Refuses to run at all unless the approved `experiment-lock.json` and every file it hashes still verify. See "Model runner" below. |
| `dashboard/` | A Streamlit live view, reading only from a DuckDB file. See "Dashboard" below. |
| `monitoring/` | An operational rule + model alert log, separate from the frozen protocol's evaluation. See "Monitoring" below. |

## Model runner, dashboard and monitoring (this milestone)

> `runner/` passed the independent runner review at c654237. It then ran the one
> locked evaluation (result above). The reservation is consumed, so a second locked
> run is refused. `generator.generate("locked", ...)` still refuses unless a caller
> explicitly passes `allow_locked=True`; only the reviewed CLI path does that. `dashboard/` and
> `monitoring/` are demo/operational code, not the frozen evaluation, and
> make no locked-evaluation or performance claim anywhere (see "What this
> demo does not claim").

### `runner/` - model-fit + threshold-selection

- `runner/lock_guard.py` - the refusal gate. `require_lock()` checks that
  `experiment-lock.json`'s own sha256 equals the sha Astra signed off for
  lock-3 (`vault:10-projects/freelance/launch/twin/astra-lock-3-signoff.md`),
  that every file hash it records (`src/generator.py`, `src/ingest.py`,
  `src/features.py`, `src/evaluator.py`, `config.yaml`, plus
  `src/twin_config.py`, `src/schema.sql`, `requirements.txt`,
  `requirements.lock`, `pytest.ini`) still matches the bytes on disk, and
  that the pinned plant-ingest submodule's actual git HEAD agrees. Any
  mismatch is a hard refusal, not a warning.
- `runner/models.py` - candidate builders only (fits nothing on validation
  or test, chooses no threshold): the static per-sensor threshold (train
  mean/std, k grid) and IsolationForest (config params taken as-is,
  contamination never tuned; train-score quantile candidate thresholds).
- `runner/pipeline.py` - `run(table_name="dev", allow_locked=False, ...)`
  orchestrates the frozen procedure once: `lock_guard.require_lock()` ->
  generate -> features -> `fit_scaler` on TRAIN only -> build+score both
  models' candidates on VALIDATION only, pooled, and select a threshold per
  model with `evaluator.select_candidate` (max event F1 within the
  false-alert cap; ties broken as frozen) -> score TEST exactly once per
  stratum (FILLER seen-asset, LABELLER unseen-asset - never pooled,
  matching `evaluator.pool_counts`'s refusal to pool any test stratum).
- `runner/cli.py` - `python3 -m runner.cli --table dev` (default; writes a
  JSON report). `--table locked` additionally requires
  `--i-have-clearance-to-run-locked-seeds`, and even then `lock_guard` and
  `generator`'s own refusal still apply underneath - nothing in this
  repo's tests, dashboard or monitoring code ever passes that flag.
- Tests (`tests/test_runner_*.py`) exercise `lock_guard` against the real
  repo plus hand-tampered `tmp_path` copies, `models.py` against hand-built
  fixtures and tiny synthetic matrices, and `pipeline.py`/`cli.py` against
  the DEVELOPMENT table only - asserting the refusal gates fire before
  anything is generated, that test strata are scored exactly once each and
  never pooled, and that the reported dataset hash matches an independent
  `generator.generate("dev")` call.

### `dashboard/` - Streamlit live view

`dashboard/seed_demo_db.py` is the only thing that writes to the demo
DuckDB file: it generates the DEVELOPMENT seed table (901-906, never
locked), ingests a bounded recent window (default 3h) per machine through
this repo's own `src/ingest.py` unmodified, ingests the **cleared plant
demo's own** deterministic synthetic day through *its own*
`ingester.Ingester` (vendor/plant-shift-oee-report, unmodified - called
directly with each event's JSON payload, no MQTT broker needed for a
one-shot seed), and runs one pass of `monitoring`'s rule + model alert log
over the freshly ingested sensor history. `dashboard/app.py`
(`streamlit run dashboard/app.py`, or set `TWIN_DB_PATH`) only ever
**reads** that file: per-machine sensor values and an ingest reconciliation
summary, the alert log, OEE computed by the plant demo's own unmodified
`report.build_report_data()` against its own STATE/COUNT/ALARM tables (not
the twin's sensor data - see its own `OEE_METHOD_NOTE`), and an inline SVG
line schematic (FILLER -> CAPPER -> LABELLER) coloured by each machine's
latest plant-demo STATE, with the state name always rendered as text (never
colour alone). Colours follow the project's validated categorical/status
palette (`dashboard/palette.py`); all interpolated strings in the SVG are
HTML-escaped.

```bash
python3 dashboard/seed_demo_db.py --db out/twin_demo.duckdb --force
streamlit run dashboard/app.py   # TWIN_DB_PATH=out/twin_demo.duckdb by default
```

Additional dependencies (`streamlit`, `altair`, `pandas`, and their
transitive packages - all OSI-licensed, see `LICENSES.md` "Dashboard
dependencies") live in `dashboard/requirements.txt`, separate from the
locked `requirements.lock`: `pip install -r dashboard/requirements.txt`.

Tested with `streamlit.testing.v1.AppTest` (`tests/test_dashboard_app.py`),
which actually executes `app.py`'s script and widget tree headlessly - no
browser is available in this build environment. A manual `streamlit run` +
HTTP smoke check (200 OK) was also run once by hand; it is not part of the
automated suite.

### `monitoring/` - rule + model alert log

Deliberately **not** the frozen protocol: `monitoring/rules.py` is a causal
expanding per-sensor z-score computed from a machine's own strictly-prior
history (never a future sample; a future-perturbation test covers this, the
same pattern `tests/test_features.py` already uses for the locked feature
pipeline). `monitoring/model_monitor.py` fits an IsolationForest once on a
fixed causal warm-up slice of a machine's own history (config's frozen
params, contamination never tuned), then scores forward against that fixed
baseline. `monitoring/alert_log.py` writes both streams to
`monitoring_alert_log` (`monitoring/schema.sql`, a new table - `src/schema.sql`
itself is never touched) in the same DuckDB file, idempotently (a
`UNIQUE` constraint plus `ON CONFLICT DO NOTHING`; note `sensor` is
`NOT NULL` - a `NULL` in a `UNIQUE` column is never equal to another `NULL`,
which would otherwise silently defeat the constraint on every rerun).

## Reuse of the cleared plant ingest (pinned reference, not a fork)

The protocol says: "Ingest reuses the CLEARED plant demo at 7e378fa ... new
sensor schemas and integration require regression tests. Integrity guarantees
are not inherited merely by importing or naming that demo." It is implemented
as follows:

- **Pinned reference.** `plant-shift-oee-report` is a git submodule at
  `vendor/plant-shift-oee-report`, pinned to commit
  `7e378facb410b529d8e2d72d329daf2eee515161`.
  - No file of it is copied into this repo, and none is edited.
  - The same SHA is recorded in `config.yaml` and `experiment-lock.json`.
  - `tests/test_ingest.py` asserts that the submodule HEAD equals that SHA and that its working tree is clean.
- **Imported unmodified.** `src/ingest.py` adds the submodule's `src/` to
  `sys.path` and runs `from ingester import _parse_ts, _content_hash`.
  - The import runs with `sys.dont_write_bytecode`, so no `__pycache__` is written into the submodule.
  - Both functions are schema-agnostic:
    - `_parse_ts`: strict parsing of timezone-aware ISO8601 timestamps.
    - `_content_hash`: sha256 of the canonical JSON of a message, used to tell a duplicate from a conflict.
  - A test asserts that both functions are defined in the submodule's `ingester.py` and are not re-implemented locally.
- **New code for the new schema.** Everything else in `src/ingest.py` is new:
  - the SENSOR grammar,
  - `schema.sql` (the plant's STATE/COUNT/ALARM/HEARTBEAT tables are not reused),
  - the categories,
  - known-total reconciliation,
  - the transaction and rollback handling,
  - restart-state recovery.

  The plant ingester's validation logic is tied to its own message types, so it was not reused. The new code follows the same contract (common-rules.md) and is proven by its own tests, not inherited.

### Ingest contract (`src/ingest.py`)

A message is one JSON object with **exactly** these keys:
`ts` (tz-aware ISO8601, whole second), `tag` (`LINE_A.FILLER|CAPPER|LABELLER`),
`type` (`"SENSOR"`), and the four sensors. Each sensor value is a finite
number inside `config.yaml sensors.ingest_range`, or `null`. At least one
sensor must be non-null.

Each handled message gets exactly one `sensor_ingest_log` row, with its raw
payload kept verbatim:

| Category | Meaning | Fact row? |
|---|---|---|
| `accepted` | Valid, and newer than the tag's newest accepted `ts` | yes |
| `duplicate` | Same `(tag, ts)` already seen, identical content hash | no |
| `reading_conflict` | Same `(tag, ts)` already seen, different content; the original row is untouched | no |
| `out_of_order_rejected` | Valid new `(tag, ts)`, but at or before the tag's watermark. There is no late window: the record is kept, quarantined, and never merged silently | no |
| `unknown_tag` | Not a configured machine | no |
| `bad_payload` | Any of: invalid JSON or UTF-8, a duplicate JSON key, a `NaN`/`Infinity` literal, a missing or extra key, a wrong type, out of range, off the 1 Hz grid, or all sensors null | no |

- **Sources.** `ingest_jsonl()` raises `IngestSourceError` with category `missing_source` or `zero_byte_source`. It never returns an empty success. Every line, including a blank one, is one accounted message.
- **Reconciliation.** `reconciliation()` checks that the category counts sum to the number of messages handled, and that fact rows equal accepted rows.
- **Durability.**
  - The log row and its fact row are committed in one DuckDB transaction, in batches of 500. A message is never split across a commit.
  - On a write failure, the transaction is rolled back and all in-memory state is rebuilt from the database.
  - A restarted ingester restores its state from the database, so a replay is classified as `duplicate`.
  - Tests cover injected write failure, restart replay, and an `os._exit` crash in the middle of a batch.

## Quickstart

```bash
git clone --recurse-submodules <this-repo-url> digital-twin-line
cd digital-twin-line
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes --only-binary=:all: -r requirements.lock
python3 -m pytest -v
# optional: DEVELOPMENT-seed data only (out/ is git-ignored)
python3 src/generator.py --out out/dev
```

The lock covers CPython 3.13 on manylinux x86_64 (wheel hashes).
`src/generator.py` has no command-line path to the locked seeds: `generate("locked")` raises
`LockedSeedsError` unless the caller passes `allow_locked=True`, and no code in this
repo does that.

## Frozen choices (full detail in `config.yaml`)

### Seeds, dates, splits

- **Locked table.**
  - train seeds 101-110 map to days 1-10;
  - validation seeds 201-204 map to days 11-14;
  - test seeds 301-306 map to days 15-20.
  - Seeds map to 20 distinct consecutive UTC dates in table order, with day 1 = 2024-03-04.
- **Assets.**
  - train and validation: FILLER and CAPPER.
  - test: LABELLER (the **unseen asset / later time** stratum) and FILLER (the **seen asset / later time** stratum). The two strata are always reported separately, and `evaluator.pool_counts` refuses to pool test strata.
  - CAPPER is not in test, exactly as in the protocol table.
- **Seed derivation.** It never uses Python `hash()`, and a test checks this with an AST scan.
  ```
  M = 2^64 - 1
  mix64(x): x = (x + 0x9E3779B97F4A7C15) & M
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
            return x ^ (x >> 31)                       # SplitMix64 finaliser
  derive_seed(date_seed, asset_code, stream) = mix64(mix64(mix64(date_seed) ^ asset_code) ^ stream)
  split_seed(seeds, stream): acc=0; for s in seeds: acc = mix64(acc ^ s); return mix64(acc ^ stream)
  ```
  - Asset codes: FILLER=1, CAPPER=2, LABELLER=3.
  - Streams: noise=1, jitter=2, fault_quota=3, fault_<type>=11..15.
  - RNG: `numpy.random.Generator(PCG64(seed))`, with numpy pinned.
- **Development seeds** (development only; never reported as train, validation or test results):
  - dev table: train 901-903, validation 904, test 905-906;
  - dates 2023-09-04 to 2023-09-09;
  - same assets and structure as the locked table.

### Signals and regimes

- **Sensors:** `vibration_rms` (mm/s), `bearing_temp_c` (°C), `motor_current_a` (A) and `line_pressure_bar` (bar), at 1 Hz on an exact integer-second grid. There is no timestamp jitter.
- **Per-asset values.** Nominal means and sigmas per asset are in `config.yaml`.
- **Noise.** A stationary Gaussian AR(1) per sensor, with phi = 0.6 / 0.98 / 0.8 / 0.5. It runs continuously across the split, and each date's innovations come from that date's seed.
- **Speed response.** The level follows the speed fraction `s`. Temperature follows a first-order lag with tau = 900 s.
- **Daily regime (UTC), identical every day and in every split:**

  | Time | Regime |
  |---|---|
  | 00:00-00:20 | startup 0 to 80% |
  | 00:20-08:00 | steady 80% |
  | 08:00-08:10 | speed change 80% to 100% |
  | 08:10-14:00 | steady 100% |
  | 14:00-14:10 | stopping |
  | 14:10-15:10 | **planned stop** |
  | 15:10-15:30 | restart |
  | 15:30-23:50 | steady 100% |
  | 23:50-24:00 | shutdown |

  Operating time is everything except PLANNED_STOP.
- **Unseen normal change (test split only).** A product changeover raises nominal motor current by +8% from 18:00 to 24:00 on every test asset-day. It is labelled `NORMAL` in the label channel and is never a fault.
- **Missingness jitter.** Each (sample, sensor) is dropped independently with probability 0.005. The draw covers every grid sample, so the pattern never depends on the faults. Its realised rate is reported.

### Faults: exact quotas, not probabilities

Quota rules, with `round_half_up(x) = floor(x + 0.5)`, sampled without
replacement from a split-level seed:

- **"1 per asset-day in 40% of days"** means that for each asset, exactly `round_half_up(0.4 × days)` of its days have one bearing-wear event.
- **"1 per N asset-days"** means that exactly `round_half_up(asset_days / N)` distinct asset-days have one event of that type.

| Type | Channel | Shape | Duration |
|---|---|---|---|
| bearing_wear | equipment | vibration + sigma × linear ramp 0.5 to 3.0 | 7200 s |
| overheating | equipment | temperature + linear ramp 1 to 6 °C | 1800 s |
| pressure_spikes | equipment | 3-5 (uniform) one-sample spikes of +4 sigma at uniformly chosen distinct seconds | 600 s |
| stuck_sensor | data | one uniformly chosen sensor holds its value at onset | uniform 1200-3600 s |
| dropout | data | all four sensors missing (whole-asset comms loss) | uniform 120-900 s |

- **Labels.** The label interval is `[onset, onset + duration)`. The effect ends at the label end.
- **Event IDs** are stable and unique: `EV-<YYYYMMDD>-<ASSET>-<CODE>-<nn>`.
- **Onset rule.** The onset is uniform over the eligible seconds. An onset is eligible only if all of the following hold:
  - the whole event plus the 10-minute scoring tail lies inside one block of allowed states on one day;
  - it is at or after the split's 30-minute warm-up;
  - it overlaps no other event (with its tail) on the same asset.
- **Allowed states.**
  - Equipment faults: steady and speed-change blocks only.
  - Data faults: any state except PLANNED_STOP. A planned stop is never a failure.
- **Training data** keeps its faults. It is not cleaned using label knowledge.
- **Prevalence.** `generator.prevalence_report()` reports the realised prevalence per split and type next to the declared rule.

### Features (`src/features.py`)

- **Windows.**
  - Output window `k` covers `[a_k, a_k + 10)` and is emitted at `e_k = a_k + 10`.
  - It uses only raw 1 Hz samples with `split_start <= t < e_k`.
  - The rolling windows are 5 min and 30 min, covering `[e - W, e)` and clipped at the split start.
- **Summaries.** They are computed from raw 1 Hz values, not 10 s aggregates:
  - mean;
  - population std (ddof = 0);
  - OLS slope of value on time, per second.
- **Minimum observations.** A summary needs at least 150 finite samples (5 min) or 900 (30 min). Below that it is `NaN` and the window is ineligible (`insufficient_observations`).
- **Missing fraction.** `missing_frac` over 10 s, 5 min and 30 min = 1 - finite samples / **declared grid seconds**. A complete dropout therefore still appears in the denominator. `NaN` and ±inf both count as missing. Any inf also makes the window `non_finite_input`.
- **Columns.** There are 36 feature columns: 4 sensors × (3 statistics × 2 windows + 3 missing fractions).
- **Leak guard.** `assert_no_leak()` enforces an exact allow-list plus a deny-list of label, event, asset, seed, split, schedule and state tokens. It is called on every output.
- **Split boundaries.** Each (split, asset) is computed from empty state. Readings that do not exactly span the split are refused. The first 30 minutes of every split are `warm_up`. The excluded windows and their durations are reported by reason.
- **Imputation.** None. Ineligible windows keep NaN, carry a reason, and are never counted as healthy.
- **Scaler.** `sklearn.preprocessing.RobustScaler(quantile_range=(25, 75))`, with scikit-learn pinned.
  - It is fitted on **train** eligible windows only. It refuses any other split.
  - Zero-variance rule: a feature whose train IQR is ≤ 1e-12 is **dropped**, and the dropped list is recorded.
  - There is no per-test-asset normalisation.

### Scoring (`src/evaluator.py`)

- **Debounce** runs per (split, asset).
  - The alert time is the window end.
  - A new episode starts when the gap (next window start minus previous window end) is greater than 60 s.
  - The episode start is the end of its first window.
- **Matching** is one-to-one per asset.
  - Events are processed in (onset, event_id) order. Each takes the earliest *unmatched* episode that starts in `[onset, end + 600]`.
  - Unmatched events are FN. Unmatched episodes are FP, including repeats inside a matched event.
  - There is no retroactive credit.
  - The full match table is returned, including each event's overlapping-event IDs.
- **Metrics.**
  - F1 = 2TP / (2TP + FP + FN).
  - Any zero denominator is reported as **N/A (None)**, never 0.
  - Precision is generic alert precision. Recall is broken down by true fault type and by channel.
- **Point level.**
  - A window is positive if it intersects `[onset, end)`. The equipment and data channels are scored separately.
  - Predictions are the raw windows before debouncing. There is no point adjustment.
  - Confusion counts are computed over eligible windows. Excluded windows and their truth are reported by reason.
- **False alerts.**
  - False alerts per operating hour = unmatched episodes that start in operating time / eligible operating hours.
  - Planned-stop false alarms are reported separately, per observed hour.
  - Unseen-normal-change false alerts, their duration and their hours are reported separately.
  - The alert-active fraction of operating time is reported.
- **Delay.** Measured for matched events only, reported with the event count and the miss rate. Median and p90 use `numpy.quantile(method="linear")`.
- **Baselines.** `no_alert` and `always_alert` are alert vectors fed to the same evaluator. `always_alert` covers every eligible window, because the constant detector has no operating-state input.
- **Selection rule.**
  - Maximise event F1 subject to at most 1 false alert per operating hour.
  - Break ties by fewer false alerts, then fewer total alerts, then the stricter threshold.
  - No-alert is always a candidate, and every candidate is retained.
  - In this milestone the rule is implemented and tested only on hand-built confusion counts.
- **Model parameters (config only; nothing in this repo fits them).**
  - k grid: {1, 2, 3, 4, 5, 6}.
  - IsolationForest: `n_estimators=200, max_samples=256, max_features=1.0, bootstrap=False, contamination=0.01` (fixed, not tuned), `random_state=7`.
  - Anomaly score: `-score_samples`.
  - Candidate thresholds: train-score quantiles {0.90, 0.95, 0.975, 0.99, 0.995, 0.999} (numpy `linear`), plus no-alert.

## Tests (`python3 -m pytest -v`)

- `tests/test_generator.py` covers:
  - determinism, both in-process and across two subprocesses with different `PYTHONHASHSEED`;
  - the SplitMix64 known-answer value and a check that builtin `hash()` is not used;
  - refusal of the locked seeds;
  - the locked table's date mapping (structure only; nothing is generated);
  - split separation in time and asset;
  - refusal of bad or overlapping split tables;
  - the fault placement rules and exact quotas;
  - the prevalence report;
  - raw and label channel separation.
- `tests/test_features.py` covers:
  - the allow-list and deny-list;
  - label-channel mutation leaving every feature unchanged;
  - future-sample perturbation leaving earlier features unchanged;
  - agreement with a brute-force implementation of the definition;
  - reset at split boundaries and warm-up exclusion;
  - a complete dropout staying in the denominator;
  - labelling of non-finite input;
  - train-only scaling and the zero-variance drop.
- `tests/test_evaluator.py` covers every amendment fixture:
  - one alert spanning two faults gives at most one TP;
  - repeated alerts give FP;
  - always-alert cannot reach perfect recall by reusing an episode;
  - no-event and no-alert strata give N/A;
  - overlapping splits are refused;
  - dropout stays in coverage accounting.

  It also covers debounce, tail inclusivity, pre-onset refusal, matching order, delay quantiles, point level without adjustment, planned-stop and normal-change accounting, refusal to pool test strata, and the selection rule on fixtures. A dev-seed wiring test checks structure only.
- `tests/test_ingest.py` covers:
  - the pinned-reuse checks;
  - every category and grammar violation;
  - known-total reconciliation;
  - missing and zero-byte sources;
  - write-failure rollback, restart replay, and a crash in the middle of a batch;
  - a dev-seed round trip from the generator through ingest back to the declared grid.
- `tests/test_lock.py` checks the submodule pin and that `experiment-lock.json` matches the files.
- `tests/test_runner_lock_guard.py` checks the refusal gate against the real repo and hand-tampered `tmp_path` copies (missing lock, disagreeing approved sha, a tampered or missing hashed file, submodule drift, and that every problem is collected, not just the first).
- `tests/test_runner_models.py` checks the static-threshold and IsolationForest candidate builders on hand-built fixtures and tiny synthetic matrices (config params taken as-is; never alerting on an ineligible or non-finite window).
- `tests/test_runner_pipeline.py` and `tests/test_runner_cli.py` check the full run on the DEVELOPMENT table only: the lock is verified before anything is generated, the locked table is refused without `allow_locked=True`, every validation candidate is retained, test is scored exactly once per stratum and never pooled, and the reported dataset hash matches an independent `generator.generate("dev")` call.
- `tests/test_monitoring_rules.py` checks the causal expanding z-score rule: no alert before `min_history`, a clear spike alerts, quiet baseline noise does not, ineligible windows never alert, and - the same pattern `test_features.py` uses for the locked pipeline - a later perturbation never changes an earlier decision.
- `tests/test_monitoring_model_monitor.py` and `tests/test_monitoring_alert_log.py` check the operational IsolationForest baseline (insufficient-baseline refusal, determinism, never alerting on an ineligible window, a gross outlier scoring above baseline noise) and the alert log (writes both streams, idempotent rerun, filters by machine).
- `tests/test_dashboard_data.py` checks every pure query/SVG function on a hand-built in-memory DuckDB (including that untrusted strings are HTML-escaped in the schematic).
- `tests/test_dashboard_seed.py` and `tests/test_dashboard_app.py` run the real seed pipeline on a short DEVELOPMENT-table window (never locked) and drive `dashboard/app.py` headlessly with `streamlit.testing.v1.AppTest`, asserting no exception and that the role-line/claim text renders.

## Open items and deviations (flagged, not decided here)

- **Bundled non-OSI sub-components.** The numpy and scipy wheels vendor a few permissive but non-OSI-listed components: a CC0 file in numpy, and Qhull in scipy. See `LICENSES.md`. scikit-learn is required by the protocol and depends on both. They are covered by the company's enumerated licence exception (see `LICENSES.md`).
- **Always-alert scope.** The baseline covers every eligible window, including planned stops. It does not cover "operating time" only, because a constant detector has no schedule input. Its planned-stop alarms are reported separately.
- **Selection objective.** Event F1 is computed over all planted events, equipment and data together, with the two validation assets pooled. The protocol does not say whether data faults count toward selection, so this choice is frozen here and open to review.
- **Dependency lock scope.** The lock is platform-specific: CPython 3.13 on manylinux x86_64.
- **One evaluation, not a benchmark.** The locked evaluation ran once. With nine events per stratum it supports no significance or generalisation claim.
- **Monitoring is not the frozen evaluation.** `monitoring/`'s rule and model baselines use a live per-machine operational fit (an expanding history / a fixed causal warm-up slice), not the protocol's train/validation/test split, and are not validation-selected. No number from `monitoring/` or `dashboard/` is a locked-evaluation result.

## What this demo does not claim

- It makes no result claim of any kind, because no results exist yet.
- Any later score measures only this declared synthetic generator. Its windows are correlated, and there is no calibrated industrial confidence and no industrial reliability claim.
- It represents no real plant, employer schema or process. The line, tags and faults are invented.
- `dashboard/`'s OEE panel is the cleared plant demo's own numbers on its own deterministic day (see `report.OEE_METHOD_NOTE`) - it is not a measurement of the twin's synthetic line, and is not combined with it into one number.
- `monitoring/`'s rule and model alerts are an operational demo over whatever history is in the database (development seeds in this milestone); they are not the frozen protocol's selected model or threshold, and carry no detection-rate claim.

## Self-check before delivery

- Tested from a clean clone into a fresh venv, installing with `--require-hashes` from `requirements.lock`, then running the full suite and this Quickstart verbatim.
- Grepped for owner, host and employer strings. None are present.
- `git -C vendor/plant-shift-oee-report status` is clean, at the pinned SHA.
- This milestone (`runner/`, `dashboard/`, `monitoring/`): full suite (`python3 -m pytest -v`, 186 tests) run in the same venv plus `dashboard/requirements.txt`; `gitleaks detect` run over the working tree and history; confirmed no test generates or reads the locked seed table (101-110 / 201-204 / 301-306) - every test asserts development-seed-only (`{901..906}`) or uses hand-built fixtures, and `runner.lock_guard.verify_lock()` passes against the unmodified repo.

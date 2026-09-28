-- DuckDB schema for the operational monitoring alert log. A NEW table in
-- the same file src/schema.sql's sensor_readings/sensor_ingest_log live in
-- (src/schema.sql itself is a locked file and is never edited).
CREATE SEQUENCE IF NOT EXISTS monitoring_alert_id_seq START 1;

-- One row per (machine, window_start, source) alert. UNIQUE makes
-- re-running the monitor over already-scored history an idempotent no-op
-- (INSERT ... ON CONFLICT DO NOTHING) rather than a duplicate row. `sensor`
-- is NOT NULL (empty string for a model alert, which has no single
-- offending sensor) deliberately: a NULL in a UNIQUE column is never equal
-- to another NULL, which would silently defeat this constraint for every
-- model-alert rerun.
CREATE TABLE IF NOT EXISTS monitoring_alert_log (
    alert_id      BIGINT PRIMARY KEY DEFAULT nextval('monitoring_alert_id_seq'),
    generated_at  TIMESTAMP NOT NULL,
    machine       TEXT NOT NULL,
    window_start  BIGINT NOT NULL,   -- epoch s, features.py 10s-window convention
    window_end    BIGINT NOT NULL,
    source        TEXT NOT NULL,     -- 'rule' | 'model'
    sensor        TEXT NOT NULL,     -- rule alerts: the sensor that tripped it, else ''
    score         DOUBLE,            -- rule: |z-score|, model: -score_samples anomaly score
    detail        TEXT NOT NULL,
    UNIQUE (machine, window_start, source, sensor)
);

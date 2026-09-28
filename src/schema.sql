-- DuckDB schema for the twin's SENSOR readings (new schema - the plant demo's
-- STATE/COUNT/ALARM/HEARTBEAT schema is not reused).
CREATE SEQUENCE IF NOT EXISTS sensor_log_id_seq START 1;

-- Exactly one row per handled message, whatever its outcome.
CREATE TABLE IF NOT EXISTS sensor_ingest_log (
    log_id        BIGINT PRIMARY KEY DEFAULT nextval('sensor_log_id_seq'),
    received_at   TIMESTAMP NOT NULL,
    category      TEXT NOT NULL,   -- accepted | duplicate | reading_conflict |
                                    -- out_of_order_rejected | unknown_tag | bad_payload
    tag           TEXT,
    ts_raw        TEXT,
    ts_epoch      BIGINT,           -- UTC epoch seconds (set once ts parsed)
    raw_payload   TEXT NOT NULL,    -- verbatim, never repaired
    reason        TEXT,
    content_hash  TEXT              -- ingester._content_hash (pinned plant demo)
);

-- One fact row per accepted message, NULL = sensor absent in that message.
CREATE TABLE IF NOT EXISTS sensor_readings (
    log_id             BIGINT PRIMARY KEY,
    machine            TEXT NOT NULL,
    ts_epoch           BIGINT NOT NULL,
    vibration_rms      DOUBLE,
    bearing_temp_c     DOUBLE,
    motor_current_a    DOUBLE,
    line_pressure_bar  DOUBLE,
    UNIQUE (machine, ts_epoch)
);

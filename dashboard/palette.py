"""The validated default palette (dataviz skill, references/palette.md),
used unchanged. Categorical hues are assigned to the four sensors in fixed
order (never cycled or re-assigned by filter); status colors are reserved
for machine/alert state and never reused as a categorical series colour.
"""
from __future__ import annotations

# Categorical slots 1-4 (light mode) - the sensor order is config.yaml
# sensors.order, so this assignment is also fixed there, not just here.
SENSOR_COLOR = {
    "vibration_rms": "#2a78d6",      # slot 1: blue
    "bearing_temp_c": "#eb6834",     # slot 2: orange
    "motor_current_a": "#1baf7a",    # slot 3: aqua
    "line_pressure_bar": "#eda100",  # slot 4: yellow
}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Machine STATE (vendor plant-shift-oee-report state_events.state) -> status role
STATE_STATUS = {"RUN": "good", "IDLE": "warning", "DOWN": "critical"}
STATE_UNKNOWN_COLOR = "#898781"   # muted ink: no state evidence yet

# Alert source -> status role (icon + label always accompany the colour)
ALERT_SOURCE_STATUS = {"rule": "warning", "model": "serious"}

SURFACE_LIGHT = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

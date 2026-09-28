"""Causal expanding per-sensor z-score alert rule (NOT a locked file).

Deliberately different from runner/models.py's static_threshold_alert: that
one uses a fixed train-only mean/std per the frozen protocol's train/
validation/test split. This one is a live operational rule with no notion
of a frozen split - at window i it compares against the expanding mean/std
of that SAME asset's own strictly-PRIOR eligible windows only (never the
current or a future window), so it can run continuously over whatever
history is in the database. A window needs at least `min_history` prior
eligible observations before the rule can alert on it at all.
"""
from __future__ import annotations

import numpy as np


def expanding_zscore_alerts(frame, sensors: list[str], k: float = 3.0,
                            min_history: int = 30):
    """Returns (alert, sensor, detail, max_abs_z): bool/object/object/float
    arrays the length of `frame`. alert is True only on eligible windows
    where at least one sensor's current {sensor}__mean_5m lies more than k
    standard deviations from that sensor's own expanding mean over strictly
    earlier eligible, finite windows. sensor/detail name the
    worst-offending sensor (empty string where alert is False)."""
    eligible = np.asarray(frame.eligible, dtype=bool)
    n = len(eligible)
    alert = np.zeros(n, dtype=bool)
    sensor_out = np.full(n, "", dtype=object)
    detail = np.full(n, "", dtype=object)
    max_abs_z = np.full(n, np.nan)

    for s in sensors:
        col = np.asarray(frame.columns[f"{s}__mean_5m"], dtype=np.float64)
        finite = eligible & np.isfinite(col)
        vals = np.where(finite, col, 0.0)
        # Causal: index i's stats come only from indices < i (prefix sums
        # shifted by one), never from i itself or anything later.
        cs = np.concatenate(([0.0], np.cumsum(vals)))
        csq = np.concatenate(([0.0], np.cumsum(vals * vals)))
        cn = np.concatenate(([0.0], np.cumsum(finite.astype(np.float64))))
        prior_n, prior_sum, prior_sumsq = cn[:-1], cs[:-1], csq[:-1]

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = prior_sum / prior_n
            var = np.maximum(prior_sumsq / prior_n - mean * mean, 0.0)
            std = np.sqrt(var)
            z = (col - mean) / std
        ready = (prior_n >= min_history) & (std > 0) & finite
        z = np.where(ready, z, np.nan)
        s_alert = ready & (np.abs(z) > k)

        better = s_alert & (np.isnan(max_abs_z) | (np.abs(z) > np.nan_to_num(max_abs_z, nan=-1.0)))
        max_abs_z[better] = np.abs(z)[better]
        sensor_out[better] = s
        for i in np.flatnonzero(better):
            detail[i] = f"{s}: |z|={abs(z[i]):.2f} (k={k}, prior_n={int(prior_n[i])})"
        alert |= s_alert

    alert &= eligible
    return alert, sensor_out, detail, max_abs_z

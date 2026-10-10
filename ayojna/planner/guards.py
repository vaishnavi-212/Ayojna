"""Forecast and anomaly guards: which tiers an extent may move to THIS hour.

  anomaly pause     flagged volume -> no moves at all (slide 4: "Anomaly pause: freezes moves")
  spike predicted   forecast spike -> no demotions, promotions still allowed (slide 3)
Compliance comes first: if an extent's current tier breaks policy, it may still leave it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ayojna.models import anomaly as anomaly_model
from ayojna.models.forecast import forecast_volumes
from ayojna.models.series import volume_hourly

NONE = {"freeze": {}, "no_demote": {}}


def guards_from(forecast: pd.DataFrame | None, anomalies: pd.DataFrame | None) -> dict:
    """{"freeze": {volume: why}, "no_demote": {volume: why}} from the two models' outputs."""
    g = {"freeze": {}, "no_demote": {}}
    if anomalies is not None and len(anomalies):
        for r in anomalies[anomalies["anomaly"]].itertuples():
            g["freeze"][r.volume] = f"anomaly pause: {r.reason}"
    if forecast is not None and len(forecast):
        for r in forecast[forecast["spike"]].itertuples():
            if r.volume not in g["freeze"]:
                g["no_demote"][r.volume] = (
                    f"spike predicted: next 24 h {r.io_next24:,.0f} I/Os vs {r.io_last24:,.0f}"
                )
    return g


def apply_guards(
    allowed: np.ndarray, current: np.ndarray, volumes: np.ndarray, guards: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (restricted allowed matrix, per-extent guard note)."""
    allowed = allowed.copy()
    n = len(current)
    rows = np.arange(n)
    notes = np.full(n, "", dtype=object)
    stay_ok = allowed[rows, current]  # current tier is compliant
    slower = np.arange(allowed.shape[1])[None, :] > current[:, None]
    for kind, by_volume in guards.items():
        for vol, why in by_volume.items():
            m = (volumes == vol) & stay_ok
            if not m.any():
                continue
            if kind == "freeze":
                allowed[m] = False
                allowed[rows[m], current[m]] = True
            else:
                allowed[m] &= ~slower[m]
            notes[m] = why
    return allowed, notes


def guard_tables(eh: pd.DataFrame, store: str, cfg: dict, start: int, end: int):
    """Guards for every replayed hour in [start, end], using only data up to the hour before.

    Returns (function hour -> guards, counts). Anomalies are scored for every hour; the
    forecast is refit every `race_refit_every` hours (Prophet is too slow to refit hourly).
    """
    fc, an = cfg["forecast"], cfg["anomaly"]
    vh = volume_hourly(eh)
    feats = anomaly_model.volume_features(eh, vh)
    try:
        model = anomaly_model.AnomalyModel.load(f"{store}/{anomaly_model.LATEST}")
    except Exception:  # no trained detector: the rule learns normal from pre-replay hours
        model = anomaly_model.train(feats, start - 1, z_max=an["z_max"], use_forest=False)
    scored = model.score(feats[feats["hour"].between(start - 1, end - 1)])
    flagged = scored[scored["anomaly"]]
    freeze: dict[int, dict] = {}
    for r in flagged.itertuples():
        freeze.setdefault(int(r.hour) + 1, {})[r.volume] = f"anomaly pause: {r.reason}"
    spikes: dict[int, dict] = {}
    step = max(1, int(fc.get("race_refit_every", 6)))
    for origin in range(start - 1, end, step):
        rows = forecast_volumes(eh, fc["model"], fc["horizon_hours"], fc["spike_ratio"],
                                fc["spike_min_io"], series=("io",), vh=vh, until_hour=origin)  # fmt: skip
        g = guards_from(rows, None)["no_demote"]
        for h in range(origin + 1, min(origin + 1 + step, end + 1)):
            spikes[h] = g

    def at(hour: int) -> dict:
        f = freeze.get(hour, {})
        return {"freeze": f, "no_demote": {v: w for v, w in spikes.get(hour, {}).items() if v not in f}}

    counts = {
        "anomaly_pause_volume_hours": int(len(flagged)),
        "spike_hold_volume_hours": int(sum(len(at(h)["no_demote"]) for h in range(start, end + 1))),
        "anomaly_method": model.metrics.get("method", "isolation_forest"),
    }
    return at, counts
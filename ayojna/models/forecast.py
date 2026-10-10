"""Capacity and I/O forecast: per volume, the next 24 hours.

Models (config/models.yaml -> forecast.model):
  prophet        Meta's Prophet, daily seasonality (pip install prophet)
  seasonal_ewma  fallback, numpy only: the same hour on previous days, recent days weigh more
  naive          reference: the next 24 h repeat the last 24 h
Each volume falls back to seasonal_ewma on its own if Prophet is missing or fails.

Scored by MAPE of the 24-hour totals on rolling origins (forecast made only from the past).
Used by the planner as a guard: a predicted spike holds demotions on that volume, so data
is not sent to a slow tier right before it is needed (slide 3: "spike seen in advance").
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ayojna.models.series import volume_hourly

SERIES = ("io", "ws_gb")  # I/Os per hour, working-set GB per hour


def naive(y: np.ndarray, h: int = 24) -> np.ndarray:
    return np.resize(y[-24:], h).astype(float) if len(y) else np.zeros(h)


def seasonal_ewma(y: np.ndarray, h: int = 24, decay: float = 0.6, days: int = 7) -> np.ndarray:
    """Weighted mean of the same hour over the last `days` days (newest weight 1, then decay^d)."""
    k = min(days, len(y) // 24)
    if k == 0:
        return np.full(h, float(np.mean(y)) if len(y) else 0.0)
    past = np.asarray(y[len(y) - 24 * k :], dtype=float).reshape(k, 24)  # row 0 = oldest day
    w = decay ** np.arange(k)[::-1]
    return np.resize((w[:, None] * past).sum(axis=0) / w.sum(), h)


def prophet(y: np.ndarray, h: int = 24) -> np.ndarray:
    for name in ("prophet", "cmdstanpy"):  # silence fitting chatter
        logging.getLogger(name).setLevel(logging.ERROR)
    from prophet import Prophet  # optional dependency: ImportError -> fallback

    ds = pd.date_range("2026-01-01", periods=len(y), freq="h")
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,  # a week of trace cannot teach a weekly pattern
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,  # point forecast only: much faster
    )
    m.fit(pd.DataFrame({"ds": ds, "y": np.asarray(y, dtype=float)}))
    future = pd.DataFrame({"ds": pd.date_range(ds[-1] + pd.Timedelta(hours=1), periods=h, freq="h")})
    return np.clip(m.predict(future)["yhat"].to_numpy(), 0, None)


MODELS = {"prophet": prophet, "seasonal_ewma": seasonal_ewma, "naive": naive}


def available(name: str) -> bool:
    if name != "prophet":
        return name in MODELS
    try:
        import prophet as _  # noqa: F401

        return True
    except Exception:
        return False


def predict(y: np.ndarray, model: str, h: int = 24) -> tuple[np.ndarray, str]:
    """Forecast with `model`; on any failure use seasonal_ewma. Returns (yhat, model used)."""
    if model != "seasonal_ewma":
        try:
            return MODELS[model](y, h), model
        except Exception:
            pass
    return seasonal_ewma(y, h), "seasonal_ewma"


def forecast_volumes(
    eh: pd.DataFrame,
    model: str = "prophet",
    horizon: int = 24,
    spike_ratio: float = 1.5,
    spike_min_io: float = 1000,
    series: tuple[str, ...] = SERIES,
    vh: pd.DataFrame | None = None,
    until_hour: int | None = None,
) -> pd.DataFrame:
    """One row per volume: last 24 h vs next 24 h (forecast), spike flag, model used."""
    vh = volume_hourly(eh) if vh is None else vh
    if until_hour is not None:
        vh = vh[vh["hour"] <= until_hour]
    rows = []
    for vol, g in vh.groupby("volume", sort=True):
        row, used = {"volume": vol}, set()
        for s in series:
            y = g[s].to_numpy(dtype=float)
            yhat, u = predict(y, model, horizon)
            used.add(u)
            agg = np.mean if s == "ws_gb" else np.sum  # working set: average, I/O: total
            row[f"{s}_last24"] = round(float(agg(y[-24:])), 3)
            row[f"{s}_next24"] = round(float(agg(yhat)), 3)
        last, nxt = row.get("io_last24", 0.0), row.get("io_next24", 0.0)
        row["io_ratio"] = round(nxt / last, 3) if last > 0 else None
        row["spike"] = bool(nxt >= spike_min_io and nxt >= spike_ratio * max(last, 1.0))
        row["model"] = "/".join(sorted(used))
        rows.append(row)
    return pd.DataFrame(rows)


def backtest(
    eh: pd.DataFrame, models: list[str], origins: int = 3, horizon: int = 24
) -> dict[str, dict]:
    """MAPE of the 24 h totals on the last `origins` days, each forecast made from the past only."""
    vh = volume_hourly(eh)
    end = int(vh["hour"].max()) + 1
    cuts = [end - horizon * (j + 1) for j in range(origins) if end - horizon * (j + 1) >= 48]
    out = {}
    for name in models:
        if not available(name):
            out[name] = {"status": "not installed"}
            continue
        err = {s: [] for s in SERIES}
        used = set()
        for vol, g in vh.groupby("volume"):
            for cut in cuts:
                for s in SERIES:
                    y = g[s].to_numpy(dtype=float)
                    yhat, u = predict(y[:cut], name, horizon)
                    used.add(u)
                    agg = np.mean if s == "ws_gb" else np.sum
                    actual, fc = float(agg(y[cut : cut + horizon])), float(agg(yhat))
                    if actual > 0:
                        err[s].append(abs(fc - actual) / actual)
        out[name] = {
            "status": "ok" if used == {name} else f"partly fell back ({'/'.join(sorted(used))})",
            "mape_io_pct": round(100 * float(np.mean(err["io"])), 1) if err["io"] else None,
            "mape_capacity_pct": round(100 * float(np.mean(err["ws_gb"])), 1)
            if err["ws_gb"]
            else None,
            "origins": len(cuts),
            "points": len(err["io"]),
        }
    return out
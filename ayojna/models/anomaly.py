"""Anomaly guard: is a volume behaving unlike its normal self this hour?

Model: scikit-learn IsolationForest over per-volume, robust-scaled workload features
(each volume is compared with ITS OWN normal, so a busy database is not "anomalous" just
for being busy). Fallback: a rule that flags any feature more than z_max robust standard
deviations from normal. Every flag says which feature was most unusual.

The planner pauses all moves on a flagged volume (slide 4: "Anomaly pause: freezes moves"):
sudden scans, bursts or broken telemetry are bad moments to act on a forecast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ayojna.models.series import volume_hourly

FEATURES = [
    "log_io",
    "read_ratio",
    "log_avg_io_size",
    "rand_ratio",
    "log_ws",
    "jump_vs_yesterday",
    "jump_vs_last_day",
]
READABLE = {
    "log_io": "I/O volume",
    "read_ratio": "read share",
    "log_avg_io_size": "I/O size",
    "rand_ratio": "random-access share",
    "log_ws": "extents touched",
    "jump_vs_yesterday": "I/O vs same hour yesterday",
    "jump_vs_last_day": "I/O vs last 24 h average",
}
LATEST = "anomaly-latest.joblib"


def volume_features(eh: pd.DataFrame, vh: pd.DataFrame | None = None) -> pd.DataFrame:
    vh = volume_hourly(eh) if vh is None else vh
    df = vh[["volume", "hour", "read_ratio", "rand_ratio"]].copy()
    df["log_io"] = np.log1p(vh["io"])
    df["log_avg_io_size"] = np.log1p(vh["avg_io_size"])
    df["log_ws"] = np.log1p(vh["extents"])
    g = df.groupby("volume", sort=False)["log_io"]
    df["jump_vs_yesterday"] = df["log_io"] - g.shift(24)
    day_mean = vh.groupby("volume", sort=False)["io"].transform(
        lambda s: s.shift(1).rolling(24, min_periods=24).mean()
    )
    df["jump_vs_last_day"] = df["log_io"] - np.log1p(day_mean)
    return df.dropna(subset=["jump_vs_yesterday", "jump_vs_last_day"]).reset_index(drop=True)


@dataclass
class AnomalyModel:
    forest: IsolationForest | None  # None = rule-only model
    center: dict  # volume -> per-feature median of its normal hours
    scale: dict  # volume -> per-feature robust spread
    z_max: float
    version: str
    metrics: dict = field(default_factory=dict)

    def zscores(self, df: pd.DataFrame) -> np.ndarray:
        g_c, g_s = self.center["__all__"], self.scale["__all__"]
        c = np.array([self.center.get(v, g_c) for v in df["volume"]])
        s = np.array([self.scale.get(v, g_s) for v in df["volume"]])
        return (df[FEATURES].to_numpy(dtype=float) - c) / s

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self.zscores(df)
        top = np.abs(z).argmax(axis=1)
        top_z = z[np.arange(len(z)), top]
        out = df[["volume", "hour"]].copy()
        if self.forest is not None:
            out["score"] = -self.forest.score_samples(z)  # higher = more unusual
            out["anomaly"] = self.forest.predict(z) == -1
            out["method"] = "isolation_forest"
        else:
            out["score"] = np.abs(top_z)
            out["anomaly"] = np.abs(top_z) > self.z_max
            out["method"] = "robust_z_rule"
        out["score"] = out["score"].round(3)
        out["top_feature"] = [READABLE[FEATURES[j]] for j in top]
        out["top_z"] = np.round(top_z, 2)
        out["reason"] = [
            f"{READABLE[FEATURES[j]]} {'+' if v > 0 else ''}{v:.1f} SD from this volume's normal"
            for j, v in zip(top, top_z)
        ]
        return out

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / LATEST
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "AnomalyModel":
        m = joblib.load(Path(path))
        if not isinstance(m, AnomalyModel):
            raise TypeError(f"{path} is not an AnomalyModel")
        return m


def _robust(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    med = x.median().to_numpy(dtype=float)
    mad = (x - med).abs().median().to_numpy(dtype=float) * 1.4826
    return med, np.maximum(mad, 0.05)  # floor: a feature that never varies still has a scale


def train(
    feats: pd.DataFrame,
    until_hour: int,
    contamination: float = 0.02,
    z_max: float = 6.0,
    use_forest: bool = True,
    seed: int = 7,
) -> AnomalyModel:
    """Learn each volume's normal from hours <= until_hour (never the scored hours)."""
    tr = feats[feats["hour"] <= until_hour]
    if len(tr) < 48:
        raise ValueError("not enough history to learn normal behaviour (need 2+ days)")
    center, scale = {}, {}
    center["__all__"], scale["__all__"] = _robust(tr[FEATURES])
    for vol, g in tr.groupby("volume"):
        center[vol], scale[vol] = _robust(g[FEATURES])
    model = AnomalyModel(None, center, scale, z_max, datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    if use_forest:
        forest = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1
        )
        model.forest = forest.fit(model.zscores(tr))
    scored = model.score(tr)
    model.metrics = {
        "method": "isolation_forest" if use_forest else "robust_z_rule",
        "train_hours": [int(tr["hour"].min()), int(tr["hour"].max())],
        "train_rows": int(len(tr)),
        "train_flag_rate": round(float(scored["anomaly"].mean()), 4),
    }
    return model


def detect(
    eh: pd.DataFrame, store: str | Path = "models_store", z_max: float = 6.0, hour: int | None = None
) -> tuple[pd.DataFrame, str, bool]:
    """Score one hour (default: the latest) for every volume.

    Returns (rows, note, degraded). Without a saved model the rule learns "normal" from
    all earlier hours on the spot, so the guard still works (degraded).
    """
    feats = volume_features(eh)
    h = int(feats["hour"].max()) if hour is None else hour
    now = feats[feats["hour"] == h]
    path = Path(store) / LATEST
    try:
        model = AnomalyModel.load(path)
        return model.score(now), f"isolation forest {model.version}", False
    except Exception as exc:
        rule = train(feats, h - 1, z_max=z_max, use_forest=False)
        return rule.score(now), f"rule fallback ({type(exc).__name__})", True


def summary(rows: pd.DataFrame) -> list[dict]:
    return json.loads(rows.to_json(orient="records"))
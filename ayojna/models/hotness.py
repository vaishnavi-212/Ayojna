"""ML hotness model: predicts hot / warm / cold for the next 24 hours.

Algorithms (a model zoo; train_hotness picks the champion on a validation window):
  lightgbm       LightGBM gradient-boosted trees          (pip install lightgbm)
  xgboost        XGBoost gradient-boosted trees           (pip install xgboost)
  random_forest  scikit-learn RandomForest
  hist_gb        scikit-learn HistGradientBoosting (LightGBM-style histogram trees)
A library that is not installed is skipped, never a crash. All use balanced class weights.

Reasons ("why"), per prediction:
  LightGBM / XGBoost -> exact TreeSHAP values computed by the library itself
  other models       -> occlusion: replace one feature with its typical value, measure the drop
Either way each driver has a signed effect: + pushed toward the prediction, - against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ayojna.contracts import Tier
from ayojna.models.features import FEATURE_COLUMNS, KEYS

CLASSES = [Tier.HOT.value, Tier.WARM.value, Tier.COLD.value]
ABSTAIN_BELOW = 0.6  # below this confidence the planner must not move the extent

READABLE = {
    "acc_1h": "I/Os in the last hour",
    "acc_6h": "I/Os in the last 6 h",
    "acc_24h": "I/Os in the last 24 h",
    "acc_72h": "I/Os in the last 72 h",
    "hours_since_access": "hours since last access",
    "trend_24_vs_72": "activity trend (24 h vs 72 h)",
    "same_hour_yesterday": "I/Os at this hour yesterday",
    "read_ratio_24h": "read share (24 h)",
    "avg_io_size_24h": "average I/O size (24 h)",
    "rand_ratio_24h": "random-access share (24 h)",
    "hour_of_day": "hour of day",
}


@dataclass
class HotnessModel:
    estimator: object  # any classifier with predict_proba and classes_
    features: list[str]
    typical: dict  # median of each feature in training data (used for reasons)
    spread: dict  # spread (IQR) of each feature in training data (used for reasons)
    version: str
    metrics: dict = field(default_factory=dict)
    algo: str = "hist_gb"

    @property
    def explainer(self) -> str:
        return "treeshap" if self.algo in TREESHAP else "occlusion"

    # ---------- prediction ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilities in CLASSES order: hot, warm, cold."""
        x = df[self.features].to_numpy(dtype=float)
        raw = self.estimator.predict_proba(x)
        order = [list(self.estimator.classes_).index(c) for c in CLASSES]
        return raw[:, order]

    def predict(self, df: pd.DataFrame, with_reasons: bool = True) -> pd.DataFrame:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"features missing: {missing}")
        p = self.predict_proba(df)
        idx = p.argmax(axis=1)
        out = df[KEYS].copy()
        out["p_hot"], out["p_warm"], out["p_cold"] = p[:, 0], p[:, 1], p[:, 2]
        out["pred"] = np.array(CLASSES)[idx]
        out["confidence"] = p.max(axis=1)
        out["abstain"] = out["confidence"] < ABSTAIN_BELOW
        if with_reasons:
            out["reasons"], out["drivers"] = self.explain(df, p, idx)
        else:
            out["reasons"], out["drivers"] = "", "[]"
        return out

    def explain(
        self, df: pd.DataFrame, p: np.ndarray, idx: np.ndarray, top_k: int = 3
    ) -> tuple[list[str], list[str]]:
        """Per-prediction drivers: TreeSHAP when the library offers it, else occlusion.

        Positive effect = pushed toward the prediction, negative = against it.
        Returns (reasons text, drivers JSON).
        """
        x = df[self.features].to_numpy(dtype=float)
        rows = np.arange(len(x))
        effect = None
        if self.algo in TREESHAP:
            try:
                effect = _treeshap(self.estimator, self.algo, x, idx)
            except Exception:  # never lose a prediction over its explanation
                effect = None
        if effect is None:
            effect = self._occlusion(x, p, idx)
        typical = np.array([self.typical[f] for f in self.features])
        spread = np.array([self.spread[f] for f in self.features])
        unusual = np.abs(x - typical) / spread
        reasons, drivers = [], []
        for i in rows:
            best = [j for j in np.argsort(-effect[i])[:top_k] if effect[i, j] > 0.01]
            if not best:  # several features agree, none decisive alone: name the most unusual ones
                best = [j for j in np.argsort(-unusual[i])[:2] if unusual[i, j] > 0]
            reasons.append(
                "; ".join(
                    f"{READABLE.get(self.features[j], self.features[j])} = {x[i, j]:g}"
                    for j in best
                )
            )
            top = np.argsort(-np.abs(effect[i]))[:5]
            drivers.append(
                json.dumps(
                    [
                        {
                            "feature": READABLE.get(self.features[j], self.features[j]),
                            "value": round(float(x[i, j]), 3),
                            "effect": round(float(effect[i, j]), 3),
                        }
                        for j in top
                        if abs(effect[i, j]) >= 0.005
                    ]
                )
            )
        return reasons, drivers

    def _occlusion(self, x: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Confidence in the predicted class minus the confidence with one feature set typical."""
        rows = np.arange(len(x))
        base = p[rows, idx]
        order = [list(self.estimator.classes_).index(c) for c in CLASSES]
        effect = np.zeros((len(x), len(self.features)))
        for j, name in enumerate(self.features):
            xj = x.copy()
            xj[:, j] = self.typical[name]
            effect[:, j] = base - self.estimator.predict_proba(xj)[:, order][rows, idx]
        return effect

    def reasons(self, df: pd.DataFrame, p: np.ndarray, idx: np.ndarray, top_k: int = 3):
        """Readable top reasons only (kept for callers that do not need drivers)."""
        return self.explain(df, p, idx, top_k)[0]

    # ---------- persistence ----------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"hotness-{self.version}.joblib"
        joblib.dump(self, path)
        (folder / f"hotness-{self.version}.json").write_text(
            json.dumps(
                {
                    "version": self.version,
                    "algo": self.algo,
                    "explainer": self.explainer,
                    "features": self.features,
                    "metrics": self.metrics,
                },
                indent=2,
            )
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "HotnessModel":
        model = joblib.load(Path(path))
        if not isinstance(model, HotnessModel):
            raise TypeError(f"{path} is not a HotnessModel")
        return model


class Encoded:
    """XGBoost wants labels 0..K-1: this wrapper keeps the string labels for everyone else."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.array(sorted(set(y)))
        codes = np.searchsorted(self.classes_, y)
        self.inner.fit(x, codes, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        return self.inner.predict_proba(x)


ALGOS = ("lightgbm", "xgboost", "random_forest", "hist_gb")
TREESHAP = ("lightgbm", "xgboost")


def make_estimator(algo: str, seed: int = 7):
    """Unfitted classifier for `algo`. Raises ImportError if its library is not installed."""
    if algo == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
        )  # fmt: skip
    if algo == "xgboost":
        from xgboost import XGBClassifier

        return Encoded(
            XGBClassifier(
                n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8,
                colsample_bytree=0.9, tree_method="hist", random_state=seed, n_jobs=-1,
            )
        )  # fmt: skip
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )  # fmt: skip
    if algo == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15, random_state=seed,
        )  # fmt: skip
    raise ValueError(f"unknown algorithm {algo!r}; choose from {ALGOS}")


def _treeshap(est, algo: str, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP of the predicted class (log-odds units), computed by LightGBM/XGBoost."""
    pos = [list(est.classes_).index(c) for c in CLASSES]  # CLASSES index -> estimator column
    k = np.array(pos)[idx]  # estimator class column of each row's prediction
    n, f = x.shape
    if algo == "lightgbm":
        raw = np.asarray(est.booster_.predict(x, pred_contrib=True))  # (n, (f+1)*classes)
        contrib = raw.reshape(n, -1, f + 1)
    else:
        import xgboost

        raw = est.inner.get_booster().predict(xgboost.DMatrix(x), pred_contribs=True)
        contrib = np.asarray(raw).reshape(n, -1, f + 1)  # (n, classes, f+1)
    return contrib[np.arange(n), k, :f]  # drop the bias column


def train(train_df: pd.DataFrame, seed: int = 7, algo: str = "hist_gb") -> HotnessModel:
    data = train_df[train_df["label"].notna()]
    if data["label"].nunique() < 2:
        raise ValueError("training data has fewer than 2 classes: replay more trace hours")
    est = make_estimator(algo, seed)
    x, y = data[FEATURE_COLUMNS].to_numpy(dtype=float), data["label"].to_numpy()
    if algo == "xgboost":  # no class_weight option: balance with sample weights
        est.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        est.fit(x, y)
    # every class must be present, otherwise CLASSES order cannot be built
    for c in CLASSES:
        if c not in est.classes_:
            raise ValueError(f"class {c!r} missing from training labels")
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    typical = {c: float(data[c].median()) for c in FEATURE_COLUMNS}
    iqr = data[FEATURE_COLUMNS].quantile(0.75) - data[FEATURE_COLUMNS].quantile(0.25)
    std = data[FEATURE_COLUMNS].std().fillna(0)
    spread = {
        c: float(iqr[c]) if iqr[c] > 0 else (float(std[c]) if std[c] > 0 else 1.0)
        for c in FEATURE_COLUMNS
    }
    return HotnessModel(est, list(FEATURE_COLUMNS), typical, spread, version, algo=algo)
"""Ayojna as a twin strategy: predictions -> policy guard -> forecast/anomaly guards -> optimizer.

At the start of hour h it uses the predictions made at hour h-1 (features built
only from hours <= h-1), so it never sees the future. The guards obey the same rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ayojna.planner.guards import apply_guards
from ayojna.planner.optimizer import TierEconomics, expected_ios, load_planner_config, plan
from ayojna.policy.guard import allowed_tiers
from ayojna.twin.sim import Twin, TwinView


class AyojnaStrategy:
    name = "ayojna"

    def __init__(
        self, twin: Twin, predictions: pd.DataFrame, features: pd.DataFrame, guards=None
    ):
        """predictions: output of predict_hotness; features: the matching feature rows;
        guards: optional function hour -> {"freeze": {...}, "no_demote": {...}}."""
        self.cfg = load_planner_config()
        self.twin = twin
        self.guards = guards
        H, N = twin.n_hours, twin.n_extents
        col = pd.DataFrame(
            {"volume": twin.volumes, "extent_id": twin.extent_ids, "col": np.arange(N)}
        )
        cols = ["volume", "extent_id", "hour", "avg_io_size_24h", "read_ratio_24h", "acc_24h"]
        df = predictions.merge(features[cols + ["future_acc", "label"]], on=cols[:3])
        df = df.merge(col, on=["volume", "extent_id"])
        # expected I/Os per hour for a hot / warm extent, learned from labelled history
        rate = (df.dropna(subset=["label"]).groupby("label")["future_acc"].mean() / 24).to_dict()
        lam = expected_ios(df["p_hot"], df["p_warm"], rate, df["acc_24h"])
        self.exp_ios = np.zeros((H, N))
        self.read_gb = np.zeros((H, N))
        self.conf = np.ones((H, N), dtype=bool)  # never-seen extents: confident they are idle
        h, c = df["hour"].to_numpy(), df["col"].to_numpy()
        self.exp_ios[h, c] = lam
        self.read_gb[h, c] = df["avg_io_size_24h"] * df["read_ratio_24h"] / 1e9
        self.conf[h, c] = df["confidence"] >= self.cfg["abstain_below"]
        self.eco = TierEconomics.from_twin(twin)
        self.since = np.zeros(N)

    def decide(self, view: TwinView) -> np.ndarray:
        if view.hour == 0:
            return view.placement
        h = view.hour - 1  # latest predictions available
        allowed, _ = allowed_tiers(view.volumes, view.placement)
        if self.guards is not None:
            allowed, _ = apply_guards(allowed, view.placement, view.volumes, self.guards(view.hour))
        choice = plan(
            self.exp_ios[h],
            self.read_gb[h],
            self.twin.sla_target_ms,
            view.placement,
            view.hour - self.since,
            allowed,
            self.conf[h],
            self.eco,
            self.cfg,
        )
        self.since[choice != view.placement] = view.hour
        return choice
"""Ayojna as a twin strategy: policy -> forecast/anomaly guards -> RL bandit -> exact optimizer.

At the start of hour h it uses the predictions made at hour h-1 (features built
only from hours <= h-1), so it never sees the future. The guards obey the same rule, and
the bandit learns from a decision only after `reward_window_hours` of real traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ayojna.planner import bandit as rl
from ayojna.planner.guards import apply_guards
from ayojna.planner.optimizer import TierEconomics, expected_ios, load_planner_config, plan
from ayojna.policy.guard import allowed_tiers
from ayojna.twin.sim import Twin, TwinView


class AyojnaStrategy:
    name = "ayojna"

    def __init__(
        self,
        twin: Twin,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        guards=None,
        use_bandit: bool | None = None,
        explore_before: int | None = None,
    ):
        """predictions: output of predict_hotness; features: the matching feature rows;
        guards: optional function hour -> {"freeze": {...}, "no_demote": {...}};
        use_bandit: None = as configured in planner.yaml;
        explore_before: explore only in hours before this one (training), then exploit."""
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
        self.solver_log: list[dict] = []
        # ---------- contextual bandit ----------
        bcfg = self.cfg.get("bandit", {})
        on = bcfg.get("enabled", False) if use_bandit is None else use_bandit
        self.bandit = rl.LinUCB(list(bcfg["arms"]), bcfg.get("alpha", 0.6)) if on else None
        self.window = int(bcfg.get("reward_window_hours", 24))
        self.rows = {v: np.flatnonzero(twin.volumes == v) for v in np.unique(twin.volumes)}
        self.learnable = [v for v, r in self.rows.items() if not twin.legal_hold[r].any()]
        self.pending: list[tuple] = []  # (hour, volume, arm, context, before, after)
        self.arm_log: list[dict] = []
        self.explore_before = explore_before

    def _learn(self, now: int) -> None:
        """Score decisions whose reward window has fully happened (hours < now only)."""
        keep = []
        for hour, vol, arm, x, before, after in self.pending:
            if hour + self.window <= now:
                hours = range(hour, hour + self.window)
                r = rl.reward(self.twin, self.rows[vol], before, after, hours, self.cfg)
                self.bandit.update(arm, x, r)
            else:
                keep.append((hour, vol, arm, x, before, after))
        self.pending = keep

    def _arms(self, view: TwinView) -> tuple[dict, dict]:
        contexts, arms = {}, {}
        for vol in self.learnable:
            r = self.rows[vol]
            hist = view.ios_past[:, r].sum(axis=1)
            contexts[vol] = rl.context(hist, float(self.twin.sla_target_ms[r].min()))
            explore = self.explore_before is None or view.hour < self.explore_before
            arms[vol] = self.bandit.choose(contexts[vol], explore)
        return contexts, arms

    def decide(self, view: TwinView) -> np.ndarray:
        if view.hour == 0:
            return view.placement
        h = view.hour - 1  # latest predictions available
        allowed, _ = allowed_tiers(view.volumes, view.placement)
        if self.guards is not None:
            allowed, _ = apply_guards(allowed, view.placement, view.volumes, self.guards(view.hour))
        knobs, contexts, arms = None, {}, {}
        if self.bandit is not None:
            self._learn(view.hour)
            contexts, arms = self._arms(view)
            knobs = rl.knobs_for(arms, view.volumes, self.cfg["bandit"]["arms"])
            self.arm_log.append({"hour": view.hour, **arms})
        info: dict = {}
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
            knobs=knobs,
            info=info,
        )
        self.solver_log.append({"hour": view.hour, **info})
        for vol, arm in arms.items():
            self.pending.append((view.hour, vol, arm, contexts[vol], view.placement, choice))
        self.since[choice != view.placement] = view.hour
        return choice

    def report(self, start: int) -> dict:
        """Solver and bandit behaviour over the scored hours (for the scoreboard)."""
        logs = [x for x in self.solver_log if x["hour"] >= start]
        solvers = pd.Series([x.get("solver", "greedy") for x in logs]).value_counts().to_dict()
        gain = [x["greedy_objective"] - x["objective"] for x in logs if "greedy_objective" in x]
        out = {
            "solver": {
                "used": {k: int(v) for k, v in solvers.items()},
                "optimal_hours": sum(x.get("status") == "optimal" for x in logs),
                "mean_ms": round(float(np.mean([x.get("ms", 0) for x in logs])), 1) if logs else 0,
                "objective_gain_vs_greedy": round(float(np.sum(gain)), 6) if gain else 0.0,
                "fallbacks": sorted({f for x in logs for f in x.get("fallbacks", [])})[:3],
            }
        }
        if self.bandit is not None:
            scored = [a for a in self.arm_log if a["hour"] >= start]
            picks = pd.DataFrame(scored).drop(columns="hour") if scored else pd.DataFrame()
            out["bandit"] = {
                "arms": self.bandit.summary(),
                "picks_scored_hours": {a: int((picks == a).sum().sum()) for a in self.bandit.arms},
                "last_choice": {k: v for k, v in (self.arm_log[-1] if self.arm_log else {}).items()
                                if k != "hour"},
            }  # fmt: skip
        return out
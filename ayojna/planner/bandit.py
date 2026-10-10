"""Contextual bandit (RL): how aggressively should each volume be re-tiered right now?

Arms (planner.yaml -> bandit.arms) change only HOW EAGER the optimizer is to move:
  cautious    high hysteresis, a move must pay back over 2 weeks
  balanced    the default settings
  aggressive  low hysteresis, a move only needs to pay back over 3 days
Context per volume, from the past only: load level, burstiness, trend, SLA strictness.
Algorithm: LinUCB (one ridge regression per arm + an exploration bonus).

Reward, measured after `reward_window_hours` of real traffic: the return on the data the
decision moved (cheaper than staying put? SLA misses and queueing priced in, move cost
charged pro rata), in [-1, 1]. The replay trains it online (the digital twin); it may only
steer the live planner after beating the static settings on a validation day.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ayojna.twin.sim import EXTENT_GB

FILE = "bandit.json"
DIM = 6


def context(io_hist: np.ndarray, sla_ms: float) -> np.ndarray:
    """Features of one volume from its hourly I/O history (past hours only)."""
    last24 = io_hist[-24:] if len(io_hist) else np.zeros(1)
    last6 = io_hist[-6:] if len(io_hist) else np.zeros(1)
    mean24 = float(np.mean(last24))
    cv = float(np.std(last24) / mean24) if mean24 > 0 else 0.0
    trend = np.log1p(float(np.mean(last6))) - np.log1p(mean24)
    return np.array([
        1.0,
        np.log1p(mean24) / 12.0,  # load level
        min(cv, 3.0) / 3.0,  # burstiness
        float(np.clip(trend, -3, 3)) / 3.0,  # heating up (+) or cooling down (-)
        1.0 if sla_ms <= 1.0 else 0.0,  # strict SLA (oltp)
        1.0 if sla_ms >= 1000.0 else 0.0,  # loose SLA (backup)
    ])  # fmt: skip


class LinUCB:
    def __init__(self, arms: list[str], alpha: float = 0.6, dim: int = DIM):
        self.arms, self.alpha = list(arms), alpha
        self.A = {a: np.eye(dim) for a in self.arms}
        self.b = {a: np.zeros(dim) for a in self.arms}
        self.n = {a: 0 for a in self.arms}
        self.reward_sum = {a: 0.0 for a in self.arms}

    def scores(self, x: np.ndarray, explore: bool = True) -> dict[str, float]:
        out = {}
        for a in self.arms:
            inv = np.linalg.inv(self.A[a])
            theta = inv @ self.b[a]
            bonus = self.alpha * float(np.sqrt(x @ inv @ x)) if explore else 0.0
            out[a] = float(theta @ x) + bonus
        return out

    def choose(self, x: np.ndarray, explore: bool = True) -> str:
        s = self.scores(x, explore)
        return max(self.arms, key=lambda a: s[a])  # ties -> first arm in config order

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.n[arm] += 1
        self.reward_sum[arm] += reward

    def summary(self) -> dict:
        return {
            a: {"pulls": self.n[a],
                "mean_reward": round(self.reward_sum[a] / self.n[a], 4) if self.n[a] else None}
            for a in self.arms
        }  # fmt: skip

    def save(self, folder: str | Path, extra: dict | None = None) -> Path:
        path = Path(folder) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"arms": self.arms, "alpha": self.alpha,
                "A": {a: self.A[a].tolist() for a in self.arms},
                "b": {a: self.b[a].tolist() for a in self.arms},
                "n": self.n, "reward_sum": self.reward_sum, **(extra or {})}  # fmt: skip
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @classmethod
    def load(cls, folder: str | Path) -> "LinUCB":
        d = json.loads((Path(folder) / FILE).read_text(encoding="utf-8"))
        m = cls(d["arms"], d["alpha"])
        m.A = {a: np.array(v) for a, v in d["A"].items()}
        m.b = {a: np.array(v) for a, v in d["b"].items()}
        m.n, m.reward_sum = d["n"], d["reward_sum"]
        return m


def live_policy(store: str | Path, vol_io: dict, vol_sla: dict) -> dict:
    """What the bandit learned in the replay, applied to the volumes right now (no exploring).

    Returns {"available": False} before any replay; otherwise whether it is promoted (only
    then do its choices drive the optimizer), its choice per volume and its arm statistics.
    """
    path = Path(store) / FILE
    if not path.exists():
        return {"available": False}
    meta = json.loads(path.read_text(encoding="utf-8"))
    b = LinUCB.load(store)
    suggested = {v: b.choose(context(io, vol_sla[v]), explore=False) for v, io in vol_io.items()}
    return {
        "available": True,
        "promoted": bool(meta.get("promoted", False)),
        "mode": "active" if meta.get("promoted") else "shadow",
        "suggested": suggested,
        "arms": b.summary(),
        "validation": meta.get("validation"),
    }


def knobs_for(arm_of_volume: dict, volumes: np.ndarray, arms_cfg: dict) -> dict:
    """Per-extent knob arrays for optimizer.plan(knobs=...)."""
    hx = np.array([arms_cfg[arm_of_volume.get(v, "balanced")]["hysteresis_x"] for v in volumes])
    am = np.array(
        [arms_cfg[arm_of_volume.get(v, "balanced")]["amortization_hours"] for v in volumes]
    )
    return {"hysteresis_x": hx, "amortization_hours": am}


def window_cost(twin, rows, full, hours, cfg) -> float:
    """$ cost of extents `rows` over `hours` if placement `full` is held, using REAL I/O.

    Storage, retrieval and an SLA penalty for every I/O whose latency (base / (1 - tier
    utilisation)) misses its target. Utilisation comes from the full placement, so queueing
    caused by other volumes counts too.
    """
    cost = 0.0
    for h in hours:
        ios = twin.ios[h]
        load = np.bincount(full, weights=ios, minlength=4)
        util = np.minimum(twin.max_util, load / twin.ios_capacity)
        lat = (twin.base_latency / (1 - util))[full[rows]]
        cost += np.sum(EXTENT_GB * twin.price[full[rows]]) / twin.hours_per_month
        cost += np.sum(twin.read_bytes[h, rows] / 1e9 * twin.retrieval[full[rows]])
        cost += np.sum(ios[rows] * (lat > twin.sla_target_ms[rows])) * cfg["sla_penalty_per_io"]
    return float(cost)


def reward(twin, rows, before, after, hours, cfg) -> float:
    """Return on the data this decision moved, in [-1, 1]; 0 if it moved nothing.

    (cost of keeping the moved extents where they were - cost after moving)
    / cost of keeping them, over the reward window, with real I/O and queueing.
    The one-time move cost is charged pro rata (window / move_amortization_hours), exactly
    as the optimizer prices it, so a move meant to pay back over a week is not judged as
    if it had to pay back within one day.
    """
    moved = rows[after[rows] != before[rows]]
    if len(moved) == 0:
        return 0.0
    stay = after.copy()
    stay[moved] = before[moved]
    share = min(1.0, len(hours) / cfg["move_amortization_hours"])
    c_stay = window_cost(twin, moved, stay, hours, cfg)
    c_move = window_cost(twin, moved, after, hours, cfg)
    c_move += share * len(moved) * EXTENT_GB * twin.move_cost_per_gb
    return float(np.clip((c_stay - c_move) / (c_stay + 1e-9), -1.0, 1.0))
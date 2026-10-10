"""Ayojna planner: choose the cheapest allowed tier for every extent.

Expected cost of extent i on tier t over the next H hours:
    storage + expected retrieval + expected SLA misses x penalty
    + (move cost + early-deletion fee, only if t changes)
Then: capacity limits on hot and warm (demote the extents that lose least), queue relief
(keep each tier's load inside its SLA, without overfilling faster tiers or churning), and
a migration budget (keep the moves with the biggest benefit).

That greedy plan is always computed. With `solver: auto | ortools | highs` (planner.yaml) the
same costs and limits are then solved EXACTLY as one optimisation problem (planner/exact.py);
the greedy plan stays as the fallback and as the yardstick the exact plan is compared with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ayojna.contracts import EXTENT_MB
from ayojna.planner import exact
from ayojna.settings import CONFIG_DIR

GB = EXTENT_MB / 1024


def load_planner_config() -> dict:
    with open(Path(CONFIG_DIR) / "planner.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TierEconomics:
    price: np.ndarray  # $ per GB-month, per tier
    retrieval: np.ndarray  # $ per GB read, per tier
    min_hours: np.ndarray  # minimum storage time, per tier
    latency_ms: np.ndarray  # base latency, per tier
    capacity: np.ndarray  # max extents, per tier
    move_cost_per_gb: float
    hours_per_month: float
    ios_capacity: np.ndarray | None = None  # I/Os per hour per tier before queues build up

    @classmethod
    def from_twin(cls, twin) -> "TierEconomics":
        return cls(
            twin.price,
            twin.retrieval,
            twin.min_hours,
            twin.base_latency,
            twin.capacity_extents,
            twin.move_cost_per_gb,
            twin.hours_per_month,
            twin.ios_capacity,
        )


def with_knobs(cfg: dict, knobs: dict | None) -> dict:
    """Per-extent aggressiveness from the bandit: scales hysteresis, sets the amortization."""
    if not knobs:
        return cfg
    return {
        **cfg,
        "hysteresis_usd": cfg["hysteresis_usd"] * np.asarray(knobs["hysteresis_x"], dtype=float),
        "move_amortization_hours": np.asarray(knobs["amortization_hours"], dtype=float),
    }


def expected_ios(p_hot, p_warm, rate: dict, acc_24h) -> np.ndarray:
    """Expected I/Os per hour of each extent over the planning horizon.

    The class forecast (P(hot) x rate of a typical hot extent + P(warm) x warm rate) is never
    allowed below the extent's OWN rate over the last 24 h: a "hot" extent doing 6,000 I/Os
    an hour must not be priced as an average hot extent, or a slow tier's queue overflows.
    """
    lam = np.asarray(p_hot) * rate.get("hot", 0.0) + np.asarray(p_warm) * rate.get("warm", 0.0)
    return np.maximum(lam, np.asarray(acc_24h, dtype=float) / 24.0)


def cost_breakdown(
    expected_ios: np.ndarray,
    read_gb_per_io: np.ndarray,
    sla_ms: np.ndarray,
    current: np.ndarray,
    held_hours: np.ndarray,
    eco: TierEconomics,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Expected cost of each extent on each tier over the horizon, split by cause ($).

    storage, retrieval, sla_risk and move are (n, 4); move is 0 on the current tier and is
    the one-time move cost + early-deletion fee + hysteresis, amortized over the expected stay.
    move_once (n,) is the un-amortized one-time cost, used for the payback time.
    """
    H = cfg["horizon_hours"]
    ios_h = (expected_ios * H)[:, None]
    remaining = np.clip(eco.min_hours[current] - held_hours, 0, None)
    leave_fee = GB * eco.price[current] * remaining / eco.hours_per_month
    move_once = GB * eco.move_cost_per_gb + leave_fee
    move = np.repeat(
        ((move_once + cfg["hysteresis_usd"]) * H / cfg["move_amortization_hours"])[:, None], 4, 1
    )
    move[np.arange(len(current)), current] = 0.0  # staying costs no move
    return {
        "storage": np.repeat((GB * eco.price * H / eco.hours_per_month)[None, :], len(current), 0),
        "retrieval": ios_h * read_gb_per_io[:, None] * eco.retrieval[None, :],
        "sla_risk": ios_h * (eco.latency_ms[None, :] > sla_ms[:, None]) * cfg["sla_penalty_per_io"],
        "move": move,
        "move_once": move_once,
    }


def _queue_repair(choice, cost, expected_ios, sla_ms, current, eco, headroom) -> None:
    """Keep each tier's expected load low enough that queueing stays inside the SLA.

    Latency on a tier = base / (1 - utilisation). For the strictest extent placed there,
    utilisation may reach 1 - base / sla. Above that (times a safety headroom), extents are
    promoted to the cheapest faster tier that is allowed AND still has room, busiest first.
    Extents already sitting on a faster tier are preferred (keeping them costs no move), so
    the plan does not churn from hour to hour. Cold first, then warm.
    """
    for t in (2, 1):
        on_t = np.flatnonzero(choice == t)
        if len(on_t) == 0:
            continue
        u_max = 1.0 - eco.latency_ms[t] / sla_ms[on_t].min()
        cap = max(0.0, u_max) * eco.ios_capacity[t] * headroom
        load = expected_ios[on_t].sum()
        if load <= cap:
            continue
        sticky = np.where(current[on_t] < t, 2.0, 1.0)  # already faster: cheaper to keep
        room = eco.capacity[:t] - np.bincount(choice, minlength=4)[:t]
        for i in on_t[np.argsort(-expected_ios[on_t] * sticky, kind="stable")]:
            if load <= cap:
                break
            options = [u for u in range(t) if room[u] > 0 and np.isfinite(cost[i, u])]
            if not options:
                continue
            u = min(options, key=lambda k: cost[i, k])
            choice[i], room[u], load = u, room[u] - 1, load - expected_ios[i]


def plan(
    expected_ios: np.ndarray,  # expected I/Os per hour, per extent
    read_gb_per_io: np.ndarray,  # GB read per I/O, per extent
    sla_ms: np.ndarray,  # latency target, per extent
    current: np.ndarray,  # current tier, per extent
    held_hours: np.ndarray,  # hours spent on the current tier
    allowed: np.ndarray,  # (n, 4) from the policy guard
    confident: np.ndarray,  # False = abstain: stay put
    eco: TierEconomics,
    cfg: dict,
    knobs: dict | None = None,  # per-extent aggressiveness chosen by the bandit
    info: dict | None = None,  # filled with the solver report (which solver, objective, ms)
) -> np.ndarray:
    cfg = with_knobs(cfg, knobs)
    n = len(current)
    rows = np.arange(n)
    parts = cost_breakdown(expected_ios, read_gb_per_io, sla_ms, current, held_hours, eco, cfg)
    cost = parts["storage"] + parts["retrieval"] + parts["sla_risk"] + parts["move"]

    ok = allowed.copy()
    stay_ok = ok[rows, current]
    ok[~confident & stay_ok] = False  # abstain: only the current tier...
    ok[rows[~confident & stay_ok], current[~confident & stay_ok]] = True
    cost = np.where(ok, cost, np.inf)
    choice = cost.argmin(axis=1)

    for t in (0, 1):  # hot, then warm: demote the extents that lose the least
        on_t = np.flatnonzero(choice == t)
        excess = len(on_t) - eco.capacity[t]
        if excess <= 0:
            continue
        slower = cost[on_t][:, t + 1 :]
        alt = slower.min(axis=1)
        regret = alt - cost[on_t, t]
        movable = np.isfinite(alt)
        order = on_t[movable][np.argsort(regret[movable])][:excess]
        choice[order] = t + 1 + cost[order][:, t + 1 :].argmin(axis=1)
    if eco.ios_capacity is not None:
        _queue_repair(
            choice, cost, expected_ios, sla_ms, current, eco, cfg.get("queue_headroom", 0.7)
        )
    moving = np.flatnonzero(choice != current)
    budget = int(cfg["max_gb_moved_per_hour"] / GB)
    if len(moving) > budget:
        benefit = cost[moving, current[moving]] - cost[moving, choice[moving]]
        keep = moving[np.argsort(-benefit)[:budget]]
        revert = np.setdiff1d(moving, keep)
        choice[revert] = current[revert]
    if cfg.get("solver", "greedy") != "greedy":
        exact_choice, meta = exact.solve(
            cost, ok, expected_ios, sla_ms, current, eco, cfg, budget,
            cfg["solver"], cfg.get("solver_time_limit_s", 5.0),
        )  # fmt: skip
        if exact_choice is not None:
            meta["greedy_objective"] = round(
                exact.objective(choice, cost, expected_ios, sla_ms, eco, cfg), 6
            )
            choice = exact_choice
        if info is not None:
            info.update(meta)
    elif info is not None:
        info.update({"solver": "greedy", "status": "configured"})
    return choice
"""Live planning for the supervisor: latest predictions + current placement -> MovePlan.

Same economics as the twin strategy (planner/strategy.py), but for ONE hour and with the
real placement from the catalog. Besides the plan it returns, for every move:
  why   - one line for tables ("I/Os in the last 72 h = 0; pii: no archive")
  cards - the full decision trace the recommendation view shows, stage by stage:
          prediction (probabilities + feature drivers) -> policy + guards -> cost per tier -> decision
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ayojna.contracts import TIER_ORDER, Envelope, Move, MovePlan
from ayojna.planner.guards import NONE, apply_guards
from ayojna.planner.optimizer import (
    GB,
    TierEconomics,
    cost_breakdown,
    expected_ios,
    load_planner_config,
    plan,
    with_knobs,
)
from ayojna.policy.guard import allowed_tiers
from ayojna.twin.sim import Twin

TIERS = [t.value for t in TIER_ORDER]


def expected_load(twin: Twin, preds: pd.DataFrame, feats: pd.DataFrame, abstain_below: float):
    """Per extent (twin order): expected I/Os per hour, GB read per I/O, confident?, prediction."""
    # learned from labelled history: average I/Os per hour of a hot / warm extent
    rate = (feats.dropna(subset=["label"]).groupby("label")["future_acc"].mean() / 24).to_dict()
    cols = ["volume", "extent_id", "hour"]
    df = preds.merge(feats[cols + ["avg_io_size_24h", "read_ratio_24h", "acc_24h"]], on=cols)
    col = pd.DataFrame(
        {"volume": twin.volumes, "extent_id": twin.extent_ids, "col": np.arange(twin.n_extents)}
    )
    df = df.merge(col, on=["volume", "extent_id"])
    exp = np.zeros(twin.n_extents)
    rgb = np.zeros(twin.n_extents)
    conf = np.ones(twin.n_extents, dtype=bool)  # never-touched extents: surely idle
    c = df["col"].to_numpy()
    exp[c] = expected_ios(df["p_hot"], df["p_warm"], rate, df["acc_24h"])
    rgb[c] = df["avg_io_size_24h"] * df["read_ratio_24h"] / 1e9
    conf[c] = df["confidence"] >= abstain_below
    if "drivers" not in df:
        df["drivers"] = "[]"
    pred = {
        int(r.col): {
            "p_hot": round(float(r.p_hot), 3),
            "p_warm": round(float(r.p_warm), 3),
            "p_cold": round(float(r.p_cold), 3),
            "label": str(r.pred),
            "confidence": round(float(r.confidence), 3),
            "reasons": str(r.reasons),
            "drivers": json.loads(r.drivers or "[]"),
        }
        for r in df.itertuples()
    }
    return exp, rgb, conf, pred


def _risk(to_tier: int, confidence: float) -> str:
    if to_tier >= 2 and confidence < 0.75:
        return "high"  # demoting to a slow tier on an unsure forecast
    return "med" if to_tier >= 2 else "low"


def live_plan(
    twin: Twin,
    preds: pd.DataFrame,
    feats: pd.DataFrame,
    current: np.ndarray,
    since: np.ndarray,
    envelope: Envelope,
    guards: dict | None = None,
    knobs: dict | None = None,
    info: dict | None = None,
) -> tuple[MovePlan, dict[str, str], dict[str, dict]]:
    """knobs: per-extent aggressiveness from a promoted bandit; info: filled with the solver
    report (which solver ran, optimal or not, objective vs the greedy plan, ms)."""
    cfg = load_planner_config()
    hour, H = int(preds["hour"].max()), cfg["horizon_hours"]
    exp, rgb, conf, pred = expected_load(twin, preds, feats, cfg["abstain_below"])
    allowed, policy = allowed_tiers(twin.volumes, current)
    allowed, guard_note = apply_guards(allowed, current, twin.volumes, guards or NONE)
    eco = TierEconomics.from_twin(twin)
    held = hour - since
    solver: dict = {}
    choice = plan(exp, rgb, twin.sla_target_ms, current, held, allowed, conf, eco, cfg,
                  knobs=knobs, info=solver)  # fmt: skip
    if info is not None:
        info.update(solver)
    parts = cost_breakdown(exp, rgb, twin.sla_target_ms, current, held, eco, with_knobs(cfg, knobs))
    run_cost = parts["storage"] + parts["retrieval"] + parts["sla_risk"]  # per 24 h, no move
    per_month = twin.hours_per_month / H
    never = {"p_hot": 0, "p_warm": 0, "p_cold": 1, "label": "cold", "confidence": 1.0,
             "reasons": "never accessed in this trace", "drivers": []}  # fmt: skip
    moves, notes, cards = [], {}, {}
    for i in np.flatnonzero(choice != current):
        c, t = int(current[i]), int(choice[i])
        p = pred.get(int(i), never)
        saving = (run_cost[i, c] - run_cost[i, t]) * per_month
        m = Move(
            volume=str(twin.volumes[i]),
            extent_id=int(twin.extent_ids[i]),
            from_tier=TIER_ORDER[c],
            to_tier=TIER_ORDER[t],
            size_gb=GB,
            expected_saving_per_month=round(float(saving), 6),
            risk=_risk(t, p["confidence"]),
        )
        moves.append(m)
        key = m.idempotency_key(envelope.run_id)
        notes[key] = "; ".join(x for x in (p["reasons"], policy[i], guard_note[i]) if x)
        options = {
            TIERS[k]: (
                {
                    "storage": round(float(parts["storage"][i, k]), 5),
                    "retrieval": round(float(parts["retrieval"][i, k]), 5),
                    "sla_risk": round(float(parts["sla_risk"][i, k]), 5),
                    "move": round(float(parts["move"][i, k]), 5),
                }
                if allowed[i, k]
                else None
            )
            for k in range(4)
        }
        alts = [k for k in range(4) if allowed[i, k] and k != t]
        runner = min(alts, key=lambda k: run_cost[i, k] + parts["move"][i, k]) if alts else None
        hourly_gain = (run_cost[i, c] - run_cost[i, t]) / H
        cards[key] = {
            "group": m.group_key(),
            "prediction": p,
            "expected_ios_per_hour": round(float(exp[i]), 2),
            "policy": {
                "allowed": [TIERS[k] for k in range(4) if allowed[i, k]],
                "rules": policy[i] or "no restrictions",
                "guard": guard_note[i] or None,
            },
            "costs_24h": options,
            "decision": {
                "from": TIERS[c],
                "to": TIERS[t],
                "runner_up": TIERS[runner] if runner is not None else None,
                "saving_per_month": round(float(saving), 5),
                "payback_hours": (
                    round(float(parts["move_once"][i] / hourly_gain), 1)
                    if hourly_gain > 0
                    else None
                ),
                "why_not_cheaper": (
                    "protects the latency SLA" if t < c else "cheapest allowed tier over 24 h"
                ),
                "solver": solver.get("solver", "greedy"),
            },
        }
    moves.sort(key=lambda m: -m.expected_saving_per_month)
    mp = MovePlan(envelope=envelope, strategy="ayojna", hour=hour, moves=moves)
    return mp, notes, cards
"""Exact placement: one optimisation problem instead of greedy repairs.

    minimise   sum cost[i,t] * x[i,t]  +  penalties for any capacity / queue overflow
    subject to each extent on exactly one allowed tier
               hot and warm hold at most their capacity (in extents)
               each tier's expected I/O load stays inside its SLA-safe queue limit
               at most `budget` discretionary moves per hour (migration budget)

The queue limit depends on the strictest SLA placed on the tier, so there is one switch
per (tier, SLA level): if any extent of that SLA sits on the tier, its limit applies.

Solvers, best first: OR-Tools CP-SAT (pip install ortools) -> HiGHS MILP (ships with SciPy)
-> the greedy planner. Overflow is a penalised slack, so the problem is always feasible.
"""

from __future__ import annotations

import time

import numpy as np

COST_SCALE = 1e7  # CP-SAT needs integers: dollars -> 1e-7 dollar units
OVERFLOW_EXTENT = 1.0  # $ penalty per extent above a capacity limit (far above any real cost)


def _queue_levels(sla_ms, ok, eco, headroom):
    """(tier, sla, limit, members) for every tier with a queue model and SLA level present."""
    out = []
    if eco.ios_capacity is None:
        return out
    for t in (1, 2):
        for s in np.unique(sla_ms):
            members = np.flatnonzero(ok[:, t] & (sla_ms == s))
            if len(members):
                u_max = max(0.0, 1.0 - eco.latency_ms[t] / s)
                out.append((t, float(s), u_max * eco.ios_capacity[t] * headroom, members))
    return out


def solve(cost, ok, expected_ios, sla_ms, current, eco, cfg, budget, solver="auto", time_limit=5.0):
    """Returns (choice per extent, info) or (None, info) if no exact solver could answer."""
    t0 = time.perf_counter()
    headroom = cfg.get("queue_headroom", 0.7)
    q_pen = cfg["sla_penalty_per_io"] * cfg["horizon_hours"] * 10  # per I/O-hour over limit
    stay_ok = ok[np.arange(len(current)), current]
    levels = _queue_levels(sla_ms, ok, eco, headroom)
    order = ["ortools", "highs"] if solver == "auto" else [solver]
    errors = []
    for name in order:
        try:
            fn = {"ortools": _cpsat, "highs": _highs}[name]
            choice, status, obj = fn(cost, ok, expected_ios, current, stay_ok, eco, levels,
                                     budget, q_pen, time_limit)  # fmt: skip
            return choice, {
                "solver": name,
                "status": status,
                "objective": round(float(obj), 6),
                "ms": round(1000 * (time.perf_counter() - t0)),
                "variables": int(ok.sum()),
                "fallbacks": errors,
            }
        except Exception as exc:  # missing library, time-out without a solution, ...
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:160])
    return None, {"solver": "greedy", "status": "fallback", "fallbacks": errors,
                  "ms": round(1000 * (time.perf_counter() - t0))}  # fmt: skip


def objective(choice, cost, expected_ios, sla_ms, eco, cfg) -> float:
    """The exact model's objective for ANY placement (used to compare with the greedy plan)."""
    rows = np.arange(len(choice))
    total = float(cost[rows, choice].sum())
    counts = np.bincount(choice, minlength=4)
    total += OVERFLOW_EXTENT * sum(max(0, counts[t] - eco.capacity[t]) for t in (0, 1))
    q_pen = cfg["sla_penalty_per_io"] * cfg["horizon_hours"] * 10
    ok = np.zeros((len(choice), 4), dtype=bool)
    ok[rows, choice] = True
    over = {1: 0.0, 2: 0.0}  # one slack per tier: the worst SLA level on it
    for t, _s, limit, _m in _queue_levels(sla_ms, ok, eco, cfg.get("queue_headroom", 0.7)):
        over[t] = max(over[t], expected_ios[choice == t].sum() - limit)
    return total + q_pen * sum(over.values())


# ---------- HiGHS (SciPy) ----------
def _highs(cost, ok, ios, current, stay_ok, eco, levels, budget, q_pen, time_limit):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n = len(current)
    pairs = np.argwhere(ok)  # (k, 2): extent, tier
    k = len(pairs)
    var = -np.ones((n, 4), dtype=int)
    var[pairs[:, 0], pairs[:, 1]] = np.arange(k)
    n_cap, n_lvl = 2, len(levels)
    # variables: x (k) | capacity slack (2) | per-level switch y (L) | per-tier queue slack (2)
    nv = k + n_cap + n_lvl + 2
    c = np.zeros(nv)
    c[:k] = cost[pairs[:, 0], pairs[:, 1]]
    c[k : k + 2] = OVERFLOW_EXTENT
    c[k + 2 + n_lvl :] = q_pen
    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(r_cols, r_vals, lo, hi):
        r = len(lb)
        rows.extend([r] * len(r_cols))
        cols.extend(r_cols)
        vals.extend(r_vals)
        lb.append(lo)
        ub.append(hi)

    for i in range(n):  # one tier each
        v = var[i][var[i] >= 0]
        add(list(v), [1.0] * len(v), 1, 1)
    for t in (0, 1):  # capacity (soft)
        v = var[:, t][var[:, t] >= 0]
        add(list(v) + [k + t], [1.0] * len(v) + [-1.0], -np.inf, float(eco.capacity[t]))
    big = float(ios.sum()) + 1.0
    for j, (t, _s, limit, members) in enumerate(levels):
        y = k + n_cap + j
        for i in members:  # y >= x[i,t]
            add([var[i, t], y], [1.0, -1.0], -np.inf, 0)
        v = var[:, t][var[:, t] >= 0]
        e = np.flatnonzero(var[:, t] >= 0)
        slack = k + n_cap + n_lvl + (t - 1)
        # load_t - slack_t <= limit + big * (1 - y)
        add(list(v) + [slack, y], list(ios[e]) + [-1.0, big], -np.inf, limit + big)
    disc = np.flatnonzero(stay_ok)  # migration budget: discretionary moves only
    mv = [var[i, t] for i in disc for t in range(4) if t != current[i] and var[i, t] >= 0]
    if mv:
        add(mv, [1.0] * len(mv), -np.inf, float(budget))
    a = coo_matrix((vals, (rows, cols)), shape=(len(lb), nv)).tocsr()
    integrality = np.zeros(nv)
    integrality[:k] = 1
    integrality[k + n_cap : k + n_cap + n_lvl] = 1
    upper = np.full(nv, np.inf)
    upper[:k] = 1
    upper[k + n_cap : k + n_cap + n_lvl] = 1
    res = milp(c, constraints=LinearConstraint(a, lb, ub), integrality=integrality,
               bounds=Bounds(np.zeros(nv), upper),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})  # fmt: skip
    if res.x is None:
        raise RuntimeError(f"no solution ({res.message})")
    x = res.x[:k] > 0.5
    choice = current.copy()
    choice[pairs[x, 0]] = pairs[x, 1]
    return choice, "optimal" if res.status == 0 else "feasible (time limit)", res.fun


# ---------- OR-Tools CP-SAT ----------
def _cpsat(cost, ok, ios, current, stay_ok, eco, levels, budget, q_pen, time_limit):
    from ortools.sat.python import cp_model

    n = len(current)
    m = cp_model.CpModel()
    x = {(i, t): m.new_bool_var(f"x{i}_{t}") for i, t in np.argwhere(ok)}
    for i in range(n):
        m.add(sum(x[i, t] for t in range(4) if (i, t) in x) == 1)
    obj = [int(round(cost[i, t] * COST_SCALE)) * v for (i, t), v in x.items()]
    for t in (0, 1):
        over = m.new_int_var(0, n, f"cap_over{t}")
        m.add(sum(v for (i, tt), v in x.items() if tt == t) - over <= int(eco.capacity[t]))
        obj.append(int(OVERFLOW_EXTENT * COST_SCALE) * over)
    iosi = np.ceil(ios).astype(int)
    total = int(iosi.sum()) + 1
    q_over = {t: m.new_int_var(0, total, f"q_over{t}") for t in (1, 2)}
    for j, (t, _s, limit, members) in enumerate(levels):
        y = m.new_bool_var(f"y{j}")
        for i in members:
            m.add_implication(x[i, t], y)
        load = sum(int(iosi[i]) * v for (i, tt), v in x.items() if tt == t)
        m.add(load - q_over[t] <= int(limit)).only_enforce_if(y)
    for t in (1, 2):
        obj.append(int(round(q_pen * COST_SCALE)) * q_over[t])
    mv = [v for (i, t), v in x.items() if stay_ok[i] and t != current[i]]
    if mv:
        m.add(sum(mv) <= budget)
    m.minimize(sum(obj))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 8
    status = s.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT status {s.status_name(status)}")
    choice = current.copy()
    for (i, t), v in x.items():
        if s.value(v):
            choice[i] = t
    label = "optimal" if status == cp_model.OPTIMAL else "feasible (time limit)"
    return choice, label, s.objective_value / COST_SCALE
"""Step 11A: exact placement (OR-Tools / HiGHS) and the contextual bandit (RL)."""

import numpy as np
import pytest

from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.models.features import build_features
from ayojna.planner import bandit as rl
from ayojna.planner import exact
from ayojna.planner.evaluate import race_all
from ayojna.planner.optimizer import TierEconomics, load_planner_config, plan

CFG = {"horizon_hours": 24, "sla_penalty_per_io": 0.0005, "hysteresis_usd": 0.0005,
       "move_amortization_hours": 168, "abstain_below": 0.6, "max_gb_moved_per_hour": 5,
       "queue_headroom": 0.7}  # fmt: skip


def _instance(seed: int, n: int = 80):
    rng = np.random.default_rng(seed)
    eco = TierEconomics(
        price=np.array([0.1, 0.045, 0.0125, 0.002]),
        retrieval=np.array([0.0, 0.01, 0.03, 0.05]),
        min_hours=np.array([0, 720, 2160, 4320]),
        latency_ms=np.array([0.1, 0.5, 8.0, 3.6e6]),
        capacity=np.array([15, 40, n, n]),
        move_cost_per_gb=0.01,
        hours_per_month=730,
        ios_capacity=np.array([2e7, 4e4, 3e3, 1e3]),
    )
    ios = rng.choice([0, 0, 2, 20, 300, 3000], n).astype(float)
    sla = rng.choice([1.0, 10.0, 1000.0], n)
    allowed = np.ones((n, 4), dtype=bool)
    allowed[sla == 1.0, 2:] = False  # oltp: not below warm
    allowed[rng.random(n) < 0.2, 3] = False
    current = np.zeros(n, dtype=int)
    args = (ios, np.full(n, 1e-5), sla, current, np.full(n, 500.0), allowed, np.ones(n, bool), eco)
    return args, eco


def _plan(args, solver, **kw):
    info = {}
    choice = plan(*args, {**CFG, "solver": solver}, info=info, **kw)
    return choice, info


def test_exact_plan_is_never_worse_than_greedy_and_respects_every_limit():
    for seed in range(5):
        args, eco = _instance(seed)
        choice, info = _plan(args, "highs")
        assert info["solver"] == "highs" and info["status"] == "optimal"
        assert info["objective"] <= info["greedy_objective"] + 1e-9
        allowed, current = args[5], args[3]
        assert allowed[np.arange(len(choice)), choice].all()  # only allowed tiers
        assert (choice != current).sum() <= int(CFG["max_gb_moved_per_hour"] / 0.25)  # budget
        assert np.sum(choice == 0) <= eco.capacity[0] or (choice != current).sum() == 20


def test_unknown_solver_falls_back_to_the_greedy_plan():
    args, _ = _instance(1)
    greedy, _ = _plan(args, "greedy")
    choice, info = _plan(args, "no_such_solver")
    assert info["solver"] == "greedy" and info["fallbacks"]
    assert (choice == greedy).all()


def test_ortools_and_highs_agree_when_ortools_is_installed():
    pytest.importorskip("ortools")
    for seed in range(3):
        args, _ = _instance(seed)
        _, a = _plan(args, "ortools")
        _, b = _plan(args, "highs")
        assert a["solver"] == "ortools", a["fallbacks"]
        assert abs(a["objective"] - b["objective"]) <= 1e-3 * max(1.0, abs(b["objective"]))


def test_bandit_learns_which_arm_fits_which_context(tmp_path):
    b = rl.LinUCB(["cautious", "aggressive"], alpha=0.3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        x = rl.context(rng.gamma(2, 50, 48) * rng.choice([1, 100]), 10.0)
        busy = x[1] > 0.5
        arm = b.choose(x)
        b.update(arm, x, 1.0 if (arm == "aggressive") == busy else 0.0)
    quiet, busy = rl.context(np.full(48, 5.0), 10.0), rl.context(np.full(48, 5e4), 10.0)
    assert b.choose(quiet, explore=False) == "cautious"
    assert b.choose(busy, explore=False) == "aggressive"
    b.save(tmp_path, {"promoted": False})
    again = rl.LinUCB.load(tmp_path)
    assert again.choose(busy, explore=False) == "aggressive"
    live = rl.live_policy(tmp_path, {"v": np.full(48, 5e4)}, {"v": 10.0})
    assert live["mode"] == "shadow" and live["suggested"] == {"v": "aggressive"}


def test_cautious_knobs_never_move_more_than_balanced():
    args, _ = _instance(3)
    n = len(args[0])
    calm = {"hysteresis_x": np.full(n, 4.0), "amortization_hours": np.full(n, 336.0)}
    eager = {"hysteresis_x": np.full(n, 0.25), "amortization_hours": np.full(n, 72.0)}
    a, _ = _plan(args, "highs", knobs=calm)
    b, _ = _plan(args, "highs", knobs=eager)
    assert (a != args[3]).sum() <= (b != args[3]).sum()


def test_replay_gates_the_bandit_and_reports_the_solver(tmp_path):
    generate(tmp_path / "raw", days=5, seed=3)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    feats = build_features(eh)
    scored, summary, _ = race_all(eh, feats, str(tmp_path / "store"), save_bandit=True)
    d = summary.attrs["decision"]
    assert "ayojna" in summary.index and set(d["solver"]["used"]) <= {"ortools", "highs", "greedy"}
    if load_planner_config()["bandit"]["enabled"]:
        b = d["bandit"]
        assert isinstance(b["promoted"], bool)
        assert d["driven_by"] == ("bandit" if b["promoted"] else "static settings")
        assert b["validation_hours"][1] < scored["hour"].min()  # validation never sees test
        assert (tmp_path / "store" / rl.FILE).exists()


def test_reward_is_zero_when_nothing_moved(tmp_path):
    from ayojna.twin.sim import Twin

    generate(tmp_path / "raw", days=2, seed=1)
    twin = Twin.from_extent_hourly(build(tmp_path / "raw", tmp_path / "eh.csv"))
    p = np.zeros(twin.n_extents, dtype=int)
    assert rl.reward(twin, np.arange(5), p, p, range(10, 20), CFG) == 0.0
    q = p.copy()
    q[:5] = 2  # idle-ish extents to cold: a real saving or loss, but bounded
    assert -1.0 <= rl.reward(twin, np.arange(5), p, q, range(10, 20), CFG) <= 1.0
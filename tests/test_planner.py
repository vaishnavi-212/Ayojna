import json

import numpy as np
import pytest

from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.models.features import build_features, time_split
from ayojna.models.hotness import train
from ayojna.models.predictor import LATEST
from ayojna.planner.evaluate import evaluate
from ayojna.planner.optimizer import TierEconomics, plan
from ayojna.policy.guard import allowed_tiers

ECO = TierEconomics(
    price=np.array([0.1, 0.045, 0.0125, 0.002]),
    retrieval=np.array([0, 0, 0.01, 0.02]),
    min_hours=np.array([0, 0, 720, 2160]),
    latency_ms=np.array([0.1, 0.5, 8, 3.6e6]),
    capacity=np.array([1, 10, 10, 10]),
    move_cost_per_gb=0.001,
    hours_per_month=730,
)
CFG = {
    "horizon_hours": 24,
    "sla_penalty_per_io": 0.0005,
    "hysteresis_usd": 0.0005,
    "move_amortization_hours": 168,
    "max_gb_moved_per_hour": 50,
}


def _plan(ios, current, allowed=None, confident=None, **kw):
    n = len(ios)
    return plan(
        np.array(ios, float),
        np.zeros(n),
        np.full(n, kw.get("sla", 1.0)),
        np.array(current),
        np.full(n, 10_000.0),
        np.ones((n, 4), bool) if allowed is None else allowed,
        np.ones(n, bool) if confident is None else confident,
        kw.get("eco", ECO),
        kw.get("cfg", CFG),
    )


def test_busy_stays_fast_idle_goes_cheap():
    assert _plan([100, 0], [0, 0], sla=0.2).tolist() == [0, 3]  # busy needs hot, idle -> archive


def test_hot_capacity_is_respected():
    choice = _plan([100, 90, 80], [0, 0, 0], sla=0.2)  # only hot meets 0.2 ms
    assert (choice == 0).sum() == 1 and choice[0] == 0  # the busiest keeps the single hot slot


def test_abstain_keeps_the_current_tier():
    assert _plan([0, 0], [0, 0], confident=np.array([False, True])).tolist() == [0, 3]


def test_never_picks_a_forbidden_tier():
    allowed = np.ones((1, 4), bool)
    allowed[0, 3] = False
    assert _plan([0], [0], allowed=allowed).tolist() == [2]


def test_migration_budget_limits_moves():
    cfg = {**CFG, "max_gb_moved_per_hour": 0.5}  # 2 extents of 0.25 GB
    assert (_plan([0] * 5, [0] * 5, cfg=cfg) != 0).sum() == 2


def test_policy_rules():
    # legal hold, financial + oltp, pii, backup
    vols = np.array(["src1_2", "web_0", "usr_0", "prn_0"])
    mask, reasons = allowed_tiers(vols, np.array([1, 0, 0, 0]))
    assert mask[0].tolist() == [False, True, False, False] and "legal hold" in reasons[0]
    assert mask[1].tolist() == [True, True, False, False]  # oltp floor = warm
    assert mask[2].tolist() == [True, True, True, False]  # pii: no archive
    assert mask[3].all()


def test_ayojna_beats_baselines_and_stays_compliant(tmp_path):
    generate(tmp_path / "raw", days=6, seed=21)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    feats = build_features(eh, hot_min=50, horizon=24)
    tr, _ = time_split(feats, test_hours=24, horizon=24)
    store = tmp_path / "store"
    path = train(tr).save(store)
    (store / LATEST).write_text(json.dumps({"file": path.name}))
    s = evaluate(eh, feats, str(store))
    rules = s.drop(index=["ayojna", "all_hot"])
    best_baseline = rules[rules["compliance_pct"] >= 100 - 1e-9]["monthly_cost"].min()  # fair: compliant
    assert s.loc["ayojna", "monthly_cost"] < best_baseline
    assert s.loc["ayojna", "compliance_pct"] == pytest.approx(100)
    assert s.loc["ayojna", "hours_over_hot_capacity"] == 0
    assert s.loc["ayojna", "sla_met_pct"] >= 99

def test_busy_tier_is_relieved_before_queueing_breaks_the_sla():
    from dataclasses import replace

    allowed = np.ones((3, 4), bool)
    allowed[:, 3] = False  # no archive
    plain = _plan([30, 30, 30], [2, 2, 2], allowed=allowed, sla=10.0)
    assert plain.tolist() == [2, 2, 2]  # base latency alone: cold (8 ms) looks fine
    eco = replace(ECO, ios_capacity=np.array([1e9, 1e9, 300.0, 10.0]))
    cfg = {**CFG, "queue_headroom": 0.7}  # cold may carry 0.2 x 300 x 0.7 = 42 I/Os per hour
    choice = _plan([30, 30, 30], [2, 2, 2], allowed=allowed, sla=10.0, eco=eco, cfg=cfg)
    assert (choice == 2).sum() == 1 and (choice < 2).sum() == 2  # two busiest promoted



def test_queue_relief_never_overfills_a_faster_tier():
    from dataclasses import replace

    allowed = np.ones((4, 4), bool)
    allowed[:, 3] = False
    eco = replace(
        ECO, capacity=np.array([1, 1, 10, 10]), ios_capacity=np.array([1e9, 1e9, 100.0, 10.0])
    )
    cfg = {**CFG, "queue_headroom": 0.7}  # cold may carry only 14 I/Os per hour
    choice = _plan([30, 30, 30, 30], [2, 2, 2, 2], allowed=allowed, sla=10.0, eco=eco, cfg=cfg)
    assert (choice == 0).sum() <= 1 and (choice == 1).sum() <= 1  # room for one each
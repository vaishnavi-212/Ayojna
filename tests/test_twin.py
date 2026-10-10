import numpy as np
import pandas as pd
import pytest

from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.twin.race import race
from ayojna.twin.sim import HOT, Twin
from ayojna.twin.strategies import AccessTimer, AgeRule, AllHot, last_access_hour


def _eh(rows):
    base = {
        "read_bytes": 0,
        "write_bytes": 0,
        "avg_io_size": 4096.0,
        "rand_ratio": 1.0,
        "writes": 0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


class Peeker:
    """A cheating strategy: records how much history it was shown."""

    name = "peeker"

    def __init__(self):
        self.seen = []

    def decide(self, view):
        self.seen.append(view.ios_past.shape[0])
        return view.placement


def test_strategies_never_see_the_future():
    twin = Twin.from_extent_hourly(
        _eh([{"volume": "web_0", "extent_id": 0, "hour": h, "reads": 1} for h in range(5)])
    )
    p = Peeker()
    twin.run(p)
    assert p.seen == [0, 1, 2, 3, 4]


def test_all_hot_storage_cost_matches_price():
    twin = Twin.from_extent_hourly(
        _eh([{"volume": "web_0", "extent_id": e, "hour": 0, "reads": 1} for e in range(4)])
    )
    m = twin.run(AllHot())
    expected = 4 * 0.25 * twin.price[HOT] / twin.hours_per_month
    assert m.loc[0, "storage_cost"] == pytest.approx(expected)


def test_age_rule_demotes_idle_extents_and_counts_moves():
    rows = [{"volume": "web_0", "extent_id": 0, "hour": h, "reads": 5} for h in range(100)]
    rows += [{"volume": "web_0", "extent_id": 1, "hour": 0, "reads": 5}]  # touched once, then idle
    twin = Twin.from_extent_hourly(_eh(rows))
    m = twin.run(AgeRule(24, 72))
    assert m["gb_moved"].sum() == pytest.approx(0.5)  # extent 1: hot->warm, warm->cold
    assert m["cost"].sum() < twin.run(AllHot())["cost"].sum()


def test_moving_legal_hold_data_breaks_compliance():
    rows = [
        {"volume": "src1_2", "extent_id": 0, "hour": 0, "reads": 1},  # src1_2 has legal_hold
        {"volume": "src1_2", "extent_id": 0, "hour": 99, "reads": 1},
    ]
    twin = Twin.from_extent_hourly(_eh(rows))
    assert twin.run(AgeRule(24, 72))["compliance_pct"].min() < 100
    assert twin.run(AllHot())["compliance_pct"].min() == 100


def test_last_access_hour():
    rows = [
        {"volume": "v_0", "extent_id": 0, "hour": 2, "reads": 1},
        {"volume": "v_0", "extent_id": 1, "hour": 6, "reads": 1},
        {"volume": "v_0", "extent_id": 0, "hour": 8, "reads": 1},
    ]
    twin = Twin.from_extent_hourly(_eh(rows))
    seen = {}

    class Spy:
        name = "spy"

        def decide(self, view):
            seen[view.hour] = last_access_hour(view).tolist()
            return view.placement

    twin.run(Spy())
    assert seen[5] == [2, -1] and seen[7] == [2, 6]


def test_race_on_synthetic_traces(tmp_path):
    generate(tmp_path / "raw", days=4, seed=5)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    metrics, summary = race(eh)
    assert set(summary.index) == {"all_hot", "age_rule", "access_timer", "lru", "lfu",
                                  "lru+policy", "lfu+policy"}  # fmt: skip
    assert summary.loc["all_hot", "saving_vs_all_hot_pct"] == pytest.approx(0)
    assert summary.loc["access_timer", "monthly_cost"] < summary.loc["all_hot", "monthly_cost"]
    assert np.isfinite(metrics["p95_latency_ms"]).all()


def test_access_timer_only_uses_hot_and_warm():
    rows = [
        {"volume": "web_0", "extent_id": e, "hour": h, "reads": 1}
        for e in range(3)
        for h in range(0, 60, 1 + e * 30)
    ]
    twin = Twin.from_extent_hourly(_eh(rows))
    seen = set()

    class Wrap(AccessTimer):
        def decide(self, view):
            out = super().decide(view)
            seen.update(out.tolist())
            return out

    twin.run(Wrap(24))
    assert seen <= {HOT, 1}


def test_idle_extents_are_part_of_the_volume():
    twin = Twin.from_extent_hourly(
        _eh([{"volume": "web_0", "extent_id": 9, "hour": 0, "reads": 1}])
    )
    assert twin.n_extents == 10  # extents 0-8 never touched, but they exist and cost money

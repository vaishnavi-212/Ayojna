"""Run every strategy on the same replayed trace and compare them.

Run:  python -m ayojna.twin.race --inp data/lake/extent_hourly.parquet --out data/lake/race.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from ayojna.io import read_table, write_table
from ayojna.twin.sim import Twin, load_twin_config
from ayojna.twin.strategies import (
    AccessTimer,
    AgeRule,
    AllHot,
    LfuCapacity,
    LruCapacity,
    PolicyAware,
)


def default_strategies() -> list:
    cfg = load_twin_config()["strategies"]
    lfu = cfg.get("lfu", {})
    return [AllHot(), AgeRule(**cfg["age_rule"]), AccessTimer(**cfg["access_timer"]),
            LruCapacity(), LfuCapacity(**lfu),
            PolicyAware(LruCapacity()), PolicyAware(LfuCapacity(**lfu))]  # fmt: skip


def summarize(metrics: pd.DataFrame, hours_per_month: float) -> pd.DataFrame:
    n_hours = metrics["hour"].nunique()
    s = metrics.groupby("strategy", sort=False).agg(
        cost=("cost", "sum"),
        sla_met_pct=("sla_met_pct", "mean"),
        gb_moved=("gb_moved", "sum"),
        compliance_pct=("compliance_pct", "min"),
        hours_over_hot_capacity=("hot_over_capacity", "sum"),
    )
    s["monthly_cost"] = s["cost"] * hours_per_month / n_hours
    reference = (
        s.loc["all_hot", "monthly_cost"] if "all_hot" in s.index else s["monthly_cost"].max()
    )
    s["saving_vs_all_hot_pct"] = 100 * (1 - s["monthly_cost"] / reference)
    return s[
        [
            "monthly_cost",
            "saving_vs_all_hot_pct",
            "sla_met_pct",
            "gb_moved",
            "compliance_pct",
            "hours_over_hot_capacity",
        ]
    ]


def race(eh: pd.DataFrame, strategies=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    twin = Twin.from_extent_hourly(eh)
    metrics = pd.concat(
        [twin.run(s) for s in (strategies or default_strategies())], ignore_index=True
    )
    return metrics, summarize(metrics, twin.hours_per_month)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--out", default="data/lake/race.parquet")
    a = ap.parse_args()
    metrics, summary = race(read_table(a.inp))
    write_table(metrics, a.out)
    pd.set_option("display.width", 140)
    print(summary.round(3).to_string())
    print(f"\nper-hour metrics -> {a.out}")

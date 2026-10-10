"""Headline result: Ayojna vs every baseline on the same trace, scored on unseen hours.

Run:
    python -m ayojna.planner.evaluate --eh data/lake/extent_hourly.parquet \
        --features data/lake/features.parquet --store models_store

The RL bandit is gated like the hotness model, with three windows that never overlap:
  hours before validation   the bandit explores and learns (digital twin = its training)
  validation (24 h)         bandit vs static settings, exploiting only -> promote or not
  test (unseen)             reported; the "ayojna" row is whichever variant was promoted
"""

from __future__ import annotations

import argparse

import pandas as pd

from ayojna.io import read_table
from ayojna.models.features import load_label_config, time_split
from ayojna.models.predictor import predict_hotness
from ayojna.planner.guards import guard_tables
from ayojna.planner.optimizer import load_planner_config
from ayojna.planner.strategy import AyojnaStrategy
from ayojna.twin.race import default_strategies, summarize
from ayojna.twin.sim import Twin


def _window(metrics: pd.DataFrame, name: str, lo: int, hi: int) -> dict:
    m = metrics[(metrics["strategy"] == name) & metrics["hour"].between(lo, hi - 1)]
    return {
        "cost": round(float(m["cost"].sum()), 6),
        "sla_met_pct": round(float(m["sla_met_pct"].mean()), 4),
    }


def race_all(
    eh: pd.DataFrame,
    feats: pd.DataFrame,
    store: str,
    with_guards: bool = True,
    save_bandit: bool = False,
):
    """All strategies on one trace -> (per-hour metrics on unseen hours, summary, note).

    summary.attrs: "guards" (how often the forecast / anomaly guards acted) and "decision"
    (solver report, bandit learning and the static-vs-bandit validation that gated it).
    """
    cfg = load_label_config()
    _, test = time_split(feats, cfg["split"]["test_hours"], cfg["labels"]["horizon_hours"])
    start = int(test["hour"].min())
    val_start = max(1, start - cfg["split"]["test_hours"])
    twin = Twin.from_extent_hourly(eh)
    preds, source, status, note = predict_hotness(feats, store)
    guards, counts = None, {}
    if with_guards:
        guards, counts = guard_tables(eh, store, cfg, start, twin.n_hours - 1)

    static = AyojnaStrategy(twin, preds, feats, guards, use_bandit=False)
    runs = [twin.run(s) for s in default_strategies()] + [twin.run(static)]
    decision: dict = {"solver": static.report(start)["solver"], "driven_by": "static settings"}
    if load_planner_config().get("bandit", {}).get("enabled", False):
        rl = AyojnaStrategy(twin, preds, feats, guards, use_bandit=True, explore_before=val_start)
        rl.name = "ayojna_rl"
        rl_runs = twin.run(rl)
        both = pd.concat([runs[-1], rl_runs], ignore_index=True)
        v_static = _window(both, "ayojna", val_start, start)
        v_rl = _window(both, "ayojna_rl", val_start, start)
        promoted = v_rl["cost"] < v_static["cost"] and v_rl["sla_met_pct"] >= v_static["sla_met_pct"]
        everything = pd.concat(runs + [rl_runs], ignore_index=True)  # all_hot = the reference
        test_rows = summarize(everything[everything["hour"] >= start], twin.hours_per_month)
        decision["bandit"] = {
            **rl.report(start)["bandit"],
            "promoted": bool(promoted),
            "validation_hours": [val_start, start - 1],
            "validation": {"static": v_static, "bandit": v_rl},
            "test": {
                "static": test_rows.loc["ayojna"].round(4).to_dict(),
                "bandit": test_rows.loc["ayojna_rl"].round(4).to_dict(),
            },
        }
        if promoted:  # the bandit drives Ayojna from now on
            runs[-1] = rl_runs.assign(strategy="ayojna")
            decision["solver"] = rl.report(start)["solver"]
            decision["driven_by"] = "bandit"
        if save_bandit:
            rl.bandit.save(store, {"promoted": bool(promoted), "validation": decision["bandit"]["validation"]})

    metrics = pd.concat(runs, ignore_index=True)
    scored = metrics[metrics["hour"] >= start].reset_index(drop=True)
    note = f"{source.value} / {status.value} ({note})"
    summary = summarize(scored, twin.hours_per_month)
    summary.attrs["guards"] = counts
    summary.attrs["decision"] = decision
    return scored, summary, note


def evaluate(eh: pd.DataFrame, feats: pd.DataFrame, store: str) -> pd.DataFrame:
    scored, summary, note = race_all(eh, feats, store)
    print(f"predictions: {note}")
    print(f"scored on unseen hours {int(scored['hour'].min())}-{int(scored['hour'].max())}")
    print(f"guards: {summary.attrs.get('guards', {})}")
    d = summary.attrs["decision"]
    print(f"solver: {d['solver']['used']} (optimal hours {d['solver']['optimal_hours']})")
    if "bandit" in d:
        b = d["bandit"]
        print(f"bandit: {'PROMOTED' if b['promoted'] else 'shadow mode (not promoted)'}; "
              f"validation cost static {b['validation']['static']['cost']} "
              f"vs bandit {b['validation']['bandit']['cost']}")  # fmt: skip
    print()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    a = ap.parse_args()
    pd.set_option("display.width", 160)
    print(evaluate(read_table(a.eh), read_table(a.features), a.store).round(3).to_string())
"""Headline result: Ayojna vs every baseline on the same trace, scored on unseen hours.

Run:
    python -m ayojna.planner.evaluate --eh data/lake/extent_hourly.parquet \
        --features data/lake/features.parquet --store models_store
"""

from __future__ import annotations

import argparse

import pandas as pd
from ayojna.io import read_table
from ayojna.models.features import load_label_config, time_split
from ayojna.models.predictor import predict_hotness
from ayojna.planner.guards import guard_tables
from ayojna.planner.strategy import AyojnaStrategy
from ayojna.twin.race import default_strategies, summarize
from ayojna.twin.sim import Twin


def race_all(eh: pd.DataFrame, feats: pd.DataFrame, store: str, with_guards: bool = True):
    """All strategies on one trace -> (per-hour metrics on unseen hours, summary, note).

    The note also reports how often the forecast and anomaly guards acted (if enabled).
    """
    cfg = load_label_config()
    _, test = time_split(feats, cfg["split"]["test_hours"], cfg["labels"]["horizon_hours"])
    start = int(test["hour"].min())
    twin = Twin.from_extent_hourly(eh)
    preds, source, status, note = predict_hotness(feats, store)
    guards, counts = None, {}
    if with_guards:
        guards, counts = guard_tables(eh, store, cfg, start, twin.n_hours - 1)
    strategies = default_strategies() + [AyojnaStrategy(twin, preds, feats, guards)]
    metrics = pd.concat([twin.run(s) for s in strategies], ignore_index=True)
    scored = metrics[metrics["hour"] >= start].reset_index(drop=True)
    note = f"{source.value} / {status.value} ({note})"
    summary = summarize(scored, twin.hours_per_month)
    summary.attrs["guards"] = counts
    return scored, summary, note


def evaluate(eh: pd.DataFrame, feats: pd.DataFrame, store: str) -> pd.DataFrame:
    scored, summary, note = race_all(eh, feats, store)
    print(f"predictions: {note}")
    print(f"scored on unseen hours {int(scored['hour'].min())}-{int(scored['hour'].max())}")
    print(f"guards: {summary.attrs.get('guards', {})}\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    a = ap.parse_args()
    pd.set_option("display.width", 160)
    print(evaluate(read_table(a.eh), read_table(a.features), a.store).round(3).to_string())
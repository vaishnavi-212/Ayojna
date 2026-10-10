"""Robustness: is the headline result stable, or one lucky replay?

    python -m ayojna.validate.robustness --runs 5 --eh data/lake/extent_hourly.parquet \
        --features data/lake/features.parquet --store models_store

Planning is sequential: a tiny difference today (a tie broken the other way) changes what is
on each tier tomorrow, so one replay is one sample. This replays the same trace N times with
noise of one part in a billion on the measured features (far below any real precision) and
reports the median and the range. Run 0 is the unperturbed replay. The KPI report judges cost
on the MEDIAN when this file exists. Writes data/lake/robustness.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ayojna.io import read_table
from ayojna.planner.evaluate import race_all
from ayojna.settings import DATA_DIR

NOISY = ["trend_24_vs_72", "read_ratio_24h", "avg_io_size_24h", "rand_ratio_24h"]


def _stats(xs: list[float]) -> dict:
    return {"median": round(float(np.median(xs)), 2), "min": round(float(min(xs)), 2),
            "max": round(float(max(xs)), 2), "runs": [round(float(x), 2) for x in xs]}  # fmt: skip


def robustness(eh: pd.DataFrame, feats: pd.DataFrame, store: str, runs: int = 5,
               noise: float = 1e-9) -> dict:  # fmt: skip
    saving, cut, sla, moved = [], [], [], []
    for seed in range(runs):
        f = feats.copy()
        if seed:
            rng = np.random.default_rng(seed)
            for c in NOISY:
                f[c] = f[c] * (1 + rng.normal(0, noise, len(f)))
        with contextlib.redirect_stdout(io.StringIO()):
            _, s, _ = race_all(eh, f, store)
        ay = s.loc["ayojna"]
        rules = s.drop(index=["ayojna", "all_hot"], errors="ignore")
        fair = rules[rules["compliance_pct"] >= 100 - 1e-9]["monthly_cost"].min()
        saving.append(ay["saving_vs_all_hot_pct"])
        cut.append(100 * (1 - ay["monthly_cost"] / fair))
        sla.append(ay["sla_met_pct"])
        moved.append(ay["gb_moved"])
        print(f"run {seed}: saving {saving[-1]:.1f}%, vs best compliant rule {cut[-1]:.1f}%, "
              f"SLA {sla[-1]:.2f}%, {moved[-1]} GB moved")  # fmt: skip
    return {"runs": runs, "noise": noise, "saving_vs_all_hot_pct": _stats(saving),
            "cut_vs_best_compliant_rule_pct": _stats(cut), "sla_met_pct": _stats(sla),
            "gb_moved": _stats(moved)}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--lake", default=str(DATA_DIR / "lake"))
    a = ap.parse_args()
    rep = robustness(read_table(a.eh), read_table(a.features), a.store, a.runs)
    Path(a.lake).mkdir(parents=True, exist_ok=True)
    Path(a.lake, "robustness.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    c = rep["cut_vs_best_compliant_rule_pct"]
    print(f"\nvs best compliant rule: median {c['median']}% (range {c['min']}-{c['max']}%) "
          f"over {a.runs} replays -> {a.lake}/robustness.json")
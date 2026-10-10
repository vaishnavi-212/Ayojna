"""Score every strategy once and save it for the dashboard (the twin is too slow per request).

Run:  python -m ayojna.api.snapshot --eh data/lake/extent_hourly.parquet \
          --features data/lake/features.parquet --store models_store
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ayojna.io import read_table
from ayojna.planner.evaluate import race_all


def snapshot(eh_path: str, features_path: str, store: str, out: str) -> dict:
    scored, summary, note = race_all(read_table(eh_path), read_table(features_path), store)
    cum = scored.pivot(index="hour", columns="strategy", values="cost").cumsum()
    board = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": note,
        "guards": summary.attrs.get("guards", {}),
        "scored_hours": [int(scored["hour"].min()), int(scored["hour"].max())],
        "summary": summary.round(4).reset_index().to_dict(orient="records"),
        "cumulative_cost": {
            "hours": [int(h) for h in cum.index],
            **{s: [round(float(v), 4) for v in cum[s]] for s in summary.index},
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(board, indent=1), encoding="utf-8")
    return board


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--out", default="data/lake/scoreboard.json")
    a = ap.parse_args()
    b = snapshot(a.eh, a.features, a.store, a.out)
    for row in b["summary"]:
        print(
            f"{row['strategy']:<13} ${row['monthly_cost']:>7.3f}/month  "
            f"saving {row['saving_vs_all_hot_pct']:>5.1f}%  SLA {row['sla_met_pct']:.2f}%"
        )
    print(f"guards in the replay: {b['guards']}")
    print(f"-> {a.out}")
"""Train and score the whole intelligence layer (slide 4: hotness · forecast · anomaly).

    python -m ayojna.models.train_all --eh data/lake/extent_hourly.parquet \
        --features data/lake/features.parquet --store models_store

1. hotness   model zoo leaderboard, champion promoted only if it beats the rules
2. forecast  MAPE backtest of every forecaster (24 h totals, past-only forecasts)
3. anomaly   Isolation Forest learns each volume's normal from the training hours
Writes data/lake/intel_report.json for the dashboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ayojna.io import read_table
from ayojna.models import anomaly, forecast, train_hotness
from ayojna.models.features import load_label_config, time_split


def main(eh_path: str, features_path: str, store: str, report: str) -> dict:
    cfg = load_label_config()
    print("== 1/3 hotness: model zoo ==")
    train_hotness.main(features_path, store)
    hot = json.loads(Path(store, train_hotness.LEADERBOARD).read_text(encoding="utf-8"))

    eh, feats = read_table(eh_path), read_table(features_path)
    fc = cfg["forecast"]
    print("\n== 2/3 capacity & I/O forecast: backtest ==")
    fboard = forecast.backtest(eh, ["naive", "seasonal_ewma", "prophet"],
                               fc["backtest_origins"], fc["horizon_hours"])  # fmt: skip
    print(f"{'model':<15}{'MAPE I/O':>10}{'MAPE capacity':>15}")
    for name, r in fboard.items():
        if r["status"] == "not installed":
            print(f"{name:<15}  not installed (pip install {name})")
        else:
            print(f"{name:<15}{r['mape_io_pct']:>9}%{r['mape_capacity_pct']:>14}%  {r['status']}")
    print(f"live model: {fc['model']} (falls back to seasonal_ewma per volume)")

    print("\n== 3/3 anomaly guard ==")
    _, te = time_split(feats, cfg["split"]["test_hours"], cfg["labels"]["horizon_hours"])
    until = int(te["hour"].min()) - 1  # never learn "normal" from the scored hours
    vf = anomaly.volume_features(eh)
    model = anomaly.train(vf, until, cfg["anomaly"]["contamination"], cfg["anomaly"]["z_max"])
    later = model.score(vf[vf["hour"] > until])
    flagged = later[later["anomaly"]]
    model.metrics.update(
        {
            "scored_hours": [until + 1, int(vf["hour"].max())],
            "scored_flag_rate": round(float(later["anomaly"].mean()), 4),
            "flags_by_volume": flagged.groupby("volume").size().to_dict(),
            "examples": flagged.sort_values("score", ascending=False)
            .head(5)[["volume", "hour", "reason"]]
            .to_dict(orient="records"),
        }
    )
    model.save(store)
    m = model.metrics
    print(f"Isolation Forest trained on hours {m['train_hours'][0]}-{m['train_hours'][1]} "
          f"({m['train_flag_rate']:.1%} flagged there)")  # fmt: skip
    print(f"later hours {m['scored_hours'][0]}-{m['scored_hours'][1]}: "
          f"{m['scored_flag_rate']:.1%} of volume-hours flagged {m['flags_by_volume']}")
    for e in m["examples"][:3]:
        print(f"  {e['volume']} hour {e['hour']}: {e['reason']}")

    out = {"hotness": hot, "forecast": {"live_model": fc["model"], "board": fboard},
           "anomaly": model.metrics}  # fmt: skip
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\n-> {report}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--report", default="data/lake/intel_report.json")
    a = ap.parse_args()
    main(a.eh, a.features, a.store, a.report)
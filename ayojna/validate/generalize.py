"""Generalization: does Ayojna work on workloads it was never trained on? (slide 3)

Train the hotness model on one set of volumes, then predict and plan for OTHER volumes:
  cross-volume   MSR volumes split in two (default)            -> runs today
  cross-dataset  all MSR volumes -> Alibaba volumes (--target)  -> once the trace is ingested

    python -m ayojna.validate.generalize --eh data/lake/extent_hourly.parquet
    python -m ayojna.validate.generalize --eh data/lake/extent_hourly.parquet \
        --target data/lake/alibaba_eh.parquet

The target volumes are unseen twice: other volumes AND only their test hours are scored.
Nothing is tuned on the target. Writes data/lake/generalization.json and .md.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import pandas as pd

from ayojna.io import read_table, write_table
from ayojna.models import train_hotness
from ayojna.models.baseline import evaluate, predict_rule
from ayojna.models.features import build_features, load_label_config, time_split
from ayojna.models.predictor import predict_hotness
from ayojna.planner.evaluate import race_all
from ayojna.settings import DATA_DIR


def split_volumes(volumes: list[str]) -> tuple[list[str], list[str]]:
    """Every other volume (sorted) trains, the rest is held out: no hand-picking."""
    vs = sorted(volumes)
    return vs[0::2], vs[1::2]


def generalize(train_eh: pd.DataFrame, test_eh: pd.DataFrame, label: str, workdir: Path) -> dict:
    cfg = load_label_config()
    hot_min, horizon = cfg["labels"]["hot_min_accesses"], cfg["labels"]["horizon_hours"]
    f_train = build_features(train_eh, hot_min, horizon)
    f_test = build_features(test_eh, hot_min, horizon)
    write_table(f_train, workdir / "train_features.csv")
    store = workdir / "store"
    with contextlib.redirect_stdout(io.StringIO()):  # the leaderboard prints a lot
        train_hotness.main(str(workdir / "train_features.csv"), str(store))
    board = json.loads((store / train_hotness.LEADERBOARD).read_text(encoding="utf-8"))

    _, te = time_split(f_test, cfg["split"]["test_hours"], horizon)
    pred, source, _, note = predict_hotness(te, store, hot_min)
    ml = evaluate(te["label"], pred["pred"])
    rule = evaluate(te["label"], predict_rule(te, hot_min)["pred"])

    with contextlib.redirect_stdout(io.StringIO()):
        scored, summary, _ = race_all(test_eh, f_test, str(store))
    rules = summary.drop(index=["ayojna", "all_hot"], errors="ignore")
    fair = rules[rules["compliance_pct"] >= 100 - 1e-9]
    ay = summary.loc["ayojna"]
    best = fair["monthly_cost"].idxmin() if len(fair) else None
    return {
        "experiment": label,
        "train_volumes": sorted(train_eh["volume"].unique().tolist()),
        "test_volumes": sorted(test_eh["volume"].unique().tolist()),
        "model": f"{board['champion']} ({source.value}: {note})",
        "scored_hours": [int(te["hour"].min()), int(te["hour"].max())],
        "hotness": {"ml_macro_f1": round(ml["macro_f1"], 3), "rule_macro_f1": round(rule["macro_f1"], 3),
                    "label_mix": ml["support"]},  # fmt: skip
        "planner": {
            "saving_vs_all_hot_pct": round(float(ay["saving_vs_all_hot_pct"]), 1),
            "sla_met_pct": round(float(ay["sla_met_pct"]), 3),
            "compliance_pct": round(float(ay["compliance_pct"]), 1),
            "best_compliant_rule": best,
            "cut_vs_best_compliant_rule_pct": round(
                100 * (1 - ay["monthly_cost"] / fair.loc[best, "monthly_cost"]), 1) if best else None,
        },  # fmt: skip
        "table": summary.round(3).reset_index().to_dict(orient="records"),
    }


def to_markdown(results: list[dict]) -> str:
    lines = ["| Experiment | Trained on | Tested on (unseen) | Hotness F1 (rule) | Saving vs all-hot "
             "| vs best compliant rule | SLA | Compliance |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in results:
        h, p = r["hotness"], r["planner"]
        c = p["cut_vs_best_compliant_rule_pct"]
        vs = "-" if c is None else f"{abs(c)}% {'cheaper' if c >= 0 else 'dearer'} than {p['best_compliant_rule']}"
        lines.append(
            f"| {r['experiment']} | {', '.join(r['train_volumes'])} | {', '.join(r['test_volumes'])} "
            f"| {h['ml_macro_f1']} ({h['rule_macro_f1']}) | {p['saving_vs_all_hot_pct']}% "
            f"| {vs} "
            f"| {p['sla_met_pct']}% | {p['compliance_pct']}% |"
        )
    return "\n".join(lines) + "\n"


def main(eh_path: str, target: str | None, lake: str) -> list[dict]:
    eh = read_table(eh_path)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        a, b = split_volumes(eh["volume"].unique().tolist())
        print(f"cross-volume: train on {a}, test on {b}")
        results.append(generalize(eh[eh["volume"].isin(a)], eh[eh["volume"].isin(b)],
                                  "MSR -> unseen MSR volumes", Path(tmp) / "xv"))  # fmt: skip
        if target:
            tgt = read_table(target)
            print(f"cross-dataset: train on all MSR, test on {sorted(tgt['volume'].unique())}")
            results.append(generalize(eh, tgt, "MSR -> Alibaba", Path(tmp) / "xd"))
    Path(lake).mkdir(parents=True, exist_ok=True)
    Path(lake, "generalization.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    Path(lake, "generalization.md").write_text(to_markdown(results), encoding="utf-8")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet", help="MSR (training) trace")
    ap.add_argument("--target", help="another dataset to test on, e.g. data/lake/alibaba_eh.parquet")
    ap.add_argument("--lake", default=str(DATA_DIR / "lake"))
    a = ap.parse_args()
    out = main(a.eh, a.target, a.lake)
    print("\n" + to_markdown(out))
    print(f"-> {a.lake}/generalization.json, generalization.md")
"""Train every hotness algorithm, pick a champion, promote it only if it beats the rules.

    python -m ayojna.models.train_hotness --features data/lake/features.parquet --store models_store

Honest selection, three windows that never overlap:
  fit         older training hours               -> every candidate learns here
  validation  last 24 labelled training hours    -> the champion is chosen here
  test        last 24 labelled hours (unseen)    -> reported only; decides promotion vs the rule
The champion is then refit on all training hours (fit + validation) before saving.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ayojna.io import read_table
from ayojna.models.baseline import evaluate, predict_rule
from ayojna.models.features import load_label_config, time_split
from ayojna.models.hotness import ALGOS, train
from ayojna.models.predictor import LATEST

LEADERBOARD = "leaderboard.json"


def main(features_path: str, store: str, algos: list[str] | None = None) -> dict:
    cfg = load_label_config()
    hot_min, horizon = cfg["labels"]["hot_min_accesses"], cfg["labels"]["horizon_hours"]
    algos = algos or cfg.get("hotness", {}).get("candidates", list(ALGOS))
    feats = read_table(features_path)
    tr, te = time_split(feats, cfg["split"]["test_hours"], horizon)
    fit, val = time_split(tr, cfg["split"]["test_hours"], horizon)
    print(
        f"fit hours {fit['hour'].min()}-{fit['hour'].max()} | "
        f"validation {val['hour'].min()}-{val['hour'].max()} | "
        f"test {te['hour'].min()}-{te['hour'].max()} (never used to choose)"
    )

    rule_val = evaluate(val["label"], predict_rule(val, hot_min)["pred"])
    rule_te = evaluate(te["label"], predict_rule(te, hot_min)["pred"])
    board = [{"model": "rule baseline", "status": "ok", "val_f1": rule_val["macro_f1"],
              "test_f1": rule_te["macro_f1"], "hot_recall": rule_te["hot_recall"],
              "train_s": 0.0}]  # fmt: skip
    for algo in algos:
        t0 = time.time()
        try:
            m = train(fit, algo=algo)
        except ImportError:
            board.append({"model": algo, "status": "not installed"})
            continue
        except Exception as exc:
            board.append({"model": algo, "status": f"failed: {exc}"})
            continue
        v = evaluate(val["label"], m.predict(val, with_reasons=False)["pred"])
        t = evaluate(te["label"], m.predict(te, with_reasons=False)["pred"])
        board.append({"model": algo, "status": "ok", "val_f1": v["macro_f1"], "test_f1": t["macro_f1"],
                      "hot_recall": t["hot_recall"], "train_s": round(time.time() - t0, 1)})  # fmt: skip

    ranked = [r for r in board[1:] if r["status"] == "ok"]
    if not ranked:
        raise SystemExit("no candidate could be trained")
    champ = max(ranked, key=lambda r: r["val_f1"])["model"]

    print(f"\n{'model':<15}{'val F1':>8}{'test F1':>9}{'hot recall':>12}{'train s':>9}")
    for r in board:
        if r["status"] != "ok":
            print(f"{r['model']:<15}  {r['status']}")
            continue
        tag = "  <- champion" if r["model"] == champ else ""
        print(f"{r['model']:<15}{r['val_f1']:>8.3f}{r['test_f1']:>9.3f}"
              f"{r['hot_recall']:>12.3f}{r['train_s']:>9.1f}{tag}")  # fmt: skip

    model = train(tr, algo=champ)  # refit on all training hours
    pred = model.predict(te, with_reasons=False)
    ml = evaluate(te["label"], pred["pred"])
    model.metrics = {"ml": ml, "rule": rule_te, "abstain_rate": float(pred["abstain"].mean())}
    path = model.save(store)
    promoted = ml["macro_f1"] > rule_te["macro_f1"]
    report = {
        "champion": champ,
        "explainer": model.explainer,
        "selected_on": "validation macro-F1",
        "windows": {
            "fit": [int(fit["hour"].min()), int(fit["hour"].max())],
            "validation": [int(val["hour"].min()), int(val["hour"].max())],
            "test": [int(te["hour"].min()), int(te["hour"].max())],
        },
        "champion_test": {"macro_f1": ml["macro_f1"], "hot_recall": ml["hot_recall"]},
        "rule_test": {"macro_f1": rule_te["macro_f1"], "hot_recall": rule_te["hot_recall"]},
        "label_mix_test": ml["support"],
        "promoted": promoted,
        "board": board,
    }
    Path(store, LEADERBOARD).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nchampion {champ} refit on all training hours: test macro-F1 {ml['macro_f1']:.3f} "
          f"(rule {rule_te['macro_f1']:.3f}), explanations by {model.explainer}")  # fmt: skip
    if promoted:
        (Path(store) / LATEST).write_text(json.dumps({"file": path.name, "version": model.version}))
        print(f"PROMOTED {path.name}")
    else:
        print(f"NOT promoted {path.name}: does not beat the rule baseline (rule fallback stays)")

    first_of_each = pred.drop_duplicates("pred").index  # one hot, one warm, one cold
    print("\nexample explanations:")
    for _, r in model.predict(te.loc[first_of_each]).iterrows():
        print(f"  {r['volume']} extent {r['extent_id']} hour {r['hour']}: {r['pred']} "
              f"({r['confidence']:.0%}) because {r['reasons']}")  # fmt: skip
    return model.metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--algos", nargs="*", help=f"subset of {', '.join(ALGOS)}")
    a = ap.parse_args()
    main(a.features, a.store, a.algos)
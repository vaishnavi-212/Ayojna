"""Slide 6 scorecard: every "measure of success" with its target, measured value and verdict.

    python -m ayojna.validate.kpi_report --eh data/lake/extent_hourly.parquet \
        --features data/lake/features.parquet --store models_store

Everything is measured on the unseen (test) hours of the replay, or read from what the
system logged (audit trail, chaos drill). A target that cannot be measured says so; nothing
is filled in by hand. Writes data/lake/kpi_report.json and kpi_report.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ayojna.api.service import Service
from ayojna.io import read_table
from ayojna.models.features import load_label_config, time_split
from ayojna.planner.evaluate import race_all
from ayojna.settings import DATA_DIR


def _row(kpi, target, value, ok, note=""):
    status = "NOT MEASURED" if ok is None else ("PASS" if ok else "MISS")
    return {"kpi": kpi, "target": target, "value": value, "status": status, "note": note}


def _json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def kpi_report(eh: pd.DataFrame, feats: pd.DataFrame, store: str, state: str, lake: str) -> dict:
    cfg = load_label_config()
    _, test = time_split(feats, cfg["split"]["test_hours"], cfg["labels"]["horizon_hours"])
    start = int(test["hour"].min())
    scored, summary, note = race_all(eh, feats, store)
    io = scored.groupby("strategy")[["ios", "hot_ios", "sla_ios"]].sum()
    io_sla = (100 * io["sla_ios"] / io["ios"]).round(3)
    hit = (100 * io["hot_ios"] / io["ios"]).round(2)
    ay = summary.loc["ayojna"]
    rules = summary.drop(index=["ayojna", "all_hot"], errors="ignore")
    rows = []

    # 1. cost vs the best rule, at equal or better SLA (and, to be fair, the same policy)
    fair = rules[(rules["compliance_pct"] >= 100 - 1e-9) & (rules["sla_met_pct"] <= ay["sla_met_pct"])]
    if len(fair):
        best = fair["monthly_cost"].idxmin()
        cut = 100 * (1 - ay["monthly_cost"] / fair.loc[best, "monthly_cost"])
        raw = rules["monthly_cost"].idxmin()
        raw_cut = 100 * (1 - ay["monthly_cost"] / rules.loc[raw, "monthly_cost"])
        rows.append(_row("Storage cost vs best rule", ">= 15% lower at equal or better SLA",
                         f"{cut:.1f}% lower than {best}", cut >= 15,
                         f"vs {raw} ignoring policy ({rules.loc[raw, 'compliance_pct']:.1f}% "
                         f"compliant): {raw_cut:+.1f}%"))  # fmt: skip
    # 2. SLA, weighted by I/O (every I/O counts once)
    rows.append(_row("Performance SLA", ">= 99% of I/Os within latency target",
                     f"{io_sla['ayojna']:.2f}%", io_sla["ayojna"] >= 99,
                     "I/O-weighted over the unseen hours"))  # fmt: skip
    # 3. hotness accuracy + hit ratio
    lb = _json(Path(store) / "leaderboard.json")
    if lb:
        f1 = lb["champion_test"]["macro_f1"]
        rows.append(_row("Hotness accuracy", "macro-F1 >= 0.85", f"{f1:.3f} ({lb['champion']})",
                         f1 >= 0.85, f"rule baseline {lb['rule_test']['macro_f1']:.3f}; "
                         f"test label mix {lb.get('label_mix_test')}"))  # fmt: skip
    else:
        rows.append(_row("Hotness accuracy", "macro-F1 >= 0.85", "-", None, "run models.train_all"))
    lru_lfu = {k: hit[k] for k in ("lru", "lfu") if k in hit}
    rows.append(_row("Hot-tier hit ratio", "above LRU/LFU", f"{hit['ayojna']:.1f}% of I/Os on hot",
                     hit["ayojna"] > max(lru_lfu.values()) if lru_lfu else None,
                     " · ".join(f"{k} {v:.1f}%" for k, v in lru_lfu.items()) +
                     " (Ayojna serves busy data from warm when the SLA allows: cheaper)"))  # fmt: skip
    # 4. capacity forecast
    intel = _json(Path(lake) / "intel_report.json")
    if intel:
        board = intel["forecast"]["board"]
        name = intel["forecast"]["live_model"]
        r = board.get(name) if board.get(name, {}).get("status") == "ok" else board.get("seasonal_ewma")
        used = name if board.get(name, {}).get("status") == "ok" else "seasonal_ewma"
        rows.append(_row("Capacity forecast", "MAPE < 15%", f"{r['mape_capacity_pct']}% ({used})",
                         r["mape_capacity_pct"] < 15,
                         f"24 h ahead (the trace is one week: 7 days ahead cannot be scored); "
                         f"I/O volume MAPE {r['mape_io_pct']}%"))  # fmt: skip
    else:
        rows.append(_row("Capacity forecast", "MAPE < 15%", "-", None, "run models.train_all"))
    # 5. compliance
    rows.append(_row("Compliance", "100% of placements pass policy", f"{ay['compliance_pct']:.1f}%",
                     ay["compliance_pct"] >= 100 - 1e-9))  # fmt: skip
    # 6. data movement vs I/O traffic
    t = eh[eh["hour"] >= start]
    io_gb = float((t["read_bytes"] + t["write_bytes"]).sum()) / 1e9
    moved = float(ay["gb_moved"])
    pct = 100 * moved / io_gb if io_gb else None
    rows.append(_row("Data movement", "<= 5% of total I/O", f"{pct:.2f}%" if pct is not None else "-",
                     pct <= 5 if pct is not None else None,
                     f"{moved:.2f} GB moved vs {io_gb:.1f} GB of I/O traffic"))  # fmt: skip
    # 7. availability + failover (logged by the supervisors / the chaos drill)
    central = Service(state, lake).central()
    chaos = _json(Path(lake) / "chaos_report.json")
    cy = central["cycles"]
    if chaos:
        v = chaos["completed_pct"]
        rows.append(_row("Availability under failure", ">= 99.9% cycles complete", f"{v}%",
                         v >= 99.9, f"chaos drill: {chaos['cycles']} cycles with injected faults, "
                         f"{chaos['acted_pct']}% acted normally, the rest held safely"))  # fmt: skip
    elif cy["total"]:
        rows.append(_row("Availability under failure", ">= 99.9% cycles complete",
                         f"{cy['completed_pct']}%", cy["completed_pct"] >= 99.9,
                         f"{cy['total']} logged cycles (run validate.chaos for a fault drill)"))  # fmt: skip
    else:
        rows.append(_row("Availability under failure", ">= 99.9% cycles complete", "-", None,
                         "run supervisors or validate.chaos"))  # fmt: skip
    f = central["failover"]
    rows.append(_row("Failover", "<= 10 s", f"{f['max_gap_s']} s worst of {f['count']}"
                     if f["count"] else "-", f["max_gap_s"] <= 10 if f["count"] else None,
                     "measured from the dead leader's last heartbeat"))  # fmt: skip

    report = {
        "scored_hours": [start, int(eh["hour"].max())],
        "predictions": note,
        "passed": sum(r["status"] == "PASS" for r in rows),
        "measured": sum(r["status"] != "NOT MEASURED" for r in rows),
        "rows": rows,
        "strategies": {
            s: {"monthly_cost": round(float(summary.loc[s, "monthly_cost"]), 4),
                "compliance_pct": round(float(summary.loc[s, "compliance_pct"]), 2),
                "io_sla_pct": float(io_sla[s]), "hot_hit_pct": float(hit[s])}
            for s in summary.index
        },  # fmt: skip
    }
    Path(lake).mkdir(parents=True, exist_ok=True)
    Path(lake, "kpi_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    Path(lake, "kpi_report.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    lines = [f"Scored on unseen hours {report['scored_hours'][0]}-{report['scored_hours'][1]}; "
             f"{report['passed']} of {report['measured']} measured targets met.", "",
             "| Measure | Target | Ayojna | Verdict | Note |", "|---|---|---|---|---|"]  # fmt: skip
    for r in report["rows"]:
        lines.append(f"| {r['kpi']} | {r['target']} | {r['value']} | {r['status']} | {r['note']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--state", default=str(DATA_DIR / "state"))
    ap.add_argument("--lake", default=str(DATA_DIR / "lake"))
    a = ap.parse_args()
    rep = kpi_report(read_table(a.eh), read_table(a.features), a.store, a.state, a.lake)
    print(to_markdown(rep))
    print(f"-> {a.lake}/kpi_report.json, kpi_report.md")
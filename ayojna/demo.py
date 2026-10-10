"""One command, whole system: data -> features -> model -> scoreboard -> supervisor -> dashboard.

    python -m ayojna.demo                                # synthetic trace (no download needed)
    python -m ayojna.demo --source real                  # real MSR traces in data/raw/msr
    python -m ayojna.demo --source real --serve          # ...and open the dashboard at the end
    python -m ayojna.demo --source lake                  # reuse data/lake/extent_hourly.parquet

Every stage runs as its own module (the same commands you can run by hand), so a
failure points at exactly one stage. Writes data/lake/results.md for the README and deck.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

from ayojna.settings import DATA_DIR, REPO_ROOT

LAKE = "data/lake"
EH, FEATS, BOARD = (
    f"{LAKE}/extent_hourly.parquet",
    f"{LAKE}/features.parquet",
    f"{LAKE}/scoreboard.json",
)
STORE = "models_store"


def run(*args: str, check: bool = True) -> bool:
    print(f"\n$ python -m {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT)
    if r.returncode != 0 and check:
        sys.exit(f"stage failed: {args[0]} (exit {r.returncode})")
    return r.returncode == 0


def results_table() -> str:
    board = json.loads((REPO_ROOT / BOARD).read_text(encoding="utf-8"))
    h0, h1 = board["scored_hours"]
    lines = [
        f"Scored on unseen hours {h0}-{h1}; predictions: {board['predictions']}",
        f"Guards in the replay: {board.get('guards', {})}",
        "",
        "| Strategy | $/month | Saving vs all-hot | SLA met | Compliance | GB moved "
        "| Over hot cap (h) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in board["summary"]:
        lines.append(
            f"| {r['strategy']} | {r['monthly_cost']:.3f} | {r['saving_vs_all_hot_pct']:.1f}% "
            f"| {r['sla_met_pct']:.2f}% | {r['compliance_pct']:.1f}% | {r['gb_moved']:.2f} "
            f"| {int(r['hours_over_hot_capacity'])} |"
        )
    return "\n".join(lines)


def main(a) -> None:
    t0 = time.time()
    print(f"== 1/6 ingest ({a.source}) ==")
    if a.source == "lake":
        if not (REPO_ROOT / EH).exists():
            sys.exit(f"{EH} not found: run with --source real or synth first")
        print(f"using the existing {EH}")
    elif a.source == "synth":
        run("ayojna.ingest.synth", "--days", "7", "--out", "data/raw/synth")
        run("ayojna.ingest.build", "--raw", "data/raw/synth", "--out", EH)
    else:
        run(
            "ayojna.ingest.real",
            "--raw",
            a.raw,
            "--out",
            EH,
            *(["--volumes", *a.volumes] if a.volumes else []),
        )

    print("\n== 2/6 features ==")
    run("ayojna.models.build_features", "--inp", EH, "--out", FEATS)

    print("\n== 3/6 intelligence layer: hotness model zoo + forecast + anomaly ==")
    if not run("ayojna.models.train_all", "--eh", EH, "--features", FEATS, "--store", STORE,
               "--report", f"{LAKE}/intel_report.json", check=False):  # fmt: skip
        (REPO_ROOT / STORE / "hotness-latest.json").unlink(missing_ok=True)
        print("model not trained on this data -> the supervisor will use the rule fallback (L1)")

    print("\n== 4/6 scoreboard (digital twin race) ==")
    run("ayojna.api.snapshot", "--eh", EH, "--features", FEATS, "--store", STORE)
    table = results_table()
    (DATA_DIR / "lake" / "results.md").write_text(table + "\n", encoding="utf-8")

    print("\n== 5/6 supervisor (fresh state) ==")
    for d in ("state", "tiers"):
        shutil.rmtree(DATA_DIR / d, ignore_errors=True)
    run(
        "ayojna.supervisor.run",
        "--cycles",
        str(a.cycles),
        "--interval",
        "1",
        "--eh",
        EH,
        "--features",
        FEATS,
        "--store",
        STORE,
    )

    print(f"\n== 6/6 results ({time.time() - t0:.0f}s) ==\n\n{table}\n")
    print("saved to data/lake/results.md")
    if a.serve:
        print("\ndashboard: http://localhost:8000   (Ctrl+C to stop)")
        run("uvicorn", "ayojna.api.app:app", "--port", "8000", check=False)
    else:
        print("dashboard: python -m uvicorn ayojna.api.app:app --port 8000")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the whole Ayojna pipeline")
    ap.add_argument("--source", choices=["synth", "real", "lake"], default="synth")
    ap.add_argument("--raw", default="data/raw/msr", help="folder with the SNIA .tar file(s)")
    ap.add_argument("--volumes", nargs="*", help="real only, e.g. web_0 usr_0 (default: 8 tagged)")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--serve", action="store_true", help="start the dashboard at the end")
    main(ap.parse_args())
"""Fault-injection drill (slide 5, step 7: "replay vs baselines + fault injection").

    python -m ayojna.validate.chaos --cycles 12 --eh data/lake/extent_hourly.parquet \
        --features data/lake/features.parquet --store models_store

Runs supervisor cycles on the real pipeline. Each cycle gets a random fault: a model, the
planner or the executor fails, or a copied extent is corrupted. Measures what slide 6 asks:
  completed  the cycle finished with a safe outcome (no crash; holds count as safe)
  acted      the cycle ran normally or degraded (L0-L2), not held (L3)
  unsafe     moves executed while the plan step had failed: must be 0
Uses its own state folder (data/state_chaos) so the live dashboard state is untouched.
Writes data/lake/chaos_report.json (read by the KPI report).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from ayojna.settings import DATA_DIR
from ayojna.supervisor.pipeline import build_steps
from ayojna.supervisor.runner import run_cycle
from ayojna.supervisor.state import StateStore

FAULTS = [None, "corrupt", "hotness", "forecast", "anomaly", "features", "plan", "execute"]


def drill(eh: str, features: str, store: str, cycles: int, seed: int, state_dir: Path) -> dict:
    shutil.rmtree(state_dir, ignore_errors=True)
    state = StateStore(state_dir, lease_ttl_s=600)
    token = state.acquire("chaos")
    rng = random.Random(seed)
    rows = []
    for i in range(cycles):
        fault = FAULTS[i % len(FAULTS)] if i < len(FAULTS) else rng.choice(FAULTS)
        steps = build_steps(eh, features, store,
                            faults={fault: "fail"} if fault not in (None, "corrupt") else {},
                            timeout_s=60, state_dir=str(state_dir),
                            tiers_root=str(state_dir / "tiers"),
                            corrupt_first_move=fault == "corrupt", store=state)  # fmt: skip
        try:
            r = run_cycle(f"chaos-{i:03d}", steps, state, token)
            ex = r["outputs"].get("execute") or {}
            plan_failed = r["steps"].get("plan", {}).get("source") != "primary"
            rows.append({
                "cycle": i, "fault": fault or "none", "level": r["level"], "completed": True,
                "moves_done": ex.get("done", 0), "rolled_back": ex.get("rolled_back", 0),
                "unsafe": bool(plan_failed and ex.get("done", 0) > 0),
            })  # fmt: skip
        except Exception as exc:  # a crash: exactly what the drill is looking for
            rows.append({"cycle": i, "fault": fault or "none", "level": "crash",
                         "completed": False, "error": f"{type(exc).__name__}: {exc}"})  # fmt: skip
        print(f"cycle {i:>2}  fault {rows[-1]['fault']:<9} -> {rows[-1]['level']}"
              f"  moves {rows[-1].get('moves_done', '-')}  rolled back {rows[-1].get('rolled_back', '-')}")  # fmt: skip
    n = len(rows)
    done = sum(r["completed"] for r in rows)
    acted = sum(r.get("level") in ("L0", "L1", "L2") for r in rows)
    return {
        "cycles": n,
        "completed_pct": round(100 * done / n, 2),
        "acted_pct": round(100 * acted / n, 2),
        "unsafe_cycles": sum(r.get("unsafe", False) for r in rows),
        "dead_letters": len(state.dead_letters(10_000)),
        "rows": rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lake", default=str(DATA_DIR / "lake"))
    a = ap.parse_args()
    rep = drill(a.eh, a.features, a.store, a.cycles, a.seed, DATA_DIR / "state_chaos")
    Path(a.lake).mkdir(parents=True, exist_ok=True)
    Path(a.lake, "chaos_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n{rep['cycles']} cycles: {rep['completed_pct']}% completed safely, "
          f"{rep['acted_pct']}% acted, {rep['unsafe_cycles']} unsafe, "
          f"{rep['dead_letters']} dead letters -> {a.lake}/chaos_report.json")
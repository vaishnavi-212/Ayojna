"""Run a supervisor instance. Start two (primary and replica) to see failover.

    python -m ayojna.supervisor.run --name primary --cycles 3
    python -m ayojna.supervisor.run --name replica --cycles 3     # in a second terminal

State:        --state-store auto (Redis if running, else files) | redis | file
              Redis URL from AYOJNA_REDIS_URL (default redis://localhost:6379/0)

Demo faults:  --fail hotness|forecast|anomaly  |  --slow features  |  --crash-after features
Executor:     --store-kind minio  |  --recommend-only  |  --corrupt-move  |  --crash-after-moves 20
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime, timezone

from ayojna.supervisor.pipeline import build_steps
from ayojna.supervisor.runner import Parallel, run_cycle
from ayojna.supervisor.state import StateStore, make_state_store


def heartbeat(store: StateStore, name: str, token: int, stop: threading.Event) -> None:
    while not stop.wait(store.ttl / 3):
        if not store.renew(name, token):
            return


def main(a) -> None:
    store = make_state_store(a.state_store, a.state, a.ttl)
    print(f"[{a.name}] shared state: {store.kind}" + (f" ({store.url})" if store.kind == "redis" else ""))
    faults = {a.fail: "fail"} if a.fail else {}
    if a.slow:
        faults[a.slow] = "slow"
    steps = build_steps(
        a.eh,
        a.features,
        a.store,
        faults,
        timeout_s=a.timeout,
        state_dir=a.state,
        tiers_root=a.tiers,
        store_kind=a.store_kind,
        recommend_only=a.recommend_only,
        corrupt_first_move=a.corrupt_move,
        crash_after_moves=a.crash_after_moves,
        approval_mode=a.approval,
        store=store,
    )
    if a.crash_after:  # crash right after this step's (or group's) checkpoint is saved
        names = [[s.name, *[m.name for m in getattr(s, "steps", [])]] for s in steps]
        i = next(k for k, n in enumerate(names) if a.crash_after in n)
        if i + 1 < len(steps):
            nxt = steps[i + 1]
            (nxt.steps[0] if isinstance(nxt, Parallel) else nxt).fn = lambda ctx: os._exit(1)
    done, token = 0, None
    while done < a.cycles:
        token = store.acquire(a.name)
        if token is None:
            print(f"[{a.name}] standby: another supervisor holds the lease")
            time.sleep(1)
            continue
        stop = threading.Event()
        threading.Thread(target=heartbeat, args=(store, a.name, token, stop), daemon=True).start()
        resumed = store.active_run()
        if store.last_takeover:  # measured: last heartbeat of the old leader -> now
            t = store.last_takeover
            print(f"[{a.name}] TAKEOVER from {t['from']} after {t['gap_s']} s without a leader")
            store.audit({"event": "failover", **t, "resumed_run": resumed, "store": store.kind})
        run_id = resumed or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")
        print(f"[{a.name}] leader (token {token}) {'RESUMING' if resumed else 'starting'} {run_id}")
        try:
            report = run_cycle(run_id, steps, store, token)
        finally:
            stop.set()
        for name, info in report["steps"].items():
            tag = " (resumed from checkpoint)" if info.get("resumed") else ""
            ms = f"{info.get('ms', 0):>6} ms" if "ms" in info else " " * 9
            print(f"    {name:<9} {info['source']:<9}{ms}  {info['note'][:60]}{tag}")
        for g, t in report.get("groups", {}).items():
            print(f"    [{g}] {' | '.join(t['steps'])} in parallel: "
                  f"{t['wall_ms']} ms wall vs {t['sum_ms']} ms one after another")  # fmt: skip
        out = report["outputs"]
        ex = out.get("execute") or {}
        if ex.get("mode") == "recommend-only" and report["level"] != "L3":
            report["level"] = "L2"  # plan shown to humans, nothing moved
        if ex.get("mode") == "safe-mode":
            report["level"] = "L4"  # operator safe mode: read-only, nothing moves
            print(f"    SAFE MODE (L4): {(ex.get('safe_mode') or {}).get('reason', '')}")
        if "plan" in out:
            p = out["plan"]
            print(f"    plan: {len(p.moves)} moves, {p.total_gb:.1f} GB ({p.strategy})")
        if ex.get("mode") in ("executed", "approval"):
            print(
                f"    execute: done {ex['done']}, pending approval {ex['pending']}, "
                f"rejected {ex['rejected']}, skipped {ex['skipped']}, "
            )
        store.audit(
            {
                "event": "cycle",
                "run_id": run_id,
                "token": token,
                "owner": a.name,
                "level": report["level"],
                "moves_planned": len(out["plan"].moves) if "plan" in out else 0,
                "moves_done": ex.get("done", 0),
                "rolled_back": ex.get("rolled_back", 0),
                "gb_moved": ex.get("gb_moved", 0),
            }
        )
        print(f"[{a.name}] level {report['level']}")
        done += 1
        if done < a.cycles:
            time.sleep(a.interval)
    if token is not None:
        store.release(a.name, token)  # clean stop: not a failover, a replica may start at once


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="primary")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--ttl", type=float, default=8.0)  # failover within ttl + 1 s poll
    ap.add_argument("--state-store", choices=["auto", "file", "redis"], default="auto")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--state", default="data/state")
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    ap.add_argument("--tiers", default="data/tiers")
    ap.add_argument("--store-kind", choices=["fs", "minio"], default="fs")
    ap.add_argument("--recommend-only", action="store_true")
    ap.add_argument("--approval", action="store_true", help="execute only approved groups")
    ap.add_argument("--corrupt-move", action="store_true")
    ap.add_argument("--crash-after-moves", type=int)
    ap.add_argument("--fail")
    ap.add_argument("--slow")
    ap.add_argument("--crash-after")
    main(ap.parse_args())
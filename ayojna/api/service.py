"""Everything the dashboard shows, as plain functions over files on disk.

Reads only what the other modules write (no extra database):
  data/state/  lease.json, active.json, audit.jsonl, catalog.json,
               last_plan.json, last_exec.json          <- supervisor + executor
  data/lake/   scoreboard.json                         <- ayojna.api.snapshot
Missing files are normal (nothing ran yet): functions return {"available": False}.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from ayojna.contracts import EXTENT_MB, TIER_ORDER
from ayojna.policy import guard
from ayojna.policy.guard import allowed_tiers
from ayojna.recommend import engine
from ayojna.recommend.approvals import load_approvals, set_decision
from ayojna.settings import load_config

GB = EXTENT_MB / 1024
L4_GRACE_S = 5.0  # a lease this long past its expiry = no supervisor alive
TIERS = [t.value for t in TIER_ORDER]


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # missing or busy (Windows): treat as "not yet"


def _tail(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        lines = deque(f, maxlen=n)
    out = []
    for x in lines:
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            continue
    return out[::-1]  # newest first


class Service:
    def __init__(self, state_dir: str | Path, lake_dir: str | Path):
        self.state, self.lake = Path(state_dir), Path(lake_dir)

    def last_cycle(self) -> dict | None:
        for e in _tail(self.state / "audit.jsonl", 500):
            if e.get("event") == "cycle":
                return e
        return None

    def status(self) -> dict:
        lease = _json(self.state / "lease.json")
        cycle = self.last_cycle()
        now = time.time()
        level, why = (cycle["level"] if cycle else None), None
        safe = _json(self.state / "safe_mode.json") or {}
        if safe.get("on"):
            level, why = "L4", f"operator safe mode: {safe.get('reason') or 'no reason given'}"
        elif lease and lease["expires"] + L4_GRACE_S < now:
            level = "L4"
            why = f"no supervisor alive for {now - lease['expires']:.0f} s: read-only, nothing moves"
        return {
            "l4_reason": why,
            "safe_mode": safe if safe.get("on") else None,
            "store": (lease or {}).get("store", "file"),
            "available": lease is not None,
            "leader": lease["owner"] if lease else None,
            "fencing_token": lease["token"] if lease else None,
            "leader_alive": bool(lease and lease["expires"] > now),
            "lease_seconds_left": round(max(0.0, lease["expires"] - now), 1) if lease else 0,
            "active_run": (_json(self.state / "active.json") or {}).get("run_id"),
            "level": level,
            "last_cycle": cycle,
        }

    def placement(self) -> dict:
        cat = _json(self.state / "catalog.json")
        if not cat:
            return {"available": False}
        price = load_config().tiers.tiers
        counts = {t: 0 for t in TIERS}
        for e in cat.values():
            counts[e["tier"]] += 1
        now = sum(n * GB * price[t].price_gb_month for t, n in zip(TIER_ORDER, counts.values()))
        all_hot = len(cat) * GB * price[TIER_ORDER[0]].price_gb_month
        return {
            "available": True,
            "extents": counts,
            "gb": {t: round(n * GB, 2) for t, n in counts.items()},
            "storage_cost_month": round(now, 4),
            "all_hot_cost_month": round(all_hot, 4),
            "saving_pct": round(100 * (1 - now / all_hot), 2) if all_hot else 0.0,
        }

    def plan(self, limit: int = 50) -> dict:
        saved = _json(self.state / "last_plan.json")
        if not saved:
            return {"available": False}
        p, why = saved["plan"], saved["why"]
        run_id = p["envelope"]["run_id"]
        moves = [
            {**m, "why": why.get(f"{run_id}:{m['volume']}:{m['extent_id']}:{m['to_tier']}", "")}
            for m in p["moves"][:limit]
        ]
        return {
            "available": True,
            "envelope": p["envelope"],
            "strategy": p["strategy"],
            "hour": p["hour"],
            "n_moves": len(p["moves"]),
            "total_gb": round(sum(m["size_gb"] for m in p["moves"]), 2),
            "saving_per_month": round(sum(m["expected_saving_per_month"] for m in p["moves"]), 4),
            "moves": moves,
        }

    def execution(self) -> dict:
        r = _json(self.state / "last_exec.json")
        if not r:
            return {"available": False}
        problems = [x for x in r.get("results", []) if x["status"] != "done"]
        return {
            "available": True,
            **{k: v for k, v in r.items() if k != "results"},
            "problems": problems[:50],
        }

    def audit(self, limit: int = 50) -> list[dict]:
        return _tail(self.state / "audit.jsonl", limit)

    def scoreboard(self) -> dict:
        board = _json(self.lake / "scoreboard.json")
        return {"available": True, **board} if board else {"available": False}

    def explain(self, volume: str, extent_id: int) -> dict:
        """Where is this extent, which tiers may it use, and why (policy + last plan)."""
        key = f"{volume}/{int(extent_id):06d}"
        entry = (_json(self.state / "catalog.json") or {}).get(key)
        current = TIERS.index(entry["tier"]) if entry else 0
        mask, reasons = allowed_tiers(np.array([volume]), np.array([current]))
        plan = self.plan(limit=10**6)
        move = next(
            (
                m
                for m in plan.get("moves", [])
                if m["volume"] == volume and m["extent_id"] == int(extent_id)
            ),
            None,
        )
        return {
            "volume": volume,
            "extent_id": int(extent_id),
            "tier": entry["tier"] if entry else None,
            "on_tier_since_hour": entry["since"] if entry else None,
            "allowed_tiers": [t for t, ok in zip(TIERS, mask[0]) if ok],
            "policy": reasons[0] or "no restrictions",
            "tags": load_config().tags_for(volume).model_dump(),
            "last_plan_move": move,
        }

    def kpis(self) -> dict:
        board, place, cycle = self.scoreboard(), self.placement(), self.last_cycle()
        ay = next((r for r in board.get("summary", []) if r["strategy"] == "ayojna"), None)
        rules = [r for r in board.get("summary", []) if r["strategy"] not in ("ayojna", "all_hot")]
        best = min(rules, key=lambda r: r["monthly_cost"]) if rules else None
        return {
            "twin_saving_vs_all_hot_pct": ay["saving_vs_all_hot_pct"] if ay else None,
            "twin_saving_vs_best_rule_pct": (
                round(100 * (1 - ay["monthly_cost"] / best["monthly_cost"]), 2)
                if ay and best
                else None
            ),
            "best_rule": best["strategy"] if best else None,
            "sla_met_pct": ay["sla_met_pct"] if ay else None,
            "compliance_pct": ay["compliance_pct"] if ay else None,
            "live_saving_pct": place.get("saving_pct"),
            "level": self.status()["level"],
            "moves_last_cycle": cycle["moves_done"] if cycle else None,
        }
    
    # ---------- intelligence layer (hotness | forecast | anomaly) ----------
    def intel(self) -> dict:
        live = _json(self.state / "last_intel.json")  # written every cycle by the supervisor
        report = _json(self.lake / "intel_report.json")  # written by models.train_all
        if not live and not report:
            return {"available": False}
        board = _json(self.lake / "scoreboard.json") or {}
        return {"available": True, "live": live, "report": report, "replay_guards": board.get("guards")}

    # ---------- central layer (supervisor, replica, shared state) ----------
    def central(self) -> dict:
        st = self.status()
        events = _tail(self.state / "audit.jsonl", 20000)
        failovers = [e for e in events if e.get("event") == "failover"]
        cycles = [e for e in events if e.get("event") == "cycle"]
        held = sum(e.get("level") == "L3" for e in cycles)
        by_level: dict = {}
        for e in cycles:
            by_level[e.get("level")] = by_level.get(e.get("level"), 0) + 1
        dlq_path = self.state / "dlq.jsonl"
        dlq_count = sum(1 for x in dlq_path.open(encoding="utf-8") if x.strip()) if dlq_path.exists() else 0
        gaps = [f["gap_s"] for f in failovers if "gap_s" in f]
        return {
            "available": st["available"] or bool(cycles),
            "status": st,
            "failover": {
                "count": len(failovers),
                "last": failovers[0] if failovers else None,  # _tail is newest first
                "max_gap_s": max(gaps) if gaps else None,
                "target_s": 10,
            },
            "cycles": {
                "total": len(cycles),
                "held_L3": held,
                "completed_pct": round(100 * (1 - held / len(cycles)), 2) if cycles else None,
                "by_level": by_level,
            },
            "dlq": {"count": dlq_count, "recent": _tail(dlq_path, 10)},
        }

    def set_safe_mode(self, on: bool, reason: str = "") -> dict:
        """Operator kill switch (L4): the next cycles plan but move nothing until it is off."""
        flag = {"on": bool(on), "reason": reason, "since": time.time()}
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "safe_mode.json").write_text(json.dumps(flag), encoding="utf-8")
        with open(self.state / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": flag["since"], "event": "safe_mode", **flag}) + "\n")
        return flag

    # ---------- decision layer (policy -> optimizer -> RL bandit) ----------
    def decision(self) -> dict:
        saved = _json(self.state / "last_plan.json") or {}
        board = _json(self.lake / "scoreboard.json") or {}
        cat = _json(self.state / "catalog.json") or {}
        volumes = sorted({k.split("/")[0] for k in cat}) or sorted(load_config().volumes)
        decisions, report = guard.decide(volumes)  # asks the policy engine right now
        return {
            "available": True,
            "policy": {**report, "volumes": decisions},
            "live": saved.get("decision"),  # solver + bandit of the last supervisor cycle
            "replay": board.get("decision"),  # solver + bandit over the whole replay
        }

    # ---------- recommendation engine ----------
    def recommendations(self) -> dict:
        saved = _json(self.state / "last_plan.json")
        if not saved:
            return {"available": False}
        return engine.build(saved, _json(self.state / "last_exec.json"), load_approvals(self.state))

    def recommendation(self, group: str) -> dict:
        saved = _json(self.state / "last_plan.json")
        found = engine.detail(saved, group) if saved else None
        return {"available": True, **found} if found else {"available": False}

    def decide(self, group: str, decision: str, note: str = "") -> dict:
        """The only write the API makes: a human decision, never data."""
        return {"group": group, **set_decision(self.state, group, decision, note)}
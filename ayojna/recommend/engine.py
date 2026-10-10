"""Turns the latest plan into ranked, explained, approvable recommendations.

One recommendation = all planned moves of one volume in one direction (e.g. usr_0 hot>cold).
Each carries: the action in words, money, size, confidence, risk, the policy that applied,
the strongest prediction drivers across its extents, its approval status and what the
executor did with it. The per-extent decision cards (prediction -> policy -> cost -> decision)
stay attached for the "why?" view.
"""

from __future__ import annotations

from collections import defaultdict

from ayojna.contracts import TIER_ORDER
from ayojna.recommend.approvals import decision_for

RANK = {t.value: i for i, t in enumerate(TIER_ORDER)}
RISK = {"low": 0, "med": 1, "high": 2}


def _verb(src: str, dst: str) -> str:
    if RANK[dst] < RANK[src]:
        return "Promote"
    return "Archive" if dst == "archive" else "Demote"


def _top_drivers(cards: list[dict], k: int = 3) -> list[dict]:
    """Average signed effect of each feature across the group's extents (strongest first)."""
    total: dict[str, list[float]] = defaultdict(list)
    for c in cards:
        for d in c["prediction"].get("drivers", []):
            total[d["feature"]].append(d["effect"])
    avg = [(f, sum(v) / len(cards)) for f, v in total.items()]
    avg.sort(key=lambda x: -abs(x[1]))
    return [{"feature": f, "effect": round(e, 3)} for f, e in avg[:k]]


def build(saved_plan: dict, last_exec: dict | None, approvals: dict) -> dict:
    p = saved_plan["plan"]
    cards, why = saved_plan.get("cards", {}), saved_plan.get("why", {})
    run_id = p["envelope"]["run_id"]
    status_of = {r["key"]: r["status"] for r in (last_exec or {}).get("results", [])}
    groups: dict[str, dict] = {}
    for m in p["moves"]:
        key = f"{run_id}:{m['volume']}:{m['extent_id']}:{m['to_tier']}"
        gid = f"{m['volume']}:{m['from_tier']}>{m['to_tier']}"
        g = groups.setdefault(
            gid, {"id": gid, "volume": m["volume"], "from": m["from_tier"], "to": m["to_tier"],
                  "keys": [], "gb": 0.0, "saving": 0.0, "risk": "low", "cards": []},
        )  # fmt: skip
        g["keys"].append(key)
        g["gb"] += m["size_gb"]
        g["saving"] += m["expected_saving_per_month"]
        g["risk"] = max(g["risk"], m["risk"], key=RISK.get)
        if key in cards:
            g["cards"].append(cards[key])
    out = []
    for g in groups.values():
        n, cs = len(g["keys"]), g["cards"]
        conf = sum(c["prediction"]["confidence"] for c in cs) / len(cs) if cs else None
        done = sum(status_of.get(k) == "done" for k in g["keys"])
        rules = cs[0]["policy"]["rules"] if cs else ""
        guard = cs[0]["policy"].get("guard") if cs else None
        verb = _verb(g["from"], g["to"])
        out.append(
            {
                "id": g["id"],
                "action": f"{verb} {n} extent{'s' * (n > 1)} ({g['gb']:.2f} GB) of {g['volume']} "
                f"from {g['from']} to {g['to']}",
                "verb": verb,
                "volume": g["volume"],
                "from": g["from"],
                "to": g["to"],
                "extents": n,
                "gb": round(g["gb"], 2),
                "saving_per_month": round(g["saving"], 4),
                "confidence": round(conf, 3) if conf is not None else None,
                "risk": g["risk"],
                "policy": rules,
                "guard": guard,
                "drivers": _top_drivers(cs),
                "reason": why.get(g["keys"][0], ""),
                "status": decision_for(approvals, g["id"]),
                "executed": done,
                "sample_keys": g["keys"][:5],
            }
        )
    out.sort(key=lambda r: (r["verb"] == "Promote", -r["saving_per_month"]))
    total = sum(r["saving_per_month"] for r in out)
    return {
        "available": True,
        "run_id": run_id,
        "hour": p["hour"],
        "model_version": p["envelope"].get("model_version"),
        "policy_version": p["envelope"].get("policy_version"),
        "summary": {
            "recommendations": len(out),
            "moves": sum(r["extents"] for r in out),
            "gb": round(sum(r["gb"] for r in out), 2),
            "saving_per_month": round(total, 4),
            "pending": sum(r["status"] == "pending" for r in out),
            "approved": sum(r["status"] == "approved" for r in out),
            "rejected": sum(r["status"] == "rejected" for r in out),
        },
        "items": out,
    }


def detail(saved_plan: dict, group: str, limit: int = 10) -> dict | None:
    """The decision cards of one recommendation (one per extent, up to `limit`)."""
    run_id = saved_plan["plan"]["envelope"]["run_id"]
    cards = saved_plan.get("cards", {})
    items = []
    for m in saved_plan["plan"]["moves"]:
        if f"{m['volume']}:{m['from_tier']}>{m['to_tier']}" != group:
            continue
        key = f"{run_id}:{m['volume']}:{m['extent_id']}:{m['to_tier']}"
        items.append(
            {"extent": f"{m['volume']}/{m['extent_id']}", "key": key, **cards.get(key, {})}
        )
        if len(items) >= limit:
            break
    return {"group": group, "items": items} if items else None
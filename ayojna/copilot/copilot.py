"""Ayojna copilot: answers operator questions in plain English, grounded in live facts.

    question -> route to an intent -> pull FACTS from the dashboard service
             -> LLM writes the answer from the facts only (optional)
             -> grounding guard checks every number -> else deterministic template

Read-only by design: it explains plans, it never makes or executes them.
Run:  python -m ayojna.copilot.copilot "why is web_0/3 on warm?"
"""

from __future__ import annotations

import json
import re
import sys
from typing import Callable

from ayojna.api.service import Service
from ayojna.copilot.guard import ungrounded_numbers
from ayojna.copilot.llm import LLMUnavailable, complete
from ayojna.settings import DATA_DIR

EXTENT = re.compile(r"\b([a-z]+\d*_\d+)\s*(?:/|#|\s+extent\s+|\s+ext\s+)\s*(\d+)\b", re.I)
VOLUME = re.compile(r"\b([a-z]+\d*_\d+)\b", re.I)
KEYWORDS = [  # first match wins
    ("execution", r"roll ?back|rolled|execut|skipp|idempot|budget|checksum|saga"),
    ("status", r"level|leader|failover|replica|token|lease|supervisor|status|health|degrad"),
    ("policy", r"polic|complian|legal|hold|archive|pii|financial|allowed|residen"),
    ("savings", r"sav|cost|cheap|money|\$|price|baseline|rule|compar|sla"),
    ("plan", r"plan|move|recommend|next|migrat"),
    ("audit", r"audit|history|log|happen"),
]
LEVELS = {
    "L0": "normal: every step used its primary path",
    "L1": "degraded: a fallback was used (e.g. rule model instead of ML)",
    "L2": "recommend-only: the plan is shown to people, nothing moves",
    "L3": "hold: a critical step failed, so nothing moves this cycle",
    "L4": "safe mode: no supervisor is in charge (or an operator switched it on); read-only, nothing moves",
}
SYSTEM = (
    "You are the Ayojna storage-tiering copilot. Answer the operator's question using ONLY "
    "the FACTS JSON. Quote numbers exactly as they appear in FACTS; never estimate or invent "
    "numbers. If FACTS do not contain the answer, say so and suggest what to run. You cannot "
    "move data or change plans; you only explain. Answer in at most 4 short sentences."
)


def route(question: str) -> tuple[str, dict]:
    q = question.strip()
    m = EXTENT.search(q)
    if m:
        return "extent", {"volume": m.group(1).lower(), "extent_id": int(m.group(2))}
    for intent, pattern in KEYWORDS:
        if re.search(pattern, q, re.I):
            v = VOLUME.search(q)
            if intent == "policy" and v:
                return "policy", {"volume": v.group(1).lower()}
            return ("policy_all" if intent == "policy" else intent), {}
    return "overview", {}


def gather(svc: Service, intent: str, args: dict) -> dict:
    if intent == "extent":
        return svc.explain(args["volume"], args["extent_id"])
    if intent == "policy":
        e = svc.explain(args["volume"], 0)
        return {k: e[k] for k in ("volume", "allowed_tiers", "policy", "tags")}
    if intent == "plan":
        p = svc.plan(limit=5)
        return (
            p
            if not p["available"]
            else {
                **p,
                "envelope": {
                    k: p["envelope"][k] for k in ("run_id", "model_version", "policy_version")
                },
            }
        )
    if intent == "execution":
        return svc.execution()
    if intent == "status":
        return svc.status()
    if intent == "audit":
        return {"events": svc.audit(10)}
    return svc.kpis()  # savings, policy_all, overview


def _pct(v) -> str:
       return "unknown" if v is None else f"{round(float(v), 1)}%"


def template(intent: str, f: dict) -> str:
    """Deterministic answer from the facts: always correct, never invents anything."""
    if f.get("available") is False:
        return (
            "Nothing to report yet. Run the supervisor (python -m ayojna.supervisor.run) "
            "and the snapshot (python -m ayojna.api.snapshot) first."
        )
    if intent == "extent":
        name = f"{f['volume']}/{f['extent_id']}"
        if f["tier"] is None:
            return f"{name} is not in the catalog. Check the name, or run the supervisor first."
        out = (
            f"{name} is on {f['tier']} (since hour {f['on_tier_since_hour']}). "
            f"Allowed tiers: {', '.join(f['allowed_tiers'])}. Policy: {f['policy']}."
        )
        m = f["last_plan_move"]
        if m:
            return out + (
                f" The latest plan moves it {m['from_tier']} -> {m['to_tier']}, saving "
                f"${m['expected_saving_per_month']}/month, because: {m['why']}."
            )
        return (
            out + " The latest plan does not move it (not worth it now, not allowed,"
            " or beyond this hour's migration budget)."
        )
    if intent == "policy":
        t = f["tags"]
        return (
            f"{f['volume']} is {t['data_class']} data with a {t['sla_class']} SLA "
            f"(legal hold: {'yes' if t['legal_hold'] else 'no'}, residency: {t['residency']}). "
            f"Allowed tiers: {', '.join(f['allowed_tiers'])}. Rules: {f['policy']}."
        )
    if intent == "plan":
        top = "; ".join(
            f"{m['volume']}/{m['extent_id']} {m['from_tier']} -> {m['to_tier']} ({m['why']})"
            for m in f["moves"][:3]
        )
        return (
            f"Latest plan ({f['strategy']}, hour {f['hour']}): {f['n_moves']} moves, "
            f"{f['total_gb']} GB, saving ${f['saving_per_month']}/month."
            + (f" Top moves: {top}." if top else " Nothing needs to move right now.")
        )
    if intent == "execution":
        if f.get("mode") == "recommend-only":
            return "The last execution was recommend-only (L2): the plan was shown, nothing moved."
        out = (
            f"Last execution: {f['done']} moves done, {f['skipped']} skipped as already "
            f"applied, {f['rolled_back']} rolled back, {f['over_budget']} over budget; "
            f"{f['gb_moved']} of {f['budget_gb']} GB budget used."
        )
        rb = [p for p in f.get("problems", []) if p["status"] == "rolled_back"]
        if rb:
            out += f" Rollback example: {rb[0]['key']} ({rb[0]['note']}); its data stayed safe."
        return out + (
            " It stopped early: a newer leader took over (fenced)." if f["fenced"] else ""
        )
    if intent == "status":
        c = f["last_cycle"] or {}
        lvl = f["level"]
        return (
            f"Leader: {f['leader']} with fencing token {f['fencing_token']} "
            f"({'lease alive' if f['leader_alive'] else 'lease expired: no cycle running'}). "
            f"Last cycle level {lvl}: {LEVELS.get(lvl, 'no cycle yet')}."
            + (f" It moved {c['moves_done']} of {c['moves_planned']} planned extents." if c else "")
        )
    if intent == "audit":
        ev = f["events"][:5]
        if not ev:
            return "The audit log is empty."
        lines = [
            (
                f"cycle {e['level']} ({e['moves_done']} moves)"
                if e.get("event") == "cycle"
                else f"step {e['step']}: {e['source']}"
            )
            for e in ev
        ]
        return "Most recent events: " + "; ".join(lines) + "."
    return (
        f"On unseen hours the digital twin shows Ayojna saving "
        f"{_pct(f['twin_saving_vs_all_hot_pct'])} vs keeping everything hot and "
        f"{_pct(f['twin_saving_vs_best_rule_pct'])} vs the best rule ({f['best_rule']}), "
        f"with SLA met {_pct(f['sla_met_pct'])} and compliance {_pct(f['compliance_pct'])}. "
        f"Live placement saves {_pct(f['live_saving_pct'])}; supervisor level {f['level']}."
    )


def ask(question: str, svc: Service, llm: Callable[[str, str], str] = complete) -> dict:
    intent, args = route(question)
    facts = gather(svc, intent, args)
    fallback = template(intent, facts)
    result = {"question": question, "intent": intent, "facts": facts}
    prompt = f"FACTS:\n{json.dumps(facts, default=str)}\n\nQUESTION: {question}"
    try:
        answer = llm(SYSTEM, prompt)
    except LLMUnavailable as exc:
        return {**result, "answer": fallback, "source": "template", "note": str(exc)}
    bad = ungrounded_numbers(answer, facts, question)
    if bad:
        note = f"LLM answer rejected by grounding guard: unsupported numbers {bad[:5]}"
        return {**result, "answer": fallback, "source": "template", "note": note}

    provider = getattr(answer, "provider", "") or "llm"
    note = f"{provider}: grounding check passed"
    return {**result, "answer": str(answer), "source": "llm", "provider": provider, "note": note}

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "how much are we saving?"
    r = ask(q, Service(DATA_DIR / "state", DATA_DIR / "lake"))
    print(f"[{r['source']} | intent {r['intent']}] {r['note']}\n\n{r['answer']}")
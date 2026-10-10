"""Prometheus metrics for Grafana (slide 5: "Grafana · live KPIs"): GET /metrics.

Plain Prometheus text format (no extra package). Everything comes from the same files the
dashboard reads, so Grafana, the web dashboard and the copilot can never disagree.
"""

from __future__ import annotations

import json

LEVEL = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


class _Out:
    """Collects samples per metric family; text() writes each family as one block."""

    def __init__(self):
        self.families: dict[str, tuple[str, str, list[str]]] = {}

    def add(self, name: str, help_: str, value, labels: dict | None = None, kind: str = "gauge"):
        if value is None:
            return  # unknown is left out, never reported as 0
        lab = ""
        if labels:
            esc = {k: str(v).replace("\\", "\\\\").replace('"', '\\"') for k, v in labels.items()}
            lab = "{" + ",".join(f'{k}="{v}"' for k, v in esc.items()) + "}"
        self.families.setdefault(name, (help_, kind, []))[2].append(f"{name}{lab} {float(value):g}")

    def text(self) -> str:
        lines = []
        for name, (help_, kind, samples) in self.families.items():
            lines += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}", *samples]
        return "\n".join(lines) + "\n"


def render(svc) -> str:
    o = _Out()
    st, k, c = svc.status(), svc.kpis(), svc.central()
    # ---- central layer ----
    o.add("ayojna_level", "Fail-safe level: 0=L0 normal .. 4=L4 safe mode", LEVEL.get(st["level"]))
    o.add("ayojna_leader_alive", "1 if a supervisor holds a live lease", int(st["leader_alive"]))
    o.add("ayojna_fencing_token", "Fencing token of the current leader", st["fencing_token"])
    o.add("ayojna_safe_mode", "1 while the operator safe-mode switch is on", int(bool(st["safe_mode"])))
    o.add("ayojna_state_store_redis", "1 if shared state is in Redis, 0 for files",
          int(st.get("store") == "redis"))  # fmt: skip
    for level, n in sorted(c["cycles"]["by_level"].items()):
        o.add("ayojna_cycles_total", "Supervisor cycles by level", n, {"level": level}, "counter")
    o.add("ayojna_cycles_completed_percent", "Cycles completed without a hold",
          c["cycles"]["completed_pct"])  # fmt: skip
    o.add("ayojna_failovers_total", "Leader failovers", c["failover"]["count"], kind="counter")
    if c["failover"]["last"]:
        o.add("ayojna_failover_last_seconds", "Last failover: seconds without a leader",
              c["failover"]["last"].get("gap_s"))  # fmt: skip
    o.add("ayojna_failover_max_seconds", "Worst failover gap", c["failover"]["max_gap_s"])
    o.add("ayojna_dead_letters_total", "Items in the dead-letter queue", c["dlq"]["count"], kind="counter")
    # ---- outcomes ----
    o.add("ayojna_saving_percent", "Cost saving vs keeping everything hot",
          k["twin_saving_vs_all_hot_pct"], {"scope": "replay"})  # fmt: skip
    o.add("ayojna_saving_percent", "Cost saving vs keeping everything hot", k["live_saving_pct"],
          {"scope": "live"})  # fmt: skip
    o.add("ayojna_saving_vs_best_rule_percent", "Cost saving vs the best policy-compliant rule",
          k["twin_saving_vs_best_rule_pct"], {"rule": k["best_rule"] or "none"})  # fmt: skip
    o.add("ayojna_sla_met_percent", "Share of I/Os within their latency target (replay)", k["sla_met_pct"])
    o.add("ayojna_compliance_percent", "Placements that pass policy (replay)", k["compliance_pct"])
    o.add("ayojna_moves_last_cycle", "Extents moved in the last cycle", k["moves_last_cycle"])
    # ---- intelligence + decision layers ----
    intel = svc.intel()
    live = (intel.get("live") or {}) if intel.get("available") else {}
    for step, info in (live.get("steps") or {}).items():
        if step in ("hotness", "forecast", "anomaly"):
            ms = info.get("ms")
            o.add("ayojna_model_seconds", "Model step time in the last cycle",
                  ms / 1000 if ms is not None else None, {"model": step})  # fmt: skip
            o.add("ayojna_model_fallback", "1 if the model used its fallback",
                  int(info.get("source") != "primary"), {"model": step})  # fmt: skip
    g = live.get("guards") or {}
    o.add("ayojna_anomaly_paused_volumes", "Volumes frozen by the anomaly guard", len(g.get("freeze", {})))
    o.add("ayojna_spike_held_volumes", "Volumes with demotions held for a predicted spike",
          len(g.get("no_demote", {})))  # fmt: skip
    d = svc.decision()
    s = (d.get("live") or {}).get("solver") or {}
    o.add("ayojna_solver_seconds", "Exact placement solve time (last cycle)",
          s["ms"] / 1000 if s.get("ms") is not None else None, {"solver": s.get("solver", "none")})  # fmt: skip
    o.add("ayojna_solver_optimal", "1 if the last plan was solved to optimality",
          int(s.get("status") == "optimal") if s else None)  # fmt: skip
    p = d["policy"]
    o.add("ayojna_policy_engine_opa", "1 if OPA answered the last policy check", int(p["engine"] == "opa"))
    o.add("ayojna_policy_engines_agree", "1 if OPA and the Python guard agree (-1 = OPA not asked)",
          -1 if p["agree"] is None else int(p["agree"]))  # fmt: skip
    # ---- slide 6 scorecard ----
    rep = _report(svc)
    if rep:
        for r in rep["rows"]:
            if r["status"] != "NOT MEASURED":
                o.add("ayojna_kpi_target_met", "1 if the slide 6 target is met",
                      int(r["status"] == "PASS"), {"kpi": r["kpi"]})  # fmt: skip
        o.add("ayojna_kpi_targets_met", "Slide 6 targets met", rep["passed"])
        o.add("ayojna_kpi_targets_measured", "Slide 6 targets measured", rep["measured"])
    return o.text()


def _report(svc) -> dict | None:
    try:
        return json.loads((svc.lake / "kpi_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
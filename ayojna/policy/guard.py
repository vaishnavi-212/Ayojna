"""Policy guard: which tiers each extent may live on. Runs before the optimizer,
so the optimizer can never choose an illegal placement (compliant by construction).

One rule set (parameters in config/policy.yaml), two independent engines:
  local  the Python rules below
  opa    config/policy.rego evaluated by Open Policy Agent (slide 4: "OPA · fails closed")
policy.yaml -> engine.mode (or env AYOJNA_POLICY_ENGINE):
  local  Python only
  opa    OPA decides; if OPA cannot answer, EVERY volume is frozen (fails closed)
  auto   OPA when it is reachable, otherwise Python
Whenever OPA answers, Python cross-checks it: a volume where the two engines disagree is
frozen (no moves) until someone looks. Each decision says which engine made it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from ayojna.contracts import TIER_ORDER
from ayojna.settings import CONFIG_DIR, load_config

TIER_NAMES = [t.value for t in TIER_ORDER]
_last: dict = {}  # report of the most recent decision (engine, agreement, ...)
_opa_down_until = 0.0  # after a failed call, do not retry OPA for a few seconds


def load_policy() -> dict:
    with open(Path(CONFIG_DIR) / "policy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rules(pol: dict) -> dict:
    return {k: pol[k] for k in ("legal_hold_freezes", "no_archive_classes", "sla_floor")}


def _tags(volumes) -> list[dict]:
    cfg = load_config()
    out = []
    for v in volumes:
        t = cfg.tags_for(str(v))
        out.append({"name": str(v), "sla_class": t.sla_class, "data_class": t.data_class,
                    "legal_hold": bool(t.legal_hold)})  # fmt: skip
    return out


def local_decisions(vols: list[dict], pol: dict) -> dict:
    """The Python engine: volume -> {"allowed": [...], "freeze": bool, "reasons": [...]}."""
    out = {}
    for v in vols:
        floor = TIER_NAMES.index(pol["sla_floor"].get(v["sla_class"], "archive"))
        allowed = TIER_NAMES[: floor + 1]
        reasons = [f"{v['sla_class']} SLA: not below {TIER_NAMES[floor]}"] if floor < 3 else []
        if v["data_class"] in pol["no_archive_classes"]:
            allowed = [t for t in allowed if t != "archive"]
            reasons.append(f"{v['data_class']}: no archive")
        freeze = bool(v["legal_hold"] and pol["legal_hold_freezes"])
        if freeze:
            reasons.append("legal hold: frozen")
        out[v["name"]] = {"allowed": allowed, "freeze": freeze, "reasons": reasons}
    return out


def opa_decisions(vols: list[dict], pol: dict, url: str, timeout_s: float) -> dict:
    """The OPA engine: same question, asked over OPA's REST API."""
    body = json.dumps({"input": {"policy": _rules(pol), "volumes": vols}}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/data/ayojna/decision", data=body,
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        result = json.loads(r.read()).get("result")
    if not isinstance(result, dict) or set(result) != {v["name"] for v in vols}:
        raise ValueError("OPA returned no decision for some volumes (is policy.rego loaded?)")
    return result


def _same(a: dict, b: dict) -> bool:
    return sorted(a["allowed"]) == sorted(b["allowed"]) and bool(a["freeze"]) == bool(b["freeze"])


def decide(volumes) -> tuple[dict, dict]:
    """Per-volume decisions + a report {"engine", "mode", "agree", "disagree", "error"}."""
    global _opa_down_until
    pol = load_policy()
    eng = pol.get("engine", {})
    mode = os.getenv("AYOJNA_POLICY_ENGINE", eng.get("mode", "local"))
    url = os.getenv("AYOJNA_OPA_URL", eng.get("opa_url", "http://localhost:8181"))
    vols = _tags(sorted({str(v) for v in volumes}))
    local = local_decisions(vols, pol)
    report = {"mode": mode, "engine": "local", "agree": None, "disagree": [], "error": None}
    if mode == "local":
        return local, report
    opa = None
    if mode == "opa" or time.time() >= _opa_down_until:
        try:
            opa = opa_decisions(vols, pol, url, float(eng.get("timeout_s", 2.0)))
        except Exception as exc:
            _opa_down_until = time.time() + 10.0
            report["error"] = f"{type(exc).__name__}: {exc}"[:160]
    if opa is None:
        if mode == "opa":  # fails closed: nobody we trust answered -> nothing moves
            report["engine"] = "none: frozen (fails closed)"
            return {n: {**d, "freeze": True, "reasons": d["reasons"] + ["policy engine unreachable: frozen"]}
                    for n, d in local.items()}, report  # fmt: skip
        return local, report
    report["engine"] = "opa"
    report["disagree"] = [n for n in opa if not _same(opa[n], local[n])]
    report["agree"] = not report["disagree"]
    for n in report["disagree"]:  # fail closed per volume
        opa[n] = {**opa[n], "freeze": True,
                  "reasons": list(opa[n]["reasons"]) + ["engines disagree: frozen"]}  # fmt: skip
    return opa, report


def allowed_tiers(volumes: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Returns a (n_extents, 4) bool mask of allowed tiers and one reason string per extent."""
    decisions, report = decide(np.unique(volumes))
    _last.clear()
    _last.update(report)
    mask = np.zeros((len(volumes), 4), dtype=bool)
    reasons = [""] * len(volumes)
    for vol, d in decisions.items():
        rows = np.flatnonzero(volumes == vol)
        if d["freeze"]:
            mask[rows, current[rows]] = True
        else:
            mask[np.ix_(rows, [TIER_NAMES.index(t) for t in d["allowed"]])] = True
        why = "; ".join(d["reasons"])
        for r in rows:
            reasons[r] = why
    return mask, reasons


def last_report() -> dict:
    return dict(_last)
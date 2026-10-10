"""Step 13B: Prometheus /metrics, Grafana provisioning, the scorecard API, the fair KPI."""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ayojna.api.app import create_app
from ayojna.settings import CONFIG_DIR

ROW = {"monthly_cost": 0, "saving_vs_all_hot_pct": 0, "sla_met_pct": 100.0, "gb_moved": 0,
       "compliance_pct": 100.0, "hours_over_hot_capacity": 0}  # fmt: skip


def _populated(tmp_path):
    """A state + lake folder like the one a demo run leaves behind."""
    state, lake = tmp_path / "state", tmp_path / "lake"
    state.mkdir()
    lake.mkdir()
    now = time.time()
    (state / "lease.json").write_text(json.dumps({"owner": "primary", "token": 2, "expires": now + 6,
                                                   "renewed_at": now - 2, "store": "redis"}))  # fmt: skip
    events = [{"event": "cycle", "level": "L1", "moves_done": 5},
              {"event": "failover", "from": "primary", "to": "replica", "gap_s": 8.6},
              {"event": "cycle", "level": "L0", "moves_done": 3}]  # fmt: skip
    (state / "audit.jsonl").write_text("".join(json.dumps({"ts": now, **e}) + "\n" for e in events))
    (state / "dlq.jsonl").write_text(json.dumps({"ts": now, "kind": "step", "step": "plan"}) + "\n")
    (state / "last_intel.json").write_text(json.dumps({
        "hour": 10, "steps": {"hotness": {"source": "primary", "ms": 300},
                              "forecast": {"source": "fallback", "ms": 90}},
        "guards": {"freeze": {"proj_0": "anomaly pause: x"}, "no_demote": {}}}))  # fmt: skip
    (state / "last_plan.json").write_text(json.dumps({
        "plan": {"envelope": {"run_id": "r"}, "moves": []},
        "decision": {"solver": {"solver": "highs", "status": "optimal", "ms": 80}}}))  # fmt: skip
    summary = [{**ROW, "strategy": "all_hot", "monthly_cost": 20.0},
               {**ROW, "strategy": "lfu", "monthly_cost": 7.8, "compliance_pct": 97.5},
               {**ROW, "strategy": "lfu+policy", "monthly_cost": 9.4},
               {**ROW, "strategy": "ayojna", "monthly_cost": 7.9, "saving_vs_all_hot_pct": 60.5}]
    (lake / "scoreboard.json").write_text(json.dumps({"summary": summary}))
    (lake / "kpi_report.json").write_text(json.dumps({
        "scored_hours": [120, 167], "passed": 1, "measured": 2, "rows": [
            {"kpi": "Compliance", "target": "100%", "value": "100%", "status": "PASS", "note": ""},
            {"kpi": 'Hot "hit" ratio', "target": "x", "value": "y", "status": "MISS", "note": ""},
            {"kpi": "Forecast", "target": "x", "value": "-", "status": "NOT MEASURED", "note": ""}]}))
    return TestClient(create_app(state, lake))


def _families(text: str) -> list[str]:
    return [line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")]


def test_fair_best_rule_ignores_rules_that_break_policy(tmp_path):
    k = _populated(tmp_path).get("/api/kpis").json()
    assert k["best_rule"] == "lfu+policy"  # lfu is cheaper only by breaking policy
    assert k["twin_saving_vs_best_rule_pct"] == round(100 * (1 - 7.9 / 9.4), 2)


def test_metrics_are_valid_prometheus_text(tmp_path):
    r = _populated(tmp_path).get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    fams = _families(text)
    assert len(fams) == len(set(fams))  # one HELP/TYPE per family
    names = [re.match(r"[a-zA-Z_:][a-zA-Z0-9_:]*", ln).group(0)
             for ln in text.splitlines() if ln and not ln.startswith("#")]  # fmt: skip
    blocks = [n for i, n in enumerate(names) if i == 0 or names[i - 1] != n]
    assert len(blocks) == len(set(blocks))  # each family's samples are contiguous
    assert 'ayojna_cycles_total{level="L1"} 1' in text and "ayojna_failover_last_seconds 8.6" in text
    assert 'ayojna_kpi_target_met{kpi="Hot \\"hit\\" ratio"} 0' in text  # quotes escaped
    assert "Forecast" not in text  # NOT MEASURED is left out, not reported as 0
    assert "ayojna_anomaly_paused_volumes 1" in text and 'ayojna_solver_seconds{solver="highs"} 0.08' in text
    promtool = shutil.which("promtool")
    if promtool:  # the real Prometheus linter, when installed
        out = subprocess.run([promtool, "check", "metrics"], input=text, text=True,
                             capture_output=True)  # fmt: skip
        assert out.returncode == 0, out.stdout + out.stderr


def test_metrics_on_an_empty_system(tmp_path):
    c = TestClient(create_app(tmp_path / "s", tmp_path / "l"))
    r = c.get("/metrics")
    assert r.status_code == 200 and "ayojna_dead_letters_total 0" in r.text
    assert c.get("/api/kpi-report").json() == {"available": False}


def test_scorecard_api(tmp_path):
    d = _populated(tmp_path).get("/api/kpi-report").json()
    assert d["available"] and d["passed"] == 1 and len(d["rows"]) == 3


def test_grafana_dashboard_only_uses_metrics_we_export(tmp_path):
    text = _populated(tmp_path).get("/metrics").text
    exported = set(_families(text))
    dash = json.loads((Path(CONFIG_DIR) / "grafana/dashboards/json/ayojna.json").read_text("utf-8"))
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids)) and dash["uid"] == "ayojna"
    for p in dash["panels"]:
        for t in p["targets"]:
            name = re.match(r"[a-z_]+", t["expr"]).group(0)
            assert name in exported, f"panel '{p['title']}' queries unknown metric {name}"
            assert p["datasource"]["uid"] == "prometheus"


def test_provisioning_files_parse():
    cfg = Path(CONFIG_DIR)
    prom = yaml.safe_load((cfg / "prometheus.yml").read_text())
    assert prom["scrape_configs"][0]["metrics_path"] == "/metrics"
    ds = yaml.safe_load((cfg / "grafana/datasources/prometheus.yml").read_text())
    assert ds["datasources"][0]["uid"] == "prometheus"
    prov = yaml.safe_load((cfg / "grafana/dashboards/ayojna.yml").read_text())
    assert prov["providers"][0]["options"]["path"].endswith("/dashboards/json")
    compose = yaml.safe_load((cfg.parent / "docker-compose.yml").read_text())
    assert {"prometheus", "grafana"} <= set(compose["services"])
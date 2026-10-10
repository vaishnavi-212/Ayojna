"""Step 14B: the validation API (generalization + robustness) and the docs stay honest."""

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from ayojna.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
STAT = {"median": 14.53, "min": 10.02, "max": 15.82, "runs": [15.82, 14.67, 13.24, 14.53, 10.02]}


def test_validation_api_empty_then_populated(tmp_path):
    lake = tmp_path / "lake"
    c = TestClient(create_app(tmp_path / "state", lake))
    assert c.get("/api/validation").json() == {"available": False, "generalization": [], "robustness": None}
    lake.mkdir()
    (lake / "robustness.json").write_text(json.dumps({"runs": 5, "cut_vs_best_compliant_rule_pct": STAT}))
    gen = [{"experiment": "MSR -> unseen MSR volumes", "test_volumes": ["mds_0"],
            "hotness": {"ml_macro_f1": 0.665, "rule_macro_f1": 0.625},
            "planner": {"saving_vs_all_hot_pct": 33.6, "sla_met_pct": 92.57}}]  # fmt: skip
    (lake / "generalization.json").write_text(json.dumps(gen))
    d = c.get("/api/validation").json()
    assert d["available"] and d["robustness"]["cut_vs_best_compliant_rule_pct"]["median"] == 14.53
    assert d["generalization"][0]["test_volumes"] == ["mds_0"]


def test_dashboard_has_the_validation_panel():
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'id="robBox"' in html and 'id="genBox"' in html and "/api/validation" in html


def test_docs_do_not_overclaim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ppt = (ROOT / "docs/PPT_CHANGES.md").read_text(encoding="utf-8")
    for doc in (readme, ppt):
        assert "14.5" in doc and "10.0" in doc and "15.8" in doc  # the headline is a median with its range
        assert "92.6" in doc  # the cross-volume limitation is stated
    assert not re.search(r"\bReact\b", readme) and "OpenTelemetry" not in readme and "DuckDB" not in readme
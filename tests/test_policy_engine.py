"""Step 11B: policy-as-code. OPA (Rego) and the Python guard decide the same, fail closed."""

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ayojna.api.app import create_app
from ayojna.policy import guard
from ayojna.settings import CONFIG_DIR, load_config

VOLS = np.array(["web_0", "src1_2", "usr_0", "ts_0", "hm_0", "unknown_vol"])
CUR = np.array([0, 1, 2, 0, 3, 1])


def _fake_opa(tamper: bool = False):
    """A stand-in OPA server: answers like policy.rego (or deliberately wrong)."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            inp = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["input"]
            result = guard.local_decisions(inp["volumes"], inp["policy"])
            if tamper:  # a bad policy push: web_0 suddenly allowed on archive
                result["web_0"] = {**result["web_0"], "allowed": ["hot", "warm", "cold", "archive"]}
            body = json.dumps({"result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _use(monkeypatch, mode, url="http://127.0.0.1:9"):
    monkeypatch.setenv("AYOJNA_POLICY_ENGINE", mode)
    monkeypatch.setenv("AYOJNA_OPA_URL", url)
    guard._opa_down_until = 0.0


def test_local_rules(monkeypatch):
    _use(monkeypatch, "local")
    mask, reasons = guard.allowed_tiers(VOLS, CUR)
    assert mask[3].tolist() == [True, True, False, False]  # ts_0 oltp: not below warm
    assert not mask[2, 3] and "pii: no archive" in reasons[2]  # usr_0
    assert mask[1].tolist() == [False, True, False, False]  # src1_2 legal hold: frozen
    assert guard.last_report()["engine"] == "local"


def test_opa_agreeing_is_used_and_matches_python(monkeypatch):
    srv, url = _fake_opa()
    try:
        _use(monkeypatch, "local")
        local_mask, _ = guard.allowed_tiers(VOLS, CUR)
        _use(monkeypatch, "auto", url)
        mask, _ = guard.allowed_tiers(VOLS, CUR)
        r = guard.last_report()
        assert r["engine"] == "opa" and r["agree"] is True
        assert (mask == local_mask).all()
    finally:
        srv.shutdown()


def test_engines_disagree_volume_is_frozen(monkeypatch):
    srv, url = _fake_opa(tamper=True)
    try:
        _use(monkeypatch, "opa", url)
        mask, reasons = guard.allowed_tiers(VOLS, CUR)
        assert guard.last_report()["disagree"] == ["web_0"]
        assert mask[0].tolist() == [True, False, False, False]  # stays where it is
        assert "engines disagree" in reasons[0]
        assert mask[2].tolist() == [True, True, True, False]  # others unaffected
    finally:
        srv.shutdown()


def test_opa_down_fails_closed_in_opa_mode_and_falls_back_in_auto(monkeypatch):
    _use(monkeypatch, "opa")  # nothing listens on port 9
    mask, reasons = guard.allowed_tiers(VOLS, CUR)
    assert (mask.sum(axis=1) == 1).all() and mask[np.arange(len(CUR)), CUR].all()
    assert guard.last_report()["engine"].startswith("none")
    assert "unreachable" in reasons[0]
    _use(monkeypatch, "auto")
    mask, _ = guard.allowed_tiers(VOLS, CUR)
    assert guard.last_report()["engine"] == "local" and mask[3].sum() == 2


def test_real_opa_agrees_with_python_when_installed(tmp_path):
    opa = shutil.which("opa")
    if opa is None:
        return  # OPA binary not installed here: the fake-server tests above still run
    names = sorted(load_config().volumes) + ["unknown_vol"]
    vols = guard._tags(names)
    pol = guard._rules(guard.load_policy())
    (tmp_path / "in.json").write_text(json.dumps({"policy": pol, "volumes": vols}))
    out = subprocess.run(
        [opa, "eval", "-d", str(Path(CONFIG_DIR) / "policy.rego"), "-i", str(tmp_path / "in.json"),
         "--format", "json", "data.ayojna.decision"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    rego = json.loads(out.stdout)["result"][0]["expressions"][0]["value"]
    py = guard.local_decisions(vols, pol)
    assert set(rego) == set(py)
    for n in py:
        assert guard._same(rego[n], py[n]), n
        assert rego[n]["reasons"] == py[n]["reasons"], n


def test_decision_api(tmp_path, monkeypatch):
    _use(monkeypatch, "local")
    state = tmp_path / "state"
    state.mkdir()
    (state / "last_plan.json").write_text(json.dumps({
        "plan": {"envelope": {"run_id": "r"}, "moves": []},
        "decision": {"solver": {"solver": "highs", "status": "optimal", "objective": 1.0,
                                "greedy_objective": 1.5, "ms": 9},
                     "bandit": {"available": True, "mode": "shadow", "suggested": {"web_0": "balanced"}}},
    }))  # fmt: skip
    d = TestClient(create_app(state, tmp_path / "lake")).get("/api/decision").json()
    assert d["available"] and d["policy"]["engine"] == "local"
    assert d["live"]["solver"]["solver"] == "highs" and d["replay"] is None
    assert "src1_2" in d["policy"]["volumes"] and d["policy"]["volumes"]["src1_2"]["freeze"]
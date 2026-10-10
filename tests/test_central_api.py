"""Step 12B: the central layer as the operator sees it: L4, safe mode, failover, dead letters."""

import json
import time

from fastapi.testclient import TestClient

from ayojna.api.app import create_app


def _client(tmp_path):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return TestClient(create_app(state, tmp_path / "lake")), state


def _lease(state, expires_in: float, store: str = "redis"):
    now = time.time()
    (state / "lease.json").write_text(json.dumps({
        "owner": "primary", "token": 3, "expires": now + expires_in,
        "renewed_at": now + expires_in - 8, "store": store}))  # fmt: skip


def _events(state, name, events):
    with open(state / name, "a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps({"ts": time.time(), **e}) + "\n")


def test_nothing_ran_yet(tmp_path):
    c, _ = _client(tmp_path)
    assert c.get("/api/central").json()["available"] is False
    assert c.get("/api/status").json()["level"] is None


def test_live_leader_keeps_its_cycle_level(tmp_path):
    c, state = _client(tmp_path)
    _lease(state, 6)
    _events(state, "audit.jsonl", [{"event": "cycle", "level": "L1", "moves_done": 3}])
    s = c.get("/api/status").json()
    assert s["level"] == "L1" and s["l4_reason"] is None and s["store"] == "redis"


def test_no_supervisor_alive_is_l4(tmp_path):
    c, state = _client(tmp_path)
    _lease(state, -30)  # last heartbeat long ago, nobody took over
    _events(state, "audit.jsonl", [{"event": "cycle", "level": "L0", "moves_done": 0}])
    s = c.get("/api/status").json()
    assert s["level"] == "L4" and "no supervisor alive" in s["l4_reason"]
    assert c.get("/api/kpis").json()["level"] == "L4"


def test_operator_safe_mode_switch(tmp_path):
    c, state = _client(tmp_path)
    _lease(state, 6)
    _events(state, "audit.jsonl", [{"event": "cycle", "level": "L0"}])
    r = c.post("/api/safe-mode", json={"on": True, "reason": "storage maintenance"}).json()
    assert r["on"] and json.loads((state / "safe_mode.json").read_text())["on"]
    s = c.get("/api/status").json()
    assert s["level"] == "L4" and "storage maintenance" in s["l4_reason"]
    c.post("/api/safe-mode", json={"on": False})
    assert c.get("/api/status").json()["level"] == "L0"
    assert '"safe_mode"' in (state / "audit.jsonl").read_text()  # the switch is audited
    assert c.post("/api/safe-mode", json={"on": True, "reason": "x" * 500}).status_code == 422


def test_failover_availability_and_dead_letters(tmp_path):
    c, state = _client(tmp_path)
    _lease(state, 6)
    _events(state, "audit.jsonl", [
        {"event": "cycle", "level": "L0"}, {"event": "cycle", "level": "L1"},
        {"event": "failover", "from": "primary", "to": "replica", "gap_s": 9.9, "resumed_run": "r0"},
        {"event": "failover", "from": "primary", "to": "replica", "gap_s": 8.5, "resumed_run": "r1"},
        {"event": "cycle", "level": "L3"}, {"event": "cycle", "level": "L1"},
    ])  # fmt: skip
    _events(state, "dlq.jsonl", [
        {"kind": "step", "step": "anomaly", "error": "RuntimeError: x", "handled_by": "fallback"},
        {"kind": "move", "key": "r1:web_0:7:cold", "status": "rolled_back", "note": "sha mismatch"},
    ])  # fmt: skip
    d = c.get("/api/central").json()
    assert d["failover"]["count"] == 2 and d["failover"]["max_gap_s"] == 9.9
    assert d["failover"]["last"]["resumed_run"] == "r1" and d["failover"]["target_s"] == 10
    assert d["cycles"]["total"] == 4 and d["cycles"]["held_L3"] == 1
    assert d["cycles"]["completed_pct"] == 75.0
    assert d["dlq"]["count"] == 2 and d["dlq"]["recent"][0]["kind"] == "move"  # newest first


def test_safe_mode_cli(tmp_path, capsys):
    from argparse import Namespace

    from ayojna.supervisor.safe_mode import main

    main(Namespace(action="on", reason="drill", state=str(tmp_path)))
    assert json.loads((tmp_path / "safe_mode.json").read_text())["reason"] == "drill"
    main(Namespace(action="off", reason="", state=str(tmp_path)))
    assert json.loads((tmp_path / "safe_mode.json").read_text())["on"] is False
"""Step 12A: shared state (files or Redis), measured failover, dead letters, safe mode."""

import json
import os
import shutil
import socket
import subprocess
import time

import pandas as pd
import pytest

from ayojna.supervisor.runner import Step, run_cycle
from ayojna.supervisor.state import RedisStateStore, StateStore, make_state_store


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_url():
    """A real Redis: AYOJNA_TEST_REDIS if set, else a throwaway redis-server, else skip."""
    url = os.getenv("AYOJNA_TEST_REDIS")
    if url:
        yield url
        return
    exe = shutil.which("redis-server")
    if exe is None:
        yield None
        return
    port = _free_port()
    proc = subprocess.Popen([exe, "--port", str(port), "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    time.sleep(0.5)
    yield f"redis://127.0.0.1:{port}/0"
    proc.terminate()


def _stores(url, tmp_path, ttl=0.5):
    if url is None:
        pytest.skip("no Redis here (run docker compose up -d redis, or set AYOJNA_TEST_REDIS)")
    a = RedisStateStore(url, tmp_path, ttl, prefix=f"t{time.time_ns()}")
    b = RedisStateStore(url, tmp_path, ttl, prefix=a.prefix)
    return a, b


# ---------- file store ----------
def test_file_store_measures_a_crash_failover_but_not_a_clean_handover(tmp_path):
    s = StateStore(tmp_path, lease_ttl_s=0.3)
    t1 = s.acquire("primary")
    assert s.acquire("replica") is None  # live lease: only one leader
    time.sleep(0.4)  # primary "crashed": no heartbeat
    t2 = s.acquire("replica")
    assert t2 == t1 + 1 and not s.is_current(t1) and s.is_current(t2)
    assert s.last_takeover["from"] == "primary" and 0.3 <= s.last_takeover["gap_s"] < 2
    s.release("replica", t2)  # clean stop
    t3 = s.acquire("primary")
    assert t3 == t2 + 1 and s.last_takeover is None  # a restart, not a failover


def test_dead_letters_and_safe_mode_flag(tmp_path):
    s = StateStore(tmp_path)
    assert s.dead_letters() == [] and s.safe_mode() is None
    s.dead_letter({"kind": "step", "step": "plan", "error": "boom"})
    assert s.dead_letters()[0]["error"] == "boom"
    (tmp_path / "safe_mode.json").write_text(json.dumps({"on": True, "reason": "maintenance"}))
    assert s.safe_mode()["reason"] == "maintenance"


def test_failed_step_goes_to_the_dead_letter_queue_with_its_real_error(tmp_path):
    s = StateStore(tmp_path)
    token = s.acquire("primary")

    def boom(ctx):
        raise RuntimeError("disk on fire")

    r = run_cycle("r1", [Step("a", boom, retries=1, fallback=lambda ctx: 1)], s, token)
    assert r["level"] == "L1"
    dlq = s.dead_letters()
    assert len(dlq) == 1 and dlq[0]["step"] == "a" and dlq[0]["attempts"] == 2
    assert "disk on fire" in dlq[0]["error"] and dlq[0]["handled_by"] == "fallback"


def test_safe_mode_plans_but_moves_nothing(tmp_path):
    from ayojna.ingest.build import build
    from ayojna.ingest.synth import generate
    from ayojna.supervisor.pipeline import build_steps

    generate(tmp_path / "raw", days=3, seed=4)
    build(tmp_path / "raw", tmp_path / "eh.csv")
    store = StateStore(tmp_path / "state", lease_ttl_s=60)
    (tmp_path / "state" / "safe_mode.json").write_text(json.dumps({"on": True, "reason": "drill"}))
    token = store.acquire("primary")
    steps = build_steps(str(tmp_path / "eh.csv"), str(tmp_path / "f.csv"), str(tmp_path / "none"),
                        state_dir=str(tmp_path / "state"), tiers_root=str(tmp_path / "tiers"),
                        store=store)  # fmt: skip
    r = run_cycle("r1", steps, store, token)
    ex = r["outputs"]["execute"]
    assert ex["mode"] == "safe-mode" and ex["results"] == [] and len(r["outputs"]["plan"].moves) > 0


def test_auto_store_falls_back_to_files_when_redis_is_down(tmp_path):
    s = make_state_store("auto", tmp_path, url=f"redis://127.0.0.1:{_free_port()}/0")
    assert s.kind == "file"
    with pytest.raises(Exception):
        make_state_store("redis", tmp_path, url=f"redis://127.0.0.1:{_free_port()}/0")


# ---------- Redis store (real server) ----------
def test_redis_lease_is_exclusive_fenced_and_measures_failover(redis_url, tmp_path):
    a, b = _stores(redis_url, tmp_path)
    t1 = a.acquire("primary")
    assert t1 and b.acquire("replica") is None
    assert a.renew("primary", t1) and a.acquire("primary") == t1  # same owner keeps its token
    time.sleep(0.7)  # primary stops heartbeating
    t2 = b.acquire("replica")
    assert t2 == t1 + 1 and b.is_current(t2) and not a.is_current(t1)
    assert not a.renew("primary", t1)  # the old leader is fenced out
    assert b.last_takeover["from"] == "primary" and b.last_takeover["gap_s"] >= 0.5
    assert json.loads((tmp_path / "lease.json").read_text())["store"] == "redis"
    b.release("replica", t2)
    assert a.acquire("primary") == t2 + 1 and a.last_takeover is None


def test_redis_checkpoints_active_run_audit_and_dead_letters(redis_url, tmp_path):
    a, b = _stores(redis_url, tmp_path)
    df = pd.DataFrame({"x": [1, 2, 3]})
    a.save_step("run-1", "features", df, {"source": "primary", "note": ""})
    out, info = b.load_step("run-1", "features")  # the replica reads what the leader saved
    assert out.equals(df) and info["source"] == "primary"
    assert b.load_step("run-1", "plan") == (None, None)
    a.set_active_run("run-1")
    assert b.active_run() == "run-1"
    a.set_active_run(None)
    assert b.active_run() is None
    a.dead_letter({"kind": "move", "key": "k1", "status": "rolled_back"})
    assert b.dead_letters()[-1]["key"] == "k1"
    a.audit({"event": "cycle", "level": "L0"})
    assert "cycle" in (tmp_path / "audit.jsonl").read_text()


def test_redis_runs_a_whole_cycle_with_resume(redis_url, tmp_path):
    a, b = _stores(redis_url, tmp_path, ttl=30)
    token = a.acquire("primary")
    ran = []
    steps = [Step("one", lambda ctx: ran.append(1) or 1), Step("two", lambda ctx: ctx["one"] + 1)]
    a.save_step("r9", "one", 41, {"source": "primary", "note": ""})  # finished before a crash
    r = run_cycle("r9", steps, a, token)
    assert ran == [] and r["outputs"]["two"] == 42 and r["steps"]["one"]["resumed"]
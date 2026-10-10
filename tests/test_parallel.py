"""Step 10B: the three models run in parallel under the supervisor, and their guards act."""

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from ayojna.api.app import create_app
from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.planner.guards import apply_guards, guards_from
from ayojna.supervisor.pipeline import build_steps
from ayojna.supervisor.runner import Parallel, Step, run_cycle
from ayojna.supervisor.state import StateStore


def _store(root):
    store = StateStore(root, lease_ttl_s=60)
    return store, store.acquire("primary")


def _slow(value, s=0.3):
    def fn(ctx):
        time.sleep(s)
        return value

    return fn


# ---------- runner ----------
def test_parallel_group_runs_at_the_same_time(tmp_path):
    store, token = _store(tmp_path)
    steps = [Parallel("models", [Step("a", _slow(1)), Step("b", _slow(2)), Step("c", _slow(3))]),
             Step("sum", lambda ctx: ctx["a"] + ctx["b"] + ctx["c"])]  # fmt: skip
    t0 = time.perf_counter()
    r = run_cycle("r1", steps, store, token)
    assert time.perf_counter() - t0 < 0.75  # three 0.3 s steps, not 0.9 s
    assert r["outputs"]["sum"] == 6 and r["level"] == "L0"
    g = r["groups"]["models"]
    assert g["sum_ms"] > g["wall_ms"] and all(r["steps"][n]["ms"] >= 250 for n in "abc")


def test_one_member_falls_back_the_others_stay_primary(tmp_path):
    store, token = _store(tmp_path)

    def boom(ctx):
        raise RuntimeError("model down")

    flaky = Step("b", boom, retries=0, fallback=lambda ctx: None, critical=False)
    group = Parallel("models", [Step("a", _slow(1, 0.05)), flaky])
    r = run_cycle("r1", [group], store, token)
    assert r["level"] == "L1"
    assert r["steps"]["a"]["source"] == "primary" and r["steps"]["b"]["source"] == "fallback"


def test_critical_member_failure_holds_the_cycle(tmp_path):
    store, token = _store(tmp_path)
    later = []

    def boom(ctx):
        raise RuntimeError("no hotness")

    steps = [Parallel("models", [Step("hot", boom, retries=0), Step("ok", _slow(1, 0.01))]),
             Step("plan", lambda ctx: later.append(1))]  # fmt: skip
    r = run_cycle("r1", steps, store, token)
    assert r["level"] == "L3" and later == []


def test_group_members_resume_from_checkpoint(tmp_path):
    store, token = _store(tmp_path)
    store.save_step("r1", "a", 41, {"source": "primary", "note": ""})
    ran = []
    group = Parallel("models", [Step("a", lambda ctx: ran.append("a")),
                                Step("b", lambda ctx: ran.append("b") or 1)])  # fmt: skip
    r = run_cycle("r1", [group], store, token)
    assert ran == ["b"] and r["outputs"]["a"] == 41 and r["steps"]["a"]["resumed"]


# ---------- guards ----------
def test_guards_freeze_or_block_demotions_but_never_block_compliance():
    vols = np.array(["v1", "v1", "v2", "v3"])
    current = np.array([0, 2, 0, 3])
    allowed = np.ones((4, 4), dtype=bool)
    allowed[3, 3] = False  # v3 sits on a forbidden tier: it must be allowed to leave
    g = {"freeze": {"v1": "anomaly pause: x", "v3": "anomaly pause: y"}, "no_demote": {"v2": "spike"}}
    ok, notes = apply_guards(allowed, current, vols, g)
    assert ok[0].tolist() == [True, False, False, False]  # frozen in place
    assert ok[1].tolist() == [False, False, True, False]
    assert ok[2].tolist() == [True, False, False, False]  # spike: may not go slower than hot
    assert ok[3].tolist() == [True, True, True, False]  # compliance first: guard skipped
    assert notes[0].startswith("anomaly") and notes[3] == ""


def test_guards_from_model_outputs():
    import pandas as pd

    an = pd.DataFrame({"volume": ["a", "b"], "anomaly": [True, False], "reason": ["burst", "ok"]})
    fc = pd.DataFrame({"volume": ["a", "c"], "spike": [True, True],
                       "io_next24": [9e5, 5e5], "io_last24": [1e5, 1e5]})  # fmt: skip
    g = guards_from(fc, an)
    assert list(g["freeze"]) == ["a"] and list(g["no_demote"]) == ["c"]  # a: pause wins
    assert guards_from(None, None) == {"freeze": {}, "no_demote": {}}


# ---------- pipeline + API ----------
def test_pipeline_runs_models_in_parallel_and_reports(tmp_path):
    generate(tmp_path / "raw", days=4, seed=4)
    build(tmp_path / "raw", tmp_path / "eh.csv")
    store, token = _store(tmp_path / "state")
    steps = build_steps(str(tmp_path / "eh.csv"), str(tmp_path / "f.csv"), str(tmp_path / "none"),
                        state_dir=str(tmp_path / "state"), tiers_root=str(tmp_path / "tiers"))  # fmt: skip
    group = next(s for s in steps if isinstance(s, Parallel))
    assert [s.name for s in group.steps] == ["hotness", "forecast", "anomaly"]
    r = run_cycle("r1", steps, store, token)
    assert r["level"] in ("L0", "L1") and "plan" in r["outputs"]
    assert r["steps"]["anomaly"]["source"] == "fallback"  # no trained detector: rule
    intel = json.loads((tmp_path / "state" / "last_intel.json").read_text())
    assert len(intel["forecast"]) == len(intel["anomaly"]) > 0
    assert intel["parallel"]["models"]["steps"] == ["hotness", "forecast", "anomaly"]
    assert set(intel["guards"]) == {"freeze", "no_demote"}

    c = TestClient(create_app(tmp_path / "state", tmp_path / "lake"))
    d = c.get("/api/intel").json()
    assert d["available"] and d["live"]["hour"] == intel["hour"]
    assert TestClient(create_app(tmp_path / "x", tmp_path / "y")).get("/api/intel").json() == {
        "available": False
    }
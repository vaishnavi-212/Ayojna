"""Step 13A: fair baselines (LFU, policy-compliant rules), the slide 6 scorecard, chaos drill."""

import json

import numpy as np
import pandas as pd

from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.models.features import build_features
from ayojna.twin.sim import Twin, TwinView
from ayojna.twin.strategies import LfuCapacity, LruCapacity, PolicyAware
from ayojna.validate.chaos import drill
from ayojna.validate.kpi_report import kpi_report, to_markdown


def _view(ios, placement, cap=(1, 1, 10, 10), volumes=None):
    n = ios.shape[1]
    return TwinView(ios.shape[0], ios, np.asarray(placement), np.array(cap),
                    np.array(volumes or ["web_0"] * n), np.arange(n))  # fmt: skip


def test_lfu_keeps_the_busiest_not_the_latest():
    # extent 0: 50 I/Os two hours ago; extent 1: 1 I/O in the last hour; extent 2: idle
    ios = np.array([[50, 0, 0], [0, 0, 0], [0, 1, 0]])
    lfu = LfuCapacity(window_hours=72).decide(_view(ios, [0, 0, 0]))
    lru = LruCapacity().decide(_view(ios, [0, 0, 0]))
    assert lfu[0] == 0 and lfu[1] == 1  # busiest on hot
    assert lru[1] == 0 and lru[0] == 1  # LRU prefers the most recent
    assert lfu[2] == 2  # never touched, was hot -> cold


def test_policy_aware_baseline_is_compliant():
    class Archive:  # a rule that wants everything on archive
        name = "archive_all"

        def decide(self, view):
            return np.full(view.placement.shape, 3)

    vols = ["ts_0", "src1_2", "prn_0"]  # oltp (floor warm), legal hold (frozen), backup
    out = PolicyAware(Archive()).decide(_view(np.zeros((2, 3)), [0, 1, 0], volumes=vols))
    assert out.tolist() == [1, 1, 3]  # nearest allowed: warm; frozen stays; backup may archive


def test_twin_reports_io_accounting():
    rows = [{"volume": "web_0", "extent_id": e, "hour": h, "reads": 5 + e}
            for h in range(4) for e in range(3)]  # fmt: skip
    eh = pd.DataFrame(rows).assign(writes=0, read_bytes=4096, write_bytes=0, avg_io_size=4096,
                                   rand_ratio=0.5)  # fmt: skip
    m = Twin.from_extent_hourly(eh).run(LruCapacity())
    assert (m["ios"] == 18).all() and (m["hot_ios"] <= m["ios"]).all()
    assert (m["sla_ios"] <= m["ios"]).all()


def test_kpi_report_and_chaos_drill(tmp_path):
    generate(tmp_path / "raw", days=5, seed=6)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    feats = build_features(eh)
    feats.to_csv(tmp_path / "f.csv", index=False)
    lake, state = tmp_path / "lake", tmp_path / "state"
    rep = drill(str(tmp_path / "eh.csv"), str(tmp_path / "f.csv"), str(tmp_path / "none"),
                cycles=4, seed=1, state_dir=state)  # fmt: skip
    assert rep["completed_pct"] == 100.0 and rep["unsafe_cycles"] == 0
    assert [r["fault"] for r in rep["rows"]] == ["none", "corrupt", "hotness", "forecast"]
    assert rep["rows"][2]["level"] == "L3" and rep["rows"][2]["moves_done"] == 0  # held safely
    lake.mkdir()
    (lake / "chaos_report.json").write_text(json.dumps(rep))
    out = kpi_report(eh, feats, str(tmp_path / "none"), str(state), str(lake))
    kpis = {r["kpi"]: r for r in out["rows"]}
    assert kpis["Compliance"]["status"] == "PASS"
    assert kpis["Hotness accuracy"]["status"] == "NOT MEASURED"  # no leaderboard: said, not faked
    assert kpis["Availability under failure"]["value"] == "100.0%"
    assert {"lfu", "lfu+policy", "ayojna"} <= set(out["strategies"])
    assert out["measured"] == sum(r["status"] != "NOT MEASURED" for r in out["rows"])
    assert (lake / "kpi_report.json").exists() and "| Measure |" in to_markdown(out)
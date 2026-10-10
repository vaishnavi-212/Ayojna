"""Step 14A: Alibaba ingest, cross-workload generalization, robustness of the headline."""

import io
import json
import tarfile

import numpy as np
import pandas as pd

from ayojna.ingest.alibaba import build_alibaba
from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.io import read_table
from ayojna.models.features import build_features
from ayojna.validate.generalize import generalize, split_volumes, to_markdown
from ayojna.validate.robustness import robustness

HOUR_US = 3_600_000_000
T0 = 1_577_808_000_000_000  # 2020-01-01, like the real trace


def _alibaba_rows(days=5):
    """3 devices; device 7 writes sequentially, device 3 reads at random, device 9 is rare."""
    rng = np.random.default_rng(0)
    rows = []
    for h in range(days * 24):
        base = T0 + h * HOUR_US
        for k in range(20):  # sequential 4 KiB writes, extent 0
            rows.append((7, "W", k * 4096, 4096, base + k))
        for k in range(15):  # random reads over 3 extents
            rows.append((3, "R", int(rng.integers(0, 3 * 2**28)), 8192, base + 100 + k))
        if h % 10 == 0:
            rows.append((9, "R", 0, 4096, base + 500))
    rows.sort(key=lambda r: r[4])  # the real trace is time-ordered across devices
    return "\n".join(",".join(map(str, r)) for r in rows) + "\n"


def _tar(tmp_path, text):
    path = tmp_path / "alibaba_block_traces_2020.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        data = text.encode()
        info = tarfile.TarInfo("alibaba_block_traces_2020/io_traces.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


def test_alibaba_streams_from_the_archive_and_stops_early(tmp_path):
    src = _tar(tmp_path, _alibaba_rows(days=5))
    eh = build_alibaba(str(src), str(tmp_path / "eh.csv"), devices=[7, 3], days=4, chunksize=500)
    assert set(eh["volume"]) == {"ali_7", "ali_3"} and eh["hour"].max() == 95  # 4 days only
    seq = eh[eh["volume"] == "ali_7"]
    assert (seq["writes"] == 20).all() and seq["rand_ratio"].max() <= 1 / 20 + 1e-9  # sequential
    rnd = eh[eh["volume"] == "ali_3"]
    assert (rnd["reads"].groupby(rnd["hour"]).sum() == 15).all() and rnd["rand_ratio"].min() > 0.5
    assert len(read_table(tmp_path / "eh.csv")) == len(eh)


def test_alibaba_first_n_devices_from_a_plain_csv(tmp_path):
    (tmp_path / "io_traces.csv").write_text(_alibaba_rows(days=2))
    eh = build_alibaba(str(tmp_path / "io_traces.csv"), str(tmp_path / "eh.csv"), first=2, days=2)
    assert eh["volume"].nunique() == 2 and eh["volume"].str.startswith("ali_").all()


def test_a_quiet_device_does_not_cut_the_window(tmp_path):
    (tmp_path / "io_traces.csv").write_text(_alibaba_rows(days=3))
    eh = build_alibaba(str(tmp_path / "io_traces.csv"), str(tmp_path / "eh.csv"), devices=[9, 7], days=3)
    assert eh["hour"].max() == 71  # device 9 is silent at the end, but the trace is not over


def test_generalization_trains_on_some_volumes_and_tests_on_others(tmp_path):
    generate(tmp_path / "raw", days=6, seed=8)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    a, b = split_volumes(eh["volume"].unique().tolist())
    assert set(a).isdisjoint(b) and sorted(a + b) == sorted(eh["volume"].unique())
    r = generalize(eh[eh["volume"].isin(a)], eh[eh["volume"].isin(b)], "test", tmp_path / "w")
    assert r["train_volumes"] == sorted(a) and r["test_volumes"] == sorted(b)
    assert 0 <= r["hotness"]["ml_macro_f1"] <= 1 and r["planner"]["compliance_pct"] == 100.0
    assert "| test |" in to_markdown([r])


def test_robustness_reports_median_and_range_and_the_scorecard_uses_it(tmp_path):
    from ayojna.validate.kpi_report import kpi_report

    generate(tmp_path / "raw", days=5, seed=9)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    feats = build_features(eh)
    rep = robustness(eh, feats, str(tmp_path / "none"), runs=2)
    c = rep["cut_vs_best_compliant_rule_pct"]
    assert len(c["runs"]) == 2 and c["min"] <= c["median"] <= c["max"]
    lake = tmp_path / "lake"
    lake.mkdir()
    (lake / "robustness.json").write_text(json.dumps(rep))
    out = kpi_report(eh, feats, str(tmp_path / "none"), str(tmp_path / "state"), str(lake))
    row = next(r for r in out["rows"] if r["kpi"] == "Storage cost vs best rule")
    assert "median of 2" in row["value"] and "range" in row["note"]
    assert row["status"] == ("PASS" if c["median"] >= 15 else "MISS")
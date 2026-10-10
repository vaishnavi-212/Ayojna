"""Step 10A: the intelligence layer (hotness zoo, forecast, anomaly) and the data window."""

import json

import numpy as np
import pandas as pd
import pytest

from ayojna.ingest.build import build
from ayojna.ingest.real import common_window
from ayojna.ingest.synth import generate
from ayojna.io import write_table
from ayojna.models import anomaly, forecast, train_hotness
from ayojna.models.features import build_features, time_split
from ayojna.models.hotness import ALGOS, _treeshap, make_estimator, train
from ayojna.planner.optimizer import expected_ios


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("intel")
    generate(d / "raw", days=6, seed=5)
    eh = build(d / "raw", d / "eh.csv")
    feats = build_features(eh)
    write_table(feats, d / "f.csv")
    return d, eh, feats


# ---------- data honesty ----------
def test_common_window_cuts_hours_after_a_trace_ends():
    t = pd.DataFrame({"volume": ["a", "a", "b", "b"], "hour": [0, 30, 0, 10]})
    assert common_window(t)["hour"].max() == 10  # b stopped at 10: a's later hours go


def test_load_estimate_never_below_the_extents_own_rate():
    rate = {"hot": 500.0, "warm": 5.0}
    lam = expected_ios([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], rate, [24 * 6000, 0, 0])
    assert lam[0] == 6000  # a very busy hot extent keeps its real rate
    assert lam[1] == 500 and lam[2] == 5  # otherwise the class forecast


# ---------- hotness model zoo ----------
def test_every_sklearn_algorithm_trains_and_predicts(synth):
    _, _, feats = synth
    tr, te = time_split(feats, 24, 24)
    for algo in ("random_forest", "hist_gb"):
        m = train(tr, algo=algo)
        p = m.predict(te.head(100))
        assert np.allclose(p[["p_hot", "p_warm", "p_cold"]].sum(axis=1), 1.0)
        assert m.algo == algo and m.explainer == "occlusion"
        assert (p["reasons"].str.len() > 0).mean() > 0.9


def test_missing_library_is_an_import_error_not_a_crash():
    for algo in ("lightgbm", "xgboost"):
        try:
            make_estimator(algo)
        except ImportError:
            pass  # not installed here: the leaderboard skips it
    with pytest.raises(ValueError):
        make_estimator("magic")


def test_treeshap_picks_the_predicted_class_block():
    class Booster:  # LightGBM layout: (n, classes*(features+1)), class-major, bias last
        def predict(self, x, pred_contrib=False):
            n, f = x.shape
            out = np.zeros((n, 3, f + 1))
            out[:, :, :f] = np.arange(3)[None, :, None] * 10 + np.arange(f)[None, None, :]
            return out.reshape(n, -1)

    class Fake:
        classes_ = np.array(["cold", "hot", "warm"])  # library order (sorted)
        booster_ = Booster()

    eff = _treeshap(Fake(), "lightgbm", np.zeros((3, 4)), np.array([0, 1, 2]))  # hot, warm, cold
    assert eff.shape == (3, 4)
    assert list(eff[:, 0]) == [10, 20, 0]  # hot -> column 1, warm -> 2, cold -> 0


def test_broken_treeshap_falls_back_to_occlusion(synth):
    _, _, feats = synth
    tr, te = time_split(feats, 24, 24)
    m = train(tr, algo="hist_gb")
    m.algo = "lightgbm"  # pretend: TreeSHAP will fail (no booster_), occlusion must answer
    p = m.predict(te.head(50))
    assert (p["drivers"].map(json.loads).map(len) > 0).mean() > 0.5


def test_leaderboard_chooses_on_validation_and_reports(synth, tmp_path):
    d, _, _ = synth
    train_hotness.main(str(d / "f.csv"), str(tmp_path), ["lightgbm", "random_forest", "hist_gb"])
    rep = json.loads((tmp_path / train_hotness.LEADERBOARD).read_text())
    ok = [r for r in rep["board"] if r["status"] == "ok" and r["model"] != "rule baseline"]
    assert rep["champion"] == max(ok, key=lambda r: r["val_f1"])["model"]
    assert rep["windows"]["validation"][1] < rep["windows"]["test"][0]  # test never chooses
    assert {r["model"] for r in rep["board"]} >= {"rule baseline", "random_forest", "hist_gb"}
    assert all(r["status"] in ("ok", "not installed") for r in rep["board"])
    assert set(ALGOS) >= {r["model"] for r in rep["board"][1:]}


# ---------- forecast ----------
def test_seasonal_forecast_repeats_a_clean_daily_pattern():
    day = np.arange(24, dtype=float)
    y = np.tile(day, 5)
    assert np.allclose(forecast.seasonal_ewma(y), day)
    assert np.allclose(forecast.naive(y), day)


def test_forecast_falls_back_when_the_model_breaks(monkeypatch):
    def broken(y, h=24):
        raise RuntimeError("cmdstan missing")

    monkeypatch.setitem(forecast.MODELS, "prophet", broken)
    yhat, used = forecast.predict(np.ones(72), "prophet")
    assert used == "seasonal_ewma" and len(yhat) == 24


def test_forecast_volumes_and_spike_flag(synth):
    _, eh, _ = synth
    rows = forecast.forecast_volumes(eh, "seasonal_ewma")
    assert set(rows["volume"]) == set(eh["volume"])
    assert {"io_last24", "io_next24", "ws_gb_next24", "spike", "model"} <= set(rows.columns)
    big = forecast.forecast_volumes(eh, "seasonal_ewma", spike_ratio=0.0, spike_min_io=0)
    assert big["spike"].all()  # thresholds at zero: everything counts as a spike


def test_backtest_reports_mape_per_model(synth):
    _, eh, _ = synth
    b = forecast.backtest(eh, ["naive", "seasonal_ewma"], origins=2)
    for name in ("naive", "seasonal_ewma"):
        assert b[name]["origins"] == 2 and b[name]["mape_capacity_pct"] >= 0


# ---------- anomaly ----------
def _with_burst(eh: pd.DataFrame, volume: str) -> pd.DataFrame:
    eh = eh.copy()
    last = eh["hour"] == eh["hour"].max()
    hit = last & (eh["volume"] == volume)
    eh.loc[hit, ["reads", "writes"]] *= 200  # a sudden scan
    eh.loc[hit, ["read_bytes", "write_bytes"]] *= 200
    return eh


def test_isolation_forest_flags_a_sudden_burst(synth, tmp_path):
    _, eh, _ = synth
    vol = "web_0"
    burst = _with_burst(eh, vol)
    feats = anomaly.volume_features(burst)
    model = anomaly.train(feats, int(feats["hour"].max()) - 1)
    model.save(tmp_path)
    rows, note, degraded = anomaly.detect(burst, tmp_path)
    assert not degraded and "isolation forest" in note
    hit = rows.set_index("volume").loc[vol]
    assert hit["anomaly"] and "SD from this volume's normal" in hit["reason"]
    assert rows["anomaly"].sum() <= 2  # the others stay calm


def test_anomaly_rule_fallback_without_a_model(synth, tmp_path):
    _, eh, _ = synth
    rows, note, degraded = anomaly.detect(_with_burst(eh, "usr_0"), tmp_path / "none")
    assert degraded and "rule fallback" in note
    assert rows.set_index("volume").loc["usr_0", "anomaly"]
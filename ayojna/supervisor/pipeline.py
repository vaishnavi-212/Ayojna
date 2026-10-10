"""The hourly decision cycle as supervised steps:

    load -> features -> [hotness | forecast | anomaly] -> plan -> execute (saga)
                         the three models run in parallel

Each step has a fallback so the cycle degrades instead of breaking:
features -> last good file, hotness -> rule model, forecast -> seasonal model (then no
spike guard), anomaly -> robust-z rule (then no pause), plan -> hold (no moves),
execute -> keep the plan as a recommendation only.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ayojna.contracts import Envelope, MovePlan, Source, Status
from ayojna.executor.catalog import Catalog
from ayojna.executor.saga import Executor
from ayojna.executor.tierstore import make_store
from ayojna.io import atomic_write_text, read_table
from ayojna.models.anomaly import detect
from ayojna.models.features import build_features, load_label_config
from ayojna.models.forecast import forecast_volumes
from ayojna.models.predictor import predict_hotness
from ayojna.models.series import volume_hourly
from ayojna.planner import bandit
from ayojna.planner.guards import guards_from
from ayojna.planner.live import live_plan
from ayojna.planner.optimizer import load_planner_config
from ayojna.recommend.approvals import decision_for, load_approvals
from ayojna.settings import CONFIG_DIR
from ayojna.supervisor.runner import Degraded, Parallel, Step
from ayojna.supervisor.state import StateStore
from ayojna.twin.sim import Twin


def _inject(fn, fault: str | None):
    """Fault injection for demos and tests: 'fail' raises, 'slow' sleeps past the timeout."""
    if fault is None:
        return fn

    def broken(ctx):
        if fault == "fail":
            raise RuntimeError("injected fault")
        time.sleep(60)
        return fn(ctx)

    return broken


def _write_json(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, default=str))


def policy_version() -> str:
    return hashlib.sha256((Path(CONFIG_DIR) / "policy.yaml").read_bytes()).hexdigest()[:8]


def build_steps(
    eh_path: str,
    features_path: str,
    model_store: str,
    faults: dict | None = None,
    timeout_s: float = 30.0,
    state_dir: str = "data/state",
    tiers_root: str = "data/tiers",
    store_kind: str = "fs",
    recommend_only: bool = False,
    corrupt_first_move: bool = False,
    crash_after_moves: int | None = None,
    approval_mode: bool = False,
    store: StateStore | None = None,
) -> list:
    faults = faults or {}
    mcfg = load_label_config()
    labels, fc, an = mcfg["labels"], mcfg["forecast"], mcfg["anomaly"]
    state = store or StateStore(state_dir)  # the run's store (file or Redis): same fencing
    catalog_path = state.root / "catalog.json"

    def load(ctx):
        return read_table(eh_path)

    def features(ctx):
        return build_features(ctx["load"], labels["hot_min_accesses"], labels["horizon_hours"])

    def features_fallback(ctx):  # last good features table on disk
        return read_table(features_path)

    def hotness(ctx):
        f = ctx["features"]
        latest = f[f["hour"] == f["hour"].max()]
        pred, source, status, note = predict_hotness(
            latest, model_store, labels["hot_min_accesses"]
        )
        pred.attrs["model_version"] = note
        return Degraded(pred, note) if source == Source.FALLBACK else pred

    def forecast(ctx):
        rows = forecast_volumes(ctx["load"], fc["model"], fc["horizon_hours"],
                                fc["spike_ratio"], fc["spike_min_io"])  # fmt: skip
        if fc["model"] != "seasonal_ewma" and not rows["model"].eq(fc["model"]).all():
            return Degraded(rows, f"{fc['model']} unavailable: seasonal_ewma used")
        return rows

    def anomaly(ctx):
        rows, note, degraded = detect(ctx["load"], model_store, an["z_max"])
        return Degraded(rows, note) if degraded else rows

    def no_guard(ctx):  # model down: plan without this guard (the level shows L1)
        return Degraded(None, "model unavailable: guard off this cycle")

    def intel_report(ctx, guards: dict) -> None:
        f, a = ctx.get("forecast"), ctx.get("anomaly")
        _write_json(
            state.root / "last_intel.json",
            {
                "run_id": ctx["run_id"],
                "hour": int(ctx["hotness"]["hour"].max()),
                "hotness_model": ctx["hotness"].attrs.get("model_version", "unknown"),
                "forecast": [] if f is None else f.to_dict(orient="records"),
                "anomaly": [] if a is None else a.to_dict(orient="records"),
                "guards": guards,
                "steps": {k: dict(v) for k, v in ctx.get("_steps", {}).items()},
                "parallel": dict(ctx.get("_groups", {})),
            },
        )

    def envelope(ctx, status=Status.OK, source=Source.PRIMARY):
        return Envelope(
            run_id=ctx["run_id"],
            data_version=Path(eh_path).name,
            model_version=ctx["hotness"].attrs.get("model_version", "unknown"),
            policy_version=policy_version(),
            fencing_token=ctx["token"],
            status=status,
            source=source,
        )

    def plan(ctx):
        preds = ctx["hotness"]
        hour = int(preds["hour"].max())
        twin = Twin.from_extent_hourly(ctx["load"])
        catalog = Catalog(catalog_path)
        catalog.seed(twin.volumes, twin.extent_ids, make_store(store_kind, tiers_root), hour)
        current, since = catalog.placement(twin.volumes, twin.extent_ids)
        guards = guards_from(ctx.get("forecast"), ctx.get("anomaly"))
        intel_report(ctx, guards)
        rl, knobs = rl_policy(ctx["load"], twin), None
        if rl.get("promoted"):  # only a bandit that beat the static settings may steer
            knobs = bandit.knobs_for(rl["suggested"], twin.volumes, load_planner_config()["bandit"]["arms"])
        solver: dict = {}
        mp, why, cards = live_plan(
            twin, preds, ctx["features"], current, since, envelope(ctx), guards, knobs, solver
        )
        _write_json(
            state.root / "last_plan.json",
            {"plan": mp.model_dump(mode="json"), "why": why, "cards": cards,
             "decision": {"solver": solver, "bandit": rl}},
        )  # fmt: skip
        return mp

    def rl_policy(eh, twin) -> dict:
        try:
            vh = volume_hourly(eh)
            frozen = {v for v in set(twin.volumes) if twin.legal_hold[twin.volumes == v].any()}
            io = {v: g["io"].to_numpy(dtype=float) for v, g in vh.groupby("volume") if v not in frozen}
            sla = {v: float(twin.sla_target_ms[twin.volumes == v].min()) for v in io}
            return bandit.live_policy(model_store, io, sla)
        except Exception as exc:  # a broken bandit file never blocks planning
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def plan_fallback(ctx):  # hold: no moves is always safe
        env = envelope(ctx, Status.DEGRADED, Source.FALLBACK)
        mp = MovePlan(envelope=env, strategy="hold", hour=int(ctx["hotness"]["hour"].max()))
        _write_json(state.root / "last_plan.json", {"plan": mp.model_dump(mode="json"), "why": {}})
        return mp

    def execute(ctx):
        mp: MovePlan = ctx["plan"]
        # a replica resuming this run re-stamps the plan with ITS token (fencing)
        env = mp.envelope.model_copy(update={"fencing_token": ctx["token"]})
        mp = mp.model_copy(update={"envelope": env})
        safe = state.safe_mode()  # operator kill switch: L4, read-only
        if recommend_only or safe:
            mode = "safe-mode" if safe else "recommend-only"
            report = {"run_id": ctx["run_id"], "mode": mode, "results": [], "safe_mode": safe}
        else:
            ex = Executor(
                make_store(store_kind, tiers_root),
                Catalog(catalog_path),
                state.root / "ledger.jsonl",
                load_planner_config()["max_gb_moved_per_hour"],
                corrupt_first=corrupt_first_move,
                crash_after_moves=crash_after_moves,
            )
            decide = None
            if approval_mode:  # only groups a person approved are executed
                approvals = load_approvals(state.root)
                decide = lambda m: decision_for(approvals, m.group_key())  # noqa: E731
            report = ex.execute(mp, state.is_current, decide)
            report["mode"] = "approval" if approval_mode else "executed"
            for r in report["results"]:
                if r["status"] == "rolled_back":  # a move that failed verification
                    state.dead_letter({"kind": "move", "run_id": ctx["run_id"], **r})
        _write_json(state.root / "last_exec.json", report)
        return report

    def execute_fallback(ctx):  # storage unreachable: keep the plan as a recommendation
        report = {"run_id": ctx["run_id"], "mode": "recommend-only", "results": []}
        _write_json(state.root / "last_exec.json", report)
        return Degraded(report, "executor unavailable: plan kept as recommendation")

    def wrap(name, fn):
        return _inject(fn, faults.get(name))

    return [
        Step("load", wrap("load", load), timeout_s),
        Step("features", wrap("features", features), timeout_s, fallback=features_fallback),
        Parallel(
            "models",
            [
                Step("hotness", wrap("hotness", hotness), timeout_s),
                Step("forecast", wrap("forecast", forecast), timeout_s, fallback=no_guard,
                     critical=False),
                Step("anomaly", wrap("anomaly", anomaly), timeout_s, fallback=no_guard,
                     critical=False),
            ],
        ),  # fmt: skip
        Step("plan", wrap("plan", plan), timeout_s, fallback=plan_fallback),
        Step("execute", wrap("execute", execute), timeout_s, fallback=execute_fallback),
    ]
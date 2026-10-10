"""Runs a cycle of steps with timeout, retry, fallback, checkpoint and a degradation level.

A `Parallel` group runs its steps at the same time (the three models: hotness, forecast,
anomaly). Each member keeps its own timeout, retry, fallback and checkpoint.

Levels: L0 normal · L1 some fallbacks used · L3 hold (a critical step failed: no moves).
(L2 recommend-only and L4 safe mode are set by the pipeline and the CLI.)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from ayojna.supervisor.state import StateStore


@dataclass
class Degraded:
    """Return this from a step that answered with its own internal fallback."""

    output: Any
    note: str


@dataclass
class Step:
    name: str
    fn: Callable[[dict], Any]  # receives the outputs of earlier steps
    timeout_s: float = 30.0
    retries: int = 1
    fallback: Callable[[dict], Any] | None = None
    critical: bool = True  # if it fails with no fallback, the cycle holds


@dataclass
class Parallel:
    """Steps that only need earlier outputs, not each other: run them concurrently."""

    name: str
    steps: list[Step]


def _call(fn, ctx, timeout_s):
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, ctx).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _run_step(step: Step, ctx: dict) -> tuple[Any, dict]:
    """Primary with retries, then fallback. Never raises. Returns (output, info)."""
    t0 = time.perf_counter()
    source, note, out = "primary", "", None
    for attempt in range(step.retries + 1):
        try:
            out = _call(step.fn, ctx, step.timeout_s)
            break
        except Exception as exc:  # includes TimeoutError
            note = f"{type(exc).__name__}: {exc}"
            if attempt < step.retries:
                time.sleep(0.2 * 2**attempt)
    else:
        if step.fallback is not None:
            try:
                out, source = _call(step.fallback, ctx, step.timeout_s), "fallback"
            except Exception as exc:
                source, note = "failed", f"fallback failed: {exc}"
        else:
            source = "failed"
    if isinstance(out, Degraded):
        out, source, note = out.output, "fallback", out.note
    return out, {"source": source, "note": note, "ms": round(1000 * (time.perf_counter() - t0))}


def run_cycle(run_id: str, steps: list, store: StateStore, token: int) -> dict:
    ctx: dict = {"run_id": run_id, "token": token}
    report = {"run_id": run_id, "level": "L0", "steps": {}, "groups": {}}
    ctx["_steps"] = report["steps"]  # later steps may read earlier timings (dashboard)
    ctx["_groups"] = report["groups"]
    store.set_active_run(run_id)
    for item in steps:
        if not store.is_current(token):
            raise PermissionError("lost the leader lease: another supervisor took over")
        group = item.steps if isinstance(item, Parallel) else [item]
        todo = []
        for step in group:
            saved, info = store.load_step(run_id, step.name)
            if info is not None:  # finished before a crash: resume, do not redo
                ctx[step.name] = saved
                report["steps"][step.name] = {**info, "resumed": True}
                if info["source"] == "fallback" and report["level"] == "L0":
                    report["level"] = "L1"
            else:
                todo.append(step)
        t0 = time.perf_counter()
        if len(todo) > 1:
            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                results = list(pool.map(lambda s: _run_step(s, ctx), todo))
        else:
            results = [_run_step(s, ctx) for s in todo]
        if isinstance(item, Parallel) and todo:
            report["groups"][item.name] = {
                "steps": [s.name for s in todo],
                "wall_ms": round(1000 * (time.perf_counter() - t0)),
                "sum_ms": sum(info["ms"] for _, info in results),
            }
        hold = False
        for step, (out, info) in zip(todo, results):
            store.audit({"run_id": run_id, "token": token, "step": step.name, **info})
            report["steps"][step.name] = info
            if info["source"] == "failed":
                hold = hold or step.critical
                continue
            if info["source"] == "fallback" and report["level"] == "L0":
                report["level"] = "L1"
            ctx[step.name] = out
            store.save_step(run_id, step.name, out, info)
        if hold:
            report["level"] = "L3"
            break
    store.set_active_run(None)
    ctx.pop("_steps", None)
    ctx.pop("_groups", None)
    report["outputs"] = ctx
    return report
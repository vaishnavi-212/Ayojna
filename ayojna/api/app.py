"""Ayojna dashboard API (FastAPI). Thin routes over service.py.

Run:  uvicorn ayojna.api.app:app --port 8000      then open http://localhost:8000
API docs (Swagger) at http://localhost:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ayojna.api.service import Service
from ayojna.copilot.copilot import ask
from ayojna.copilot.llm import config_from_env

from ayojna.settings import DATA_DIR, REPO_ROOT

WEB = REPO_ROOT / "web" / "index.html"


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)



class SafeMode(BaseModel):
    on: bool
    reason: str = Field(default="", max_length=200)


class Decision(BaseModel):
    group: str = Field(min_length=3, max_length=80)
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=200)

def create_app(state_dir: str | Path | None = None, lake_dir: str | Path | None = None) -> FastAPI:
    svc = Service(
        state_dir or os.getenv("AYOJNA_STATE", DATA_DIR / "state"),
        lake_dir or os.getenv("AYOJNA_LAKE", DATA_DIR / "lake"),
    )
    api = FastAPI(title="Ayojna", description="AI that plans where every byte should live")

    @api.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(WEB)

    @api.get("/api/kpis")
    def kpis():
        return svc.kpis()

    @api.get("/api/status")
    def status():
        return svc.status()

    @api.get("/api/scoreboard")
    def scoreboard():
        return svc.scoreboard()

    @api.get("/api/central")
    def central():
        return svc.central()

    @api.post("/api/safe-mode")
    def safe_mode(s: SafeMode):
        return svc.set_safe_mode(s.on, s.reason)

    @api.get("/api/decision")
    def decision():
        return svc.decision()

    @api.get("/api/intel")
    def intel():
        return svc.intel()

    @api.get("/api/placement")
    def placement():
        return svc.placement()

    @api.get("/api/plan")
    def plan(limit: int = 50):
        return svc.plan(limit)

    @api.get("/api/execution")
    def execution():
        return svc.execution()

    @api.get("/api/audit")
    def audit(limit: int = 50):
        return svc.audit(limit)

    @api.get("/api/explain/{volume}/{extent_id}")
    def explain(volume: str, extent_id: int):
        return svc.explain(volume, extent_id)

    @api.get("/api/recommendations")
    def recommendations():
        return svc.recommendations()

    @api.get("/api/recommendations/detail")
    def recommendation(group: str):
        return svc.recommendation(group)

    @api.post("/api/recommendations/decide")
    def decide(d: Decision):
        return svc.decide(d.group, d.decision, d.note)
    
    @api.post("/api/ask")
    def ask_copilot(q: Question):
        return ask(q.question, svc)

    @api.get("/api/copilot")
    def copilot_info():
        cfg = config_from_env()
        return {"provider": cfg.provider, "model": cfg.model if cfg.provider != "none" else None}
    
    return api


app = create_app()
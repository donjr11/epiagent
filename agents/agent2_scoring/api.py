"""FastAPI app for the Scoring Service (design doc §9.2).

Endpoints:
  POST /score           -> rank models for a given capability/hardware query
  GET  /capabilities    -> list supported capabilities
  GET  /healthz         -> trivial liveness probe
  POST /ingest/trigger  -> manual ingestion kickoff (for operators)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.agent1_discovery import run_ingestion
from agents.agent2_scoring import scorer
from core.database import get_conn, init_db
from core.models.dto import CAPABILITIES, ScoreRequest, ScoreResponse

log = logging.getLogger(__name__)

app = FastAPI(title="EpiAgent Scoring Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # internal operators only in v1; lock down post-auth
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/capabilities")
def list_capabilities() -> dict:
    return {"capabilities": CAPABILITIES}


@app.post("/score", response_model=ScoreResponse)
def post_score(req: ScoreRequest) -> ScoreResponse:
    unknown = [c for c in req.capabilities if c not in CAPABILITIES]
    if unknown:
        raise HTTPException(400, f"Unknown capabilities: {unknown}")
    with get_conn() as conn:
        return scorer.score(conn, req)


@app.post("/ingest/trigger")
def trigger_ingestion() -> dict:
    summary = run_ingestion()
    return summary

"""FastAPI app for the Scoring Service (design doc §9.2).

Endpoints:
  POST /score           -> rank models for a given capability/hardware query
  GET  /capabilities    -> list supported capabilities
  GET  /healthz         -> trivial liveness probe
  POST /ingest/trigger  -> manual ingestion kickoff (for operators)
  GET  /digest/daily    -> render today's digest as HTML (design §9.3)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agents.agent1_discovery import run_ingestion
from agents.agent2_scoring import scorer
from agents.agent2_scoring.profiles import ProfileMissingError
from agents.agent3_notifications import digest as digest_mod
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
    try:
        with get_conn() as conn:
            return scorer.score(conn, req)
    except ProfileMissingError as exc:
        # No silent uniform-weight fallback (design §6.4): tell the operator.
        log.error("scoring profile missing: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Scoring profile not configured for capability "
                f"'{exc.capability}'. Run scripts.seed_strawman_profiles or the "
                f"BWM CLI before requesting scores."
            ),
        ) from exc


@app.post("/ingest/trigger")
def trigger_ingestion() -> dict:
    summary = run_ingestion()
    return summary


@app.get("/digest/daily", response_class=HTMLResponse)
def get_digest_daily() -> HTMLResponse:
    """Render the daily digest as HTML (design §9.3).

    Side effects: persists the new top-5 snapshot + last_digest_at to
    notification_state. Does NOT send email — that path is owned by the
    APScheduler job, which calls this endpoint and then forwards the HTML
    to the SMTP sender.
    """
    composed = digest_mod.compose_digest()
    return HTMLResponse(content=composed["html"])

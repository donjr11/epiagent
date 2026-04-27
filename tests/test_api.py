"""HTTP-level tests for agents.agent2_scoring.api."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_digest_daily_returns_200_with_html(tmp_db):
    """GET /digest/daily must return 200 with a non-empty HTML body
    rendered from the existing digest composer (design §9.3)."""
    # Import after tmp_db fixture has rewired the DB path & reloaded core.config.
    from agents.agent2_scoring.api import app

    with TestClient(app) as client:
        resp = client.get("/digest/daily")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    body = resp.text
    assert body, "digest HTML body must be non-empty"
    # Sanity: the digest template's title text should be present.
    assert "EpiAgent daily digest" in body

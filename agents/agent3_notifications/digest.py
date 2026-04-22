"""Daily digest composer.

Flow:
  1. Find models whose first_seen_at > last_digest_at.
  2. Compare current per-capability top-5 against the previous snapshot
     stored in notification_state.last_snapshot; flag |Δrank| >= threshold.
  3. Check latest ingestion_runs for a failed/partial status since last digest.
  4. Render template, send via SMTP, update notification_state.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agents.agent2_scoring import scorer
from agents.agent3_notifications import sender
from core import config
from core.database import get_conn, init_db
from core.models.dto import ScoreRequest, CAPABILITIES

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(("html", "htm")),
    )


def _new_models(conn: sqlite3.Connection, since: str | None) -> list[dict]:
    if since is None:
        return []
    rows = conn.execute(
        """
        SELECT huggingface_id, name, parameter_count_b, license_spdx, is_on_whitelist
        FROM models WHERE first_seen_at > ?
        ORDER BY first_seen_at DESC LIMIT 50
        """,
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def _top5_snapshot(conn: sqlite3.Connection) -> dict[str, list[tuple[int, str]]]:
    """For each capability, rank the top 5 models using default query (no VRAM limit)."""
    snap: dict[str, list[tuple[int, str]]] = {}
    for cap in CAPABILITIES:
        try:
            resp = scorer.score(
                conn,
                ScoreRequest(capabilities=[cap], top_n=5),
            )
        except Exception as exc:
            log.warning("snapshot score for %s failed: %s", cap, exc)
            continue
        snap[cap] = [(0, row.huggingface_id) for row in resp.ranked]
    return snap


def _rank_changes(
    prev: dict[str, list[tuple[int, str]]] | None,
    curr: dict[str, list[tuple[int, str]]],
    threshold: int,
) -> list[dict]:
    out: list[dict] = []
    if not prev:
        return out
    for cap, curr_list in curr.items():
        prev_list = prev.get(cap) or []
        prev_ids = [hid for _, hid in prev_list]
        curr_ids = [hid for _, hid in curr_list]
        for i, hid in enumerate(curr_ids):
            prev_rank = prev_ids.index(hid) + 1 if hid in prev_ids else 99
            now_rank = i + 1
            if abs(prev_rank - now_rank) >= threshold:
                out.append({
                    "capability": cap,
                    "name": hid,
                    "prev": prev_rank if prev_rank < 99 else "—",
                    "now": now_rank,
                })
    return out


def _recent_ingestion_failure(conn: sqlite3.Connection, since: str | None) -> str | None:
    if since is None:
        row = conn.execute(
            "SELECT status, warnings FROM ingestion_runs "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT status, warnings FROM ingestion_runs "
            "WHERE started_at > ? AND status != 'ok' "
            "ORDER BY started_at DESC LIMIT 1",
            (since,),
        ).fetchone()
    if row and row["status"] != "ok":
        return f"status={row['status']}; warnings={row['warnings']}"
    return None


def build_and_send() -> dict:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        state = conn.execute(
            "SELECT last_digest_at, last_snapshot FROM notification_state WHERE id=1"
        ).fetchone()
        last_digest_at = state["last_digest_at"] if state else None
        last_snapshot = json.loads(state["last_snapshot"]) if state and state["last_snapshot"] else None

        new_models = _new_models(conn, last_digest_at)
        curr_snapshot = _top5_snapshot(conn)
        rank_changes = _rank_changes(last_snapshot, curr_snapshot, config.RANK_CHANGE_THRESHOLD)
        ingestion_failure = _recent_ingestion_failure(conn, last_digest_at)

        # Persist new snapshot + timestamp.
        snap_json = json.dumps(curr_snapshot)
        if state is None:
            conn.execute(
                "INSERT INTO notification_state (id, last_digest_at, last_snapshot) VALUES (1, ?, ?)",
                (now, snap_json),
            )
        else:
            conn.execute(
                "UPDATE notification_state SET last_digest_at=?, last_snapshot=? WHERE id=1",
                (now, snap_json),
            )

    html = _env().get_template("digest.html.j2").render(
        date=now[:10],
        new_models=new_models,
        rank_changes=rank_changes,
        ingestion_failure=ingestion_failure,
        threshold=config.RANK_CHANGE_THRESHOLD,
    )
    err = sender.send_html(
        subject=f"EpiAgent digest {now[:10]}",
        html=html,
        to=config.ADMIN_EMAILS,
    )
    return {
        "sent": err is None,
        "error": err,
        "new_model_count": len(new_models),
        "rank_change_count": len(rank_changes),
        "ingestion_failure": ingestion_failure,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build_and_send(), indent=2))

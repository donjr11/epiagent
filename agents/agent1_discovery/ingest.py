"""Ingestion orchestration.

Flow (Section 9.1):
  1. List HF candidates and filter via policy.is_candidate.
  2. Upsert models + capabilities; log metadata changes.
  3. Call VRAM estimator per model, upsert hardware rows.
  4. Pull AA benchmarks, append new rows.
  5. Fall back to HF Leaderboard v2 for missing metrics.
  6. Write an ingestion_runs log entry.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Iterable

from agents.agent1_discovery import policy, vram_estimator
from agents.agent1_discovery.sources import (
    artificial_analysis,
    hf_leaderboard,
    huggingface,
)
from core import config, licenses
from core.database import get_conn, init_db

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_model(conn: sqlite3.Connection, rec: dict) -> tuple[int, bool]:
    """Insert or update a model. Returns (model_id, created)."""
    license_ = licenses.normalize(rec.get("license"))
    on_whitelist = 1 if licenses.is_whitelisted(license_) else 0
    now = _now()

    row = conn.execute(
        "SELECT id, name, license_spdx, context_window, parameter_count_b, is_on_whitelist "
        "FROM models WHERE huggingface_id = ?",
        (rec["huggingface_id"],),
    ).fetchone()

    if row is None:
        cur = conn.execute(
            """
            INSERT INTO models (
              huggingface_id, name, provider, parameter_count_b, license_spdx,
              is_on_whitelist, context_window, release_date, first_seen_at, last_seen_at,
              is_active
            ) VALUES (?,?,?,?,?,?,?,?,?,?,1)
            """,
            (
                rec["huggingface_id"],
                rec["name"],
                rec.get("provider"),
                rec.get("parameter_count_b"),
                license_,
                on_whitelist,
                rec.get("context_window"),
                (rec.get("release_date") or "")[:10] or None,
                now,
                now,
            ),
        )
        return int(cur.lastrowid), True

    model_id = int(row["id"])
    # Log meaningful field changes; keep history in model_change_log.
    tracked = {
        "name": rec["name"],
        "license_spdx": license_,
        "context_window": rec.get("context_window"),
        "parameter_count_b": rec.get("parameter_count_b"),
        "is_on_whitelist": on_whitelist,
    }
    for field, new_val in tracked.items():
        old_val = row[field]
        if new_val is not None and old_val != new_val:
            conn.execute(
                "INSERT INTO model_change_log (model_id, field, old_value, new_value, changed_at) "
                "VALUES (?,?,?,?,?)",
                (model_id, field, str(old_val), str(new_val), now),
            )

    conn.execute(
        """
        UPDATE models SET
          name = COALESCE(?, name),
          license_spdx = COALESCE(?, license_spdx),
          is_on_whitelist = ?,
          parameter_count_b = COALESCE(?, parameter_count_b),
          context_window = COALESCE(?, context_window),
          last_seen_at = ?,
          is_active = 1
        WHERE id = ?
        """,
        (
            rec["name"],
            license_,
            on_whitelist,
            rec.get("parameter_count_b"),
            rec.get("context_window"),
            now,
            model_id,
        ),
    )
    return model_id, False


def _upsert_capabilities(conn: sqlite3.Connection, model_id: int, caps: Iterable[str]) -> None:
    for cap in caps:
        conn.execute(
            "INSERT OR IGNORE INTO model_capabilities (model_id, capability) VALUES (?, ?)",
            (model_id, cap),
        )


def _upsert_hardware(conn: sqlite3.Connection, model_id: int, rec: dict) -> None:
    params_b = rec.get("parameter_count_b")
    ctx = rec.get("context_window") or 8192
    if not params_b:
        return
    now = _now()
    for est in vram_estimator.estimate_all_quants(params_b, ctx):
        conn.execute(
            """
            INSERT INTO model_hardware_requirements (
                model_id, quantization, min_vram_gb, recommended_vram_gb,
                min_ram_gb, estimator_source, estimated_at
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(model_id, quantization) DO UPDATE SET
                min_vram_gb = excluded.min_vram_gb,
                recommended_vram_gb = excluded.recommended_vram_gb,
                min_ram_gb = excluded.min_ram_gb,
                estimator_source = excluded.estimator_source,
                estimated_at = excluded.estimated_at
            """,
            (
                model_id,
                est.quantization,
                est.min_vram_gb,
                est.recommended_vram_gb,
                est.min_ram_gb,
                est.estimator_source,
                now,
            ),
        )


def _append_benchmark(conn: sqlite3.Connection, row: dict) -> None:
    mid = conn.execute(
        "SELECT id FROM models WHERE huggingface_id = ?", (row["huggingface_id"],)
    ).fetchone()
    if not mid:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO model_benchmarks
            (model_id, metric_name, source, n_shot, cot, score, measured_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            int(mid["id"]),
            row["metric_name"],
            row["source"],
            int(row.get("n_shot", -1)),
            int(row.get("cot", 0)),
            float(row["score"]),
            row["measured_at"],
        ),
    )


def run(
    hf_iter: Callable[[], Iterable[dict]] | None = None,
    aa_iter: Callable[[], Iterable[dict]] | None = None,
    leaderboard_iter: Callable[[], Iterable[dict]] | None = None,
) -> dict:
    """Execute one ingestion cycle.

    All sources are injectable so tests can pass deterministic generators.
    """
    init_db()
    started = _now()
    warnings: list[str] = []
    candidates_seen = 0
    models_upserted = 0

    hf_iter = hf_iter or (lambda: huggingface.list_candidate_models())
    aa_iter = aa_iter or (lambda: artificial_analysis.fetch_all())
    leaderboard_iter = leaderboard_iter or (lambda: hf_leaderboard.fetch_all())

    with get_conn() as conn:
        for rec in hf_iter():
            candidates_seen += 1
            ok, reason = policy.is_candidate(rec)
            if not ok:
                warnings.append(f"rejected {rec.get('huggingface_id')}: {reason}")
                continue
            try:
                model_id, _created = _upsert_model(conn, rec)
                _upsert_capabilities(conn, model_id, rec.get("capabilities") or ["text"])
                _upsert_hardware(conn, model_id, rec)
                models_upserted += 1
            except Exception as exc:
                warnings.append(f"upsert failure {rec.get('huggingface_id')}: {exc!r}")

        # Benchmarks: AA primary.
        try:
            for row in aa_iter():
                _append_benchmark(conn, row)
        except Exception as exc:
            warnings.append(f"AA source failed: {exc!r}")

        # Benchmarks: HF leaderboard fallback.
        try:
            for row in leaderboard_iter():
                _append_benchmark(conn, row)
        except Exception as exc:
            warnings.append(f"HF leaderboard source failed: {exc!r}")

        conn.execute(
            """
            INSERT INTO ingestion_runs (
                started_at, finished_at, status, candidates_seen, models_upserted, warnings
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                started,
                _now(),
                "ok" if not warnings else "partial",
                candidates_seen,
                models_upserted,
                json.dumps(warnings[-200:]),  # cap log size
            ),
        )

    return {
        "candidates_seen": candidates_seen,
        "models_upserted": models_upserted,
        "warnings": warnings,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = run()
    print(json.dumps(summary, indent=2))

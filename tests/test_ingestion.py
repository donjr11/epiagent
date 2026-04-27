"""Integration test for the ingestion service against mocked HF/AA iterators."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from agents.agent1_discovery import ingest
from agents.agent1_discovery.sources import artificial_analysis as aa
from core.database import get_conn


def _hf_fixture():
    """Two candidates: one that passes, one rejected by policy."""
    now = datetime.now(timezone.utc).isoformat()
    yield {
        "huggingface_id": "acme/llama-7b",
        "name": "llama-7b",
        "provider": "acme",
        "parameter_count_b": 7.0,
        "license": "apache-2.0",
        "context_window": 8192,
        "pipeline_tag": "text-generation",
        "tags": ["text-generation"],
        "capabilities": ["text"],
        "downloads": 50000,
        "last_modified": now,
    }
    # Rejected: downloads below threshold
    yield {
        "huggingface_id": "nobody/tiny",
        "name": "tiny",
        "parameter_count_b": 0.2,
        "license": "mit",
        "context_window": 2048,
        "pipeline_tag": "text-generation",
        "tags": ["text-generation"],
        "capabilities": ["text"],
        "downloads": 5,
        "last_modified": now,
    }


def _aa_fixture():
    yield {
        "huggingface_id": "acme/llama-7b",
        "metric_name": "MMLU-Pro",
        "source": "artificial_analysis",
        "n_shot": 5,
        "cot": 1,
        "score": 62.3,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def _lb_fixture():
    # No additional rows; exercise the empty-generator path.
    return iter(())


def test_ingestion_upserts_model_and_benchmark_and_hw(tmp_db):
    summary = ingest.run(
        hf_iter=lambda: _hf_fixture(),
        aa_iter=lambda: _aa_fixture(),
        leaderboard_iter=lambda: _lb_fixture(),
    )
    assert summary["candidates_seen"] == 2
    assert summary["models_upserted"] == 1
    # Rejected candidate should produce a warning entry
    assert any("nobody/tiny" in w for w in summary["warnings"])

    with get_conn(tmp_db) as conn:
        row = conn.execute(
            "SELECT id, name, is_on_whitelist, context_window "
            "FROM models WHERE huggingface_id = ?",
            ("acme/llama-7b",),
        ).fetchone()
        assert row is not None
        assert row["is_on_whitelist"] == 1
        assert row["context_window"] == 8192

        bench = conn.execute(
            "SELECT COUNT(*) AS n FROM model_benchmarks WHERE model_id = ?",
            (row["id"],),
        ).fetchone()
        assert bench["n"] == 1

        hw = conn.execute(
            "SELECT COUNT(*) AS n FROM model_hardware_requirements WHERE model_id = ?",
            (row["id"],),
        ).fetchone()
        # estimator_all_quants emits 4 rows (fp16, int8, int4, q4_k_m)
        assert hw["n"] == 4


def test_ingestion_change_log_records_license_change(tmp_db):
    ingest.run(
        hf_iter=lambda: iter([{
            "huggingface_id": "acme/x",
            "name": "x",
            "parameter_count_b": 3.0,
            "license": "apache-2.0",
            "context_window": 4096,
            "pipeline_tag": "text-generation",
            "tags": ["text-generation"],
            "capabilities": ["text"],
            "downloads": 10000,
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }]),
        aa_iter=lambda: iter(()),
        leaderboard_iter=lambda: iter(()),
    )
    # Re-ingest with a different license
    ingest.run(
        hf_iter=lambda: iter([{
            "huggingface_id": "acme/x",
            "name": "x",
            "parameter_count_b": 3.0,
            "license": "mit",
            "context_window": 4096,
            "pipeline_tag": "text-generation",
            "tags": ["text-generation"],
            "capabilities": ["text"],
            "downloads": 10000,
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }]),
        aa_iter=lambda: iter(()),
        leaderboard_iter=lambda: iter(()),
    )

    with get_conn(tmp_db) as conn:
        changes = conn.execute(
            "SELECT field, old_value, new_value FROM model_change_log"
        ).fetchall()
    fields = {c["field"] for c in changes}
    assert "license_spdx" in fields


def test_aa_missing_key_logs_error_and_increments_counter(tmp_db, monkeypatch, caplog):
    """When ARTIFICIAL_ANALYSIS_KEY is unset, fetch_all must:
      - log at ERROR (not WARNING)
      - increment SKIPPED_DUE_TO_MISSING_KEY
      - yield nothing (still iterable; ingest must not require the key)
    """
    monkeypatch.setattr(aa.config, "ARTIFICIAL_ANALYSIS_KEY", "")
    before = aa.SKIPPED_DUE_TO_MISSING_KEY

    with caplog.at_level(logging.ERROR, logger=aa.log.name):
        rows = list(aa.fetch_all())

    assert rows == []
    assert aa.SKIPPED_DUE_TO_MISSING_KEY == before + 1
    assert any(
        "Artificial Analysis key not configured" in rec.message
        and rec.levelno == logging.ERROR
        for rec in caplog.records
    ), f"expected ERROR log; got {[(r.levelname, r.message) for r in caplog.records]}"


def test_ingest_emits_warn_alert_when_only_hf_leaderboard_yields(tmp_db):
    """If AA produces 0 rows (the only non-HF-Leaderboard benchmark source),
    the ingest summary must contain a WARN-prefixed alert."""
    summary = ingest.run(
        hf_iter=lambda: _hf_fixture(),
        aa_iter=lambda: iter(()),                # AA empty
        leaderboard_iter=lambda: iter(()),       # also empty; doesn't matter
    )
    warns = [w for w in summary["warnings"] if w.startswith("WARN:")]
    assert warns, (
        f"expected a WARN-prefixed alert for empty non-HF-Leaderboard sources; "
        f"got warnings={summary['warnings']}"
    )
    assert "non-HF-Leaderboard" in warns[0]


def test_ingest_no_warn_alert_when_aa_yields_rows(tmp_db):
    """Sanity: when AA produces rows, the WARN alert must NOT fire."""
    summary = ingest.run(
        hf_iter=lambda: _hf_fixture(),
        aa_iter=lambda: _aa_fixture(),
        leaderboard_iter=lambda: iter(()),
    )
    warns = [w for w in summary["warnings"] if w.startswith("WARN:")]
    assert not warns, f"unexpected WARN alert: {warns}"

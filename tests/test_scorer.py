"""Property tests for the scorer (Stage 1 + Stage 2 + geometric mean)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agents.agent2_scoring import scorer
from core.database import get_conn
from core.models.dto import ScoreRequest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_model(
    conn,
    hf_id: str,
    name: str,
    params_b: float,
    context_window: int,
    license_spdx: str,
    whitelist: int,
    benchmarks: dict[str, float],
    hw: list[tuple[str, float]],
    caps: list[str] | None = None,
) -> int:
    now = _now()
    cur = conn.execute(
        """
        INSERT INTO models (huggingface_id, name, provider, parameter_count_b,
            license_spdx, is_on_whitelist, context_window, first_seen_at,
            last_seen_at, is_active)
        VALUES (?,?,?,?,?,?,?,?,?,1)
        """,
        (hf_id, name, "test", params_b, license_spdx, whitelist, context_window, now, now),
    )
    mid = int(cur.lastrowid)
    for c in caps or ["text"]:
        conn.execute(
            "INSERT INTO model_capabilities (model_id, capability) VALUES (?,?)",
            (mid, c),
        )
    for metric, score in benchmarks.items():
        conn.execute(
            """
            INSERT INTO model_benchmarks (model_id, metric_name, source, n_shot, cot,
                score, measured_at) VALUES (?,?,?,?,?,?,?)
            """,
            (mid, metric, "artificial_analysis", 5, 1, score, now),
        )
    for q, vram in hw:
        conn.execute(
            """
            INSERT INTO model_hardware_requirements (model_id, quantization,
                min_vram_gb, recommended_vram_gb, min_ram_gb, estimator_source,
                estimated_at) VALUES (?,?,?,?,?,?,?)
            """,
            (mid, q, vram, vram * 1.25, vram * 1.5, "formula", now),
        )
    return mid


def test_hard_filters_exclude_non_whitelisted(tmp_db):
    with get_conn(tmp_db) as conn:
        _seed_model(conn, "a/x", "X", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 70.0}, [("fp16", 14.0)])
        _seed_model(conn, "b/y", "Y", 7.0, 8192, "mrl-non-commercial", 0,
                    {"MMLU-Pro": 90.0}, [("fp16", 14.0)])

    with get_conn(tmp_db) as conn:
        resp = scorer.score(conn, ScoreRequest(capabilities=["reasoning"]))
    assert resp.shortlist_size == 1
    assert resp.ranked[0].huggingface_id == "a/x"


def test_vram_filter_eliminates_oversized_models(tmp_db):
    with get_conn(tmp_db) as conn:
        _seed_model(conn, "p/s", "Small", 3.0, 8192, "mit", 1,
                    {"MMLU-Pro": 60.0}, [("fp16", 6.0), ("int4", 2.0)])
        _seed_model(conn, "p/b", "Big", 70.0, 8192, "mit", 1,
                    {"MMLU-Pro": 85.0}, [("fp16", 140.0)])
    with get_conn(tmp_db) as conn:
        resp = scorer.score(
            conn, ScoreRequest(capabilities=["reasoning"], vram_gb=24.0)
        )
    assert resp.shortlist_size == 1
    assert resp.ranked[0].huggingface_id == "p/s"


def test_missing_benchmark_on_selected_capability_drops_to_zero(tmp_db):
    """Design §6.3: a model missing a benchmark in any selected capability scores 0."""
    with get_conn(tmp_db) as conn:
        # Model A has both reasoning (MMLU-Pro) and coding (LiveCodeBench) benchmarks
        _seed_model(conn, "a/full", "Full", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 75.0, "LiveCodeBench": 50.0},
                    [("fp16", 14.0)])
        # Model B only has reasoning; should zero out when coding is selected too
        _seed_model(conn, "b/part", "Partial", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 90.0},
                    [("fp16", 14.0)])

    with get_conn(tmp_db) as conn:
        resp = scorer.score(
            conn, ScoreRequest(capabilities=["reasoning", "coding"])
        )
    assert resp.shortlist_size == 2
    # Partial model must be ranked after the full one, with combined_score == 0
    by_id = {r.huggingface_id: r for r in resp.ranked}
    assert by_id["b/part"].combined_score == 0.0
    assert by_id["a/full"].combined_score > 0.0
    assert by_id["a/full"].rank < by_id["b/part"].rank


def test_monotonicity_of_single_capability_score(tmp_db):
    """Higher benchmark => higher score on a single-capability query."""
    with get_conn(tmp_db) as conn:
        _seed_model(conn, "m/low", "L", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 50.0, "GPQA-Diamond": 20.0, "MATH-500": 40.0, "AIME": 10.0},
                    [("fp16", 14.0)])
        _seed_model(conn, "m/high", "H", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 85.0, "GPQA-Diamond": 55.0, "MATH-500": 80.0, "AIME": 40.0},
                    [("fp16", 14.0)])

    with get_conn(tmp_db) as conn:
        resp = scorer.score(conn, ScoreRequest(capabilities=["reasoning"]))
    assert resp.ranked[0].huggingface_id == "m/high"
    assert resp.ranked[1].huggingface_id == "m/low"


def test_empty_shortlist_returns_diagnostic(tmp_db):
    with get_conn(tmp_db) as conn:
        _seed_model(conn, "x/1", "Only", 7.0, 4096, "apache-2.0", 1,
                    {"MMLU-Pro": 70.0}, [("fp16", 14.0)])
    with get_conn(tmp_db) as conn:
        resp = scorer.score(
            conn, ScoreRequest(capabilities=["reasoning"], context_required=32768)
        )
    assert resp.shortlist_size == 0
    assert resp.diagnostic is not None


def test_geometric_mean_zero_anywhere_is_zero():
    assert scorer._geometric_mean([0.9, 0.0, 0.7]) == 0.0
    assert scorer._geometric_mean([0.25, 1.0]) == pytest.approx(0.5, abs=1e-9)

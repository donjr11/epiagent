"""Property tests for the scorer (Stage 1 + Stage 2 + geometric mean)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agents.agent2_scoring import capabilities as cap_mod
from agents.agent2_scoring import scorer
from core.database import get_conn
from core.models.dto import CAPABILITIES, ScoreRequest


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
    # Default to a generic text+reasoning model; tests that exercise tag
    # filtering should pass an explicit `caps`.
    for c in caps or ["text", "reasoning"]:
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
        # Both models advertise reasoning + coding so the tag filter doesn't
        # cull them; the test isolates the *missing benchmark* behavior.
        _seed_model(conn, "a/full", "Full", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 75.0, "LiveCodeBench": 50.0},
                    [("fp16", 14.0)],
                    caps=["text", "reasoning", "coding"])
        # Model B only has reasoning benchmarks; should zero out when coding
        # is selected too.
        _seed_model(conn, "b/part", "Partial", 7.0, 8192, "apache-2.0", 1,
                    {"MMLU-Pro": 90.0},
                    [("fp16", 14.0)],
                    caps=["text", "reasoning", "coding"])

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


def _seed_capability_property_corpus(conn) -> None:
    """Seed two models per capability: one with the matching tag, one without.

    The model lacking the tag must be excluded from the ranking; the tagged
    model is the only legal answer when its capability is queried.
    """
    rich_bench = {
        "MMLU-Pro": 70.0, "GPQA-Diamond": 40.0, "MATH-500": 60.0, "AIME": 25.0,
        "LiveCodeBench": 45.0, "SWE-Bench-Verified": 20.0, "RULER": 80.0,
        "IFEval": 85.0, "Terminal-Bench": 50.0, "MMBench": 75.0, "MMMU": 50.0,
        "AudioBench": 60.0, "MMAU": 55.0,
    }
    # tag-bearing variants
    _seed_model(conn, "tag/coding", "TagCoding", 7.0, 8192, "apache-2.0", 1,
                rich_bench, [("fp16", 14.0)],
                caps=["text", "reasoning", "coding", "tools"])
    _seed_model(conn, "tag/reasoning", "TagReasoning", 7.0, 8192, "apache-2.0", 1,
                rich_bench, [("fp16", 14.0)],
                caps=["text", "reasoning"])
    _seed_model(conn, "tag/tools", "TagTools", 7.0, 8192, "apache-2.0", 1,
                rich_bench, [("fp16", 14.0)],
                caps=["text", "reasoning", "tools"])
    _seed_model(conn, "tag/vision", "TagVision", 7.0, 8192, "apache-2.0", 1,
                rich_bench, [("fp16", 14.0)],
                caps=["text", "vision"])
    _seed_model(conn, "tag/audio", "TagAudio", 7.0, 8192, "apache-2.0", 1,
                rich_bench, [("fp16", 14.0)],
                caps=["text", "audio"])
    # untagged "text-only" variant — must NEVER appear under tag-gated capability
    _seed_model(conn, "untagged/plain", "Untagged", 7.0, 8192, "apache-2.0", 1,
                rich_bench, [("fp16", 14.0)],
                caps=["text"])


@pytest.mark.parametrize("capability", CAPABILITIES)
def test_capability_required_tag_filters_untagged_models(tmp_db, capability):
    """For each of the 7 capabilities, a model lacking the corresponding
    required tag (if any) MUST NOT appear in the ranking.

    For capabilities whose required tag is None (cross-cutting), the property
    is vacuous and any model is permitted.
    """
    with get_conn(tmp_db) as conn:
        _seed_capability_property_corpus(conn)

    with get_conn(tmp_db) as conn:
        resp = scorer.score(conn, ScoreRequest(capabilities=[capability]))

    required_tag = cap_mod.CAPABILITY_REQUIRED_TAG.get(capability)
    if required_tag is None:
        # No tag-gate; we still expect at least one ranked model from the corpus.
        assert resp.shortlist_size >= 1, (
            f"capability={capability} (cross-cutting): expected non-empty "
            f"shortlist, got {resp.shortlist_size}"
        )
        return

    # Tag-gated: every ranked model must carry the required tag.
    with get_conn(tmp_db) as conn:
        for row in resp.ranked:
            mid = conn.execute(
                "SELECT id FROM models WHERE huggingface_id = ?",
                (row.huggingface_id,),
            ).fetchone()["id"]
            tags = {
                r["capability"]
                for r in conn.execute(
                    "SELECT capability FROM model_capabilities WHERE model_id=?",
                    (mid,),
                )
            }
            assert required_tag in tags, (
                f"capability={capability}: ranked model {row.huggingface_id} "
                f"lacks required tag '{required_tag}' (its tags: {tags})"
            )

    # And the untagged plain-text model must be excluded for tag-gated caps.
    ranked_ids = {r.huggingface_id for r in resp.ranked}
    assert "untagged/plain" not in ranked_ids, (
        f"capability={capability}: untagged/plain leaked into ranking"
    )

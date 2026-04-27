"""Artificial Analysis client: primary per-capability benchmarks.

Uses AA's public models endpoint. ToS verification is pending (Open Item #5);
this client logs an ERROR (not a silent warning) when no API key is configured
and returns an empty iterator so ingestion degrades gracefully to the HF
fallback. The skip count is tracked in `SKIPPED_DUE_TO_MISSING_KEY` so the
ingest summary can surface chronic mis-configuration.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator

import httpx

from core import config

log = logging.getLogger(__name__)


# Process-lifetime counter incremented every time fetch_all is called without
# an API key configured. Read by the ingest orchestrator for summary alerts.
SKIPPED_DUE_TO_MISSING_KEY: int = 0


# Map AA metric labels onto the canonical metric names used by scoring_profiles.
AA_METRIC_MAP = {
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA-Diamond",
    "livecodebench": "LiveCodeBench",
    "math_500": "MATH-500",
    "aime": "AIME",
    "scicode": "SciCode",
    "ifbench": "IFBench",
}


def fetch_all() -> Iterator[dict]:
    """Yield benchmark rows {huggingface_id, metric_name, score, source, ...}.

    If the API key is unset, logs an ERROR (so the operator notices) and
    returns an empty iterator. The key is NOT required — ingestion proceeds.
    """
    global SKIPPED_DUE_TO_MISSING_KEY
    if not config.ARTIFICIAL_ANALYSIS_KEY:
        SKIPPED_DUE_TO_MISSING_KEY += 1
        log.error(
            "Artificial Analysis key not configured; skipping AA ingestion "
            "(skip_count=%d). Falling back to HF-Leaderboard only.",
            SKIPPED_DUE_TO_MISSING_KEY,
        )
        return
    headers = {
        "Accept": "application/json",
        "x-api-key": config.ARTIFICIAL_ANALYSIS_KEY,
    }
    now = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=60.0, headers=headers) as client:
        resp = client.get(f"{config.ARTIFICIAL_ANALYSIS_BASE}/models")
        resp.raise_for_status()
        payload = resp.json()
        for m in payload.get("data", payload if isinstance(payload, list) else []):
            hf_id = m.get("huggingface_id") or m.get("model_id")
            if not hf_id:
                continue
            evals = m.get("evaluations") or {}
            for aa_key, canonical in AA_METRIC_MAP.items():
                if aa_key not in evals:
                    continue
                score = evals[aa_key]
                if score is None:
                    continue
                yield {
                    "huggingface_id": hf_id,
                    "metric_name": canonical,
                    "source": "artificial_analysis",
                    "n_shot": 5,
                    "cot": 1,
                    "score": float(score),
                    "measured_at": now,
                }

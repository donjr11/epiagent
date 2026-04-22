"""Artificial Analysis client: primary per-capability benchmarks.

Uses AA's public models endpoint. ToS verification is pending (Open Item #5);
this client logs a single warning when no API key is configured and returns
an empty iterator so ingestion degrades gracefully to the HF fallback.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator

import httpx

from core import config

log = logging.getLogger(__name__)


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
    """Yield benchmark rows {huggingface_id, metric_name, score, source, ...}."""
    if not config.ARTIFICIAL_ANALYSIS_KEY:
        log.warning("ARTIFICIAL_ANALYSIS_KEY not configured; AA source skipped.")
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

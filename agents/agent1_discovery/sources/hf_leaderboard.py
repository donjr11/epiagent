"""HF Open LLM Leaderboard v2 client (secondary source).

Pulls parquet-backed dataset rows via the datasets-server API. Returns
benchmark rows keyed by huggingface_id. Used to fill coverage gaps when
Artificial Analysis has no data for a model.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator

import httpx

from core import config

log = logging.getLogger(__name__)

HF_DATASETS_API = "https://datasets-server.huggingface.co/rows"

# Canonical metric names aligned with scoring_profiles.
LEADERBOARD_METRIC_MAP = {
    "ifeval": "IFEval",
    "bbh": "BBH",
    "gpqa": "GPQA",
    "math_lvl_5": "MATH",
    "mmlu_pro": "MMLU-Pro",
    "musr": "MuSR",
}


def _headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if config.HF_API_TOKEN:
        h["Authorization"] = f"Bearer {config.HF_API_TOKEN}"
    return h


def fetch_all(page_size: int = 100, max_rows: int = 2000) -> Iterator[dict]:
    dataset = config.HF_LEADERBOARD_DATASET
    now = datetime.now(timezone.utc).isoformat()
    offset = 0
    with httpx.Client(timeout=60.0, headers=_headers()) as client:
        while offset < max_rows:
            resp = client.get(
                HF_DATASETS_API,
                params={
                    "dataset": dataset,
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": page_size,
                },
            )
            if resp.status_code >= 400:
                log.warning("HF leaderboard fetch failed: %s", resp.text[:200])
                return
            rows = resp.json().get("rows", [])
            if not rows:
                return
            for row in rows:
                rec = row.get("row", {})
                hf_id = rec.get("fullname") or rec.get("model") or rec.get("eval_name")
                if not hf_id:
                    continue
                for key, canonical in LEADERBOARD_METRIC_MAP.items():
                    if key in rec and rec[key] is not None:
                        yield {
                            "huggingface_id": hf_id,
                            "metric_name": canonical,
                            "source": "hf_leaderboard_v2",
                            "n_shot": 0,
                            "cot": 0,
                            "score": float(rec[key]),
                            "measured_at": now,
                        }
            offset += page_size
            if len(rows) < page_size:
                return

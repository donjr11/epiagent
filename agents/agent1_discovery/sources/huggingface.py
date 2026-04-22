"""HuggingFace Hub client: metadata authoritative source.

Yields raw records shaped for the policy + ingest writer. Network calls are
isolated here so `tests/test_ingestion.py` can swap in a fake iterator.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import httpx

from core import config

HF_API = "https://huggingface.co/api"


def _headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if config.HF_API_TOKEN:
        h["Authorization"] = f"Bearer {config.HF_API_TOKEN}"
    return h


def list_candidate_models(
    pipelines: Iterable[str] = ("text-generation", "image-text-to-text"),
    min_downloads: int = 1000,
    limit: int = 500,
) -> Iterator[dict]:
    """Yield sorted candidate models by downloads, per pipeline tag."""
    with httpx.Client(timeout=30.0, headers=_headers()) as client:
        for pipeline in pipelines:
            params = {
                "pipeline_tag": pipeline,
                "sort": "downloads",
                "direction": "-1",
                "limit": limit,
                "full": "true",
            }
            resp = client.get(f"{HF_API}/models", params=params)
            resp.raise_for_status()
            for raw in resp.json():
                if (raw.get("downloads") or 0) < min_downloads:
                    continue
                yield _normalize(raw)


def get_model(huggingface_id: str) -> dict | None:
    with httpx.Client(timeout=30.0, headers=_headers()) as client:
        resp = client.get(f"{HF_API}/models/{huggingface_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _normalize(resp.json())


def _normalize(raw: dict) -> dict:
    """Project the HF API payload into EpiAgent's canonical record."""
    card = raw.get("cardData") or {}
    config_dict = raw.get("config") or {}
    tags = raw.get("tags") or []

    # params in Billions if available
    params_b = None
    if raw.get("safetensors") and raw["safetensors"].get("total"):
        params_b = round(raw["safetensors"]["total"] / 1e9, 3)

    context_window = None
    for key in ("max_position_embeddings", "max_seq_len", "n_positions"):
        if key in config_dict:
            context_window = int(config_dict[key])
            break

    capability_tags: list[str] = ["text"]
    if any("vision" in t for t in tags) or "image-text-to-text" in tags:
        capability_tags.append("vision")
    if any("audio" in t for t in tags) or "audio-text-to-text" in tags:
        capability_tags.append("audio")
    if "code" in tags or any("coder" in t for t in tags):
        capability_tags.append("coding")

    return {
        "huggingface_id": raw["id"],
        "name": raw.get("modelId", raw["id"]).split("/")[-1],
        "provider": raw["id"].split("/")[0] if "/" in raw["id"] else None,
        "parameter_count_b": params_b,
        "license": card.get("license") or raw.get("license"),
        "context_window": context_window,
        "pipeline_tag": raw.get("pipeline_tag"),
        "tags": tags,
        "capabilities": capability_tags,
        "downloads": raw.get("downloads"),
        "last_modified": raw.get("lastModified"),
        "release_date": raw.get("createdAt"),
    }

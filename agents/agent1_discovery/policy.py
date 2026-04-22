"""Ingestion candidate-filter policy (Section 9.1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from core import config


def is_candidate(model_info: dict[str, Any]) -> tuple[bool, str | None]:
    """Decide whether a raw HF model record is worth ingesting.

    Returns (ok, reason_if_rejected).
    """
    tags = set(model_info.get("tags") or [])
    pipeline = model_info.get("pipeline_tag")

    # text / vision-text / audio-text generation models only
    allowed_pipelines = {
        "text-generation",
        "image-text-to-text",
        "audio-text-to-text",
        "automatic-speech-recognition",  # multi-modal boundary; kept optional
    }
    if pipeline not in allowed_pipelines and not (tags & allowed_pipelines):
        return False, f"pipeline_tag={pipeline!r} not in allowed set"

    downloads = model_info.get("downloads") or 0
    if downloads < config.MIN_DOWNLOADS:
        return False, f"downloads={downloads} < {config.MIN_DOWNLOADS}"

    last_modified = model_info.get("last_modified") or model_info.get("lastModified")
    if last_modified:
        try:
            if isinstance(last_modified, str):
                ts = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            else:
                ts = last_modified
            cutoff = datetime.now(timezone.utc) - timedelta(days=config.MODIFIED_WITHIN_DAYS)
            if ts < cutoff and not model_info.get("_already_in_db"):
                return False, "not modified within window and not already tracked"
        except Exception:
            pass  # permissive: don't reject for parse errors

    params_b = model_info.get("parameter_count_b")
    if params_b is not None:
        if params_b < config.MIN_PARAMS_B or params_b > config.MAX_PARAMS_B:
            return False, f"param_count_b={params_b} out of [{config.MIN_PARAMS_B}, {config.MAX_PARAMS_B}]"

    license_ = model_info.get("license")
    if not license_:
        return False, "missing license"

    return True, None

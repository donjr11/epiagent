"""License whitelist (Section 5 of design doc).

Pending PO confirmation. Models outside the whitelist are still ingested but
flagged `is_on_whitelist=false` and excluded from scoring.
"""
from __future__ import annotations

# SPDX identifiers where available; custom community licenses use slugs
# documented on HuggingFace (license field).
WHITELIST: dict[str, dict] = {
    "apache-2.0": {"commercial": True, "note": "Unambiguous commercial use."},
    "mit": {"commercial": True, "note": "Unambiguous commercial use."},
    "bsd-3-clause": {"commercial": True, "note": "Unambiguous commercial use."},
    "llama3": {"commercial": True, "note": "Commercial OK below 700M MAU."},
    "llama3.1": {"commercial": True, "note": "Commercial OK below 700M MAU."},
    "llama3.2": {"commercial": True, "note": "Commercial OK below 700M MAU."},
    "llama3.3": {"commercial": True, "note": "Commercial OK below 700M MAU."},
    "gemma": {"commercial": True, "note": "Google custom; check use-case restrictions."},
    "gemma-terms-of-use": {"commercial": True, "note": "Google custom."},
    "qwen": {"commercial": True, "note": "Generally permissive."},
    "deepseek-license": {"commercial": True, "note": "Generally permissive."},
    # Flagged non-commercial but kept in whitelist for research shortlist. PO
    # may decide to exclude.
    "mrl": {"commercial": False, "note": "Mistral Research License, non-commercial."},
}


def normalize(license_str: str | None) -> str | None:
    if not license_str:
        return None
    return license_str.strip().lower()


def is_whitelisted(license_str: str | None) -> bool:
    key = normalize(license_str)
    if not key:
        return False
    return key in WHITELIST

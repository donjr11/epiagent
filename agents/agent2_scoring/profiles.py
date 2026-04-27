"""Read/write `scoring_profiles` rows (BWM output storage)."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from agents.agent2_scoring import bwm, capabilities

log = logging.getLogger(__name__)


class ProfileMissingError(LookupError):
    """Raised by load_profile when no scoring_profiles row exists for a capability.

    The scorer must NOT silently fall back to uniform weights (design §6.4).
    The API layer should translate this to HTTP 503 so the operator notices.
    """

    def __init__(self, capability: str):
        super().__init__(
            f"No scoring_profiles row for capability '{capability}'. "
            f"Run `python -m scripts.seed_strawman_profiles` or run the BWM CLI."
        )
        self.capability = capability


@dataclass
class Profile:
    capability: str
    version: int
    criteria: list[str]
    weights: dict[str, float]
    source_preferences: dict[str, list[tuple[str, int, int]]]
    consistency_ratio: Optional[float]


def load_profile(conn: sqlite3.Connection, capability: str) -> Profile:
    row = conn.execute(
        "SELECT version, criteria, weights, source_preferences, consistency_ratio "
        "FROM scoring_profiles WHERE capability = ?",
        (capability,),
    ).fetchone()
    if row is None:
        # Loud failure: no silent uniform-weight fallback (design §6.4).
        log.error(
            "scoring_profiles row missing for capability=%r; refusing to score. "
            "Run scripts.seed_strawman_profiles or the BWM CLI.",
            capability,
        )
        raise ProfileMissingError(capability)
    return Profile(
        capability=capability,
        version=int(row["version"]),
        criteria=json.loads(row["criteria"]),
        weights=json.loads(row["weights"]),
        source_preferences={
            k: [tuple(t) for t in v]
            for k, v in json.loads(row["source_preferences"]).items()
        },
        consistency_ratio=row["consistency_ratio"],
    )


def save_profile_from_bwm(
    conn: sqlite3.Connection,
    capability: str,
    solution: bwm.BWMSolution,
    source_preferences: dict[str, list[tuple[str, int, int]]] | None = None,
    created_by: str = "po",
) -> None:
    """Persist a BWM solution to scoring_profiles.

    `created_by` is recorded inside the bwm_input JSON blob (the schema has no
    dedicated column). Use "adam_strawman" for placeholder weights and "po" for
    PO-elicited ones.
    """
    if solution.consistency_ratio > 0.10:
        raise ValueError(
            f"CR={solution.consistency_ratio:.3f} > 0.10; PO must revise."
        )
    prefs = source_preferences or {
        c: capabilities.DEFAULT_SOURCE_PREFERENCES.get(c, [])
        for c in solution.criteria
    }
    existing = conn.execute(
        "SELECT version FROM scoring_profiles WHERE capability = ?", (capability,)
    ).fetchone()
    next_version = (int(existing["version"]) + 1) if existing else 1
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO scoring_profiles (
            capability, version, criteria, weights, source_preferences,
            bwm_input, consistency_ratio, created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(capability) DO UPDATE SET
            version = excluded.version,
            criteria = excluded.criteria,
            weights = excluded.weights,
            source_preferences = excluded.source_preferences,
            bwm_input = excluded.bwm_input,
            consistency_ratio = excluded.consistency_ratio,
            created_at = excluded.created_at
        """,
        (
            capability,
            next_version,
            json.dumps(solution.criteria),
            json.dumps(solution.weights),
            json.dumps(prefs),
            json.dumps({
                "best": solution.best,
                "worst": solution.worst,
                "created_by": created_by,
            }),
            solution.consistency_ratio,
            now,
        ),
    )

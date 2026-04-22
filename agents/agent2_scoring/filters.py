"""Stage 1 hard filters (Section 6.1)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

from agents.agent2_scoring import capabilities


@dataclass
class FilterInput:
    capabilities: list[str]
    vram_gb: Optional[float]
    context_required: Optional[int]
    quantization: Optional[str]


def shortlist(conn: sqlite3.Connection, f: FilterInput) -> list[int]:
    """Return model ids passing all hard filters, preserving whitelist policy."""
    required_tags: set[str] = set()
    for cap in f.capabilities:
        tag = capabilities.CAPABILITY_REQUIRED_TAG.get(cap)
        if tag:
            required_tags.add(tag)

    sql = [
        "SELECT m.id FROM models m",
        "WHERE m.is_active = 1 AND m.is_on_whitelist = 1",
    ]
    params: list = []

    if f.context_required is not None:
        sql.append("AND COALESCE(m.context_window, 0) >= ?")
        params.append(int(f.context_required))

    if f.vram_gb is not None:
        # at least one quantization row satisfies min_vram_gb <= vram_gb
        cond = (
            "AND EXISTS (SELECT 1 FROM model_hardware_requirements h "
            "WHERE h.model_id = m.id AND h.min_vram_gb <= ?"
        )
        params.append(float(f.vram_gb))
        if f.quantization:
            cond += " AND h.quantization = ?"
            params.append(f.quantization)
        cond += ")"
        sql.append(cond)

    for tag in required_tags:
        sql.append(
            "AND EXISTS (SELECT 1 FROM model_capabilities c "
            "WHERE c.model_id = m.id AND c.capability = ?)"
        )
        params.append(tag)

    rows = conn.execute(" ".join(sql), params).fetchall()
    return [int(r["id"]) for r in rows]

"""Pydantic DTOs for the scoring API surface."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


CAPABILITIES = [
    "coding",
    "reasoning",
    "rag_long_context",
    "agentic",
    "vision_language",
    "audio_language",
    "minimum_cost",
]


class ScoreRequest(BaseModel):
    capabilities: list[str] = Field(..., min_length=1, max_length=7)
    vram_gb: Optional[float] = None
    context_required: Optional[int] = None
    quantization: Optional[str] = None           # fp16|int8|int4|...
    top_n: int = 25


class CapabilityBreakdown(BaseModel):
    capability: str
    score: float                                  # [0,1]
    contributions: dict[str, float]               # metric -> w_i * x_hat_i


class ModelHardwareInfo(BaseModel):
    quantization: str
    min_vram_gb: float
    recommended_vram_gb: Optional[float] = None


class ModelRankRow(BaseModel):
    rank: int
    huggingface_id: str
    name: str
    license_spdx: Optional[str]
    parameter_count_b: Optional[float]
    context_window: Optional[int]
    combined_score: float
    per_capability: list[CapabilityBreakdown]
    hardware: list[ModelHardwareInfo]
    why: str                                      # short human summary


class ScoreResponse(BaseModel):
    shortlist_size: int
    ranked: list[ModelRankRow]
    diagnostic: Optional[str] = None

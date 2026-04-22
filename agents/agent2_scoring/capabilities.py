"""Capability definitions (Section 7 of design doc).

Each capability lists its criteria (metric names), whether each metric is a
benefit or cost, and a default source-preference ordering used to resolve
which (source, n_shot, cot) tuple to read from model_benchmarks.

Default weights are uniform until BWM sessions run with the PO; stored
versions in `scoring_profiles` override these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricKind = Literal["benefit", "cost"]


@dataclass(frozen=True)
class Criterion:
    name: str
    kind: MetricKind          # "benefit" or "cost"
    source: str = "model_benchmarks"   # or "models" for context_window etc.


# Metric fields exposed directly on the models row
MODEL_FIELD_METRICS = {
    "context_window": Criterion("context_window", "benefit", source="models"),
    "min_vram_gb": Criterion("min_vram_gb", "cost", source="hardware"),
    "throughput_tok_s": Criterion("throughput_tok_s", "benefit"),
    "ttft_ms": Criterion("ttft_ms", "cost"),
    "tool_calling": Criterion("tool_calling", "benefit", source="capability_flag"),
}


CAPABILITY_CRITERIA: dict[str, list[Criterion]] = {
    "coding": [
        Criterion("LiveCodeBench", "benefit"),
        Criterion("SWE-Bench-Verified", "benefit"),
        MODEL_FIELD_METRICS["context_window"],
        MODEL_FIELD_METRICS["throughput_tok_s"],
        MODEL_FIELD_METRICS["min_vram_gb"],
    ],
    "reasoning": [
        Criterion("MMLU-Pro", "benefit"),
        Criterion("GPQA-Diamond", "benefit"),
        Criterion("MATH-500", "benefit"),
        Criterion("AIME", "benefit"),
        MODEL_FIELD_METRICS["context_window"],
        MODEL_FIELD_METRICS["min_vram_gb"],
    ],
    "rag_long_context": [
        MODEL_FIELD_METRICS["context_window"],
        Criterion("RULER", "benefit"),
        Criterion("MMLU-Pro", "benefit"),
        MODEL_FIELD_METRICS["throughput_tok_s"],
        MODEL_FIELD_METRICS["ttft_ms"],
        MODEL_FIELD_METRICS["min_vram_gb"],
    ],
    "agentic": [
        Criterion("Terminal-Bench", "benefit"),
        Criterion("IFEval", "benefit"),
        MODEL_FIELD_METRICS["tool_calling"],
        MODEL_FIELD_METRICS["context_window"],
        Criterion("LiveCodeBench", "benefit"),
    ],
    "vision_language": [
        Criterion("MMBench", "benefit"),
        Criterion("MMMU", "benefit"),
        MODEL_FIELD_METRICS["context_window"],
        MODEL_FIELD_METRICS["min_vram_gb"],
    ],
    "audio_language": [
        Criterion("AudioBench", "benefit"),
        Criterion("MMAU", "benefit"),
        MODEL_FIELD_METRICS["context_window"],
        MODEL_FIELD_METRICS["min_vram_gb"],
    ],
    "minimum_cost": [
        MODEL_FIELD_METRICS["min_vram_gb"],
        MODEL_FIELD_METRICS["throughput_tok_s"],
        Criterion("MMLU-Pro", "benefit"),
    ],
}


# Gate tags: if a capability requires a capability-tag, add it here so that
# Stage 1 filter can apply it. Maps capability -> required model_capabilities tag.
CAPABILITY_REQUIRED_TAG: dict[str, str | None] = {
    "coding": None,
    "reasoning": None,
    "rag_long_context": None,
    "agentic": None,
    "vision_language": "vision",
    "audio_language": "audio",
    "minimum_cost": None,
}


# Default source preferences for metrics. Higher priority first.
DEFAULT_SOURCE_PREFERENCES: dict[str, list[tuple[str, int, int]]] = {
    "MMLU-Pro": [("artificial_analysis", 5, 1), ("hf_leaderboard_v2", 0, 0)],
    "GPQA-Diamond": [("artificial_analysis", 5, 1)],
    "GPQA": [("hf_leaderboard_v2", 0, 0)],
    "LiveCodeBench": [("artificial_analysis", 5, 1)],
    "SWE-Bench-Verified": [("artificial_analysis", 0, 0)],
    "MATH-500": [("artificial_analysis", 5, 1)],
    "MATH": [("hf_leaderboard_v2", 0, 0)],
    "AIME": [("artificial_analysis", 5, 1)],
    "SciCode": [("artificial_analysis", 5, 1)],
    "IFBench": [("artificial_analysis", 5, 1)],
    "IFEval": [("artificial_analysis", 5, 1), ("hf_leaderboard_v2", 0, 0)],
    "BBH": [("hf_leaderboard_v2", 0, 0)],
    "RULER": [("artificial_analysis", 0, 0)],
    "MMBench": [("artificial_analysis", 0, 0)],
    "MMMU": [("artificial_analysis", 0, 0)],
    "AudioBench": [("artificial_analysis", 0, 0)],
    "MMAU": [("artificial_analysis", 0, 0)],
    "Terminal-Bench": [("artificial_analysis", 0, 0)],
}

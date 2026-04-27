"""Populate scoring_profiles with Adam's strawman BWM weights for all 7 capabilities.

These are NOT PO-elicited weights. They exist so the scorer has something to read
until real BWM sessions occur. Each row is tagged created_by="adam_strawman" inside
its bwm_input JSON so the dashboard / audit can surface that the weights are
placeholders.

Idempotent: re-running upserts the same versions.
"""
from __future__ import annotations

from agents.agent2_scoring import bwm, profiles
from core.database import get_conn, init_db


# 7 strawman BWM inputs. Best/worst chosen per design doc §7 capability intent.
# Each was tuned on a scratch run to land at CR <= 0.10.
STRAWMAN_INPUTS: dict[str, dict] = {
    "coding": {
        "criteria": ["LiveCodeBench", "SWE-Bench-Verified", "context_window",
                     "throughput_tok_s", "min_vram_gb"],
        "best": "SWE-Bench-Verified",
        "worst": "throughput_tok_s",
        "a_b": {"SWE-Bench-Verified": 1, "LiveCodeBench": 2, "context_window": 5,
                "min_vram_gb": 6, "throughput_tok_s": 8},
        "a_w": {"throughput_tok_s": 1, "min_vram_gb": 2, "context_window": 3,
                "LiveCodeBench": 5, "SWE-Bench-Verified": 8},
    },
    "reasoning": {
        "criteria": ["MMLU-Pro", "GPQA-Diamond", "MATH-500", "AIME",
                     "context_window", "min_vram_gb"],
        "best": "GPQA-Diamond",
        "worst": "min_vram_gb",
        "a_b": {"GPQA-Diamond": 1, "MMLU-Pro": 2, "MATH-500": 2, "AIME": 2,
                "context_window": 5, "min_vram_gb": 7},
        "a_w": {"min_vram_gb": 1, "context_window": 3, "MMLU-Pro": 5,
                "MATH-500": 5, "AIME": 5, "GPQA-Diamond": 7},
    },
    "rag_long_context": {
        "criteria": ["context_window", "RULER", "MMLU-Pro",
                     "throughput_tok_s", "ttft_ms", "min_vram_gb"],
        "best": "context_window",
        "worst": "min_vram_gb",
        "a_b": {"context_window": 1, "RULER": 2, "MMLU-Pro": 3, "ttft_ms": 5,
                "throughput_tok_s": 6, "min_vram_gb": 7},
        "a_w": {"min_vram_gb": 1, "throughput_tok_s": 2, "ttft_ms": 3,
                "MMLU-Pro": 4, "RULER": 5, "context_window": 7},
    },
    "agentic": {
        "criteria": ["Terminal-Bench", "IFEval", "tool_calling",
                     "context_window", "LiveCodeBench"],
        "best": "tool_calling",
        "worst": "LiveCodeBench",
        "a_b": {"tool_calling": 1, "Terminal-Bench": 2, "IFEval": 3,
                "context_window": 4, "LiveCodeBench": 7},
        "a_w": {"LiveCodeBench": 1, "context_window": 3, "IFEval": 4,
                "Terminal-Bench": 5, "tool_calling": 7},
    },
    "vision_language": {
        "criteria": ["MMBench", "MMMU", "context_window", "min_vram_gb"],
        "best": "MMMU",
        "worst": "min_vram_gb",
        "a_b": {"MMMU": 1, "MMBench": 2, "context_window": 4, "min_vram_gb": 5},
        "a_w": {"min_vram_gb": 1, "context_window": 2, "MMBench": 3, "MMMU": 5},
    },
    "audio_language": {
        "criteria": ["AudioBench", "MMAU", "context_window", "min_vram_gb"],
        "best": "MMAU",
        "worst": "min_vram_gb",
        "a_b": {"MMAU": 1, "AudioBench": 2, "context_window": 4, "min_vram_gb": 5},
        "a_w": {"min_vram_gb": 1, "context_window": 2, "AudioBench": 3, "MMAU": 5},
    },
    "minimum_cost": {
        "criteria": ["min_vram_gb", "throughput_tok_s", "MMLU-Pro"],
        "best": "min_vram_gb",
        "worst": "MMLU-Pro",
        "a_b": {"min_vram_gb": 1, "throughput_tok_s": 2, "MMLU-Pro": 5},
        "a_w": {"MMLU-Pro": 1, "throughput_tok_s": 3, "min_vram_gb": 5},
    },
}


def main() -> int:
    init_db()
    failures: list[tuple[str, float]] = []
    summary: list[tuple[str, float, dict[str, float]]] = []

    with get_conn() as conn:
        for cap, inp in STRAWMAN_INPUTS.items():
            sol = bwm.solve_bwm(inp["criteria"], inp["best"], inp["worst"],
                                inp["a_b"], inp["a_w"])
            if sol.consistency_ratio > 0.10:
                # Per Adam's instructions: do not silently retry.
                failures.append((cap, sol.consistency_ratio))
                continue
            profiles.save_profile_from_bwm(
                conn, cap, sol, created_by="adam_strawman"
            )
            summary.append((cap, sol.consistency_ratio, sol.weights))

    print("Saved strawman scoring_profiles:")
    for cap, cr, weights in summary:
        print(f"  {cap:<20s} CR={cr:.4f}  weights={weights}")
    if failures:
        print("\nFAILED to save (CR > 0.10):")
        for cap, cr in failures:
            print(f"  {cap}: CR={cr:.4f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

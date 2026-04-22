"""VRAM estimator sidecar.

Formula-based. Kept deliberately simple for v1 — labelled "approximate" in
the UI. Matches the design doc risk note: v2 may integrate a real
serving-stack measurement.

For a decoder-only LLM at inference time, dominant memory is weights plus
KV-cache. We approximate:

    weights_gb    = params_B * bytes_per_param
    kv_cache_gb   = 2 * n_layers * n_heads * head_dim * context * bytes_per_param_kv / 1e9

We lack n_layers/n_heads/head_dim for most models without probing the config,
so we use a compact heuristic keyed on parameter count. Accuracy target:
within ~25% for typical transformer shapes; flagged approximate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Quant = Literal["fp32", "fp16", "int8", "int4", "q4_k_m"]

_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "q4_k_m": 0.55,   # GGUF q4_k_m averages ~4.4 bits per weight
}

# Architectural rough shape as a function of params_B. Used only for
# KV-cache estimation. Values are representative for modern dense decoders.
def _arch_from_params(params_b: float) -> tuple[int, int, int]:
    if params_b <= 1.5:
        return 22, 16, 64     # ~1B
    if params_b <= 4:
        return 32, 24, 80     # ~3B
    if params_b <= 9:
        return 32, 32, 128    # ~7B
    if params_b <= 16:
        return 40, 40, 128    # ~13B
    if params_b <= 40:
        return 60, 48, 128    # ~30B
    if params_b <= 80:
        return 80, 64, 128    # ~70B
    return 120, 96, 128       # 100B+


@dataclass
class VramEstimate:
    quantization: str
    min_vram_gb: float
    recommended_vram_gb: float
    min_ram_gb: float
    estimator_source: str = "formula"


def estimate(
    params_b: float,
    context_window: int,
    quantization: Quant = "fp16",
    kv_cache_bytes_per_elem: float = 2.0,
) -> VramEstimate:
    """Return a conservative VRAM estimate (GB) for params + KV cache."""
    bpp = _BYTES_PER_PARAM.get(quantization, 2.0)
    weights_gb = params_b * bpp   # params_b is already in billions -> GB direct

    n_layers, n_heads, head_dim = _arch_from_params(params_b)
    # KV cache: 2 (K and V) * layers * heads * head_dim * seq_len * bytes
    kv_bytes = 2.0 * n_layers * n_heads * head_dim * context_window * kv_cache_bytes_per_elem
    kv_gb = kv_bytes / 1e9

    overhead_gb = 1.0 + 0.05 * weights_gb  # activations, CUDA runtime, fragmentation
    min_vram = weights_gb + kv_gb + overhead_gb
    recommended = min_vram * 1.25
    min_ram = max(weights_gb * 1.5, 8.0)
    return VramEstimate(
        quantization=quantization,
        min_vram_gb=round(min_vram, 2),
        recommended_vram_gb=round(recommended, 2),
        min_ram_gb=round(min_ram, 2),
    )


def estimate_all_quants(params_b: float, context_window: int) -> list[VramEstimate]:
    """Produce one row per quantization level for the hardware table."""
    out: list[VramEstimate] = []
    for q in ("fp16", "int8", "int4", "q4_k_m"):
        out.append(estimate(params_b, context_window, q))
    return out

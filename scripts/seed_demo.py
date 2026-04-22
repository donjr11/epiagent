"""Seed a small demo dataset so the dashboard has something to render before
the nightly ingestion runs. Idempotent: reruns upsert the same models."""
from __future__ import annotations

from datetime import datetime, timezone

from core.database import get_conn, init_db

_NOW = datetime.now(timezone.utc).isoformat()

DEMO_MODELS = [
    dict(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama-3.1-8B-Instruct",
        provider="meta-llama",
        params=8.0,
        license="llama3.1",
        ctx=131072,
        caps=["text"],
        bench={"MMLU-Pro": 48.3, "GPQA-Diamond": 30.4, "MATH-500": 51.9,
               "AIME": 15.3, "LiveCodeBench": 28.6, "IFEval": 80.4},
        hw=[("fp16", 18.0), ("int8", 10.0), ("int4", 6.0), ("q4_k_m", 6.5)],
    ),
    dict(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen2.5-7B-Instruct",
        provider="Qwen",
        params=7.6,
        license="apache-2.0",
        ctx=131072,
        caps=["text"],
        bench={"MMLU-Pro": 56.3, "GPQA-Diamond": 36.4, "MATH-500": 75.5,
               "AIME": 23.3, "LiveCodeBench": 39.0, "IFEval": 82.6},
        hw=[("fp16", 16.5), ("int8", 9.0), ("int4", 5.5), ("q4_k_m", 6.0)],
    ),
    dict(
        hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        name="Qwen2.5-Coder-7B-Instruct",
        provider="Qwen",
        params=7.6,
        license="apache-2.0",
        ctx=131072,
        caps=["text", "coding"],
        bench={"MMLU-Pro": 47.6, "LiveCodeBench": 55.3,
               "SWE-Bench-Verified": 16.3, "MATH-500": 62.2,
               "IFEval": 78.0},
        hw=[("fp16", 16.5), ("int8", 9.0), ("int4", 5.5), ("q4_k_m", 6.0)],
    ),
    dict(
        hf_id="mistralai/Mistral-Small-24B-Instruct-2501",
        name="Mistral-Small-24B-Instruct",
        provider="mistralai",
        params=24.0,
        license="apache-2.0",
        ctx=32768,
        caps=["text"],
        bench={"MMLU-Pro": 66.2, "GPQA-Diamond": 45.3, "MATH-500": 70.1,
               "LiveCodeBench": 41.7, "IFEval": 82.1},
        hw=[("fp16", 52.0), ("int8", 28.0), ("int4", 15.0), ("q4_k_m", 16.0)],
    ),
    dict(
        hf_id="Qwen/Qwen2-VL-7B-Instruct",
        name="Qwen2-VL-7B-Instruct",
        provider="Qwen",
        params=8.3,
        license="apache-2.0",
        ctx=32768,
        caps=["text", "vision"],
        bench={"MMBench": 81.0, "MMMU": 54.1, "MMLU-Pro": 47.0},
        hw=[("fp16", 19.0), ("int8", 11.0), ("int4", 7.0), ("q4_k_m", 7.5)],
    ),
    dict(
        hf_id="google/gemma-2-9b-it",
        name="Gemma-2-9B-Instruct",
        provider="google",
        params=9.2,
        license="gemma",
        ctx=8192,
        caps=["text"],
        bench={"MMLU-Pro": 52.1, "GPQA-Diamond": 32.8, "MATH-500": 55.0,
               "IFEval": 74.4},
        hw=[("fp16", 20.5), ("int8", 11.5), ("int4", 7.0), ("q4_k_m", 7.5)],
    ),
    dict(
        hf_id="meta-llama/Llama-3.3-70B-Instruct",
        name="Llama-3.3-70B-Instruct",
        provider="meta-llama",
        params=70.6,
        license="llama3.3",
        ctx=131072,
        caps=["text"],
        bench={"MMLU-Pro": 68.9, "GPQA-Diamond": 50.5, "MATH-500": 77.0,
               "AIME": 35.0, "LiveCodeBench": 54.8, "IFEval": 92.1},
        hw=[("fp16", 150.0), ("int8", 80.0), ("int4", 42.0), ("q4_k_m", 44.0)],
    ),
]


def _insert_model(conn, m: dict) -> None:
    license_ = m["license"].lower()
    from core import licenses
    whitelist = 1 if licenses.is_whitelisted(license_) else 0
    conn.execute(
        """
        INSERT INTO models (huggingface_id, name, provider, parameter_count_b,
            license_spdx, is_on_whitelist, context_window, first_seen_at,
            last_seen_at, is_active)
        VALUES (?,?,?,?,?,?,?,?,?,1)
        ON CONFLICT(huggingface_id) DO UPDATE SET
            name=excluded.name, provider=excluded.provider,
            parameter_count_b=excluded.parameter_count_b,
            license_spdx=excluded.license_spdx,
            is_on_whitelist=excluded.is_on_whitelist,
            context_window=excluded.context_window,
            last_seen_at=excluded.last_seen_at
        """,
        (m["hf_id"], m["name"], m["provider"], m["params"], license_, whitelist,
         m["ctx"], _NOW, _NOW),
    )
    mid = conn.execute(
        "SELECT id FROM models WHERE huggingface_id=?", (m["hf_id"],)
    ).fetchone()["id"]
    for c in m["caps"]:
        conn.execute(
            "INSERT OR IGNORE INTO model_capabilities (model_id, capability) VALUES (?,?)",
            (mid, c),
        )
    for metric, score in m["bench"].items():
        conn.execute(
            """
            INSERT OR IGNORE INTO model_benchmarks (model_id, metric_name, source,
                n_shot, cot, score, measured_at) VALUES (?,?,?,?,?,?,?)
            """,
            (mid, metric, "artificial_analysis", 5, 1, score, _NOW),
        )
    for q, vram in m["hw"]:
        conn.execute(
            """
            INSERT INTO model_hardware_requirements (model_id, quantization,
                min_vram_gb, recommended_vram_gb, min_ram_gb, estimator_source,
                estimated_at) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(model_id, quantization) DO UPDATE SET
                min_vram_gb=excluded.min_vram_gb,
                recommended_vram_gb=excluded.recommended_vram_gb,
                min_ram_gb=excluded.min_ram_gb,
                estimator_source=excluded.estimator_source,
                estimated_at=excluded.estimated_at
            """,
            (mid, q, vram, vram * 1.25, vram * 1.5, "formula", _NOW),
        )


def main() -> None:
    init_db()
    with get_conn() as conn:
        for m in DEMO_MODELS:
            _insert_model(conn, m)
    print(f"Seeded {len(DEMO_MODELS)} demo models.")


if __name__ == "__main__":
    main()

"""Stage 2 scoring: per-capability weighted sum + geometric mean combination.

Ref: design doc §6.2-6.3. Normalization is local to the shortlist S so that
scores reflect rank among feasible alternatives for the user's query.

Missing benchmarks propagate to score = 0 for that capability, which through
the geometric mean zeros out the combined score — the design's intended
"unknown benchmark => drop out" semantics.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from agents.agent2_scoring import capabilities as cap_mod
from agents.agent2_scoring import filters, profiles
from core.models.dto import (
    CapabilityBreakdown,
    ModelHardwareInfo,
    ModelRankRow,
    ScoreRequest,
    ScoreResponse,
)


@dataclass
class _ModelData:
    id: int
    huggingface_id: str
    name: str
    license_spdx: Optional[str]
    parameter_count_b: Optional[float]
    context_window: Optional[int]
    benchmarks: dict[str, float]                       # metric -> score
    hardware: list[tuple[str, float, Optional[float]]] # (quant, min_vram, rec_vram)
    capability_tags: set[str]
    min_vram_for_user: Optional[float]                 # at chosen quant, or min across quants
    throughput_tok_s: Optional[float]
    ttft_ms: Optional[float]


def _latest_benchmark(
    conn: sqlite3.Connection,
    model_id: int,
    metric: str,
    prefs: list[tuple[str, int, int]],
) -> Optional[float]:
    """Walk source_preferences in order; return the freshest matching row."""
    for source, n_shot, cot in prefs:
        row = conn.execute(
            """
            SELECT score FROM model_benchmarks
            WHERE model_id=? AND metric_name=? AND source=? AND n_shot=? AND cot=?
            ORDER BY measured_at DESC LIMIT 1
            """,
            (model_id, metric, source, n_shot, cot),
        ).fetchone()
        if row is not None:
            return float(row["score"])
    # fall back: any source for this metric
    row = conn.execute(
        """
        SELECT score FROM model_benchmarks
        WHERE model_id=? AND metric_name=?
        ORDER BY measured_at DESC LIMIT 1
        """,
        (model_id, metric),
    ).fetchone()
    return float(row["score"]) if row else None


def _load_model_data(
    conn: sqlite3.Connection,
    model_ids: list[int],
    criteria_names: set[str],
    prefs_by_metric: dict[str, list[tuple[str, int, int]]],
    quant: Optional[str],
    vram_gb: Optional[float],
) -> list[_ModelData]:
    out: list[_ModelData] = []
    for mid in model_ids:
        m = conn.execute(
            "SELECT id, huggingface_id, name, license_spdx, parameter_count_b, context_window "
            "FROM models WHERE id=?",
            (mid,),
        ).fetchone()
        if m is None:
            continue

        # capability tags
        tags = {
            r["capability"]
            for r in conn.execute(
                "SELECT capability FROM model_capabilities WHERE model_id=?",
                (mid,),
            )
        }

        # hardware rows
        hw_rows = conn.execute(
            "SELECT quantization, min_vram_gb, recommended_vram_gb "
            "FROM model_hardware_requirements WHERE model_id=?",
            (mid,),
        ).fetchall()
        hw = [(r["quantization"], r["min_vram_gb"], r["recommended_vram_gb"]) for r in hw_rows]

        # choose min_vram_for_user: at chosen quant if given; else min across feasible quants
        min_vram_for_user: Optional[float] = None
        if hw:
            if quant:
                for q, mv, _rv in hw:
                    if q == quant:
                        min_vram_for_user = mv
                        break
            else:
                feasible = [mv for _q, mv, _rv in hw if (vram_gb is None or mv <= vram_gb)]
                min_vram_for_user = min(feasible) if feasible else min(mv for _q, mv, _rv in hw)

        # benchmarks for criteria we care about
        bench: dict[str, float] = {}
        for metric in criteria_names:
            if metric in {"context_window", "min_vram_gb", "throughput_tok_s", "ttft_ms", "tool_calling"}:
                continue
            val = _latest_benchmark(conn, mid, metric, prefs_by_metric.get(metric, []))
            if val is not None:
                bench[metric] = val

        # throughput / ttft if present as benchmarks
        tp = _latest_benchmark(conn, mid, "throughput_tok_s", prefs_by_metric.get("throughput_tok_s", []))
        ttft = _latest_benchmark(conn, mid, "ttft_ms", prefs_by_metric.get("ttft_ms", []))

        out.append(
            _ModelData(
                id=int(m["id"]),
                huggingface_id=m["huggingface_id"],
                name=m["name"],
                license_spdx=m["license_spdx"],
                parameter_count_b=m["parameter_count_b"],
                context_window=m["context_window"],
                benchmarks=bench,
                hardware=hw,
                capability_tags=tags,
                min_vram_for_user=min_vram_for_user,
                throughput_tok_s=tp,
                ttft_ms=ttft,
            )
        )
    return out


def _raw_value(md: _ModelData, criterion: cap_mod.Criterion) -> Optional[float]:
    n = criterion.name
    if n == "context_window":
        return float(md.context_window) if md.context_window else None
    if n == "min_vram_gb":
        return md.min_vram_for_user
    if n == "throughput_tok_s":
        return md.throughput_tok_s
    if n == "ttft_ms":
        return md.ttft_ms
    if n == "tool_calling":
        return 1.0 if "agentic" in md.capability_tags or "tools" in md.capability_tags else 0.0
    return md.benchmarks.get(n)


def _normalize(
    vals: list[Optional[float]], kind: str
) -> list[float]:
    """Min-max normalize. Missing values become 0 (drop out via geo mean)."""
    present = [v for v in vals if v is not None]
    if not present:
        return [0.0 for _ in vals]
    lo, hi = min(present), max(present)
    if hi == lo:
        # all equal (incl. singleton S): neutral 1.0 for benefit, 1.0 for cost
        return [1.0 if v is not None else 0.0 for v in vals]
    out: list[float] = []
    for v in vals:
        if v is None:
            out.append(0.0)
            continue
        if kind == "benefit":
            out.append((v - lo) / (hi - lo))
        else:  # cost
            out.append((hi - v) / (hi - lo))
    return out


_MODEL_FIELD_NAMES = {
    "context_window",
    "min_vram_gb",
    "throughput_tok_s",
    "ttft_ms",
    "tool_calling",
}


def _score_capability(
    capability: str,
    profile: profiles.Profile,
    models: list[_ModelData],
) -> tuple[list[float], list[dict[str, float]]]:
    """Return (per-model score in [0,1], per-model contributions dict).

    Design §6.3: "A model with zero or missing benchmark in any selected
    capability gets score 0 and drops out." We interpret this as: if the
    model has NO benchmark data (none of the capability's actual benchmark
    criteria) for this capability, its capability score is 0. Model-field
    metrics (context_window, min_vram_gb, etc.) alone cannot carry a score.
    """
    criteria = cap_mod.CAPABILITY_CRITERIA[capability]
    crit_by_name = {c.name: c for c in criteria}
    benchmark_names = {
        c.name for c in criteria if c.name not in _MODEL_FIELD_NAMES
    }

    # raw vectors per criterion
    normed_by_name: dict[str, list[float]] = {}
    for cname in profile.criteria:
        crit = crit_by_name.get(cname)
        if crit is None:
            normed_by_name[cname] = [0.0] * len(models)
            continue
        raws = [_raw_value(m, crit) for m in models]
        normed_by_name[cname] = _normalize(raws, crit.kind)

    scores: list[float] = []
    contribs: list[dict[str, float]] = []
    for i, m in enumerate(models):
        has_any_benchmark = any(
            bm in m.benchmarks for bm in benchmark_names
        )
        contrib: dict[str, float] = {}
        if not has_any_benchmark and benchmark_names:
            # Design §6.3: no benchmark data => capability score 0
            for cname in profile.criteria:
                contrib[cname] = 0.0
            scores.append(0.0)
            contribs.append(contrib)
            continue

        total = 0.0
        for cname in profile.criteria:
            w = profile.weights.get(cname, 0.0)
            x_hat = normed_by_name[cname][i]
            c = w * x_hat
            contrib[cname] = c
            total += c
        scores.append(total)
        contribs.append(contrib)
    return scores, contribs


def _geometric_mean(per_cap_scores: list[float]) -> float:
    """Strict geometric mean; zero-anywhere zeros the product (design §6.3)."""
    if not per_cap_scores:
        return 0.0
    prod = 1.0
    for s in per_cap_scores:
        if s <= 0.0:
            return 0.0
        prod *= s
    return prod ** (1.0 / len(per_cap_scores))


def score(
    conn: sqlite3.Connection,
    req: ScoreRequest,
) -> ScoreResponse:
    # Stage 1
    sl = filters.shortlist(
        conn,
        filters.FilterInput(
            capabilities=req.capabilities,
            vram_gb=req.vram_gb,
            context_required=req.context_required,
            quantization=req.quantization,
        ),
    )
    if not sl:
        return ScoreResponse(
            shortlist_size=0,
            ranked=[],
            diagnostic="No models passed hard filters. Loosen VRAM or context constraints.",
        )

    # load profiles + resolve criteria names for data pull
    profs = {cap: profiles.load_profile(conn, cap) for cap in req.capabilities}
    criteria_names: set[str] = set()
    prefs_by_metric: dict[str, list[tuple[str, int, int]]] = {}
    for p in profs.values():
        criteria_names.update(p.criteria)
        for metric, pref in p.source_preferences.items():
            prefs_by_metric.setdefault(metric, pref)

    models = _load_model_data(conn, sl, criteria_names, prefs_by_metric, req.quantization, req.vram_gb)

    # Stage 2 per-capability
    per_cap_scores: dict[str, list[float]] = {}
    per_cap_contribs: dict[str, list[dict[str, float]]] = {}
    for cap in req.capabilities:
        s, c = _score_capability(cap, profs[cap], models)
        per_cap_scores[cap] = s
        per_cap_contribs[cap] = c

    # Combine via geometric mean
    combined: list[float] = []
    for i in range(len(models)):
        combined.append(
            _geometric_mean([per_cap_scores[cap][i] for cap in req.capabilities])
        )

    # Sort descending and persist
    order = sorted(range(len(models)), key=lambda i: combined[i], reverse=True)
    now = datetime.now(timezone.utc).isoformat()
    run_caps_json = json.dumps(req.capabilities)
    ranked_rows: list[ModelRankRow] = []

    for rank, i in enumerate(order[: req.top_n], start=1):
        md = models[i]
        per_cap_breakdown = [
            CapabilityBreakdown(
                capability=cap,
                score=per_cap_scores[cap][i],
                contributions=per_cap_contribs[cap][i],
            )
            for cap in req.capabilities
        ]
        hw_info = [
            ModelHardwareInfo(
                quantization=q, min_vram_gb=mv, recommended_vram_gb=rv
            )
            for q, mv, rv in md.hardware
        ]
        why = _why_string(req.capabilities, per_cap_breakdown, combined[i])
        ranked_rows.append(
            ModelRankRow(
                rank=rank,
                huggingface_id=md.huggingface_id,
                name=md.name,
                license_spdx=md.license_spdx,
                parameter_count_b=md.parameter_count_b,
                context_window=md.context_window,
                combined_score=combined[i],
                per_capability=per_cap_breakdown,
                hardware=hw_info,
                why=why,
            )
        )

        # Write audit row
        conn.execute(
            """
            INSERT INTO scoring_runs (
                capabilities, model_id, rank, combined_score,
                per_cap_scores, contributions,
                user_vram_gb, user_context_req, run_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                run_caps_json,
                md.id,
                rank,
                combined[i],
                json.dumps({cap: per_cap_scores[cap][i] for cap in req.capabilities}),
                json.dumps({cap: per_cap_contribs[cap][i] for cap in req.capabilities}),
                req.vram_gb,
                req.context_required,
                now,
            ),
        )
    conn.commit()

    return ScoreResponse(
        shortlist_size=len(models),
        ranked=ranked_rows,
        diagnostic=None,
    )


def _why_string(caps: list[str], breakdown: list[CapabilityBreakdown], combined: float) -> str:
    parts = []
    for b in breakdown:
        top_metric = max(b.contributions.items(), key=lambda kv: kv[1]) if b.contributions else ("-", 0.0)
        parts.append(f"{b.capability}={b.score:.2f} (top: {top_metric[0]})")
    return f"combined={combined:.3f}; " + "; ".join(parts)

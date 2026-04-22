"use client";

import { useEffect, useMemo, useState } from "react";

type CapabilityBreakdown = {
  capability: string;
  score: number;
  contributions: Record<string, number>;
};
type Hardware = { quantization: string; min_vram_gb: number; recommended_vram_gb?: number | null };
type Row = {
  rank: number;
  huggingface_id: string;
  name: string;
  license_spdx: string | null;
  parameter_count_b: number | null;
  context_window: number | null;
  combined_score: number;
  per_capability: CapabilityBreakdown[];
  hardware: Hardware[];
  why: string;
};
type Resp = { shortlist_size: number; ranked: Row[]; diagnostic: string | null };

const CAPABILITY_LABELS: Record<string, string> = {
  coding: "Coding",
  reasoning: "Reasoning",
  rag_long_context: "RAG / Long-Context",
  agentic: "Agentic",
  vision_language: "Vision-Language",
  audio_language: "Audio-Language",
  minimum_cost: "Minimum-Cost",
};

export default function Page() {
  const [caps, setCaps] = useState<string[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set(["reasoning"]));
  const [vram, setVram] = useState<string>("");
  const [ctx, setCtx] = useState<string>("");
  const [quant, setQuant] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/capabilities")
      .then((r) => r.json())
      .then((j) => setCaps(j.capabilities))
      .catch(() => setCaps(Object.keys(CAPABILITY_LABELS)));
  }, []);

  const selectedList = useMemo(() => Array.from(selected), [selected]);

  async function runScore() {
    setLoading(true);
    setErr(null);
    setData(null);
    try {
      const resp = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          capabilities: selectedList,
          vram_gb: vram ? parseFloat(vram) : null,
          context_required: ctx ? parseInt(ctx, 10) : null,
          quantization: quant || null,
          top_n: 25,
        }),
      });
      if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
      setData(await resp.json());
    } catch (e: any) {
      setErr(e.message || "request failed");
    } finally {
      setLoading(false);
    }
  }

  function toggle(c: string) {
    const next = new Set(selected);
    if (next.has(c)) next.delete(c);
    else next.add(c);
    setSelected(next);
  }

  return (
    <div className="container">
      <h1>EpiAgent — Self-hostable LLM selector</h1>
      <p className="muted">
        Pick the capabilities you care about and (optionally) your compute budget. Results are ranked
        locally across feasible models using BWM-derived weights and a geometric-mean combination.
      </p>

      <div className="card">
        <h2>Capabilities</h2>
        <div className="chip-row">
          {caps.map((c) => (
            <div
              key={c}
              className={`chip ${selected.has(c) ? "active" : ""}`}
              onClick={() => toggle(c)}
            >
              {CAPABILITY_LABELS[c] || c}
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Compute (optional)</h2>
        <div className="row">
          <div className="field">
            <label>Available VRAM (GB)</label>
            <input
              value={vram}
              onChange={(e) => setVram(e.target.value)}
              placeholder="e.g. 24"
              type="number"
            />
          </div>
          <div className="field">
            <label>Required context (tokens)</label>
            <input
              value={ctx}
              onChange={(e) => setCtx(e.target.value)}
              placeholder="e.g. 32768"
              type="number"
            />
          </div>
          <div className="field">
            <label>Quantization</label>
            <select value={quant} onChange={(e) => setQuant(e.target.value)}>
              <option value="">any</option>
              <option value="fp16">fp16</option>
              <option value="int8">int8</option>
              <option value="int4">int4</option>
              <option value="q4_k_m">q4_k_m</option>
            </select>
          </div>
        </div>
      </div>

      <button
        className="primary"
        onClick={runScore}
        disabled={loading || selectedList.length === 0}
      >
        {loading ? "Scoring…" : "Rank models"}
      </button>

      {err && <div className="diag" style={{ marginTop: 16 }}>{err}</div>}

      {data && (
        <div className="card" style={{ marginTop: 16 }}>
          <h2>
            Results — shortlist of {data.shortlist_size} model
            {data.shortlist_size === 1 ? "" : "s"}
          </h2>
          {data.diagnostic && <div className="diag">{data.diagnostic}</div>}
          {data.ranked.length > 0 && (
            <table className="results">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Model</th>
                  <th>Overall</th>
                  {selectedList.map((c) => (
                    <th key={c}>{CAPABILITY_LABELS[c] || c}</th>
                  ))}
                  <th>Min VRAM</th>
                  <th>License</th>
                </tr>
              </thead>
              <tbody>
                {data.ranked.map((r) => {
                  const minVram = Math.min(...r.hardware.map((h) => h.min_vram_gb));
                  return (
                    <tr key={r.huggingface_id}>
                      <td>{r.rank}</td>
                      <td>
                        <div>{r.name}</div>
                        <div className="muted">{r.huggingface_id}</div>
                      </td>
                      <td className="score">
                        <span className="tooltip">
                          {r.combined_score.toFixed(3)}
                          <span className="tip">{r.why}</span>
                        </span>
                      </td>
                      {selectedList.map((c) => {
                        const b = r.per_capability.find((x) => x.capability === c);
                        return (
                          <td key={c} className="score">
                            <span className="tooltip">
                              {b ? b.score.toFixed(3) : "—"}
                              <span className="tip">
                                {b
                                  ? Object.entries(b.contributions)
                                      .sort((a, z) => z[1] - a[1])
                                      .map(
                                        ([metric, contrib]) =>
                                          `${metric}: ${(contrib as number).toFixed(3)}`,
                                      )
                                      .join("\n")
                                  : "no data"}
                              </span>
                            </span>
                          </td>
                        );
                      })}
                      <td className="score">
                        {isFinite(minVram) ? `${minVram.toFixed(1)} GB` : "—"}
                      </td>
                      <td>
                        <span className="badge">{r.license_spdx || "?"}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

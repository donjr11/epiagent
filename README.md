# EpiAgent

Dashboard that helps Epineon's clients select the best self-hostable LLM for
their needs. Users pick the capabilities they care about (Coding, Reasoning,
Vision, etc.) and, optionally, their available compute (VRAM, context window,
quantization), and receive a ranked list of open-weight LLMs with per-criterion
score breakdowns and hardware requirements.

See the design doc (`EpiAgent Design Document`, Epineon AI, v0.2 Draft) for
full specs. This is the v1 reference implementation.

**What it is:** a filter-and-rank engine over a curated database of
open-weight LLMs.
**What it isn't:** an agentic chatbot, an auto-deployer.

## Layout

```
agents/
  agent1_discovery/        Ingestion Service — daily pull from HF Hub, AA, HF Leaderboard
  agent2_scoring/          Scoring Service — FastAPI, BWM solver, hard filters, geometric mean
  agent3_notifications/    Notification Service — daily digest via SMTP
  agent4_deployment/       (out of scope for v1)
core/
  database/                SQLite schema + connection helpers
  models/                  Pydantic DTOs for the API
  config.py                env-driven config
  licenses.py              whitelist (Section 5)
orchestrator/cron.py       APScheduler entry point: nightly ingestion + daily digest
dashboard/                 Next.js 14 single-page dashboard
scripts/seed_demo.py       Seed a small demo dataset
tests/                     pytest suite (BWM, scorer, ingestion)
```

## Three services

Per the design doc:

1. **Ingestion** — nightly job that pulls model specs, benchmarks, and
   hardware requirements from HuggingFace Hub, Artificial Analysis, and the
   HF Open LLM Leaderboard v2. Stores an append-only benchmark history.
2. **Scoring** — on-demand API that applies hard filters (license, VRAM,
   capability, context) and ranks the shortlist using BWM-derived weights and
   a weighted-sum score. Multi-capability selection is combined via geometric
   mean across per-capability scores.
3. **Notifications** — daily email digest of new models, significant rank
   changes, and ingestion failures.

## Quick start (local)

```
# Python side
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # fill in secrets as needed

# Seed a small demo dataset so the dashboard has something to show
python -m scripts.seed_demo

# Run the scoring API on :8000
uvicorn agents.agent2_scoring.api:app --reload --port 8000

# In a separate shell, run the dashboard on :3000
cd dashboard && npm install && npm run dev
```

Open http://localhost:3000.

## Running ingestion manually

```
python -m agents.agent1_discovery.ingest
# or via the API:
curl -X POST http://localhost:8000/ingest/trigger
```

## Running the cron

```
python -m orchestrator.cron
```

Schedules:
- `02:00 UTC` — nightly ingestion
- `07:00 UTC` — daily digest

## BWM weight elicitation

One session per capability with the PO:

```
python -m agents.agent2_scoring.bwm_cli --capability coding
```

The CLI prompts for (best, worst, A_B, A_W) on the Saaty-like 1-9 scale, solves
the nonlinear program, reports the consistency ratio, and persists the
solution to `scoring_profiles` iff `CR ≤ 0.10`.

Until BWM runs are complete, scoring uses uniform weights per capability
(labelled `version=0` in the profile table).

## Tests

```
pytest
```

Covers:
- BWM solver (weights sum to 1, best-worst ordering, consistency flagging)
- Scorer property tests (hard filters, VRAM filter, monotonicity, geometric-mean
  zero-on-missing behavior, empty-shortlist diagnostic)
- Ingestion integration against mocked HF/AA iterators + change-log auditing

## Open items (blocking PO confirmation)

See design doc §11:

1. License whitelist as proposed in `core/licenses.py` — needs PO sign-off.
2. Repo ownership (personal vs Epineon org) — affects where this code lives.
3. Final capability list — currently seven, matches §7.
4. PO availability for ~1h BWM session per capability (7 total).
5. Artificial Analysis ToS verification — ingestion gracefully degrades to the
   HF Leaderboard fallback if the AA key is missing, but the ToS question must
   still be answered before v1 demo.

## Operational cost target

< $25 / month (HF Pro $9, SMTP $0-5, VPS $5-10, AA $0 TBC). See design doc §10.2.

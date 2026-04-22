"""Central configuration. Reads from env; safe defaults for local dev."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Database
DB_PATH = Path(os.getenv("EPIAGENT_DB_PATH", ROOT / "epiagent.db"))

# External APIs
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
ARTIFICIAL_ANALYSIS_KEY = os.getenv("ARTIFICIAL_ANALYSIS_KEY", "")
ARTIFICIAL_ANALYSIS_BASE = os.getenv(
    "ARTIFICIAL_ANALYSIS_BASE", "https://artificialanalysis.ai/api/v2"
)
HF_LEADERBOARD_DATASET = os.getenv(
    "HF_LEADERBOARD_DATASET", "open-llm-leaderboard/contents"
)

# Ingestion policy
MIN_DOWNLOADS = int(os.getenv("EPIAGENT_MIN_DOWNLOADS", "1000"))
MIN_PARAMS_B = float(os.getenv("EPIAGENT_MIN_PARAMS_B", "0.5"))
MAX_PARAMS_B = float(os.getenv("EPIAGENT_MAX_PARAMS_B", "500"))
MODIFIED_WITHIN_DAYS = int(os.getenv("EPIAGENT_MODIFIED_WITHIN_DAYS", "90"))

# SMTP
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "25"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "epiagent@localhost")
ADMIN_EMAILS = [e.strip() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()]

# Notification thresholds
RANK_CHANGE_THRESHOLD = int(os.getenv("EPIAGENT_RANK_CHANGE_THRESHOLD", "3"))

# API
SCORING_HOST = os.getenv("EPIAGENT_HOST", "127.0.0.1")
SCORING_PORT = int(os.getenv("EPIAGENT_PORT", "8000"))

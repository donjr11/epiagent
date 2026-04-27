"""Shared pytest fixtures."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated SQLite file per test, wired through core.config.

    Seeds the 7 strawman scoring_profiles so the scorer can run; without these
    rows it (correctly) raises ProfileMissingError per design §6.4.
    """
    db_path = tmp_path / "epiagent_test.db"
    monkeypatch.setenv("EPIAGENT_DB_PATH", str(db_path))
    # Force reload of config / connection modules so they pick up the env var.
    from importlib import reload
    from core import config as cfg
    from core.database import connection as conn_mod
    reload(cfg)
    reload(conn_mod)
    conn_mod.init_db()

    from agents.agent2_scoring import bwm, profiles
    from scripts.seed_strawman_profiles import STRAWMAN_INPUTS

    with conn_mod.get_conn() as conn:
        for cap, inp in STRAWMAN_INPUTS.items():
            sol = bwm.solve_bwm(
                inp["criteria"], inp["best"], inp["worst"], inp["a_b"], inp["a_w"]
            )
            profiles.save_profile_from_bwm(
                conn, cap, sol, created_by="adam_strawman"
            )

    return db_path

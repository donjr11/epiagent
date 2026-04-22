"""Shared pytest fixtures."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated SQLite file per test, wired through core.config."""
    db_path = tmp_path / "epiagent_test.db"
    monkeypatch.setenv("EPIAGENT_DB_PATH", str(db_path))
    # Force reload of config / connection modules so they pick up the env var.
    from importlib import reload
    from core import config as cfg
    from core.database import connection as conn_mod
    reload(cfg)
    reload(conn_mod)
    conn_mod.init_db()
    return db_path

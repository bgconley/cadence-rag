from __future__ import annotations

import importlib
import os
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

DEFAULT_BASE_URL = "postgresql+psycopg://rag:rag@localhost:5432/rag"
DEFAULT_SCHEMA = "rag_test"


def _admin_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.split("postgresql+psycopg://", 1)[1]
    return url.split("?", 1)[0]


@pytest.fixture(scope="session")
def test_database_url() -> str:
    base_url = os.getenv("TEST_DATABASE_URL", DEFAULT_BASE_URL)
    schema = os.getenv("TEST_SCHEMA", DEFAULT_SCHEMA)
    admin_url = _admin_url(base_url)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    return f"{base_url}?options=-csearch_path={schema},public"


@pytest.fixture(scope="session")
def apply_migrations(test_database_url: str) -> None:
    os.environ["DATABASE_URL"] = test_database_url
    os.environ["SKIP_VERSION_CHECK"] = "true"
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")


@pytest.fixture()
def client(test_database_url: str, apply_migrations: None) -> TestClient:
    os.environ["DATABASE_URL"] = test_database_url
    os.environ["SKIP_VERSION_CHECK"] = "true"

    import app.config as config_module
    import app.db as db_module
    import app.ingest as ingest_module
    import app.retrieve as retrieve_module
    import app.main as main_module

    importlib.reload(config_module)
    importlib.reload(db_module)
    importlib.reload(ingest_module)
    importlib.reload(retrieve_module)
    importlib.reload(main_module)

    return TestClient(main_module.app)

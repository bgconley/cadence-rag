from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

DEFAULT_BASE_URL = "postgresql+psycopg://rag:rag@10.25.0.50:5432/rag"
DEFAULT_SCHEMA_PREFIX = "rag_test"
SAFE_TEST_SCHEMA_RE = re.compile(r"^rag_test(?:_[a-z0-9]+)?$")


def _admin_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.split("postgresql+psycopg://", 1)[1]
    return url.split("?", 1)[0]


def _resolve_test_schema() -> str:
    configured_schema = os.getenv("TEST_SCHEMA")
    schema = configured_schema or f"{DEFAULT_SCHEMA_PREFIX}_{uuid4().hex[:8]}"
    if not SAFE_TEST_SCHEMA_RE.fullmatch(schema):
        raise ValueError(
            "TEST_SCHEMA must match 'rag_test' or 'rag_test_<suffix>' "
            "to prevent accidental destructive operations."
        )
    return schema


@pytest.fixture(scope="session")
def test_database_url() -> Iterator[str]:
    base_url = os.getenv("TEST_DATABASE_URL", DEFAULT_BASE_URL)
    schema = _resolve_test_schema()
    admin_url = _admin_url(base_url)
    keep_schema = os.getenv("TEST_SCHEMA_KEEP", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(schema)
            )
        )
        conn.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )

    separator = "&" if "?" in base_url else "?"
    db_url = f"{base_url}{separator}options=-csearch_path={schema},public"
    yield db_url

    if keep_schema:
        return

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(schema)
            )
        )


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

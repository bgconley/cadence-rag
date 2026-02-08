from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Iterator
from urllib.parse import quote
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
def test_schema_name() -> str:
    return _resolve_test_schema()


@pytest.fixture(scope="session")
def test_database_url(test_schema_name: str) -> Iterator[str]:
    base_url = os.getenv("TEST_DATABASE_URL", DEFAULT_BASE_URL)
    schema = test_schema_name
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
    search_path_option = quote(f"-c search_path={schema},public")
    db_url = f"{base_url}{separator}options={search_path_option}"
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
def apply_migrations(test_database_url: str, test_schema_name: str) -> None:
    os.environ["DATABASE_URL"] = test_database_url
    os.environ["SKIP_VERSION_CHECK"] = "true"
    os.environ["EMBEDDINGS_BASE_URL"] = ""
    os.environ["ALEMBIC_VERSION_TABLE_SCHEMA"] = test_schema_name
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")


@pytest.fixture()
def client(test_database_url: str, apply_migrations: None) -> TestClient:
    os.environ["DATABASE_URL"] = test_database_url
    os.environ["SKIP_VERSION_CHECK"] = "true"
    os.environ["EMBEDDINGS_BASE_URL"] = ""
    os.environ["EMBEDDINGS_MODEL_ID"] = "Qwen/Qwen3-Embedding-4B"
    os.environ["EMBEDDINGS_DIM"] = "1024"
    os.environ["EMBEDDINGS_TIMEOUT_S"] = "30"
    os.environ["EMBEDDINGS_BATCH_SIZE"] = "8"
    os.environ["EMBEDDINGS_EXACT_SCAN_THRESHOLD"] = "2000"
    os.environ["EMBEDDINGS_HNSW_EF_SEARCH"] = "80"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["INGEST_QUEUE_NAME"] = "ingest"
    os.environ["INGEST_ROOT_DIR"] = "/tmp/personal_rag_ingest_tests"
    os.environ["INGEST_POLL_SECONDS"] = "1"
    os.environ["INGEST_JOB_MAX_ATTEMPTS"] = "3"
    os.environ["INGEST_JOB_RETRY_BACKOFF_S"] = "1"
    os.environ["LOG_LEVEL"] = "INFO"

    import app.config as config_module
    import app.db as db_module
    import app.browse as browse_module
    import app.ingest_fs as ingest_fs_module
    import app.ingest as ingest_module
    import app.retrieve as retrieve_module
    import app.main as main_module

    importlib.reload(config_module)
    importlib.reload(db_module)
    importlib.reload(browse_module)
    importlib.reload(ingest_fs_module)
    importlib.reload(ingest_module)
    importlib.reload(retrieve_module)
    importlib.reload(main_module)

    return TestClient(main_module.app)

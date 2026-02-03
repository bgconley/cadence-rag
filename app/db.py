import re
from typing import Dict, Tuple

from sqlalchemy import create_engine, text

from .config import settings


_VERSION_RE = re.compile(r"^(\d+\.\d+)")

engine = create_engine(settings.database_url, pool_pre_ping=True)


def _parse_pg_version(raw: str) -> str:
    match = _VERSION_RE.match(raw.strip())
    return match.group(1) if match else raw.strip()


def fetch_db_info() -> Dict[str, object]:
    with engine.connect() as conn:
        server_version_raw = conn.execute(text("SHOW server_version")).scalar()
        ext_rows = conn.execute(
            text(
                "SELECT extname, extversion "
                "FROM pg_extension "
                "WHERE extname IN ('pg_search','vector')"
            )
        ).fetchall()

    ext_versions = {row[0]: row[1] for row in ext_rows}
    return {
        "server_version_raw": server_version_raw,
        "server_version": _parse_pg_version(server_version_raw),
        "extensions": ext_versions,
    }


def validate_versions() -> Tuple[bool, str]:
    info = fetch_db_info()
    server_version = info["server_version"]
    extensions = info["extensions"]

    if server_version != settings.expected_pg_version:
        return False, (
            f"Postgres version mismatch: expected {settings.expected_pg_version}, "
            f"got {server_version}"
        )

    pg_search_version = extensions.get("pg_search")
    if pg_search_version != settings.expected_pg_search_version:
        return False, (
            f"pg_search version mismatch: expected {settings.expected_pg_search_version}, "
            f"got {pg_search_version}"
        )

    pgvector_version = extensions.get("vector")
    if pgvector_version != settings.expected_pgvector_version:
        return False, (
            f"pgvector version mismatch: expected {settings.expected_pgvector_version}, "
            f"got {pgvector_version}"
        )

    return True, "ok"

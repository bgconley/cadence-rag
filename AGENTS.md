# AGENTS.md

## Canonical docs
- `APP_SPEC.md`: product/system spec + contracts (schema, API shapes, evidence-pack format).
- `IMPLEMENTATION_PLAN.md`: engineering principles/guardrails + rationale.
- `PHASED_PLAN.md`: phase sequencing + deliverables/acceptance criteria (**follow this ordering**).
- Keep docs in sync: if a phase introduces/changes a contract or schema, update `APP_SPEC.md` and/or `IMPLEMENTATION_PLAN.md` in the same change set.

## Doc precedence (when they conflict)
- Contract/schema/API behavior → follow `APP_SPEC.md`.
- Phase ordering/scope → follow `PHASED_PLAN.md`.
- Non-negotiables (deterministic retrieval, citation gating, version pinning) → follow `IMPLEMENTATION_PLAN.md`.

## Core decisions (do not drift)
- Canonical DB: **Postgres 18** via **ParadeDB** (extensions: `pg_search`, `pgvector`, `pgcrypto`, optional `pg_trgm`).
- Retrieval is **deterministic and server-orchestrated**; avoid “LLM decides tools” loops.
- Evidence-pack answering with **strict citation gating** (no uncited sentences).
- Embedding dimension is standardized to **1024**.

## Repo conventions (when implemented)
- API: FastAPI; background work via a worker (Celery/RQ/Arq); schema changes via migrations (e.g., Alembic).
- Prefer small, testable changes; avoid introducing new services unless required for throughput.

## Privacy & logging
- Treat transcripts/artifacts as sensitive.
- Do not log transcript text by default; logs should use IDs and timings only.
- Test fixtures should avoid real PII.

## Dev workflow (when files exist)
- **All migrations/tests must target the ParadeDB instance on P620** (`10.25.0.50`), not local Mac Postgres.
- Required env before migration/test commands:
  - `export DATABASE_URL='postgresql+psycopg://rag:rag@10.25.0.50:5432/rag'`
  - `export TEST_DATABASE_URL='postgresql+psycopg://rag:rag@10.25.0.50:5432/rag'`
  - Optional: `export TEST_SCHEMA='rag_test_<suffix>'` (must start with `rag_test`; default is auto-generated per test run).
  - Optional: `export TEST_SCHEMA_KEEP=true` to retain the test schema for debugging.
- Run migrations: `uv run alembic upgrade head`
- Run API (local process, remote DB): `uv run uvicorn app.main:app --reload --port 8001`
- Run tests (remote DB): `uv run pytest`

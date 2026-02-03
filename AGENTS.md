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
- Start DB: `docker compose up -d db`
- Run migrations: `uv run alembic upgrade head`
- Run API (local): `uv run uvicorn app.main:app --reload --port 8001`
- Run tests: `uv run pytest`

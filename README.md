# Personal RAG — Foundation

## Quickstart (Phase 0)

### 1) Start core infrastructure
```bash
docker compose up -d db redis
```

If you previously ran with an older Postgres data layout, reset the volume:
```bash
docker compose down -v
docker compose up -d db redis
```

### 2) Create a Python env with uv
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3) Run migrations
```bash
uv run alembic upgrade head
```

### 4) Run the API locally
```bash
uv run uvicorn app.main:app --reload --port 8001
```

API endpoints:
- `GET /health`
- `GET /diagnostics`
- `POST /ingest/call`
- `POST /ingest/transcript`
- `POST /ingest/analysis`
- `GET /ingest/jobs`
- `GET /ingest/jobs/{ingest_job_id}`
- `GET /calls`
- `GET /calls/{call_id}`
- `GET /chunks/{chunk_id}`
- `POST /expand`
- `POST /retrieve`

### 4b) Run background ingest services (for drop-folder ingest)
```bash
docker compose up -d ingest_scanner ingest_worker
```

### 5) Enable dense embeddings lane (Phase 3)
Set embedding service env vars in `.env`:
```bash
EMBEDDINGS_BASE_URL=http://<p620-host-or-ip>:8101
EMBEDDINGS_MODEL_ID=Qwen/Qwen3-Embedding-4B
EMBEDDINGS_DIM=1024
EMBEDDINGS_TIMEOUT_S=180
EMBEDDINGS_BATCH_SIZE=32
EMBEDDINGS_EXACT_SCAN_THRESHOLD=2000
EMBEDDINGS_HNSW_EF_SEARCH=80
```

Backfill existing rows with null embeddings:
```bash
uv run python -m app.scripts.embed_backfill
```

Backfill `tech_tokens` after token-rule updates (recommended after lexicon changes):
```bash
uv run python -m app.scripts.tech_tokens_backfill --dry-run
uv run python -m app.scripts.tech_tokens_backfill
```

### 6) Filesystem ingest queue (drop-folder workflow)
Start queue services (if not already running):
```bash
docker compose up -d redis ingest_scanner ingest_worker
```

Drop a bundle:
```text
ingest/
  inbox/
    demo-call-001/
      manifest.json
      transcript.json
      analysis/summary.md
      analysis/action_items.md
      _READY
```

`manifest.json` example:
```json
{
  "bundle_id": "demo-call-001",
  "call_ref": {
    "external_source": "manual",
    "external_id": "demo-call-001",
    "started_at": "2026-02-07T18:00:00Z",
    "title": "Demo call"
  },
  "transcript": {
    "path": "transcript.json",
    "format": "json_turns"
  },
  "analysis": [
    { "kind": "summary", "path": "analysis/summary.md" },
    { "kind": "action_items", "path": "analysis/action_items.md" }
  ]
}
```

Check job status:
```bash
curl -s http://localhost:8001/ingest/jobs | jq
```

## Run modes (important)

### First run (new environment / fresh clone)
1. `docker compose up -d db redis`
2. `uv venv && source .venv/bin/activate && uv pip install -e .`
3. `uv run alembic upgrade head`
4. Start API and ingest services:
   - `docker compose up -d api ingest_scanner ingest_worker`

### Normal daily run (no schema changes)
1. `docker compose up -d db redis api ingest_scanner ingest_worker`
2. Skip migrations unless new migration files were added.

### After pulling code with new migrations
1. `docker compose up -d db redis`
2. `uv run alembic upgrade head`
3. `docker compose up -d api ingest_scanner ingest_worker`

### If you do not need drop-folder ingest
You can run only DB + API:
```bash
docker compose up -d db api
```

## Notes
- `DATABASE_URL` and version expectations live in `.env` (see `.env.example`).
- The app fails fast on version mismatches unless `SKIP_VERSION_CHECK=true`.

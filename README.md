# Personal RAG — Foundation

## Quickstart (Phase 0)

### 1) Start Postgres (ParadeDB)
```bash
docker compose up -d db
```

If you previously ran with an older Postgres data layout, reset the volume:
```bash
docker compose down -v
docker compose up -d db
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
- `GET /calls`
- `GET /calls/{call_id}`
- `GET /chunks/{chunk_id}`
- `POST /expand`
- `POST /retrieve`

## Notes
- `DATABASE_URL` and version expectations live in `.env` (see `.env.example`).
- The app fails fast on version mismatches unless `SKIP_VERSION_CHECK=true`.

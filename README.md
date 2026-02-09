# Cadence RAG — Foundation

## Quickstart (Phase 0)

### Stack naming (`cadence-rag`)
`docker-compose.yml` now sets compose project name to `cadence-rag` and fixed container names:
- `cadence_rag_db`
- `cadence_rag_redis`
- `cadence_rag_api`
- `cadence_rag_ingest_worker`
- `cadence_rag_ingest_scanner`

If you are migrating from older `personal_rag_*` containers on a host, run this once:
```bash
# 1) Stop old containers if present
docker rm -f \
  personal_rag_ingest_scanner \
  personal_rag_ingest_worker \
  personal_rag_api \
  personal_rag_redis \
  personal_rag_db 2>/dev/null || true

# 2) Migrate DB volume data old -> new
docker volume rm cadence_rag_db_data 2>/dev/null || true
docker volume create cadence_rag_db_data
docker run --rm \
  -v personal-rag_db_data:/from \
  -v cadence_rag_db_data:/to \
  alpine sh -lc 'cd /from && cp -a . /to/'

# 3) Start renamed stack
docker compose up -d db redis api ingest_scanner ingest_worker
```

Verify:
```bash
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep cadence_rag_
```

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
INGEST_AUTO_EMBED_ON_SUCCESS=true
INGEST_AUTO_EMBED_FAIL_ON_ERROR=false
```

Notes:
- New ingest jobs auto-embed `chunks` and `artifact_chunks` for that call when `INGEST_AUTO_EMBED_ON_SUCCESS=true` and `EMBEDDINGS_BASE_URL` is set.
- `EMBEDDINGS_BATCH_SIZE` should respect your embedding endpoint limit (for example `8` on your current Triton model). The backfill script adaptively shrinks batch size when provider errors indicate an upper bound.
- `INGEST_AUTO_EMBED_FAIL_ON_ERROR=false` keeps ingest fail-open if embedding service is unavailable; set to `true` for strict fail-closed behavior.

Backfill existing rows with null embeddings (historical data / catch-up):
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

No-manifest quick drop (recommended):
```text
ingest/
  inbox/
    starcluster-call-20251125/
      Starcluster call 20251125 maor.md
      analysis/
        action_items.csv
      _READY
```

When `INGEST_AUTO_MANIFEST=true` (default), scanner auto-generates `manifest.json` from files.
It infers transcript + analysis formats and call reference metadata deterministically.

Single-file quick drop (no folder, no manifest):
```text
ingest/
  inbox/
    Starcluster call 20251125 maor.md
```

Scanner will auto-wrap the file into an internal bundle in `processing/`, generate a manifest, and queue it.
To avoid ingesting files while they are still being copied, scanner waits `INGEST_SINGLE_FILE_MIN_AGE_S` seconds (default `5`).

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
    "format": "auto"
  },
  "analysis": [
    { "kind": "summary", "path": "analysis/summary.md", "format": "markdown" },
    { "kind": "action_items", "path": "analysis/action_items.csv", "format": "csv" }
  ]
}
```

Filesystem ingest format notes:
- Transcript `format` supports `json_turns` (strict), `markdown_turns` (speaker/timestamp Markdown transcripts), and `auto` (adapter normalization).
- Analysis `format` supports `auto`, `text`, `markdown`, `csv`, `tsv`, `json`, `html`, `docx`, and `pdf`.
- With `auto`, transcript/analysis files are normalized by extension/content so table-heavy exports can still be ingested.
- PDF extraction uses native text extraction first. Optional OCR fallback can be enabled for scanned PDFs when extracted text quality is low.
- Auto-manifest mode can be disabled with `INGEST_AUTO_MANIFEST=false` (then `manifest.json` is required).

Optional OCR fallback for scanned PDFs:
```bash
# Docker mode: rebuild images so worker/scanner have OCR dependencies
docker compose build api ingest_worker ingest_scanner

# Local uv mode (no containers): install OCR tools on host once
sudo apt-get update
sudo apt-get install -y ocrmypdf tesseract-ocr ghostscript
```

`.env` controls:
```bash
ANALYSIS_PDF_OCR_ENABLED=true
ANALYSIS_PDF_OCR_COMMAND=ocrmypdf
ANALYSIS_PDF_OCR_LANGUAGES=eng
ANALYSIS_PDF_OCR_MIN_CHARS=400
ANALYSIS_PDF_OCR_MIN_ALPHA_RATIO=0.55
ANALYSIS_PDF_OCR_MAX_PAGES=150
ANALYSIS_PDF_OCR_TIMEOUT_S=600
ANALYSIS_PDF_OCR_FORCE=false
```

Check job status:
```bash
curl -s http://localhost:8001/ingest/jobs | jq
```

Queue retry/backoff is controlled via env:
```bash
INGEST_JOB_MAX_ATTEMPTS=3
INGEST_JOB_RETRY_BACKOFF_S=10
```

Transcript ingest idempotency:
- Re-ingesting the same normalized transcript payload (same call + same chunking options) is a no-op.
- API returns `utterances_ingested=0` and `chunks_created=0` for duplicate payloads.

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
- API responses include `X-Request-ID` for correlation with logs.
- Evaluation commands are documented in `eval/README.md`, including regression gating.

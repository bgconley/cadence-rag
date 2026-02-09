# Cadence RAG — Phased Implementation Plan (Forward Roadmap)
**Status:** Canonical  \
**Last updated:** 2026-02-09

## Summary (where we are now)
**Already implemented and tested:**
- Postgres 18.1 via ParadeDB (`pg_search` 0.21.5, `pgvector` 0.8.1) with fail-fast version checks.
- Alembic migrations: initial schema + `calls.external_id/external_source` + BM25 indexes (including ngram aliases).
- API endpoints:
  - `GET /health`, `GET /diagnostics`
  - `POST /ingest/call`
  - `POST /ingest/transcript` (currently `json_turns` only)
  - `POST /ingest/analysis` (supports analysis-only ingest)
  - `GET /ingest/jobs`, `GET /ingest/jobs/{ingest_job_id}`
  - `GET /calls`, `GET /calls/{call_id}`, `GET /chunks/{chunk_id}`
  - `POST /expand`
  - `POST /retrieve` (BM25 + tech-token lanes; RRF fusion; budgeted evidence-pack output; `debug` and `ids_only` modes)
- Analysis retrieval now uses `artifact_chunks` (chunk-level), including `A-<artifact_chunk_id>` evidence IDs and `artifact_chunk:<id>` ids-only responses.
- Filesystem ingest queue baseline:
  - drop-folder contract (`ingest/inbox/<bundle_id>/manifest.json` + `_READY`)
  - scanner validates and enqueues to Redis
  - RQ worker processes jobs and tracks status in `ingest_jobs`/`ingest_job_files`
  - bounded retry/backoff for transient failures before terminal `failed`
- Unit + integration tests (Pytest) covering chunking, tech-token extraction, ingest + retrieve roundtrip, browse/expand, ids-only retrieval, and budget enforcement.
- FastAPI startup migrated to lifespan (no deprecation warnings).
- Dense embeddings lane is implemented in `/retrieve`:
  - embedding client via `EMBEDDINGS_*` settings
  - planner path for exact vs ANN vector queries
  - configurable filtered ANN behavior (`EMBEDDINGS_HNSW_EF_SEARCH`)
- Embedding operations hardening is implemented:
  - `embed_backfill` supports adaptive batch-size downshift when provider max-batch limits are hit
  - ingest worker can auto-embed newly ingested call rows (`INGEST_AUTO_EMBED_ON_SUCCESS`)
  - auto-embed failure behavior is configurable (`INGEST_AUTO_EMBED_FAIL_ON_ERROR`)

**Not implemented yet (next):**
- GPU reranking.
- `/answer` with citation gating and LLM gateway.
- Entity extraction + entity filters + faceting.
- Evaluation harness wired to the live `/retrieve` endpoint (currently metric script only).

---

## Global decisions (locked)
- **Datastore:** Postgres 18.1 via ParadeDB image pinned (tag + digest).
- **Search lanes:** BM25 (`pg_search`) + exact-token lane (`tech_tokens`) + dense vectors (`pgvector`) when added.
- **Model serving:** **HTTP services** (TEI/vLLM/etc) called from API via env vars; API must degrade gracefully when services are absent.
- **LLM integration:** implement **both local + remote** via an **OpenAI-compatible gateway** (local servers like LM Studio/Ollama OpenAI-compat; and remote providers).
- **Plan doc location:** this file (`PHASED_PLAN.md`) is the phase sequencing source of truth.

---

## Phase 2A — Finish Phase 1 ergonomics (browse + provenance)
**Goal:** make the current system usable without SQL access; ensure provenance is explorable.

### Deliverables
1) **Add `POST /ingest/call`**
   - Purpose: explicitly create/update a call record (metadata-only).
   - Input: `CallRef` (same as ingest endpoints) and optional metadata fields.
   - Output: `{ call_id, created:boolean }`.
   - Behavior: uses the same call resolution rules as ingest.

2) **Add browse endpoints**
   - `GET /calls` (cursor pagination; filters: date range, tags, external_id/source)
   - `GET /calls/{call_id}` (call metadata + artifact list + counts)
   - `GET /chunks/{chunk_id}` (chunk text + call_id + timestamps + speaker)

3) **Add `POST /expand`**
   - Input: `{ evidence_id, window_ms?, max_chars }` (compatible with MCP intent).
   - Behavior:
     - For chunk evidence: use `chunk_utterances` to fetch adjacent utterances deterministically (by ordinal), then optionally apply `window_ms`.
     - For analysis evidence: return bounded excerpt from `artifact_chunks.content` (or from the parent artifact using `start_char/end_char` when available).

4) **DB idempotency hardening**
   - Add a unique index on `calls(source_uri, source_hash)` where both are non-null.
   - Update call resolution logic to rely on the index guarantee (still return 409 if duplicates exist from prior data).

5) **Start recording ingestion configs**
   - On ingest transcript and ingest analysis: insert an `ingestion_runs` row capturing:
     - `pipeline_version` (currently `"v2"`; bump whenever chunking/token/NER rules change)
     - `chunking_config` (actual options used)
     - `embedding_config` (even if embeddings are disabled: `{enabled:false, model_id:null, dim:1024}`)
     - `ner_config` (even if disabled)

### Tests
- Integration: create call via `POST /ingest/call`, then ingest transcript using only `external_id/source`, verify same call_id.
- Integration: `/expand` returns utterance-bounded output and respects `max_chars`.
- Unit: call_ref resolution precedence rules, 409 ambiguity cases.

### Acceptance criteria
- You can ingest transcript+analysis, then browse call and expand evidence without touching SQL.
- Ingestion runs are recorded for every ingest request.

---

## Phase 2B — Retrieval contract hardening (pre-dense baseline)
**Goal:** make `/retrieve` “production-shaped” for downstream `/answer` and evaluation.

### Deliverables
1) **Stabilize the evidence-pack schema**
   - Ensure `/retrieve` output matches `APP_SPEC.md` contract:
     - stable `query_id`
     - explicit `notes.retrieval` config snapshot
     - consistent evidence IDs (`A-<artifact_chunk_id>`, `Q-<chunk_id>`)

2) **Add optional debug/trace mode**
   - Request field: `debug: bool` (default false)
   - When true: include lane-specific topK ranks/scores for reproducibility (not transcript text beyond snippets).

3) **Support “return IDs only” mode for eval**
   - Request field: `return_style: "evidence_pack_json" | "ids_only"`
   - `ids_only` returns `{ query_id, retrieved_ids:[...] }` using `chunk:<id>` / `artifact_chunk:<id>`.

4) **BM25 ngram robustness (ASR noise)**
   - Add a new migration that upgrades BM25 indexes to include an ngram-tokenized alias for:
     - `chunks.text`
     - `artifact_chunks.content` (preferred; retrieval units)
     - `analysis_artifacts.content` (optional coarse fallback)
   - Keep existing trigram index as fallback; do not worry about index bloat.

### Tests
- Integration: ids-only retrieval produces stable `retrieved_ids`.
- Integration: BM25 still returns exact matches for `ECONNRESET`, `ABC-123`, etc.
- Unit: budget enforcement (max items, max chars) is strict.

### Acceptance criteria
- `/retrieve` output is stable enough to be a contract for `/answer` and eval harness.

---

## Phase 2C — Analysis artifact chunking (`artifact_chunks`) + artifact-lane fixups (embedding-ready)
**Goal:** make analysis artifacts behave like transcripts in retrieval: chunk-level candidates, relevant snippets, and future-ready embeddings.

### Deliverables
1) **Schema: add `artifact_chunks`**
   - New table: `artifact_chunks` (FK to `analysis_artifacts`, plus `call_id`, `call_started_at`, `kind`, `ordinal`, `content`, `token_count`, optional `start_char/end_char`).
   - Indexes:
     - BM25 (+ ngram alias) on `artifact_chunks.content` (primary lexical lane for analysis evidence)
     - GIN on `artifact_chunks.tech_tokens`
     - HNSW on `artifact_chunks.embedding vector(1024)`

2) **Ingest: chunk analysis artifacts deterministically**
   - On `POST /ingest/analysis`:
     - Store the canonical full artifact in `analysis_artifacts` (unchanged).
     - Create retrieval units in `artifact_chunks`:
       - chunk by headings/bullets/paragraphs (kind-aware)
       - `action_items`: prefer 1 chunk per item
       - `decisions`: prefer 1 chunk per decision bullet/paragraph
     - Compute `tech_tokens` per chunk.
     - Record the chunking config in `ingestion_runs.chunking_config`.

3) **Retrieval: switch analysis lanes to `artifact_chunks`**
   - Replace artifact BM25 + tech-token lanes to query `artifact_chunks` instead of `analysis_artifacts`.
   - Evidence-pack artifacts now reference **artifact chunks** (not whole artifacts).
   - Update `return_style=ids_only` to return:
     - `chunk:<id>` (transcript chunks)
     - `artifact_chunk:<id>` (analysis chunks)

4) **Evidence IDs + expand**
   - Evidence ID formats (for `/retrieve` and `/expand`):
     - `Q-<chunk_id>` (transcript evidence)
     - `A-<artifact_chunk_id>` (analysis evidence)
   - Update `POST /expand`:
     - `Q-*`: unchanged (uses `chunk_utterances` + optional window)
     - `A-*`: returns bounded excerpt from `artifact_chunks.content` (or from parent using `start_char/end_char` when available)

### Tests
- Integration: ingest analysis artifacts with multiple paragraphs/bullets, verify `artifact_chunks` created and retrievable.
- Integration: `/retrieve` returns `A-*` evidence IDs and snippets that actually contain the relevant passage.
- Integration: `/expand` supports `A-*` and respects `max_chars`.
- Unit: chunking is deterministic (same input → same chunk boundaries and ordinals).

### Acceptance criteria
- Analysis retrieval is chunk-granular and budget-friendly; snippets are relevant.
- Evidence IDs are stable and can round-trip through `/expand`.

---

## Phase 2D — Filesystem ingest queue (`inbox` scanner + Redis worker)
**Goal:** support unattended ingest by dropping bundles into a watched directory, with deterministic validation and observable job status.

### Deliverables
1) **Schema: ingest jobs + file audit tables**
   - Add `ingest_jobs`:
     - `bundle_id`, `status`, `queue_name`, `source_path`, `manifest_path`, `call_ref`, `call_id`, `error`, `attempts`, timestamps
   - Add `ingest_job_files`:
     - one row per file (`kind`, `relative_path`, `file_sha256`, `file_size_bytes`)

2) **Filesystem bundle contract**
   - Root from `INGEST_ROOT_DIR` with subdirs:
     - `inbox/`, `processing/`, `done/`, `failed/`
   - Bundle is ready only when `_READY` sentinel exists.
   - Required `manifest.json` with:
     - `call_ref`
     - optional transcript descriptor (`path`, `format=json_turns`, optional `sha256`, optional chunking options)
     - optional analysis file descriptors (`kind`, `path`, optional `sha256`, optional metadata)

3) **Scanner service**
   - Polls `inbox/` on interval.
   - Validates bundle and optional checksums.
   - Moves valid bundles to `processing/`.
   - Enqueues job ID in Redis queue.
   - Marks invalid bundles and moves to `failed/`.

4) **Worker service**
   - Consumes queue jobs.
   - Executes ingest pipeline:
     - resolve/create call from `call_ref`
     - transcript ingest (if present)
     - analysis ingest (if present)
   - Transitions status:
     - `queued` → `running` → `succeeded|failed`
   - Retries transient failures with bounded exponential backoff.
   - Moves processed bundle to `done/` or `failed/`.

5) **Job visibility API**
   - `GET /ingest/jobs`
   - `GET /ingest/jobs/{ingest_job_id}`

### Tests
- Unit: manifest validation (required files, path safety, checksum mismatch).
- Integration: ingest job endpoints return expected status and file audit payload.
- Smoke: scanner `--once` works and worker entrypoint boots.

### Acceptance criteria
- A valid bundle dropped into `inbox/` is ingested end-to-end without manual API calls.
- Invalid bundles never enter retrieval tables and are observable as `invalid|failed`.
- Job status is queryable via API for operational debugging.

---

## Phase 3 — Dense embeddings lane (pgvector) via HTTP embedding service
**Status:** Implemented in core API/scripts; continue tuning batch sizes and ANN thresholds per deployment.

**Goal:** add semantic recall while keeping tool usage deterministic.

### Deliverables
1) **Embedding client (HTTP)**
   - Env vars (defaults included in `.env.example`):
     - `EMBEDDINGS_BASE_URL`
     - `EMBEDDINGS_MODEL_ID`
     - `EMBEDDINGS_DIM=1024`
     - `EMBEDDINGS_TIMEOUT_S`
   - Client uses `httpx` and supports batch embedding.

2) **Embedding backfill command**
   - `uv run python -m app.scripts.embed_backfill`
   - Behavior:
     - Finds rows in `chunks` and `artifact_chunks` where `embedding IS NULL`.
     - Batches embedding requests.
     - Downshifts batch size automatically when provider limits are lower than configured batch size.
     - Updates embeddings in DB.
     - Writes an `ingestion_runs` update/record indicating embedding completion.

2b) **Optional auto-embed on ingest success**
   - Worker can embed the just-ingested call immediately after successful transcript/artifact ingest.
   - Controlled by:
     - `INGEST_AUTO_EMBED_ON_SUCCESS`
     - `INGEST_AUTO_EMBED_FAIL_ON_ERROR`

3) **Dense retrieval lane + planner**
   - Add to `/retrieve`:
     - If `EMBEDDINGS_BASE_URL` not set: skip dense lane and record in `notes`.
     - Else:
       - embed query
       - run pgvector search for chunks + artifact_chunks
       - planner chooses:
         - **exact scan** when filtered candidate set is small (e.g., scoped call_ids) OR
         - **HNSW** otherwise
       - for filtered ANN queries: use `SET LOCAL hnsw.iterative_scan = relaxed_order` and a configured `ef_search`.

4) **RRF fusion now merges 3 lanes**
   - `bm25 + tech_tokens + dense`

### Tests
- Unit: planner decision table (filters → exact scan vs ANN).
- Unit: “no embeddings configured” keeps endpoint functional.
- (Optional) Integration: enable a fake embedding service in tests and verify dense lane contributes.

### Acceptance criteria
- Semantic queries improve recall without breaking exact-token performance.
- Filtered queries do not “mysteriously return nothing” due to ANN overfiltering.
- Embedding backfill and ingest-time embedding tolerate provider batch-limit differences without operator hand-holding.

---

## Phase 4 — Reranking (HTTP) + evidence-pack selection
**Goal:** improve top-of-pack precision and reduce redundancy.

### Deliverables
1) **Rerank client (HTTP)**
   - Env vars:
     - `RERANK_BASE_URL`
     - `RERANK_MODEL_ID`
     - `RERANK_TIMEOUT_S`
     - `RERANK_MAX_CHARS_PER_DOC` (truncate before rerank)
     - `RERANK_TOPN_IN` (e.g., 80)
     - `RERANK_TOPM_OUT` (e.g., 12)

2) **Pipeline integration**
   - `/retrieve`:
     - candidate fetch per lane → RRF → take top N → rerank → select top M
     - candidates include transcript chunks (`chunks`) and analysis chunks (`artifact_chunks`)
     - enforce per-call diversity caps and dedupe rules

### Tests
- Unit: rerank ordering + truncation.
- Integration: with stub reranker returns deterministic scores, verify ordering changes and budgets enforced.

### Acceptance criteria
- Evidence packs become more precise with fewer irrelevant snippets, staying under budgets.

---

## Phase 5 — `/answer` (citation-gated) + LLM gateway (local + remote)
**Goal:** one-shot answering grounded strictly in evidence.

### Deliverables
1) **LLM gateway**
   - One OpenAI-compatible client that can target:
     - local base URL (LM Studio / Ollama OpenAI-compat)
     - remote provider base URL + API key
   - Env vars:
     - `LLM_BASE_URL`
     - `LLM_API_KEY` (optional for local)
     - `LLM_MODEL`
     - `LLM_TIMEOUT_S`

2) **Citation format + validator**
   - Standardize answer citations to: `... sentence text. [Q-123]` or `... [A-45][Q-12]`
   - Validator checks:
     - every sentence has ≥1 citation
     - citations reference evidence IDs in the pack
   - Repair loop:
     - if invalid, reprompt with validator failure report and require corrected output
     - bounded retries (e.g., 2)

3) **`POST /answer`**
   - Calls `/retrieve`
   - Calls LLM once (plus repair retries only if needed)
   - Returns:
     - `answer`
     - `citations` normalized list
     - `evidence_pack` (optional echo)

### Tests
- Unit: citation validator (good/bad cases).
- Integration: stub LLM returning uncited text triggers repair, then passes.

### Acceptance criteria
- No uncited sentences ever pass.
- When evidence is insufficient, return a structured “cannot answer from evidence.”

---

## Phase 6 — Entities + faceting (rule-based first, GLiNER second)
**Goal:** enable filtering (“service=api-gateway”, “ticket=ABC-123”) and better timelines.

### Deliverables
1) **Rule-based entity extraction into `entities/*_entities`**
   - Reuse existing tech-token regexes, but persist as entities + mentions:
     - `ERROR_CODE`, `TICKET_ID`, `URL`, `IP_ADDRESS`, `FILE_PATH`, `VERSION`, etc.
   - Backfill script: `uv run python -m app.scripts.entity_backfill`

2) **GLiNER integration (HTTP or worker)**
   - Env vars:
     - `NER_BASE_URL`
     - `NER_MODEL_ID`
   - Populate `entities` and `chunk_entities/artifact_entities` (entity filters should work for `artifact_chunks` via `artifact_id`).

3) **Retrieval filters**
   - `RetrieveFilters.entity_filters = [{label, value}]`
   - Implement filters in SQL using joins to entities/mentions.

### Tests
- Unit: normalization + dedupe behavior.
- Integration: entity filter reduces result set but still returns relevant evidence.

### Acceptance criteria
- “ticket ABC-123” works even with ASR noise due to combined BM25 ngrams + tech/entity lanes.
- Facets can be computed without heavy LLM steps.

---

## Phase 7 — Evaluation harness wiring + regression gates
**Goal:** make quality measurable and stable across changes.

### Deliverables
- Script to run the gold set through `/retrieve` and output results JSONL in the expected eval format.
- Store run artifacts: timestamp, git SHA, config snapshot.
- Add minimal regression gate: fail if recall@20 drops below chosen threshold.

### Acceptance criteria
- One command produces metrics and a stored results artifact.

---

## Operational notes (defaults)
- Dev default: DB in Docker, API local on `8001`.
- Keep transcript text out of logs by default; only log IDs + timings.
- Any architectural change must be reflected in `APP_SPEC.md` and `IMPLEMENTATION_PLAN.md` as part of the same change set.

# Cadence RAG (Transcript-Centric) — App Spec
**Status:** Canonical  
**Last updated:** 2026-02-03  

## 1) Summary
Build a **local-first, transcript-centric Cadence RAG** optimized for **technical call transcripts**. The system is designed for **high recall on exact technical strings** (error codes, ticket IDs, stack traces) while still answering semantic questions accurately.

Core strategy:
- **Deterministic, server-orchestrated retrieval** (no LLM tool-choice loops)
- **Hybrid retrieval**: dense vectors + BM25 + exact-token lane
- **GPU reranking**
- **Entity extraction** (NER + rule-based “technical entities”)
- **Evidence-pack answering** with strict **citation gating** (answers must be provably grounded)

## 2) Goals / Non-goals

### Goals (MVP)
- Ingest call transcripts + post-call artifacts (summary / decisions / action items)
- Allow ingest of transcript, analysis, or both without confusion; preserve provenance and links
- Ask questions and get **citation-backed answers** with speaker + timestamps
- Guarantee strong behavior on **technical tokens** (IDs, errors, versions, stack traces)
- Keep data **local-first** and reduce operational/tooling bloat
- Provide a stable, auditable retrieval pipeline with explicit budgets

### Non-goals (MVP)
- Perfect diarization correction UI
- Autonomous agents that run multi-step tool loops
- Multi-user RBAC and cross-device sync (planned later)

## 3) Primary user workflows
1. **Decision recall**: “What did we decide about migrating auth to X?”
2. **Action items**: “What action items were assigned to me last week?”
3. **Who said what**: “Who proposed the caching change and why?”
4. **Troubleshooting**: “Where did we discuss `ECONNRESET` in `api-gateway`?”
5. **Status across calls**: “What’s the latest status of ticket `ABC-123`?”

## 4) Success metrics
- **Retrieval**
  - Recall@20 and MRR on a gold set (defined in the Implementation Plan)
  - Fewer candidates needed for reranking while maintaining recall
- **Faithfulness**
  - 0 tolerance for uncited claims: every answer sentence must cite evidence
  - Automated post-checks catch and repair missing/invalid citations
- **Latency**
  - Interactive Q&A in seconds locally (reranker dominates)
- **Token discipline**
  - Evidence pack stays within strict max bytes/tokens consistently

## 5) Core technical decisions (opinionated defaults)

### 5.1 Canonical datastore
Use **Postgres 18** as the canonical store, deployed via **ParadeDB** (for `pg_search`) with `pgvector` included.

Rationale:
- One canonical store for metadata joins, filtering, provenance, and retrieval
- Best ergonomics when `pg_search` (BM25) and `pgvector` are colocated
- Postgres remains the source of truth; optional graph is enrichment only

Operational rule:
- **Pin ParadeDB image tags** (including Postgres major) and **assert extension versions at startup** (fail fast if unexpected).

Pinned versions (recommended):
- Postgres: **18.1**
- ParadeDB / `pg_search`: **0.21.5**
- pgvector (`vector` extension): **0.8.1**
- Docker image: `paradedb/paradedb:0.21.5-pg18` (then pin by digest; at time of writing this tag resolves to `sha256:919c0e96ae44845643683ecc67a532926eed84090867277ab4f1dd0a0080db6e`)

### 5.2 Extensions (baseline)
- `pg_search` (BM25 lexical search)
- `pgvector` (dense vector search)
- `pgcrypto` (UUID generation)
- `pg_trgm` (optional substring/fuzzy fallback for noisy ASR)

### 5.3 Embedding standard
- Standardize embedding dimension to **1024** across GPU profiles for one vector column shape.

### 5.4 Reliability principle
**Never rely on the model to decide which tools to call.**  
Retrieval + rerank + evidence-pack compression run deterministically **before** the LLM call.

## 6) Architecture

### 6.1 Default deployment shape (modular monolith)
Start as a **single FastAPI service** plus a **background worker** (Celery/RQ/Arq), a **filesystem ingest scanner**, and separate model inference endpoints if needed.

```
Client (CLI/Web)
   |
   v
API (FastAPI)
  - ingest endpoints
  - ingest job status endpoints
  - retrieve endpoint (deterministic)
  - answer endpoint (LLM once, citation-gated)
   |
   +--> Redis queue
   |
   +--> Scanner (polls ingest/inbox for ready bundles, validates, enqueues)
   |
   +--> Worker (async jobs: ingest bundles, chunking, embeddings, NER, reindex)
   |
   +--> Model endpoints (optional separate processes/containers)
         - embeddings
         - rerank
         - NER
   |
   v
Postgres 18 (ParadeDB): pg_search (BM25) + pgvector
```

Notes:
- Split “Retrieval Service” into its own service only if throughput/concurrency demands it.
- Keep LLM calls behind an “LLM Gateway” interface so you can swap local vs remote.

### 6.2 Why this avoids tool bloat
- Retrieval is not a tool the LLM chooses.
- The LLM receives a **budgeted evidence pack**, not a pile of chunks.
- Answers are validated post-hoc for citation completeness and repaired deterministically.

## 7) Data model (Postgres)

### 7.1 Design requirements
- Preserve provenance: `call_id`, timestamps, speaker attribution
- Support hierarchical retrieval: call-level artifacts first, then transcript quotes
- Support exact technical retrieval: IDs/errors/versions must be reliably matchable
- Support repeatable pipelines: store pipeline + model configuration per ingestion

### 7.2 Core tables (DDL)
> The DDL below is the canonical shape; index syntax for `pg_search` BM25 should be implemented per the pinned `pg_search` version.

```sql
-- Extensions
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 0) Corpus / namespace (optional but recommended)
CREATE TABLE IF NOT EXISTS corpora (
  corpus_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name              TEXT NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name)
);

-- 1) Calls (one row per call/session)
CREATE TABLE IF NOT EXISTS calls (
  call_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  corpus_id         UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
  external_id       TEXT,                      -- stable external key
  external_source   TEXT,                      -- namespace/source of external_id
  started_at        TIMESTAMPTZ NOT NULL,
  ended_at          TIMESTAMPTZ,
  title             TEXT,
  source_uri        TEXT,                      -- file path, URL, etc.
  source_hash       TEXT,                      -- content hash for idempotency
  participants      JSONB,                     -- [{name,email,role}, ...]
  tags              TEXT[],
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS calls_started_at_idx ON calls (started_at DESC);
CREATE INDEX IF NOT EXISTS calls_tags_gin_idx   ON calls USING GIN (tags);
CREATE INDEX IF NOT EXISTS calls_meta_gin_idx   ON calls USING GIN (metadata);
CREATE UNIQUE INDEX IF NOT EXISTS calls_external_id_uq
  ON calls (external_source, external_id)
  WHERE external_id IS NOT NULL;

-- 2) Utterances (raw speaker turns)
CREATE TABLE IF NOT EXISTS utterances (
  utterance_id      BIGSERIAL PRIMARY KEY,
  call_id           UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  speaker           TEXT,
  speaker_id        TEXT,                      -- diarization id if present
  start_ts_ms       BIGINT NOT NULL,
  end_ts_ms         BIGINT NOT NULL,
  confidence        REAL,
  text              TEXT NOT NULL,
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS utterances_call_ts_idx
  ON utterances (call_id, start_ts_ms);

-- 3) Chunks (retrieval units)
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id          BIGSERIAL PRIMARY KEY,
  call_id           UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  corpus_id         UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,

  -- Denormalized for hot filtering (avoid join in retrieval paths)
  call_started_at   TIMESTAMPTZ NOT NULL,

  speaker           TEXT,                      -- dominant speaker or 'MULTI'
  start_ts_ms       BIGINT NOT NULL,
  end_ts_ms         BIGINT NOT NULL,
  token_count       INT NOT NULL,
  text              TEXT NOT NULL,

  -- Dense lane
  embedding         vector(1024),

  -- Exact-token lane (deterministic match for technical identifiers)
  tech_tokens       TEXT[] NOT NULL DEFAULT '{}',

  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS chunks_call_ts_idx ON chunks (call_id, start_ts_ms);
CREATE INDEX IF NOT EXISTS chunks_started_at_idx ON chunks (call_started_at DESC);
CREATE INDEX IF NOT EXISTS chunks_meta_gin_idx ON chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS chunks_tech_tokens_gin ON chunks USING GIN (tech_tokens);
CREATE INDEX IF NOT EXISTS chunks_text_trgm ON chunks USING GIN (text gin_trgm_ops);

-- Dense ANN
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 4) Chunk provenance: which utterances built the chunk
CREATE TABLE IF NOT EXISTS chunk_utterances (
  chunk_id          BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  utterance_id      BIGINT NOT NULL REFERENCES utterances(utterance_id) ON DELETE CASCADE,
  ordinal           INT NOT NULL,
  PRIMARY KEY (chunk_id, utterance_id)
);

CREATE INDEX IF NOT EXISTS chunk_utterances_chunk_ordinal_idx
  ON chunk_utterances (chunk_id, ordinal);

-- 5) Analysis artifacts (call-level)
CREATE TABLE IF NOT EXISTS analysis_artifacts (
  artifact_id       BIGSERIAL PRIMARY KEY,
  call_id           UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  corpus_id         UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
  call_started_at   TIMESTAMPTZ NOT NULL,

  kind              TEXT NOT NULL,             -- summary | decisions | action_items | tech_notes | etc
  content           TEXT NOT NULL,
  token_count       INT NOT NULL,
  embedding         vector(1024),
  tech_tokens       TEXT[] NOT NULL DEFAULT '{}',
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS artifacts_call_kind_idx
  ON analysis_artifacts (call_id, kind);
CREATE INDEX IF NOT EXISTS artifacts_started_at_idx
  ON analysis_artifacts (call_started_at DESC);
CREATE INDEX IF NOT EXISTS artifacts_meta_gin
  ON analysis_artifacts USING GIN (metadata);
CREATE INDEX IF NOT EXISTS artifacts_tech_tokens_gin
  ON analysis_artifacts USING GIN (tech_tokens);
CREATE INDEX IF NOT EXISTS artifacts_embedding_hnsw
  ON analysis_artifacts USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 6) Artifact chunks (analysis retrieval units)
--
-- Purpose:
-- - `analysis_artifacts` remains the canonical, full document for provenance/browsing.
-- - `artifact_chunks` are the *retrieval units* (mirrors `utterances -> chunks`).
--
-- Why:
-- - Avoid coarse retrieval: whole-document artifacts are too big for evidence-pack budgets.
-- - Avoid poor dense retrieval: a single embedding for a long artifact is a semantic average.
-- - Ensure artifact snippets are actually relevant (not just "first N chars of the doc").
--
-- Notes:
-- - `content` is the chunk text used for BM25 + tech token lanes.
-- - `start_char`/`end_char` are optional offsets into the parent `analysis_artifacts.content` (if computed).
CREATE TABLE IF NOT EXISTS artifact_chunks (
  artifact_chunk_id BIGSERIAL PRIMARY KEY,
  artifact_id       BIGINT NOT NULL REFERENCES analysis_artifacts(artifact_id) ON DELETE CASCADE,
  call_id           UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  corpus_id         UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
  call_started_at   TIMESTAMPTZ NOT NULL,

  kind              TEXT NOT NULL,             -- denormalized from analysis_artifacts.kind
  ordinal           INT NOT NULL,              -- order within (artifact_id)
  content           TEXT NOT NULL,
  token_count       INT NOT NULL,
  start_char        INT,                       -- optional: offset into analysis_artifacts.content
  end_char          INT,                       -- optional: offset into analysis_artifacts.content

  embedding         vector(1024),
  tech_tokens       TEXT[] NOT NULL DEFAULT '{}',
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (artifact_id, ordinal)
);

CREATE INDEX IF NOT EXISTS artifact_chunks_artifact_ordinal_idx
  ON artifact_chunks (artifact_id, ordinal);
CREATE INDEX IF NOT EXISTS artifact_chunks_call_kind_idx
  ON artifact_chunks (call_id, kind);
CREATE INDEX IF NOT EXISTS artifact_chunks_started_at_idx
  ON artifact_chunks (call_started_at DESC);
CREATE INDEX IF NOT EXISTS artifact_chunks_meta_gin
  ON artifact_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS artifact_chunks_tech_tokens_gin
  ON artifact_chunks USING GIN (tech_tokens);
CREATE INDEX IF NOT EXISTS artifact_chunks_embedding_hnsw
  ON artifact_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 7) Entities (semantic + technical)
CREATE TABLE IF NOT EXISTS entities (
  entity_id         BIGSERIAL PRIMARY KEY,
  label             TEXT NOT NULL,             -- PERSON, ORG, SERVICE, ERROR_CODE, TICKET_ID, ...
  canonical         TEXT NOT NULL,
  normalized        TEXT NOT NULL,
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (label, normalized)
);

-- 8) Mentions
CREATE TABLE IF NOT EXISTS chunk_entities (
  chunk_id          BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  entity_id         BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  start_char        INT NOT NULL,
  end_char          INT NOT NULL,
  confidence        REAL,
  PRIMARY KEY (chunk_id, entity_id, start_char, end_char)
);
CREATE INDEX IF NOT EXISTS chunk_entities_entity_idx
  ON chunk_entities (entity_id);

CREATE TABLE IF NOT EXISTS artifact_entities (
  artifact_id       BIGINT NOT NULL REFERENCES analysis_artifacts(artifact_id) ON DELETE CASCADE,
  entity_id         BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  start_char        INT NOT NULL,
  end_char          INT NOT NULL,
  confidence        REAL,
  PRIMARY KEY (artifact_id, entity_id, start_char, end_char)
);
CREATE INDEX IF NOT EXISTS artifact_entities_entity_idx
  ON artifact_entities (entity_id);

-- 9) Pipeline config (reproducibility)
CREATE TABLE IF NOT EXISTS ingestion_runs (
  ingestion_run_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id           UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  pipeline_version  TEXT NOT NULL,             -- bump when chunking/NER/token rules change
  chunking_config   JSONB NOT NULL,
  embedding_config  JSONB NOT NULL,
  ner_config        JSONB NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingestion_runs_call_idx ON ingestion_runs (call_id);

-- 10) Transcript ingest dedupe (idempotency)
CREATE TABLE IF NOT EXISTS transcript_ingests (
  transcript_ingest_id BIGSERIAL PRIMARY KEY,
  call_id              UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
  transcript_hash      TEXT NOT NULL,
  utterance_count      INT NOT NULL DEFAULT 0,
  chunk_count          INT NOT NULL DEFAULT 0,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (call_id, transcript_hash)
);
CREATE INDEX IF NOT EXISTS transcript_ingests_call_idx
  ON transcript_ingests (call_id, created_at DESC);

-- 11) Ingest jobs (filesystem queue ingest)
CREATE TABLE IF NOT EXISTS ingest_jobs (
  ingest_job_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  bundle_id         TEXT NOT NULL UNIQUE,      -- stable bundle identifier
  status            TEXT NOT NULL,             -- queued|running|succeeded|failed|invalid
  queue_name        TEXT NOT NULL,
  source_path       TEXT NOT NULL,             -- bundle path in inbox/processing/done/failed
  manifest_path     TEXT NOT NULL,
  call_ref          JSONB NOT NULL DEFAULT '{}'::jsonb,
  call_id           UUID REFERENCES calls(call_id) ON DELETE SET NULL,
  error             TEXT,
  attempts          INT NOT NULL DEFAULT 0,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at        TIMESTAMPTZ,
  completed_at      TIMESTAMPTZ,
  CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'invalid'))
);
CREATE INDEX IF NOT EXISTS ingest_jobs_status_created_idx
  ON ingest_jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS ingest_jobs_call_idx
  ON ingest_jobs (call_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ingest_job_files (
  ingest_job_file_id BIGSERIAL PRIMARY KEY,
  ingest_job_id      UUID NOT NULL REFERENCES ingest_jobs(ingest_job_id) ON DELETE CASCADE,
  kind               TEXT NOT NULL,            -- manifest|transcript|analysis:<kind>
  relative_path      TEXT NOT NULL,
  file_sha256        TEXT NOT NULL,
  file_size_bytes    BIGINT NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (ingest_job_id, relative_path)
);
CREATE INDEX IF NOT EXISTS ingest_job_files_job_idx ON ingest_job_files (ingest_job_id);
```

### 7.3 Lexical search with pg_search (BM25)
Use `pg_search` BM25 indexes as the **primary lexical lane** for:
- chunks.text
- artifact_chunks.content (preferred; retrieval units)
- analysis_artifacts.content (optional coarse fallback)
- (optional) calls.title / metadata-derived fields

Tokenizer guidance:
- Use a word tokenizer (with stemming) for general language retrieval.
- Use an n-gram tokenizer for robustness to ASR noise and for “substring-ish” technical queries.

Canonical BM25 index examples (adjust to the pinned `pg_search` version; commit final DDL in migrations):

```sql
-- BM25 over chunks
CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON chunks
USING bm25 (
  chunk_id,
  text,
  (text::pdb.ngram(3,3, 'alias=text_ngram'))
)
WITH (key_field='chunk_id');

-- BM25 over artifact chunks (preferred for evidence packs)
CREATE INDEX IF NOT EXISTS artifact_chunks_bm25_idx ON artifact_chunks
USING bm25 (
  artifact_chunk_id,
  content,
  (content::pdb.ngram(3,3, 'alias=artifact_chunk_content_ngram'))
)
WITH (key_field='artifact_chunk_id');

-- BM25 over whole artifacts (optional coarse fallback)
CREATE INDEX IF NOT EXISTS artifacts_bm25_idx ON analysis_artifacts
USING bm25 (
  artifact_id,
  content,
  (content::pdb.ngram(3,3, 'alias=content_ngram'))
)
WITH (key_field='artifact_id');
```

### 7.4 Exact-token lane (must-have for technical reliability)
At ingestion, extract high-signal technical tokens and store them in:
- `chunks.tech_tokens`
- `artifact_chunks.tech_tokens` (preferred; retrieval units)
- `analysis_artifacts.tech_tokens` (optional coarse fallback)

Examples:
- ticket IDs: `ABC-123`, `JIRA-9381`
- error codes: `ECONNRESET`, `ORA-00001`, `SQLSTATE 23505`
- versions: `v1.2.3`, `2026.01`
- commit hashes, IPs, hostnames, URLs, file paths
- sales/SE domain terms: `BOM`, `build`, `SSD`, `object store`, `tiering`
- vendor/cloud + competition signals: `Lenovo`, `Dell`, `Supermicro`/`SMC`, `AWS`/`Azure`/`GCP`/`OCI`, `competitive`, `bake-off`, `head-to-head`, `vs`, `incumbent`

This lane provides deterministic matches even if BM25 tokenization/stemming misses edge cases.
When token extraction rules are updated, run a deterministic `tech_tokens` backfill over existing rows.

### 7.5 Call linking & provenance (transcript + analysis)
Support ingesting **transcripts**, **analysis artifacts**, or **both**, while keeping all outputs linked and auditable.

Core rules:
- A **call can exist without a transcript** (analysis-only ingest). It remains the canonical anchor.
- Every `analysis_artifacts` row **must link to a call** via `call_id`.
- Transcript content links via `utterances` → `chunks` → `call_id`.

Ingest must accept a **call_ref** object to resolve linkage deterministically:
```json
{
  "call_id": "uuid (optional, preferred when known)",
  "external_id": "string (optional, stable external key)",
  "external_source": "string (optional, namespace for external_id)",
  "source_uri": "string (optional)",
  "source_hash": "string (optional)",
  "started_at": "timestamp (optional)",
  "ended_at": "timestamp (optional)",
  "title": "string (optional)"
}
```

Resolution rules:
1. If `call_id` is provided: attach to that call (error if missing).
2. Else if `external_id` is provided: find or create a call with `calls.external_id = external_id` (scoped by `external_source` if provided).
3. Else if `source_uri + source_hash` provided: find or create a call with those fields.
4. Else: create a new call with whatever metadata was provided.
5. If multiple matches occur: return **409 Conflict** and require `call_id`.

Provenance storage (recommended in `metadata`):
- `calls.metadata.external_id` (legacy/multi-id; primary is `calls.external_id`)
- `analysis_artifacts.metadata.analysis_source_uri`
- `analysis_artifacts.metadata.analysis_hash`
- `analysis_artifacts.metadata.analysis_version`
- `analysis_artifacts.metadata.produced_by`

## 8) Ingestion pipeline

### 8.1 Inputs
- Transcript formats: `json_turns` plus adapter-normalized JSON via `format=auto` (for common export variants)
- Post-call artifacts: summary, decisions, action items, tech notes (`markdown`, `text`, `csv`, `tsv`, `json`, `html`, `docx`, `pdf`)

### 8.2 Pipeline stages
1. **Parse + normalize**
   - Normalize whitespace, keep code blocks/logs intact
   - Ensure timestamps, speaker labels (or UNKNOWN)
2. **Store utterances**
   - One row per turn with timestamps and speaker
3. **Chunking**
   - Transcript chunking (`utterances` → `chunks`)
     - Target 250–450 tokens, hard max 600
     - Overlap ~50 tokens (or 1–2 utterances)
     - Prefer semantic boundaries; avoid splitting code/log blocks
     - Write `chunk_utterances` mapping for deterministic expand
   - Analysis artifact chunking (`analysis_artifacts` → `artifact_chunks`)
     - Chunk by document structure (headings, bullet groups, paragraphs) rather than speaker turns
     - Use `kind` (`summary`, `decisions`, `action_items`, `tech_notes`, etc.) to guide chunking:
       - `action_items`: prefer 1 chunk per action item
       - `decisions`: prefer 1 chunk per decision bullet / paragraph
     - Store `artifact_id`, `ordinal`, and (optionally) `start_char`/`end_char` into the parent artifact
4. **Tech token extraction**
   - Regex/rule extraction for IDs/errors/versions/etc → `tech_tokens`
5. **Entity extraction**
   - NER (GLiNER) for semantic entities
   - Rule-based entities for technical patterns (high precision)
   - Store in `entities` + mention tables
6. **Embeddings**
   - Embed transcript chunks and artifact chunks (dim=1024)
   - Default worker behavior: after successful ingest, auto-embed that call's new `chunks` and `artifact_chunks` when embedding service is configured
   - Auto-embed failure mode is configurable (fail-open by default; optional fail-closed)
   - (Optional) also embed whole artifacts (`analysis_artifacts.embedding`) as a coarse fallback vector
   - Store embedding config in `ingestion_runs`
7. **Indexing**
   - pgvector HNSW for dense
   - pg_search BM25 for lexical

### 8.3 Idempotency and reprocessing
- Deduplicate ingest by `(source_uri, source_hash)` when possible
- Deduplicate transcript ingest per call via `(call_id, transcript_hash)`:
  - same normalized transcript + chunking options for the same call becomes a no-op
- Pipeline versioning ensures re-chunk/re-embed jobs are traceable

### 8.4 Filesystem ingest queue contract
Use a deterministic drop-folder contract for unattended ingest:

- Root: `INGEST_ROOT_DIR` (default `./ingest`)
- Scanner input: `INGEST_ROOT_DIR/inbox/<bundle_id>/`
- Scanner also supports single-file drops directly in `INGEST_ROOT_DIR/inbox/` (auto-wrap mode)
- Required files in each bundle directory:
  - `_READY` sentinel (scanner ignores bundles until this exists)
- `manifest.json` is optional when auto-manifest is enabled (`INGEST_AUTO_MANIFEST=true`):
  - scanner infers transcript + analysis files and writes deterministic `manifest.json`
  - scanner sets `call_ref.external_source=filesystem` and `call_ref.external_id=<bundle_id>`
- Scanner lifecycle directories:
  - `inbox/` → `processing/` → `done/` (or `failed/`)

`manifest.json` contract:
```json
{
  "bundle_id": "optional-string; defaults to folder name",
  "call_ref": {
    "external_source": "gong",
    "external_id": "abc-123",
    "started_at": "2026-02-07T18:00:00Z",
    "title": "Storage architecture call"
  },
  "transcript": {
    "path": "transcript.json",
    "format": "auto",
    "sha256": "optional sha256 hex",
    "options": { "target_tokens": 350, "max_tokens": 600, "overlap_tokens": 50 }
  },
  "analysis": [
    {
      "kind": "summary",
      "path": "analysis/summary.md",
      "format": "markdown",
      "sha256": "optional sha256 hex",
      "metadata": { "producer": "post-call-bot" }
    }
  ]
}
```

Format behavior:
- `transcript.format`:
  - `json_turns`: strict schema (`speaker`, `start_ts_ms`, `end_ts_ms`, `text`)
  - `markdown_turns`: Markdown transcripts (`**Speaker**: text` + timestamp lines)
  - `auto`: adapter maps common JSON export variants and Markdown transcripts into canonical `json_turns`
- `analysis[].format`:
  - `auto`, `text`, `markdown`, `csv`, `tsv`, `json`, `html`, `docx`, `pdf`
  - tabular/structured formats are normalized to retrieval-friendly text before ingest
  - `pdf` defaults to native text extraction; optional OCR fallback may run for low-quality text when enabled (`ANALYSIS_PDF_OCR_*`)

Operational behavior:
- If `manifest.json` is absent and auto-manifest is enabled, scanner generates one before validation.
- In single-file auto-wrap mode, scanner creates a processing bundle from the file, generates a manifest, then enqueues normally.
- Scanner applies a minimum file age gate before consuming direct inbox files (`INGEST_SINGLE_FILE_MIN_AGE_S`) to avoid partial-copy ingest.
- Scanner validates required files and optional `sha256` checks before enqueueing.
- Scanner writes one row in `ingest_jobs` and per-file metadata in `ingest_job_files`.
- Worker updates status transitions: `queued` → `running` → `succeeded|failed`.
- Worker can auto-embed newly ingested call rows when `EMBEDDINGS_BASE_URL` is configured; this is controlled by `INGEST_AUTO_EMBED_ON_SUCCESS` and `INGEST_AUTO_EMBED_FAIL_ON_ERROR`.
- OCR behavior for scanned PDFs is deterministic and gated by config thresholds (`ANALYSIS_PDF_OCR_MIN_CHARS`, `ANALYSIS_PDF_OCR_MIN_ALPHA_RATIO`, `ANALYSIS_PDF_OCR_MAX_PAGES`), and is disabled by default.
- Worker retry behavior:
  - retry transient failures with bounded exponential backoff (`INGEST_JOB_MAX_ATTEMPTS`, `INGEST_JOB_RETRY_BACKOFF_S`)
  - mark terminal failures only after max attempts are exhausted
- Invalid bundles are marked `invalid` and moved to `failed/`.
- Job status is observable via `GET /ingest/jobs` and `GET /ingest/jobs/{id}`.

## 9) Query / retrieval pipeline (deterministic)

### 9.1 Query analysis (no LLM required)
- Intent heuristics: decision/action-items/who-said/troubleshooting/status/auto
- Extract technical tokens from the query (same rules as ingest)
- Extract explicit filters (date range, tags, entity constraints)

### 9.2 Hierarchical retrieval (default strategy, with recall-safe fallback)
1. **Call/artifact retrieval**
   - Lexical BM25 on **artifact chunks** (decisions/action items/summaries/tech notes)
   - Dense vector search on **artifact chunk embeddings**
   - Result: shortlist of calls and high-signal artifact chunks (retrieval units)
2. **Chunk retrieval scoped to shortlisted calls**
   - Lexical BM25 on chunks.text (within call subset)
   - Dense vector search on chunks.embedding (within call subset)
   - Exact-token lane via `tech_tokens @> ARRAY[...]` (within call subset)

Benefits:
- Improves relevance and diversity across calls
- Makes filtering cheaper and more reliable
- Produces better evidence packs (fewer redundant chunks)

Recall-safe fallback (avoid “artifact shortlist missed the right call”):
- If the artifact stage is “low confidence” (e.g., too few calls shortlisted, weak/flat scores, or insufficient candidates), run a **global chunk search** lane (BM25 + dense + `tech_tokens`) without call scoping and proceed with fusion+rERANK from that set.
- If the query contains extracted `tech_tokens`, include a **global `tech_tokens` match lane** regardless of call scoping (cheap insurance for exact identifiers).

### 9.3 Retrieval planner (ANN vs exact)
For each dense query (chunks or artifact_chunks), choose between:
- **ANN (HNSW)** when the candidate set is large and filters are broad
- **Exact scan** when filters narrow the candidate set sufficiently (e.g., “within top calls” or tight date range)

Planner inputs:
- estimated rows after filters
- desired topK
- observed latency targets

### 9.4 Fusion + rerank
- Fuse candidates from lanes using **RRF** (dense + BM25 + exact-token)
- GPU rerank top N (e.g., 60–120) to top M (e.g., 8–16)
- Truncate reranker inputs (e.g., 512–2048 tokens) for throughput

### 9.5 Evidence pack construction (budgeted)
Default targets:
- 1–2 artifact chunks (decision/action/summary) when relevant
- 2–6 transcript quotes
- Max 2 quotes per call (unless deep-dive requested)

Each evidence item includes:
- stable evidence_id
- provenance (call_id, chunk_id and/or artifact_chunk_id + artifact_id, speaker, timestamps)
- bounded snippet
- 1-sentence “why relevant”
- scores and lane metadata

### 9.6 Answer synthesis + citation gating
One LLM call gets:
- user query
- intent
- evidence pack JSON
- strict instruction: answer only from evidence; cite every sentence

Post-checks (deterministic):
- Every sentence has ≥1 citation
- Citations reference evidence IDs present in the evidence pack
- If failed: auto-reprompt with the failure report and require corrected output

## 10) Model stack (GPU profiles)

### Profile A (10GB VRAM)
- Embeddings: Qwen3-Embedding-0.6B (dim=1024)
- Reranker: Qwen3-Reranker-0.6B
- NER: GLiNER small/medium + rule-based extractors

### Profile B (24GB VRAM)
- Embeddings: Qwen3-Embedding-4B (dim forced to 1024)
- Reranker: Mixedbread rerank-large-v2 (or Qwen reranker 4B for single-family stack)
- NER: GLiNER large-v2 + rule-based extractors

### Model serving playbook (recommended defaults)
Pick the simplest option that meets your latency/throughput needs, and keep it reproducible.

Option A (recommended for clean separation):
- Embeddings: **TEI** (Text Embeddings Inference) for high-throughput batched embedding.
- Rerank + NER: **PyTorch/Transformers** service (batched), separate from the API process.

Option B (simplest for MVP):
- A single Python “model service” process hosting embedding + rerank + NER with batching.

Operational defaults:
- Rerank input truncation: start with **1024 tokens** (try 512/1024/2048 in eval).
- Rerank batch sizing:
  - 10GB: prioritize smaller batches; keep truncation conservative.
  - 24GB: increase batch size first; then raise truncation if needed.
- Always expose:
  - max sequence length per endpoint
  - batch size limits
  - GPU memory utilization knobs (server-dependent)

## 11) Evidence pack contract (server → LLM)
```json
{
  "query_id": "uuid",
  "intent": "auto | decision | action_items | who_said | troubleshooting | status",
  "budget": { "max_evidence_items": 8, "max_total_chars": 6000 },
  "artifacts": [
    {
      "evidence_id": "A-789",
      "call_id": "uuid",
      "artifact_id": 456,
      "artifact_chunk_id": 789,
      "kind": "decisions",
      "snippet": "…",
      "why_relevant": "…"
    }
  ],
  "quotes": [
    {
      "evidence_id": "Q-789",
      "call_id": "uuid",
      "chunk_id": 999,
      "speaker": "Alice",
      "start_ts_ms": 123456,
      "end_ts_ms": 130000,
      "snippet": "…",
      "why_relevant": "…"
    }
  ],
  "notes": {
    "retrieval": {
      "planner": "lexical_only|ann|exact",
      "dense_topk": 0,
      "lex_topk": 50,
      "artifact_chunk_lex_topk": 10,
      "tech_token_topk": 50,
      "reranked_from": null,
      "lanes": { "bm25": true, "tech_tokens": true, "dense": false }
    }
  }
}
```

Evidence ID formats (for citation gating):
- Transcript quote evidence: `Q-<chunk_id>`
- Analysis evidence: `A-<artifact_chunk_id>` (includes `artifact_id` for provenance)

## 12) API surface (OpenAPI-oriented)

### Ingest
- `POST /ingest/call` (create/update call record from `call_ref`)
- `POST /ingest/transcript` (accepts `call_ref` for linking)
- `POST /ingest/analysis` (accepts `call_ref` for linking)
- `GET /ingest/jobs` (queue status and per-file audit trail)
- `GET /ingest/jobs/{ingest_job_id}` (single job detail)

### Retrieval + answering
- `POST /retrieve` → evidence pack only
  - Optional request fields:
    - `return_style`: `evidence_pack_json` (default) or `ids_only`
    - `debug`: boolean; when true includes lane ranks/scores (no extra text beyond snippets)
  - If `return_style=ids_only`, response is `{ query_id, retrieved_ids:[ "chunk:<id>", "artifact_chunk:<id>" ] }`
- `POST /answer` → one-shot answer (includes evidence pack + citations)
- `POST /expand` → bounded context expansion for one evidence item (uses `chunk_utterances` to reconstruct transcript context; uses `artifact_chunks` for analysis evidence)

### Browsing
- `GET /calls/{id}`
- `GET /chunks/{id}`
- `GET /artifacts/{id}`

## 13) Operational requirements
- Local-first: default storage on local disk; no transcript text in logs
- Structured logs include correlation IDs (`X-Request-ID`) and IDs/timings; avoid transcript/body content in logs
- Pin versions:
  - Docker image tag (ParadeDB version + Postgres major)
  - Docker image digest (after first pull)
  - `pg_search` version
  - `pgvector` version
  - model IDs for embedding/rerank/NER
- Fail-fast startup checks:
  - Assert expected extension versions via `pg_extension`
  - Expose a diagnostics endpoint (DB + extension + model versions)
- Backups: regular `pg_dump` or filesystem snapshot
- Config capture: record server + extension versions in a diagnostics endpoint
- Operational run modes:
  - First run/new environment: bring up `db` + `redis`, apply migrations, then start `api` + scanner + worker.
  - Normal run/no schema changes: start services directly; no migration step required.
  - Post-schema-change run: apply migrations before (re)starting API/worker paths that touch updated tables.

## 14) Future extensions (explicitly optional)
- **Graph layer** (Neo4j) only as enrichment after entity extraction stabilizes and you have graph-native use cases
- **Dedicated vector DB** (Qdrant) only if you hit performance walls at very large scale
- **MCP wrapper** only as a thin front-end: `kb.search` (evidence pack) and `kb.expand` (bounded context)

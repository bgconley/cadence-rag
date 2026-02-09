# Cadence RAG (Transcript-Centric) — Implementation Plan
**Status:** Canonical  
**Last updated:** 2026-02-09  

## 0) Guiding rules (non-negotiable)
- Retrieval is deterministic and server-orchestrated; the LLM never “chooses tools.”
- The LLM sees only a budgeted evidence pack and must cite every sentence.
- Exact technical token recall is treated as a first-class requirement (not “nice to have”).
- Pin versions early (DB image + extensions + models), **assert at startup**, and record configs per ingestion run.
- Canonical phase sequencing and execution scope are tracked in `PHASED_PLAN.md`; this doc focuses on implementation principles and capability design.

## 0.1) Pinned versions (as of 2026-02-04)
- Postgres: **18.1**
- ParadeDB Docker image: `paradedb/paradedb:0.21.5-pg18` (then pin by digest)
- `pg_search` (extversion): **0.21.5**
- `pgvector` / `vector` (extversion): **0.8.1**

## 0.2) Current execution checkpoint (2026-02-09)
- Implemented in codebase:
  - Phases 0 through 2D from `PHASED_PLAN.md`
  - Phase 3 dense retrieval lane (`/retrieve`) with embedding client + planner (exact vs ANN)
  - Embedding ops hardening:
    - adaptive batch-size downshift in `embed_backfill`
    - optional ingest-time auto-embed of newly ingested call rows
    - configurable fail-open vs fail-closed behavior for auto-embed
- Next in execution order:
  - Phase 4 reranking integration
  - Phase 5 `/answer` with citation gating

## 1) Phase 0 — Project bootstrap + database foundation
**Goal:** runnable local stack with the canonical schema and a minimal API skeleton.

### Deliverables
- Docker Compose (or equivalent) with:
  - Postgres 18 via a **pinned ParadeDB image tag** (includes `pg_search` + `pgvector`): `paradedb/paradedb:0.21.5-pg18`
  - (recommended) pin the image digest after the first pull
  - API service skeleton
  - Worker skeleton (optional in this phase)
- Migration framework (Alembic or similar)
- Initial schema applied (calls/utterances/chunks/chunk_utterances/analysis_artifacts/artifact_chunks/entities)
- Health/diagnostics endpoints:
  - DB connectivity
  - `server_version`
  - installed extension versions
  - configured model IDs (if present)
- Startup “fail fast” checks:
  - assert expected `pg_search` and `pgvector` versions via `pg_extension`

### Acceptance criteria
- `docker compose up` starts DB + API
- Schema applies cleanly on a fresh volume
- Diagnostics endpoint reports DB + extension versions
- Startup checks fail fast if:
  - Postgres is not 18.1
  - `pg_search` extversion is not 0.21.5
  - `vector` extversion is not 0.8.1

## 2) Phase 0.5 — Minimal evaluation harness (do this early)
**Goal:** establish feedback loops before tuning chunking/retrieval/reranking.

### Deliverables
- Gold set format (YAML/JSONL) with:
  - query text
  - required relevant IDs (artifact_chunk_id/chunk_id)
  - expected answer bullets (optional)
- Metrics scripts:
  - recall@k, MRR, nDCG@k
  - citation completeness checks for answers

### Acceptance criteria
- You can run a single command to evaluate current retrieval against the gold set
- Results are stored with a timestamp and config snapshot

## 3) Phase 1 — Ingestion MVP (transcripts + artifacts)
**Goal:** reliably ingest transcripts into utterances/chunks with deterministic provenance.

### Deliverables
- Transcript parsers (start with one canonical format; add others incrementally)
- Utterance normalization (speaker, timestamps, text cleanup rules)
- Chunker that:
  - targets 250–450 tokens; hard max 600
  - preserves code/log blocks
  - writes `chunk_utterances` mappings
- Call resolution via `call_ref`:
  - resolve by `call_id` → `external_id` → `source_uri+source_hash` → create new
  - return 409 on ambiguous matches
- `calls.external_id` + `calls.external_source` columns with a unique index (per spec)
- Idempotent ingest:
  - `(source_uri, source_hash)` detection
  - safe re-run behavior
  - transcript-level dedupe per call via transcript hash reservation (`transcript_ingests`)
- Artifact ingest endpoint (summary/decisions/action items), supporting **analysis-only ingest**
- Filesystem ingest conveyor:
  - drop bundles into `INGEST_ROOT_DIR/inbox/<bundle_id>/` with `manifest.json` + `_READY`
  - scanner validates bundles and enqueues jobs to Redis
  - RQ worker processes jobs and updates `ingest_jobs` / `ingest_job_files`
  - bounded retry/backoff for transient failures before terminal failure state
  - status APIs expose queued/running/succeeded/failed/invalid states

### Acceptance criteria
- Ingest N calls end-to-end and browse call → chunks → utterances provenance
- `expand` can reconstruct bounded context deterministically (via `chunk_utterances`)
- Ingest analysis-only and later transcript; both attach to the same call via `call_ref`
- Drop-folder ingest works end-to-end without manual API calls per file

## 4) Capability band — Search lanes (BM25 + exact-token + dense)
Mapping note: corresponds to Phases 2B/2C/3 in `PHASED_PLAN.md`.
**Goal:** implement the three retrieval lanes with stable, explainable outputs.

### Deliverables
- Technical token extraction (regex/rules) for:
  - ticket IDs, errors, versions, hashes, URLs, file paths, IPs/hosts
  - sales/SE terms and vendor/cloud aliases (e.g., BOM/build/object-store/tiering, OEM/cloud providers, competition signals)
  - stored in `tech_tokens` arrays
  - provide a replay/backfill path so historical rows adopt rule updates without full reingest
- BM25 search using `pg_search` for:
  - chunks.text
  - artifact_chunks.content (preferred for analysis evidence packs)
- Pin and implement canonical BM25 indexes in migrations (include n-gram tokenization where needed)
- Dense vector search using `pgvector` HNSW for:
  - chunks.embedding
  - artifact_chunks.embedding
- Embedding operations hardening:
  - call-scoped auto-embed after successful ingest (config-gated)
  - adaptive batch-size downshift for backfill when provider-enforced limits are lower than configured batch size
  - explicit fail-open/fail-closed control for embedding during ingest
- Retrieval planner:
  - decides ANN vs exact scan for dense queries based on filter selectivity / estimated rows
- A single `POST /retrieve` endpoint returning:
  - lane results with scores and provenance
  - fused candidates (RRF) before reranking
- Implement hierarchical retrieval with a recall-safe fallback:
  - artifact_chunks → shortlist calls → scoped chunk search
  - fallback to global chunk search when shortlist confidence is low

### Acceptance criteria
- Queries containing exact tokens like `ECONNRESET` / `ABC-123` reliably return matches
- Dense retrieval works with filters without “empty result surprises” (planner chooses exact scan when appropriate)
- Dense ingestion/backfill flows complete reliably across embedding services with different batch ceilings.

## 5) Capability band — Reranking + evidence packs
Mapping note: corresponds to Phase 4 in `PHASED_PLAN.md`.
**Goal:** turn candidate lists into compact, high-signal evidence packs.

### Deliverables
- GPU reranker integration (batch scoring; truncation controls)
- Document and implement the model serving approach (TEI/vLLM/torch), including batching + max-length defaults per GPU profile
- Fusion + rerank pipeline:
  - BM25 + dense + tech_tokens → RRF → rerank top N → select top M
- Evidence pack generator:
  - max items and max chars enforcement
  - diversity rules (cap per call)
  - snippets with timestamps + speaker
- `POST /retrieve` returns evidence pack JSON (canonical contract)

### Acceptance criteria
- Evidence packs stay under budget for all gold-set queries
- Top evidence is not overly redundant (dedupe + per-call caps)

## 6) Capability band — Answer endpoint with citation gating (reliability hardening)
Mapping note: corresponds to Phase 5 in `PHASED_PLAN.md`.
**Goal:** one-shot answers with strict grounding guarantees.

### Deliverables
- `POST /answer` endpoint:
  - calls `/retrieve`
  - calls LLM once with evidence pack + strict citation instructions
  - runs deterministic post-checks:
    - every sentence has citations
    - citations refer to evidence IDs present in pack
  - auto-reprompt on failure with a structured failure report
- Output format:
  - answer text
  - citations list (evidence_id + provenance)
  - echo evidence pack (optional)

### Acceptance criteria
- No uncited sentences pass the validator
- Failure modes are repaired automatically (or return a structured “cannot answer from evidence” response)

## 7) Capability band — Entities + faceting
Mapping note: corresponds to Phase 6 in `PHASED_PLAN.md`.
**Goal:** improve filtering, timelines, and query understanding.

### Deliverables
- GLiNER extraction for semantic entities (PERSON/ORG/SERVICE/etc)
- Rule-based entities for technical patterns (ERROR_CODE/TICKET_ID/etc)
- Entity normalization rules and alias strategy (optional)
- Add retrieval filters:
  - date range, tags
  - entity filters (label/value)
- Basic UI/CLI facets (even a minimal TUI/CLI is fine)

### Acceptance criteria
- Entity filters narrow results without recall collapse (planner chooses exact scan as needed)
- “ticket ABC-123” works even with ASR noise via combined BM25 + ngrams/trgm + tech_tokens

## 8) Capability band — Performance + regression hardening
Mapping note: corresponds to Phase 7 in `PHASED_PLAN.md`.
**Goal:** make performance predictable and prevent quality regressions.

### Deliverables
- Profiling harness:
  - embed throughput
  - retrieval latency per lane
  - rerank throughput pairs/sec at different truncation lengths
- Caching:
  - query hash → evidence pack
  - rerank cache for (query_id, candidate_id) pairs (optional)
- Regression tests:
  - retrieval metrics thresholds on gold set
  - answer citation compliance tests
- Operational guardrails:
  - no transcript text in logs by default
  - configurable redaction at ingest (email/phone)

### Acceptance criteria
- Gold-set metrics do not regress beyond a configured tolerance
- Reranker truncation length is tuned to fit latency targets

## 9) Optional phases (only if justified)

### 9.1 Graph enrichment (Neo4j) — only after you have real graph-native queries
- Sync entities/relationships from Postgres; keep Postgres canonical
- Deterministic Cypher templates (no LLM-generated Cypher)

### 9.2 MCP wrapper — only as a thin front-end
- Tools:
  - `kb.search` → evidence pack only
  - `kb.expand` → bounded excerpt only

## 10) Key risks and mitigations
- **Exact-token misses** → tech_tokens lane + BM25 ngrams + trigram fallback
- **Filtered ANN recall collapse** → retrieval planner (exact scan when selective) + iterative scans as fallback
- **Hierarchical retrieval misses** → fallback to global chunk retrieval when artifact shortlist confidence is low
- **Context bloat** → hard evidence pack budgets + diversity rules + dedupe
- **Hallucinated glue text** → citation gating + automatic repair loop
- **Model upgrades break behavior** → pin model versions + store configs per ingestion run + regression suite
- **Licensing (`pg_search` is AGPL)** → document early; plan for fallback to built-in Postgres FTS if requirements change

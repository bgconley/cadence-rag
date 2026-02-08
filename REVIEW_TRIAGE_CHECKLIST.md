# Review Triage Checklist

This checklist captures external review findings and their disposition against the current roadmap.

## Accept now (implement immediately)

- [x] Add transcript-level idempotency/deduplication for `POST /ingest/transcript`.
- [x] Add retry/backoff behavior for failed filesystem ingest jobs.
- [x] Add structured logging with correlation/request IDs for API + ingest pipeline paths.
- [x] Tighten request/model validation (chunking constraints and bounded expand payload).
- [x] Improve eval harness with a larger sample set and a regression-gate command.

## Accept later (roadmap-aligned)

- [ ] GPU reranker integration (Phase 4).
- [ ] `/answer` endpoint with citation gating (Phase 5).
- [ ] GLiNER/entity pipeline + entity filters/faceting (Phase 6).
- [ ] Corpus CRUD/filter APIs when multi-corpus workflows are actively needed.
- [ ] Deployment hardening (healthchecks, restart policies, secrets, non-root runtime).

## Partially accept / clarify

- [ ] Keep embedding dim fixed at `1024` by contract; avoid making runtime dimension dynamic unless contract changes.
- [ ] Keep `CallRef` permissive for analysis-only/new-call workflows; add quality safeguards without breaking this behavior.
- [ ] Add retrieval failure logging and stable error envelopes as hardening, not as a precondition for Phase 3-5 work.
- [ ] Keep per-call quote cap server-controlled default for now; revisit request-level configurability later.

## Defer / not adopting as stated

- [ ] Do not force ORM migration right now; continue with tested parameterized SQL.
- [ ] Do not require API version prefix yet (`/v1`) until external client compatibility pressure appears.
- [ ] Do not introduce async DB retrieval lanes yet; optimize after baseline profiling indicates need.

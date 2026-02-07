from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query

from .browse import expand_evidence, get_call, get_chunk, list_calls
from .config import settings
from .db import fetch_db_info, validate_versions
from .ingest_fs import get_ingest_job, list_ingest_jobs
from .ingest import ingest_analysis, ingest_call, ingest_transcript
from .retrieve import retrieve_evidence
from .schemas import (
    AnalysisIngestRequest,
    CallIngestRequest,
    ChunkingOptions,
    ExpandRequest,
    RetrieveRequest,
    TranscriptIngestRequest,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.skip_version_check:
        ok, message = validate_versions()
        if not ok:
            raise RuntimeError(message)
    yield


app = FastAPI(title="Personal RAG API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    try:
        info = fetch_db_info()
    except Exception as exc:  # pragma: no cover - safety
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "db": info}


@app.get("/diagnostics")
def diagnostics() -> dict:
    try:
        info = fetch_db_info()
        ok, message = validate_versions()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return {
        "status": "ok" if ok else "mismatch",
        "detail": message,
        "db": info,
        "expected": {
            "postgres": settings.expected_pg_version,
            "pg_search": settings.expected_pg_search_version,
            "pgvector": settings.expected_pgvector_version,
        },
    }


@app.post("/ingest/transcript")
def ingest_transcript_endpoint(payload: TranscriptIngestRequest) -> dict:
    if payload.transcript.format != "json_turns":
        raise HTTPException(status_code=400, detail="unsupported transcript format")
    options = payload.options or ChunkingOptions()
    call_id, utterances_ingested, chunks_created = ingest_transcript(
        call_ref=payload.call_ref,
        utterances_in=payload.transcript.content,
        options=options,
    )
    return {
        "call_id": str(call_id),
        "utterances_ingested": utterances_ingested,
        "chunks_created": chunks_created,
    }


@app.post("/ingest/call")
def ingest_call_endpoint(payload: CallIngestRequest) -> dict:
    call_id, created = ingest_call(payload.call_ref)
    return {"call_id": str(call_id), "created": created}


@app.post("/ingest/analysis")
def ingest_analysis_endpoint(payload: AnalysisIngestRequest) -> dict:
    if not payload.artifacts:
        raise HTTPException(status_code=400, detail="no artifacts provided")
    call_id, created = ingest_analysis(
        call_ref=payload.call_ref,
        artifacts=payload.artifacts,
    )
    return {"call_id": str(call_id), "artifacts_created": created}


@app.get("/ingest/jobs")
def list_ingest_jobs_endpoint(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    allowed = {"queued", "running", "succeeded", "failed", "invalid"}
    if status is not None and status not in allowed:
        raise HTTPException(status_code=400, detail="invalid ingest job status filter")
    return list_ingest_jobs(status=status, limit=limit)


@app.get("/ingest/jobs/{ingest_job_id}")
def get_ingest_job_endpoint(ingest_job_id: UUID) -> dict:
    try:
        return get_ingest_job(ingest_job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/calls")
def list_calls_endpoint(
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    tags: Optional[List[str]] = Query(None),
    external_id: Optional[str] = None,
    external_source: Optional[str] = None,
) -> dict:
    return list_calls(
        limit=limit,
        cursor=cursor,
        date_from=date_from,
        date_to=date_to,
        tags=tags,
        external_id=external_id,
        external_source=external_source,
    )


@app.get("/calls/{call_id}")
def get_call_endpoint(call_id: UUID) -> dict:
    return get_call(call_id)


@app.get("/chunks/{chunk_id}")
def get_chunk_endpoint(chunk_id: int) -> dict:
    return get_chunk(chunk_id)


@app.post("/expand")
def expand_endpoint(payload: ExpandRequest) -> dict:
    return expand_evidence(
        payload.evidence_id,
        window_ms=payload.window_ms,
        max_chars=payload.max_chars,
    )


@app.post("/retrieve")
def retrieve_endpoint(payload: RetrieveRequest) -> dict:
    return retrieve_evidence(payload)

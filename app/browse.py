from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text

from .db import engine


def _encode_cursor(started_at: datetime, call_id: UUID) -> str:
    raw = f"{started_at.isoformat()}|{call_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8")


def _decode_cursor(cursor: str) -> Tuple[datetime, UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8")
        started_at_raw, call_id_raw = raw.split("|", 1)
        return datetime.fromisoformat(started_at_raw), UUID(call_id_raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid cursor") from exc


def list_calls(
    *,
    limit: int,
    cursor: Optional[str],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
    tags: Optional[List[str]],
    external_id: Optional[str],
    external_source: Optional[str],
) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    clauses: List[str] = []
    params: Dict[str, Any] = {"limit": limit + 1}

    if date_from:
        clauses.append("started_at >= :date_from")
        params["date_from"] = date_from
    if date_to:
        clauses.append("started_at <= :date_to")
        params["date_to"] = date_to
    if tags:
        clauses.append("tags && :tags")
        params["tags"] = tags
    if external_id:
        clauses.append("external_id = :external_id")
        params["external_id"] = external_id
        if external_source is not None:
            clauses.append("external_source IS NOT DISTINCT FROM :external_source")
            params["external_source"] = external_source
    elif external_source:
        clauses.append("external_source = :external_source")
        params["external_source"] = external_source

    if cursor:
        cursor_started_at, cursor_call_id = _decode_cursor(cursor)
        clauses.append("(started_at, call_id) < (:cursor_started_at, :cursor_call_id)")
        params["cursor_started_at"] = cursor_started_at
        params["cursor_call_id"] = cursor_call_id

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    rows = []
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT call_id, started_at, ended_at, title, external_id, external_source,
                       source_uri, source_hash, tags, participants, metadata, created_at
                FROM calls
                WHERE {where_sql}
                ORDER BY started_at DESC, call_id DESC
                LIMIT :limit
                """
            ),
            params,
        ).mappings().all()

    next_cursor = None
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = _encode_cursor(last["started_at"], last["call_id"])
        rows = rows[:limit]

    items = [
        {
            "call_id": str(row["call_id"]),
            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
            "ended_at": row["ended_at"].isoformat() if row["ended_at"] else None,
            "title": row["title"],
            "external_id": row["external_id"],
            "external_source": row["external_source"],
            "source_uri": row["source_uri"],
            "source_hash": row["source_hash"],
            "tags": row["tags"] or [],
            "participants": row["participants"],
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]

    return {"items": items, "next_cursor": next_cursor}


def get_call(call_id: UUID) -> Dict[str, Any]:
    with engine.connect() as conn:
        call_row = conn.execute(
            text(
                """
                SELECT call_id, started_at, ended_at, title, external_id, external_source,
                       source_uri, source_hash, tags, participants, metadata, created_at
                FROM calls
                WHERE call_id = :call_id
                """
            ),
            {"call_id": call_id},
        ).mappings().fetchone()

        if not call_row:
            raise HTTPException(status_code=404, detail="call not found")

        counts = conn.execute(
            text(
                """
                SELECT
                  (SELECT count(*) FROM utterances WHERE call_id = :call_id) AS utterances,
                  (SELECT count(*) FROM chunks WHERE call_id = :call_id) AS chunks,
                  (SELECT count(*) FROM analysis_artifacts WHERE call_id = :call_id) AS artifacts
                """
            ),
            {"call_id": call_id},
        ).mappings().fetchone()

        artifacts = conn.execute(
            text(
                """
                SELECT artifact_id, kind, token_count, created_at
                FROM analysis_artifacts
                WHERE call_id = :call_id
                ORDER BY created_at ASC
                """
            ),
            {"call_id": call_id},
        ).mappings().all()

    return {
        "call": {
            "call_id": str(call_row["call_id"]),
            "started_at": call_row["started_at"].isoformat()
            if call_row["started_at"]
            else None,
            "ended_at": call_row["ended_at"].isoformat()
            if call_row["ended_at"]
            else None,
            "title": call_row["title"],
            "external_id": call_row["external_id"],
            "external_source": call_row["external_source"],
            "source_uri": call_row["source_uri"],
            "source_hash": call_row["source_hash"],
            "tags": call_row["tags"] or [],
            "participants": call_row["participants"],
            "metadata": call_row["metadata"],
            "created_at": call_row["created_at"].isoformat()
            if call_row["created_at"]
            else None,
        },
        "counts": dict(counts) if counts else {"utterances": 0, "chunks": 0, "artifacts": 0},
        "artifacts": [
            {
                "artifact_id": row["artifact_id"],
                "kind": row["kind"],
                "token_count": row["token_count"],
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
            }
            for row in artifacts
        ],
    }


def get_chunk(chunk_id: int) -> Dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT chunk_id, call_id, speaker, start_ts_ms, end_ts_ms,
                       token_count, text, tech_tokens
                FROM chunks
                WHERE chunk_id = :chunk_id
                """
            ),
            {"chunk_id": chunk_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="chunk not found")

    return {
        "chunk_id": row["chunk_id"],
        "call_id": str(row["call_id"]),
        "speaker": row["speaker"],
        "start_ts_ms": row["start_ts_ms"],
        "end_ts_ms": row["end_ts_ms"],
        "token_count": row["token_count"],
        "text": row["text"],
        "tech_tokens": row["tech_tokens"] or [],
    }


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_utterance(speaker: Optional[str], text: str) -> str:
    return f"{speaker}: {text}" if speaker else text


def expand_evidence(
    evidence_id: str, *, window_ms: Optional[int], max_chars: int
) -> Dict[str, Any]:
    if evidence_id.startswith("Q-"):
        chunk_id = int(evidence_id.split("-", 1)[1])
        with engine.connect() as conn:
            chunk = conn.execute(
                text(
                    """
                    SELECT chunk_id, call_id, start_ts_ms, end_ts_ms
                    FROM chunks
                    WHERE chunk_id = :chunk_id
                    """
                ),
                {"chunk_id": chunk_id},
            ).mappings().fetchone()
            if not chunk:
                raise HTTPException(status_code=404, detail="chunk not found")

            if window_ms and window_ms > 0:
                min_ts = chunk["start_ts_ms"] - window_ms
                max_ts = chunk["end_ts_ms"] + window_ms
                utterances = conn.execute(
                    text(
                        """
                        SELECT speaker, start_ts_ms, end_ts_ms, text
                        FROM utterances
                        WHERE call_id = :call_id
                          AND start_ts_ms <= :max_ts
                          AND end_ts_ms >= :min_ts
                        ORDER BY start_ts_ms ASC
                        """
                    ),
                    {"call_id": chunk["call_id"], "min_ts": min_ts, "max_ts": max_ts},
                ).mappings().all()
            else:
                utterances = conn.execute(
                    text(
                        """
                        SELECT u.speaker, u.start_ts_ms, u.end_ts_ms, u.text
                        FROM chunk_utterances cu
                        JOIN utterances u ON u.utterance_id = cu.utterance_id
                        WHERE cu.chunk_id = :chunk_id
                        ORDER BY cu.ordinal ASC
                        """
                    ),
                    {"chunk_id": chunk_id},
                ).mappings().all()

        if utterances:
            snippet = "\n".join(
                _format_utterance(u["speaker"], u["text"]) for u in utterances
            )
            start_ts_ms = utterances[0]["start_ts_ms"]
            end_ts_ms = utterances[-1]["end_ts_ms"]
        else:
            snippet = ""
            start_ts_ms = chunk["start_ts_ms"]
            end_ts_ms = chunk["end_ts_ms"]

        return {
            "evidence_id": evidence_id,
            "call_id": str(chunk["call_id"]),
            "chunk_id": chunk_id,
            "start_ts_ms": start_ts_ms,
            "end_ts_ms": end_ts_ms,
            "snippet": _clip(snippet, max_chars),
        }

    if evidence_id.startswith("A-"):
        artifact_chunk_id = int(evidence_id.split("-", 1)[1])
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT artifact_chunk_id, artifact_id, call_id, kind, content
                    FROM artifact_chunks
                    WHERE artifact_chunk_id = :artifact_chunk_id
                    """
                ),
                {"artifact_chunk_id": artifact_chunk_id},
            ).mappings().fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="artifact chunk not found")
        return {
            "evidence_id": evidence_id,
            "call_id": str(row["call_id"]),
            "artifact_id": row["artifact_id"],
            "artifact_chunk_id": row["artifact_chunk_id"],
            "kind": row["kind"],
            "snippet": _clip(row["content"], max_chars),
        }

    raise HTTPException(status_code=400, detail="unsupported evidence_id")

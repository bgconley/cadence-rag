from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from uuid import UUID, uuid4

from sqlalchemy import text

from .db import engine
from .ingest import extract_tech_tokens
from .schemas import Budget, RetrieveFilters, RetrieveRequest

DEFAULT_RRF_K = 60
DEFAULT_CHUNK_BM25_TOPK = 50
DEFAULT_ARTIFACT_CHUNK_BM25_TOPK = 10
DEFAULT_TECH_TOPK = 50
DEFAULT_MAX_ARTIFACTS = 2
DEFAULT_MAX_QUOTES_PER_CALL = 2
DEFAULT_SNIPPET_CHARS = 800


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _build_debug_lane(
    rows: Sequence[Dict[str, Any]], id_field: str
) -> List[Dict[str, Any]]:
    lane = []
    for rank, row in enumerate(rows, start=1):
        lane.append(
            {id_field: row[id_field], "rank": rank, "score": row.get("score")}
        )
    return lane


def _resolve_call_ids(
    conn, filters: Optional[RetrieveFilters]
) -> Optional[List[UUID]]:
    if not filters:
        return None

    call_ids: Optional[Set[UUID]] = set(filters.call_ids or [])
    if filters.external_id:
        if filters.external_source is None:
            rows = conn.execute(
                text(
                    """
                    SELECT call_id
                    FROM calls
                    WHERE external_id = :external_id
                    """
                ),
                {"external_id": filters.external_id},
            ).fetchall()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT call_id
                    FROM calls
                    WHERE external_id = :external_id
                      AND (external_source IS NOT DISTINCT FROM :external_source)
                    """
                ),
                {
                    "external_id": filters.external_id,
                    "external_source": filters.external_source,
                },
            ).fetchall()
        resolved = {row[0] for row in rows}
        if call_ids:
            call_ids &= resolved
        else:
            call_ids = resolved

    if call_ids is None:
        return None
    return list(call_ids)


def _build_filter_clause(
    filters: Optional[RetrieveFilters],
    alias: str,
    call_ids: Optional[Sequence[UUID]],
) -> Tuple[str, Dict[str, Any], bool]:
    clauses: List[str] = []
    params: Dict[str, Any] = {}
    join_calls = False

    if filters:
        if filters.date_from:
            clauses.append(f"{alias}.call_started_at >= :date_from")
            params["date_from"] = filters.date_from
        if filters.date_to:
            clauses.append(f"{alias}.call_started_at <= :date_to")
            params["date_to"] = filters.date_to
        if call_ids:
            clauses.append(f"{alias}.call_id = ANY(:call_ids)")
            params["call_ids"] = list(call_ids)
        if filters.call_tags:
            join_calls = True
            clauses.append("c.tags && :call_tags")
            params["call_tags"] = filters.call_tags

    if not clauses:
        return "TRUE", params, join_calls

    return " AND ".join(clauses), params, join_calls


def _fetch_chunks_bm25(
    conn, query: str, filters: Optional[RetrieveFilters], limit: int
) -> List[Dict[str, Any]]:
    call_ids = _resolve_call_ids(conn, filters)
    if call_ids == []:
        return []
    where_sql, params, join_calls = _build_filter_clause(filters, "chunks", call_ids)
    join_sql = "JOIN calls c ON c.call_id = chunks.call_id" if join_calls else ""
    params.update({"query": query, "limit": limit})
    rows = conn.execute(
        text(
            f"""
            SELECT chunk_id, call_id, speaker, start_ts_ms, end_ts_ms, text,
                   pdb.score(chunks) AS score
            FROM chunks
            {join_sql}
            WHERE {where_sql}
              AND chunks.text @@@ :query
            ORDER BY pdb.score(chunks) DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return list(rows)


def _fetch_artifacts_bm25(
    conn, query: str, filters: Optional[RetrieveFilters], limit: int
) -> List[Dict[str, Any]]:
    call_ids = _resolve_call_ids(conn, filters)
    if call_ids == []:
        return []
    where_sql, params, join_calls = _build_filter_clause(
        filters, "artifact_chunks", call_ids
    )
    join_sql = (
        "JOIN calls c ON c.call_id = artifact_chunks.call_id" if join_calls else ""
    )
    params.update({"query": query, "limit": limit})
    rows = conn.execute(
        text(
            f"""
            SELECT artifact_chunk_id, artifact_id, call_id, kind, content,
                   pdb.score(artifact_chunks) AS score
            FROM artifact_chunks
            {join_sql}
            WHERE {where_sql}
              AND artifact_chunks.content @@@ :query
            ORDER BY pdb.score(artifact_chunks) DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return list(rows)


def _fetch_chunks_tech(
    conn, tokens: Sequence[str], filters: Optional[RetrieveFilters], limit: int
) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    call_ids = _resolve_call_ids(conn, filters)
    if call_ids == []:
        return []
    where_sql, params, join_calls = _build_filter_clause(filters, "chunks", call_ids)
    join_sql = "JOIN calls c ON c.call_id = chunks.call_id" if join_calls else ""
    params.update({"tech_tokens": list(tokens), "limit": limit})
    rows = conn.execute(
        text(
            f"""
            SELECT chunk_id, call_id, speaker, start_ts_ms, end_ts_ms, text
            FROM chunks
            {join_sql}
            WHERE {where_sql}
              AND chunks.tech_tokens && :tech_tokens
            ORDER BY chunks.call_started_at DESC, chunks.chunk_id ASC
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return list(rows)


def _fetch_artifacts_tech(
    conn, tokens: Sequence[str], filters: Optional[RetrieveFilters], limit: int
) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    call_ids = _resolve_call_ids(conn, filters)
    if call_ids == []:
        return []
    where_sql, params, join_calls = _build_filter_clause(
        filters, "artifact_chunks", call_ids
    )
    join_sql = (
        "JOIN calls c ON c.call_id = artifact_chunks.call_id" if join_calls else ""
    )
    params.update({"tech_tokens": list(tokens), "limit": limit})
    rows = conn.execute(
        text(
            f"""
            SELECT artifact_chunk_id, artifact_id, call_id, kind, content
            FROM artifact_chunks
            {join_sql}
            WHERE {where_sql}
              AND artifact_chunks.tech_tokens && :tech_tokens
            ORDER BY artifact_chunks.call_started_at DESC, artifact_chunks.artifact_chunk_id ASC
            LIMIT :limit
            """
        ),
        params,
    ).mappings()
    return list(rows)


def _rrf_merge(
    lanes: Dict[str, Sequence[Dict[str, Any]]],
    key_field: str,
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[Dict[str, Any], Set[str], float]]:
    scores: Dict[Any, float] = {}
    items: Dict[Any, Dict[str, Any]] = {}
    lane_hits: Dict[Any, Set[str]] = {}
    for lane_name, rows in lanes.items():
        for rank, row in enumerate(rows, start=1):
            key = row[key_field]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, row)
            lane_hits.setdefault(key, set()).add(lane_name)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(items[key], lane_hits[key], score) for key, score in ordered]


def retrieve_evidence(payload: RetrieveRequest) -> Dict[str, Any]:
    query_id = str(uuid4())
    query = payload.query.strip()
    budget = payload.budget or Budget()
    return_style = payload.return_style

    if not query:
        if return_style == "ids_only":
            return {"query_id": query_id, "retrieved_ids": []}
        return {
            "query_id": query_id,
            "intent": payload.intent,
            "budget": budget.model_dump(),
            "artifacts": [],
            "quotes": [],
            "notes": {"error": "empty query"},
        }

    filters = payload.filters
    tech_tokens = extract_tech_tokens(query)

    with engine.connect() as conn:
        bm25_chunks = _fetch_chunks_bm25(
            conn, query, filters, DEFAULT_CHUNK_BM25_TOPK
        )
        bm25_artifacts = _fetch_artifacts_bm25(
            conn, query, filters, DEFAULT_ARTIFACT_CHUNK_BM25_TOPK
        )
        tech_chunks = _fetch_chunks_tech(conn, tech_tokens, filters, DEFAULT_TECH_TOPK)
        tech_artifacts = _fetch_artifacts_tech(
            conn, tech_tokens, filters, DEFAULT_TECH_TOPK
        )

    debug_payload = None
    if payload.debug:
        debug_payload = {
            "lanes": {
                "chunks": {
                    "bm25": _build_debug_lane(bm25_chunks, "chunk_id"),
                    "tech_tokens": _build_debug_lane(tech_chunks, "chunk_id"),
                },
                "artifacts": {
                    "bm25": _build_debug_lane(bm25_artifacts, "artifact_chunk_id"),
                    "tech_tokens": _build_debug_lane(
                        tech_artifacts, "artifact_chunk_id"
                    ),
                },
            },
            "limits": {
                "bm25_chunk_topk": DEFAULT_CHUNK_BM25_TOPK,
                "bm25_artifact_chunk_topk": DEFAULT_ARTIFACT_CHUNK_BM25_TOPK,
                "tech_token_topk": DEFAULT_TECH_TOPK,
            },
        }

    chunk_ranked = _rrf_merge(
        {"bm25": bm25_chunks, "tech_tokens": tech_chunks}, "chunk_id"
    )
    artifact_ranked = _rrf_merge(
        {"bm25": bm25_artifacts, "tech_tokens": tech_artifacts}, "artifact_chunk_id"
    )

    if return_style == "ids_only":
        combined: List[Tuple[str, int, float]] = []
        for row, _lanes, score in artifact_ranked:
            combined.append(("artifact_chunk", row["artifact_chunk_id"], score))
        for row, _lanes, score in chunk_ranked:
            combined.append(("chunk", row["chunk_id"], score))
        kind_order = {"artifact_chunk": 0, "chunk": 1}
        combined.sort(key=lambda item: (-item[2], kind_order[item[0]], item[1]))
        retrieved_ids = [f"{kind}:{item_id}" for kind, item_id, _ in combined]
        response: Dict[str, Any] = {
            "query_id": query_id,
            "retrieved_ids": retrieved_ids,
        }
        if debug_payload is not None:
            response["debug"] = debug_payload
        return response

    max_items = budget.max_evidence_items
    remaining_chars = budget.max_total_chars

    artifacts_out: List[Dict[str, Any]] = []
    quotes_out: List[Dict[str, Any]] = []

    max_artifacts = min(DEFAULT_MAX_ARTIFACTS, max_items)
    evidence_count = 0

    for row, lanes, _score in artifact_ranked:
        if evidence_count >= max_items or len(artifacts_out) >= max_artifacts:
            break
        if remaining_chars <= 0:
            break
        snippet = _clip(row["content"], min(DEFAULT_SNIPPET_CHARS, remaining_chars))
        remaining_chars -= len(snippet)
        artifacts_out.append(
            {
                "evidence_id": f"A-{row['artifact_chunk_id']}",
                "call_id": str(row["call_id"]),
                "artifact_id": row["artifact_id"],
                "artifact_chunk_id": row["artifact_chunk_id"],
                "kind": row["kind"],
                "snippet": snippet,
                "why_relevant": " + ".join(sorted(lanes)),
            }
        )
        evidence_count += 1

    quotes_per_call: Dict[str, int] = {}
    for row, lanes, _score in chunk_ranked:
        if evidence_count >= max_items:
            break
        if remaining_chars <= 0:
            break
        call_id = str(row["call_id"])
        if quotes_per_call.get(call_id, 0) >= DEFAULT_MAX_QUOTES_PER_CALL:
            continue
        snippet = _clip(row["text"], min(DEFAULT_SNIPPET_CHARS, remaining_chars))
        remaining_chars -= len(snippet)
        quotes_out.append(
            {
                "evidence_id": f"Q-{row['chunk_id']}",
                "call_id": call_id,
                "chunk_id": row["chunk_id"],
                "speaker": row["speaker"],
                "start_ts_ms": row["start_ts_ms"],
                "end_ts_ms": row["end_ts_ms"],
                "snippet": snippet,
                "why_relevant": " + ".join(sorted(lanes)),
            }
        )
        quotes_per_call[call_id] = quotes_per_call.get(call_id, 0) + 1
        evidence_count += 1

    response: Dict[str, Any] = {
        "query_id": query_id,
        "intent": payload.intent,
        "budget": budget.model_dump(),
        "artifacts": artifacts_out,
        "quotes": quotes_out,
        "notes": {
            "retrieval": {
                "planner": "lexical_only",
                "dense_topk": 0,
                "lex_topk": DEFAULT_CHUNK_BM25_TOPK,
                "artifact_chunk_lex_topk": DEFAULT_ARTIFACT_CHUNK_BM25_TOPK,
                "reranked_from": None,
                "bm25_chunk_topk": DEFAULT_CHUNK_BM25_TOPK,
                "bm25_artifact_chunk_topk": DEFAULT_ARTIFACT_CHUNK_BM25_TOPK,
                "tech_token_topk": DEFAULT_TECH_TOPK,
                "tech_tokens": tech_tokens,
                "lanes": {"bm25": True, "tech_tokens": True, "dense": False},
            }
        },
    }
    if debug_payload is not None:
        response["debug"] = debug_payload
    return response

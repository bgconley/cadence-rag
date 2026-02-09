from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set
from uuid import UUID

from sqlalchemy import text

from .config import settings
from .db import engine
from .embeddings import EmbeddingClientError, EmbeddingResult, embed_texts, embeddings_enabled
from .ingest import NER_CONFIG_DISABLED, PIPELINE_VERSION


@dataclass(frozen=True)
class PendingRow:
    row_id: int
    call_id: UUID
    content: str


@dataclass(frozen=True)
class TableSpec:
    table: str
    id_column: str
    text_column: str
    order_column: str


@dataclass(frozen=True)
class BackfillSummary:
    rows_updated: int
    calls_touched: int
    ingestion_runs_inserted: int
    model_used: str
    per_table: Dict[str, int]


TABLE_SPECS: Sequence[TableSpec] = (
    TableSpec(
        table="chunks",
        id_column="chunk_id",
        text_column="text",
        order_column="chunk_id",
    ),
    TableSpec(
        table="artifact_chunks",
        id_column="artifact_chunk_id",
        text_column="content",
        order_column="artifact_chunk_id",
    ),
)

_BATCH_SIZE_LIMIT_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"batch[- ]size[^0-9]{0,40}<=\s*(\d+)", re.IGNORECASE),
    re.compile(r"max(?:imum)?\s+batch[- ]size[^0-9]{0,40}(\d+)", re.IGNORECASE),
)


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".10g") for value in values) + "]"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_batch_size_limit(error_message: str) -> Optional[int]:
    message = (error_message or "").strip()
    if not message:
        return None
    for pattern in _BATCH_SIZE_LIMIT_PATTERNS:
        match = pattern.search(message)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _embed_texts_adaptive(texts: Sequence[str], batch_size: int) -> EmbeddingResult:
    cleaned = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        raise EmbeddingClientError("embedding request requires at least one non-empty text")

    current_batch = max(1, int(batch_size))
    vectors: List[List[float]] = []
    model_used = settings.embeddings_model_id
    index = 0

    while index < len(cleaned):
        upper = min(index + current_batch, len(cleaned))
        chunk = cleaned[index:upper]
        try:
            result = embed_texts(chunk)
        except EmbeddingClientError as exc:
            # Retry with smaller batches when provider enforces max batch limits.
            if len(chunk) <= 1:
                raise
            inferred = infer_batch_size_limit(str(exc))
            if inferred is not None and inferred < len(chunk):
                current_batch = max(1, inferred)
            else:
                current_batch = max(1, len(chunk) // 2)
            continue

        vectors.extend(result.vectors)
        model_used = result.model
        index = upper

    return EmbeddingResult(vectors=vectors, model=model_used)


def _fetch_pending_rows(spec: TableSpec, limit: int, call_id: Optional[UUID]) -> List[PendingRow]:
    call_filter = "AND call_id = :call_id" if call_id is not None else ""
    params = {"limit": limit}
    if call_id is not None:
        params["call_id"] = call_id

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT {spec.id_column} AS row_id, call_id, {spec.text_column} AS content
                FROM {spec.table}
                WHERE embedding IS NULL
                  AND {spec.text_column} IS NOT NULL
                  AND length(trim({spec.text_column})) > 0
                  {call_filter}
                ORDER BY {spec.order_column} ASC
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
        return [
            PendingRow(row_id=row["row_id"], call_id=row["call_id"], content=row["content"])
            for row in rows
        ]


def _update_embeddings(
    spec: TableSpec, rows: Sequence[PendingRow], vectors: Sequence[Sequence[float]]
) -> None:
    if len(rows) != len(vectors):
        raise RuntimeError(
            f"row/vector mismatch for {spec.table}: {len(rows)} rows vs {len(vectors)} vectors"
        )
    vector_dim = max(1, int(settings.embeddings_dim))
    with engine.begin() as conn:
        for row, vector in zip(rows, vectors):
            conn.execute(
                text(
                    f"""
                    UPDATE {spec.table}
                    SET embedding = CAST(:embedding AS vector({vector_dim}))
                    WHERE {spec.id_column} = :row_id
                    """
                ),
                {"row_id": row.row_id, "embedding": _vector_literal(vector)},
            )


def _record_embedding_runs(call_ids: Iterable[UUID], model_id: str, dim: int, source: str) -> int:
    inserted = 0
    chunking_config = json.dumps(
        {
            "enabled": True,
            "mode": "existing_chunks",
            "source": source,
        }
    )
    embedding_config = json.dumps(
        {
            "enabled": True,
            "mode": "http_backfill_v1",
            "model_id": model_id,
            "dim": dim,
            "base_url": settings.embeddings_base_url,
            "timestamp": _now_utc_iso(),
            "source": source,
        }
    )
    ner_config = json.dumps(NER_CONFIG_DISABLED)

    with engine.begin() as conn:
        for call_id in sorted(set(call_ids), key=str):
            conn.execute(
                text(
                    """
                    INSERT INTO ingestion_runs
                      (call_id, pipeline_version, chunking_config, embedding_config, ner_config)
                    VALUES
                      (:call_id, :pipeline_version, CAST(:chunking_config AS jsonb),
                       CAST(:embedding_config AS jsonb), CAST(:ner_config AS jsonb))
                    """
                ),
                {
                    "call_id": call_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "chunking_config": chunking_config,
                    "embedding_config": embedding_config,
                    "ner_config": ner_config,
                },
            )
            inserted += 1
    return inserted


def _backfill_table(
    spec: TableSpec,
    *,
    batch_size: int,
    call_id: Optional[UUID],
) -> tuple[int, Set[UUID], str]:
    updated = 0
    touched_calls: Set[UUID] = set()
    model_used = settings.embeddings_model_id

    while True:
        batch = _fetch_pending_rows(spec, batch_size, call_id=call_id)
        if not batch:
            break
        texts = [row.content for row in batch]
        result = _embed_texts_adaptive(texts, batch_size=batch_size)
        _update_embeddings(spec, batch, result.vectors)
        touched_calls.update(row.call_id for row in batch)
        updated += len(batch)
        model_used = result.model

    return updated, touched_calls, model_used


def run_embedding_backfill(
    *,
    batch_size: int,
    call_id: Optional[UUID] = None,
    source: str = "embed_backfill",
) -> BackfillSummary:
    if not embeddings_enabled():
        raise RuntimeError("EMBEDDINGS_BASE_URL must be set to run embedding backfill")
    if settings.embeddings_dim <= 0:
        raise RuntimeError("EMBEDDINGS_DIM must be > 0")
    if batch_size <= 0:
        raise RuntimeError("EMBEDDINGS_BATCH_SIZE must be > 0")

    total_updated = 0
    all_calls: Set[UUID] = set()
    model_used = settings.embeddings_model_id
    per_table: Dict[str, int] = {}

    for spec in TABLE_SPECS:
        updated, touched_calls, model_from_table = _backfill_table(
            spec,
            batch_size=batch_size,
            call_id=call_id,
        )
        per_table[spec.table] = updated
        total_updated += updated
        all_calls.update(touched_calls)
        model_used = model_from_table

    ingestion_rows = _record_embedding_runs(
        call_ids=all_calls,
        model_id=model_used,
        dim=settings.embeddings_dim,
        source=source,
    )
    return BackfillSummary(
        rows_updated=total_updated,
        calls_touched=len(all_calls),
        ingestion_runs_inserted=ingestion_rows,
        model_used=model_used,
        per_table=per_table,
    )

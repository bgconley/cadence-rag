from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Set
from uuid import UUID

from sqlalchemy import text

from app.config import settings
from app.db import engine
from app.embeddings import (
    EmbeddingClientError,
    embed_texts_batched,
    embeddings_enabled,
)
from app.ingest import NER_CONFIG_DISABLED, PIPELINE_VERSION


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


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".10g") for value in values) + "]"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_pending_rows(spec: TableSpec, limit: int) -> List[PendingRow]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT {spec.id_column} AS row_id, call_id, {spec.text_column} AS content
                FROM {spec.table}
                WHERE embedding IS NULL
                  AND {spec.text_column} IS NOT NULL
                  AND length(trim({spec.text_column})) > 0
                ORDER BY {spec.order_column} ASC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings()
        return [
            PendingRow(row_id=row["row_id"], call_id=row["call_id"], content=row["content"])
            for row in rows
        ]


def _update_embeddings(spec: TableSpec, rows: Sequence[PendingRow], vectors: Sequence[Sequence[float]]) -> None:
    if len(rows) != len(vectors):
        raise RuntimeError(
            f"row/vector mismatch for {spec.table}: {len(rows)} rows vs {len(vectors)} vectors"
        )
    with engine.begin() as conn:
        for row, vector in zip(rows, vectors):
            conn.execute(
                text(
                    f"""
                    UPDATE {spec.table}
                    SET embedding = CAST(:embedding AS vector(1024))
                    WHERE {spec.id_column} = :row_id
                    """
                ),
                {"row_id": row.row_id, "embedding": _vector_literal(vector)},
            )


def _record_embedding_runs(call_ids: Iterable[UUID], model_id: str, dim: int) -> int:
    inserted = 0
    chunking_config = json.dumps(
        {
            "enabled": True,
            "mode": "existing_chunks",
            "source": "embed_backfill",
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


def _backfill_table(spec: TableSpec, batch_size: int) -> tuple[int, Set[UUID], str]:
    updated = 0
    touched_calls: Set[UUID] = set()
    model_used = settings.embeddings_model_id

    while True:
        batch = _fetch_pending_rows(spec, batch_size)
        if not batch:
            break
        texts = [row.content for row in batch]
        result = embed_texts_batched(texts, batch_size=batch_size)
        _update_embeddings(spec, batch, result.vectors)
        touched_calls.update(row.call_id for row in batch)
        updated += len(batch)
        model_used = result.model
        print(
            f"[embed_backfill] table={spec.table} batch_updated={len(batch)} total_updated={updated}"
        )

    return updated, touched_calls, model_used


def main() -> None:
    if not embeddings_enabled():
        raise RuntimeError("EMBEDDINGS_BASE_URL must be set to run embed_backfill")
    if settings.embeddings_dim != 1024:
        raise RuntimeError("EMBEDDINGS_DIM must be 1024 for this schema")

    batch_size = settings.embeddings_batch_size
    if batch_size <= 0:
        raise RuntimeError("EMBEDDINGS_BATCH_SIZE must be > 0")

    total_updated = 0
    all_calls: Set[UUID] = set()
    model_used = settings.embeddings_model_id

    for spec in TABLE_SPECS:
        updated, touched_calls, model_from_table = _backfill_table(spec, batch_size)
        total_updated += updated
        all_calls.update(touched_calls)
        model_used = model_from_table
        print(
            f"[embed_backfill] finished table={spec.table} updated={updated} calls={len(touched_calls)}"
        )

    ingestion_rows = _record_embedding_runs(
        call_ids=all_calls, model_id=model_used, dim=settings.embeddings_dim
    )
    print(
        "[embed_backfill] complete "
        f"rows_updated={total_updated} calls_touched={len(all_calls)} "
        f"ingestion_runs_inserted={ingestion_rows} model={model_used}"
    )


if __name__ == "__main__":
    try:
        main()
    except EmbeddingClientError as exc:
        raise SystemExit(f"embed_backfill failed: {exc}") from exc

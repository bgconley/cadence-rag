from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Set
from uuid import UUID

from sqlalchemy import text

from app.db import engine
from app.ingest import (
    EMBEDDING_CONFIG_DISABLED,
    NER_CONFIG_DISABLED,
    PIPELINE_VERSION,
    extract_tech_tokens,
)


@dataclass(frozen=True)
class TableSpec:
    table: str
    id_column: str
    text_column: str


@dataclass(frozen=True)
class PendingRow:
    row_id: int
    call_id: UUID
    content: str
    tech_tokens: List[str]


TABLE_SPECS: Sequence[TableSpec] = (
    TableSpec(table="chunks", id_column="chunk_id", text_column="text"),
    TableSpec(
        table="artifact_chunks",
        id_column="artifact_chunk_id",
        text_column="content",
    ),
    TableSpec(
        table="analysis_artifacts",
        id_column="artifact_id",
        text_column="content",
    ),
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_rows(spec: TableSpec, batch_size: int, after_id: int) -> List[PendingRow]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT
                  {spec.id_column} AS row_id,
                  call_id,
                  {spec.text_column} AS content,
                  tech_tokens
                FROM {spec.table}
                WHERE {spec.id_column} > :after_id
                  AND {spec.text_column} IS NOT NULL
                  AND length(trim({spec.text_column})) > 0
                ORDER BY {spec.id_column} ASC
                LIMIT :limit
                """
            ),
            {"after_id": after_id, "limit": batch_size},
        ).mappings()
        return [
            PendingRow(
                row_id=row["row_id"],
                call_id=row["call_id"],
                content=row["content"],
                tech_tokens=list(row["tech_tokens"] or []),
            )
            for row in rows
        ]


def _update_tokens(spec: TableSpec, row_id: int, tech_tokens: Sequence[str]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                UPDATE {spec.table}
                SET tech_tokens = :tech_tokens
                WHERE {spec.id_column} = :row_id
                """
            ),
            {"row_id": row_id, "tech_tokens": list(tech_tokens)},
        )


def _record_ingestion_runs(call_ids: Iterable[UUID], source: str) -> int:
    inserted = 0
    chunking_config = json.dumps(
        {
            "enabled": True,
            "mode": "existing_chunks",
            "source": source,
            "timestamp": _now_utc_iso(),
        }
    )
    embedding_config = json.dumps(EMBEDDING_CONFIG_DISABLED)
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
    batch_size: int,
    dry_run: bool,
) -> tuple[int, int, Set[UUID]]:
    scanned = 0
    updated = 0
    touched_calls: Set[UUID] = set()
    after_id = 0

    while True:
        rows = _fetch_rows(spec, batch_size=batch_size, after_id=after_id)
        if not rows:
            break
        scanned += len(rows)
        after_id = rows[-1].row_id

        for row in rows:
            new_tokens = extract_tech_tokens(row.content)
            if new_tokens == row.tech_tokens:
                continue
            if not dry_run:
                _update_tokens(spec, row.row_id, new_tokens)
            updated += 1
            touched_calls.add(row.call_id)

        print(
            f"[tech_tokens_backfill] table={spec.table} scanned={scanned} updated={updated}"
        )

    return scanned, updated, touched_calls


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute tech_tokens across existing rows after token-rule updates."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows fetched per batch (default: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report updates without writing to the database.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    total_scanned = 0
    total_updated = 0
    all_calls: Set[UUID] = set()

    for spec in TABLE_SPECS:
        scanned, updated, touched_calls = _backfill_table(
            spec=spec, batch_size=args.batch_size, dry_run=args.dry_run
        )
        total_scanned += scanned
        total_updated += updated
        all_calls.update(touched_calls)
        print(
            f"[tech_tokens_backfill] finished table={spec.table} scanned={scanned} updated={updated}"
        )

    ingestion_rows = 0
    if not args.dry_run and all_calls:
        ingestion_rows = _record_ingestion_runs(
            call_ids=all_calls, source="tech_tokens_backfill_v1"
        )

    print(
        "[tech_tokens_backfill] complete "
        f"scanned={total_scanned} updated={total_updated} "
        f"calls_touched={len(all_calls)} ingestion_runs_inserted={ingestion_rows} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()

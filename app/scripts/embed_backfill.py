from __future__ import annotations

from app.config import settings
from app.embedding_pipeline import run_embedding_backfill
from app.embeddings import EmbeddingClientError


def main() -> None:
    summary = run_embedding_backfill(
        batch_size=settings.embeddings_batch_size,
        source="embed_backfill",
    )
    for table_name, updated in summary.per_table.items():
        print(
            f"[embed_backfill] finished table={table_name} updated={updated}"
        )
    print(
        "[embed_backfill] complete "
        f"rows_updated={summary.rows_updated} "
        f"calls_touched={summary.calls_touched} "
        f"ingestion_runs_inserted={summary.ingestion_runs_inserted} "
        f"model={summary.model_used}"
    )


if __name__ == "__main__":
    try:
        main()
    except (EmbeddingClientError, RuntimeError) as exc:
        raise SystemExit(f"embed_backfill failed: {exc}") from exc

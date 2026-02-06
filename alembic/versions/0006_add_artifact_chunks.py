"""add artifact_chunks table and indexes

Revision ID: 0006_add_artifact_chunks
Revises: 0005_add_bm25_ngram
Create Date: 2026-02-06

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_add_artifact_chunks"
down_revision = "0005_add_bm25_ngram"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_chunks (
          artifact_chunk_id BIGSERIAL PRIMARY KEY,
          artifact_id       BIGINT NOT NULL REFERENCES analysis_artifacts(artifact_id) ON DELETE CASCADE,
          call_id           UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
          corpus_id         UUID REFERENCES corpora(corpus_id) ON DELETE SET NULL,
          call_started_at   TIMESTAMPTZ NOT NULL,
          kind              TEXT NOT NULL,
          ordinal           INT NOT NULL,
          content           TEXT NOT NULL,
          token_count       INT NOT NULL,
          start_char        INT,
          end_char          INT,
          embedding         vector(1024),
          tech_tokens       TEXT[] NOT NULL DEFAULT '{}',
          metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (artifact_id, ordinal)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_artifact_ordinal_idx "
        "ON artifact_chunks (artifact_id, ordinal);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_call_kind_idx "
        "ON artifact_chunks (call_id, kind);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_started_at_idx "
        "ON artifact_chunks (call_started_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_meta_gin "
        "ON artifact_chunks USING GIN (metadata);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_tech_tokens_gin "
        "ON artifact_chunks USING GIN (tech_tokens);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_embedding_hnsw "
        "ON artifact_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifact_chunks_bm25_idx "
        "ON artifact_chunks USING bm25 (artifact_chunk_id, content, "
        "(content::pdb.ngram(3,3, 'alias=artifact_chunk_content_ngram'))) "
        "WITH (key_field='artifact_chunk_id');"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS artifact_chunks_bm25_idx;")
    op.execute("DROP TABLE IF EXISTS artifact_chunks;")

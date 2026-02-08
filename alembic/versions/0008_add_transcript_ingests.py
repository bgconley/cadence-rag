"""add transcript ingest dedupe table

Revision ID: 0008_add_transcript_ingests
Revises: 0007_add_ingest_jobs
Create Date: 2026-02-07

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0008_add_transcript_ingests"
down_revision = "0007_add_ingest_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_ingests (
          transcript_ingest_id BIGSERIAL PRIMARY KEY,
          call_id              UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
          transcript_hash      TEXT NOT NULL,
          utterance_count      INT NOT NULL DEFAULT 0,
          chunk_count          INT NOT NULL DEFAULT 0,
          created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (call_id, transcript_hash)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS transcript_ingests_call_idx "
        "ON transcript_ingests (call_id, created_at DESC);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS transcript_ingests;")

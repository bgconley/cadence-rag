"""add ingest job tracking tables

Revision ID: 0007_add_ingest_jobs
Revises: 0006_add_artifact_chunks
Create Date: 2026-02-07

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_add_ingest_jobs"
down_revision = "0006_add_artifact_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_jobs (
          ingest_job_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          bundle_id       TEXT NOT NULL UNIQUE,
          status          TEXT NOT NULL,
          queue_name      TEXT NOT NULL,
          source_path     TEXT NOT NULL,
          manifest_path   TEXT NOT NULL,
          call_ref        JSONB NOT NULL DEFAULT '{}'::jsonb,
          call_id         UUID REFERENCES calls(call_id) ON DELETE SET NULL,
          error           TEXT,
          attempts        INT NOT NULL DEFAULT 0,
          created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
          started_at      TIMESTAMPTZ,
          completed_at    TIMESTAMPTZ,
          CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'invalid'))
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ingest_jobs_status_created_idx "
        "ON ingest_jobs (status, created_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ingest_jobs_call_idx "
        "ON ingest_jobs (call_id, created_at DESC);"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_job_files (
          ingest_job_file_id BIGSERIAL PRIMARY KEY,
          ingest_job_id      UUID NOT NULL REFERENCES ingest_jobs(ingest_job_id) ON DELETE CASCADE,
          kind               TEXT NOT NULL,
          relative_path      TEXT NOT NULL,
          file_sha256        TEXT NOT NULL,
          file_size_bytes    BIGINT NOT NULL,
          created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (ingest_job_id, relative_path)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ingest_job_files_job_idx "
        "ON ingest_job_files (ingest_job_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ingest_job_files;")
    op.execute("DROP TABLE IF EXISTS ingest_jobs;")

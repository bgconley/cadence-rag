"""add bm25 indexes

Revision ID: 0003_add_bm25_indexes
Revises: 0002_add_external_id_columns
Create Date: 2026-02-03
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_add_bm25_indexes"
down_revision = "0002_add_external_id_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_bm25_idx "
        "ON chunks USING bm25 (chunk_id, text) "
        "WITH (key_field='chunk_id');"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_bm25_idx "
        "ON analysis_artifacts USING bm25 (artifact_id, content) "
        "WITH (key_field='artifact_id');"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_bm25_idx;")
    op.execute("DROP INDEX IF EXISTS artifacts_bm25_idx;")

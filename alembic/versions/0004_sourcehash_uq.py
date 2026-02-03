"""add source_uri/source_hash unique index

Revision ID: 0004_sourcehash_uq
Revises: 0003_add_bm25_indexes
Create Date: 2026-02-03
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "0004_sourcehash_uq"
down_revision = "0003_add_bm25_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS calls_source_uri_hash_uq "
        "ON calls (source_uri, source_hash) "
        "WHERE source_uri IS NOT NULL AND source_hash IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS calls_source_uri_hash_uq;")

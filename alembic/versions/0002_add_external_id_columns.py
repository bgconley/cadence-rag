"""add external_id columns

Revision ID: 0002_add_external_id_columns
Revises: 0001_initial_schema
Create Date: 2026-02-03

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "0002_add_external_id_columns"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS external_id TEXT;")
    op.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS external_source TEXT;")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS calls_external_id_uq "
        "ON calls (external_source, external_id) "
        "WHERE external_id IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS calls_external_id_uq;")
    op.execute("ALTER TABLE calls DROP COLUMN IF EXISTS external_source;")
    op.execute("ALTER TABLE calls DROP COLUMN IF EXISTS external_id;")

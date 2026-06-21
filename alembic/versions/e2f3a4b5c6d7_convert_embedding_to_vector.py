"""convert embedding to pgvector type

Revision ID: e2f3a4b5c6d7
Revises: d1a2b3c4d5e6
Create Date: 2026-06-21
"""

from alembic import op

revision = "e2f3a4b5c6d7"
down_revision = "d1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE jobs ADD COLUMN embedding vector(1024)")
    op.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS embedding vector(1024)")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE jobs ADD COLUMN embedding TEXT")
    op.execute("ALTER TABLE resumes DROP COLUMN IF EXISTS embedding")

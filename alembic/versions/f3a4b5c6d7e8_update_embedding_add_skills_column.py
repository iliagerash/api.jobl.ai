"""update embedding to vector, drop embedded_at, add skills columns

Revision ID: f3a4b5c6d7e8
Revises: d1a2b3c4d5e6
Create Date: 2026-06-21
"""

from alembic import op
import sqlalchemy as sa

revision = "f3a4b5c6d7e8"
down_revision = "d1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Jobs: convert embedding TEXT → vector(1024), drop embedded_at, add skills
    # IF EXISTS needed because partial migration runs may have already dropped some columns
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS embedding CASCADE")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS embedded_at CASCADE")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedding vector(1024)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS skills TEXT")

    # Resumes: add embedding + skills
    op.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS embedding vector(1024)")
    op.execute("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS skills TEXT")


def downgrade() -> None:
    # Jobs: restore original columns
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS embedding")
    op.add_column("jobs", sa.Column("embedding", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))
    op.drop_column("jobs", "skills")

    # Resumes: remove new columns
    op.execute("ALTER TABLE resumes DROP COLUMN IF EXISTS embedding")
    op.drop_column("resumes", "skills")

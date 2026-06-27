"""add intelligence columns to jobs (category_id, destination, destination_job_id)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-27
"""
from alembic import op
import sqlalchemy as sa


revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS category_id SMALLINT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS destination VARCHAR(100)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS destination_job_id BIGINT")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_category_id ON jobs (category_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_destination ON jobs (destination, destination_job_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_jobs_destination")
    op.execute("DROP INDEX IF EXISTS idx_jobs_category_id")
    op.drop_column("jobs", "destination_job_id")
    op.drop_column("jobs", "destination")
    op.drop_column("jobs", "category_id")

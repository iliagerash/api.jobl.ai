"""add prediction columns to jobs

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa


revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS predicted_salary_min NUMERIC(15, 2)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS predicted_salary_max NUMERIC(15, 2)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS predicted_salary_period VARCHAR(10)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS predicted_revenue NUMERIC(10, 6)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS predicted_rps NUMERIC(10, 6)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS priority_tier VARCHAR(10)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS keyword_id BIGINT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS keyword_title VARCHAR(255)")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS keyword_distance NUMERIC(6, 4)")


def downgrade() -> None:
    op.drop_column("jobs", "keyword_distance")
    op.drop_column("jobs", "keyword_title")
    op.drop_column("jobs", "keyword_id")
    op.drop_column("jobs", "priority_tier")
    op.drop_column("jobs", "predicted_rps")
    op.drop_column("jobs", "predicted_revenue")
    op.drop_column("jobs", "predicted_salary_period")
    op.drop_column("jobs", "predicted_salary_max")
    op.drop_column("jobs", "predicted_salary_min")

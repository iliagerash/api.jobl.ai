"""add language_code to jobs

Revision ID: 6b51f1ce3b6d
Revises: 4df52e56f2ef
Create Date: 2026-03-08 17:10:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6b51f1ce3b6d"
down_revision = "4df52e56f2ef"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("language_code", sa.CHAR(length=2), nullable=True))
    op.create_index("idx_jobs_language_code", "jobs", ["language_code"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_jobs_language_code", table_name="jobs")
    op.drop_column("jobs", "language_code")

"""add normalization backlog index

Revision ID: 2f40abe9cf1a
Revises: 8fbdbc0245c6
Create Date: 2026-03-08 15:09:35.825984
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2f40abe9cf1a'
down_revision = '8fbdbc0245c6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_jobs_normalize_backlog",
        "jobs",
        ["id"],
        unique=False,
        postgresql_where=sa.text(
            "title_normalized IS NULL OR description_clean IS NULL OR description_html IS NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index("idx_jobs_normalize_backlog", table_name="jobs")

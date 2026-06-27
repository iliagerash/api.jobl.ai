"""add keywords and job_keywords tables

Revision ID: a1b2c3d4e5f6
Revises: f3a4b5c6d7e8
Create Date: 2026-06-27
"""
from alembic import op
import sqlalchemy as sa


revision = "a1b2c3d4e5f6"
down_revision = "f3a4b5c6d7e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "keywords",
        sa.Column("id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("canonical_id", sa.BIGINT(), nullable=False),
        sa.Column("title", sa.VARCHAR(255), nullable=False),
        sa.Column("language_code", sa.CHAR(2), nullable=False),
        sa.Column("category_id", sa.BIGINT(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("title", "language_code", "category_id", name="uq_keywords_title_lang_cat"),
    )
    op.execute("ALTER TABLE keywords ADD COLUMN embedding vector(1024)")
    op.execute("CREATE INDEX idx_keywords_canonical ON keywords (canonical_id)")
    op.execute("CREATE INDEX idx_keywords_cat_lang ON keywords (category_id, language_code)")

    op.create_table(
        "job_keywords",
        sa.Column("job_id", sa.BIGINT(), nullable=False),
        sa.Column("keyword_id", sa.BIGINT(), nullable=False),
        sa.Column("distance", sa.NUMERIC(6, 4), nullable=True),
        sa.PrimaryKeyConstraint("job_id", "keyword_id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["keyword_id"], ["keywords.id"], ondelete="CASCADE"),
    )
    op.execute("CREATE INDEX idx_job_keywords_keyword ON job_keywords (keyword_id)")


def downgrade() -> None:
    op.drop_table("job_keywords")
    op.drop_table("keywords")

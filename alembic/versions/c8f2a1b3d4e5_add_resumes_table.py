"""add resumes table

Revision ID: c8f2a1b3d4e5
Revises: 6643976ba29a
Create Date: 2026-06-03 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = "c8f2a1b3d4e5"
down_revision = "6643976ba29a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resumes",
        sa.Column("id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("source_website", sa.VARCHAR(length=128), nullable=False),
        sa.Column("external_id", sa.BIGINT(), nullable=False),
        sa.Column("title", sa.VARCHAR(length=255), nullable=False),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("city_title", sa.VARCHAR(length=255), nullable=True),
        sa.Column("region_title", sa.VARCHAR(length=255), nullable=True),
        sa.Column("country_code", sa.CHAR(length=2), nullable=True),
        sa.Column("salary", sa.NUMERIC(precision=15, scale=2), nullable=True),
        sa.Column("salary_period", sa.VARCHAR(length=20), nullable=True),
        sa.Column("salary_currency", sa.CHAR(length=3), nullable=True),
        sa.Column("contract", sa.VARCHAR(length=24), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_remote", sa.BOOLEAN(), server_default="false", nullable=False),
        sa.Column("language_code", sa.CHAR(length=2), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_website", "external_id", name="uq_resumes_source_external"),
    )
    op.create_index("idx_resumes_language_code", "resumes", ["language_code"], unique=False)
    op.create_index("idx_resumes_source_published", "resumes", ["source_website", "published_at"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_resumes_source_published", table_name="resumes")
    op.drop_index("idx_resumes_language_code", table_name="resumes")
    op.drop_table("resumes")

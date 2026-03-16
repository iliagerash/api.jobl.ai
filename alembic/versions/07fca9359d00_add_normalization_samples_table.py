"""add normalization samples table

Revision ID: 07fca9359d00
Revises: 2f40abe9cf1a
Create Date: 2026-03-08 15:24:01.965830
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '07fca9359d00'
down_revision = '2f40abe9cf1a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "normalization_samples",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source_db", sa.String(length=128), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=True),
        sa.Column("site_id", sa.SmallInteger(), nullable=True),
        sa.Column("source_job_id", sa.BigInteger(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("title_raw", sa.Text(), nullable=False),
        sa.Column("description_raw", sa.Text(), nullable=False),
        sa.Column("expected_title_normalized", sa.Text(), nullable=True),
        sa.Column("expected_description_clean", sa.Text(), nullable=True),
        sa.Column("expected_description_html", sa.Text(), nullable=True),
        sa.Column("generated_title_normalized", sa.Text(), nullable=True),
        sa.Column("generated_description_clean", sa.Text(), nullable=True),
        sa.Column("generated_description_html", sa.Text(), nullable=True),
        sa.Column("title_match", sa.Boolean(), nullable=True),
        sa.Column("description_clean_match", sa.Boolean(), nullable=True),
        sa.Column("description_html_match", sa.Boolean(), nullable=True),
        sa.Column("review_status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("batch_tag", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name="ck_normalization_samples_review_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_normalization_samples_source",
        "normalization_samples",
        ["source_db", "country_code", "site_id"],
        unique=False,
    )
    op.create_index(
        "idx_normalization_samples_review_status",
        "normalization_samples",
        ["review_status"],
        unique=False,
    )
    op.create_index(
        "idx_normalization_samples_batch_tag",
        "normalization_samples",
        ["batch_tag"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_normalization_samples_batch_tag", table_name="normalization_samples")
    op.drop_index("idx_normalization_samples_review_status", table_name="normalization_samples")
    op.drop_index("idx_normalization_samples_source", table_name="normalization_samples")
    op.drop_table("normalization_samples")

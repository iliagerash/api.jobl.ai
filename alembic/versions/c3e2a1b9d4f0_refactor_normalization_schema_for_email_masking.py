"""refactor normalization schema for email masking and title cleaning

Revision ID: c3e2a1b9d4f0
Revises: b2ff36274cb3
Create Date: 2026-03-12 12:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c3e2a1b9d4f0"
down_revision = "b2ff36274cb3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop dependent artifacts that reference old jobs columns.
    op.execute("DROP VIEW IF EXISTS jobs_active")
    op.drop_index("idx_jobs_normalize_backlog", table_name="jobs")

    # jobs table changes.
    op.alter_column("jobs", "title_normalized", new_column_name="title_clean")
    op.drop_column("jobs", "description_html")
    op.add_column("jobs", sa.Column("email", sa.Text(), nullable=True))

    # normalization_samples table changes.
    op.alter_column("normalization_samples", "description_raw", new_column_name="description")
    op.drop_column("normalization_samples", "expected_description_clean")
    op.drop_column("normalization_samples", "expected_description_html")
    op.drop_column("normalization_samples", "generated_description_clean")
    op.drop_column("normalization_samples", "generated_description_html")
    op.drop_column("normalization_samples", "description_clean_match")
    op.drop_column("normalization_samples", "description_html_match")

    # Recreate normalized backlog index with new title column name.
    op.create_index(
        "idx_jobs_normalize_backlog",
        "jobs",
        ["id"],
        unique=False,
        postgresql_where=sa.text("title_clean IS NULL OR description_clean IS NULL"),
    )

    # Recreate active jobs view with updated projection.
    op.execute(
        """
        CREATE OR REPLACE VIEW jobs_active AS
        SELECT
            id,
            source_db,
            source_job_id,
            site_id,
            external_id,
            title,
            description,
            company_id,
            company_name,
            site_title,
            url,
            email,
            city_id,
            city_title,
            region_id,
            region_title,
            country_code,
            salary_min,
            salary_max,
            salary_period,
            salary_currency,
            contract,
            experience,
            education,
            published_at,
            expires_at,
            is_remote,
            is_active,
            title_clean,
            description_clean,
            embedding,
            embedded_at
        FROM jobs
        WHERE is_active = TRUE
          AND (expires_at IS NULL OR expires_at > NOW())
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS jobs_active")
    op.drop_index("idx_jobs_normalize_backlog", table_name="jobs")

    # Revert jobs table changes.
    op.drop_column("jobs", "email")
    op.add_column("jobs", sa.Column("description_html", sa.Text(), nullable=True))
    op.alter_column("jobs", "title_clean", new_column_name="title_normalized")

    # Revert normalization_samples table changes.
    op.add_column("normalization_samples", sa.Column("description_html_match", sa.Boolean(), nullable=True))
    op.add_column("normalization_samples", sa.Column("description_clean_match", sa.Boolean(), nullable=True))
    op.add_column("normalization_samples", sa.Column("generated_description_html", sa.Text(), nullable=True))
    op.add_column("normalization_samples", sa.Column("generated_description_clean", sa.Text(), nullable=True))
    op.add_column("normalization_samples", sa.Column("expected_description_html", sa.Text(), nullable=True))
    op.add_column("normalization_samples", sa.Column("expected_description_clean", sa.Text(), nullable=True))
    op.alter_column("normalization_samples", "description", new_column_name="description_raw")

    op.create_index(
        "idx_jobs_normalize_backlog",
        "jobs",
        ["id"],
        unique=False,
        postgresql_where=sa.text(
            "title_normalized IS NULL OR description_clean IS NULL OR description_html IS NULL"
        ),
    )

    op.execute(
        """
        CREATE OR REPLACE VIEW jobs_active AS
        SELECT
            id,
            source_db,
            source_job_id,
            site_id,
            external_id,
            title,
            description,
            company_id,
            company_name,
            site_title,
            url,
            city_id,
            city_title,
            region_id,
            region_title,
            country_code,
            salary_min,
            salary_max,
            salary_period,
            salary_currency,
            contract,
            experience,
            education,
            published_at,
            expires_at,
            is_remote,
            is_active,
            title_normalized,
            description_clean,
            description_html,
            embedding,
            embedded_at
        FROM jobs
        WHERE is_active = TRUE
          AND (expires_at IS NULL OR expires_at > NOW())
        """
    )

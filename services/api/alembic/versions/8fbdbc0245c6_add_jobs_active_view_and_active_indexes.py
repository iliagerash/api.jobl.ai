"""add jobs_active view and active indexes

Revision ID: 8fbdbc0245c6
Revises: 2e77decb51a7
Create Date: 2026-03-08 15:01:26.005705
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8fbdbc0245c6'
down_revision = '2e77decb51a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
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

    op.create_index(
        "idx_jobs_active_expires",
        "jobs",
        ["is_active", "expires_at"],
        unique=False,
    )

    # Speeds common frontend filters (country + remote) for active jobs.
    op.create_index(
        "idx_jobs_country_remote_pub_active",
        "jobs",
        ["country_code", "is_remote", sa.text("published_at DESC")],
        unique=False,
        postgresql_where=sa.text("is_active = TRUE"),
    )


def downgrade() -> None:
    op.drop_index("idx_jobs_country_remote_pub_active", table_name="jobs")
    op.drop_index("idx_jobs_active_expires", table_name="jobs")
    op.execute("DROP VIEW IF EXISTS jobs_active")

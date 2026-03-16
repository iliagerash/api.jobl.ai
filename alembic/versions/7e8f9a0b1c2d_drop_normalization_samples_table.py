"""drop_normalization_samples_table

Revision ID: 7e8f9a0b1c2d
Revises: 3c1d2e4f5a6b
Create Date: 2026-03-16 21:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7e8f9a0b1c2d'
down_revision = '3c1d2e4f5a6b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table('normalization_samples')


def downgrade() -> None:
    op.create_table(
        'normalization_samples',
        sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column('source_db', sa.VARCHAR(length=128), nullable=False),
        sa.Column('country_code', sa.VARCHAR(length=2), nullable=True),
        sa.Column('language_code', sa.VARCHAR(length=2), nullable=True),
        sa.Column('country_name', sa.VARCHAR(length=128), nullable=True),
        sa.Column('city_title', sa.VARCHAR(length=255), nullable=True),
        sa.Column('region_title', sa.VARCHAR(length=255), nullable=True),
        sa.Column('site_id', sa.SMALLINT(), nullable=True),
        sa.Column('source_job_id', sa.BIGINT(), nullable=True),
        sa.Column('url', sa.TEXT(), nullable=True),
        sa.Column('company_name', sa.VARCHAR(length=255), nullable=True),
        sa.Column('email', sa.TEXT(), nullable=True),
        sa.Column('title', sa.TEXT(), nullable=False),
        sa.Column('description', sa.TEXT(), nullable=False),
        sa.Column('expected_title_normalized', sa.TEXT(), nullable=True),
        sa.Column('generated_title_normalized', sa.TEXT(), nullable=True),
        sa.Column('title_match', sa.BOOLEAN(), nullable=True),
        sa.Column('review_status', sa.VARCHAR(length=20), nullable=False, server_default='pending'),
        sa.Column('review_notes', sa.TEXT(), nullable=True),
        sa.Column('batch_tag', sa.VARCHAR(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name='ck_normalization_samples_review_status',
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_normalization_samples_source', 'normalization_samples', ['source_db', 'country_code', 'site_id'])
    op.create_index('idx_normalization_samples_language_code', 'normalization_samples', ['language_code'])
    op.create_index('idx_normalization_samples_review_status', 'normalization_samples', ['review_status'])
    op.create_index('idx_normalization_samples_batch_tag', 'normalization_samples', ['batch_tag'])

"""add_job_labelling_table

Revision ID: 1430cb678afb
Revises: b5c6d7e8f9a0
Create Date: 2026-03-18 22:32:15.279781
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1430cb678afb'
down_revision = 'b5c6d7e8f9a0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'job_labelling',
        sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column('job_id', sa.BIGINT(), nullable=False),
        sa.Column('title', sa.TEXT(), nullable=False),
        sa.Column('title_clean', sa.TEXT(), nullable=True),
        sa.Column('description', sa.TEXT(), nullable=True),
        sa.Column('description_clean', sa.TEXT(), nullable=True),
        sa.Column('company_name', sa.VARCHAR(255), nullable=True),
        sa.Column('country_code', sa.CHAR(2), nullable=True),
        sa.Column('language_code', sa.CHAR(2), nullable=True),
        sa.Column('original_category', sa.VARCHAR(255), nullable=True),
        sa.Column('email', sa.TEXT(), nullable=True),
        sa.Column('expiry_date', sa.DATE(), nullable=True),
        sa.Column('category_id', sa.BIGINT(), nullable=False),
        sa.Column('labelled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', name='uq_job_labelling_job_id'),
    )
    op.create_index('idx_job_labelling_category', 'job_labelling', ['category_id'])


def downgrade() -> None:
    op.drop_index('idx_job_labelling_category', table_name='job_labelling')
    op.drop_table('job_labelling')

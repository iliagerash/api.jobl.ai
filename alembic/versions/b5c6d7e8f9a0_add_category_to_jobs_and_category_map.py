"""add_category_to_jobs_and_category_map

Revision ID: b5c6d7e8f9a0
Revises: 3c1d2e4f5a6b
Create Date: 2026-03-17 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b5c6d7e8f9a0'
down_revision = '3c1d2e4f5a6b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('category', sa.VARCHAR(255), nullable=True))
    op.create_table(
        'category_map',
        sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column('original_category', sa.VARCHAR(255), nullable=False),
        sa.Column('category_id', sa.BIGINT(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('original_category', name='uq_category_map_original'),
    )


def downgrade() -> None:
    op.drop_table('category_map')
    op.drop_column('jobs', 'category')

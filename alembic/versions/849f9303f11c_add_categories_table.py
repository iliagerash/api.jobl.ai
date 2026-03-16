"""add_categories_table

Revision ID: 849f9303f11c
Revises: f7a8b9c1d2e3
Create Date: 2026-03-16 20:08:00.704947
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '849f9303f11c'
down_revision = 'f7a8b9c1d2e3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'categories',
        sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column('title', sa.VARCHAR(length=128), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('title'),
    )


def downgrade() -> None:
    op.drop_table('categories')

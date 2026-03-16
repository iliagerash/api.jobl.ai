"""add_sync_state_table

Revision ID: 3c1d2e4f5a6b
Revises: 849f9303f11c
Create Date: 2026-03-16 20:30:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3c1d2e4f5a6b'
down_revision = '849f9303f11c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'sync_state',
        sa.Column('source_db', sa.VARCHAR(length=128), nullable=False),
        sa.Column('destination', sa.VARCHAR(length=100), nullable=False),
        sa.Column('last_job_id', sa.BIGINT(), nullable=False, server_default='0'),
        sa.Column(
            'updated_at',
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
        sa.PrimaryKeyConstraint('source_db', 'destination'),
    )


def downgrade() -> None:
    op.drop_table('sync_state')

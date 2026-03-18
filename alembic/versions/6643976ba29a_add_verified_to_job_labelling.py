"""add_verified_to_job_labelling

Revision ID: 6643976ba29a
Revises: 1430cb678afb
Create Date: 2026-03-18 23:59:19.774685
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6643976ba29a'
down_revision = '1430cb678afb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "job_labelling",
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    op.drop_column("job_labelling", "verified")

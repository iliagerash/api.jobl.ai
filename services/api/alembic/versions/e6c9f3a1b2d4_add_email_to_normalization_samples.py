"""add email to normalization_samples

Revision ID: e6c9f3a1b2d4
Revises: d4f1b8e7a2c9
Create Date: 2026-03-12 14:20:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e6c9f3a1b2d4"
down_revision = "d4f1b8e7a2c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("normalization_samples", sa.Column("email", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("normalization_samples", "email")

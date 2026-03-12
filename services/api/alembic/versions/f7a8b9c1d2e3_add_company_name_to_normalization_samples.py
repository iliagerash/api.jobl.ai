"""add company_name to normalization_samples

Revision ID: f7a8b9c1d2e3
Revises: e6c9f3a1b2d4
Create Date: 2026-03-12 15:05:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f7a8b9c1d2e3"
down_revision = "e6c9f3a1b2d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("normalization_samples", sa.Column("company_name", sa.VARCHAR(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("normalization_samples", "company_name")

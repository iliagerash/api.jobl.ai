"""add language_code to normalization_samples

Revision ID: 1c96ff4d0e2e
Revises: 6b51f1ce3b6d
Create Date: 2026-03-08 17:20:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1c96ff4d0e2e"
down_revision = "6b51f1ce3b6d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("normalization_samples", sa.Column("language_code", sa.VARCHAR(length=2), nullable=True))
    op.create_index("idx_normalization_samples_language_code", "normalization_samples", ["language_code"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_normalization_samples_language_code", table_name="normalization_samples")
    op.drop_column("normalization_samples", "language_code")

"""rename normalization_samples.title_raw to title

Revision ID: d4f1b8e7a2c9
Revises: c3e2a1b9d4f0
Create Date: 2026-03-12 12:45:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "d4f1b8e7a2c9"
down_revision = "c3e2a1b9d4f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("normalization_samples", "title_raw", new_column_name="title")


def downgrade() -> None:
    op.alter_column("normalization_samples", "title", new_column_name="title_raw")

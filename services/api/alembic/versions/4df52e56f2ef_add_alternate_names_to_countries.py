"""add alternate_names to countries

Revision ID: 4df52e56f2ef
Revises: 0784978dd5ac
Create Date: 2026-03-08 16:20:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "4df52e56f2ef"
down_revision = "0784978dd5ac"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("countries", sa.Column("alternate_names", sa.ARRAY(sa.TEXT()), nullable=True))


def downgrade() -> None:
    op.drop_column("countries", "alternate_names")

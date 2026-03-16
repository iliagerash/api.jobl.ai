"""add language_codes array to countries

Revision ID: b2ff36274cb3
Revises: 1c96ff4d0e2e
Create Date: 2026-03-08 19:22:12.889758
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2ff36274cb3'
down_revision = '1c96ff4d0e2e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("countries", sa.Column("language_codes", sa.ARRAY(sa.TEXT()), nullable=True))


def downgrade() -> None:
    op.drop_column("countries", "language_codes")

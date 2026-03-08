"""add countries lookup and sample location columns

Revision ID: 0784978dd5ac
Revises: 07fca9359d00
Create Date: 2026-03-08 16:11:51.856184
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0784978dd5ac'
down_revision = '07fca9359d00'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "countries",
        sa.Column("code", sa.CHAR(length=2), nullable=False),
        sa.Column("name", sa.VARCHAR(length=128), nullable=False),
        sa.PrimaryKeyConstraint("code"),
        sa.UniqueConstraint("name"),
    )
    op.add_column("normalization_samples", sa.Column("country_name", sa.VARCHAR(length=128), nullable=True))
    op.add_column("normalization_samples", sa.Column("city_title", sa.VARCHAR(length=255), nullable=True))
    op.add_column("normalization_samples", sa.Column("region_title", sa.VARCHAR(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("normalization_samples", "region_title")
    op.drop_column("normalization_samples", "city_title")
    op.drop_column("normalization_samples", "country_name")
    op.drop_table("countries")

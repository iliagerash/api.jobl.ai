"""add skills tables

Revision ID: d1a2b3c4d5e6
Revises: c8f2a1b3d4e5
Create Date: 2026-06-21
"""

from alembic import op
import sqlalchemy as sa

revision = "d1a2b3c4d5e6"
down_revision = "c8f2a1b3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "skills",
        sa.Column("uri", sa.String(255), primary_key=True),
        sa.Column("preferred_label_en", sa.Text(), nullable=False),
        sa.Column("skill_type", sa.String(50)),
    )

    op.create_table(
        "skill_labels",
        sa.Column("skill_uri", sa.String(255), sa.ForeignKey("skills.uri", ondelete="CASCADE"), primary_key=True),
        sa.Column("language_code", sa.String(10), primary_key=True),
        sa.Column("label", sa.Text(), primary_key=True),
    )

    op.create_index("idx_skill_labels_lookup", "skill_labels", ["language_code", "label"])


def downgrade() -> None:
    op.drop_index("idx_skill_labels_lookup", table_name="skill_labels")
    op.drop_table("skill_labels")
    op.drop_table("skills")

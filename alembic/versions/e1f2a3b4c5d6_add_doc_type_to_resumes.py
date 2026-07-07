"""add doc_type to resumes

Revision ID: e1f2a3b4c5d6
Revises: d4e5f6a7b8c9
Create Date: 2026-07-07

Adds doc_type_enum PG type and doc_type column to the resumes table.
Used by the resume_classify task on /v1/embed (type=resume).
"""

from alembic import op

revision = "e1f2a3b4c5d6"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'doc_type_enum') THEN
                CREATE TYPE doc_type_enum AS ENUM ('resume', 'cover_letter', 'stub', 'other');
            END IF;
        END
        $$;
    """)
    op.execute("""
        ALTER TABLE resumes
        ADD COLUMN IF NOT EXISTS doc_type doc_type_enum NULL
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_resumes_doc_type ON resumes (doc_type)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_resumes_doc_type")
    op.execute("ALTER TABLE resumes DROP COLUMN IF EXISTS doc_type")
    op.execute("DROP TYPE IF EXISTS doc_type_enum")

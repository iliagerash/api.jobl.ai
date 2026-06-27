from sqlalchemy import BIGINT, CHAR, NUMERIC, VARCHAR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Keyword(Base):
    __tablename__ = "keywords"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    canonical_id: Mapped[int] = mapped_column(BIGINT, nullable=False)
    title: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    language_code: Mapped[str] = mapped_column(CHAR(2), nullable=False)
    category_id: Mapped[int] = mapped_column(BIGINT, nullable=False)
    embedding: Mapped[str | None] = mapped_column(nullable=True)  # vector(1024)


class JobKeyword(Base):
    __tablename__ = "job_keywords"

    job_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    keyword_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    distance: Mapped[float | None] = mapped_column(NUMERIC(6, 4), nullable=True)

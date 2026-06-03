from datetime import datetime
from decimal import Decimal

from sqlalchemy import BIGINT, BOOLEAN, CHAR, DateTime, Index, NUMERIC, TEXT, VARCHAR, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Resume(Base):
    __tablename__ = "resumes"
    __table_args__ = (
        UniqueConstraint("source_website", "external_id", name="uq_resumes_source_external"),
        Index("idx_resumes_language_code", "language_code"),
        Index("idx_resumes_source_published", "source_website", "published_at"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)

    source_website: Mapped[str] = mapped_column(VARCHAR(128), nullable=False)
    external_id: Mapped[int] = mapped_column(BIGINT, nullable=False)

    title: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)

    city_title: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    region_title: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    country_code: Mapped[str | None] = mapped_column(CHAR(2), nullable=True)

    salary: Mapped[Decimal | None] = mapped_column(NUMERIC(15, 2), nullable=True)
    salary_period: Mapped[str | None] = mapped_column(VARCHAR(20), nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(CHAR(3), nullable=True)
    contract: Mapped[str | None] = mapped_column(VARCHAR(24), nullable=True)

    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_remote: Mapped[bool] = mapped_column(BOOLEAN, nullable=False, default=False, server_default="false")
    language_code: Mapped[str | None] = mapped_column(CHAR(2), nullable=True)

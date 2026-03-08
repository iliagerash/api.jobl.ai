from datetime import datetime

from sqlalchemy import (
    BIGINT,
    BOOLEAN,
    CHAR,
    DateTime,
    Index,
    NUMERIC,
    SMALLINT,
    TEXT,
    VARCHAR,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("source_db", "source_job_id", name="uq_jobs_source_pk"),
        UniqueConstraint("source_db", "site_id", "external_id", name="uq_jobs_source_external"),
        Index("idx_jobs_active_pub", "is_active", "published_at"),
        Index("idx_jobs_country_active_pub", "country_code", "is_active", "published_at"),
        Index("idx_jobs_remote_active_pub", "is_remote", "is_active", "published_at"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)

    # Source identity
    source_db: Mapped[str] = mapped_column(VARCHAR(128), nullable=False)
    source_job_id: Mapped[int] = mapped_column(BIGINT, nullable=False)
    site_id: Mapped[int] = mapped_column(SMALLINT, nullable=False)
    external_id: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)

    # Content
    title: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    company_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    company_name: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    site_title: Mapped[str | None] = mapped_column(VARCHAR(100), nullable=True)
    url: Mapped[str] = mapped_column(TEXT, nullable=False)

    # Location
    city_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    city_title: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    region_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    region_title: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    country_code: Mapped[str | None] = mapped_column(CHAR(2), nullable=True)

    # Compensation / attributes
    salary_min: Mapped[float | None] = mapped_column(NUMERIC(15, 2), nullable=True)
    salary_max: Mapped[float | None] = mapped_column(NUMERIC(15, 2), nullable=True)
    salary_period: Mapped[str | None] = mapped_column(VARCHAR(10), nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(CHAR(3), nullable=True)
    contract: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    experience: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    education: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)

    # Lifecycle
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_remote: Mapped[bool] = mapped_column(BOOLEAN, nullable=False, default=False, server_default="false")
    is_active: Mapped[bool] = mapped_column(BOOLEAN, nullable=False, default=True, server_default="true")

    # Local AI processing
    title_normalized: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    description_clean: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    description_html: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    embedding: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

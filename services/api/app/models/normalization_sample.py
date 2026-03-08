from datetime import datetime

from sqlalchemy import BIGINT, BOOLEAN, CheckConstraint, DateTime, Index, SMALLINT, TEXT, VARCHAR, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NormalizationSample(Base):
    __tablename__ = "normalization_samples"
    __table_args__ = (
        CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name="ck_normalization_samples_review_status",
        ),
        Index("idx_normalization_samples_source", "source_db", "country_code", "site_id"),
        Index("idx_normalization_samples_review_status", "review_status"),
        Index("idx_normalization_samples_batch_tag", "batch_tag"),
    )

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    source_db: Mapped[str] = mapped_column(VARCHAR(128), nullable=False)
    country_code: Mapped[str | None] = mapped_column(VARCHAR(2), nullable=True)
    country_name: Mapped[str | None] = mapped_column(VARCHAR(128), nullable=True)
    city_title: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    region_title: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    site_id: Mapped[int | None] = mapped_column(SMALLINT, nullable=True)
    source_job_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    url: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    title_raw: Mapped[str] = mapped_column(TEXT, nullable=False)
    description_raw: Mapped[str] = mapped_column(TEXT, nullable=False)

    expected_title_normalized: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    expected_description_clean: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    expected_description_html: Mapped[str | None] = mapped_column(TEXT, nullable=True)

    generated_title_normalized: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    generated_description_clean: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    generated_description_html: Mapped[str | None] = mapped_column(TEXT, nullable=True)

    title_match: Mapped[bool | None] = mapped_column(BOOLEAN, nullable=True)
    description_clean_match: Mapped[bool | None] = mapped_column(BOOLEAN, nullable=True)
    description_html_match: Mapped[bool | None] = mapped_column(BOOLEAN, nullable=True)

    review_status: Mapped[str] = mapped_column(VARCHAR(20), nullable=False, server_default="pending")
    review_notes: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    batch_tag: Mapped[str | None] = mapped_column(VARCHAR(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

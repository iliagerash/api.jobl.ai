from typing import Any

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SourceCountry(Base):
    __tablename__ = "source_countries"

    db_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

from sqlalchemy import CHAR, TEXT, VARCHAR
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Country(Base):
    __tablename__ = "countries"

    code: Mapped[str] = mapped_column(CHAR(2), primary_key=True)
    name: Mapped[str] = mapped_column(VARCHAR(128), nullable=False, unique=True)
    alternate_names: Mapped[list[str] | None] = mapped_column(ARRAY(TEXT), nullable=True)

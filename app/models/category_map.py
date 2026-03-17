from sqlalchemy import BIGINT, VARCHAR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CategoryMap(Base):
    __tablename__ = "category_map"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    original_category: Mapped[str] = mapped_column(VARCHAR(255), nullable=False, unique=True)
    category_id: Mapped[int] = mapped_column(BIGINT, nullable=False)

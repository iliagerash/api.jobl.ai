from sqlalchemy import BIGINT, VARCHAR, TIMESTAMP, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SyncState(Base):
    __tablename__ = "sync_state"

    source_db: Mapped[str] = mapped_column(VARCHAR(128), primary_key=True, nullable=False)
    destination: Mapped[str] = mapped_column(VARCHAR(100), primary_key=True, nullable=False)
    last_job_id: Mapped[int] = mapped_column(BIGINT, nullable=False, default=0)
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")
    )

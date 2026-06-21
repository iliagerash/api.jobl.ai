from sqlalchemy import Column, ForeignKey, Index, String, Text
from app.db.base import Base


class Skill(Base):
    __tablename__ = "skills"

    uri = Column(String(255), primary_key=True)
    preferred_label_en = Column(Text, nullable=False)
    skill_type = Column(String(50))


class SkillLabel(Base):
    __tablename__ = "skill_labels"

    skill_uri = Column(String(255), ForeignKey("skills.uri", ondelete="CASCADE"), primary_key=True)
    language_code = Column(String(10), primary_key=True)
    label = Column(Text, primary_key=True)

    __table_args__ = (
        Index("idx_skill_labels_lookup", "language_code", "label"),
    )

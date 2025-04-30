from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.database import Base  # 또는 SQLAlchemy Base

class EmotionStatistics(Base):
    __tablename__ = "emotionStatistics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    quadrant = Column(Integer, nullable=True)
    emotiontype_id = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
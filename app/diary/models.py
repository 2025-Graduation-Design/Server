from sqlalchemy import Column, Integer, ForeignKey, Text, Float, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
from app.emotion.models import EmotionType

class Diary(Base):
    __tablename__ = "diary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    emotiontype_id = Column(Integer, ForeignKey("emotionType.id", ondelete="SET NULL"), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ✅ 클래스명을 정확히 사용 (User, EmotionType)
    user = relationship("User", back_populates="diaries")
    emotion = relationship(EmotionType, back_populates="diaries")
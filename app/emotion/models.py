from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from app.database import Base

class EmotionType(Base):
    __tablename__ = "emotionType"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    quadrant = Column(Integer, nullable=False)
    related_emotion_id = Column(Integer, ForeignKey("emotionType.id", ondelete="SET NULL"), nullable=True)

    # ✅ 관계 명칭을 `diaries`로 변경
    diaries = relationship("Diary", back_populates="emotion")
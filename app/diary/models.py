from sqlalchemy import Column, Integer, ForeignKey, Text, Float, DateTime, JSON, String
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
from app.emotion.models import EmotionType

class Diary(Base):
    __tablename__ = "diary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    best_sentence = Column(Text, nullable=False)
    emotiontype_id = Column(Integer, ForeignKey("emotionType.id", ondelete="SET NULL"), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="diaries")
    emotion = relationship(EmotionType, back_populates="diaries")
    recommended_songs = relationship("RecommendedSong", back_populates="diary", cascade="all, delete-orphan")


class diaryEmbedding(Base):
    __tablename__ = "diaryEmbedding"

    id = Column(Integer, primary_key=True, autoincrement=True)
    diary_id = Column(Integer, nullable=False)
    embedding = Column(JSON, nullable=False)


class RecommendedSong(Base):
    __tablename__ = "recommendedSongs"

    id = Column(Integer, primary_key=True)
    diary_id = Column(Integer, ForeignKey('diary.id'), nullable=False)
    song_id = Column(Integer, nullable=False)
    song_name = Column(String(256))
    artist = Column(JSON)
    genre = Column(String(64))
    album_image = Column(String(512))
    best_lyric = Column(Text)
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    diary = relationship("Diary", back_populates="recommended_songs")
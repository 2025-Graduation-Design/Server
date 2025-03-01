from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from app.database import Base

class Genre(Base):
    __tablename__ = "genre"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)

    user_genres = relationship("UserGenre", back_populates="genre", cascade="all, delete")

class UserGenre(Base):
    __tablename__ = "userGenre"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    genre_id = Column(Integer, ForeignKey("genre.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="user_genres")
    genre = relationship("Genre", back_populates="user_genres")
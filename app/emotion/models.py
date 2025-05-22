import torch
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from app.database import Base

emotion_labels = [
    "신남", "기대", "만족", "편안", "허무", "우울", "슬픔", "분노"
]

# 모델 예측 인덱스 (0~7) → DB emotionType.id (1~8)
model_index_to_db_emotion_id = {
    0: 1,  # 신남
    1: 2,  # 기대
    2: 3,  # 만족
    3: 4,  # 편안
    4: 5,  # 허무
    5: 6,  # 우울
    6: 7,  # 슬픔
    7: 8,  # 분노
}

class EmotionType(Base):
    __tablename__ = "emotionType"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    quadrant = Column(Integer, nullable=False)
    related_emotion_id = Column(Integer, ForeignKey("emotionType.id", ondelete="SET NULL"), nullable=True)

    diaries = relationship("Diary", back_populates="emotion")

class DiaryEmotionTag(Base):
    __tablename__ = "diaryEmotionTags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    diary_id = Column(Integer, ForeignKey("diary.id", ondelete="CASCADE"), nullable=False)
    emotiontype_id = Column(Integer, ForeignKey("emotionType.id", ondelete="CASCADE"), nullable=False)
    score = Column(Float, nullable=False)

    diary = relationship("Diary", back_populates="emotion_tags")
    emotion = relationship("EmotionType")

tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=8)
model.load_state_dict(torch.load("app/emotion/best_model(7th).pt", map_location="cpu"))
model.eval()

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel

from sqlalchemy import Column, Integer, ForeignKey, String
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

    # ✅ 관계 명칭을 `diaries`로 변경
    diaries = relationship("Diary", back_populates="emotion")

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))

tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

model = EmotionClassifier(num_classes=len(emotion_labels))
state_dict = torch.load("app/emotion/best_model(1st).pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()
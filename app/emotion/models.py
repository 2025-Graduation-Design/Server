import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from app.database import Base

emotion_labels = [
    "신남", "기대", "편안", "만족", "허무", "우울", "슬픔", "분노"
]

# 모델 예측 인덱스 (0~7) → DB emotionType.id (1~8)
model_index_to_db_emotion_id = {
    0: 8,  # 분노
    1: 7,  # 슬픔
    2: 1,  # 신남/행복
    3: 5,  # 허무
    4: 3,  # 만족
    5: 4,  # 편안
    6: 2,  # 기대
    7: 6,  # 우울
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
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

model = EmotionClassifier(num_classes=len(emotion_labels))
state_dict = torch.load("app/emotion/2Cycle_best_model_epoch5.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()
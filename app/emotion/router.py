import logging
import torch
from app.emotion.models import tokenizer, model
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.emotion.models import EmotionType

router = APIRouter()
logger = logging.getLogger(__name__)

def predict_emotion(text: str):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()

        topk = torch.topk(probs, k=3)
        for i in range(3):
            idx = topk.indices[0][i].item()
            score = topk.values[0][i].item()
            logger.info(f"[감정 분석 Top-{i + 1}] ID: {idx}, 확신도: {score:.4f}")

    return pred_index, probs.squeeze().tolist()

@router.get("/emotions", response_model=list[dict])
def get_emotions(db: Session = Depends(get_db)):
    emotions = db.query(EmotionType).all()
    return [{"id": e.id, "name": e.name} for e in emotions]
import logging
import torch
from app.emotion.models import tokenizer, model

logger = logging.getLogger(__name__)

def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = torch.softmax(outputs, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()

        # 🪵 Top-3 감정 ID와 확률 로그 출력
        topk = torch.topk(probs, k=3)
        for i in range(3):
            idx = topk.indices[0][i].item()
            score = topk.values[0][i].item()
            logger.info(f"[감정 분석 Top-{i+1}] ID: {idx}, 확신도: {score:.4f}")

    return pred_index, probs.squeeze().tolist()
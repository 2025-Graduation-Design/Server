from pydantic import BaseModel
from typing import Optional, Dict

class EmotionStatisticsSchema(BaseModel):
    emotiontype_id: int
    count: int
    quadrant: Optional[int]

    class Config:
        orm_mode = True

class EmotionSummaryResponse(BaseModel):
    top_emotion_group: str
    group_distribution: Dict[str, int]
    message: str
    suggestion: str
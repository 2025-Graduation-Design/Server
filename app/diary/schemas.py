from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Request
class DiaryCreateRequest(BaseModel):
    content: str

class DiaryUpdateRequest(BaseModel):
    content: Optional[str] = None

# Response
class DiaryResponse(BaseModel):
    id: int
    user_id: int
    content: str
    emotiontype_id: Optional[int] = None
    confidence: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
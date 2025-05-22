from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Union


class EmotionScore(BaseModel):
    emotion_id: int
    score: float

# Request
class DiaryCreateRequest(BaseModel):
    content: str

class DiaryUpdateRequest(BaseModel):
    content: Optional[str] = None

# Response
class SongResponse(BaseModel):
    song_id: int  # MongoDB _id → 문자열 변환
    song_name: str
    artist: list[str]
    genre: str
    album_image: str
    best_lyric: str
    similarity_score: float

    class Config:
        allow_population_by_field_name = True

class DiaryResponse(BaseModel):
    id: int
    user_id: int
    content: str
    emotiontype_id: Optional[int] = None
    confidence: Optional[float] = None
    recommended_songs: list[SongResponse]
    top_emotions: list[EmotionScore] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # pydantic v2용 (orm_mode → from_attributes)

class RecommendSongResponse(BaseModel):
    id: int
    song_id: int
    song_name: str
    artist: List[str]
    genre: str
    album_image: str
    best_lyric: str
    similarity_score: float

    class Config:
        orm_mode = True

class SentenceEmotion(BaseModel):
    sentence: str
    predicted_emotion_id: int
    confidence: float
    top3: List[Dict[str, Union[int, float]]]

class DiaryPreviewResponse(BaseModel):
    id: int
    user_id: int
    content: str
    emotiontype_id: Optional[int] = None
    confidence: Optional[float] = None
    recommended_songs: list[SongResponse]
    top_emotions: list[EmotionScore] = None
    created_at: datetime
    updated_at: datetime
    sentence_emotions: List[SentenceEmotion]

    class Config:
        from_attributes = True  # pydantic v2용 (orm_mode → from_attributes)


class DiaryCountResponse(BaseModel):
    user_id: int
    year: int
    month: int
    diary_count: int

class TopEmotion(BaseModel):
    emotion_id: int
    score: float

class BestSentence(BaseModel):
    sentence: str
    predicted_emotion_id: int
    confidence: float

class SentenceEmotion(BaseModel):
    sentence: str
    predicted_emotion_id: int
    confidence: float
    top3: List[TopEmotion]

class EmotionPreviewResponse(BaseModel):
    content: str
    emotiontype_id: int
    confidence: float
    top_emotions: List[TopEmotion]
    best_sentence: BestSentence
    sentence_emotions: List[SentenceEmotion]


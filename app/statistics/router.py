from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import extract
from sqlalchemy.orm import Session
from app.database import get_db
from datetime import datetime
from . import models, schemas
from .models import EmotionStatistics
from .utils import summarize_emotions
from ..user.auth import get_current_user
from ..user.models import User

router = APIRouter()

@router.get(
    "/current-month",
    response_model=schemas.EmotionSummaryResponse,
    summary="현재 달 감정 통계",
    description="로그인한 사용자의 이번 달 감정 통계를 요약해 반환합니다."
)
def get_current_month_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.now()
    stats = db.query(models.EmotionStatistics).filter(
        models.EmotionStatistics.user_id == current_user.id,
        extract("year", models.EmotionStatistics.created_at) == now.year,
        extract("month", models.EmotionStatistics.created_at) == now.month
    ).all()

    if not stats:
        raise HTTPException(status_code=404, detail="이번 달 통계가 없습니다.")

    emotion_ids = []
    for s in stats:
        if s.emotiontype_id is not None:
            emotion_ids.extend([s.emotiontype_id] * s.count)

    return summarize_emotions(emotion_ids)

@router.get(
    "",
    response_model=List[schemas.EmotionStatisticsSchema],
    summary="전체 감정 통계",
    description="로그인한 사용자의 전체 감정 통계를 반환합니다."
)
def get_all_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stats = db.query(models.EmotionStatistics).filter(
        models.EmotionStatistics.user_id == current_user.id
    ).all()

    if not stats:
        raise HTTPException(status_code=404, detail="감정 통계가 없습니다.")
    return stats

@router.get(
    "/{year}/{month}",
    response_model=schemas.EmotionSummaryResponse,
    summary="특정 연/월 감정 통계",
    description="로그인한 사용자의 특정 연도와 월의 감정 통계를 요약해 반환합니다."
)
def get_statistics_by_month(
    year: int,
    month: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stats = db.query(EmotionStatistics).filter(
        EmotionStatistics.user_id == current_user.id,
        extract("year", EmotionStatistics.created_at) == year,
        extract("month", EmotionStatistics.created_at) == month
    ).all()

    if not stats:
        raise HTTPException(status_code=404, detail="해당 연도/월 통계가 없습니다.")

    emotion_ids = [s.emotiontype_id for s in stats if s.emotiontype_id is not None]
    return summarize_emotions(emotion_ids)
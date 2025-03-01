from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import extract
from app.database import get_db
from app.user.auth import get_current_user
from app.diary.models import Diary
from app.diary.schemas import DiaryCreateRequest, DiaryUpdateRequest, DiaryResponse
from app.user.models import User
from typing import List

router = APIRouter()

# 📝 일기 작성 API
@router.post("", response_model=DiaryResponse, status_code=status.HTTP_201_CREATED, summary="일기 작성", description="새로운 일기를 작성합니다.")
def create_diary(
    diary_request: DiaryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_diary = Diary(
        user_id=current_user.id,
        content=diary_request.content
    )

    db.add(new_diary)
    db.commit()
    db.refresh(new_diary)

    # 🚀 [추후 감정 분석 모델 적용 예정]
    # 1. KoBERT 모델을 사용하여 감정을 분석한다.
    # 2. 감정 분석 결과를 `new_diary.emotiontype_id`, `new_diary.confidence`에 저장한다.
    # 3. 다시 DB에 반영한다.

    return new_diary


@router.get("/{diary_id}", response_model=DiaryResponse, summary="일기 조회", description="특정 일기를 조회합니다.")
def get_diary(
    diary_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(Diary.id == diary_id, Diary.user_id == current_user.id).first()
    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    return diary

@router.get("", response_model=list[DiaryResponse], summary="내 일기 목록 조회", description="사용자가 작성한 모든 일기를 조회합니다.")
def get_all_diaries(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diaries = db.query(Diary).filter(Diary.user_id == current_user.id).all()
    return diaries

@router.put("/{diary_id}", response_model=DiaryResponse, summary="일기 수정", description="작성한 일기를 수정합니다.")
def update_diary(
    diary_id: int,
    diary_update: DiaryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(Diary.id == diary_id, Diary.user_id == current_user.id).first()
    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    if diary_update.content:
        diary.content = diary_update.content

    db.commit()
    db.refresh(diary)
    return diary

@router.delete("/{diary_id}", summary="일기 삭제", description="작성한 일기를 삭제합니다.")
def delete_diary(
    diary_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(Diary.id == diary_id, Diary.user_id == current_user.id).first()
    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    db.delete(diary)
    db.commit()

    return {"message": "일기가 삭제되었습니다!"}

@router.get("/{year}/{month}", response_model=List[DiaryResponse], summary="특정 연도와 월의 일기 조회", description="입력한 연도와 월에 해당하는 모든 일기를 조회합니다.")
def get_diaries_by_month(
    year: int,
    month: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diaries = db.query(Diary).filter(
        Diary.user_id == current_user.id,
        extract("year", Diary.created_at) == year,
        extract("month", Diary.created_at) == month
    ).all()

    if not diaries:
        raise HTTPException(status_code=404, detail="해당 연도와 월에 작성된 일기가 없습니다.")

    return diaries
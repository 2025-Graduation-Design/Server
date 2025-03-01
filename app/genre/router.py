from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.genre.models import Genre
from app.genre.schemas import GenreResponse

router = APIRouter()

@router.get("", response_model=list[GenreResponse], summary="모든 음악 장르 조회", description="저장된 모든 음악 장르를 조회합니다.")
def get_all_genres(db: Session = Depends(get_db)):
    genres = db.query(Genre).all()
    return genres

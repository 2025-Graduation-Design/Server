from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, get_mongodb
from app.genre.models import Genre
from app.genre.schemas import GenreResponse

router = APIRouter()

@router.get("", response_model=list[GenreResponse], summary="모든 음악 장르 조회", description="저장된 모든 음악 장르를 조회합니다.")
def get_all_genres(db: Session = Depends(get_db)):
    genres = db.query(Genre).all()
    return genres

@router.post("/new", summary="장르 추가", description="현재 DB에 있는 음악 장르를 추가합니다.")
async def add_genres_from_mongodb(mongodb=Depends(get_mongodb), db: Session = Depends(get_db)):
    """
    MongoDB에서 'genre' 컬럼을 가져와 MySQL의 Genre 테이블에 추가
    (중복 제거 후 추가)
    """
    mongo_genres = await mongodb["song_meta"].distinct("genre")
    genre_set = set()

    for genre in mongo_genres:
        if genre:
            split_genres = [g.strip() for g in genre.split(",")]
            genre_set.update(split_genres)

    existing_genres = db.query(Genre.name).all()
    existing_genres = {g[0] for g in existing_genres}

    new_genres = [Genre(name=genre) for genre in genre_set if genre not in existing_genres]

    if new_genres:
        db.add_all(new_genres)
        db.commit()

    return {"message": f"{len(new_genres)}개 장르 추가 완료"}

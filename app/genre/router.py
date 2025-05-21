from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, get_mongodb
from app.genre.models import Genre
from app.genre.schemas import GenreResponse
from fastapi.responses import JSONResponse
from collections import Counter
router = APIRouter()

@router.get("", response_model=list[GenreResponse], summary="모든 음악 장르 조회", description="저장된 모든 음악 장르를 조회합니다.")
def get_all_genres(db: Session = Depends(get_db)):
    genres = db.query(Genre).all()
    return genres

@router.post("/new", summary="장르 추가", description="현재 DB에 있는 음악 장르를 추가합니다.")
async def add_genres_from_mongodb(mongodb=Depends(get_mongodb), db: Session = Depends(get_db)):
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

@router.get("/songs/genre-counts")
async def get_genre_counts(db: Session = Depends(get_mongodb)):
    genre_counter = Counter()

    cursor = db.song_meta.find({}, {"genre": 1})

    async for doc in cursor:
        genre_str = doc.get("genre", "")
        genres = [g.strip() for g in genre_str.split(",") if g.strip()]
        genre_counter.update(genres)

    return JSONResponse(content=dict(genre_counter))
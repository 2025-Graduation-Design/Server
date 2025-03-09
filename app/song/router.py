from fastapi import APIRouter, Depends
from app.database import get_mongodb

router = APIRouter()

@router.get("")
async def get_songs(mongodb=Depends(get_mongodb), limit: int = 10):
    """MongoDB에서 노래 데이터 조회"""
    songs = await mongodb["song_meta"].find().to_list(length=limit)

    for song in songs:
        song["_id"] = str(song["_id"])

    return songs
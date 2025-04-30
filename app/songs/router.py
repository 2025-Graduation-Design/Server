from typing import List

from fastapi import APIRouter, Depends, Query
from app.database import get_mongodb
from pymongo.database import Database
from app.songs.schemas import SongResponse

router = APIRouter()


@router.get("", response_model=List[SongResponse], summary="노래 조회", description="전체 노래 리스트를 조회합니다.")
async def get_songs(mongodb: Database = Depends(get_mongodb)):
    songs = await mongodb["song_meta"].find(
        {}, {
            "id": 1,
            "song_name": 1,
            "artist_name_basket": 1,
            "genre": 1,
            "lyrics": 1,
            "album_id": 1,
            "album_name": 1,
            "album_image": 1,
            "uri": 1
        }
    ).to_list(length=100)

    return [
        {
            "id": song.get("id"),
            "title": song.get("song_name"),
            "artist": ", ".join(song.get("artist_name_basket", [])),
            "genre": song.get("genre"),
            "lyrics": song.get("lyrics"),
            "album_id": song.get("album_id"),
            "album_name": song.get("album_name"),
            "album_image": song.get("album_image"),
            "uri": song.get("uri")
        } for song in songs
    ]


@router.get("/genre", response_model=List[SongResponse], summary="장르 기반 노래 조회",
            description="특정 장르에 해당하는 노래 목록을 조회합니다. (예: 발라드, 인디음악 등)")
async def get_songs_with_genres(
    genre: str = Query(..., description="조회할 장르 (예: '발라드')"),
    mongodb: Database = Depends(get_mongodb),
):
    songs = await mongodb["song_meta"].find(
        {"genre": {"$regex": genre, "$options": "i"}},
        {
            "id": 1,
            "song_name": 1,
            "artist_name_basket": 1,
            "genre": 1,
            "lyrics": 1,
            "album_id": 1,
            "album_name": 1,
            "album_image": 1,
            "uri": 1
        }
    ).to_list(length=100)

    return [
        {
            "id": song.get("id"),
            "title": song.get("song_name"),
            "artist": ", ".join(song.get("artist_name_basket", [])),
            "genre": song.get("genre"),
            "lyrics": song.get("lyrics"),
            "album_id": song.get("album_id"),
            "album_name": song.get("album_name"),
            "album_image": song.get("album_image"),
            "uri": song.get("uri")
        } for song in songs
    ]

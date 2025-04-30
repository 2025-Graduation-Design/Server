import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, get_mongodb
from app.embedding.models import kobert, get_existing_song_ids, save_song_embedding, load_songs_from_mongodb
from app.embedding.schemas import EmbeddingResponse
import time

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/embed-songs",
            response_model=EmbeddingResponse,
            summary="가사 임베딩",
            description="MongoDB에서 가사 데이터를 불러와 KoBERT로 임베딩하고, MySQL에 저장합니다.")
async def embed_songs(
    db: Session = Depends(get_db),
    mongodb = Depends(get_mongodb)
):
    songs = await load_songs_from_mongodb(mongodb)
    existing_song_ids = get_existing_song_ids(db)

    unprocessed_songs = [
        song for song in songs
        if song["song_id"] not in existing_song_ids
    ]

    BATCH_SIZE = 10
    processed_songs = []

    for i in range(0, len(unprocessed_songs), BATCH_SIZE):
        batch = unprocessed_songs[i:i + BATCH_SIZE]

        for song in batch:
            song_id = song.get("song_id")
            lyrics = song.get("lyrics", [])

            if not song_id or not lyrics:
                continue

            try:
                embeddings = [
                    kobert.get_embedding(line)
                    for line in lyrics if line.strip()
                ]
                save_song_embedding(db, song_id, embeddings)
                processed_songs.append({"song_id": song_id, "status": "embedded"})
            except Exception as e:
                logger.error(f"임베딩 실패 (song_id={song_id}): {e}")
                continue

        db.commit()
        time.sleep(3)  # 한 배치 끝나고 쿨다운

    return {
        "total_songs": len(songs),
        "embedded_songs": len(processed_songs),
        "processed_details": processed_songs
    }
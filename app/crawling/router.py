from fastapi import APIRouter, Depends
from app.database import get_mongodb
from app.crawling.melonCrawler import MelonCrawler

router = APIRouter()

@router.post("/melon")
async def crawl_melon(mongodb=Depends(get_mongodb), limit: int = 100):
    """멜론에서 상위 N개의 노래를 크롤링하여 MongoDB에 저장"""
    crawler = MelonCrawler()
    song_data = crawler.crawl_songs_with_details(limit)  # ✅ 함수명 수정

    new_songs = []
    for song in song_data:
        existing_song = await mongodb["song_meta"].find_one({"id": song["id"]})
        if not existing_song:
            new_songs.append(song)

    if new_songs:
        await mongodb["song_meta"].insert_many(new_songs)

    return {"message": f"{len(new_songs)}개 곡 저장 완료"}
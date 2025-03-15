from fastapi import APIRouter, Depends
from app.database import get_mongodb
from app.crawling.melonCrawler import MelonCrawler

router = APIRouter()

@router.post("/melon/top100", summary="탑 100 노래 크롤링", description="100곡까지 크롤링 가능합니다")
async def crawl_melon(mongodb=Depends(get_mongodb), limit: int = 100):
    """멜론에서 상위 N개의 노래를 크롤링하여 MongoDB에 저장"""
    crawler = MelonCrawler()
    song_data = crawler.crawl_songs_with_details(limit)

    new_songs = []
    for song in song_data:
        existing_song = await mongodb["song_meta"].find_one({"id": song["id"]})
        if not existing_song:
            new_songs.append(song)

    if new_songs:
        await mongodb["song_meta"].insert_many(new_songs)

    return {"message": f"{len(new_songs)}개 곡 저장 완료"}

@router.post("/melon/genre/top100", summary="음악 장르로 탑 100 크롤링", description="""
    GENRE_CODES = { "ballad", "dance", "hiphop", "randb", "indie", "rock" }
    앞에 있는 이름들 이용해서 크롤링하면 됩니다""")
async def crawl_genre_songs_with_details(mongodb=Depends(get_mongodb), genre: str = "ballad"):
    """멜론 특정 장르의 인기곡을 크롤링하고, 가사 및 장르 추가"""
    crawler = MelonCrawler()
    song_data = crawler.crawl_songs_with_details_genre(genre=genre)

    new_songs = []
    for song in song_data:
        existing_song = await mongodb["song_meta"].find_one({"id": song["id"]})
        if not existing_song:
            new_songs.append(song)

    if new_songs:
        await mongodb["song_meta"].insert_many(new_songs)

    return {"message": f"{len(new_songs)}개 {genre} 장르 곡 저장 완료"}
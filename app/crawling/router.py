from datetime import datetime

from fastapi import APIRouter, Depends
from app.database import get_mongodb
from app.crawling.melonCrawler import MelonCrawler

router = APIRouter()

@router.post("/melon/top100",
             summary="실시간 TOP100 곡 크롤링",
             description="멜론에서 실시간 인기곡 최대 100곡까지 크롤링하고, 중복되지 않은 곡을 MongoDB에 저장합니다.")
async def crawl_melon(
    mongodb = Depends(get_mongodb),
    limit: int = 100
):
    """멜론에서 실시간 인기곡을 크롤링하여 MongoDB에 저장합니다."""
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

@router.post("/melon/week/genre/top100",
             summary="주간 장르별 인기곡 크롤링",
             description="""
멜론 장르별 주간 인기곡을 크롤링합니다.  
지원 장르: `ballad`, `dance`, `hiphop`, `randb`, `indie`, `rock`  
중복되지 않은 곡만 MongoDB에 저장됩니다.
""")
async def crawl_genre_songs_with_details(
    mongodb = Depends(get_mongodb),
    genre: str = "ballad"
):
    """멜론 특정 장르의 주간 인기곡을 크롤링하고 MongoDB에 저장합니다."""
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

@router.post("/melon/month/genre/top100",
             summary="월간 장르별 인기곡 크롤링",
             description="""
지정한 연월(YYYYMM)의 멜론 월간 장르별 인기곡을 크롤링하고,  
가사 및 장르 정보를 포함한 상세 정보를 MongoDB에 저장합니다.  
기본값: `genre=ballad`, `year_month=현재 연월`
""")
async def crawl_monthly_genre_songs(
    mongodb = Depends(get_mongodb),
    genre: str = "ballad",
    year_month: str = datetime.now().strftime("%Y%m")
):
    """멜론 특정 장르의 월간 인기곡을 크롤링하고, 가사 및 장르 정보를 추가하여 저장합니다."""
    crawler = MelonCrawler()
    song_data = crawler.crawl_genre_songs_monthly(genre=genre, year_month=year_month, limit=100)

    new_songs = []
    for song in song_data:
        existing_song = await mongodb["song_meta"].find_one({"id": song["id"]})
        if not existing_song:
            details = crawler.crawl_song_details(song["id"])
            song.update(details)
            new_songs.append(song)

    if new_songs:
        await mongodb["song_meta"].insert_many(new_songs)

    return {"message": f"{len(new_songs)}개 {year_month} 월간 {genre} 장르 곡 저장 완료"}
from fastapi import FastAPI
from app.user.router import router as user_router# 🐾 유저 라우터 임포트
from app.diary.router import router as diary_router
from app.genre.router import router as genre_router
from app.crawling.router import router as crawling_router
from app.song.router import router as song_router

app = FastAPI()

app.include_router(user_router, prefix="/user", tags=["user"])
app.include_router(diary_router, prefix="/diary", tags=["diary"])
app.include_router(genre_router, prefix="/genre", tags=["genre"])
app.include_router(crawling_router, prefix="/crawl", tags=["crawl"])
app.include_router(song_router, prefix="/song", tags=["song"])

@app.get("/")
def read_root():
    return {"message": "Hello, Melog API!"}
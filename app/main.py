from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.user.router import router as user_router
from app.diary.router import router as diary_router
from app.genre.router import router as genre_router
from app.crawling.router import router as crawling_router
from app.songs.router import router as song_router
from app.embedding.router import router as embedding_router
from app.statistics.router import router as statistics_router

app = FastAPI()

app.include_router(user_router, prefix="/user", tags=["user"])
app.include_router(diary_router, prefix="/diary", tags=["diary"])
app.include_router(genre_router, prefix="/genre", tags=["genre"])
app.include_router(crawling_router, prefix="/crawl", tags=["crawl"])
app.include_router(song_router, prefix="/songs", tags=["songs"])
app.include_router(embedding_router, prefix="/embedding", tags=["embedding"])
app.include_router(statistics_router, prefix="/statistics", tags=["statistics"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "멜로그 시작 화면에 뜰 메세지에요~"}
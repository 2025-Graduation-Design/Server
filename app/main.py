from fastapi import FastAPI
from app.user.router import router as user_router  # 🐾 유저 라우터 임포트

app = FastAPI()

# 🐾 라우터 등록!
app.include_router(user_router, prefix="/user", tags=["User"])

@app.get("/")
def read_root():
    return {"message": "Hello, Melog API!"}
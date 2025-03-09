from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import redis
from app.database import get_db, get_redis, get_mongodb

router = APIRouter()

# ✅ MySQL 연결 테스트
@router.get("/test/mysql")
async def test_mysql(db: Session = Depends(get_db)):
    result = db.execute("SELECT DATABASE()").fetchone()
    return {"MySQL 연결됨": result[0]}

# ✅ Redis 연결 테스트
@router.get("/test/redis")
async def test_redis(redis_client: redis.StrictRedis = Depends(get_redis)):
    redis_client.set("test_key", "Hello, Redis!")
    value = redis_client.get("test_key")
    return {"Redis 연결 테스트": value}

# ✅ MongoDB 연결 테스트
@router.get("/test/mongodb")
async def test_mongodb(mongodb=Depends(get_mongodb)):
    collections = await mongodb.list_collection_names()
    return {"MongoDB 컬렉션 목록": collections}
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# ✅ MySQL 설정
user = os.getenv("DB_USER")
passwd = os.getenv("DB_PASSWD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

DB_URL = f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset=utf8'

# SQLAlchemy 엔진 및 세션 생성
engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI 의존성 주입을 위한 MySQL 세션 생성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Redis 설정
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Redis 클라이언트 생성
redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True  # 문자열 데이터를 반환하도록 설정
)

# FastAPI 의존성 주입을 위한 Redis 클라이언트 함수
def get_redis():
    try:
        yield redis_client
    finally:
        pass  # Redis는 별도 연결 종료가 필요 없음

# ✅ MongoDB 설정
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "melog")

class MongoDB:
    def __init__(self):
        self.client = AsyncIOMotorClient(MONGO_URI)
        self.database = self.client[MONGO_DB_NAME]

    def get_database(self):
        return self.database

# FastAPI 의존성 주입을 위한 MongoDB 연결 함수
def get_mongodb():
    mongodb = MongoDB()
    try:
        yield mongodb.get_database()
    finally:
        pass  # MongoDB는 연결을 유지하는 방식이므로 별도 종료 X
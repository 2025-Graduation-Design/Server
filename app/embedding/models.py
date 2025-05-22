import json
import logging
import torch
import re
import numpy as np
from kiwipiepy import Kiwi
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.emotion.models import tokenizer, model
from motor.motor_asyncio import AsyncIOMotorDatabase
from transformers import AutoTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

kiwi = Kiwi()
logger = logging.getLogger(__name__)

SPECIAL_SPLIT_PATTERNS = [
    r"(ㅋ+)", r"(ㅎ+)", r"(ㅠ+)", r"(ㅜ+)",
    r"(\.\.\.+)", r"(ㅡㅡ+)", r"(--+)", r"(;;+)",
    r"(!+)", r"(\?+)", r"(~+)"
]

MIN_SENTENCE_LENGTH = 3

def split_sentences(text: str):
    # 1차: kiwi로 문장 단위 분리
    sentences = [s.text.strip() for s in kiwi.split_into_sents(text)]

    # 너무 짧은 문장은 다음 문장과 병합
    refined = []
    buffer = ""
    for s in sentences:
        if len(s) < 5:
            buffer += " " + s
        else:
            if buffer:
                refined.append(buffer.strip())
                buffer = ""
            refined.append(s)
    if buffer:
        refined.append(buffer.strip())

    return refined

def split_by_special_patterns(sentence: str):
    """특정 감정 표현(ㅋㅋ, ㅠㅠ 등) 주변에서 문장 분리를 시도하되, 짧은 파편은 병합"""
    if not sentence:
        return []

    pattern = "|".join(SPECIAL_SPLIT_PATTERNS)
    split_result = re.split(f"({pattern})", sentence)

    result = []
    buffer = ""

    for part in split_result:
        if part is None or part.strip() == "":
            continue

        buffer += part
        # 만약 이게 감정 패턴이라면, 앞에 쌓인 걸 하나의 문장으로 처리
        if re.fullmatch(pattern, part):
            if len(buffer.strip()) >= MIN_SENTENCE_LENGTH:
                result.append(buffer.strip())
                buffer = ""
    # 마지막 남은 부분 처리
    if buffer.strip():
        result.append(buffer.strip())

    # 최종 공백 제거 + 너무 짧은 단위 필터링
    return [s for s in result if len(s.strip()) >= MIN_SENTENCE_LENGTH]

# KoBERT 임베딩 클래스
class KoBERTEmbedding:
    def __init__(self, fine_tuned_model=None):
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

        if fine_tuned_model is not None:
            self.model = fine_tuned_model
        else:
            self.model = BertModel.from_pretrained("skt/kobert-base-v1")

        self.model.eval()

    def get_embedding(self, text: str, decimal_places=4):
        """문장을 KoBERT 임베딩 벡터로 변환"""
        if not text.strip():
            return None

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).flatten().cpu().numpy()
        rounded = np.round(embedding, decimal_places).tolist()
        return [round(val, decimal_places) for val in rounded]

kobert = KoBERTEmbedding(fine_tuned_model=model.bert)

# MongoDB에서 가사 데이터 불러오기
async def load_songs_from_mongodb(mongodb: AsyncIOMotorDatabase):
    """MongoDB에서 id, 가사만 가져옴"""
    songs = await mongodb.song_meta.find({}, {"id": 1, "lyrics": 1}).to_list(length=None)
    return [{"song_id": song["id"], "lyrics": song["lyrics"]} for song in songs]


# MySQL에서 임베딩된 노래 목록 가져오기
def get_existing_song_ids(db: Session):
    """이미 임베딩된 song_id 목록 조회"""
    result = db.execute(text("SELECT song_id FROM songLyricsEmbedding"))
    return {str(row[0]) for row in result.fetchall()}


# 노래 임베딩 저장
def save_song_embedding(db: Session, song_id: str, embedding: list):
    """MySQL에 노래 가사 임베딩 저장"""
    embedding_json = json.dumps(embedding)
    query = text("""
        INSERT INTO songLyricsEmbedding (song_id, embedding) 
        VALUES (:song_id, :embedding)
        ON DUPLICATE KEY UPDATE embedding = :embedding
    """)
    db.execute(query, {"song_id": song_id, "embedding": embedding_json})
    db.commit()


# 일기 임베딩 저장
def save_diary_embedding(db: Session, diary_id: int, embedding: list):
    """MySQL에 일기 임베딩 저장"""
    embedding_json = json.dumps(embedding)
    query = text("""
        INSERT INTO DiaryEmbedding (diary_id, embedding) 
        VALUES (:diary_id, :embedding)
        ON DUPLICATE KEY UPDATE embedding = :embedding
    """)
    db.execute(query, {"diary_id": diary_id, "embedding": embedding_json})
    db.commit()


# 유저의 선호 장르 이름 가져오기
def get_user_preferred_genres(db: Session, user_id: int):
    """유저의 선호 장르명 리스트 반환"""
    result = db.execute(
        text("SELECT genre_id FROM userGenre WHERE user_id = :user_id"),
        {"user_id": user_id}
    )
    genre_ids = {row[0] for row in result.fetchall()}
    if not genre_ids:
        return []

    result = db.execute(
        text("SELECT id, name FROM Genre WHERE id IN :genre_ids"),
        {"genre_ids": tuple(genre_ids)}
    )
    return [row[1] for row in result.fetchall()]


# MongoDB에서 특정 장르의 노래 불러오기
async def get_songs_by_genre(mongodb: AsyncIOMotorDatabase, genre_names):
    """MongoDB에서 장르명 기준으로 곡 불러오기"""
    fields = {"id": 1, "lyrics": 1, "song_name": 1, "genre": 1,
              "album_image": 1, "artist_name_basket": 1}
    query = {
        "$or": [
            {"genre": {"$regex": genre, "$options": "i"}} for genre in genre_names
        ]
    }
    songs = await mongodb.song_meta.find(query, fields).to_list(length=None)
    return songs


# 여러 노래의 임베딩 가져오기
def get_song_embeddings(db: Session, song_ids):
    """MySQL에서 song_id들에 대한 가사 임베딩 조회"""
    result = db.execute(
        text("SELECT song_id, embedding FROM songLyricsEmbedding WHERE song_id IN :song_ids"),
        {"song_ids": tuple(song_ids)}
    )

    song_embeddings = {}
    for row in result.fetchall():
        song_id = row[0]
        song_embeddings[song_id] = json.loads(row[1])
    return song_embeddings


# 유사도 계산: 일기 ↔ 노래 가사
def calculate_similarity(diary_embedding, song_embeddings):
    """일기 임베딩과 모든 곡 임베딩 비교 → 최고 유사도 곡 리턴"""
    best_match = None
    highest_score = -1

    for song_id, embeddings in song_embeddings.items():
        sim_scores = cosine_similarity([diary_embedding], np.array(embeddings))[0]
        best_idx = np.argmax(sim_scores)
        best_score = sim_scores[best_idx]

        if best_score > highest_score:
            highest_score = best_score
            best_match = (song_id, best_idx, best_score)

    return best_match
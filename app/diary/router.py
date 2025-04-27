from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import extract, text
from app.database import get_db, get_mongodb, get_redis
from app.emotion.models import model_index_to_db_emotion_id
from app.emotion.router import predict_emotion
from app.statistics.models import EmotionStatistics
from app.user.auth import get_current_user
from app.diary.models import Diary
from app.diary.schemas import DiaryCreateRequest, DiaryUpdateRequest, DiaryResponse
from app.user.models import User
from app.embedding.models import kobert, save_diary_embedding, split_sentences, get_user_preferred_genres, \
    get_songs_by_genre, get_song_embeddings, calculate_similarity
from app.transaction import transactional_session
from typing import List
import logging
import json
import torch
import numpy as np
import heapq
import torch.nn.functional as F
from datetime import datetime

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LYRIC_WINDOW_SIZE = 3  # 가사 비교 단위 (3줄)
MIN_LYRIC_LINES = 5    # 최소 가사 줄 수 필터링

# 📝 일기 작성 API
@router.post("", response_model=DiaryResponse, status_code=201, summary="일기 작성 & 노래 추천",
             description="일기를 작성하면 자동으로 임베딩을 진행하고, 사용자의 선호 장르 내에서 가장 유사한 노래를 추천합니다.")
async def create_diary(
        diary_request: DiaryCreateRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
        mongodb=Depends(get_mongodb)
):
    """
    1. 새로운 일기를 DB에 저장
    2. Kiwi를 이용해 문장 분리 후 KoBERT로 임베딩
    3. DiaryEmbedding 테이블에 저장
    4. 유저의 선호 장르 기반으로 MongoDB에서 노래 리스트 가져오기
    5. 가사와 일기 텍스트 임베딩 값 비교 후 가장 유사한 노래 추천
    """

    with transactional_session(db) as session:
        sentences = split_sentences(diary_request.content)
        logger.info(f"[일기 문장 분리] - 원본: {diary_request.content}")
        for idx, sentence in enumerate(sentences):
            logger.info(f"    ▶ 문장 {idx + 1}: {sentence}")

        embeddings = [kobert.get_embedding(sentence) for sentence in sentences if sentence.strip()]
        if not embeddings:
            logger.warning("KoBERT 임베딩 결과가 없음")
            return {"message": "임베딩할 문장이 없습니다."}

        logger.info(f"[KoBERT 임베딩 완료] - {len(embeddings)}개 문장 처리 완료")

        # 2) 유저 선호 장르 가져오기
        user_id = current_user.id
        genre_names = get_user_preferred_genres(session, user_id)
        if not genre_names:
            logger.warning(f"유저 {user_id}의 선호 장르가 설정되지 않음")
            return {"message": "유저의 선호 장르가 설정되지 않았습니다."}

        logger.info(f"🎵 [유저 선호 장르] - {genre_names}")

        # 3) MongoDB에서 해당 장르의 노래 가져오기
        songs = await get_songs_by_genre(mongodb, genre_names)
        if not songs:
            logger.warning("해당 장르에 노래가 없음")
            return {"message": "해당 장르에 노래가 없습니다."}

        song_ids = [song["id"] for song in songs]
        logger.info(f"🎼 [가져온 노래 개수] - {len(songs)}")

        # 4) 노래 가사 임베딩 불러오기 및 유사도 계산
        song_embeddings = get_song_embeddings(session, song_ids)
        best_match = calculate_similarity(embeddings[0], song_embeddings)  # 첫 번째 문장만 비교

        if not best_match:
            logger.warning("유사한 가사를 찾을 수 없음")
            return {"message": "유사한 가사를 찾을 수 없습니다."}

        song_id, best_idx, similarity_score = best_match
        matching_song = next((song for song in songs if song["id"] == str(song_id)), None)

        if matching_song is None:
            logger.error(f"추천된 song_id {song_id}가 MongoDB에서 찾을 수 없음")
            return {"message": "추천된 노래를 찾을 수 없습니다."}

        # best_idx가 가사 범위를 벗어나지 않는지 확인
        if best_idx >= len(matching_song["lyrics"]):
            logger.error(f"best_idx {best_idx}가 가사 범위를 초과함 (가사 개수: {len(matching_song['lyrics'])})")
            return {"message": "유사한 가사를 찾을 수 없습니다."}

        start = max(0, best_idx - 1)
        end = min(len(matching_song["lyrics"]), best_idx + 2)

        context_lyrics = matching_song["lyrics"][start:end]
        best_lyric = " ".join(context_lyrics)

        # 5) 모든 과정 완료 후 일기 저장 (트랜잭션 보장)
        new_diary = Diary(
            user_id=current_user.id,
            content=diary_request.content
        )
        session.add(new_diary)
        session.commit()
        session.refresh(new_diary)

        logger.info(f"[📖 일기 저장 완료] - {new_diary.content}")

        save_diary_embedding(session, new_diary.id, embeddings)

        response_data = {
            "id": new_diary.id,
            "user_id": new_diary.user_id,
            "content": new_diary.content,
            "created_at": new_diary.created_at,
            "updated_at": new_diary.updated_at,
            "recommended_song": {
                "song_id": song_id,
                "song_name": matching_song.get("song_name", "제목 없음"),
                "best_lyric": best_lyric,
                "similarity_score": round(float(similarity_score), 4),
                "album_image": matching_song.get("album_image", "이미지 없음"),
                "artist": matching_song.get("artist_name_basket", ["아티스트 없음"]),
                "genre": matching_song.get("genre", "장르 없음")
            }
        }

        logger.info(f" [응답 데이터] - {json.dumps(response_data, ensure_ascii=False, indent=4, default=str)}")

        return response_data


@router.post("/main", response_model=DiaryResponse, status_code=201,
             summary="일기 작성 & Top-3 유사 가사 기반 노래 추천",
             description="일기를 작성하면 KoBERT 임베딩 후, 선호 장르 내에서 가사 유사도 기반으로 Top-3 노래를 추천합니다.")
async def create_diary_with_music_recommend_top3(
    diary_request: DiaryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    mongodb = Depends(get_mongodb),
    redis = Depends(get_redis)
):
    """MySQL songLyricsEmbedding 테이블 구조에 최적화된 버전 (Top-3 우선순위 큐 적용)"""

    with transactional_session(db) as session:
        emotion_id, probabilities = predict_emotion(diary_request.content)
        logger.info(f"[감정 분석 결과] 감정 ID: {emotion_id} | 신뢰도: {probabilities[emotion_id]:.4f}")
        confidence = probabilities[emotion_id]

        emotion_id_db = model_index_to_db_emotion_id[emotion_id]

        # 1) 문장 분리 + KoBERT 임베딩
        sentences = [s.strip() for s in split_sentences(diary_request.content) if s.strip()]
        if not sentences:
            raise HTTPException(status_code=400, detail="분석할 문장이 없습니다")

        embeddings = []
        for sentence in sentences:
            try:
                emb = kobert.get_embedding(sentence)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"임베딩 생성 실패: {e}")
        if not embeddings:
            raise HTTPException(status_code=500, detail="임베딩 생성 실패")

        combined_embedding = np.mean(embeddings, axis=0)

        # 2) 선호 장르 조회
        genre_names = get_user_preferred_genres(session, current_user.id)
        if not genre_names:
            raise HTTPException(status_code=400, detail="선호 장르가 설정되지 않았습니다")

        # 3) MongoDB에서 해당 장르 노래 가져오기
        songs = await get_songs_by_genre(mongodb, genre_names)
        if not songs:
            raise HTTPException(status_code=404, detail="해당 장르에 노래가 없습니다")
        logger.info(f"🎼 [가져온 노래 개수] - {len(songs)}")

        # 4) 유사도 계산 및 Top-3 추천곡 선정
        heap = []
        counter = 0
        for song in songs:
            try:
                song_id = int(song["id"])
                cache_key = f"lyrics_emb:{song_id}"
                cached = await redis.get(cache_key)

                if cached:
                    lyrics_embedding = np.array(json.loads(cached))
                else:
                    result = session.execute(
                        text("SELECT embedding FROM songLyricsEmbedding WHERE song_id = :song_id"),
                        {"song_id": song_id}
                    ).fetchone()
                    if not result:
                        continue
                    lyrics_embedding = np.array(json.loads(result[0]))
                    await redis.set(cache_key, json.dumps(lyrics_embedding.tolist()))

                if len(lyrics_embedding.shape) != 2:
                    continue

                lyrics = song.get("lyrics", [])
                if len(lyrics) < MIN_LYRIC_LINES or len(lyrics_embedding) < LYRIC_WINDOW_SIZE:
                    continue

                for i in range(len(lyrics_embedding) - LYRIC_WINDOW_SIZE + 1):
                    chunk_avg = np.mean(lyrics_embedding[i:i + LYRIC_WINDOW_SIZE], axis=0)

                    similarity = F.cosine_similarity(
                        torch.tensor(combined_embedding).unsqueeze(0),
                        torch.tensor(chunk_avg).unsqueeze(0)
                    ).item()

                    if similarity < 0.7:
                        continue

                    heapq.heappush(heap, (
                        similarity,
                        counter,
                        {
                            "song_id": song["id"],
                            "lyric_chunk": lyrics[i:i + LYRIC_WINDOW_SIZE],
                            "similarity": similarity,
                            "metadata": {
                                "song_name": song.get("song_name"),
                                "album_image": song.get("album_image"),
                                "artist": song.get("artist_name_basket", []),
                                "genre": song.get("genre")
                            }
                        }
                    ))
                    counter += 1

            except Exception as e:
                logger.error(f"노래 처리 중 오류: {e}")
                continue

        raw_top = heapq.nlargest(10, heap, key=lambda x: (x[0], x[1]))

        # 5) Top-3 유사한 노래 선택
        seen_song_ids = set()
        top_3 = []
        for sim, _, match in raw_top:
            if match["song_id"] not in seen_song_ids:
                top_3.append((sim, match))
                seen_song_ids.add(match["song_id"])
            if len(top_3) >= 3:
                break

        if not top_3:
            raise HTTPException(status_code=404, detail="적합한 노래를 찾을 수 없습니다")

        # 6) 일기 저장
        new_diary = Diary(
            user_id=current_user.id,
            content=diary_request.content,
            emotiontype_id=emotion_id_db,
            confidence=confidence,
            created_at=datetime.utcnow()
        )
        session.add(new_diary)
        session.commit()
        session.refresh(new_diary)

        # 6-1) 감정 통계 업데이트 or 추가
        existing_stat = session.query(EmotionStatistics).filter(
            EmotionStatistics.user_id == current_user.id,
            extract("year", EmotionStatistics.created_at) == new_diary.created_at.year,
            extract("month", EmotionStatistics.created_at) == new_diary.created_at.month,
            EmotionStatistics.emotiontype_id == emotion_id
        ).first()

        if existing_stat:
            existing_stat.count += 1
            existing_stat.total_diaries += 1
        else:
            new_stat = EmotionStatistics(
                user_id=current_user.id,
                year=new_diary.created_at.year,
                month=new_diary.created_at.month,
                emotiontype_id=emotion_id_db,
                quadrant=None,  # 필요한 경우 감정 ID 기반으로 매핑
                count=1,
                total_diaries=1,
                created_at=new_diary.created_at
            )
            session.add(new_stat)

        # 같은 달의 다른 감정도 있을 수 있으므로, 해당 달 전체 일기 수 기준으로 다른 감정의 total_diaries도 업데이트
        session.query(EmotionStatistics).filter(
            EmotionStatistics.user_id == current_user.id,
            extract("year", EmotionStatistics.created_at) == new_diary.created_at.year,
            extract("month", EmotionStatistics.created_at) == new_diary.created_at.month
        ).update({
            EmotionStatistics.total_diaries: EmotionStatistics.total_diaries + 1
        }, synchronize_session=False)

        # 7) DiaryEmbedding 테이블 저장
        session.execute(
            text("INSERT INTO diaryEmbedding (diary_id, embedding) VALUES (:diary_id, :embedding)"),
            {"diary_id": new_diary.id, "embedding": json.dumps(combined_embedding.tolist())}
        )

        # 8) 응답 구성
        recommended_songs = [
            {
                "song_id": match["song_id"],
                "song_name": match["metadata"]["song_name"],
                "best_lyric": " ".join(match["lyric_chunk"]),
                "similarity_score": round(float(sim), 4),
                "album_image": match["metadata"]["album_image"],
                "artist": match["metadata"]["artist"],
                "genre": match["metadata"]["genre"]
            }
            for sim, match in top_3
        ]

        response_data = {
            "id": new_diary.id,
            "user_id": new_diary.user_id,
            "content": new_diary.content,
            "emotiontype_id": emotion_id_db,
            "confidence": confidence,
            "created_at": new_diary.created_at,
            "updated_at": new_diary.updated_at,
            "recommended_songs": recommended_songs
        }

        logger.info("추천 결과: %s", json.dumps(response_data, indent=2, ensure_ascii=False, default=str))
        return response_data



@router.get("/{diary_id}", response_model=DiaryResponse,
            summary="일기 조회",
            description="특정 일기의 상세 정보를 조회합니다.")
def get_diary(
    diary_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(
        Diary.id == diary_id,
        Diary.user_id == current_user.id
    ).first()

    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    return diary

@router.get("", response_model=List[DiaryResponse],
            summary="내 일기 목록 조회",
            description="로그인한 사용자가 작성한 모든 일기를 조회합니다.")
def get_all_diaries(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diaries = db.query(Diary).filter(
        Diary.user_id == current_user.id
    ).all()

    return diaries

@router.put("/{diary_id}", response_model=DiaryResponse,
            summary="일기 수정",
            description="작성한 일기의 내용을 수정합니다.")
def update_diary(
    diary_id: int,
    diary_update: DiaryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(
        Diary.id == diary_id,
        Diary.user_id == current_user.id
    ).first()

    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    if diary_update.content:
        diary.content = diary_update.content

    db.commit()
    db.refresh(diary)

    return diary

@router.delete("/{diary_id}",
               summary="일기 삭제",
               description="작성한 일기를 삭제합니다.")
def delete_diary(
    diary_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(
        Diary.id == diary_id,
        Diary.user_id == current_user.id
    ).first()

    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    db.delete(diary)
    db.commit()

    return {"message": "일기가 삭제되었습니다!"}

@router.get("/{year}/{month}", response_model=List[DiaryResponse],
            summary="특정 연도/월의 일기 조회",
            description="입력한 연도와 월에 해당하는 일기를 모두 조회합니다.")
def get_diaries_by_month(
    year: int,
    month: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diaries = db.query(Diary).filter(
        Diary.user_id == current_user.id,
        extract("year", Diary.created_at) == year,
        extract("month", Diary.created_at) == month
    ).all()

    if not diaries:
        raise HTTPException(status_code=404, detail="해당 연도와 월에 작성된 일기가 없습니다.")

    return diaries


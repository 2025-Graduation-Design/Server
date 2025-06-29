import os
import re

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import extract, text, func
from app.database import get_db, get_mongodb, get_redis
from app.emotion.models import model_index_to_db_emotion_id, DiaryEmotionTag
from app.emotion.router import predict_emotion
from app.statistics.models import EmotionStatistics
from app.user.auth import get_current_user
from app.diary.models import Diary, RecommendedSong
from app.diary.schemas import DiaryCreateRequest, DiaryUpdateRequest, DiaryResponse, DiaryCountResponse, SongResponse, \
    DiaryPreviewResponse, RecommendSongResponse, EmotionTag
from app.user.models import User
from app.embedding.models import kobert, save_diary_embedding, split_sentences, get_user_preferred_genres, \
    get_songs_by_genre
from app.transaction import transactional_session
from typing import List, Set
import logging
import json
import torch
import numpy as np
import heapq
from datetime import datetime

router = APIRouter()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

emotion_to_genres = {
    0: ["댄스", "록/메탈"],
    1: ["댄스", "포크/블루스"],
    2: ["포크/블루스", "R&B/Soul"],
    3: ["포크/블루스", "인디음악", "R&B/Soul"],
    4: ["포크/블루스", "인디음악", "R&B/Soul"],
    5: ["발라드", "포크/블루스"],
    6: ["발라드", "R&B/Soul", "포크/블루스"],
    7: ["랩/힙합", "록/메탈"]
}

def get_recently_recommended_lyrics(session, user_id: int, limit: int = 20) -> Set[str]:
    """
    최근 일기에서 추천된 가사 블럭(best_lyric)들을 반환
    """
    subquery = (
        session.query(Diary.id)
        .filter(Diary.user_id == user_id)
        .order_by(Diary.created_at.desc())
        .limit(limit)
        .subquery()
    )

    lyrics = (
        session.query(RecommendedSong.best_lyric)
        .filter(RecommendedSong.diary_id.in_(subquery))
        .distinct()
        .all()
    )

    return set(lyric[0] for lyric in lyrics)

def normalize_lyric(text: str) -> str:
    """
    공백, 개행 등 제거해서 비교 용도로만 사용하는 정제 함수
    """
    return re.sub(r"\s+", " ", text.strip()).lower()

def is_repetitive_lyric(text: str) -> bool:
    patterns = [
        r"(라|아|어|우|으|하|랄)\1{2,}",                      # 라라라, 아아아, 하하하 등 3회 이상 반복
        r"\b(ah|oh|ha|woo|la)(\s?\1){2,}\b",                  # ah ah ah 등 3회 이상 반복
        r"^([가-힣a-zA-Z])\1{3,}$",                           # 같은 문자만 4회 이상 반복 (e.g., ㅋㅋㅋㅋ)
        r"^([가-힣a-zA-Z\s]{1,5})$",                          # 의미 없는 5자 이하 문장
    ]
    for pattern in patterns:
        if re.search(pattern, text.strip(), re.IGNORECASE):
            return True
    return False

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
    with transactional_session(db) as session:
        # 1) 문장 분리
        sentences = split_sentences(diary_request.content)
        if not sentences:
            raise HTTPException(status_code=400, detail="분석할 문장이 없습니다.")

        sentence_confidences = []
        emotion_vote_counter = {}

        logger.info("[문장별 감정 분석 시작]")
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            logger.info(f"    ▶ 문장 {idx + 1}: {sentence}")

            emotion_id, probabilities = predict_emotion(sentence)
            confidence = max(probabilities)

            logger.info(f"       ▶ 문장 {idx + 1} 예측 감정 ID: {emotion_id}, 확신도: {confidence:.4f}")
            sentence_confidences.append((sentence, emotion_id, confidence))

            # Top-3 감정 모두 누적
            probs_tensor = torch.tensor(probabilities)
            topk = torch.topk(probs_tensor, k=3)

            for i in range(3):
                emo_id = topk.indices[i].item()
                score = topk.values[i].item()

                if score < 0.05:
                    continue

                if emo_id not in emotion_vote_counter:
                    emotion_vote_counter[emo_id] = 0.0
                emotion_vote_counter[emo_id] += score

        # 확신도 총합이 가장 높은 감정 선택
        emotion_id_full = max(emotion_vote_counter.items(), key=lambda x: x[1])[0]
        confidence_full = emotion_vote_counter[emotion_id_full]
        emotion_id_db = model_index_to_db_emotion_id[emotion_id_full]

        # 3) 전체 감정 분석 결과 출력
        logger.info("[문장별 감정 통계 기반 전체 감정 분석 결과]")
        for emo_id, score in sorted(emotion_vote_counter.items(), key=lambda x: -x[1]):
            logger.info(f"    ▶ 감정 ID={emo_id}, 확신도 총합={score:.4f}")

        logger.info(f"    ▶ 최종 전체 감정 ID: {emotion_id_full}, 확신도 총합: {confidence_full:.4f}")

        # 4) Top-1 감정과 일치하는 문장 중 가장 확신도 높은 문장 선택
        top1_emotion_id = emotion_id_full  # 모델 기준 감정 ID
        filtered_sentences = [
            (sentence, emo_id, conf)
            for sentence, emo_id, conf in sentence_confidences
            if emo_id == top1_emotion_id
        ]

        if not filtered_sentences:
            raise HTTPException(status_code=500, detail="Top 감정에 해당하는 문장이 없습니다.")

        best_sentence, best_emotion_id, best_confidence = max(filtered_sentences, key=lambda x: x[2])
        logger.info(f"[Top 감정에서 가장 강한 문장 선택] {best_sentence} (감정 ID={best_emotion_id}, 확신도={best_confidence:.4f})")

        # 5) best_sentence를 KoBERT 임베딩
        combined_embedding = kobert.get_embedding(best_sentence)

        # 6) 선호 장르 조회
        genre_names = get_user_preferred_genres(session, current_user.id)
        if not genre_names:
            raise HTTPException(status_code=400, detail="선호 장르가 설정되지 않았습니다.")

        # 7) MongoDB에서 해당 장르 노래 가져오기
        songs = await get_songs_by_genre(mongodb, genre_names)
        if not songs:
            raise HTTPException(status_code=404, detail="해당 장르에 노래가 없습니다.")
        logger.info(f"🎼 [가져온 노래 개수] - {len(songs)}")

        # 8) 유사도 계산 및 Top-3 추천곡 선정
        heap = []
        counter = 0

        # 1) 곡 ID 추출
        song_id_map = {int(song["id"]): song for song in songs}
        song_ids = list(song_id_map.keys())
        cache_keys = [f"lyrics_emb:{song_id}" for song_id in song_ids]

        # 2) Redis 일괄 조회
        cached_values = await redis.mget(cache_keys)

        # 3) 유사도 계산
        combined_np = np.array(combined_embedding)  # (768,)
        for song_id, cached in zip(song_ids, cached_values):
            try:
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
                    await redis.set(f"lyrics_emb:{song_id}", json.dumps(lyrics_embedding.tolist()), ex=60*60*24*30)

                if len(lyrics_embedding.shape) != 2:
                    continue

                song = song_id_map[song_id]
                lyrics = song.get("lyrics", [])
                if len(lyrics) < 1 or len(lyrics_embedding) != len(lyrics):
                    continue

                # 4) 전체 블럭과 유사도 한 번에 계산
                dot = np.dot(lyrics_embedding, combined_np)  # (n,)
                norm_block = np.linalg.norm(lyrics_embedding, axis=1)
                norm_query = np.linalg.norm(combined_np)
                similarities = dot / (norm_block * norm_query + 1e-8)

                for idx, similarity in enumerate(similarities):
                    heapq.heappush(heap, (
                        similarity,
                        counter,
                        {
                            "song_id": song_id,
                            "lyric_chunk": [lyrics[idx]],
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

        # 이후 raw_top, top_3, recommended_songs 생성은 기존 코드 그대로 유지
        raw_top = heapq.nlargest(10, heap, key=lambda x: (x[0], x[1]))

        recent_lyrics = get_recently_recommended_lyrics(session, user_id=current_user.id)

        seen_song_ids = set()
        seen_lyrics = set()
        top_3 = []

        for sim, _, match in raw_top:
            song_id = match["song_id"]
            lyric_chunk = " ".join(match["lyric_chunk"]).strip()

            if song_id in seen_song_ids:
                continue
            if lyric_chunk in recent_lyrics:
                logger.info(f"최근 추천된 동일 가사 블럭 제외됨: {lyric_chunk}")
                continue

            top_3.append((sim, match))
            seen_song_ids.add(song_id)
            seen_lyrics.add(lyric_chunk)

            if len(top_3) >= 3:
                break

        if not top_3:
            raise HTTPException(status_code=404, detail="적합한 노래를 찾을 수 없습니다.")

        # 9) 일기 저장
        new_diary = Diary(
            user_id=current_user.id,
            content=diary_request.content,
            emotiontype_id=emotion_id_db,
            confidence=confidence_full,
            best_sentence=best_sentence,
            created_at=datetime.utcnow()
        )
        session.add(new_diary)
        session.commit()
        session.refresh(new_diary)

        save_diary_embedding(session, new_diary.id, combined_embedding)

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

        for song_data in recommended_songs:
            new_song = RecommendedSong(
                diary_id=new_diary.id,
                song_id=song_data["song_id"],  # MongoDB ID 문자열 변환
                song_name=song_data["song_name"],
                artist=song_data["artist"],
                genre=song_data["genre"],
                album_image=song_data["album_image"],
                best_lyric=song_data["best_lyric"],
                similarity_score=song_data["similarity_score"]
            )
            session.add(new_song)

        # 9-1) 감정 통계 업데이트 또는 추가
        existing_stat = session.query(EmotionStatistics).filter(
            EmotionStatistics.user_id == current_user.id,
            EmotionStatistics.year == new_diary.created_at.year,
            EmotionStatistics.month == new_diary.created_at.month,
            EmotionStatistics.emotiontype_id == emotion_id_db
        ).first()

        if existing_stat:
            existing_stat.count += 1
        else:
            new_stat = EmotionStatistics(
                user_id=current_user.id,
                year=new_diary.created_at.year,
                month=new_diary.created_at.month,
                emotiontype_id=emotion_id_db,
                quadrant=None,  # quadrant 나중에 추가할거면 매핑
                count=1,
                created_at=new_diary.created_at
            )
            session.add(new_stat)

        session.commit()

        # 10) 응답 구성
        response_data = {
            "id": new_diary.id,
            "user_id": new_diary.user_id,
            "content": new_diary.content,
            "emotiontype_id": emotion_id_db,
            "confidence": confidence_full,
            "created_at": new_diary.created_at,
            "updated_at": new_diary.updated_at,
            "recommended_songs": recommended_songs,
            "top_emotions": [{"emotion_id": emo_id, "score": round(score, 4)}
                            for emo_id, score in sorted(emotion_vote_counter.items(), key=lambda x: -x[1])[:3]]
        }

        logger.info("추천 결과: %s", json.dumps(response_data, indent=2, ensure_ascii=False, default=str))
        return response_data

@router.post("/preview", summary="일기 감정 분석 + 추천 미리보기", response_model=DiaryPreviewResponse)
async def preview_diary_with_music_recommend_top3(
    diary_request: DiaryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    mongodb = Depends(get_mongodb),
    redis = Depends(get_redis)
):
    sentences = split_sentences(diary_request.content)
    if not sentences:
        raise HTTPException(status_code=400, detail="분석할 문장이 없습니다.")

    sentence_emotions = []
    sentence_confidences = []
    emotion_vote_counter = {}

    for sentence in sentences:
        emotion_id, probabilities = predict_emotion(sentence)
        confidence = max(probabilities)

        topk = torch.topk(torch.tensor(probabilities), k=3)
        top3 = [
            {"emotion_id": topk.indices[i].item(), "score": round(topk.values[i].item(), 4)}
            for i in range(3) if topk.values[i].item() >= 0.01
        ]

        sentence_confidences.append((sentence, emotion_id, confidence))
        sentence_emotions.append({
            "sentence": sentence,
            "predicted_emotion_id": emotion_id,
            "confidence": round(confidence, 4),
            "top3": top3
        })

        for i in range(3):
            emo_id = topk.indices[i].item()
            score = topk.values[i].item()
            if score < 0.05:
                continue
            emotion_vote_counter[emo_id] = emotion_vote_counter.get(emo_id, 0) + score

    if not emotion_vote_counter:
        raise HTTPException(status_code=500, detail="감정 분석 실패")

    top1_emotion_id = max(emotion_vote_counter.items(), key=lambda x: x[1])[0]
    confidence_full = emotion_vote_counter[top1_emotion_id]
    emotion_id_db = model_index_to_db_emotion_id[top1_emotion_id]

    # Top 감정 기준 가장 강한 문장
    filtered_sentences = [
        (s, eid, c) for (s, eid, c) in sentence_confidences if eid == top1_emotion_id
    ]
    if not filtered_sentences:
        raise HTTPException(status_code=500, detail="Top 감정 문장 없음")
    best_sentence, best_emotion_id, best_confidence = max(filtered_sentences, key=lambda x: x[2])

    combined_embedding = kobert.get_embedding(best_sentence)

    genre_names = get_user_preferred_genres(db, current_user.id)
    if not genre_names:
        raise HTTPException(status_code=400, detail="선호 장르가 설정되지 않았습니다.")

    songs = await get_songs_by_genre(mongodb, genre_names)
    if not songs:
        raise HTTPException(status_code=404, detail="해당 장르에 노래가 없습니다.")

    heap = []
    counter = 0
    song_id_map = {int(song["id"]): song for song in songs}
    song_ids = list(song_id_map.keys())
    cache_keys = [f"lyrics_emb:{song_id}" for song_id in song_ids]
    cached_values = await redis.mget(cache_keys)

    combined_np = np.array(combined_embedding)
    for song_id, cached in zip(song_ids, cached_values):
        try:
            if cached:
                lyrics_embedding = np.array(json.loads(cached))
            else:
                result = db.execute(
                    text("SELECT embedding FROM songLyricsEmbedding WHERE song_id = :song_id"),
                    {"song_id": song_id}
                ).fetchone()
                if not result:
                    continue
                lyrics_embedding = np.array(json.loads(result[0]))
                await redis.set(f"lyrics_emb:{song_id}", json.dumps(lyrics_embedding.tolist()), ex=60*60*24*30)

            if len(lyrics_embedding.shape) != 2:
                continue

            song = song_id_map[song_id]
            lyrics = song.get("lyrics", [])
            if len(lyrics) < 1 or len(lyrics_embedding) != len(lyrics):
                continue

            dot = np.dot(lyrics_embedding, combined_np)
            norm_block = np.linalg.norm(lyrics_embedding, axis=1)
            norm_query = np.linalg.norm(combined_np)
            similarities = dot / (norm_block * norm_query + 1e-8)

            for idx, similarity in enumerate(similarities):
                heapq.heappush(heap, (
                    similarity,
                    counter,
                    {
                        "song_id": song_id,
                        "lyric_chunk": [lyrics[idx]],
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
            logger.error(f"[preview] 노래 유사도 처리 오류: {e}")
            continue

    raw_top = heapq.nlargest(10, heap, key=lambda x: (x[0], x[1]))
    seen_song_ids = set()
    top_3 = []
    for sim, _, match in raw_top:
        if match["song_id"] not in seen_song_ids:
            top_3.append((sim, match))
            seen_song_ids.add(match["song_id"])
        if len(top_3) >= 3:
            break

    if not top_3:
        raise HTTPException(status_code=404, detail="적합한 노래를 찾을 수 없습니다.")

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

    return {
        "id": -1,
        "user_id": current_user.id,
        "content": diary_request.content,
        "emotiontype_id": emotion_id_db,
        "confidence": confidence_full,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "recommended_songs": recommended_songs,
        "top_emotions": [
            {"emotion_id": emo_id, "score": round(score, 4)}
            for emo_id, score in sorted(emotion_vote_counter.items(), key=lambda x: -x[1])[:3]
        ],
        "best_sentence": {
            "sentence": best_sentence,
            "predicted_emotion_id": best_emotion_id,
            "confidence": round(best_confidence, 4)
        },
        "sentence_emotions": sentence_emotions
    }

@router.post("/emotion-based", response_model=DiaryResponse, status_code=201,
             summary="일기 작성 & 감정 기반 Top-3 노래 추천",
             description="일기를 작성하면 KoBERT로 감정을 분석하고, 해당 감정에 맞는 장르에서 가사 유사도를 기준으로 노래를 추천합니다.")
async def create_diary_with_emotion_based_recommendation(
    diary_request: DiaryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    mongodb = Depends(get_mongodb),
    redis = Depends(get_redis)
):
    with transactional_session(db) as session:
        sentences = split_sentences(diary_request.content)
        if not sentences:
            raise HTTPException(status_code=400, detail="분석할 문장이 없습니다.")

        sentence_confidences = []
        emotion_vote_counter = {}

        logger.info("[문장별 감정 분석 시작]")
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            logger.info(f"    ▶ 문장 {idx + 1}: {sentence}")

            emotion_id, probabilities = predict_emotion(sentence)
            confidence = max(probabilities)

            logger.info(f"       ▶ 문장 {idx + 1} 예측 감정 ID: {emotion_id}, 확신도: {confidence:.4f}")
            sentence_confidences.append((sentence, emotion_id, confidence))

            probs_tensor = torch.tensor(probabilities)
            topk = torch.topk(probs_tensor, k=3)

            for i in range(3):
                emo_id = topk.indices[i].item()
                score = topk.values[i].item()

                if score < 0.05:
                    continue

                if emo_id not in emotion_vote_counter:
                    emotion_vote_counter[emo_id] = 0.0
                emotion_vote_counter[emo_id] += score

        emotion_id_full = max(emotion_vote_counter.items(), key=lambda x: x[1])[0]
        confidence_full = emotion_vote_counter[emotion_id_full]
        emotion_id_db = model_index_to_db_emotion_id[emotion_id_full]

        logger.info("[문장별 감정 통계 기반 전체 감정 분석 결과]")
        for emo_id, score in sorted(emotion_vote_counter.items(), key=lambda x: -x[1]):
            logger.info(f"    ▶ 감정 ID={emo_id}, 확신도 총합={score:.4f}")

        logger.info(f"    ▶ 최종 전체 감정 ID: {emotion_id_full}, 확신도 총합: {confidence_full:.4f}")

        top1_emotion_id = emotion_id_full
        filtered_sentences = [
            (sentence, emo_id, conf)
            for sentence, emo_id, conf in sentence_confidences
            if emo_id == top1_emotion_id
        ]

        if not filtered_sentences:
            raise HTTPException(status_code=500, detail="Top 감정에 해당하는 문장이 없습니다.")

        best_sentence, best_emotion_id, best_confidence = max(filtered_sentences, key=lambda x: x[2])
        logger.info(f"[Top 감정에서 가장 강한 문장 선택] {best_sentence} (감정 ID={best_emotion_id}, 확신도={best_confidence:.4f})")

        combined_embedding = kobert.get_embedding(best_sentence)

        genre_names = emotion_to_genres.get(emotion_id_full)
        if not genre_names:
            raise HTTPException(status_code=400, detail="감정에 대응되는 장르가 없습니다.")

        songs = await get_songs_by_genre(mongodb, genre_names)
        if not songs:
            raise HTTPException(status_code=404, detail="해당 감정의 장르에 노래가 없습니다.")

        song_id_map = {int(song["id"]): song for song in songs}
        song_ids = list(song_id_map.keys())
        cache_keys = [f"lyrics_emb:{song_id}" for song_id in song_ids]
        cached_values = await redis.mget(cache_keys)

        combined_np = np.array(combined_embedding)
        combined_np = combined_np / (np.linalg.norm(combined_np) + 1e-8)

        all_embeddings = []
        meta_infos = []

        for song_id, cached in zip(song_ids, cached_values):
            try:
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
                    await redis.set(f"lyrics_emb:{song_id}", json.dumps(lyrics_embedding.tolist()), ex=60*60*24*30)

                if len(lyrics_embedding.shape) != 2:
                    continue

                song = song_id_map[song_id]
                lyrics = song.get("lyrics", [])
                if len(lyrics) < 1 or len(lyrics_embedding) != len(lyrics):
                    continue

                normed = lyrics_embedding / (np.linalg.norm(lyrics_embedding, axis=1, keepdims=True) + 1e-8)
                all_embeddings.append(normed)

                for idx in range(len(normed)):
                    meta_infos.append({
                        "song_id": song_id,
                        "lyric": lyrics[idx],
                        "metadata": {
                            "song_name": song.get("song_name"),
                            "album_image": song.get("album_image"),
                            "artist": song.get("artist_name_basket", []),
                            "genre": song.get("genre")
                        }
                    })
            except Exception as e:
                logger.error(f"노래 처리 중 오류: {e}")
                continue

        if not all_embeddings:
            raise HTTPException(status_code=500, detail="유사도 계산을 위한 데이터가 부족합니다.")

        E = np.vstack(all_embeddings)
        sims = np.dot(E, combined_np)

        heap = []
        for i, similarity in enumerate(sims):
            heapq.heappush(heap, (
                similarity,
                i,
                {
                    "song_id": meta_infos[i]["song_id"],
                    "lyric_chunk": [meta_infos[i]["lyric"]],
                    "similarity": float(similarity),
                    "metadata": meta_infos[i]["metadata"]
                }
            ))

        top_6_raw = heapq.nlargest(10, heap, key=lambda x: (x[0], x[1]))

        recent_lyrics = get_recently_recommended_lyrics(session, user_id=current_user.id)
        seen_song_ids = set()
        seen_lyrics = set()
        top_6 = []

        for sim, _, match in top_6_raw:
            song_id = match["song_id"]
            lyric_chunk = " ".join(match["lyric_chunk"]).strip()

            if is_repetitive_lyric(lyric_chunk):
                logger.info(f"무의미 반복 가사 제외됨: {lyric_chunk}")
                continue
            if song_id in seen_song_ids:
                continue
            if lyric_chunk in recent_lyrics:
                logger.info(f"최근 추천된 동일 가사 블럭 제외됨: {lyric_chunk}")
                continue

            top_6.append((sim, match))
            seen_song_ids.add(song_id)
            seen_lyrics.add(lyric_chunk)

            if len(top_6) >= 6:
                break

        if not top_6:
            raise HTTPException(status_code=404, detail="적합한 노래를 찾을 수 없습니다.")

        new_diary = Diary(
            user_id=current_user.id,
            content=diary_request.content,
            emotiontype_id=emotion_id_db,
            confidence=confidence_full,
            best_sentence=best_sentence,
            created_at=datetime.utcnow()
        )
        session.add(new_diary)
        session.commit()
        session.refresh(new_diary)

        for eid, score in sorted(emotion_vote_counter.items(), key=lambda x: -x[1])[:3]:
            avg_score = score / len(sentences)
            if avg_score >= 0.15:
                tag = DiaryEmotionTag(
                    diary_id=new_diary.id,
                    emotiontype_id=model_index_to_db_emotion_id[eid],
                    score=round(avg_score, 4)
                )
                session.add(tag)

        save_diary_embedding(session, new_diary.id, combined_embedding)

        recommended_songs = []
        for sim, match in top_6:
            song_data = RecommendedSong(
                diary_id=new_diary.id,
                song_id=match["song_id"],
                song_name=match["metadata"]["song_name"],
                artist=match["metadata"]["artist"],
                genre=match["metadata"]["genre"],
                album_image=match["metadata"]["album_image"],
                best_lyric=" ".join(match["lyric_chunk"]),
                similarity_score=round(float(sim), 4)
            )
            session.add(song_data)
            session.flush()

            recommended_songs.append({
                "id": song_data.id,
                "song_id": song_data.song_id,
                "song_name": song_data.song_name,
                "artist": song_data.artist,
                "genre": song_data.genre,
                "album_image": song_data.album_image,
                "best_lyric": song_data.best_lyric,
                "similarity_score": song_data.similarity_score,
            })
        # 9-1) 감정 통계 업데이트 또는 추가
        existing_stat = session.query(EmotionStatistics).filter(
            EmotionStatistics.user_id == current_user.id,
            EmotionStatistics.year == new_diary.created_at.year,
            EmotionStatistics.month == new_diary.created_at.month,
            EmotionStatistics.emotiontype_id == emotion_id_db
        ).first()

        if existing_stat:
            existing_stat.count += 1
        else:
            new_stat = EmotionStatistics(
                user_id=current_user.id,
                year=new_diary.created_at.year,
                month=new_diary.created_at.month,
                emotiontype_id=emotion_id_db,
                quadrant=None,  # quadrant 나중에 추가할거면 매핑
                count=1,
                created_at=new_diary.created_at
            )
            session.add(new_stat)

        session.commit()

        response_data = {
            "id": new_diary.id,
            "user_id": new_diary.user_id,
            "content": new_diary.content,
            "emotiontype_id": emotion_id_db,
            "confidence": confidence_full,
            "best_sentence": new_diary.best_sentence,
            "created_at": new_diary.created_at,
            "updated_at": new_diary.updated_at,
            "recommended_songs": recommended_songs[:3],
            "top_emotions": [
                {"emotion_id": eid, "score": round(score, 4)}
                for eid, score in sorted(emotion_vote_counter.items(), key=lambda x: -x[1])[:3]
            ]
        }

        logger.info("추천 결과: %s", json.dumps(response_data, indent=2, ensure_ascii=False, default=str))
        return response_data


@router.get("/{diary_id}", response_model=DiaryResponse, summary = "일기 단편 조회")
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

    recommended_songs = db.query(RecommendedSong).filter(
        RecommendedSong.diary_id == diary.id
    ).order_by(RecommendedSong.similarity_score.desc()).all()

    emotion_tags = db.query(DiaryEmotionTag).filter(DiaryEmotionTag.diary_id == diary.id).all()

    main_song = db.query(RecommendedSong).get(diary.main_recommended_song_id)

    return DiaryResponse(
        id=diary.id,
        user_id=diary.user_id,
        content=diary.content,
        emotiontype_id=diary.emotiontype_id,
        confidence=diary.confidence,
        best_sentence=diary.best_sentence,
        created_at=diary.created_at,
        updated_at=diary.updated_at,
        recommended_songs=recommended_songs,
        emotion_tags=emotion_tags,
        main_recommend_song=main_song,
        top_emotions=[]
    )

@router.get("/{diary_id}/recommended-songs", response_model=List[RecommendSongResponse],
            summary="추천 노래 조회",
            description="특정 일기에 대한 추천 노래 리스트를 조회합니다.")
def get_recommended_songs_by_diary(
    diary_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(
        Diary.id == diary_id,
        Diary.user_id == current_user.id
    ).first()

    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    songs = db.query(RecommendedSong).filter(
        RecommendedSong.diary_id == diary_id
    ).order_by(RecommendedSong.similarity_score.desc()).all()

    if not songs:
        raise HTTPException(status_code=404, detail="추천된 노래가 없습니다.")

    return songs

@router.get("", response_model=List[DiaryResponse], summary="내 일기 목록 조회", description="로그인한 사용자가 작성한 모든 일기를 조회합니다.")
def get_all_diaries(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diaries = db.query(Diary).filter(
        Diary.user_id == current_user.id
    ).all()

    results = []
    for diary in diaries:
        recommended_songs = db.query(RecommendedSong).filter(
            RecommendedSong.diary_id == diary.id
        ).order_by(RecommendedSong.similarity_score.desc()).all()

        main_song = db.query(RecommendedSong).get(diary.main_recommended_song_id)

        results.append(DiaryResponse(
            id=diary.id,
            user_id=diary.user_id,
            content=diary.content,
            emotiontype_id=diary.emotiontype_id,
            confidence=diary.confidence,
            best_sentence=diary.best_sentence,
            created_at=diary.created_at,
            updated_at=diary.updated_at,
            recommended_songs=recommended_songs,
            main_recommend_song=main_song,
            top_emotions=[]
        ))

    return results

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

@router.get("/{diary_id}/emotion-tags", response_model=List[EmotionTag],
            summary="감정 태그 조회", description="특정 일기의 감정 분석 결과(상위 3개)를 조회합니다.")
def get_emotion_tags_by_diary(
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

    tags = db.query(DiaryEmotionTag).filter(
        DiaryEmotionTag.diary_id == diary_id
    ).order_by(DiaryEmotionTag.score.desc()).all()

    return tags

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


@router.get("/{year}/{month}/count", response_model=DiaryCountResponse,
            summary="특정 연/월의 일기 개수 조회",
            description="입력한 연도와 월에 해당하는 일기 개수를 반환합니다.")
def get_diary_count_by_month(
    year: int,
    month: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="월(month)은 1~12 사이여야 합니다.")

    diary_count = db.query(func.count(Diary.id)).filter(
        Diary.user_id == current_user.id,
        extract("year", Diary.created_at) == year,
        extract("month", Diary.created_at) == month
    ).scalar()

    return DiaryCountResponse(
        user_id=current_user.id,
        year=year,
        month=month,
        diary_count=diary_count
    )

@router.put("/{diary_id}/set-main-song", status_code=200, summary = "대표 노래 선정")
async def set_main_song(
    diary_id: int,
    recommended_song_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(
        Diary.id == diary_id,
        Diary.user_id == current_user.id
    ).first()

    if not diary:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

    song = db.query(RecommendedSong).filter(
        RecommendedSong.id == recommended_song_id,
        RecommendedSong.diary_id == diary.id
    ).first()

    if not song:
        raise HTTPException(status_code=400, detail="추천곡이 일기와 일치하지 않습니다.")

    if not song.youtube_url:
        query = f"{song.song_name} {' '.join(song.artist)}"
        api_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "key": YOUTUBE_API_KEY
        }

        res = requests.get(api_url, params=params)
        data = res.json()

        if not data.get("items"):
            raise HTTPException(status_code=404, detail="YouTube 영상이 없습니다.")

        video_id = data["items"][0]["id"]["videoId"]
        song.youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    diary.main_recommended_song_id = song.id
    db.commit()

    return {
        "message": "대표 음악이 설정되었습니다.",
        "youtube_url": song.youtube_url
    }

@router.get("/{diary_id}/recommended-songs/extra", response_model=List[RecommendSongResponse],
            summary="일기 리롤 버튼",
            description="일기 리롤 가능합니다. 1회.")
def get_additional_recommended_songs(
    diary_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diary = db.query(Diary).filter(
        Diary.id == diary_id,
        Diary.user_id == current_user.id
    ).first()

    if not diary:
        raise HTTPException(404, detail="일기를 찾을 수 없습니다.")

    all_songs = db.query(RecommendedSong).filter(
        RecommendedSong.diary_id == diary_id
    ).order_by(RecommendedSong.similarity_score.desc()).all()

    if len(all_songs) <= 3:
        raise HTTPException(404, detail="추가 추천곡이 없습니다.")

    return all_songs[3:]


"""
안 씀
@router.get("/recommended-songs/{recommended_song_id}/youtube-link-direct")
def get_direct_youtube_link(
    recommended_song_id: int,
    db: Session = Depends(get_db)
):
    song = db.query(RecommendedSong).filter(RecommendedSong.id == recommended_song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="추천곡을 찾을 수 없습니다.")

    # 이미 저장된 경우 → 캐시된 URL 반환
    if song.youtube_url:
        return {
            "recommended_song_id": song.id,
            "youtube_url": song.youtube_url
        }

    # YouTube 검색 API 호출
    query = f"{song.song_name} {' '.join(song.artist)}"
    api_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 1,
        "key": YOUTUBE_API_KEY
    }

    res = requests.get(api_url, params=params)
    data = res.json()

    if not data.get("items"):
        raise HTTPException(status_code=404, detail="YouTube 영상이 없습니다.")

    video_id = data["items"][0]["id"]["videoId"]
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    song.youtube_url = youtube_url
    db.commit()

    return {
        "recommended_song_id": song.id,
        "youtube_url": youtube_url
    }
"""

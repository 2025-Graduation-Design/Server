import re

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db, get_redis
from app.genre.models import UserGenre
from app.user.models import User
from app.user.auth import (
    hash_password, verify_password, create_access_token, create_refresh_token, decode_access_token, get_current_user,
    logout
)
from app.user.schemas import UserLoginRequest, UserLoginResponse, UserResponse, UserCreateRequest, UserUpdateRequest, \
    UserPasswordUpdateRequest, UserGenreResponse, UserRegisterResponse

router = APIRouter()

@router.post("/register", response_model=UserRegisterResponse, status_code=status.HTTP_201_CREATED, summary="회원가입",
             description="사용자 정보를 기반으로 회원가입을 진행합니다. 비밀번호 보안 정책을 따릅니다.")
def register_user(user_create: UserCreateRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.user_id == user_create.user_id).first():
        raise HTTPException(status_code=409, detail="이미 존재하는 사용자 ID입니다.")

    if len(user_create.password) < 8 or \
       not re.search(r'[a-z]', user_create.password) or \
       not re.search(r'[0-9]', user_create.password) or \
       not re.search(r'[!@#$%^&*(),.?]', user_create.password):
        raise HTTPException(status_code=400, detail="비밀번호는 영문, 숫자, 특수문자를 포함해야 합니다.")

    new_user = User(
        user_id=user_create.user_id,
        password=hash_password(user_create.password),
        nickname=user_create.nickname,
        phone=user_create.phone
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserRegisterResponse(user_id=new_user.user_id, nickname=new_user.nickname, phone=new_user.phone)


@router.post("/login", response_model=UserLoginResponse, summary="로그인",
             description="ID, 비밀번호로 로그인하여 access/refresh 토큰을 발급합니다.")
def login(user_request: UserLoginRequest, db: Session = Depends(get_db), redis_client=Depends(get_redis)):
    user = db.query(User).filter(User.user_id == user_request.user_id).first()
    if not user or not verify_password(user_request.password, user.password):
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    access_token = create_access_token({"sub": user.user_id})
    refresh_token = create_refresh_token({"sub": user.user_id})
    redis_client.setex(f"refresh_token:{user.user_id}", 86400 * 14, refresh_token)

    return {"access_token": access_token, "refresh_token": refresh_token}


@router.post("/logout", summary="로그아웃", description="로그인된 사용자의 세션을 종료합니다.")
def user_logout(current_user: User = Depends(get_current_user), redis_client=Depends(get_redis)):
    return logout(current_user.user_id, redis_client)


@router.post("/check-password", summary="비밀번호 확인", description="현재 비밀번호가 맞는지 확인합니다.")
def check_password(
    user_password_update: UserPasswordUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    if not verify_password(user_password_update.password, current_user.password):
        raise HTTPException(status_code=400, detail="비밀번호가 올바르지 않습니다.")
    return {"message": "비밀번호 확인 성공!"}


@router.post("/refresh", summary="리프레시 토큰 발급", description="리프레시 토큰을 통해 access_token을 재발급합니다.")
def refresh_token(refresh_token: str, redis_client=Depends(get_redis), db: Session = Depends(get_db)):
    payload = decode_access_token(refresh_token)
    if "sub" not in payload:
        raise HTTPException(status_code=401, detail="유효하지 않은 리프레시 토큰입니다.")

    user_id = payload["sub"]
    stored = redis_client.get(f"refresh_token:{user_id}")
    if not stored or stored != refresh_token:
        raise HTTPException(status_code=401, detail="리프레시 토큰이 만료됨")

    if not db.query(User).filter(User.user_id == user_id).first():
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    return {"access_token": create_access_token({"sub": user_id})}


@router.get("/me", response_model=UserResponse, summary="내 정보 조회", description="현재 로그인된 사용자 정보 조회")
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/check/auth", summary="로그인 상태 확인", description="access_token으로 인증된 사용자 상태 확인")
def check_auth(current_user: User = Depends(get_current_user)):
    return {"message": "인증된 사용자입니다.", "user_id": current_user.user_id}


@router.put("/change/info", response_model=UserResponse, summary="유저 정보 수정", description="닉네임/전화번호 수정")
def update_user_info(
    user_update: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user_update.nickname:
        current_user.nickname = user_update.nickname
    if user_update.phone:
        current_user.phone = user_update.phone
    db.commit()
    db.refresh(current_user)
    return current_user


@router.put("/change/password", summary="비밀번호 변경", description="비밀번호를 새 값으로 변경합니다.")
def update_user_password(
    user_password_update: UserPasswordUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    password = user_password_update.password
    if len(password) < 8 or \
       not re.search(r'[a-z]', password) or \
       not re.search(r'[0-9]', password) or \
       not re.search(r'[!@#$%^&*(),.?]', password):
        raise HTTPException(status_code=400, detail="비밀번호는 영문, 숫자, 특수문자를 포함해야 합니다.")

    current_user.password = hash_password(password)
    db.commit()
    db.refresh(current_user)
    return {"message": "비밀번호 변경이 완료되었어요!"}


@router.delete("/delete", summary="회원 탈퇴", description="계정 삭제 및 리프레시 토큰 폐기")
def delete_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis)
):
    redis_client.delete(f"refresh_token:{current_user.user_id}")
    db.delete(current_user)
    db.commit()
    return {"message": "회원 탈퇴가 완료되었어요. 다음에 다시 만나요!"}


@router.post("/genre/{genre_id}", response_model=UserGenreResponse, status_code=status.HTTP_201_CREATED,
             summary="선호 장르 추가", description="사용자의 선호 음악 장르를 추가합니다.")
def add_user_genre(
    genre_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if db.query(UserGenre).filter(UserGenre.user_id == current_user.id, UserGenre.genre_id == genre_id).first():
        raise HTTPException(status_code=400, detail="이미 선호 장르로 등록된 장르입니다.")

    new_user_genre = UserGenre(user_id=current_user.id, genre_id=genre_id)
    db.add(new_user_genre)
    db.commit()
    db.refresh(new_user_genre)
    return new_user_genre


@router.get("/genre", response_model=list[UserGenreResponse], summary="선호 장르 조회", description="유저가 등록한 선호 장르 목록")
def get_user_genres(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(UserGenre).filter(UserGenre.user_id == current_user.id).all()


@router.delete("/genre/{genre_id}", summary="선호 장르 삭제", description="등록된 선호 장르를 제거합니다.")
def delete_user_genre(
    genre_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    genre = db.query(UserGenre).filter(UserGenre.user_id == current_user.id, UserGenre.genre_id == genre_id).first()
    if not genre:
        raise HTTPException(status_code=400, detail="등록되지 않은 장르입니다.")

    db.delete(genre)
    db.commit()
    return {"message": "장르가 삭제되었어요."}
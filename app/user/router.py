from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db, get_redis
from app.user.models import User
from app.user.auth import (
    hash_password, verify_password, create_access_token, create_refresh_token, decode_access_token, get_current_user, logout
)
from app.user.schemas import UserLoginRequest, UserLoginResponse, UserResponse, UserCreateRequest

router = APIRouter()

# 📝 회원가입 API
@router.post("/register", status_code=status.HTTP_201_CREATED, summary="회원가입", description="새로운 사용자를 등록합니다.")
def register_user(user: UserCreateRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.user_id == user.user_id).first()
    if existing_user:
        raise HTTPException(status_code=409, detail="이미 존재하는 사용자 ID입니다.")

    hashed_pw = hash_password(user.password)
    new_user = User(user_id=user.user_id, password=hashed_pw, nickname=user.nickname, phone=user.phone)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "회원가입 성공!", "user_id": new_user.user_id}

# 📝 로그인 API
@router.post("/login", summary="로그인", description="로그인 진행", response_model=UserLoginResponse)
def login(user_request: UserLoginRequest, db: Session = Depends(get_db), redis_client=Depends(get_redis)):
    db_user = db.query(User).filter(User.user_id == user_request.user_id).first()
    if not db_user or not verify_password(user_request.password, db_user.password):
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    access_token = create_access_token({"sub": db_user.user_id})
    refresh_token = create_refresh_token({"sub": db_user.user_id})

    redis_client.setex(f"refresh_token:{db_user.user_id}", 86400 * 14, refresh_token)  # 리프레시 토큰 14일 유지

    return {"access_token": access_token, "refresh_token": refresh_token}

# 📝 로그아웃 API
@router.post("/logout", summary="로그아웃", description="로그아웃 진행")
def user_logout(current_user: User = Depends(get_current_user), redis_client=Depends(get_redis)):
    return logout(current_user.user_id, redis_client)

# 📝 액세스 토큰 갱신 API
@router.post("/refresh", summary="리프레시 토큰 발급", description="리프레시 토큰 재발급 로직")
def refresh_token(refresh_token: str, redis_client=Depends(get_redis), db: Session = Depends(get_db)):
    payload = decode_access_token(refresh_token)
    if "sub" not in payload:
        raise HTTPException(status_code=401, detail="유효하지 않은 리프레시 토큰입니다.")

    user_id = payload["sub"]
    stored_refresh_token = redis_client.get(f"refresh_token:{user_id}")

    if stored_refresh_token is None or stored_refresh_token != refresh_token:
        raise HTTPException(status_code=401, detail="리프레시 토큰이 만료됨")

    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    new_access_token = create_access_token({"sub": db_user.user_id})

    return {"access_token": new_access_token}

# 📝 현재 로그인된 사용자 정보 조회
@router.get("/me", response_model=UserResponse, summary="내 정보 조회", description="현재 로그인된 사용자의 정보를 조회")
def me(current_user: User = Depends(get_current_user)):
    return current_user
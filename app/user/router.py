from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.user.models import User
from app.user.auth import hash_password, verify_password, create_access_token, decode_access_token, get_current_user
from app.user.schemas import UserResponse, UserCreateRequest, UserLoginRequest

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
@router.post("/login", summary="로그인", description="사용자 로그인 후 JWT 액세스 토큰을 발급")
def login(user: UserLoginRequest, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_id == user.user_id).first()

    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    access_token = create_access_token({"sub": db_user.user_id})
    return {"access_token": access_token, "token_type": "bearer"}

# 📝 내 정보 조회 API
@router.get("/me", response_model=UserResponse, summary="내 정보 조회", description="현재 로그인된 사용자의 정보를 조회")
def me(current_user: User = Depends(get_current_user)):
    return current_user

# 📝 토큰 갱신 API
@router.post("/refresh", summary="액세스 토큰 갱신", description="리프레시 토큰을 사용하여 새로운 액세스 토큰을 발급")
def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    payload = decode_access_token(refresh_token)
    if payload is None or "sub" not in payload:
        raise HTTPException(status_code=401, detail="refresh token을 다시 발급받으세요.")

    user_id = payload["sub"]
    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User를 찾을 수 없습니다.")

    new_access_token = create_access_token({"sub": db_user.user_id})
    return {"access_token": new_access_token, "token_type": "bearer"}
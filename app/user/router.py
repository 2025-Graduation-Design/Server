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
    UserPasswordUpdateRequest, UserGenreResponse

router = APIRouter()

@router.post(
    "/register", status_code=status.HTTP_201_CREATED, summary="회원가입",
    description="""
    새로운 사용자를 등록합니다.

    - `user_id`: 유저 ID (중복 불가)
    - `password`: 비밀번호 (8자 이상, 영문+숫자+특수문자 포함)
    - `nickname`: 유저 닉네임
    - `phone`: 전화번호 (선택 사항)

    제약 조건:
    - 비밀번호가 보안 규칙에 맞지 않으면 400 오류 발생(특수문자 허용 :[!@#$%^&*(),.?])
    - ID가 이미 존재하면 409 오류 발생
    """,
    response_model=UserResponse,
    responses={
        201: {"description": "created", "content": {"application/json": {"example": {"user_id": "duehee", "nickname": "듀히", "phone": "010-1234-5678"}}}},
        400: {"description": "Bad Request", "content": {"application/json": {"example": {"detail": "비밀번호는 최소 8자 이상이어야 합니다."}}}},
        409: {"description": "duplicated", "content": {"application/json": {"example": {"detail": "이미 존재하는 사용자 ID입니다."}}}},
    }
)
def register_user(user_create: UserCreateRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.user_id == user_create.user_id).first()
    if existing_user:
        raise HTTPException(status_code=409, detail="이미 존재하는 사용자 ID입니다.")

    if len(user_create.password) < 8:
        raise HTTPException(status_code=400, detail="비밀번호는 최소 8자 이상이어야 합니다.")

    if not re.search(r'[a-z]', user_create.password) or not re.search(r'[0-9]'
            , user_create.password) or not re.search(r'[!@#$%^&*(),.?]', user_create.password):
        raise HTTPException(status_code=400, detail="비밀번호는 영문, 숫자, 특수문자를 포함해야 합니다.")

    hashed_pw = hash_password(user_create.password)
    new_user = User(user_id=user_create.user_id, password=hashed_pw,
                    nickname=user_create.nickname, phone=user_create.phone)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserResponse(
        user_id=new_user.user_id,
        nickname=new_user.nickname,
        phone=new_user.phone
    )

@router.post("/login", summary="로그인", description="""
    사용자 로그인 후 **JWT 액세스 토큰과 리프레시 토큰**을 발급합니다.

    - `user_id`: 유저 ID  
    - `password`: 비밀번호  

    **응답**  
    - 성공 시, `access_token`과 `refresh_token` 반환  
    - 아이디 또는 비밀번호가 틀리면 400 오류 발생  
    """,
    response_model=UserLoginResponse)
def login(user_request: UserLoginRequest, db: Session = Depends(get_db), redis_client=Depends(get_redis)):
    db_user = db.query(User).filter(User.user_id == user_request.user_id).first()
    if not db_user or not verify_password(user_request.password, db_user.password):
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    access_token = create_access_token({"sub": db_user.user_id})
    refresh_token = create_refresh_token({"sub": db_user.user_id})

    redis_client.setex(f"refresh_token:{db_user.user_id}", 86400 * 14, refresh_token)  # 리프레시 토큰 14일 유지

    return {"access_token": access_token, "refresh_token": refresh_token}

@router.post("/logout", summary="로그아웃", description="""
    현재 로그인된 사용자의 세션을 종료합니다.

    **기능**  
    - Redis에서 `refresh_token`을 삭제하여 강제 로그아웃  
    - 이후 `refresh_token`을 사용한 토큰 갱신 불가능  
    """)
def user_logout(current_user: User = Depends(get_current_user), redis_client=Depends(get_redis)):
    return logout(current_user.user_id, redis_client)

@router.post("/check-password", summary="비밀번호 확인", description="""
    현재 사용자의 비밀번호가 올바른지 확인합니다.

    **기능**  
    - 비밀번호가 맞으면 `비밀번호 확인 성공!` 반환  
    - 틀리면 400 오류 발생  
    """)
def check_password(
    user_password_update: UserPasswordUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    if not verify_password(user_password_update.password, current_user.password):
        raise HTTPException(status_code=400, detail="비밀번호가 올바르지 않습니다.")

    return {"message": "비밀번호 확인 성공!"}

@router.post("/refresh", summary="리프레시 토큰 발급", description="""
    리프레시 토큰을 사용하여 **새로운 액세스 토큰**을 발급합니다.

    **기능**  
    - 기존 `refresh_token`이 유효한 경우, 새로운 `access_token`을 발급  
    - `refresh_token`이 만료되었거나 올바르지 않으면 401 오류 발생  
    """)
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

@router.get("/me", response_model=UserResponse, summary="내 정보 조회", description="""
    현재 로그인된 사용자의 정보를 조회합니다.

    **기능**  
    - `access_token`을 이용해 인증된 사용자 정보를 반환  
    - 로그인하지 않은 경우 401 오류 발생  
    """)
def me(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/check/auth", summary="로그인 상태 확인", description="""
    현재 로그인 상태를 확인합니다.

    **기능**  
    - `access_token`이 유효하면 `인증된 사용자입니다.` 응답  
    - 인증되지 않은 경우 401 오류 발생  
    """)
def check_auth(current_user: User = Depends(get_current_user)):
    return {"message": "인증된 사용자입니다.", "user_id": current_user.user_id}

@router.put("/change/info", response_model=UserResponse, summary="유저 정보 수정", description="""
    현재 로그인된 사용자의 정보를 수정합니다.

    **변경 가능 필드**  
    - `nickname`: 닉네임 변경  
    - `phone`: 전화번호 변경  

    **기능**  
    - 수정 완료 후, 변경된 사용자 정보 반환  
    """
)
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

@router.put("/change/password", summary = "유저 비밀번호 변경", description="""
    현재 사용자의 비밀번호를 변경합니다.

    **제약 조건**  
    - 최소 8자 이상, 영문+숫자+특수문자 포함  
    - 변경 후, 새로운 비밀번호로 로그인 필요  

    **응답**  
    - 성공 시 `"비밀번호 변경이 완료되었어요!"` 반환  
    - 비밀번호 규칙을 지키지 않으면 400 오류 발생  
    """)
def update_user_password(
        user_password_update: UserPasswordUpdateRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if len(user_password_update.password) < 8:
        raise HTTPException(status_code=400, detail="비밀번호는 최소 8자 이상이어야 합니다.")

    if not re.search(r'[a-z]', user_password_update.password) or not re.search(r'[0-9]'
            , user_password_update.password) or not re.search(r'[!@#$%^&*(),.?]', user_password_update.password):
        raise HTTPException(status_code=400, detail="비밀번호는 영문, 숫자, 특수문자를 포함해야 합니다.")

    hashed_pw = hash_password(user_password_update.password)

    current_user.password = hashed_pw
    db.commit()
    db.refresh(current_user)

    return {"message": "비밀번호 변경이 완료되었어요!"}

@router.delete("/delete", summary="회원 탈퇴", description="""
    현재 사용자의 계정을 삭제합니다.

    **기능**  
    - 계정 정보 및 `refresh_token` 삭제  
    - 이후 로그인 불가, 모든 데이터 삭제됨  
    - 성공 시 `"회원 탈퇴가 완료되었어요. 다음에 다시 만나요!"` 응답  
    """)
def delete_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis)
):
    redis_client.delete(f"refresh_token:{current_user.user_id}")

    db.delete(current_user)
    db.commit()

    return {"message": "회원 탈퇴가 완료되었어요. 다음에 다시 만나요!"}

@router.post("/genre/{genre_id}", response_model=UserGenreResponse, status_code=status.HTTP_201_CREATED, summary="사용자 선호 장르 추가", description="사용자가 선호하는 음악 장르를 추가합니다.")
def add_user_genre(
    genre_id = int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    existing_user_genre = db.query(UserGenre).filter(
        UserGenre.user_id == current_user.id,
        UserGenre.genre_id == genre_id
    ).first()

    if existing_user_genre:
        raise HTTPException(status_code=400, detail="이미 선호 장르로 등록된 장르입니다.")

    new_user_genre = UserGenre(user_id=current_user.id, genre_id=genre_id)
    db.add(new_user_genre)
    db.commit()
    db.refresh(new_user_genre)

    return new_user_genre

@router.get("/genre", response_model=list[UserGenreResponse], summary="사용자 선호 장르 조회",
            description="사용자가 등록한 선호 음악 장르 목록을 조회합니다.")
def get_user_genres(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_genres = db.query(UserGenre).filter(UserGenre.user_id == current_user.id).all()
    return user_genres

@router.delete("/genre/{genre_id}", summary="사용자 선호 장르 삭제", description="사용자가 등록한 선호 음악 장르를 삭제합니다.")
def delete_user_genre(
        genre_id = int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    existing_user_genre = db.query(UserGenre).filter(
        UserGenre.user_id == current_user.id,
        UserGenre.genre_id == genre_id
    ).first()

    if not existing_user_genre:
        raise HTTPException(status_code=400, detail="등록되지 않은 장르입니다.")

    db.delete(existing_user_genre)
    db.commit()

    return {"messages" : "장르가 삭제되었어요."}
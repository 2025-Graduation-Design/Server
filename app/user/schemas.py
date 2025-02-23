from pydantic import BaseModel
from typing import Optional

# 회원가입 요청 모델
class UserCreateRequest(BaseModel):
    user_id: str
    password: str
    nickname: str
    phone: Optional[str] = None

# 로그인 요청 모델
class UserLoginRequest(BaseModel):
    user_id: str
    password: str

# 유저 응답 모델
class UserResponse(BaseModel):
    user_id: str
    nickname: str
    phone: Optional[str] = None

    class Config:
        orm_mode = True
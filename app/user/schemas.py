from datetime import datetime

from pydantic import BaseModel
from typing import Optional, List

from app.genre.schemas import GenreResponse, UserGenreInfoResponse


# Request
class UserCreateRequest(BaseModel):
    user_id: str
    password: str
    nickname: str
    phone: Optional[str] = None

class UserLoginRequest(BaseModel):
    user_id: str
    password: str

class UserPasswordRequest(BaseModel):
    password: str

class UserUpdateRequest(BaseModel):
    nickname: Optional[str] = None
    phone: Optional[str] = None

class UserPasswordUpdateRequest(BaseModel):
    password: str

# Response
class UserRegisterResponse(BaseModel):
    user_id: str
    nickname: str
    phone: Optional[str]

class UserLoginResponse(BaseModel):
    access_token: str
    refresh_token: str

class UserGenreResponse(BaseModel):
    id: int
    user_id: int
    genre: GenreResponse
    created_at: datetime

    class Config:
        orm_mode = True

class UserResponse(BaseModel):
    user_id: str
    nickname: str
    phone: Optional[str] = None
    user_genres: List[UserGenreInfoResponse]

    class Config:
        orm_mode = True


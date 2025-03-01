from pydantic import BaseModel
from datetime import datetime

class GenreResponse(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True

class UserGenreResponse(BaseModel):
    id: int
    user_id: int
    genre: GenreResponse
    created_at: datetime

    class Config:
        orm_mode = True
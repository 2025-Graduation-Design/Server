from pydantic import BaseModel

class GenreResponse(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True

class UserGenreInfoResponse(BaseModel):
    genre: GenreResponse

    class Config:
        orm_mode = True
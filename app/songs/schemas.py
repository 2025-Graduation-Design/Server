from pydantic import BaseModel

from typing import List

class SongResponse(BaseModel):
    id: str
    title: str
    artist: str
    genre: str
    lyrics: List[str]

    class Config:
        orm_mode = True
from pydantic import BaseModel

from typing import List

class SongResponse(BaseModel):
    id: str
    title: str
    artist: str
    genre: str
    lyrics: List[str]
    album_id: str
    album_name: str
    album_image: str
    uri: str

    class Config:
        orm_mode = True
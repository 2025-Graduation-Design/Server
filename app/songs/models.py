from pydantic import BaseModel

class Song(BaseModel):
    id: str
    song_name: str
    artist_name_basket: list
    album_id: str
    album_name: str
    lyrics: str
    uri: str

    class Config:
        from_attributes = True


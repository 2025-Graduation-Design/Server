from pydantic import BaseModel
from typing import List, Dict

class EmbeddingResponse(BaseModel):
    total_songs: int
    embedded_songs: int
    processed_details: List[Dict[str, str]]
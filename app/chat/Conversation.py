from datetime import datetime
from typing import List
from pydantic import BaseModel

from app.chat.Message import Message

class Conversation(BaseModel):
    id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
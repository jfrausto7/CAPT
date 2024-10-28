from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.conversation_manager import ConversationManager


app = FastAPI()
conversation_manager = ConversationManager()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class MessageRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None

# API endpoints
@app.post("/api/messages")
async def create_message(message_request: MessageRequest):
    try:
        response = await conversation_manager.create_message(
            text=message_request.text,
            conversation_id=message_request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success"}
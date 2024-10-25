from datetime import datetime
import sqlite3
import uuid
from contextlib import contextmanager
from typing import List, Optional, Dict
from pydantic import BaseModel

from agents.TherapyAgent import TherapyAgent

# Pydantic models for type safety
class Message(BaseModel):
    text: str
    sender: str
    timestamp: datetime = datetime.now()

class Conversation(BaseModel):
    id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime

class ConversationManager:
    def __init__(self, db_path: str = "data/cache_therapy_chat.db"):
        self.db_path = db_path
        self.init_db()
        self.setup_llm()

    def init_db(self):
        with self.get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    text TEXT,
                    sender TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)

    def setup_llm(self):
        self.llm = TherapyAgent(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            temperature=0.3,
            max_tokens=512
        )

    @contextmanager
    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_conversation_messages(self, conn, conversation_id: str) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def format_prompt(self, conversation_history: List[dict], user_message: str) -> str:
        # Format conversation history into a prompt
        history = "\n".join([
            f"{msg['sender']}: {msg['text']}" 
            for msg in conversation_history[-5:]  # Last 5 messages for context
        ])

        # TODO: DO NOT USE THIS PROMPT! IT WILL STILL OVERCHARGE (FIGURE OUT CONTEXT WINDOW LENGTH ABOVE.)
        
        prompt = f"""You are CAPT, a compassionate and skilled therapist specializing in psychedelic-assisted therapy. 
        Your responses should be empathetic, non-judgmental, and focused on creating a safe space for clients.
        Always maintain professional boundaries and ethical guidelines.

        Previous conversation:
        {history}

        Client: {user_message}

        Respond as the therapist:"""
        
        return prompt

    async def get_llm_response(self, conversation_history: List[dict], user_message: str) -> str:
        prompt = self.format_prompt(conversation_history, user_message)
        response = await self.llm.complete(prompt)
        return response.strip()

    async def create_message(self, text: str, conversation_id: Optional[str] = None) -> Dict:
        with self.get_db() as conn:
            # Get or create conversation
            conv_id = conversation_id or str(uuid.uuid4())
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conv_id, datetime.now(), datetime.now())
            )
            
            # Create user message
            user_message = Message(
                text=text,
                sender="user"
            )
            
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, text, sender, timestamp) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conv_id, user_message.text, user_message.sender, user_message.timestamp)
            )
            
            # Get conversation history and LLM response
            messages = self.get_conversation_messages(conn, conv_id)
            llm_response_text = await self.get_llm_response(messages, text)
            
            # Create therapist message
            therapist_message = Message(
                text=llm_response_text,
                sender="therapist"
            )
            
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, text, sender, timestamp) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conv_id, therapist_message.text, therapist_message.sender, therapist_message.timestamp)
            )
            
            # Update conversation timestamp
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.now(), conv_id)
            )
            
            conn.commit()
            
            return {
                "conversation_id": conv_id,
                "message": user_message.dict(),
                "response": therapist_message.dict()
            }

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation = cursor.fetchone()
            
            if not conversation:
                return None
                
            messages = self.get_conversation_messages(conn, conversation_id)
            
            return {
                "id": conversation["id"],
                "messages": messages,
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"]
            }

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cursor.rowcount > 0
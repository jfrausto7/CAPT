from datetime import datetime
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import List, Optional, Dict
from langchain.chains import RetrievalQA
from langchain_together import Together

from agents.TherapyAgent import TherapyAgent
from agents.llm_integration import IntentClassifier
from agents.retrieval.MultiVectorstoreRetriever import MultiVectorstoreRetriever
from app.chat.ContextManager import ContextManager, ContextWindow, ConversationContext
from app.chat.Message import Message
from app.chat.Conversation import Conversation
from app.safety_mechanisms import SafetyMechanisms
from config import RATE_LIMIT_BREAK

class ConversationManager:
    def __init__(self, db_path: str = "data/cache_therapy_chat.db"):
        self.db_path = db_path
        self.max_tokens = 512
        self.init_db()
        self.setup_agents()
        self.safety_filter = SafetyMechanisms(self.therapy_agent)
        self.is_terminated = False
        self.context_manager = ContextManager(self.therapy_agent)
        self.contexts: Dict[str, ConversationContext] = {}

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

    def get_or_create_context(self, conversation_id: str) -> ConversationContext:
        """Get existing context or create new one for a conversation."""
        if conversation_id not in self.contexts:
            self.contexts[conversation_id] = ConversationContext(
                critical_context=[],
                therapeutic_context=ContextWindow(
                    messages=[],
                    summary="",
                    key_points=[],
                    emotional_state="",
                    therapy_goals=[],
                    last_summary_index=0
                ),
                background_context=""
            )
        return self.contexts[conversation_id]

    def setup_agents(self):
        # Initialize both agents
        self.therapy_agent = TherapyAgent(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            temperature=0.3,
            max_tokens=self.max_tokens
        )

        self.retrieval_model = Together(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            temperature=0.3,
            max_tokens=256
        )

        self.qa_chain = self.create_RAG_retrieval_chain(
            vector_dir="app/vectorstore/store",
            model=self.retrieval_model
        )
        
        self.intent_classifier = IntentClassifier(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            temperature=0.3,
            max_tokens=64,
            qa_chain=self.qa_chain
        )

    @contextmanager
    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            text=row['text'],
            sender=row['sender'],
            timestamp=datetime.fromisoformat(row['timestamp'])
        )

    def get_conversation_messages(self, conn, conversation_id: str) -> List[Message]:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def format_therapy_prompt(self, context:str, user_message: str, is_escalated: bool = False) -> str:
        prompt = f"""You are CAPT, a compassionate and skilled therapist specializing in psychedelic-assisted therapy. 
        Your responses should be empathetic, non-judgmental, and focused on creating a safe space for clients.
        Always maintain professional boundaries and ethical guidelines.

        {context}

        Client message: {user_message}

        Respond as the therapist in {self.max_tokens} tokens or less. ONLY PROVIDE THE RESPONSE ITSELF, NOTHING ELSE:"""
        
        if is_escalated:
            prompt = f"""IMPORTANT: This conversation requires additional care and sensitivity.
            Maintain a calm, supportive tone. Focus on safety and well-being.
            If appropriate, gently suggest professional resources or support services.
            
            {prompt}"""
            
        return prompt

    async def get_response(self, conversation_history: List[Message], user_message: str, conversation_id: Optional[str] = None) -> str:
        # Create message and conversation objects for safety processing
        current_message = Message(
            text=user_message,
            sender="user",
            timestamp=datetime.now()
        )

        # Get or create conversation context
        context = self.get_or_create_context(conversation_id)

        current_conversation = Conversation(
            id="",
            messages=conversation_history,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Process message through safety filter
        is_escalated, is_terminated = self.safety_filter.process_message(current_message, current_conversation)
        
        if is_terminated or self.is_terminated:
            self.is_terminated = True
            context.critical_context.append({
                "type": "safety_termination",
                "timestamp": datetime.now().isoformat(),
                "reason": "Crisis intervention required"
            })
            return """I need to pause our conversation here. Your safety is my top priority, and I'm concerned about what you've shared. Please immediately contact emergency services or a crisis hotline:
            
            \n National Suicide Prevention Lifeline: 988
            \n Crisis Text Line: Text HOME to 741741
            \n Fireside Project: Text or call 62-FIRESIDE (623-473-7433)
            
            \n These services are available 24/7 and have trained professionals who can provide the immediate support you need."""

        # If not terminated, process through intent classifier
        intent_response = await self.intent_classifier.complete(user_message)
        time.sleep(RATE_LIMIT_BREAK)

        # Update context with new message
        context = await self.context_manager.update_context(context, [current_message])
        formatted_context = self.context_manager.format_context_for_prompt(context)
        
        # RAG
        if intent_response is not None and intent_response['result'] != " I don't know.":
            # For RAG-based responses, still check if we need to add safety messaging
            response = intent_response['result']
            if is_escalated:
                response += "\n\nIf you're feeling overwhelmed or need immediate support, please don't hesitate to reach out to mental health professionals or emergency services."
                # Add critical context for safety termination
                context.critical_context.append({
                    "type": "safety_termination",
                    "timestamp": datetime.now().isoformat(),
                    "reason": "Crisis intervention required"
                })
            
            # Store therapy response in context
            therapeutic_window = context.therapeutic_context
            therapeutic_window.messages.append(Message(
                text=response,
                sender="assistant",
                timestamp=datetime.now()
            ))
            return response
        # Default Fallback TherapyAgent
        else:
            # Default to therapy agent with safety-aware prompt
            time.sleep(RATE_LIMIT_BREAK)
            prompt = self.format_therapy_prompt(formatted_context, user_message, is_escalated)

            # Get response from therapy agent
            response = (await self.therapy_agent.complete(prompt)).strip()
            
            # Store therapy response in context
            therapeutic_window = context.therapeutic_context
            therapeutic_window.messages.append(Message(
                text=response,
                sender="assistant",
                timestamp=datetime.now()
            ))
            
            # Update emotional state if needed
            if len(therapeutic_window.messages) % 3 == 0:  # Update every 3 messages
                therapeutic_window.emotional_state = await self.context_manager.assess_emotional_state(
                    therapeutic_window.messages[-3:]
                )
            
            return response

    async def create_message(self, text: str, conversation_id: Optional[str] = None) -> Dict:
        with self.get_db() as conn:
            conv_id = conversation_id or str(uuid.uuid4())
            current_time = datetime.now()
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conv_id, current_time, current_time)
            )
            
            user_message = Message(
                text=text,
                sender="user",
                timestamp=current_time
            )
            
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, text, sender, timestamp) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conv_id, user_message.text, user_message.sender, user_message.timestamp)
            )
            
            messages = self.get_conversation_messages(conn, conv_id)
            response_text = await self.get_response(messages, text, conv_id)
            
            response_message = Message(
                text=response_text,
                sender="assistant",
                timestamp=datetime.now()
            )
            
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, text, sender, timestamp) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conv_id, response_message.text, response_message.sender, response_message.timestamp)
            )
            
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.now(), conv_id)
            )
            
            conn.commit()
            
            return {
                "conversation_id": conv_id,
                "message": user_message.model_dump(),
                "response": response_message.model_dump()
            }
    
    def create_RAG_retrieval_chain(
        self,
        vector_dir: str,
        model,
        show_progress: bool = True
    ):
        # Initialize retriever
        retriever = MultiVectorstoreRetriever(
            vector_dir=vector_dir,
            k=10,  # Number of documents to retrieve per query
            score_threshold=0.7,  # Minimum similarity score
            show_progress=show_progress
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",  # or "map_reduce" for longer contexts
            retriever=retriever,
            return_source_documents=True  # Include source docs in response
        )

        return qa_chain

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation_row = cursor.fetchone()
            
            if not conversation_row:
                return None
                
            messages = self.get_conversation_messages(conn, conversation_id)
            
            return Conversation(
                id=conversation_row["id"],
                messages=messages,
                created_at=datetime.fromisoformat(conversation_row["created_at"]),
                updated_at=datetime.fromisoformat(conversation_row["updated_at"])
            )

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cursor.rowcount > 0
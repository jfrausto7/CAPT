from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

from app.chat.Message import Message

@dataclass
class ContextWindow:
    """Represents a dynamic context window with relevant conversation history."""
    messages: List[Message]
    summary: str
    key_points: List[str]
    emotional_state: str
    therapy_goals: List[str]
    last_summary_index: int

class ContextType(Enum):
    CRITICAL = "critical"
    THERAPEUTIC = "therapeutic"
    BACKGROUND = "background"

@dataclass
class ConversationContext:
    """Stores different types of context with varying retention priorities."""
    critical_context: List[Dict]  # Never summarized, always retained
    therapeutic_context: ContextWindow  # Dynamically summarized
    background_context: str  # Long-term summary

class ContextManager:
    def __init__(self, model, max_window_size: int = 10):
        self.model = model
        self.max_window_size = max_window_size
        self.summarization_threshold = max_window_size // 2

    async def create_summary(self, messages: List[Message]) -> str:
        """Create a concise summary of the conversation segment."""
        messages_text = "\n".join([f"{msg.sender}: {msg.text}" for msg in messages])
        prompt = f"""Summarize the key points of this therapy conversation, focusing on:
        1. Main themes and concerns discussed
        2. Client's emotional state
        3. Any progress or insights gained
        4. Action items or homework discussed
        
        Conversation:
        {messages_text}
        
        Provide a concise summary:"""
        
        return await self.model.complete(prompt)

    async def extract_key_points(self, messages: List[Message]) -> List[str]:
        """Extract critical information that should be retained."""
        messages_text = "\n".join([f"{msg.sender}: {msg.text}" for msg in messages])
        prompt = f"""Identify the most important therapeutic information from this conversation:
        - Triggers or risk factors
        - Therapeutic goals
        - Key breakthroughs
        - Safety concerns
        - Coping strategies discussed
        
        Conversation:
        {messages_text}
        
        List the key points:"""
        
        response = await self.model.complete(prompt)
        return [point.strip() for point in response.split('\n') if point.strip()]

    async def assess_emotional_state(self, messages: List[Message]) -> str:
        """Assess the client's current emotional state."""
        recent_messages = messages[-3:]  # Focus on most recent messages
        messages_text = "\n".join([f"{msg.sender}: {msg.text}" for msg in recent_messages])
        prompt = f"""Assess the client's current emotional state based on their recent messages:
        
        {messages_text}
        
        Provide a brief emotional state assessment:"""
        
        return await self.model.complete(prompt)

    async def update_context(self, context: ConversationContext, new_messages: List[Message]) -> ConversationContext:
        """Update the conversation context with new messages."""
        therapeutic_window = context.therapeutic_context
        therapeutic_window.messages.extend(new_messages)
        
        # Check if we need to summarize
        if len(therapeutic_window.messages) >= self.max_window_size:
            # Create new summary
            new_summary = await self.create_summary(therapeutic_window.messages)
            key_points = await self.extract_key_points(therapeutic_window.messages)
            emotional_state = await self.assess_emotional_state(therapeutic_window.messages)
            
            # Keep only recent messages
            therapeutic_window.messages = therapeutic_window.messages[-self.summarization_threshold:]
            therapeutic_window.summary = new_summary
            therapeutic_window.key_points = key_points
            therapeutic_window.emotional_state = emotional_state
            therapeutic_window.last_summary_index = len(therapeutic_window.messages)
        
        return context

    def format_context_for_prompt(self, context: ConversationContext) -> str:
        """Format the context for inclusion in the therapy prompt."""
        therapeutic_window = context.therapeutic_context
        
        prompt_parts = []
        
        # Add critical context
        if context.critical_context:
            prompt_parts.append("Important Context:")
            for item in context.critical_context:
                prompt_parts.append(f"- {item}")
        
        # Add conversation summary if available
        if therapeutic_window.summary:
            prompt_parts.append("\nConversation Summary:")
            prompt_parts.append(therapeutic_window.summary)
        
        # Add key therapeutic points
        if therapeutic_window.key_points:
            prompt_parts.append("\nKey Points:")
            for point in therapeutic_window.key_points:
                prompt_parts.append(f"- {point}")
        
        # Add emotional state
        if therapeutic_window.emotional_state:
            prompt_parts.append(f"\nCurrent Emotional State: {therapeutic_window.emotional_state}")
        
        # Add recent messages
        prompt_parts.append("\nRecent Conversation:")
        for msg in therapeutic_window.messages[-5:]:
            prompt_parts.append(f"{msg.sender}: {msg.text}")
        
        return "\n".join(prompt_parts)
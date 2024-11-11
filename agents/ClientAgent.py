from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
from langchain_together import Together

@dataclass
class ClientMessage:
    text: str
    timestamp: datetime = datetime.now()

@dataclass
class ClientProfile:
    age: int
    gender: str
    occupation: str
    grief_duration: int  # months
    grief_feelings: List[str]
    integration_goals: List[str]
    attitude_to_therapy: str
    strengths: List[str]

class ClientAgent:
    def __init__(self, model: str, temperature: float, max_tokens: int, persona_text: str):
        self.agent = Together(model=model, temperature=temperature, max_tokens=max_tokens)
        self.profile = self.extract_profile_from_text(persona_text)
        self.messages: List[ClientMessage] = []

    def extract_profile_from_text(self, persona_text: str) -> ClientProfile:
        """Extract profile information from structured text without using LLM"""
        profile_data = {}
        
        # Split the text into lines and process each line
        for line in persona_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle list fields
                if key == 'Grief Feelings':
                    profile_data[key] = [item.strip() for item in value.split(',')]
                elif key == 'Integration Goals':
                    profile_data[key] = [item.strip() for item in value.split(',')]
                elif key == 'Strengths':
                    profile_data[key] = [item.strip() for item in value.split(',')]
                else:
                    profile_data[key] = value

        return ClientProfile(
            age=int(profile_data['Age']),
            gender=profile_data['Gender'],
            occupation=profile_data['Occupation'],
            grief_duration=int(profile_data['Grief Duration']),
            grief_feelings=profile_data['Grief Feelings'],
            integration_goals=profile_data['Integration Goals'],
            attitude_to_therapy=profile_data['Attitude to Therapy'],
            strengths=profile_data['Strengths']
        )
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return [
            {"sender": "client", "text": msg.text, "timestamp": msg.timestamp.isoformat()}
            for msg in self.messages
        ]
    
    def format_prompt(self, conversation_history: List[Dict[str, str]], user_message: str) -> str:
        # Format conversation history into a prompt
        history = "\n".join([
            f"{msg['sender']}: {msg['text']}"
            for msg in conversation_history[-5:]  # Last 5 messages for context
        ])

        prompt = f"""You are a client seeking psychedelic-assisted therapy. 
        You have been experiencing the following feelings for about {self.profile.grief_duration} months: {', '.join(self.profile.grief_feelings)}.
        Your main goals for therapy are to {', '.join(self.profile.integration_goals)}. 
        You are {self.profile.attitude_to_therapy} about therapy helping you.

        Previous conversation:
        {history}

        You: {user_message}

        Respond as the client. ONLY PROVIDE THE RESPONSE ITSELF, NOTHING ELSE:"""

        return prompt

    async def complete(self, conversation_history: List[Dict[str, str]], user_message: str) -> str:
        prompt = self.format_prompt(conversation_history, user_message)
        response = await self.agent.agenerate([prompt], stop=["therapist"])
        return response.generations[0][0].text.strip()
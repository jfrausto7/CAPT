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
    def __init__(self, model: str, temperature: float, max_tokens: int, persona_text):
        self.agent = Together(model=model, temperature=temperature, max_tokens=max_tokens)
        self.profile = self.extract_profile_from_text(persona_text)
        self.messages: List[ClientMessage] = []

    async def extract_profile_from_text(self, persona_text: str) -> ClientProfile:
        # Use the LLM to extract the client profile information from the persona text
        prompt = f"Extract the following information from the provided client persona text:\n\n{persona_text}\n\nOutput the information in this format:\n\nAge: <age>\nGender: <gender>\nOccupation: <occupation>\nGrief Duration: <grief_duration>\nGrief Feelings: <grief_feelings>\nIntegration Goals: <integration_goals>\nAttitude to Therapy: <attitude_to_therapy>\nStrengths: <strengths>"

        response = await self.llm.agenerate([prompt])
        profile_data = {}
        for line in response[0][0].split('\n'):
            if ':' in line:
                key, value = line.split(': ', 1)
                if key == 'Grief Feelings':
                    profile_data[key] = [feeling.strip() for feeling in value.split(',')]
                elif key == 'Integration Goals':
                    profile_data[key] = [goal.strip() for goal in value.split(',')]
                elif key == 'Strengths':
                    profile_data[key] = [strength.strip() for strength in value.split(',')]
                else:
                    profile_data[key] = value.strip()

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

            Respond as the client:"""

            return prompt

    async def complete(self, conversation_history: List[Dict[str, str]], user_message: str) -> str:
        prompt = self.format_prompt(conversation_history, user_message)
        response = await self.llm.agenerate([prompt])
        return response[0][0].strip()
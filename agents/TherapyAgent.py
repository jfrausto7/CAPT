
from langchain_together import Together


class TherapyAgent:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.agent = Together(model=model, temperature=temperature, max_tokens=max_tokens)
    
    async def complete(self, prompt: str) -> str:
        # TODO: replace with actual integration, chains, & prompts
        return self.agent.invoke(prompt)
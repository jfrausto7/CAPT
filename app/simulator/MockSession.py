import csv
from datetime import datetime
import asyncio
import time
import uuid
import random

from agents.ClientAgent import ClientAgent
from agents.TherapyAgent import TherapyAgent
from config import RATE_LIMIT_BREAK

# Sample client personas from "Deliberate Practice in Psychedelic-Assisted Therapy"
PERSONAS = [
    """Age: 35
Gender: Female
Occupation: Social Worker
Grief Duration: 8
Grief Feelings: Sadness, Guilt, Anger
Integration Goals: Find meaning, Improve relationships, Reduce anxiety
Attitude to Therapy: Cautiously optimistic
Strengths: Empathetic, Hardworking, Resilient""",

    """Age: 42
Gender: Male
Occupation: Artist
Grief Duration: 14
Grief Feelings: Despair, Hopelessness, Confusion
Integration Goals: Reconnect with creativity, Improve self-worth, Find purpose
Attitude to Therapy: Skeptical but willing to try
Strengths: Imaginative, Sensitive, Introspective""",

    """Age: 28
Gender: Non-binary
Occupation: Software Engineer
Grief Duration: 6
Grief Feelings: Anxiety, Loneliness, Emptiness
Integration Goals: Reduce social isolation, Improve self-acceptance, Find meaning
Attitude to Therapy: Nervous but hopeful
Strengths: Intelligent, Logical, Determined"""
]

def format_therapy_prompt(conversation_history):
    """Format conversation history into a prompt for the therapy agent."""
    # Convert the last 5 messages into a formatted string
    recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    formatted_history = "\n".join([
        f"{msg['sender']}: {msg['text']}"
        for msg in recent_messages
    ])
    
    prompt = f"""You are a compassionate and skilled therapist conducting an integration session 
    after psychedelic-assisted therapy. Your role is to help the client process their experience 
    and integrate insights into their daily life.

    Previous conversation:
    {formatted_history}

    Provide a thoughtful and empathetic response as the therapist. ONLY PROVIDE THE RESPONSE ITSELF, NOTHING ELSE:"""
    
    return prompt

async def run_mock_session(session_id: str, persona: str):
    """Run a single mock therapy session and save the transcript."""
    
    # Initialize agents
    client_agent = ClientAgent(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        temperature=0.7,
        max_tokens=512,
        persona_text=persona.strip()  # Ensure no extra whitespace
    )

    therapy_agent = TherapyAgent(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        temperature=0.5,
        max_tokens=512
    )

    # Predefined opening statements/questions for the therapist
    opening_statements = [
        "How are you feeling today as we begin our integration session?",
        "What's been coming up for you since our last session?",
        "What would you like to focus on in today's integration session?",
        "How have you been processing your experience since we last met?"
    ]

    # Start conversation with therapist's opening
    conversation_history = []
    opening = opening_statements[len(conversation_history) % len(opening_statements)]
    
    # Save the opening statement
    with open("data/transcripts/therapy_transcripts.csv", "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        timestamp = datetime.now().isoformat()
        writer.writerow([session_id, timestamp, "therapist", opening])
    
    # Initial conversation history entry
    conversation_history.append({
        "sender": "therapist",
        "text": opening,
        "timestamp": timestamp
    })
    
    # Simulate random number of exchanges
    num_exchanges = random.randint(5, 25)
    
    for _ in range(num_exchanges):
        # Get client response
        client_response = await client_agent.complete(conversation_history, opening)
        print("CLIENT RESPONSE:")
        print(client_response)
        timestamp = datetime.now().isoformat()
        
        # Save client response
        with open("data/transcripts/therapy_transcripts.csv", "a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, timestamp, "client", client_response])
        
        # Update conversation history
        conversation_history.append({
            "sender": "client",
            "text": client_response,
            "timestamp": timestamp
        })
        
        # Format prompt for therapy agent
        therapy_prompt = format_therapy_prompt(conversation_history)
        
        # Get therapist response
        time.sleep(RATE_LIMIT_BREAK)  # need this to avoid rate limiting
        therapist_response = await therapy_agent.complete(therapy_prompt)
        print("THERAPIST RESPONSE:")
        print(therapist_response)
        time.sleep(RATE_LIMIT_BREAK)  # need this to avoid rate limiting
        timestamp = datetime.now().isoformat()
        
        # Save therapist response
        with open("data/transcripts/therapy_transcripts.csv", "a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, timestamp, "therapist", therapist_response])
        
        # Update conversation history
        conversation_history.append({
            "sender": "therapist",
            "text": therapist_response,
            "timestamp": timestamp
        })
        
        # Set up next exchange
        opening = therapist_response

async def main():
    # Create or check if CSV exists with headers
    with open("data/transcripts/therapy_transcripts.csv", "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        # Check if file is empty
        f.seek(0, 2)  # Seek to end of file
        if f.tell() == 0:  # If file is empty
            writer.writerow(["session_id", "timestamp", "speaker", "message"])
    
    while True:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Let user choose a persona
        print("\nAvailable personas:")
        for i, persona in enumerate(PERSONAS, 1):
            print(f"\nPersona {i}:")
            print(persona)
        
        choice = int(input("\nChoose a persona (1-3): ")) - 1
        if 0 <= choice < len(PERSONAS):
            selected_persona = PERSONAS[choice]
            print(f"\nRunning session with selected persona...")
            await run_mock_session(session_id, selected_persona)
        
        if input("\nRun another session? (y/n): ").lower() != 'y':
            break

    print("\nAll sessions completed. Transcripts saved to data/transcripts/therapy_transcripts.csv")

if __name__ == "__main__":
    asyncio.run(main())
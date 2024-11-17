import asyncio
import os
import time
from app.conversation_manager import ConversationManager
from config import RATE_LIMIT_BREAK

async def test_conversation_manager():
    # Use a test database file
    test_db_path = "test_therapy_chat.db"
    
    # Clean up any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Initialize ConversationManager
    manager = ConversationManager(db_path=test_db_path)

    print("Testing ConversationManager...")
    print("-" * 50)
    
    # Test 1: Create a new conversation
    print("\nTest 1: Creating new conversation...")
    conversation_id = "TEST123"
    result = await manager.create_message(text="I've just had the craziest LSD experience... I have no idea how to process what I experienced. What should I do?'", conversation_id=conversation_id)
    print(f"Conversation created with ID: {conversation_id}")
    print(f"User message: {result['message']['text']}")
    print(f"Assistant response: {result['response']['text']}")
        
    # Test 2: Continue the conversation
    print("\nTest 2: Continuing conversation...")
    time.sleep(RATE_LIMIT_BREAK)
    result = await manager.create_message(
        "I'm interested in learning more about psilocybin.",
        conversation_id
    )
    print(f"User message: {result['message']['text']}")
    print(f"Assistant response: {result['response']['text']}")
        
    # Test 3: Retrieve conversation history
    print("\nTest 3: Retrieving conversation history...")
    conversation = manager.get_conversation(conversation_id)
    if conversation:
        print(f"Conversation created at: {conversation.created_at}")
        print("Messages:")
        for msg in conversation.messages:
            print(f"{msg.sender}: {msg.text}")
    else:
        print("Failed to retrieve conversation")
        
    # Test 4: Test clinical trial query
    time.sleep(RATE_LIMIT_BREAK)
    print("\nTest 4: Testing clinical trial query...")
    result = await manager.create_message(
        "Are there any MDMA clinical trials for PTSD in California?",
        conversation_id
    )
    print(f"User message: {result['message']['text']}")
    print(f"Assistant response: {result['response']['text']}")
        
    # Test 5: Delete conversation
    print("\nTest 5: Deleting conversation...")
    deleted = manager.delete_conversation(conversation_id)
    print(f"Conversation deleted: {deleted}")
    
    # Verify deletion
    conversation = manager.get_conversation(conversation_id)
    print(f"Conversation retrieval after deletion: {'Failed' if conversation is None else 'Still exists'}")

if __name__ == "__main__":
    asyncio.run(test_conversation_manager())
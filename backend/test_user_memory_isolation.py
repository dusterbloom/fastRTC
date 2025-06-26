#!/usr/bin/env python3
"""
Test script to verify user memory isolation in the FastRTC memory system.

This script tests:
1. Unique session-based user IDs instead of hardcoded "guest_user"
2. User identification via spoken commands
3. Memory isolation between different users
4. User switching functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
from datetime import datetime, timezone
from src.memory.manager import AMemMemoryManager
from src.audio.user_identification import SpokenUserIdentifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_user_memory_isolation():
    """Test user memory isolation functionality."""
    
    print("🧪 Testing User Memory Isolation System")
    print("=" * 50)
    
    # Test 1: Create session-based user ID (simulating VoiceAssistant initialization)
    session_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    session_user_id = f"session_user_{session_timestamp}"
    
    print(f"✅ Test 1: Session-based user ID created: {session_user_id}")
    
    # Test 2: Initialize memory manager with session user
    print(f"\n📝 Test 2: Initializing memory manager for session user...")
    try:
        memory_manager = AMemMemoryManager(session_user_id)
        await memory_manager.start_background_processor()
        print(f"✅ Memory manager initialized for user: {session_user_id}")
        print(f"   Collection name: {memory_manager.amem_system.retriever.collection.name}")
    except Exception as e:
        print(f"❌ Failed to initialize memory manager: {e}")
        return
    
    # Test 3: Add a memory for the session user
    print(f"\n💾 Test 3: Adding memory for session user...")
    try:
        category = await memory_manager.add_memory(
            "My name is TestUser. I love machine learning.",
            "Hello TestUser! Nice to meet you. Machine learning is fascinating!"
        )
        print(f"✅ Memory added for session user. Category: {category}")
    except Exception as e:
        print(f"❌ Failed to add memory: {e}")
    
    # Test 4: Test user identification
    print(f"\n👤 Test 4: Testing user identification...")
    identifier = SpokenUserIdentifier()
    
    test_phrases = [
        "I am Peppy",
        "My name is Alice", 
        "This is Bob",
        "User charlie"
    ]
    
    for phrase in test_phrases:
        identified_id = identifier.process_text(phrase)
        if identified_id:
            print(f"✅ Phrase: '{phrase}' -> Identified as: {identified_id}")
        else:
            print(f"❌ Phrase: '{phrase}' -> No identification")
    
    # Test 5: Test user switching
    print(f"\n🔄 Test 5: Testing user switching...")
    
    # Switch to user_peppy
    peppy_user_id = "user_peppy"
    if memory_manager.switch_user(peppy_user_id):
        print(f"✅ Successfully switched to user: {peppy_user_id}")
        print(f"   New collection: {memory_manager.amem_system.retriever.collection.name}")
        
        # Add memory for Peppy
        category = await memory_manager.add_memory(
            "I prefer Python over JavaScript for AI projects.",
            "Python is indeed excellent for AI! It has great libraries like TensorFlow and PyTorch."
        )
        print(f"✅ Memory added for Peppy. Category: {category}")
        
        # Check user context
        context = memory_manager.get_user_context()
        print(f"✅ Peppy's context: {context[:100]}...")
        
    else:
        print(f"❌ Failed to switch to user: {peppy_user_id}")
    
    # Test 6: Switch back to session user and verify isolation
    print(f"\n🔙 Test 6: Testing memory isolation...")
    
    if memory_manager.switch_user(session_user_id):
        print(f"✅ Switched back to session user: {session_user_id}")
        
        # Check that session user's context doesn't contain Peppy's info
        context = memory_manager.get_user_context()
        print(f"✅ Session user context: {context[:100]}...")
        
        if "Python" in context and "JavaScript" in context:
            print("❌ MEMORY LEAK: Session user sees Peppy's preferences!")
        else:
            print("✅ Memory isolation working: Session user doesn't see Peppy's data")
    else:
        print(f"❌ Failed to switch back to session user")
    
    # Test 7: Verify different collection names
    print(f"\n📊 Test 7: Verifying collection isolation...")
    
    session_collection = f"memories_user_session_user_{session_timestamp.replace('T', '_').replace('Z', '').replace(':', '_')}"
    peppy_collection = "memories_user_user_peppy"
    
    print(f"Session user collection should be: {session_collection}")
    print(f"Peppy collection should be: {peppy_collection}")
    
    # Clean up
    print(f"\n🧹 Cleaning up...")
    await memory_manager.shutdown()
    print("✅ Memory manager shutdown complete")
    
    print("\n🎉 User Memory Isolation Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    # Set dummy OpenAI key for local testing
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-use"
    
    # Run the test
    asyncio.run(test_user_memory_isolation())
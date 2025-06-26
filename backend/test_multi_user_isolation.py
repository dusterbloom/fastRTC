#!/usr/bin/env python3
"""
Test multi-user memory isolation in FastRTC.

This script verifies that:
1. Different users have completely isolated memory spaces
2. User switching works correctly
3. No memory leakage between users
4. Collection naming is consistent
5. User identification works properly
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logging import setup_logging
from src.memory.manager import AMemMemoryManager
from src.audio.user_identification import SpokenUserIdentifier
from src.a_mem.retrievers import ChromaRetriever
import chromadb

async def test_user_isolation():
    """Test that different users have isolated memory spaces."""
    
    setup_logging(log_level="INFO", console_output=True, colored_output=True)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Multi-User Memory Isolation")
    logger.info("=" * 50)
    
    # Test users
    test_users = ['alice', 'bob', 'charlie']
    user_memories = {}
    
    # Test 1: Create memories for different users
    logger.info("📝 Test 1: Creating user-specific memories")
    
    for username in test_users:
        user_id = f"user_{username}"
        logger.info(f"   Creating memory manager for: {user_id}")
        
        # Create memory manager for this user
        memory_manager = AMemMemoryManager(user_id=user_id)
        
        # Add user-specific memory
        user_memory = f"Hello, I am {username}. I like {username}_specific_activity."
        note_id = memory_manager.amem_system.add_note(
            content=user_memory,
            tags=[f"{username}_info", "personal_info"]
        )
        
        logger.info(f"   Added memory for {username}: '{user_memory[:30]}...'")
        user_memories[user_id] = {
            'manager': memory_manager,
            'memory': user_memory,
            'note_id': note_id
        }
    
    await asyncio.sleep(1)  # Let background processing complete
    
    # Test 2: Verify memory isolation
    logger.info("\n🔍 Test 2: Verifying memory isolation")
    
    for user_id, data in user_memories.items():
        memory_manager = data['manager']
        expected_memory = data['memory']
        
        # Search for user's own memory
        results = memory_manager.amem_system.search(f"Hello, I am", k=5)
        found_own_memory = any(expected_memory in r['content'] for r in results)
        
        # Search for other users' memories
        other_users = [u for u in test_users if f"user_{u}" != user_id]
        found_other_memory = False
        
        for other_user in other_users:
            other_memory = user_memories[f"user_{other_user}"]['memory']
            if any(other_memory in r['content'] for r in results):
                found_other_memory = True
                break
        
        status_own = "✅ PASS" if found_own_memory else "❌ FAIL"
        status_other = "✅ PASS" if not found_other_memory else "❌ FAIL"
        
        logger.info(f"   {user_id}:")
        logger.info(f"     Can access own memory: {status_own}")
        logger.info(f"     Cannot access other memories: {status_other}")
    
    # Test 3: User switching
    logger.info("\n🔄 Test 3: Testing user switching")
    
    # Use alice's manager and switch to bob
    alice_manager = user_memories['user_alice']['manager']
    original_user = alice_manager.user_id
    
    logger.info(f"   Original user: {original_user}")
    
    # Switch to bob
    switch_success = alice_manager.switch_user('user_bob')
    new_user = alice_manager.user_id
    
    logger.info(f"   Switch to bob: {'✅ SUCCESS' if switch_success else '❌ FAILED'}")
    logger.info(f"   New user: {new_user}")
    
    # Verify we can now access bob's memories but not alice's
    results = alice_manager.amem_system.search("Hello, I am", k=5)
    
    alice_memory = user_memories['user_alice']['memory']
    bob_memory = user_memories['user_bob']['memory']
    
    has_alice_memory = any(alice_memory in r['content'] for r in results)
    has_bob_memory = any(bob_memory in r['content'] for r in results)
    
    logger.info(f"   Can access Alice's memory: {'❌ FAIL' if has_alice_memory else '✅ PASS'}")
    logger.info(f"   Can access Bob's memory: {'✅ PASS' if has_bob_memory else '❌ FAIL'}")
    
    # Test 4: Collection naming validation
    logger.info("\n📋 Test 4: Validating collection names")
    
    db_path = '/mnt/c/Users/PC/Dev/fastRTC/backend/chroma_db'
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    
    expected_collections = [f"memories_user_{user}" for user in test_users]
    
    for expected in expected_collections:
        found = any(c.name == expected for c in collections)
        status = "✅ FOUND" if found else "❌ MISSING"
        logger.info(f"   {expected}: {status}")
    
    # Test 5: User identification
    logger.info("\n👤 Test 5: Testing user identification")
    
    identifier = SpokenUserIdentifier()
    
    test_phrases = [
        ("I am alice", "user_alice"),
        ("my name is bob", "user_bob"),
        ("this is charlie", "user_charlie"),
        ("user alice", "user_alice")
    ]
    
    for phrase, expected_id in test_phrases:
        result = identifier.process_text(phrase)
        status = "✅ PASS" if result == expected_id else "❌ FAIL"
        logger.info(f"   '{phrase}' -> {result} (expected {expected_id}): {status}")
    
    # Test 6: Cross-contamination check
    logger.info("\n🚫 Test 6: Cross-contamination check")
    
    # Add a secret to Alice and verify Bob can't see it
    alice_manager = user_memories['user_alice']['manager']
    alice_manager.switch_user('user_alice')
    
    secret_memory = "Alice's secret: she loves ice cream"
    alice_manager.amem_system.add_note(content=secret_memory, tags=["secret"])
    
    # Switch to Bob and search for Alice's secret
    alice_manager.switch_user('user_bob')
    results = alice_manager.amem_system.search("secret ice cream", k=10)
    
    found_secret = any("Alice's secret" in r['content'] for r in results)
    status = "✅ SECURE" if not found_secret else "❌ LEAKED"
    
    logger.info(f"   Bob cannot see Alice's secret: {status}")
    
    # Final report
    logger.info("\n📊 FINAL REPORT")
    logger.info("=" * 30)
    
    # Count total memories per user
    total_collections = 0
    total_memories = 0
    
    for collection in collections:
        if collection.name.startswith("memories_user_"):
            total_collections += 1
            count = collection.count()
            total_memories += count
            user_name = collection.name.replace("memories_user_", "")
            logger.info(f"   {user_name}: {count} memories")
    
    logger.info(f"\nTotal user collections: {total_collections}")
    logger.info(f"Total memories: {total_memories}")
    logger.info("✅ Multi-user memory isolation test completed!")

def test_collection_mapping():
    """Test the collection name mapping logic."""
    
    logger = logging.getLogger(__name__)
    logger.info("\n🔧 Testing Collection Name Mapping")
    
    retriever = ChromaRetriever()
    
    test_cases = [
        ("alice", "memories_user_alice"),
        ("user_bob", "memories_user_user_bob"),  # This shows the double prefix issue
        ("guest_user", "memories_user_guest_user"),
        ("test-user", "memories_user_testuser"),
        ("user@domain.com", "memories_user_userdomaincom")
    ]
    
    for user_id, expected in test_cases:
        result = retriever.get_user_collection_name(user_id)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        logger.info(f"   {user_id} -> {result}: {status}")

async def main():
    """Main test function."""
    
    print("🚀 FastRTC Multi-User Memory Isolation Test")
    print("=" * 60)
    print("This test verifies that user memories are properly isolated")
    print("and that there's no cross-contamination between users.")
    print("=" * 60)
    
    try:
        await test_user_isolation()
        test_collection_mapping()
        
        print("\n✅ All tests completed! Check the log output above for detailed results.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
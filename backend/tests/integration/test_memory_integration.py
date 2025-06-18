"""Integration tests for memory components."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, Mock

from src.memory.manager import AMemMemoryManager
from src.memory.cache import ResponseCache
from src.memory.conversation import ConversationBuffer
from src.core.exceptions import MemoryError


class TestMemoryIntegration:
    """Integration tests for memory components working together."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield os.path.join(temp_dir, "test_memory.db")
    
    @pytest.fixture
    def mock_amem_system(self):
        """Mock A-MEM system for integration tests."""
        with patch('src.memory.manager.AgenticMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_instance.memories = {}
            mock_instance.add_note.return_value = "test_memory_id"
            mock_instance.search_agentic.return_value = []
            mock_instance.consolidate_memories.return_value = None
            mock_instance.evo_cnt = 0
            
            # Mock retriever
            mock_retriever = Mock()
            mock_collection = Mock()
            mock_collection.get.return_value = {'ids': []}
            mock_retriever.collection = mock_collection
            mock_instance.retriever = mock_retriever
            
            mock_class.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def memory_components(self, mock_amem_system):
        """Create integrated memory components."""
        memory_manager = AMemMemoryManager(user_id="test_user")
        response_cache = ResponseCache(ttl_seconds=300, max_entries=100)
        conversation_buffer = ConversationBuffer(max_turns=10, max_context_turns=3)
        
        return memory_manager, response_cache, conversation_buffer
    
    @pytest.mark.asyncio
    async def test_memory_workflow_integration(self, memory_components, mock_amem_system):
        """Test complete memory workflow integration."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Start background processor
        await memory_manager.start_background_processor()
        
        try:
            # Simulate conversation flow
            user_inputs = [
                "Hello, my name is Alice",
                "I like chocolate and reading books",
                "What's my name?",
                "What do I like?"
            ]
            
            assistant_responses = [
                "Nice to meet you, Alice!",
                "Great to know about your preferences!",
                "Your name is Alice.",
                "You like chocolate and reading books."
            ]
            
            # Process conversation turns
            for user_text, assistant_text in zip(user_inputs, assistant_responses):
                # Add to memory manager
                category = await memory_manager.add_memory(user_text, assistant_text)
                
                # Add to conversation buffer
                conversation_buffer.add_turn(user_text, assistant_text, language="en")
                
                # Cache response
                response_cache.put(user_text, assistant_text)
                
                # Verify integration
                if "my name is" in user_text.lower():
                    assert category == "personal_info"
                elif "i like" in user_text.lower():
                    assert category == "preference"
                # Questions asking for recall should return None
                elif user_text.lower().startswith("what"):
                    assert category is None
            
            # Test memory retrieval
            context = memory_manager.get_user_context()
            assert "Alice" in context
            
            # Test conversation context
            recent_context = conversation_buffer.get_recent_context()
            assert "Alice" in recent_context
            
            # Test cache retrieval
            cached_response = response_cache.get("Hello, my name is Alice")
            assert cached_response == "Nice to meet you, Alice!"
            
            # Verify statistics
            memory_stats = memory_manager.get_stats()
            cache_stats = response_cache.get_stats()
            conversation_stats = conversation_buffer.get_stats()
            
            assert memory_stats['mem_ops'] > 0
            assert cache_stats['size'] > 0
            assert conversation_stats['current_turns'] > 0
            
        finally:
            await memory_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_search_integration(self, memory_components, mock_amem_system):
        """Test memory search integration across components."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Setup mock search results
        mock_amem_system.search_agentic.return_value = [
            {'content': 'User: I love pizza\nAssistant: Great choice!'},
            {'content': 'User: My favorite color is blue\nAssistant: Nice!'}
        ]
        
        # Add some conversation history
        conversation_buffer.add_turn("I love pizza", "Great choice!")
        conversation_buffer.add_turn("My favorite color is blue", "Nice!")
        
        # Test search
        search_result = await memory_manager.search_memories("food preferences")
        assert "pizza" in search_result
        
        # Test name query from cache
        memory_manager.memory_cache['user_name'] = "Bob"
        name_result = await memory_manager.search_memories("what is my name")
        assert "Bob" in name_result
        assert memory_manager.cache_hits > 0
    
    @pytest.mark.asyncio
    async def test_memory_persistence_simulation(self, memory_components, mock_amem_system):
        """Test memory persistence simulation."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Simulate existing memories in A-MEM
        existing_memory = Mock()
        existing_memory.content = "User: My name is Charlie\nAssistant: Hello Charlie!"
        existing_memory.tags = ["personal_info", "conversation"]
        existing_memory.timestamp = "202301011200"
        
        mock_amem_system.memories = {"memory_1": existing_memory}
        
        # Reload memories (simulate restart)
        memory_manager._load_existing_memories()
        
        # Verify loaded data
        assert memory_manager.memory_cache['user_name'] == "Charlie"
        
        # Test context generation with loaded data
        context = memory_manager.get_user_context()
        assert "Charlie" in context
    
    @pytest.mark.asyncio
    async def test_cache_and_memory_consistency(self, memory_components):
        """Test consistency between cache and memory."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Add data to both systems
        user_text = "What's the weather like?"
        assistant_text = "I don't have access to weather data."
        
        # Store in memory
        await memory_manager.add_memory(user_text, assistant_text)
        
        # Store in cache
        response_cache.put(user_text, assistant_text)
        
        # Verify both systems have the data
        cached_response = response_cache.get(user_text)
        assert cached_response == assistant_text
        
        # Verify memory operations were recorded
        assert memory_manager.memory_operations > 0
    
    @pytest.mark.asyncio
    async def test_conversation_buffer_memory_integration(self, memory_components):
        """Test conversation buffer integration with memory."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Add conversation turns
        turns = [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well!"),
            ("What's new?", "Just chatting with you!")
        ]
        
        for user_text, assistant_text in turns:
            # Add to conversation buffer
            conversation_buffer.add_turn(user_text, assistant_text, language="en")
            
            # Add to memory if significant
            category = await memory_manager.add_memory(user_text, assistant_text)
        
        # Test context generation uses conversation buffer
        context = conversation_buffer.get_recent_context()
        assert "How are you?" in context
        assert "What's new?" in context
        
        # Verify conversation statistics
        stats = conversation_buffer.get_stats()
        assert stats['current_turns'] == 3
        assert "en" in stats['languages_used']
    
    @pytest.mark.asyncio
    async def test_memory_error_handling_integration(self, mock_amem_system):
        """Test error handling across memory components."""
        # Test A-MEM initialization failure
        with patch('src.memory.manager.AgenticMemorySystem', side_effect=Exception("Init failed")):
            with pytest.raises(MemoryError, match="Failed to initialize A-MEM system"):
                AMemMemoryManager(user_id="test_user")
        
        # Test memory manager with working components
        memory_manager = AMemMemoryManager(user_id="test_user")
        response_cache = ResponseCache()
        conversation_buffer = ConversationBuffer()
        
        # Test graceful degradation
        mock_amem_system.add_note.side_effect = Exception("Storage failed")
        
        # Should not raise exception, but handle gracefully
        category = await memory_manager.add_memory("Hello", "Hi!")
        # Memory operations should still be counted even if storage fails
        assert memory_manager.memory_operations > 0
    
    @pytest.mark.asyncio
    async def test_multilingual_memory_integration(self, memory_components):
        """Test multilingual support across memory components."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Add multilingual conversations
        multilingual_turns = [
            ("Hello", "Hi there!", "en"),
            ("Ciao", "Ciao! Come stai?", "it"),
            ("Bonjour", "Salut!", "fr"),
            ("Hola", "¡Hola! ¿Cómo estás?", "es")
        ]
        
        for user_text, assistant_text, language in multilingual_turns:
            # Add to conversation buffer with language
            conversation_buffer.add_turn(user_text, assistant_text, language=language)
            
            # Cache responses
            response_cache.put(user_text, assistant_text)
            
            # Add to memory
            await memory_manager.add_memory(user_text, assistant_text)
        
        # Test language distribution
        lang_dist = conversation_buffer.get_language_distribution()
        assert len(lang_dist) == 4
        assert lang_dist["en"] == 1
        assert lang_dist["it"] == 1
        
        # Test cache works across languages
        assert response_cache.get("Ciao") == "Ciao! Come stai?"
        assert response_cache.get("Bonjour") == "Salut!"
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_integration(self, memory_components):
        """Test cleanup and shutdown integration."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        # Start background processor
        await memory_manager.start_background_processor()
        
        # Add some data
        await memory_manager.add_memory("Test message", "Test response")
        response_cache.put("Test", "Response")
        conversation_buffer.add_turn("Test", "Response")
        
        # Test individual cleanup
        response_cache.clear()
        assert len(response_cache) == 0
        
        conversation_buffer.clear()
        assert len(conversation_buffer) == 0
        
        # Test memory manager shutdown
        await memory_manager.shutdown()
        
        # Verify background task is cleaned up
        assert memory_manager.background_task.cancelled() or memory_manager.background_task.done()
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, memory_components):
        """Test performance characteristics of integrated components."""
        memory_manager, response_cache, conversation_buffer = memory_components
        
        import time
        
        # Measure performance of integrated operations
        start_time = time.time()
        
        # Perform many operations
        for i in range(100):
            user_text = f"Message {i}"
            assistant_text = f"Response {i}"
            
            # Add to all components
            await memory_manager.add_memory(user_text, assistant_text)
            response_cache.put(user_text, assistant_text)
            conversation_buffer.add_turn(user_text, assistant_text)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 100 operations
        
        # Verify all components handled the load
        assert memory_manager.memory_operations == 100
        assert len(response_cache) == 100
        assert len(conversation_buffer) == 10  # Limited by max_turns
        
        # Test retrieval performance
        start_time = time.time()
        
        for i in range(50):
            response_cache.get(f"Message {i}")
            conversation_buffer.get_recent_context()
            memory_manager.get_user_context()
        
        end_time = time.time()
        retrieval_duration = end_time - start_time
        
        # Retrieval should be fast
        assert retrieval_duration < 1.0  # 1 second for 150 operations
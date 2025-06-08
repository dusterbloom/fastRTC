"""Integration tests for LLM services."""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch

from src.services.llm_service import LLMService
from src.memory.cache import ResponseCache
from src.memory.conversation import ConversationBuffer
from src.core.exceptions import LLMError
from tests.mocks.mock_memory import MockMemoryManager
from tests.mocks.mock_llm import create_mock_http_session


class TestLLMIntegration:
    """Integration tests for LLM service with other components."""
    
    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance."""
        return LLMService(
            use_ollama=True,
            ollama_url="http://localhost:11434",
            ollama_model="test-model",
            timeout=5.0
        )
    
    @pytest.fixture
    def mock_http_session(self):
        """Create mock HTTP session."""
        return create_mock_http_session()
    
    @pytest.fixture
    def response_cache(self):
        """Create response cache."""
        return ResponseCache(ttl_seconds=300, max_entries=100)
    
    @pytest.fixture
    def conversation_buffer(self):
        """Create conversation buffer."""
        return ConversationBuffer(max_turns=10, max_context_turns=3)
    
    @pytest.fixture
    def memory_manager(self):
        """Create mock memory manager."""
        return MockMemoryManager(user_name="Alice", preferences={"food": "pizza"})
    
    @pytest.fixture
    async def integrated_llm_service(self, llm_service, mock_http_session, 
                                   response_cache, conversation_buffer, memory_manager):
        """Create fully integrated LLM service."""
        await llm_service.initialize(
            http_session=mock_http_session,
            response_cache=response_cache,
            conversation_buffer=conversation_buffer,
            memory_manager=memory_manager
        )
        return llm_service
    
    @pytest.mark.asyncio
    async def test_llm_with_memory_integration(self, integrated_llm_service, memory_manager):
        """Test LLM service integration with memory manager."""
        # Test name recall from memory
        response = await integrated_llm_service.get_response("what is my name", "")
        assert "Alice" in response
        assert memory_manager.cache_hits > 0
        
        # Test memory deletion
        response = await integrated_llm_service.get_response("forget everything", "")
        assert "erased all my memories" in response
        
        # Test name extraction and acknowledgment
        response = await integrated_llm_service.get_response("my name is Bob", "")
        assert "Got it, Bob!" in response
        assert len(memory_manager.add_memory_calls) > 0
    
    @pytest.mark.asyncio
    async def test_llm_with_cache_integration(self, integrated_llm_service, response_cache):
        """Test LLM service integration with response cache."""
        # First request should miss cache and call LLM
        response1 = await integrated_llm_service.get_response("Hello", "Context")
        assert response1 is not None
        assert integrated_llm_service._stats['requests'] == 1
        
        # Second identical request should hit cache
        response2 = await integrated_llm_service.get_response("Hello", "Context")
        assert response2 == response1
        assert integrated_llm_service._stats['cache_hits'] == 1
        
        # Verify cache statistics
        cache_stats = response_cache.get_stats()
        assert cache_stats['hits'] == 1
        assert cache_stats['size'] == 1
    
    @pytest.mark.asyncio
    async def test_llm_with_conversation_integration(self, integrated_llm_service, conversation_buffer):
        """Test LLM service integration with conversation buffer."""
        # Add some conversation history
        conversation_buffer.add_turn("Hello", "Hi there!")
        conversation_buffer.add_turn("How are you?", "I'm doing well!")
        
        # Test that conversation context is included in LLM prompt
        context = "User context"
        prompt = integrated_llm_service._get_llm_context_prompt(context)
        
        assert "Recent Conversation:" in prompt
        assert "Hello" in prompt
        assert "How are you?" in prompt
        assert "User context" in prompt
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, integrated_llm_service, 
                                        memory_manager, response_cache, conversation_buffer):
        """Test complete conversation flow with all components."""
        # Simulate a conversation
        conversation_turns = [
            ("Hello", "Hi there! How can I help you?"),
            ("My name is Charlie", "Got it, Charlie! I'll remember that."),
            ("I like music", "Great to know about your preferences!"),
            ("What's my name?", "Your name is Charlie."),
            ("What do I like?", "You mentioned you like music.")
        ]
        
        for user_text, expected_response_type in conversation_turns:
            # Get LLM response
            response = await integrated_llm_service.get_response(user_text, "")
            
            # Verify response is reasonable
            assert response is not None
            assert len(response) > 0
            
            # Add to conversation buffer (simulating full system)
            conversation_buffer.add_turn(user_text, response)
        
        # Verify all components have been updated
        assert len(conversation_buffer) == 5
        assert len(memory_manager.add_memory_calls) > 0
        assert response_cache.get_stats()['size'] > 0
        
        # Verify memory manager has learned user info
        context = memory_manager.get_user_context()
        assert "Charlie" in context or memory_manager.user_name == "Charlie"
    
    @pytest.mark.asyncio
    async def test_llm_backend_switching(self, mock_http_session, response_cache, 
                                       conversation_buffer, memory_manager):
        """Test switching between Ollama and LM Studio backends."""
        # Test Ollama backend
        ollama_service = LLMService(use_ollama=True, ollama_model="ollama-model")
        await ollama_service.initialize(
            http_session=mock_http_session,
            response_cache=response_cache,
            conversation_buffer=conversation_buffer,
            memory_manager=memory_manager
        )
        
        response1 = await ollama_service.get_response("Hello", "Context")
        assert response1 is not None
        
        # Test LM Studio backend
        lm_studio_service = LLMService(use_ollama=False, lm_studio_model="lm-studio-model")
        await lm_studio_service.initialize(
            http_session=mock_http_session,
            response_cache=response_cache,
            conversation_buffer=conversation_buffer,
            memory_manager=memory_manager
        )
        
        response2 = await lm_studio_service.get_response("Hello", "Context")
        assert response2 is not None
        
        # Verify different backends can be used
        ollama_stats = ollama_service.get_stats()
        lm_studio_stats = lm_studio_service.get_stats()
        
        assert ollama_stats['backend'] == 'Ollama'
        assert lm_studio_stats['backend'] == 'LM Studio'
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, llm_service, response_cache, 
                                            conversation_buffer, memory_manager):
        """Test error handling across integrated components."""
        # Create failing HTTP session
        failing_session = Mock()
        failing_session.post.side_effect = aiohttp.ClientConnectorError(
            connection_key=Mock(), os_error=Mock()
        )
        
        await llm_service.initialize(
            http_session=failing_session,
            response_cache=response_cache,
            conversation_buffer=conversation_buffer,
            memory_manager=memory_manager
        )
        
        # Test that connection errors are handled gracefully
        with pytest.raises(LLMError, match="Unable to connect to LLM server"):
            await llm_service.get_response("Hello", "Context")
        
        # Verify error statistics
        stats = llm_service.get_stats()
        assert stats['connection_errors'] > 0
        assert stats['failures'] > 0
    
    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self, llm_service, mock_http_session,
                                              response_cache, conversation_buffer, memory_manager):
        """Test timeout handling in integrated environment."""
        # Create session that times out
        timeout_session = Mock()
        timeout_session.post.side_effect = asyncio.TimeoutError()
        
        await llm_service.initialize(
            http_session=timeout_session,
            response_cache=response_cache,
            conversation_buffer=conversation_buffer,
            memory_manager=memory_manager
        )
        
        # Test timeout handling
        with pytest.raises(LLMError, match="Request is taking longer than usual"):
            await llm_service.get_response("Hello", "Context")
        
        # Verify timeout statistics
        stats = llm_service.get_stats()
        assert stats['timeouts'] > 0
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, integrated_llm_service):
        """Test health check integration."""
        # Test successful health check
        is_healthy = await integrated_llm_service.health_check()
        assert is_healthy is True
        
        # Test availability
        assert integrated_llm_service.is_available() is True
    
    @pytest.mark.asyncio
    async def test_multilingual_llm_integration(self, integrated_llm_service, conversation_buffer):
        """Test multilingual support in LLM integration."""
        # Test different languages
        multilingual_inputs = [
            ("Hello", "en"),
            ("Ciao", "it"),
            ("Bonjour", "fr"),
            ("Hola", "es")
        ]
        
        for user_text, language in multilingual_inputs:
            response = await integrated_llm_service.get_response(user_text, "")
            assert response is not None
            
            # Add to conversation buffer with language
            conversation_buffer.add_turn(user_text, response, language=language)
        
        # Verify language distribution
        lang_dist = conversation_buffer.get_language_distribution()
        assert len(lang_dist) == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_integration(self, integrated_llm_service):
        """Test concurrent LLM requests with integrated components."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = integrated_llm_service.get_response(f"Message {i}", "Context")
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed successfully
        for response in responses:
            assert not isinstance(response, Exception)
            assert response is not None
        
        # Verify statistics
        stats = integrated_llm_service.get_stats()
        assert stats['total_requests'] == 10
        assert stats['successes'] == 10
    
    @pytest.mark.asyncio
    async def test_memory_search_llm_integration(self, integrated_llm_service, memory_manager):
        """Test memory search integration with LLM responses."""
        # Setup memory search results
        memory_manager.set_search_result("food preferences", "I found that you like pizza and chocolate.")
        
        # Test memory search through LLM
        response = await integrated_llm_service.get_response("what do you remember about my food preferences", "")
        assert "pizza" in response or "chocolate" in response
        
        # Verify search was called
        assert len(memory_manager.search_memory_calls) > 0
    
    @pytest.mark.asyncio
    async def test_context_building_integration(self, integrated_llm_service, 
                                              memory_manager, conversation_buffer):
        """Test context building with all components."""
        # Setup memory context
        memory_manager.set_user_name("David")
        memory_manager.add_preference("hobby", "photography")
        
        # Setup conversation context
        conversation_buffer.add_turn("I went hiking yesterday", "That sounds fun!")
        conversation_buffer.add_turn("The weather was perfect", "Great hiking weather!")
        
        # Test context building
        memory_context = memory_manager.get_user_context()
        llm_prompt = integrated_llm_service._get_llm_context_prompt(memory_context)
        
        # Verify all context is included
        assert "David" in llm_prompt
        assert "photography" in llm_prompt
        assert "hiking" in llm_prompt
        assert "Recent Conversation:" in llm_prompt
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, integrated_llm_service):
        """Test performance of integrated LLM service."""
        import time
        
        # Measure response time for multiple requests
        start_time = time.time()
        
        for i in range(20):
            await integrated_llm_service.get_response(f"Test message {i}", "Context")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert duration < 10.0  # 10 seconds for 20 requests
        
        # Verify statistics
        stats = integrated_llm_service.get_stats()
        assert stats['total_requests'] == 20
        assert stats['success_rate'] == "100.0%"
    
    @pytest.mark.asyncio
    async def test_shutdown_integration(self, integrated_llm_service):
        """Test graceful shutdown of integrated LLM service."""
        # Perform some operations
        await integrated_llm_service.get_response("Hello", "Context")
        
        # Test shutdown
        await integrated_llm_service.shutdown()
        
        # Should complete without errors
        assert True  # If we reach here, shutdown was successful
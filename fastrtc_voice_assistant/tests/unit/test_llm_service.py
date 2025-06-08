"""Unit tests for LLM service."""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from aiohttp import web

from src.services.llm_service import LLMService
from src.core.exceptions import LLMError


class TestLLMService:
    """Test cases for LLMService."""
    
    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance."""
        return LLMService(
            use_ollama=True,
            ollama_url="http://localhost:11434",
            ollama_model="test-model",
            timeout=5.0,
            max_tokens=100,
            temperature=0.7
        )
    
    @pytest.fixture
    def mock_http_session(self):
        """Mock HTTP session."""
        session = Mock(spec=aiohttp.ClientSession)
        return session
    
    @pytest.fixture
    def mock_response_cache(self):
        """Mock response cache."""
        cache = Mock()
        cache.get.return_value = None
        cache.put = Mock()
        return cache
    
    @pytest.fixture
    def mock_conversation_buffer(self):
        """Mock conversation buffer."""
        buffer = Mock()
        buffer.get_recent_context.return_value = "\n---\nRecent Conversation:\nUser: Hello\nAssistant: Hi there!"
        return buffer
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager."""
        manager = Mock()
        manager.search_memories = AsyncMock(return_value="I found some memories about you.")
        manager.clear_memory = AsyncMock(return_value=True)
        manager.extract_user_name.return_value = None
        manager.update_local_cache = Mock()
        manager.add_memory = AsyncMock()
        return manager
    
    def test_initialization(self, llm_service):
        """Test LLM service initialization."""
        assert llm_service.use_ollama is True
        assert llm_service.ollama_url == "http://localhost:11434"
        assert llm_service.ollama_model == "test-model"
        assert llm_service.timeout == 5.0
        assert llm_service.max_tokens == 100
        assert llm_service.temperature == 0.7
        assert llm_service.http_session is None
        assert llm_service._stats['requests'] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_dependencies(self, llm_service, mock_http_session, 
                                         mock_response_cache, mock_conversation_buffer, 
                                         mock_memory_manager):
        """Test dependency initialization."""
        await llm_service.initialize(
            http_session=mock_http_session,
            response_cache=mock_response_cache,
            conversation_buffer=mock_conversation_buffer,
            memory_manager=mock_memory_manager
        )
        
        assert llm_service.http_session == mock_http_session
        assert llm_service.response_cache == mock_response_cache
        assert llm_service.conversation_buffer == mock_conversation_buffer
        assert llm_service.memory_manager == mock_memory_manager
    
    def test_get_llm_context_prompt(self, llm_service, mock_conversation_buffer):
        """Test LLM context prompt generation."""
        llm_service.conversation_buffer = mock_conversation_buffer
        
        context = "User's name is John"
        prompt = llm_service._get_llm_context_prompt(context)
        
        assert "Echo" in prompt
        assert "User's name is John" in prompt
        assert "Recent Conversation" in prompt
        assert "multilingual voice assistant" in prompt
    
    @pytest.mark.asyncio
    async def test_get_response_cache_hit(self, llm_service, mock_http_session, mock_response_cache):
        """Test response with cache hit."""
        await llm_service.initialize(http_session=mock_http_session, response_cache=mock_response_cache)
        
        # Setup cache hit
        mock_response_cache.get.return_value = "Cached response"
        
        response = await llm_service.get_response("Hello", "Context")
        
        assert response == "Cached response"
        assert llm_service._stats['cache_hits'] == 1
        assert llm_service._stats['requests'] == 1
    
    @pytest.mark.asyncio
    async def test_get_response_recall_phrase(self, llm_service, mock_http_session, 
                                            mock_response_cache, mock_memory_manager):
        """Test response with recall phrase."""
        await llm_service.initialize(
            http_session=mock_http_session,
            response_cache=mock_response_cache,
            memory_manager=mock_memory_manager
        )
        
        response = await llm_service.get_response("what do you remember about me", "Context")
        
        assert response == "I found some memories about you."
        mock_memory_manager.search_memories.assert_called_once()
        mock_response_cache.put.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_response_delete_memory(self, llm_service, mock_http_session, 
                                            mock_response_cache, mock_memory_manager):
        """Test response with memory deletion request."""
        await llm_service.initialize(
            http_session=mock_http_session,
            response_cache=mock_response_cache,
            memory_manager=mock_memory_manager
        )
        
        response = await llm_service.get_response("forget everything", "Context")
        
        assert "erased all my memories" in response
        mock_memory_manager.clear_memory.assert_called_once()
        mock_response_cache.put.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_response_name_extraction(self, llm_service, mock_http_session, 
                                              mock_response_cache, mock_memory_manager):
        """Test response with name extraction."""
        await llm_service.initialize(
            http_session=mock_http_session,
            response_cache=mock_response_cache,
            memory_manager=mock_memory_manager
        )
        
        # Setup name extraction
        mock_memory_manager.extract_user_name.return_value = "John"
        
        response = await llm_service.get_response("my name is John", "Context")
        
        assert "Got it, John!" in response
        mock_memory_manager.update_local_cache.assert_called_once()
        mock_memory_manager.add_memory.assert_called_once()
        mock_response_cache.put.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_ollama_success(self, llm_service):
        """Test successful Ollama API call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message": {"content": "Hello! How can I help you?"}
        })
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        llm_service.http_session = mock_session
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await llm_service._call_ollama(messages)
        
        assert response == "Hello! How can I help you?"
    
    @pytest.mark.asyncio
    async def test_call_ollama_error(self, llm_service):
        """Test Ollama API call error."""
        # Mock error response
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        llm_service.http_session = mock_session
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(LLMError, match="Ollama request failed"):
            await llm_service._call_ollama(messages)
    
    @pytest.mark.asyncio
    async def test_call_lm_studio_success(self, llm_service):
        """Test successful LM Studio API call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello! How can I help you?"}}]
        })
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        llm_service.http_session = mock_session
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await llm_service._call_lm_studio(messages)
        
        assert response == "Hello! How can I help you?"
    
    @pytest.mark.asyncio
    async def test_call_lm_studio_error(self, llm_service):
        """Test LM Studio API call error."""
        # Mock error response
        mock_response = Mock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad request")
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        llm_service.http_session = mock_session
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(LLMError, match="LM Studio request failed"):
            await llm_service._call_lm_studio(messages)
    
    @pytest.mark.asyncio
    async def test_generate_llm_response_no_session(self, llm_service):
        """Test LLM response generation without HTTP session."""
        with pytest.raises(LLMError, match="HTTP session not available"):
            await llm_service._generate_llm_response("Hello", "Context")
    
    @pytest.mark.asyncio
    async def test_generate_llm_response_connection_error(self, llm_service):
        """Test LLM response generation with connection error."""
        mock_session = Mock()
        mock_session.post.side_effect = aiohttp.ClientConnectorError(
            connection_key=Mock(), os_error=Mock()
        )
        llm_service.http_session = mock_session
        
        with pytest.raises(LLMError, match="Unable to connect to LLM server"):
            await llm_service._generate_llm_response("Hello", "Context")
        
        assert llm_service._stats['connection_errors'] == 1
    
    @pytest.mark.asyncio
    async def test_generate_llm_response_timeout(self, llm_service):
        """Test LLM response generation with timeout."""
        mock_session = Mock()
        mock_session.post.side_effect = asyncio.TimeoutError()
        llm_service.http_session = mock_session
        
        with pytest.raises(LLMError, match="Request is taking longer than usual"):
            await llm_service._generate_llm_response("Hello", "Context")
        
        assert llm_service._stats['timeouts'] == 1
    
    @pytest.mark.asyncio
    async def test_health_check_ollama_success(self, llm_service):
        """Test successful Ollama health check."""
        mock_response = Mock()
        mock_response.status = 200
        
        mock_session = Mock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        llm_service.http_session = mock_session
        
        is_healthy = await llm_service.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_lm_studio_success(self, llm_service):
        """Test successful LM Studio health check."""
        llm_service.use_ollama = False
        
        mock_response = Mock()
        mock_response.status = 200
        
        mock_session = Mock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        llm_service.http_session = mock_session
        
        is_healthy = await llm_service.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service):
        """Test health check failure."""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection failed")
        llm_service.http_session = mock_session
        
        is_healthy = await llm_service.health_check()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_no_session(self, llm_service):
        """Test health check without HTTP session."""
        is_healthy = await llm_service.health_check()
        assert is_healthy is False
    
    def test_is_available(self, llm_service, mock_http_session):
        """Test availability check."""
        assert llm_service.is_available() is False
        
        llm_service.http_session = mock_http_session
        assert llm_service.is_available() is True
    
    def test_get_stats(self, llm_service):
        """Test statistics generation."""
        llm_service._stats = {
            'requests': 10,
            'successes': 8,
            'failures': 2,
            'cache_hits': 3,
            'timeouts': 1,
            'connection_errors': 1
        }
        
        stats = llm_service.get_stats()
        
        assert stats['backend'] == 'Ollama'
        assert stats['model'] == 'test-model'
        assert stats['total_requests'] == 10
        assert stats['successes'] == 8
        assert stats['failures'] == 2
        assert stats['success_rate'] == "80.0%"
        assert stats['cache_hits'] == 3
        assert stats['timeouts'] == 1
        assert stats['connection_errors'] == 1
        assert stats['timeout_seconds'] == 5.0
        assert stats['max_tokens'] == 100
        assert stats['temperature'] == 0.7
    
    @pytest.mark.asyncio
    async def test_shutdown(self, llm_service):
        """Test LLM service shutdown."""
        await llm_service.shutdown()
        # Should complete without error
    
    @pytest.mark.asyncio
    async def test_full_response_flow(self, llm_service, mock_http_session, 
                                    mock_response_cache, mock_conversation_buffer, 
                                    mock_memory_manager):
        """Test full response generation flow."""
        await llm_service.initialize(
            http_session=mock_http_session,
            response_cache=mock_response_cache,
            conversation_buffer=mock_conversation_buffer,
            memory_manager=mock_memory_manager
        )
        
        # Mock successful Ollama response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message": {"content": "Hello! How can I help you today?"}
        })
        
        mock_http_session.post.return_value.__aenter__.return_value = mock_response
        
        response = await llm_service.get_response("Hello there", "User context")
        
        assert response == "Hello! How can I help you today?"
        assert llm_service._stats['requests'] == 1
        assert llm_service._stats['successes'] == 1
        
        # Verify memory and cache interactions
        mock_memory_manager.add_memory.assert_called_once()
        mock_response_cache.put.assert_called_once()
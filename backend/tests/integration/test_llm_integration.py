"""Integration tests for LLM services with real components."""

import pytest
import pytest_asyncio
import asyncio
import aiohttp
import time
import shutil
import os
import chromadb
import tempfile
from unittest.mock import Mock

from src.services.llm_service import LLMService
from src.memory.manager import AMemMemoryManager as MemoryManager
from src.a_mem.retrievers import ChromaRetriever
from src.memory.cache import ResponseCache
from src.memory.conversation import ConversationBuffer
from src.core.exceptions import LLMError
from src.memory.redis_cache import MemoryRedisCache
from tests.test_config import get_test_config

# Get test configuration
config = get_test_config()

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def llm_service():
    """Create LLM service instance with real Ollama for the entire module."""
    llm_config = config.llm
    return LLMService(
        use_ollama=llm_config.use_ollama,
        ollama_url=llm_config.ollama_url,
        ollama_model=llm_config.ollama_model,
        timeout=llm_config.timeout,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature
    )

@pytest_asyncio.fixture(scope="function")
async def memory_manager():
    """Create a real MemoryManager instance for each test function."""
    test_user_id = "test_user"
    # Create a unique temporary directory for each test
    temp_dir = tempfile.mkdtemp()

    # Clear Redis cache for the test user
    redis_cache = MemoryRedisCache()
    redis_cache.clear_user_data(test_user_id)
    
    manager = MemoryManager(user_id=test_user_id)
    await manager.start_background_processor()
    yield manager
    await manager.shutdown()
    # Clean up the temporary directory after the test
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def response_cache():
    """Create a new ResponseCache for each test function."""
    return ResponseCache(ttl_seconds=config.memory.cache_ttl_seconds, max_entries=100)

@pytest.fixture(scope="function")
def conversation_buffer():
    """Create a new ConversationBuffer for each test function."""
    return ConversationBuffer(max_turns=10, max_context_turns=3)

@pytest_asyncio.fixture(scope="function")
async def integrated_llm_service(llm_service, memory_manager, response_cache, conversation_buffer):
    """Create a fully integrated LLM service with real components for each test function."""
    async with aiohttp.ClientSession() as session:
        await llm_service.initialize(
            http_session=session,
            response_cache=response_cache,
            conversation_buffer=conversation_buffer,
            memory_manager=memory_manager
        )
        yield llm_service

class TestLLMIntegration:
    """Integration tests for LLM service with other components."""

    @pytest.mark.asyncio
    async def test_llm_with_memory_integration(self, integrated_llm_service, memory_manager):
        """Test LLM service integration with a real memory manager."""
        # Test name extraction and storage
        full_response_content = []
        async for chunk in integrated_llm_service.get_response("my name is Alice", ""):
            full_response_content.append(chunk)
        response = "".join(full_response_content)
        assert "Alice" in response
        
        # Verify that the name is stored in memory
        context = memory_manager.get_user_context()
        assert "Alice" in context

        # Test name recall from memory
        full_response_content = []
        async for chunk in integrated_llm_service.get_response("what is my name", ""):
            full_response_content.append(chunk)
        response = "".join(full_response_content)
        assert "Alice" in response

    @pytest.mark.asyncio
    async def test_performance_and_benchmarking(self, integrated_llm_service):
        """Benchmark performance of the integrated LLM service."""
        num_requests = 10
        latencies = []
        
        start_time = time.monotonic()
        
        for i in range(num_requests):
            req_start_time = time.monotonic()
            full_response_content = []
            async for chunk in integrated_llm_service.get_response(f"Test message {i}", "Context"):
                full_response_content.append(chunk)
            # response = "".join(full_response_content) # Not needed for this test, just need to consume the generator
            req_end_time = time.monotonic()
            latencies.append(req_end_time - req_start_time)
            
        total_duration = time.monotonic() - start_time
        
        avg_latency = sum(latencies) / num_requests
        requests_per_second = num_requests / total_duration
        
        print(f"\n--- Performance Benchmark ---")
        print(f"Total requests: {num_requests}")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"Requests per second: {requests_per_second:.2f}")
        
        assert total_duration < 20.0  # Set a reasonable threshold
        assert avg_latency < 2.0

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, integrated_llm_service, 
                                        memory_manager, response_cache, conversation_buffer):
        """Test complete conversation flow with all real components."""
        conversation_turns = [
            ("Hello", "Hello"),
            ("My name is Charlie", "Charlie"),
            ("I like music", "music"),
            ("What's my name?", "Charlie"),
            ("What do I like?", "music")
        ]
        
        for user_text, expected_keyword in conversation_turns:
            full_response_content = []
            async for chunk in integrated_llm_service.get_response(user_text, ""):
                full_response_content.append(chunk)
            response = "".join(full_response_content)
            assert expected_keyword in response
            conversation_buffer.add_turn(user_text, response)
            
        assert len(conversation_buffer) == 5
        context = memory_manager.get_user_context()
        assert "Charlie" in context
        assert "music" in context


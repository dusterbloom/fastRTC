"""Mock LLM utilities for testing."""

from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Optional


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """Initialize mock LLM service.
        
        Args:
            responses: Dictionary mapping input text to responses
        """
        self.responses = responses or {}
        self.default_response = "This is a mock response."
        self.call_count = 0
        self.last_user_text = None
        self.last_context = None
        
        # Mock statistics
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'cache_hits': 0,
            'timeouts': 0,
            'connection_errors': 0
        }
    
    async def get_response(self, user_text: str, context: str) -> str:
        """Mock get_response method."""
        self.call_count += 1
        self.last_user_text = user_text
        self.last_context = context
        self._stats['requests'] += 1
        
        # Return predefined response if available
        if user_text.lower() in self.responses:
            response = self.responses[user_text.lower()]
        else:
            response = self.default_response
        
        self._stats['successes'] += 1
        return response
    
    async def health_check(self) -> bool:
        """Mock health check."""
        return True
    
    def is_available(self) -> bool:
        """Mock availability check."""
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Mock statistics."""
        return {
            'backend': 'Mock',
            'model': 'mock-model',
            'url': 'http://mock-url',
            'total_requests': self._stats['requests'],
            'successes': self._stats['successes'],
            'failures': self._stats['failures'],
            'success_rate': "100.0%",
            'cache_hits': self._stats['cache_hits'],
            'timeouts': self._stats['timeouts'],
            'connection_errors': self._stats['connection_errors'],
            'timeout_seconds': 30.0,
            'max_tokens': 250,
            'temperature': 0.7
        }
    
    async def shutdown(self):
        """Mock shutdown."""
        pass


class MockOllamaServer:
    """Mock Ollama server for testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None, 
                 status_code: int = 200, should_fail: bool = False):
        """Initialize mock Ollama server.
        
        Args:
            responses: Dictionary mapping input to responses
            status_code: HTTP status code to return
            should_fail: Whether requests should fail
        """
        self.responses = responses or {}
        self.status_code = status_code
        self.should_fail = should_fail
        self.request_count = 0
        self.last_request = None
    
    async def handle_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mock chat request."""
        self.request_count += 1
        self.last_request = request_data
        
        if self.should_fail:
            raise Exception("Mock server failure")
        
        # Extract user message
        messages = request_data.get('messages', [])
        user_message = None
        for msg in messages:
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        # Get response
        if user_message and user_message.lower() in self.responses:
            response_text = self.responses[user_message.lower()]
        else:
            response_text = "Mock Ollama response"
        
        return {
            'message': {
                'content': response_text
            }
        }


class MockLMStudioServer:
    """Mock LM Studio server for testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None, 
                 status_code: int = 200, should_fail: bool = False):
        """Initialize mock LM Studio server.
        
        Args:
            responses: Dictionary mapping input to responses
            status_code: HTTP status code to return
            should_fail: Whether requests should fail
        """
        self.responses = responses or {}
        self.status_code = status_code
        self.should_fail = should_fail
        self.request_count = 0
        self.last_request = None
    
    async def handle_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mock chat request."""
        self.request_count += 1
        self.last_request = request_data
        
        if self.should_fail:
            raise Exception("Mock server failure")
        
        # Extract user message
        messages = request_data.get('messages', [])
        user_message = None
        for msg in messages:
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        # Get response
        if user_message and user_message.lower() in self.responses:
            response_text = self.responses[user_message.lower()]
        else:
            response_text = "Mock LM Studio response"
        
        return {
            'choices': [
                {
                    'message': {
                        'content': response_text
                    }
                }
            ]
        }


def create_mock_http_session(server_responses: Optional[Dict[str, Any]] = None,
                           status_code: int = 200,
                           should_fail: bool = False):
    """Create a mock HTTP session for testing.
    
    Args:
        server_responses: Dictionary mapping URLs to response data
        status_code: HTTP status code to return
        should_fail: Whether requests should fail
        
    Returns:
        Mock HTTP session
    """
    session = Mock()
    
    # Mock response object
    mock_response = Mock()
    mock_response.status = status_code
    
    if should_fail:
        mock_response.json = AsyncMock(side_effect=Exception("Mock failure"))
        mock_response.text = AsyncMock(return_value="Mock error")
    else:
        # Default responses
        default_responses = {
            'ollama_chat': {
                'message': {'content': 'Mock Ollama response'}
            },
            'lm_studio_chat': {
                'choices': [{'message': {'content': 'Mock LM Studio response'}}]
            },
            'ollama_tags': {'models': []},
            'lm_studio_models': {'data': []}
        }
        
        responses = server_responses or default_responses
        
        def get_response_for_url(url):
            if 'ollama' in url and 'chat' in url:
                return responses.get('ollama_chat', default_responses['ollama_chat'])
            elif 'ollama' in url and 'tags' in url:
                return responses.get('ollama_tags', default_responses['ollama_tags'])
            elif 'chat/completions' in url:
                return responses.get('lm_studio_chat', default_responses['lm_studio_chat'])
            elif 'models' in url:
                return responses.get('lm_studio_models', default_responses['lm_studio_models'])
            else:
                return {'message': 'Unknown endpoint'}
        
        # Mock the response based on URL
        async def mock_json(*args, **kwargs):
            # This is a simplified approach - in real tests you'd want to 
            # inspect the actual URL being called
            return get_response_for_url('default')
        
        mock_response.json = mock_json
        mock_response.text = AsyncMock(return_value="Mock response text")
    
    # Mock context manager for requests
    mock_context = Mock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    session.post.return_value = mock_context
    session.get.return_value = mock_context
    
    return session


def create_mock_llm_with_predefined_responses():
    """Create a mock LLM service with predefined responses for common queries."""
    responses = {
        "hello": "Hello! How can I help you today?",
        "what is my name": "I don't have your name stored yet.",
        "my name is john": "Got it, John! I'll remember that.",
        "what do you remember about me": "I found some memories about you.",
        "forget everything": "I've erased all my memories and reset my knowledge network.",
        "how are you": "I'm doing well, thank you for asking!",
        "goodbye": "Goodbye! Have a great day!",
        "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
        "what's the weather": "I don't have access to current weather data.",
        "help": "I'm here to help! You can ask me questions or have a conversation."
    }
    
    return MockLLMService(responses=responses)


def create_failing_mock_llm():
    """Create a mock LLM service that simulates failures."""
    mock = MockLLMService()
    
    async def failing_get_response(user_text: str, context: str) -> str:
        mock._stats['requests'] += 1
        mock._stats['failures'] += 1
        raise Exception("Mock LLM failure")
    
    mock.get_response = failing_get_response
    mock.is_available = lambda: False
    mock.health_check = AsyncMock(return_value=False)
    
    return mock


def create_slow_mock_llm(delay: float = 1.0):
    """Create a mock LLM service that simulates slow responses.
    
    Args:
        delay: Delay in seconds before responding
        
    Returns:
        Mock LLM service with artificial delay
    """
    import asyncio
    
    mock = MockLLMService()
    original_get_response = mock.get_response
    
    async def slow_get_response(user_text: str, context: str) -> str:
        await asyncio.sleep(delay)
        return await original_get_response(user_text, context)
    
    mock.get_response = slow_get_response
    
    return mock
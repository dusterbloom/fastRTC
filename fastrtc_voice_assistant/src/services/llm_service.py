"""LLM service implementation for FastRTC Voice Assistant.

This module provides LLM interaction capabilities supporting both
Ollama and LM Studio backends with context building and error handling.
"""

import asyncio
import aiohttp
import re
from typing import Optional, Dict, Any

from ..core.interfaces import LLMService as LLMServiceInterface
from ..core.exceptions import LLMError
from ..memory.cache import ResponseCache
from ..memory.conversation import ConversationBuffer
from ..memory.manager import AMemMemoryManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMService(LLMServiceInterface):
    """LLM service for conversation handling.
    
    This class provides LLM interaction capabilities with support for
    both Ollama and LM Studio backends, context building with memory
    integration, and comprehensive error handling.
    """
    
    def __init__(self, 
                 use_ollama: bool = True,
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2:3b",
                 lm_studio_url: str = "http://localhost:1234/v1",
                 lm_studio_model: str = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                 timeout: float = 30.0,
                 max_tokens: int = 250,
                 temperature: float = 0.7):
        """Initialize the LLM service.
        
        Args:
            use_ollama: Whether to use Ollama (True) or LM Studio (False)
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            lm_studio_url: LM Studio server URL
            lm_studio_model: LM Studio model name
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.lm_studio_url = lm_studio_url
        self.lm_studio_model = lm_studio_model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.response_cache: Optional[ResponseCache] = None
        self.conversation_buffer: Optional[ConversationBuffer] = None
        self.memory_manager: Optional[AMemMemoryManager] = None
        
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'cache_hits': 0,
            'timeouts': 0,
            'connection_errors': 0
        }
        
        logger.info(f"LLM service initialized with backend: {'Ollama' if use_ollama else 'LM Studio'}")
    
    async def initialize(self, http_session: aiohttp.ClientSession,
                        response_cache: Optional[ResponseCache] = None,
                        conversation_buffer: Optional[ConversationBuffer] = None,
                        memory_manager: Optional[AMemMemoryManager] = None):
        """Initialize the LLM service with dependencies.
        
        Args:
            http_session: HTTP session for making requests
            response_cache: Response cache instance
            conversation_buffer: Conversation buffer instance
            memory_manager: Memory manager instance
        """
        self.http_session = http_session
        self.response_cache = response_cache
        self.conversation_buffer = conversation_buffer
        self.memory_manager = memory_manager
        
        logger.info("LLM service dependencies initialized")
    
    def _get_llm_context_prompt(self, context: str) -> str:
        """Build LLM context prompt with memory and conversation history.
        
        Args:
            context: User context from memory
            
        Returns:
            str: Complete system prompt with context
        """
        # Get recent conversation context
        recent_conv = ""
        if self.conversation_buffer:
            recent_conv = self.conversation_buffer.get_recent_context()
        
        system_prompt = f"""You are Echo, a friendly and multilingual voice assistant with advanced memory capabilities.
        Keep responses concise and natural for voice interaction.

        {context}

        Remember:
        - Your name is Echo (the assistant)
        - You have an advanced agentic memory system that learns and evolves
        - You can form connections between different memories and concepts
        - You answers the user always in the same language they used to ask
        - You can remember the user's name and preferences
        - You can forget all memories if the user requests it
        - Be warm and conversational
        {recent_conv}"""
        
        return system_prompt.strip()
    
    async def get_response(self, user_text: str, context: str) -> str:
        """Get LLM response to user input.
        
        Args:
            user_text: User's input text
            context: Conversation context from memory
            
        Returns:
            str: LLM response text
            
        Raises:
            LLMError: If LLM request fails
        """
        self._stats['requests'] += 1
        
        # Check cache first
        if self.response_cache:
            cached_response = self.response_cache.get(user_text)
            if cached_response:
                self._stats['cache_hits'] += 1
                logger.debug(f"Cache hit for user text: {user_text[:50]}...")
                return cached_response
        
        # Handle recall phrases
        recall_phrases = [
            'what do you remember', 'what do you know about me', 
            'tell me about myself', 'what is my name', 'who am i'
        ]
        if any(phrase in user_text.lower() for phrase in recall_phrases):
            if self.memory_manager:
                memory_search_result = await self.memory_manager.search_memories(user_text)
                if self.response_cache:
                    self.response_cache.put(user_text, memory_search_result)
                self._stats['successes'] += 1
                return memory_search_result
        
        # Handle memory deletion requests
        delete_phrases = ["delete all your memory", "reset", "forget everything"]
        if any(phrase in user_text.lower() for phrase in delete_phrases):
            if self.memory_manager:
                result = await self.memory_manager.clear_memory()
                response = ("I've erased all my memories and reset my knowledge network." 
                          if result else "Sorry, I couldn't erase my memories due to an internal error.")
                if self.response_cache:
                    self.response_cache.put(user_text, response)
                self._stats['successes'] += 1
                return response
        
        # Handle name extraction and acknowledgment
        if self.memory_manager:
            potential_name = self.memory_manager.extract_user_name(user_text)
            if potential_name:
                self.memory_manager.update_local_cache(user_text, "personal_info", is_current_turn_extraction=True)
                
                # Check if this is a simple name introduction
                name_pattern = rf"(my name is|i'?m|call me|i am)\s+{re.escape(potential_name)}\s*\.?"
                if re.fullmatch(name_pattern, user_text.lower().strip(), re.IGNORECASE):
                    ack = f"Got it, {potential_name}! I'll remember that and my memory system will create connections with this information."
                    await self.memory_manager.add_memory(user_text, ack)
                    if self.response_cache:
                        self.response_cache.put(user_text, ack)
                    self._stats['successes'] += 1
                    return ack
        
        # Generate LLM response
        try:
            response = await self._generate_llm_response(user_text, context)
            
            # Store in memory and cache
            if self.memory_manager:
                await self.memory_manager.add_memory(user_text, response)
            
            if self.response_cache:
                self.response_cache.put(user_text, response)
            
            self._stats['successes'] += 1
            return response
            
        except Exception as e:
            self._stats['failures'] += 1
            logger.error(f"LLM request failed: {e}")
            raise LLMError(f"LLM request failed: {e}")
    
    async def _generate_llm_response(self, user_text: str, context: str) -> str:
        """Generate response using configured LLM backend.
        
        Args:
            user_text: User's input text
            context: Conversation context
            
        Returns:
            str: Generated response
            
        Raises:
            LLMError: If LLM request fails
        """
        if not self.http_session:
            raise LLMError("HTTP session not available for LLM call")
        
        system_prompt = self._get_llm_context_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            if self.use_ollama:
                return await self._call_ollama(messages)
            else:
                return await self._call_lm_studio(messages)
                
        except aiohttp.ClientConnectorError as e:
            self._stats['connection_errors'] += 1
            url_used = self.ollama_url if self.use_ollama else self.lm_studio_url
            error_msg = f"Unable to connect to LLM server at {url_used}. Is the server running?"
            logger.error(f"❌ LLM Connection Error: {e}. {error_msg}")
            raise LLMError(error_msg)
            
        except asyncio.TimeoutError:
            self._stats['timeouts'] += 1
            url_used = self.ollama_url if self.use_ollama else self.lm_studio_url
            error_msg = f"LLM request timed out after {self.timeout}s to {url_used}"
            logger.error(f"❌ {error_msg}")
            raise LLMError("Request is taking longer than usual. Please try again.")
            
        except Exception as e:
            logger.error(f"❌ Unexpected LLM request error: {e}")
            raise LLMError(f"Unexpected error during LLM request: {e}")
    
    async def _call_ollama(self, messages: list) -> str:
        """Call Ollama API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Generated response
            
        Raises:
            LLMError: If Ollama request fails
        """
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with self.http_session.post(
            f"{self.ollama_url}/api/chat", 
            json=payload, 
            timeout=timeout
        ) as response:
            if response.status == 200:
                data = await response.json()
                content = data.get("message", {}).get("content", "").strip()
                if not content:
                    raise LLMError("Empty response from Ollama")
                return content
            else:
                error_body = await response.text()
                logger.error(f"⚠️ Ollama request failed: Status {response.status}, Body: {error_body[:200]}")
                raise LLMError(f"Ollama request failed with status {response.status}")
    
    async def _call_lm_studio(self, messages: list) -> str:
        """Call LM Studio API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Generated response
            
        Raises:
            LLMError: If LM Studio request fails
        """
        payload = {
            "model": self.lm_studio_model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with self.http_session.post(
            f"{self.lm_studio_url}/chat/completions", 
            json=payload, 
            timeout=timeout
        ) as response:
            if response.status == 200:
                data = await response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMError("No choices in LM Studio response")
                
                content = choices[0].get("message", {}).get("content", "").strip()
                if not content:
                    raise LLMError("Empty response from LM Studio")
                return content
            else:
                error_body = await response.text()
                logger.error(f"⚠️ LM Studio request failed: Status {response.status}, Body: {error_body[:200]}")
                raise LLMError(f"LM Studio request failed with status {response.status}")
    
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy and responsive.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        if not self.http_session:
            return False
        
        try:
            if self.use_ollama:
                # Check Ollama health
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with self.http_session.get(
                    f"{self.ollama_url}/api/tags", 
                    timeout=timeout
                ) as response:
                    return response.status == 200
            else:
                # Check LM Studio health
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with self.http_session.get(
                    f"{self.lm_studio_url}/models", 
                    timeout=timeout
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if the LLM service is available and ready.
        
        Returns:
            bool: True if service is ready, False otherwise
        """
        return self.http_session is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM service statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        total_requests = self._stats['requests']
        success_rate = (self._stats['successes'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'backend': 'Ollama' if self.use_ollama else 'LM Studio',
            'model': self.ollama_model if self.use_ollama else self.lm_studio_model,
            'url': self.ollama_url if self.use_ollama else self.lm_studio_url,
            'total_requests': total_requests,
            'successes': self._stats['successes'],
            'failures': self._stats['failures'],
            'success_rate': f"{success_rate:.1f}%",
            'cache_hits': self._stats['cache_hits'],
            'timeouts': self._stats['timeouts'],
            'connection_errors': self._stats['connection_errors'],
            'timeout_seconds': self.timeout,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
    
    async def shutdown(self):
        """Shutdown the LLM service gracefully."""
        logger.info("LLM service shutdown complete")
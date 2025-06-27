"""Async-compatible LLM service implementation for Threading Pipeline.

This module provides an async-compatible LLM service that works within FastRTC's
WebRTC requirements while maintaining the threading pipeline architecture.
"""

import asyncio
import aiohttp
import json
import re
from typing import Optional, Dict, Any, Generator
from ..core.exceptions import LLMError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SyncLLMService:
    """
    Async-compatible LLM service for threading pipeline.
    
    This class provides the same interface as the async LLMService
    but uses asyncio.run() to handle async HTTP requests within sync context,
    ensuring compatibility with FastRTC's WebRTC requirements.
    """
    
    def __init__(self, 
                 use_ollama: bool = True,
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3:8b-instruct-q4_K_M",
                 lm_studio_url: str = "http://localhost:1234/v1",
                 lm_studio_model: str = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                 timeout: float = 30.0,
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        """Initialize the sync LLM service.
        
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
        
        # Create aiohttp session for async requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'timeouts': 0,
            'connection_errors': 0
        }
        
        logger.info(f"Async-compatible LLM service initialized with backend: {'Ollama' if use_ollama else 'LM Studio'}")
        logger.info(f"🔧 Async LLM Debug: max_tokens={self.max_tokens}, temperature={self.temperature}, timeout={self.timeout}")
    
    def get_response(self, user_text: str, context: str = "") -> str:
        """Get LLM response to user input using async patterns in sync context.
        
        Args:
            user_text: User's input text
            context: Conversation context from memory
            
        Returns:
            str: LLM response text
            
        Raises:
            LLMError: If LLM request fails
        """
        self._stats['requests'] += 1
        
        try:
            # Run async function in new event loop
            response = asyncio.run(self._generate_llm_response_async(user_text, context))
            self._stats['successes'] += 1
            return response
            
        except Exception as e:
            self._stats['failures'] += 1
            logger.error(f"Async LLM request failed: {e}")
            raise LLMError(f"Async LLM request failed: {e}")
    
    def stream_response(self, user_text: str, context: str = "") -> Generator[str, None, None]:
        """
        Stream LLM response token by token using async patterns in sync context.
        
        Args:
            user_text: User's input text
            context: Conversation context from memory
            
        Yields:
            str: Individual tokens or token chunks from LLM
            
        Raises:
            LLMError: If streaming request fails
        """
        self._stats['requests'] += 1
        
        try:
            # Run async streaming in new event loop
            async def _run_stream():
                tokens = []
                async for token in self._stream_llm_response_async(user_text, context):
                    tokens.append(token)
                return tokens
            
            tokens = asyncio.run(_run_stream())
            for token in tokens:
                yield token
            self._stats['successes'] += 1
            
        except Exception as e:
            self._stats['failures'] += 1
            logger.error(f"Async LLM streaming request failed: {e}")
            raise LLMError(f"Async LLM streaming request failed: {e}")
    
    async def _generate_llm_response_async(self, user_text: str, context: str) -> str:
        """Generate response using configured LLM backend asynchronously.
        
        Args:
            user_text: User's input text
            context: Conversation context
            
        Returns:
            str: Generated response
            
        Raises:
            LLMError: If LLM request fails
        """
        system_prompt = self._get_llm_context_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            if self.use_ollama:
                return await self._call_ollama_async(messages)
            else:
                return await self._call_lm_studio_async(messages)
                
        except aiohttp.ClientConnectorError as e:
            self._stats['connection_errors'] += 1
            url_used = self.ollama_url if self.use_ollama else self.lm_studio_url
            error_msg = f"Unable to connect to LLM server at {url_used}. Is the server running?"
            logger.error(f"❌ Async LLM Connection Error: {e}. {error_msg}")
            raise LLMError(error_msg)
            
        except asyncio.TimeoutError:
            self._stats['timeouts'] += 1
            url_used = self.ollama_url if self.use_ollama else self.lm_studio_url
            error_msg = f"LLM request timed out after {self.timeout}s to {url_used}"
            logger.error(f"❌ {error_msg}")
            raise LLMError("Request is taking longer than usual. Please try again.")
            
        except Exception as e:
            logger.error(f"❌ Unexpected async LLM request error: {e}")
            raise LLMError(f"Unexpected error during async LLM request: {e}")
    
    async def _stream_llm_response_async(self, user_text: str, context: str):
        """
        Stream response from configured LLM backend asynchronously.
        
        Args:
            user_text: User's input text
            context: Conversation context
            
        Yields:
            str: Token chunks from LLM
            
        Raises:
            LLMError: If streaming request fails
        """
        system_prompt = self._get_llm_context_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            if self.use_ollama:
                async for token in self._stream_ollama_async(messages):
                    yield token
            else:
                async for token in self._stream_lm_studio_async(messages):
                    yield token
                    
        except aiohttp.ClientConnectorError as e:
            self._stats['connection_errors'] += 1
            url_used = self.ollama_url if self.use_ollama else self.lm_studio_url
            error_msg = f"Unable to connect to LLM server at {url_used}. Is the server running?"
            logger.error(f"❌ Async LLM Connection Error: {e}. {error_msg}")
            raise LLMError(error_msg)
            
        except asyncio.TimeoutError:
            self._stats['timeouts'] += 1
            url_used = self.ollama_url if self.use_ollama else self.lm_studio_url
            error_msg = f"LLM streaming request timed out after {self.timeout}s to {url_used}"
            logger.error(f"❌ {error_msg}")
            raise LLMError("Request is taking longer than usual. Please try again.")
            
        except Exception as e:
            logger.error(f"❌ Unexpected async LLM streaming error: {e}")
            raise LLMError(f"Unexpected error during async LLM streaming: {e}")
    
    async def _call_ollama_async(self, messages: list) -> str:
        """Call Ollama API asynchronously.
        
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
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.ollama_url}/api/chat", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if not content:
                        raise LLMError("Empty response from Ollama")
                    
                    # Log response length for debugging
                    logger.info(f"🔧 Async LLM Debug: Ollama response length: {len(content)} chars, ~{len(content.split())} words")
                    return content
                else:
                    error_body = await response.text()
                    logger.error(f"⚠️ Ollama request failed: Status {response.status}, Body: {error_body[:200]}")
                    raise LLMError(f"Ollama request failed with status {response.status}")
    
    async def _call_lm_studio_async(self, messages: list) -> str:
        """Call LM Studio API asynchronously.
        
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
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.lm_studio_url}/chat/completions", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMError("No choices in LM Studio response")
                    
                    content = choices[0].get("message", {}).get("content", "").strip()
                    if not content:
                        raise LLMError("Empty response from LM Studio")
                    
                    # Log response length for debugging
                    logger.info(f"🔧 Async LLM Debug: LM Studio response length: {len(content)} chars, ~{len(content.split())} words")
                    return content
                else:
                    error_body = await response.text()
                    logger.error(f"⚠️ LM Studio request failed: Status {response.status}, Body: {error_body[:200]}")
                    raise LLMError(f"LM Studio request failed with status {response.status}")
    
    async def _stream_ollama_async(self, messages: list):
        """
        Stream response from Ollama API asynchronously.
        
        Args:
            messages: List of message dictionaries
            
        Yields:
            str: Token chunks from Ollama
            
        Raises:
            LLMError: If Ollama streaming fails
        """
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.ollama_url}/api/chat", 
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                line_text = line.decode('utf-8').strip()
                                if not line_text:
                                    continue
                                    
                                chunk_data = json.loads(line_text)
                                
                                if chunk_data.get("done", False):
                                    break
                                
                                content = chunk_data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                                    
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.warning(f"Error parsing Ollama stream chunk: {e}")
                                continue
                else:
                    error_body = await response.text()
                    logger.error(f"⚠️ Ollama streaming failed: Status {response.status}, Body: {error_body[:200]}")
                    raise LLMError(f"Ollama streaming failed with status {response.status}")
    
    async def _stream_lm_studio_async(self, messages: list):
        """
        Stream response from LM Studio API asynchronously.
        
        Args:
            messages: List of message dictionaries
            
        Yields:
            str: Token chunks from LM Studio
            
        Raises:
            LLMError: If LM Studio streaming fails
        """
        payload = {
            "model": self.lm_studio_model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True  # Enable streaming
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.lm_studio_url}/chat/completions", 
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                line_text = line.decode('utf-8').strip()
                                
                                # Skip SSE prefixes and empty lines
                                if line_text.startswith('data: '):
                                    line_text = line_text[6:]
                                
                                if not line_text or line_text == '[DONE]':
                                    continue
                                
                                chunk_data = json.loads(line_text)
                                choices = chunk_data.get("choices", [])
                                
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                        
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.warning(f"Error parsing LM Studio stream chunk: {e}")
                                continue
                else:
                    error_body = await response.text()
                    logger.error(f"⚠️ LM Studio streaming failed: Status {response.status}, Body: {error_body[:200]}")
                    raise LLMError(f"LM Studio streaming failed with status {response.status}")
    
    def _get_llm_context_prompt(self, context: str) -> str:
        """Build LLM context prompt.
        
        Args:
            context: User context from memory
            
        Returns:
            str: Complete system prompt with context
        """
        # For threading pipeline, use a simplified prompt since we don't have 
        # access to conversation buffer and memory in the sync context
        system_prompt = f"""You are Echo, a friendly and multilingual voice assistant.
Keep responses concise and natural for voice interaction.

{context}

Remember:
- Your name is Echo (the assistant)
- You answer the user always in the same language they used to ask
- Be warm and conversational
- Keep responses brief and to the point"""
        
        return system_prompt.strip()
    
    def health_check(self) -> bool:
        """Check if the LLM service is healthy and responsive.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            return asyncio.run(self._health_check_async())
        except Exception as e:
            logger.warning(f"Async LLM health check failed: {e}")
            return False
    
    async def _health_check_async(self) -> bool:
        """Async health check implementation."""
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.use_ollama:
                    # Check Ollama health
                    async with session.get(f"{self.ollama_url}/api/tags") as response:
                        return response.status == 200
                else:
                    # Check LM Studio health
                    async with session.get(f"{self.lm_studio_url}/models") as response:
                        return response.status == 200
                        
        except Exception as e:
            logger.warning(f"Async LLM health check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if the LLM service is available and ready.
        
        Returns:
            bool: True if service is ready, False otherwise
        """
        return True  # Service is always available since we create sessions per request
    
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
            'timeouts': self._stats['timeouts'],
            'connection_errors': self._stats['connection_errors'],
            'timeout_seconds': self.timeout,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'service_type': 'async-compatible'
        }
    
    def shutdown(self):
        """Shutdown the async-compatible LLM service gracefully."""
        # No persistent session to close since we create sessions per request
        logger.info("Async-compatible LLM service shutdown complete")
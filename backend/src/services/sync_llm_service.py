"""
Synchronous LLM Service for Threading Pipeline

Pure threading-based LLM service that uses standard HTTP requests
to call Ollama without any async/await dependencies.
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SyncLLMService:
    """
    Synchronous LLM service for pure threading pipeline.
    
    Uses requests library to make direct HTTP calls to Ollama API
    without any async dependencies.
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b",
        timeout: float = 30.0,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        Initialize synchronous LLM service.
        
        Args:
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Statistics
        self.stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'timeouts': 0,
            'connection_errors': 0
        }
        
        logger.info(f"🧠 SyncLLMService initialized: {ollama_url} model={ollama_model}")
    
    def get_response(self, user_text: str, context: str = "") -> str:
        """
        Get LLM response synchronously.
        
        Args:
            user_text: User input text
            context: Additional context (unused in simple implementation)
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If LLM request fails
        """
        self.stats['requests'] += 1
        start_time = time.time()
        
        try:
            logger.info(f"🧠 [SYNC_LLM] Calling Ollama for: '{user_text[:50]}...'")
            
            # Prepare messages in Ollama format
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Respond naturally and concisely."
                },
                {
                    "role": "user", 
                    "content": user_text
                }
            ]
            
            # Prepare Ollama API payload
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            # Make HTTP request to Ollama
            logger.debug(f"🧠 [SYNC_LLM] POST to {self.ollama_url}/api/chat")
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Check response status
            if response.status_code == 200:
                try:
                    result = response.json()
                    response_text = result.get("message", {}).get("content", "")
                    
                    if response_text:
                        processing_time = time.time() - start_time
                        self.stats['successes'] += 1
                        logger.info(f"🧠 [SYNC_LLM] Got response in {processing_time:.2f}s: '{response_text[:100]}...'")
                        return response_text.strip()
                    else:
                        logger.error("🧠 [SYNC_LLM] Empty response from Ollama")
                        raise Exception("Empty response from LLM")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"🧠 [SYNC_LLM] Invalid JSON response: {e}")
                    raise Exception(f"Invalid JSON response from LLM: {e}")
            else:
                logger.error(f"🧠 [SYNC_LLM] HTTP {response.status_code}: {response.text}")
                raise Exception(f"HTTP {response.status_code} from LLM server")
                
        except requests.exceptions.Timeout:
            self.stats['timeouts'] += 1
            error_msg = f"LLM request timed out after {self.timeout}s"
            logger.error(f"🧠 [SYNC_LLM] {error_msg}")
            raise Exception(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            self.stats['connection_errors'] += 1
            error_msg = f"Cannot connect to Ollama at {self.ollama_url}. Is it running?"
            logger.error(f"🧠 [SYNC_LLM] Connection error: {e}")
            raise Exception(error_msg)
            
        except Exception as e:
            self.stats['failures'] += 1
            logger.error(f"🧠 [SYNC_LLM] Unexpected error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.stats.copy()
    
    def health_check(self) -> bool:
        """
        Check if Ollama is accessible.
        
        Returns:
            True if Ollama is responding, False otherwise
        """
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
"""Network and service configuration for FastRTC Voice Assistant."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import aiohttp

@dataclass
class ServiceEndpoints:
    """Service endpoint configuration."""
    ollama: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    lm_studio: str = field(default_factory=lambda: os.getenv("LM_STUDIO_URL", "http://192.168.1.5:1234/v1"))
    qdrant: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    redis: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    prometheus: str = field(default_factory=lambda: os.getenv("PROMETHEUS_URL", "http://localhost:9090"))
    grafana: str = field(default_factory=lambda: os.getenv("GRAFANA_URL", "http://localhost:3000"))

@dataclass
class TimeoutConfig:
    """Timeout configuration for various operations."""
    # Core operation timeouts
    stt_transcription: float = 8.0
    llm_response: float = 10.0
    tts_synthesis: float = 15.0
    memory_operation: float = 4.0
    
    # Network timeouts
    http_total: float = 20.0
    http_connect: float = 5.0
    http_read: float = 15.0
    
    # System timeouts
    startup: float = 30.0
    shutdown: float = 15.0
    health_check: float = 5.0
    
    # Async operation timeouts
    async_task: float = 30.0
    queue_join: float = 10.0

@dataclass
class HTTPClientConfig:
    """HTTP client configuration."""
    max_connections: int = 100
    max_connections_per_host: int = 5
    enable_ssl: bool = True
    user_agent: str = "FastRTC-Voice-Assistant/1.0.0"
    
    def create_timeout(self, timeout_config: TimeoutConfig) -> aiohttp.ClientTimeout:
        """Create aiohttp ClientTimeout from timeout configuration."""
        return aiohttp.ClientTimeout(
            total=timeout_config.http_total,
            connect=timeout_config.http_connect,
            sock_read=timeout_config.http_read
        )
    
    def create_connector(self) -> aiohttp.TCPConnector:
        """Create aiohttp TCPConnector from configuration."""
        return aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ssl=self.enable_ssl
        )
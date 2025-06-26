"""Test-specific configuration."""

import os
from dataclasses import dataclass, field
from typing import List

# Ensure the test chroma_db directory exists
TEST_CHROMA_DB_PATH = "backend/tests/chroma_db"
os.makedirs(TEST_CHROMA_DB_PATH, exist_ok=True)

@dataclass
class TestLLMConfig:
    """LLM configuration for testing."""
    use_ollama: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b-instruct-q4_K_M"
    timeout: float = 15.0
    max_tokens: int = 150
    temperature: float = 0.1

@dataclass
class TestMemoryConfig:
    """Memory configuration for testing."""
    cache_ttl_seconds: int = 60
    db_path: str = TEST_CHROMA_DB_PATH

@dataclass
class TestAudioConfig:
    """Audio configuration for testing."""
    sample_rate: int = 16000
    chunk_duration: float = 5.0

@dataclass
class TestConfig:
    """Main container for test configurations."""
    llm: TestLLMConfig = field(default_factory=TestLLMConfig)
    memory: TestMemoryConfig = field(default_factory=TestMemoryConfig)
    audio: TestAudioConfig = field(default_factory=TestAudioConfig)

def get_test_config() -> TestConfig:
    """Returns a complete configuration for testing."""
    return TestConfig()
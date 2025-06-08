# Configuration Guide

## Overview

This guide provides comprehensive information about configuring the FastRTC Voice Assistant system. The configuration covers vector databases (ChromaDB/Qdrant), memory systems (A-MEM/Mem0), LLM services, and all other components.

## Architecture Overview

The FastRTC Voice Assistant uses the following data storage and processing components:

### Vector Databases
- **ChromaDB**: Primary vector database for embeddings (persistent local storage)
- **Qdrant**: Alternative vector database option (for distributed deployments)

### Memory Systems
- **A-MEM**: Custom agentic memory system with evolution capabilities
- **Mem0**: Advanced memory framework integration
- **Redis**: Caching layer for responses and rate limiting

### LLM Services
- **Ollama**: Local LLM inference (primary)
- **LM Studio**: Alternative local LLM service
- **External APIs**: Support for cloud LLM services

### Audio Processing
- **FastRTC**: WebRTC integration for real-time audio
- **HuggingFace**: STT models and transformers
- **Kokoro**: Multilingual TTS engine

## Configuration Structure

### Main Configuration File

```yaml
# config/production.yml
app:
  name: "FastRTC Voice Assistant"
  version: "1.0.0"
  environment: "production"
  debug: false

# Audio processing configuration
audio:
  sample_rate: 16000
  chunk_duration: 2.0
  noise_threshold: 0.15
  minimal_silent_frame_duration_ms: 20
  max_audio_length_seconds: 30
  
  # Audio processing engines
  stt:
    engine: "huggingface"  # huggingface, whisper
    model: "openai/whisper-base"
    device: "auto"  # auto, cpu, cuda
    
  tts:
    engine: "kokoro"
    preferred_voice: "af_heart"
    fallback_voices: ["af_bella", "af_sarah"]
    speed: 1.05
    quality: "high"

# Vector database configuration
vector_db:
  # Primary vector database
  primary: "chromadb"  # chromadb, qdrant
  
  # ChromaDB configuration
  chromadb:
    persist_directory: "./chroma_db"
    collection_name: "memories"
    embedding_model: "all-MiniLM-L6-v2"
    
  # Qdrant configuration (alternative)
  qdrant:
    url: "http://localhost:6333"
    collection_name: "voice_memories"
    vector_size: 384
    distance: "cosine"

# Memory system configuration
memory:
  # A-MEM configuration
  a_mem:
    llm_model: "llama3.2:3b"
    embedder_model: "nomic-embed-text"
    evolution_threshold: 50
    max_memory_entries: 10000
    
  # Mem0 configuration
  mem0:
    enabled: true
    config_path: "./config/mem0_config.yml"
    
  # Response caching
  cache:
    enabled: true
    ttl_seconds: 180
    max_entries: 1000
    backend: "redis"  # redis, memory

# LLM service configuration
llm:
  # Primary LLM service
  primary: "ollama"  # ollama, lm_studio, openai, anthropic
  
  # Ollama configuration
  ollama:
    url: "http://localhost:11434"
    model: "llama3:8b-instruct-q4_K_M"
    temperature: 0.7
    max_tokens: 2048
    timeout: 30
    
  # LM Studio configuration
  lm_studio:
    url: "http://192.168.1.5:1234/v1"
    model: "mistral-nemo-instruct-2407"
    api_key: null
    
  # External API configuration
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    organization: null
    
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet-20240229"

# Redis configuration
redis:
  url: "redis://localhost:6379/0"
  password: null
  ssl: false
  
  # Connection pool settings
  pool:
    max_connections: 20
    retry_on_timeout: true
    socket_timeout: 5
    
  # Cache settings
  cache:
    default_ttl: 3600
    key_prefix: "voice_assistant:"

# Language detection configuration
language:
  # Primary detector
  primary: "hybrid"  # mediapipe, keyword, hybrid
  
  # MediaPipe configuration
  mediapipe:
    model_path: "./models/mediapipe/language_detector.tflite"
    confidence_threshold: 0.7
    
  # Keyword detector (fallback)
  keyword:
    confidence_threshold: 0.2
    min_matches: 2
    
  # Voice mapping
  voice_mapping:
    default_voice: "af_heart"
    language_voices:
      a: ["af_heart", "af_bella", "af_sarah"]           # American English
      b: ["bf_emma", "bf_isabella", "bm_george"]        # British English
      i: ["if_sara", "im_nicola"]                       # Italian
      e: ["ef_dora", "em_alex", "em_santa"]             # Spanish
      f: ["ff_siwis"]                                   # French
      p: ["pf_dora", "pm_alex", "pm_santa"]             # Portuguese
      j: ["jf_alpha", "jf_gongitsune", "jm_kumo"]       # Japanese
      z: ["zf_xiaobei", "zf_xiaoni", "zm_yunjian", "zm_yunxi"]  # Chinese
      h: ["hf_alpha", "hf_beta", "hm_omega"]            # Hindi

# FastRTC integration
fastrtc:
  # WebRTC configuration
  ice_servers:
    - urls: "stun:stun.l.google.com:19302"
    - urls: "turn:your-turn-server.com:3478"
      username: "your-username"
      credential: "your-password"
      
  # Audio stream settings
  audio:
    codec: "opus"
    sample_rate: 48000
    channels: 1
    bitrate: 64000
    
  # Connection settings
  connection:
    timeout: 30
    max_retries: 3
    keepalive_interval: 25

# Security configuration
security:
  # Authentication
  auth:
    enabled: true
    method: "api_key"  # api_key, jwt, oauth
    
  # API key authentication
  api_key:
    header_name: "X-API-Key"
    valid_keys:
      - "${API_KEY_1}"
      - "${API_KEY_2}"
      
  # JWT authentication
  jwt:
    secret: "${JWT_SECRET}"
    algorithm: "HS256"
    expiration: 86400  # 24 hours
    
  # Rate limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
    
  # CORS settings
  cors:
    enabled: true
    origins: ["https://yourdomain.com"]
    methods: ["GET", "POST"]
    headers: ["Content-Type", "Authorization"]

# Monitoring and logging
monitoring:
  # Prometheus metrics
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
    
  # Health checks
  health:
    enabled: true
    path: "/health"
    interval: 30
    
  # Performance monitoring
  performance:
    track_latency: true
    track_memory: true
    track_errors: true

# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json"  # json, text
  
  # File logging
  file:
    enabled: true
    path: "/var/log/voice-assistant/app.log"
    max_size: "100MB"
    backup_count: 5
    
  # Console logging
  console:
    enabled: true
    colored: true
    
  # Structured logging fields
  fields:
    service: "voice-assistant"
    version: "1.0.0"
    environment: "production"

# Development settings
development:
  # Hot reload
  hot_reload: false
  
  # Debug features
  debug_audio: false
  debug_memory: false
  debug_llm: false
  
  # Test mode
  test_mode: false
  mock_services: false
```

## Environment Variables

### Core Environment Variables

```bash
# Application settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Vector database settings
CHROMADB_PATH=./chroma_db
QDRANT_URL=http://localhost:6333

# Redis settings
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# LLM service settings
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct-q4_K_M
LM_STUDIO_URL=http://192.168.1.5:1234/v1

# External API keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Security settings
API_KEY_1=your-primary-api-key
API_KEY_2=your-secondary-api-key
JWT_SECRET=your-jwt-secret-key

# Audio processing
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_DURATION=2.0

# Memory settings
MEMORY_EVOLUTION_THRESHOLD=50
MEMORY_CACHE_TTL=180

# Performance settings
MAX_CONCURRENT_SESSIONS=100
REQUEST_TIMEOUT=30
MEMORY_LIMIT=500MB
```

## Vector Database Configuration

### ChromaDB Setup

ChromaDB is the primary vector database for local deployments:

```python
# ChromaDB configuration
CHROMADB_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "memories",
    "embedding_function": "all-MiniLM-L6-v2",
    "distance_metric": "cosine",
    "settings": {
        "anonymized_telemetry": False,
        "allow_reset": True
    }
}
```

#### ChromaDB Docker Configuration

```yaml
# For containerized ChromaDB (if needed)
services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chromadb_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
```

### Qdrant Setup

Qdrant is available as an alternative for distributed deployments:

```python
# Qdrant configuration
QDRANT_CONFIG = {
    "url": "http://localhost:6333",
    "collection_name": "voice_memories",
    "vector_config": {
        "size": 384,
        "distance": "Cosine"
    },
    "timeout": 30
}
```

#### Qdrant Docker Configuration

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
```

## Memory System Configuration

### A-MEM Configuration

The A-MEM (Agentic Memory) system provides evolving memory capabilities:

```yaml
# A-MEM specific configuration
a_mem:
  # Core settings
  llm_model: "llama3.2:3b"
  embedder_model: "nomic-embed-text"
  
  # Evolution settings
  evolution_threshold: 50
  consolidation_interval: 3600  # 1 hour
  max_memory_age_days: 30
  
  # Retrieval settings
  max_retrieval_results: 10
  similarity_threshold: 0.7
  
  # Storage settings
  persist_directory: "./a_mem_data"
  backup_interval: 86400  # 24 hours
```

### Mem0 Configuration

Mem0 provides advanced memory framework integration:

```yaml
# config/mem0_config.yml
version: "1.0"

# Vector store configuration
vector_store:
  provider: "chromadb"  # chromadb, qdrant, pinecone
  config:
    collection_name: "mem0_memories"
    persist_directory: "./mem0_db"

# Embedding configuration
embedder:
  provider: "sentence_transformers"
  config:
    model: "all-MiniLM-L6-v2"

# LLM configuration for memory operations
llm:
  provider: "ollama"
  config:
    model: "llama3:8b-instruct-q4_K_M"
    base_url: "http://localhost:11434"
    
# Memory configuration
memory:
  # Memory types
  types:
    - "episodic"    # Conversation memories
    - "semantic"    # Knowledge memories
    - "procedural"  # Skill memories
    
  # Retention policies
  retention:
    episodic_days: 30
    semantic_days: 365
    procedural_days: 180
    
  # Evolution settings
  evolution:
    enabled: true
    threshold: 50
    consolidation_strategy: "similarity_clustering"
```

## LLM Service Configuration

### Ollama Configuration

Ollama provides local LLM inference:

```yaml
ollama:
  # Connection settings
  base_url: "http://localhost:11434"
  timeout: 30
  max_retries: 3
  
  # Model settings
  model: "llama3:8b-instruct-q4_K_M"
  temperature: 0.7
  max_tokens: 2048
  top_p: 0.9
  
  # Performance settings
  num_ctx: 4096
  num_predict: 512
  repeat_penalty: 1.1
  
  # Available models
  models:
    - "llama3:8b-instruct-q4_K_M"
    - "llama3.2:3b"
    - "mistral:7b-instruct"
    - "codellama:7b-instruct"
```

### LM Studio Configuration

LM Studio as alternative local LLM service:

```yaml
lm_studio:
  # Connection settings
  base_url: "http://192.168.1.5:1234/v1"
  api_key: null  # Usually not required for local
  timeout: 30
  
  # Model settings
  model: "mistral-nemo-instruct-2407"
  temperature: 0.7
  max_tokens: 2048
  
  # Streaming settings
  stream: true
  stream_timeout: 5
```

## Audio Configuration

### STT (Speech-to-Text) Configuration

```yaml
stt:
  # Engine selection
  engine: "huggingface"  # huggingface, whisper, fastrtc
  
  # HuggingFace STT
  huggingface:
    model: "openai/whisper-base"
    device: "auto"  # auto, cpu, cuda:0
    torch_dtype: "float16"
    language: "auto"
    
  # Whisper configuration
  whisper:
    model_size: "base"  # tiny, base, small, medium, large
    language: "auto"
    device: "auto"
    
  # FastRTC Whisper
  fastrtc_whisper:
    model_path: "./models/whisper-base.bin"
    language: "auto"
    threads: 4
```

### TTS (Text-to-Speech) Configuration

```yaml
tts:
  # Engine selection
  engine: "kokoro"
  
  # Kokoro TTS
  kokoro:
    model_path: "./models/kokoro"
    sample_rate: 24000
    speed: 1.05
    
    # Voice configuration
    voices:
      a: ["af_heart", "af_bella", "af_sarah"]           # American English
      b: ["bf_emma", "bf_isabella", "bm_george"]        # British English
      i: ["if_sara", "im_nicola"]                       # Italian
      e: ["ef_dora", "em_alex", "em_santa"]             # Spanish
      f: ["ff_siwis"]                                   # French
      p: ["pf_dora", "pm_alex", "pm_santa"]             # Portuguese
      j: ["jf_alpha", "jf_gongitsune", "jm_kumo"]       # Japanese
      z: ["zf_xiaobei", "zf_xiaoni", "zm_yunjian", "zm_yunxi"]  # Chinese
      h: ["hf_alpha", "hf_beta", "hm_omega"]            # Hindi
      
    # Quality settings
    quality: "high"  # low, medium, high
    streaming: true
```

## Development Configuration

### Development Environment

```yaml
# config/development.yml
app:
  environment: "development"
  debug: true
  
# Relaxed settings for development
audio:
  sample_rate: 16000
  chunk_duration: 1.0  # Shorter for faster testing
  
memory:
  evolution_threshold: 5  # Lower threshold for testing
  cache_ttl_seconds: 60   # Shorter TTL for testing
  
llm:
  ollama:
    model: "llama3.2:3b"  # Smaller model for development
    
security:
  auth:
    enabled: false  # Disable auth for development
  rate_limiting:
    enabled: false  # Disable rate limiting
    
logging:
  level: "DEBUG"
  console:
    colored: true
    
development:
  hot_reload: true
  debug_audio: true
  debug_memory: true
  mock_services: false  # Set to true for offline development
```

### Testing Configuration

```yaml
# config/testing.yml
app:
  environment: "testing"
  debug: true
  
# Use in-memory databases for testing
vector_db:
  primary: "chromadb"
  chromadb:
    persist_directory: ":memory:"
    
redis:
  url: "redis://localhost:6379/15"  # Use different DB for tests
  
memory:
  evolution_threshold: 3  # Very low for testing
  cache_ttl_seconds: 10
  
llm:
  primary: "mock"  # Use mock LLM for testing
  
development:
  test_mode: true
  mock_services: true
```

## Configuration Validation

### Validation Script

```python
# scripts/validate_config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def validate_config(config_path: str) -> bool:
    """Validate configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = [
            'app', 'audio', 'vector_db', 'memory', 
            'llm', 'security', 'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        # Validate vector database configuration
        if not validate_vector_db_config(config['vector_db']):
            return False
            
        # Validate LLM configuration
        if not validate_llm_config(config['llm']):
            return False
            
        print("Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def validate_vector_db_config(config: Dict[str, Any]) -> bool:
    """Validate vector database configuration."""
    primary = config.get('primary')
    if primary not in ['chromadb', 'qdrant']:
        print(f"Invalid vector database: {primary}")
        return False
        
    if primary == 'chromadb':
        if 'chromadb' not in config:
            print("ChromaDB configuration missing")
            return False
    elif primary == 'qdrant':
        if 'qdrant' not in config:
            print("Qdrant configuration missing")
            return False
            
    return True

def validate_llm_config(config: Dict[str, Any]) -> bool:
    """Validate LLM configuration."""
    primary = config.get('primary')
    if primary not in ['ollama', 'lm_studio', 'openai', 'anthropic']:
        print(f"Invalid LLM service: {primary}")
        return False
        
    if primary in config:
        llm_config = config[primary]
        if 'model' not in llm_config:
            print(f"Model not specified for {primary}")
            return False
            
    return True

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/production.yml"
    validate_config(config_path)
```

## Configuration Management

### Environment-Specific Configurations

```bash
# Load configuration based on environment
export ENVIRONMENT=production
python -m src.core.main --config config/${ENVIRONMENT}.yml
```

### Configuration Override

```python
# Override configuration with environment variables
import os
from src.config.settings import load_config

config = load_config("config/production.yml")

# Override with environment variables
if os.getenv("OLLAMA_URL"):
    config.llm.ollama.url = os.getenv("OLLAMA_URL")
    
if os.getenv("REDIS_URL"):
    config.redis.url = os.getenv("REDIS_URL")
```

### Hot Configuration Reload

```python
# Hot reload configuration without restart
from src.config.settings import ConfigManager

config_manager = ConfigManager("config/production.yml")

# Watch for configuration changes
config_manager.watch_for_changes()

# Reload configuration
new_config = config_manager.reload()
```

This configuration guide reflects the actual architecture using ChromaDB, Qdrant, A-MEM, Mem0, and Redis as intended, rather than PostgreSQL which was incorrectly mentioned in the deployment documentation.
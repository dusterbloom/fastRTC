# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the FastRTC Voice Assistant in production environments. It covers deployment strategies, configuration management, monitoring, and maintenance procedures.

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB
- **Storage**: 20 GB available space
- **Network**: Stable internet connection (100 Mbps recommended)
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+

#### Recommended Requirements
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 50 GB SSD
- **Network**: Dedicated network interface
- **OS**: Ubuntu 22.04 LTS

#### GPU Requirements (Optional)
- **NVIDIA GPU**: GTX 1060 or better
- **VRAM**: 6 GB minimum
- **CUDA**: Version 11.8+
- **cuDNN**: Version 8.6+

### Software Dependencies

#### Core Dependencies
```bash
# Python 3.9+
python3 --version

# System packages
sudo apt-get update
sudo apt-get install -y \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    git \
    curl \
    wget
```

#### Optional Dependencies
```bash
# For GPU acceleration
sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit

# For monitoring
sudo apt-get install -y htop iotop nethogs

# For containerization
sudo apt-get install -y docker.io docker-compose
```

## Installation

### Production Installation

#### 1. Clone Repository
```bash
git clone https://github.com/your-org/fastrtc-voice-assistant.git
cd fastrtc-voice-assistant
```

#### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install tf-keras  # For compatibility
```

#### 4. Install Optional Dependencies
```bash
# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For monitoring
pip install prometheus-client grafana-api

# For advanced features
pip install mediapipe kokoro-onnx
```

### Docker Installation

#### 1. Build Docker Image
```bash
docker build -t fastrtc-voice-assistant:latest .
```

#### 2. Run with Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  voice-assistant:
    build: .
    ports:
      - "8080:8080"
      - "8443:8443"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
```

```bash
docker-compose up -d
```

## Configuration

### Environment Configuration

#### Production Environment Variables
```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_DURATION=2.0
AUDIO_NOISE_THRESHOLD=0.15

# Memory Configuration
MEMORY_LLM_MODEL=llama3.2:3b
MEMORY_EMBEDDER_MODEL=nomic-embed-text
MEMORY_EVOLUTION_THRESHOLD=50
MEMORY_CACHE_TTL_SECONDS=180

# LLM Configuration
LLM_USE_OLLAMA=true
LLM_OLLAMA_URL=http://localhost:11434
LLM_OLLAMA_MODEL=llama3:8b-instruct-q4_K_M
LLM_LM_STUDIO_URL=http://192.168.1.5:1234/v1
LLM_LM_STUDIO_MODEL=mistral-nemo-instruct-2407

# TTS Configuration
TTS_PREFERRED_VOICE=af_heart
TTS_FALLBACK_VOICES=af_alloy,af_bella
TTS_SPEED=1.05

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
JWT_SECRET=your-jwt-secret-here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/voice_assistant
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
METRICS_ENABLED=true

# Performance
MAX_CONCURRENT_SESSIONS=100
REQUEST_TIMEOUT=30
MEMORY_LIMIT=500MB
```

#### Configuration File Structure
```yaml
# config/production.yml
app:
  name: "FastRTC Voice Assistant"
  version: "1.0.0"
  environment: "production"
  debug: false

server:
  host: "0.0.0.0"
  port: 8080
  ssl_port: 8443
  ssl_cert: "/etc/ssl/certs/voice-assistant.crt"
  ssl_key: "/etc/ssl/private/voice-assistant.key"

audio:
  sample_rate: 16000
  chunk_duration: 2.0
  noise_threshold: 0.15
  minimal_silent_frame_duration_ms: 20
  max_audio_length_seconds: 30

memory:
  llm_model: "llama3.2:3b"
  embedder_model: "nomic-embed-text"
  evolution_threshold: 50
  cache_ttl_seconds: 180
  max_memory_entries: 10000

llm:
  use_ollama: true
  ollama_url: "http://localhost:11434"
  ollama_model: "llama3:8b-instruct-q4_K_M"
  lm_studio_url: "http://192.168.1.5:1234/v1"
  lm_studio_model: "mistral-nemo-instruct-2407"
  max_tokens: 2048
  temperature: 0.7

tts:
  preferred_voice: "af_heart"
  fallback_voices: ["af_alloy", "af_bella"]
  speed: 1.05
  quality: "high"

security:
  enable_auth: true
  api_key_required: true
  rate_limiting: true
  max_requests_per_minute: 60
  cors_origins: ["https://yourdomain.com"]

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/voice-assistant/app.log"
  max_size: "100MB"
  backup_count: 5
  
monitoring:
  prometheus_enabled: true
  prometheus_port: 9090
  health_check_interval: 30
  metrics_retention_days: 30
```

### SSL/TLS Configuration

#### Generate SSL Certificates
```bash
# Self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout voice-assistant.key -out voice-assistant.crt -days 365 -nodes

# Let's Encrypt (production)
sudo apt-get install certbot
sudo certbot certonly --standalone -d yourdomain.com
```

#### Configure SSL in Application
```python
# config/ssl_config.py
SSL_CONFIG = {
    "cert_file": "/etc/ssl/certs/voice-assistant.crt",
    "key_file": "/etc/ssl/private/voice-assistant.key",
    "ca_certs": "/etc/ssl/certs/ca-certificates.crt",
    "ssl_version": "TLSv1_2",
    "ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
}
```

## Deployment Strategies

### 1. Single Server Deployment

#### Basic Setup
```bash
# Install and configure
git clone https://github.com/your-org/fastrtc-voice-assistant.git
cd fastrtc-voice-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp config/production.yml.example config/production.yml
# Edit configuration as needed

# Start services
python -m src.core.main --config config/production.yml
```

#### Systemd Service
```ini
# /etc/systemd/system/voice-assistant.service
[Unit]
Description=FastRTC Voice Assistant
After=network.target

[Service]
Type=simple
User=voice-assistant
Group=voice-assistant
WorkingDirectory=/opt/voice-assistant
Environment=PATH=/opt/voice-assistant/venv/bin
ExecStart=/opt/voice-assistant/venv/bin/python -m src.core.main --config config/production.yml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant
sudo systemctl status voice-assistant
```

### 2. Load Balanced Deployment

#### Nginx Load Balancer
```nginx
# /etc/nginx/sites-available/voice-assistant
upstream voice_assistant {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/voice-assistant.crt;
    ssl_certificate_key /etc/ssl/private/voice-assistant.key;

    location / {
        proxy_pass http://voice_assistant;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://voice_assistant/health;
        access_log off;
    }
}
```

#### HAProxy Configuration
```
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend voice_assistant_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/voice-assistant.pem
    redirect scheme https if !{ ssl_fc }
    default_backend voice_assistant_backend

backend voice_assistant_backend
    balance roundrobin
    option httpchk GET /health
    server app1 127.0.0.1:8080 check
    server app2 127.0.0.1:8081 check
    server app3 127.0.0.1:8082 check
```

### 3. Containerized Deployment

#### Kubernetes Deployment
```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-assistant
  labels:
    app: voice-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-assistant
  template:
    metadata:
      labels:
        app: voice-assistant
    spec:
      containers:
      - name: voice-assistant
        image: fastrtc-voice-assistant:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: voice-assistant-service
spec:
  selector:
    app: voice-assistant
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### Helm Chart
```yaml
# helm/voice-assistant/values.yml
replicaCount: 3

image:
  repository: fastrtc-voice-assistant
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: voice-assistant.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: voice-assistant-tls
      hosts:
        - voice-assistant.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
```

## Monitoring and Observability

### Health Checks

#### Application Health Endpoints
```python
# src/monitoring/health.py
from fastapi import APIRouter
from src.core.voice_assistant import VoiceAssistant

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if all components are ready
    checks = {
        "stt_engine": await check_stt_health(),
        "tts_engine": await check_tts_health(),
        "llm_service": await check_llm_health(),
        "memory_system": await check_memory_health(),
    }
    
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    
    return {"status": "ready" if all_ready else "not_ready", "checks": checks}

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_prometheus_metrics()
```

### Prometheus Metrics

#### Custom Metrics
```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
REQUEST_COUNT = Counter('voice_assistant_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('voice_assistant_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('voice_assistant_active_sessions', 'Active voice sessions')

# Audio processing metrics
AUDIO_PROCESSING_DURATION = Histogram('voice_assistant_audio_processing_seconds', 'Audio processing time')
STT_REQUESTS = Counter('voice_assistant_stt_requests_total', 'STT requests', ['language'])
TTS_REQUESTS = Counter('voice_assistant_tts_requests_total', 'TTS requests', ['voice'])

# Memory metrics
MEMORY_USAGE = Gauge('voice_assistant_memory_usage_bytes', 'Memory usage in bytes')
MEMORY_OPERATIONS = Counter('voice_assistant_memory_operations_total', 'Memory operations', ['operation'])

# Error metrics
ERROR_COUNT = Counter('voice_assistant_errors_total', 'Total errors', ['type'])

def generate_prometheus_metrics():
    """Generate Prometheus metrics."""
    return generate_latest()
```

#### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "voice_assistant_rules.yml"

scrape_configs:
  - job_name: 'voice-assistant'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

#### Voice Assistant Dashboard
```json
{
  "dashboard": {
    "title": "FastRTC Voice Assistant",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(voice_assistant_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(voice_assistant_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Sessions",
        "type": "singlestat",
        "targets": [
          {
            "expr": "voice_assistant_active_sessions"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "voice_assistant_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      }
    ]
  }
}
```

### Logging

#### Structured Logging Configuration
```python
# src/utils/logging.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('/var/log/voice-assistant/app.log'),
        logging.StreamHandler()
    ]
)

# Set JSON formatter for file handler
file_handler = logging.FileHandler('/var/log/voice-assistant/app.log')
file_handler.setFormatter(JSONFormatter())
```

#### Log Aggregation with ELK Stack
```yaml
# docker-compose.elk.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## Security

### Authentication and Authorization

#### API Key Authentication
```python
# src/security/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    api_key = credentials.credentials
    
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

def is_valid_api_key(api_key: str) -> bool:
    """Validate API key against database or configuration."""
    valid_keys = get_valid_api_keys()
    return api_key in valid_keys
```

#### JWT Authentication
```python
# src/security/jwt_auth.py
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

def create_jwt_token(user_id: str, expires_delta: timedelta = None):
    """Create JWT token for user."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    payload = {
        "user_id": user_id,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

async def verify_jwt_token(token: str = Depends(HTTPBearer())):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Rate Limiting

#### Redis-based Rate Limiting
```python
# src/security/rate_limiting.py
import redis
from fastapi import HTTPException, Request
from datetime import datetime, timedelta

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def rate_limit(request: Request, max_requests: int = 60, window_seconds: int = 60):
    """Rate limiting middleware."""
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    
    current_requests = redis_client.get(key)
    
    if current_requests is None:
        redis_client.setex(key, window_seconds, 1)
        return
    
    if int(current_requests) >= max_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    redis_client.incr(key)
```

### Data Protection

#### Audio Data Encryption
```python
# src/security/encryption.py
from cryptography.fernet import Fernet
import base64

class AudioEncryption:
    def __init__(self, key: bytes = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
    
    def encrypt_audio(self, audio_data: bytes) -> str:
        """Encrypt audio data."""
        encrypted_data = self.cipher.encrypt(audio_data)
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_audio(self, encrypted_data: str) -> bytes:
        """Decrypt audio data."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        return self.cipher.decrypt(encrypted_bytes)
```

## Performance Optimization

### Caching Strategies

#### Redis Caching
```python
# src/caching/redis_cache.py
import redis
import json
import pickle
from typing import Any, Optional

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        data = self.redis_client.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        data = pickle.dumps(value)
        self.redis_client.setex(key, ttl, data)
    
    async def delete(self, key: str):
        """Delete key from cache."""
        self.redis_client.delete(key)
```

#### Memory Caching
```python
# src/caching/memory_cache.py
from functools import lru_cache
import asyncio
from typing import Dict, Any

class AsyncLRUCache:
    def __init__(self, maxsize: int = 128):
        self.cache: Dict[str, Any] = {}
        self.maxsize = maxsize
    
    async def get(self, key: str) -> Any:
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any):
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
```

### Database Optimization

#### Connection Pooling
```python
# src/database/pool.py
import asyncpg
from typing import Optional

class DatabasePool:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self, database_url: str, min_size: int = 10, max_size: int = 20):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60
        )
    
    async def execute(self, query: str, *args):
        """Execute query using pool."""
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch results using pool."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
```

## Backup and Recovery

### Database Backup

#### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

# Configuration
DB_NAME="voice_assistant"
DB_USER="postgres"
BACKUP_DIR="/var/backups/voice-assistant"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/voice_assistant_$DATE.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -U $DB_USER -h localhost $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

#### Backup Cron Job
```bash
# Add to crontab
0 2 * * * /opt/voice-assistant/scripts/backup.sh
```

### Disaster Recovery

#### Recovery Procedures
```bash
# 1. Stop application
sudo systemctl stop voice-assistant

# 2. Restore database
gunzip -c /var/backups/voice-assistant/voice_assistant_20231201_020000.sql.gz | psql -U postgres voice_assistant

# 3. Restore application files
tar -xzf /var/backups/voice-assistant/app_backup_20231201.tar.gz -C /opt/voice-assistant/

# 4. Start application
sudo systemctl start voice-assistant

# 5. Verify functionality
curl http://localhost:8080/health
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check application memory
sudo systemctl status voice-assistant
journalctl -u voice-assistant -f

# Solutions:
# 1. Increase memory limits
# 2. Optimize cache settings
# 3. Check for memory leaks
```

#### High CPU Usage
```bash
# Check CPU usage
top -p $(pgrep -f voice-assistant)
htop

# Profile application
python -m cProfile -o profile.stats -m src.core.main

# Solutions:
# 1. Optimize audio processing
# 2. Implement request queuing
# 3. Scale horizontally
```

#### Connection Issues
```bash
# Check network connectivity
netstat -tlnp | grep 8080
ss -tlnp | grep 8080

# Check SSL certificates
openssl x509 -in /etc/ssl/certs/voice-assistant.crt -text -noout

# Check logs
tail -f /var/log/voice-assistant/app.log
journalctl -u voice-assistant -f
```

### Log Analysis

#### Common Log Patterns
```bash
# Error analysis
grep "ERROR" /var/log/voice-assistant/app.log | tail -20

# Performance analysis
grep "duration" /var/log/voice-assistant/app.log | awk '{print $NF}' | sort -n

# User activity
grep "user_id" /var/log/voice-assistant/app.log | cut -d'"' -f4 | sort | uniq -c
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
- Monitor system health and performance
- Check error logs for issues
- Verify backup completion
- Monitor disk space usage

#### Weekly Tasks
- Review performance metrics
- Update security patches
-
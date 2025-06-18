# Task Template

## 1. Task & Context
**Task:** Implement production-grade Docker configuration for fastRTC with integrated STUN/TURN servers and WebRTC networking
**Scope:** Root-level docker-compose.yml, backend/start_clean.py WebRTC configuration, STUN/TURN server integration
**Branch:** feature/production-docker-webrtc

## 2. Quick Plan
**Approach:** Deploy self-hosted CoTURN server alongside fastRTC backend using host networking mode, configure ICE servers, and implement external IP detection for containerized WebRTC deployment
**Complexity:** 3-Complex
**Uncertainty:** 3-High
**Unknowns:** 
- Host server external IP address detection method
- Optimal port range allocation for TURN relay (default 49152-65535 may conflict)
- SSL certificate provisioning for TURNS (secure TURN over TLS)
- Authentication secret generation and persistence across container restarts
- Network interface selection on multi-homed hosts
- CPU core allocation strategy for real-time media processing
- STUN/TURN server geographic proximity optimization
**Human Input Needed:** Yes - External IP address, SSL certificate paths, port range availability, authentication method preference (static vs dynamic credentials)

## 3. Implementation
```yaml
# docker-compose.yml (project root) - Production WebRTC deployment
version: '3.8'

services:
  # CoTURN STUN/TURN server for NAT traversal
  coturn:
    image: coturn/coturn:4.6.3
    container_name: fastrtc-coturn
    restart: unless-stopped
    network_mode: host  # Required for WebRTC ICE candidate gathering
    environment:
      # Auto-detect external IP - UNCERTAINTY: May fail on cloud instances
      DETECT_EXTERNAL_IP: "yes"
      # Generate secure auth secret - UNCERTAINTY: Persistence across restarts
      STATIC_AUTH_SECRET: "${TURN_AUTH_SECRET}"
    volumes:
      - ./docker/coturn/turnserver.conf:/etc/coturn/turnserver.conf:ro
      - coturn-data:/var/lib/coturn
      - coturn-logs:/var/log
    # UNCERTAINTY: Port range conflicts on shared hosts
    # ports:
    #   - "3478:3478"          # STUN/TURN
    #   - "3478:3478/udp"      
    #   - "5349:5349"          # TURNS (TLS)
    #   - "5349:5349/udp"      
    #   - "49152-65535:49152-65535/udp"  # RTP relay range

  # FastRTC backend with WebRTC configuration
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: fastrtc-backend
    restart: unless-stopped
    network_mode: host  # Required for WebRTC media stream access
    # UNCERTAINTY: CPU allocation strategy for real-time processing
    cpuset: "0-7"  # Allocate specific cores to prevent timing jitter
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      # UNCERTAINTY: External IP detection method
      - EXTERNAL_IP=${EXTERNAL_IP}
      - STUN_SERVERS=${STUN_SERVERS:-stun:${EXTERNAL_IP}:3478}
      - TURN_SERVERS=${TURN_SERVERS:-turn:${EXTERNAL_IP}:3478}
      - TURN_AUTH_SECRET=${TURN_AUTH_SECRET}
      # WebRTC media port configuration  
      - WEBRTC_MEDIA_PORT_FROM=31001
      - WEBRTC_MEDIA_PORT_TO=40000
    volumes:
      - hf_cache:/root/.cache/huggingface/hub
      - ./backend/chroma_db:/app/backend/chroma_db
      - ./docker/webrtc:/app/config/webrtc:ro
    depends_on:
      - coturn

  # Frontend with WebRTC client configuration
  frontend:
    build:
      context: ./frontend/react-vite
      dockerfile: Dockerfile
    container_name: fastrtc-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      # UNCERTAINTY: Backend connectivity when using host networking
      - NEXT_PUBLIC_API_URL=http://${EXTERNAL_IP:-localhost}:8000
      - NEXT_PUBLIC_STUN_SERVERS=${STUN_SERVERS:-stun:${EXTERNAL_IP}:3478}
      - NEXT_PUBLIC_TURN_SERVERS=${TURN_SERVERS:-turn:${EXTERNAL_IP}:3478}
    depends_on:
      - backend

volumes:
  hf_cache:
  coturn-data:
  coturn-logs:
```

```bash
# .env.production (project root) - Production environment configuration
# UNCERTAINTY: These values must be configured per deployment environment

# External server IP - REQUIRED
EXTERNAL_IP=YOUR_SERVER_PUBLIC_IP

# TURN authentication secret - Generate once and keep secure
TURN_AUTH_SECRET=GENERATE_WITH_openssl_rand_hex_16

# ICE server configuration - UNCERTAINTY: Geographic optimization needed
STUN_SERVERS=stun:YOUR_SERVER_PUBLIC_IP:3478
TURN_SERVERS=turn:YOUR_SERVER_PUBLIC_IP:3478

# SSL certificate paths for TURNS (optional but recommended for production)
# SSL_CERT_PATH=/path/to/cert.pem
# SSL_KEY_PATH=/path/to/key.pem
```

```ini
# docker/coturn/turnserver.conf - CoTURN server configuration
# UNCERTAINTY: Optimal configuration for production scale

# Listener interface - UNCERTAINTY: Network interface detection
listening-ip=0.0.0.0
listening-port=3478
tls-listening-port=5349

# External IP - Will be set via environment variable
# external-ip=EXTERNAL_IP_PLACEHOLDER

# Relay configuration - UNCERTAINTY: Port range optimization
min-port=49152
max-port=65535

# Authentication - UNCERTAINTY: Static vs dynamic credentials
use-auth-secret
static-auth-secret=PLACEHOLDER_AUTH_SECRET

# Database for persistent sessions - UNCERTAINTY: Required for scale?
# userdb=/var/lib/coturn/turndb

# Security settings
no-multicast-peers
no-cli
no-tlsv1
no-tlsv1_1

# Logging - UNCERTAINTY: Log level for production
verbose
log-file=/var/log/turnserver.log

# Realm configuration
realm=fastrtc.local

# SSL certificates for TURNS - UNCERTAINTY: Certificate provisioning
# cert=/etc/ssl/certs/turn_server_cert.pem
# pkey=/etc/ssl/private/turn_server_pkey.pem
```

```python
# backend/src/config/webrtc_config.py (new file) - WebRTC configuration module
"""WebRTC configuration for production Docker deployment."""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class WebRTCConfig:
    """Production WebRTC configuration with STUN/TURN integration."""
    
    # External IP detection - UNCERTAINTY: Cloud instance detection
    external_ip: str = os.getenv("EXTERNAL_IP", "localhost")
    
    # ICE server configuration
    stun_servers: List[str] = None
    turn_servers: List[str] = None
    turn_auth_secret: str = os.getenv("TURN_AUTH_SECRET", "")
    
    # Media port configuration
    media_port_from: int = int(os.getenv("WEBRTC_MEDIA_PORT_FROM", "31001"))
    media_port_to: int = int(os.getenv("WEBRTC_MEDIA_PORT_TO", "40000"))
    
    def __post_init__(self):
        """Initialize ICE servers if not provided."""
        if self.stun_servers is None:
            self.stun_servers = [f"stun:{self.external_ip}:3478"]
        
        if self.turn_servers is None:
            self.turn_servers = [f"turn:{self.external_ip}:3478"]
            
        # UNCERTAINTY: Validate external IP is reachable
        if self.external_ip == "localhost":
            logger.warning("WebRTC configured with localhost - this will not work in production")
    
    def get_ice_servers(self) -> List[dict]:
        """Generate ICE servers configuration for WebRTC clients."""
        ice_servers = []
        
        # Add STUN servers
        for stun_url in self.stun_servers:
            ice_servers.append({"urls": stun_url})
        
        # Add TURN servers with authentication
        for turn_url in self.turn_servers:
            ice_servers.append({
                "urls": turn_url,
                "username": "fastrtc",
                "credential": self.turn_auth_secret
            })
        
        return ice_servers
```

```python
# backend/start_clean.py - line 19, add import
from src.config.webrtc_config import WebRTCConfig

# backend/start_clean.py - line 45, add global variable
webrtc_config: Optional[WebRTCConfig] = None

# backend/start_clean.py - line 52, update initialize_voice_assistant
async def initialize_voice_assistant():
    """Initialize the voice assistant and all its components."""
    global voice_assistant, fastrtc_bridge, callback_handler, async_env_manager, webrtc_config
    
    if voice_assistant is not None:
        logger.info("🔄 Voice assistant already initialized")
        return
    
    try:
        logger.info("🚀 Initializing voice assistant components...")
        
        # Initialize WebRTC configuration
        webrtc_config = WebRTCConfig()
        logger.info(f"🌐 WebRTC configured with external IP: {webrtc_config.external_ip}")
        
        # Load application configuration
        config = load_config()
        
        # Rest of existing initialization...
```

```python
# backend/start_clean.py - line 92, update create_fastrtc_stream
async def create_fastrtc_stream():
    """Create and configure the FastRTC stream with WebRTC networking."""
    if not all([voice_assistant, fastrtc_bridge, callback_handler, webrtc_config]):
        await initialize_voice_assistant()
    
    # Create the FastRTC stream with WebRTC configuration
    from fastrtc import SileroVadOptions
    stream = fastrtc_bridge.create_stream(
        callback_function=callback_handler.process_audio_stream,
        speech_threshold=0.05,
        server_name=webrtc_config.external_ip,  # Use external IP for ICE candidates
        server_port=8000,
        share=False,
        # UNCERTAINTY: FastRTC API may not support these parameters
        ice_servers=webrtc_config.get_ice_servers(),
        media_port_range=(webrtc_config.media_port_from, webrtc_config.media_port_to)
    )
    
    logger.info(f"🎤 FastRTC stream created with external IP: {webrtc_config.external_ip}")
    logger.info(f"🧊 ICE servers configured: {len(webrtc_config.get_ice_servers())} servers")
    return stream
```

```dockerfile
# Dockerfile.backend (project root) - Backend container with WebRTC support
FROM python:3.11-slim

# Install system dependencies for WebRTC
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY run.py .

# Create configuration directory
RUN mkdir -p /app/config/webrtc

# Expose port (when not using host networking)
EXPOSE 8000

CMD ["python", "run.py"]
```

```dockerfile
# frontend/react-vite/Dockerfile - Frontend container with WebRTC environment
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build for production with WebRTC support
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

```bash
# scripts/deploy.sh (new file) - Production deployment script
#!/bin/bash
set -e

echo "🚀 Deploying fastRTC with WebRTC support..."

# UNCERTAINTY: External IP detection method varies by cloud provider
if [ -z "$EXTERNAL_IP" ]; then
    echo "🔍 Detecting external IP..."
    EXTERNAL_IP=$(curl -s ifconfig.me || curl -s ipecho.net/plain || echo "localhost")
    echo "📍 Detected external IP: $EXTERNAL_IP"
fi

# Generate TURN authentication secret
if [ -z "$TURN_AUTH_SECRET" ]; then
    echo "🔐 Generating TURN authentication secret..."
    TURN_AUTH_SECRET=$(openssl rand -hex 16)
    echo "🔑 Generated TURN secret: $TURN_AUTH_SECRET"
fi

# Export environment variables
export EXTERNAL_IP
export TURN_AUTH_SECRET
export STUN_SERVERS="stun:${EXTERNAL_IP}:3478"
export TURN_SERVERS="turn:${EXTERNAL_IP}:3478"

# Deploy with Docker Compose
echo "📦 Starting Docker containers..."
docker compose -f docker-compose.yml --env-file .env.production up -d

echo "✅ Deployment complete!"
echo "🌐 Frontend: http://${EXTERNAL_IP}:3000"
echo "🔧 Backend API: http://${EXTERNAL_IP}:8000"
echo "🧊 STUN server: stun:${EXTERNAL_IP}:3478"
echo "🔄 TURN server: turn:${EXTERNAL_IP}:3478"
```

## 4. Check & Commit
**Changes Made:**
- Implemented production Docker configuration with CoTURN STUN/TURN server
- Added WebRTC configuration module with ICE server management
- Updated FastRTC stream creation to use external IP and ICE servers
- Created deployment script with external IP detection and secret generation
- Added CoTURN configuration with production security settings

**Commit Message:** feat: implement production Docker deployment with integrated STUN/TURN servers

**Status:** Incomplete - High uncertainty items require production environment testing and configuration

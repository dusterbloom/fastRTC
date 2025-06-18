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
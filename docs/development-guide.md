# FastRTC Development Guide

## Development Environment Options

FastRTC supports two development approaches. Choose based on your current development needs:

### Local Development (Recommended for Feature Work)

**Use When**: Building features, debugging application logic, rapid iteration

**Setup**:
```bash
# Backend
cd backend && python ../run.py

# Frontend  
cd frontend/react-vite && npm run dev
```

**Access**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- WebRTC: Works directly via localhost

**Advantages**:
- Faster iteration (no container rebuilds)
- Direct debugging access
- No Docker networking complexity
- WebRTC connects immediately on localhost

### Docker Development (Required for Network Testing)

**Use When**: Testing WebRTC across networks, integration testing, production-like environment validation

**Setup**:
```bash
# Set your external IP
export EXTERNAL_IP=$(curl -s ifconfig.me)

# Deploy with Docker
docker compose up
```

**Access**:
- Frontend: http://YOUR_EXTERNAL_IP:3000
- Backend API: http://YOUR_EXTERNAL_IP:8000
- WebRTC: Includes STUN/TURN servers for NAT traversal

**Advantages**:
- Production-like environment
- Includes STUN/TURN servers
- Tests real network conditions
- Consistent across team

## Environment Configuration

### Local Development Environment Variables

**Frontend** (`frontend/react-vite/.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Backend** (`backend/.env.development`):
```bash
HOST=0.0.0.0
PORT=8000
EXTERNAL_IP=localhost
```

### Docker Development Environment Variables

**Production** (`.env.production`):
```bash
EXTERNAL_IP=YOUR_SERVER_PUBLIC_IP
TURN_AUTH_SECRET=GENERATE_WITH_openssl_rand_hex_16
```

## Switching Between Environments

**No code changes required**. Environment detection is automatic:

```typescript
// WebRTC client automatically detects environment
const API_URL = process.env.NEXT_PUBLIC_API_URL || 
  (window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : `http://${window.location.hostname}:8000`);
```

## WebRTC Connectivity

### Local Development
- Direct peer-to-peer connection via localhost
- No STUN/TURN servers needed
- Immediate connectivity for same-machine testing

### Docker Development  
- Includes CoTURN STUN/TURN server
- Handles NAT traversal for cross-network testing
- Required for testing real-world network conditions

## Troubleshooting

### Local Development Issues
- **Port conflicts**: Ensure ports 3000 and 8000 are available
- **WebRTC fails**: Check browser permissions for microphone access

### Docker Development Issues
- **External IP detection**: Manually set `EXTERNAL_IP` if auto-detection fails
- **Port conflicts**: Ensure ports 3000, 8000, 3478, 5349, and 49152-65535 are available
- **STUN/TURN auth**: Regenerate `TURN_AUTH_SECRET` if authentication fails

## Development Workflow Recommendation

1. **Start with Local**: Use local development for feature implementation
2. **Test with Docker**: Switch to Docker for integration and network testing
3. **Deploy with Docker**: Use Docker configuration for staging and production

Both environments maintain full feature parity - choose based on your current development phase.

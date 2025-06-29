# FastRTC - Real-Time Voice Assistant

A high-performance real-time voice assistant built with WebRTC, featuring ultra-fast speech-to-text, LLM processing, multi-language TTS, and persistent memory system. Now with optimized database handling and streaming performance.

## 🚀 Quick Start

### One-Command Deployment

FastRTC now features a unified deployment script that handles all modes with a single command:

```bash
# Local development (Python + Node, no Docker)
./fastrtc.sh dev

# Docker testing (Linux/macOS)
./fastrtc.sh docker

# Production deployment
EXTERNAL_IP=your.server.ip ./fastrtc.sh prod
```

## ⚠️ Important: CUDA Dependency Management

**FastRTC uses PyTorch, CTranslate2, and Resemblyzer which can conflict if not properly managed.**

- ✅ **Use the provided setup script** to avoid dependency hell
- ✅ **Validate your environment** before development  
- ✅ **Use async pipeline** if threading pipeline fails
- ❌ **Don't manually install CUDA packages** without version pinning

See `CUDA_DEPENDENCY_HELL_PREVENTION_GUIDE.md` for details.

### Installation

#### Prerequisites
- **Windows 11 with WSL2** (recommended)
- **NVIDIA RTX 3090** or compatible GPU
- **Latest NVIDIA drivers** (525.xx+)
- **Python 3.10+**

#### Quick Setup
```bash
git clone https://github.com/your-repo/fastRTC
cd fastRTC

# Automated environment setup (prevents CUDA dependency hell)
./setup_dev_env.sh

# Validate environment
python validate_environment.py

# Start development
./fastrtc.sh dev
```

## 📋 Deployment Modes

### 🖥️ Development Mode (`./fastrtc.sh dev`)

**Best for**: Local development and testing
**Requirements**: Python 3.8+, Node.js 18+, Redis, Ollama/LM Studio

```bash
./fastrtc.sh dev
```

**What it does**:
- Starts Python backend on port 8000
- Starts React frontend on port 3000
- Uses localhost for all services
- Automatic dependency checking
- Graceful shutdown with Ctrl+C

**Access Points**:
- Frontend: http://localhost:3001
- Backend API: http://localhost:8000
- Health Check: http://localhost:8000/health
- WebRTC Endpoint: http://localhost:8000/assistant

### 🐳 Docker Mode (`./fastrtc.sh docker`)

**Best for**: Testing containerized deployment on Linux/macOS
**Requirements**: Docker, Docker Compose

```bash
./fastrtc.sh docker
```

**What it does**:
- Builds and starts all services in Docker
- Includes CoTURN STUN/TURN server for WebRTC
- Uses bridge networking (no complex host mode)
- Health checks and service dependencies
- Log streaming

**Access Points**:
- Frontend: http://localhost:3001
- Backend API: http://localhost:8000
- STUN/TURN: localhost:3478

### 🚀 Production Mode (`./fastrtc.sh prod`)

**Best for**: Production deployment with external IP
**Requirements**: Docker, Docker Compose, external IP address

```bash
# Auto-detect external IP
./fastrtc.sh prod

# Or specify external IP
EXTERNAL_IP=1.2.3.4 ./fastrtc.sh prod
```

**What it does**:
- Auto-detects external IP for WebRTC
- Generates secure TURN authentication
- Configures production security settings
- Full Docker deployment with proper networking

**Access Points**:
- Frontend: http://YOUR_IP:3001
- Backend API: http://YOUR_IP:8000
- STUN Server: stun:YOUR_IP:3478
- TURN Server: turn:YOUR_IP:3478

## 🔧 Configuration

### Environment Files

FastRTC uses a clean three-file environment strategy:

```
.env.development    # Local development settings
.env.docker         # Docker mode settings  
.env.production     # Production deployment settings
```

### Custom Configuration

Create `.env.local` for user-specific overrides (gitignored):

```bash
# .env.local - Custom settings that override defaults
OLLAMA_URL=http://custom-host:11434
GEMINI_API_KEY=your-api-key
EXTERNAL_IP=your-custom-ip
```

### Key Configuration Options

#### LLM Services
```env
# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_CONVERSATIONAL_MODEL=llama3:8b-instruct-q4_K_M

# LM Studio
LM_STUDIO_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=mistral-nemo-instruct-2407

# Gemini API
GEMINI_API_KEY=your-api-key
```

#### Speech-to-Text
```env
STT_BACKEND=faster                    # or 'huggingface'
HF_MODEL_ID=openai/whisper-large-v3
WHISPER_MODEL=base
```

#### WebRTC (Production)
```env
EXTERNAL_IP=auto                      # Auto-detect or specify
TURN_AUTH_SECRET=your-secure-secret   # Generate with: openssl rand -hex 16
```

## 🎤 STT Backend Comparison

### Faster-Whisper (Recommended)
```bash
STT_BACKEND=faster ./fastrtc.sh dev
```
- **Model loading**: ~3-5 seconds
- **VRAM usage**: ~1GB  
- **First token latency**: ~200-300ms
- **Advantages**: Faster, lower memory, INT8 quantization

### HuggingFace Transformers
```bash
STT_BACKEND=huggingface ./fastrtc.sh dev
```
- **Model loading**: ~10-15 seconds
- **VRAM usage**: ~3GB
- **First token latency**: ~500-800ms
- **Advantages**: More features, direct HuggingFace integration

## 🏗️ Architecture

### Docker Architecture (Simplified)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   CoTURN        │
│   (React/Next)  │    │   (FastAPI)     │    │   (STUN/TURN)   │
│   Port: 3001    │    │   Port: 8000    │    │   Port: 3478    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Docker Network │
                    │  (Bridge Mode)  │
                    └─────────────────┘
```

### Development Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   npm run dev   │    │   Python        │    │   Services      │
│   Port: 3000    │    │   Port: 8000    │    │   (Ollama etc.) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         localhost networking
```

## 🐛 Debug & Troubleshooting

### Debug Logging

FastRTC includes comprehensive debug logging to help diagnose issues. There are several debug modes you can enable:

#### Debug Environment Variables

Add these to your environment configuration to enable detailed logging:

```bash
# Enable all debug logging
DEBUG_TIMING=true       # Performance timing for all operations
DEBUG_TTS=true          # Text-to-Speech processing details  
DEBUG_STREAMING=true    # Step-by-step streaming pipeline logs
DEBUG_MEMORY=true       # Memory system operations
LOG_LEVEL=DEBUG         # Show all debug messages
```

#### Quick Debug Setup

**Option 1: Edit environment file directly**
```bash
# For development mode
vim .env.development

# Add these lines:
DEBUG_TIMING=true
DEBUG_TTS=true  
DEBUG_STREAMING=true
LOG_LEVEL=DEBUG
```

**Option 2: Create local override file**
```bash
# Create .env.local (takes precedence over other configs)
cat > .env.local << EOF
# Enhanced Debug Logging
DEBUG_TIMING=true
DEBUG_TTS=true
DEBUG_STREAMING=true
DEBUG_MEMORY=true
LOG_LEVEL=DEBUG
EOF
```

**Option 3: Runtime environment variables** (Development mode only)
```bash
# Export for current session
export DEBUG_STREAMING=true
export DEBUG_TTS=true
export LOG_LEVEL=DEBUG

# Start development mode
./fastrtc.sh dev
```

#### What Each Debug Mode Shows

**DEBUG_STREAMING=true:**
```
🎯 Step 1: Starting STT processing (audio size: 16000 samples)
🔍 Step 2: Quality check - Text: 'hello world' (confidence: 0.95, words: 2)
✅ Step 3: Starting LLM→TTS pipeline for: 'hello world'
🧠 Step 4: Starting LLM streaming for: 'hello world'
📝 Step 5: Sentence #1 complete: 'Hello! How can I help you?' (after 24 tokens)
✂️ Sentence split: Found ending '!' in: 'Hello! How can I help you?'
```

**DEBUG_TTS=true:**
```
🔤 TTS Text Input: 'Hello! How can I help you?' (length: 27 chars, words: 6)
🔧 TTS Options: voice='af_sarah', lang='en-us', speed=1.05
🌊 TTS Stream Input: 'Hello!' (length: 6 chars, words: 1)
📝 TTS Sentence Input: 'Hello!' (length: 6 chars, words: 1, voice: af_sarah, lang: en)
```

**DEBUG_TIMING=true:**
```
⏱️ TTS Synthesis: 0.234s
⏱️ STT Processing: 0.156s  
⏱️ LLM to TTS Pipeline: 1.234s
⏱️ Full Audio Stream Processing: 1.625s
```

**DEBUG_MEMORY=true:**
```
🧠 Memory retrieval: 0.045s (3 relevant memories found)
💾 Memory storage: 0.023s (conversation saved)
🔍 A-MEM query: 'user said hello' -> 2 matches
```

#### Debugging Specific Issues

**TTS Word-Breaking Issues:**
```bash
# Enable streaming and TTS debug to see sentence detection
DEBUG_STREAMING=true
DEBUG_TTS=true

# Look for these logs:
# ✂️ Sentence split: Found ending '.' in: 'complete sentence'
# ⚠️ Potential word break: 'incompl'  # Should not appear
```

**Audio Processing Issues:**
```bash
# Full pipeline visibility
DEBUG_STREAMING=true
DEBUG_TIMING=true

# Watch for:
# 🎯 Step 1: Starting STT processing
# ❌ Step 1: No transcript result - yielding empty
```

**Performance Issues:**
```bash
# Timing analysis
DEBUG_TIMING=true

# Look for slow operations:
# ⏱️ TTS Synthesis: 2.345s  # Should be < 1s typically
# ⚠️ Slow streaming request: 3.456s
```

#### Testing Debug Features

You can test the debug logging with the included test script:

```bash
# Activate virtual environment and run tests
cd backend
source venv/bin/activate
python test_sentence_detection_fixes.py

# Tests word-breaking fixes and sentence detection logic
# Shows comprehensive validation of streaming improvements
```

### Script Issues
```bash
# Check script permissions
chmod +x fastrtc.sh

# View help
./fastrtc.sh help

# Check dependencies
./fastrtc.sh dev  # Will check Python, Node automatically
./fastrtc.sh docker  # Will check Docker automatically
```

### WebRTC Connection Issues
- Ensure external IP is correctly detected: Check script output
- Verify ports 3478 and 49160-49200 are accessible
- Check TURN server: `docker-compose logs fastrtc-coturn`

### Development Mode Issues
```bash
# Backend issues
cd backend && python start_deferred.py  # Check direct backend

# Frontend issues  
cd frontend/react-vite && npm run dev   # Check direct frontend

# Check health
curl http://localhost:8000/health
```

### Docker Mode Issues
```bash
# View logs
docker-compose logs -f

# Check container status
docker-compose ps

# Restart services
docker-compose restart

# Rebuild from scratch
docker-compose down && docker-compose build --no-cache
```

### Model Loading Issues
```bash
# Pre-download models (optional)
huggingface-cli download openai/whisper-large-v3
huggingface-cli download Systran/faster-whisper-large-v3
```

## 🚫 WSL2 Notice

**This version no longer supports WSL2/Windows Docker complexity.** For WSL2 users:

1. **Use development mode**: `./fastrtc.sh dev` (works perfectly in WSL2)
2. **Deploy to Linux server**: Use `./fastrtc.sh prod` on a real Linux server
3. **Use GitHub Codespaces**: Full Docker support in cloud environment

This change eliminates networking issues and provides a much better experience on native Linux/macOS.

## 📁 Project Structure

```
fastRTC/
├── fastrtc.sh              # 🌟 Universal deployment script
├── .env.development        # Development configuration  
├── .env.docker            # Docker configuration
├── .env.production        # Production configuration
├── docker-compose.yml     # Simplified Docker setup
├── chroma_db/             # 🧠 Persistent user memory database
├── backend/               # Python FastAPI backend
│   ├── src/
│   │   ├── a_mem/         # Agentic memory system
│   │   ├── audio/         # STT/TTS engines
│   │   ├── integration/   # WebRTC & streaming
│   │   ├── memory/        # Memory management
│   │   └── services/      # LLM services
│   ├── requirements.txt
│   └── start_deferred.py  # 🚀 Non-blocking startup
├── frontend/react-vite/   # React/Next.js frontend
│   ├── lib/webrtc-client.ts
│   ├── components/ui/     # Voice input, language selector
│   └── package.json
├── tasks/                 # 📋 Development tasks & reports
└── docs/                  # 📚 Documentation & performance reports
```

## 🚀 Performance Tips

### Database & Memory
1. **Fixed Path Issues**: ChromaDB now uses correct paths - no more nested directories
2. **Persistent Memory**: User conversations persist across restarts automatically
3. **User Isolation**: Each user gets their own memory space for privacy

### Audio Processing
4. **Use Faster-Whisper**: `STT_BACKEND=faster` for 25% faster STT processing
5. **Streaming Optimization**: Enhanced callback handlers for real-time responses
6. **Multi-Language TTS**: Kokoro engine supports 8+ languages efficiently

### System Resources
7. **GPU Acceleration**: Ensure CUDA is available for model inference  
8. **Memory Management**: Monitor usage with `docker stats` or `nvidia-smi`
9. **Resource Allocation**: Development mode uses fewer resources than Docker
10. **Performance Monitoring**: Check `/docs/performance_analysis_report.md` for detailed metrics

## 📝 Development Workflow

### Recommended Workflow
1. **Local Development**: `./fastrtc.sh dev` for fast iteration
2. **Docker Testing**: `./fastrtc.sh docker` to test containerization
3. **Production**: `./fastrtc.sh prod` on Linux/macOS server

### Environment Progression
```bash
# Development
./fastrtc.sh dev

# Test containerization  
./fastrtc.sh docker

# Deploy to production
EXTERNAL_IP=production.server.ip ./fastrtc.sh prod
```

## 🌟 What's New

### Latest Updates (December 2024)
- **Fixed ChromaDB Path Issues**: Eliminated nested `backend/backend/chroma_db` creation
- **Persistent Memory System**: User memories now properly persist across restarts
- **Enhanced Streaming Performance**: Optimized audio callback handlers with auto-commit
- **Multi-Language TTS**: Kokoro ONNX engine with 8+ language support
- **User Isolation**: Complete memory separation between users
- **Performance Monitoring**: Comprehensive benchmarking and optimization (26% faster pipeline)

### Core Features
- **Single command deployment**: One script for all modes
- **Simplified Docker**: No more WSL2 complexity  
- **Clean environment management**: Three clear configuration files
- **Linux/macOS focused**: First-class support for native platforms
- **Raspberry Pi ready**: Standard Docker works on Pi 5
- **Better developer experience**: Faster setup, clearer errors

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. View logs: `./fastrtc.sh docker` and check output
3. Open an issue with your `fastrtc.sh` output

---

**Previous complex Docker scripts have been replaced with the unified `fastrtc.sh`. For legacy documentation, see git history.**
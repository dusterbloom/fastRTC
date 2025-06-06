#!/bin/bash
set -e  # Exit on any error

# FastRTC Quick Setup - Voice Assistant with Memory
# Works on Linux/Mac/WSL

echo "🚀 FastRTC Quick Setup - Voice Assistant with Intelligent Memory!"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found! Please install Python 3.8+ first.${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo -e "${BLUE}📍 Using Python: $(which $PYTHON_CMD)${NC}"
echo -e "${BLUE}📍 Python version: $($PYTHON_CMD --version)${NC}"

# Create virtual environment
echo -e "${BLUE}🐍 Creating virtual environment...${NC}"
$PYTHON_CMD -m venv env

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source env/bin/activate

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip

# Install FastRTC with all features
echo -e "${BLUE}⬇️  Installing FastRTC (this may take 2-3 minutes)...${NC}"
pip install "fastrtc[vad,stt,tts,stopword]"

# Install additional dependencies
echo -e "${BLUE}⬇️  Installing additional dependencies...${NC}"
pip install requests soundfile

# Install memory system dependencies
echo -e "${BLUE}🧠 Installing memory system (ChromaDB + embeddings)...${NC}"
pip install chromadb sentence-transformers

# Install Qdrant docker client 
#docker pull qdrant/qdrant

#docker run -p 6333:6333 -p 6334:6334 \
#    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
#    qdrant/qdrant
    
# Create necessary directories
echo -e "${BLUE}📁 Creating data directories...${NC}"
mkdir -p voice_memory
mkdir -p conversations

# Download embedding model (optional - will auto-download on first use)
echo -e "${YELLOW}📥 Pre-downloading embedding model (optional)...${NC}"
$PYTHON_CMD -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" 2>/dev/null || echo -e "${YELLOW}⚠️  Model will download on first use${NC}"

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}📋 What's new:${NC}"
echo "   ✨ ChromaDB for semantic memory search"
echo "   ✨ Sentence transformers for embeddings"
echo "   ✨ Persistent user profiles"
echo "   ✨ Intelligent fact extraction"
echo ""
echo -e "${GREEN}🎯 Next steps:${NC}"
echo "   1. Make sure LM Studio is running at http://192.168.1.5:1234"
echo "   2. Run: python voice_assistant.py"
echo "   3. Open browser when prompted"
echo "   4. Start talking - the assistant will remember you!"
echo ""
echo -e "${BLUE}💡 Memory commands:${NC}"
echo '   - "My name is [name]" - introduces yourself'
echo '   - "Remember that..." - stores explicit facts'
echo '   - "What do you remember about me?" - recalls information'
echo '   - "Forget everything" - clears all memory'
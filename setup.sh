#!/bin/bash
set -e  # Exit on any error

# FastRTC Quick Setup - Get voice assistant running in 5 minutes
# Works on Linux/Mac/WSL

echo "🚀 FastRTC Quick Setup - Voice Assistant in 5 minutes!"
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

# Install additional dependencies for better compatibility
echo -e "${BLUE}⬇️  Installing additional dependencies...${NC}"
pip install requests soundfile

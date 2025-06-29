#!/bin/bash
# FastRTC Development Environment Setup
# For Windows 11 WSL2 + RTX 3090
# Prevents CUDA dependency hell

set -e

echo "🚀 FastRTC Development Environment Setup"
echo "Target: Windows 11 WSL2 + RTX 3090"
echo "=" * 50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
log_info "Checking prerequisites..."

# Check if we're in WSL2
if ! grep -q "microsoft" /proc/version 2>/dev/null; then
    log_warn "Not running in WSL2. This script is optimized for WSL2."
fi

# Check GPU access
if ! nvidia-smi > /dev/null 2>&1; then
    log_error "NVIDIA GPU not detected. Please check:"
    echo "  1. NVIDIA drivers installed on Windows host"
    echo "  2. WSL2 GPU support enabled"
    echo "  3. Run 'nvidia-smi' to verify"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)
log_success "GPU detected: $GPU_NAME"
log_info "CUDA Driver Version: $CUDA_VERSION"

# Check Python version
if ! python3.10 --version > /dev/null 2>&1; then
    log_error "Python 3.10 not found. Installing..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
fi

PYTHON_VERSION=$(python3.10 --version)
log_success "Python: $PYTHON_VERSION"

# Determine setup method
if command -v conda &> /dev/null; then
    USE_CONDA=true
    log_info "Conda detected - using conda environment"
else
    USE_CONDA=false
    log_info "Conda not found - using Python venv"
fi

# Setup environment
if [ "$USE_CONDA" = true ]; then
    log_info "Creating conda environment 'fastrtc'..."
    
    # Check if environment exists
    if conda env list | grep -q "fastrtc"; then
        log_warn "Environment 'fastrtc' already exists. Removing..."
        conda env remove -n fastrtc -y
    fi
    
    # Create environment with specific versions
    conda create -n fastrtc python=3.10 -y
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate fastrtc
    
    # Install PyTorch with CUDA 11.8 (compatible with RTX 3090)
    log_info "Installing PyTorch with CUDA 11.8..."
    conda install pytorch=2.1.1 torchaudio=2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install other packages via pip
    log_info "Installing additional packages..."
    pip install ctranslate2==4.2.0
    pip install faster-whisper==1.1.0
    pip install resemblyzer==0.1.4
    
    # Install FastRTC dependencies
    cd backend
    pip install -r requirements.txt
    pip install --no-deps fastrtc==0.0.28
    cd ..
    
    log_success "Conda environment 'fastrtc' created successfully"
    log_info "Activate with: conda activate fastrtc"
    
else
    log_info "Creating Python virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d "backend/venv" ]; then
        log_warn "Removing existing virtual environment..."
        rm -rf backend/venv
    fi
    
    # Create new venv
    cd backend
    python3.10 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with specific CUDA version FIRST
    log_info "Installing PyTorch with CUDA 11.8..."
    pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 torchvision==0.16.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
    
    # Install CTranslate2 (compatible version)
    log_info "Installing CTranslate2..."
    pip install ctranslate2==4.2.0
    
    # Install FasterWhisper
    log_info "Installing FasterWhisper..."
    pip install faster-whisper==1.1.0
    
    # Install Resemblyzer
    log_info "Installing Resemblyzer..."
    pip install resemblyzer==0.1.4
    
    # Install other requirements
    log_info "Installing other dependencies..."
    pip install fastapi uvicorn[standard]
    pip install numpy==1.26.4 scipy==1.13.0
    pip install librosa==0.11.0 soundfile==0.12.1
    pip install chromadb==0.4.22 rank_bm25==0.2.2
    pip install requests aiohttp transformers
    pip install huggingface_hub[hf_transfer]==0.25.0
    pip install ollama redis nltk
    pip install pytest pytest-asyncio
    
    # Install FastRTC
    pip install --no-deps fastrtc==0.0.28
    
    cd ..
    
    log_success "Python virtual environment created successfully"
    log_info "Activate with: source backend/venv/bin/activate"
fi

# Verify installation
log_info "Verifying installation..."

if [ "$USE_CONDA" = true ]; then
    eval "$(conda shell.bash hook)"
    conda activate fastrtc
else
    source backend/venv/bin/activate
fi

# Run verification script
python3 -c "
import sys
try:
    import torch
    import ctranslate2
    from resemblyzer import VoiceEncoder
    
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
    print(f'✅ CTranslate2 version: {ctranslate2.__version__}')
    print(f'✅ CTranslate2 CUDA devices: {ctranslate2.get_cuda_device_count()}')
    print(f'✅ Resemblyzer import successful')
    
    # Quick test
    if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0:
        print('🎉 All CUDA components working!')
    else:
        print('⚠️  CUDA components may have issues')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Verification failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    log_success "Environment setup completed successfully!"
    echo ""
    echo "🎯 Next steps:"
    if [ "$USE_CONDA" = true ]; then
        echo "  1. conda activate fastrtc"
    else
        echo "  1. source backend/venv/bin/activate"
    fi
    echo "  2. python validate_environment.py"
    echo "  3. ./fastrtc.sh dev"
    echo ""
    echo "💡 If you encounter issues, run: python validate_environment.py"
else
    log_error "Environment setup failed during verification"
    exit 1
fi
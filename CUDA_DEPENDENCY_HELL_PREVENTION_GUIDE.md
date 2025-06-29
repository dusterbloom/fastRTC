# CUDA Dependency Hell Prevention Guide
## How to Save Your Team's Sanity

**Target Environment:** Windows 11 + WSL2 + RTX 3090  
**Problem:** CUDA library conflicts causing silent failures  
**Solution:** Bulletproof development environment setup  

---

## 🎯 **The Golden Rules**

### **Rule #1: Pin EVERYTHING**
Never use flexible version ranges for CUDA-related packages:
```bash
# ❌ BAD - Leads to dependency hell
torch>=2.1.0
ctranslate2>=4.0.0

# ✅ GOOD - Explicit and reproducible
torch==2.1.1+cu118
ctranslate2==4.2.0
```

### **Rule #2: Match CUDA Versions**
All packages must use the same CUDA version:
```bash
# ✅ All using CUDA 11.8
torch==2.1.1+cu118
torchaudio==2.1.1+cu118
torchvision==0.16.1+cu118
ctranslate2==4.2.0  # Built with CUDA 11.8
```

### **Rule #3: Document Your Environment**
Create a detailed environment specification that works.

---

## 🛠️ **Bulletproof Setup for RTX 3090 + WSL2**

### **Step 1: System Prerequisites**

#### **Windows 11 Host:**
```powershell
# Install latest NVIDIA driver (525.xx or newer)
# Download from: https://www.nvidia.com/drivers/
# This provides CUDA 12.2+ support

# Verify installation
nvidia-smi.exe
```

#### **WSL2 Ubuntu:**
```bash
# DO NOT install CUDA toolkit in WSL2!
# WSL2 uses Windows driver directly

# Verify GPU access
nvidia-smi
# Should show: CUDA Version: 12.2 (or similar)

# Install basic dev tools
sudo apt update
sudo apt install python3.10-venv python3.10-dev build-essential
```

### **Step 2: Create Locked Environment**

#### **Create `environment.yml` (Conda approach - RECOMMENDED):**
```yaml
name: fastrtc
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1.1
  - torchaudio=2.1.1
  - pytorch-cuda=11.8
  - numpy=1.26.4
  - pip
  - pip:
    - ctranslate2==4.2.0
    - faster-whisper==1.1.0
    - resemblyzer==0.1.4
    - fastapi==0.104.1
    - uvicorn[standard]==0.24.0
    - chromadb==0.4.22
    - rank_bm25==0.2.2
    - librosa==0.11.0
    - scipy==1.13.0
    - soxr==0.5.0.post1
    - soundfile==0.12.1
    - requests==2.31.0
    - aiohttp==3.9.1
    - transformers==4.36.2
    - huggingface_hub[hf_transfer]==0.25.0
    - ollama==0.1.0
    - redis==4.5.0
    - nltk==3.8.1
    - pytest==7.4.3
    - pytest-asyncio==0.21.1
```

#### **Alternative: Locked `requirements.txt`:**
```bash
# requirements-locked.txt
# Generated on: 2025-06-28
# Platform: Windows 11 WSL2 + RTX 3090
# CUDA: 11.8 (all packages)

# Core ML stack - CUDA 11.8
torch==2.1.1+cu118
torchaudio==2.1.1+cu118
torchvision==0.16.1+cu118

# STT/TTS stack
ctranslate2==4.2.0
faster-whisper==1.1.0
resemblyzer==0.1.4

# Exact versions of everything else
numpy==1.26.4
scipy==1.13.0
librosa==0.11.0
soundfile==0.12.1
# ... (all other packages with exact versions)
```

### **Step 3: Environment Setup Script**

#### **Create `setup_dev_env.sh`:**
```bash
#!/bin/bash
# FastRTC Development Environment Setup
# For Windows 11 WSL2 + RTX 3090

set -e

echo "🚀 Setting up FastRTC development environment..."

# Check prerequisites
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA GPU not detected. Check WSL2 GPU support."
    exit 1
fi

echo "✅ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Method 1: Conda (RECOMMENDED)
if command -v conda &> /dev/null; then
    echo "📦 Using Conda environment..."
    conda env create -f environment.yml
    conda activate fastrtc
    echo "✅ Conda environment 'fastrtc' created"
    
# Method 2: Python venv
else
    echo "🐍 Using Python venv..."
    python3.10 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with specific CUDA version FIRST
    pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 torchvision==0.16.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
    
    # Install other packages
    pip install -r requirements-locked.txt
    
    echo "✅ Python venv created and packages installed"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch
import ctranslate2
from resemblyzer import VoiceEncoder

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CTranslate2 version: {ctranslate2.__version__}')
print(f'CUDA devices: {ctranslate2.get_cuda_device_count()}')
print('Resemblyzer import: ✅')
"

echo "✅ Environment setup complete!"
echo "💡 Activate with: conda activate fastrtc (or source venv/bin/activate)"
```

### **Step 4: Validation Script**

#### **Create `validate_environment.py`:**
```python
#!/usr/bin/env python3
"""
Environment Validation Script
Ensures CUDA dependencies are compatible before development
"""

import sys
import subprocess
from pathlib import Path

def check_gpu():
    """Check GPU availability"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected and accessible")
            return True
        else:
            print("❌ GPU not accessible")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def check_cuda_compatibility():
    """Check CUDA library compatibility"""
    try:
        import torch
        import ctranslate2
        
        print(f"✅ PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
        print(f"✅ CTranslate2 {ctranslate2.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA devices: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.get_device_name()}")
        else:
            print("❌ CUDA not available to PyTorch")
            return False
            
        # Test CTranslate2 CUDA
        ct2_devices = ctranslate2.get_cuda_device_count()
        if ct2_devices > 0:
            print(f"✅ CTranslate2 CUDA devices: {ct2_devices}")
        else:
            print("❌ CTranslate2 cannot access CUDA")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_resemblyzer():
    """Test Resemblyzer initialization"""
    try:
        from resemblyzer import VoiceEncoder
        import time
        
        start = time.time()
        encoder = VoiceEncoder()
        init_time = time.time() - start
        
        print(f"✅ Resemblyzer initialized in {init_time:.2f}s")
        print(f"✅ Encoder device: {encoder.device}")
        return True
        
    except Exception as e:
        print(f"❌ Resemblyzer failed: {e}")
        return False

def test_pipeline_components():
    """Test FastRTC pipeline components"""
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        
        from src.audio.engines.stt.faster_whisper_stt import FasterWhisperSTT
        from src.core.voice_assistant import VoiceAssistant
        from src.config.settings import load_config
        
        print("✅ FastRTC imports successful")
        
        # Test STT engine
        stt = FasterWhisperSTT()
        print("✅ STT engine initialized")
        
        # Test Voice Assistant
        config = load_config()
        va = VoiceAssistant(config=config)
        print("✅ Voice Assistant initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🔍 FastRTC Environment Validation")
    print("=" * 40)
    
    checks = [
        ("GPU Access", check_gpu),
        ("CUDA Compatibility", check_cuda_compatibility),
        ("Resemblyzer", test_resemblyzer),
        ("Pipeline Components", test_pipeline_components),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n🧪 Testing: {name}")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 40)
    print("📋 Validation Summary:")
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Environment is ready for development!")
        return 0
    else:
        print("\n💥 Environment has issues. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 📋 **Team Onboarding Checklist**

### **For New Developers:**

```bash
# 1. Clone repository
git clone <repo-url>
cd fastrtc

# 2. Check system prerequisites
nvidia-smi  # Should show RTX 3090

# 3. Run setup script
chmod +x setup_dev_env.sh
./setup_dev_env.sh

# 4. Validate environment
python validate_environment.py

# 5. Test application
./fastrtc.sh dev  # Should work without issues
```

### **Required Documentation in README.md:**

```markdown
## Development Environment Setup

### Prerequisites
- Windows 11 with WSL2
- NVIDIA RTX 3090 (or compatible GPU)
- Latest NVIDIA drivers (525.xx+)

### Quick Setup
```bash
./setup_dev_env.sh
python validate_environment.py
```

### Troubleshooting
If validation fails:
1. Check `nvidia-smi` works
2. Ensure WSL2 GPU support is enabled
3. Run `validate_environment.py` for detailed diagnostics
4. See `CUDA_DEPENDENCY_HELL_PREVENTION_GUIDE.md`
```

---

## 🚨 **CI/CD Integration**

### **GitHub Actions Workflow:**

```yaml
name: Environment Validation
on: [push, pull_request]

jobs:
  validate-environment:
    runs-on: self-hosted  # Use your RTX 3090 machine
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Environment
        run: ./setup_dev_env.sh
        
      - name: Validate Environment
        run: python validate_environment.py
        
      - name: Test Pipeline
        run: |
          source venv/bin/activate  # or conda activate fastrtc
          python backend/test_pipeline_debug.py
```

---

## 🔒 **Dependency Lock Strategy**

### **1. Generate Lock Files:**
```bash
# After successful setup, generate exact versions
pip freeze > requirements-exact.txt
conda env export > environment-exact.yml

# Commit these to repository
git add requirements-exact.txt environment-exact.yml
git commit -m "Lock working dependency versions"
```

### **2. Version Update Process:**
```bash
# Create branch for dependency updates
git checkout -b update-dependencies

# Test new versions in isolated environment
python -m venv test_env
source test_env/bin/activate

# Install new versions
pip install torch==2.2.0+cu118  # New version

# Run full validation
python validate_environment.py
python backend/test_pipeline_debug.py

# If all tests pass, update lock files
pip freeze > requirements-exact.txt
```

### **3. Automated Dependency Monitoring:**
```python
# Add to CI/CD: dependency_monitor.py
import subprocess
import json

def check_for_updates():
    """Check for available updates and test compatibility"""
    result = subprocess.run(['pip', 'list', '--outdated', '--format=json'], 
                          capture_output=True, text=True)
    outdated = json.loads(result.stdout)
    
    critical_packages = ['torch', 'ctranslate2', 'resemblyzer']
    
    for pkg in outdated:
        if pkg['name'] in critical_packages:
            print(f"⚠️  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            # Trigger automated testing with new version
```

---

## 🎯 **Key Takeaways for Your Team**

### **DO:**
- ✅ Use exact version pinning for all CUDA-related packages
- ✅ Test environment setup on clean machines
- ✅ Document working configurations
- ✅ Use validation scripts before development
- ✅ Prefer Conda over pip for ML environments

### **DON'T:**
- ❌ Use flexible version ranges (`>=`, `~=`) for ML packages
- ❌ Mix CUDA versions across packages
- ❌ Install CUDA toolkit in WSL2 (use Windows driver)
- ❌ Skip environment validation
- ❌ Assume "it works on my machine" means it works everywhere

### **Emergency Procedures:**
```bash
# If environment breaks:
1. Run: python validate_environment.py
2. Check: git log --oneline (what changed?)
3. Revert: git checkout HEAD~1 -- requirements-exact.txt
4. Rebuild: ./setup_dev_env.sh
5. Validate: python validate_environment.py
```

This approach will save your team countless hours of debugging CUDA dependency hell! 🎉
#!/usr/bin/env python3
"""
Dependency Analysis Script
=========================

Analyzes current dependency state and identifies conflicts that cause
the threading pipeline to hang.
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def get_installed_packages():
    """Get currently installed packages"""
    stdout, stderr, code = run_command("pip list --format=json")
    if code == 0:
        return json.loads(stdout)
    return []

def check_pytorch_compatibility():
    """Check PyTorch and related packages compatibility"""
    print("🔍 PyTorch Ecosystem Analysis")
    print("=" * 40)
    
    packages = get_installed_packages()
    torch_packages = [p for p in packages if 'torch' in p['name'].lower()]
    
    for pkg in torch_packages:
        print(f"  {pkg['name']}: {pkg['version']}")
    
    # Check CUDA availability
    try:
        import torch
        print(f"\n🔧 PyTorch CUDA Status:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")

def check_ctranslate2_compatibility():
    """Check CTranslate2 compatibility"""
    print("\n🔍 CTranslate2 Analysis")
    print("=" * 40)
    
    try:
        import ctranslate2
        print(f"  CTranslate2 version: {ctranslate2.__version__}")
        print(f"  CUDA support: {ctranslate2.get_cuda_device_count() > 0}")
        print(f"  CUDA devices: {ctranslate2.get_cuda_device_count()}")
    except Exception as e:
        print(f"❌ CTranslate2 import failed: {e}")

def check_resemblyzer_compatibility():
    """Check Resemblyzer compatibility"""
    print("\n🔍 Resemblyzer Analysis")
    print("=" * 40)
    
    try:
        from resemblyzer import VoiceEncoder
        print("  Resemblyzer import: ✅ Success")
        
        # Test encoder initialization
        import time
        start_time = time.time()
        encoder = VoiceEncoder()
        init_time = time.time() - start_time
        print(f"  Encoder init time: {init_time:.2f}s")
        print(f"  Encoder device: {encoder.device}")
        
    except Exception as e:
        print(f"❌ Resemblyzer failed: {e}")

def check_faster_whisper_compatibility():
    """Check FasterWhisper compatibility"""
    print("\n🔍 FasterWhisper Analysis")
    print("=" * 40)
    
    try:
        from faster_whisper import WhisperModel
        print("  FasterWhisper import: ✅ Success")
        
        # Test model initialization
        try:
            model = WhisperModel("tiny", device="cpu")  # Use tiny model for speed
            print("  Model init (CPU): ✅ Success")
        except Exception as e:
            print(f"  Model init (CPU): ❌ Failed - {e}")
            
        try:
            model = WhisperModel("tiny", device="cuda")
            print("  Model init (CUDA): ✅ Success")
        except Exception as e:
            print(f"  Model init (CUDA): ❌ Failed - {e}")
            
    except Exception as e:
        print(f"❌ FasterWhisper failed: {e}")

def analyze_version_conflicts():
    """Analyze potential version conflicts"""
    print("\n🔍 Version Conflict Analysis")
    print("=" * 40)
    
    packages = get_installed_packages()
    pkg_dict = {p['name'].lower(): p['version'] for p in packages}
    
    # Check critical packages
    critical_packages = [
        'torch', 'torchaudio', 'torchvision',
        'ctranslate2', 'faster-whisper', 'resemblyzer',
        'numpy', 'transformers'
    ]
    
    print("📦 Critical Package Versions:")
    for pkg in critical_packages:
        version = pkg_dict.get(pkg, "NOT INSTALLED")
        status = "✅" if version != "NOT INSTALLED" else "❌"
        print(f"  {status} {pkg}: {version}")
    
    # Check for known conflicts
    print("\n⚠️  Known Conflict Checks:")
    
    # PyTorch 2.7+ with older CTranslate2
    torch_version = pkg_dict.get('torch', '0.0.0')
    ctranslate_version = pkg_dict.get('ctranslate2', '0.0.0')
    
    if torch_version.startswith('2.7') and ctranslate_version.startswith('4.'):
        print("  🚨 PyTorch 2.7+ with CTranslate2 4.x - POTENTIAL CONFLICT")
    else:
        print("  ✅ PyTorch/CTranslate2 versions seem compatible")
    
    # Numpy 2.0+ conflicts
    numpy_version = pkg_dict.get('numpy', '0.0.0')
    if numpy_version.startswith('2.'):
        print("  🚨 NumPy 2.0+ detected - may cause issues with older packages")
    else:
        print("  ✅ NumPy version < 2.0")

def test_threading_components():
    """Test components that are used in threading pipeline"""
    print("\n🔍 Threading Pipeline Component Test")
    print("=" * 40)
    
    # Test STT engine
    try:
        sys.path.insert(0, str(Path(__file__).parent.resolve()))
        from src.audio.engines.stt.faster_whisper_stt import FasterWhisperSTT
        
        print("  STT Engine import: ✅ Success")
        
        stt = FasterWhisperSTT()
        print("  STT Engine init: ✅ Success")
        
        # Test async transcription
        import asyncio
        import numpy as np
        
        async def test_stt():
            dummy_audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
            result = await stt.transcribe(dummy_audio)
            return result
        
        result = asyncio.run(test_stt())
        print(f"  STT Transcription: ✅ Success - '{result.text}'")
        
    except Exception as e:
        print(f"  ❌ STT Engine failed: {e}")
    
    # Test Voice Assistant
    try:
        from src.core.voice_assistant import VoiceAssistant
        from src.config.settings import load_config
        
        print("  Voice Assistant import: ✅ Success")
        
        config = load_config()
        va = VoiceAssistant(config=config)
        print("  Voice Assistant init: ✅ Success")
        
    except Exception as e:
        print(f"  ❌ Voice Assistant failed: {e}")

def generate_recommendations():
    """Generate specific recommendations based on analysis"""
    print("\n🎯 Recommendations")
    print("=" * 40)
    
    packages = get_installed_packages()
    pkg_dict = {p['name'].lower(): p['version'] for p in packages}
    
    torch_version = pkg_dict.get('torch', '0.0.0')
    
    if torch_version.startswith('2.7'):
        print("🔧 IMMEDIATE FIXES:")
        print("  1. Downgrade PyTorch to working version:")
        print("     pip install torch==2.1.1 torchaudio==2.1.1 --force-reinstall")
        print()
        print("  2. Or use async pipeline (bypass threading issues):")
        print("     ./fastrtc.sh dev  # Remove --threading flag")
        print()
        print("  3. Or disable voice auth temporarily:")
        print("     Edit src/audio/user_identification.py line 51")
        print("     Set: self.enable_voice_auth = False")
    
    print("\n🔬 TESTING STRATEGY:")
    print("  1. Test with working requirements.txt from 2 commits ago")
    print("  2. Add Resemblyzer incrementally with version constraints")
    print("  3. Test each change with threading pipeline")
    print("  4. Document working version combinations")
    
    print("\n📋 DEPENDENCY MANAGEMENT:")
    print("  1. Pin all critical package versions in requirements.txt")
    print("  2. Use virtual environments for testing")
    print("  3. Add dependency compatibility tests to CI/CD")
    print("  4. Consider Docker for production deployment")

def main():
    """Run complete dependency analysis"""
    print("🔍 FastRTC Dependency Analysis")
    print("=" * 50)
    print("Analyzing current dependency state and conflicts...")
    print()
    
    check_pytorch_compatibility()
    check_ctranslate2_compatibility()
    check_resemblyzer_compatibility()
    check_faster_whisper_compatibility()
    analyze_version_conflicts()
    test_threading_components()
    generate_recommendations()
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")
    print("📄 See VOICE_AUTH_PIPELINE_HANG_REPORT.md for detailed findings")

if __name__ == "__main__":
    main()
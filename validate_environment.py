#!/usr/bin/env python3
"""
FastRTC Environment Validation Script
====================================

Ensures CUDA dependencies are compatible before development.
Specifically designed for Windows 11 WSL2 + RTX 3090 setup.

Usage:
    python validate_environment.py
"""

import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n🔍 {title}")
    print("=" * (len(title) + 4))

def check_system_prerequisites():
    """Check system-level prerequisites"""
    print_header("System Prerequisites")
    
    checks_passed = 0
    total_checks = 4
    
    # Check if running in WSL2
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
        if 'microsoft' in version_info.lower():
            print("✅ Running in WSL2")
            checks_passed += 1
        else:
            print("⚠️  Not running in WSL2 (may still work)")
            checks_passed += 1
    except:
        print("❌ Cannot determine if running in WSL2")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi accessible")
            checks_passed += 1
            
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GeForce' in line or 'Tesla' in line:
                    gpu_info = line.strip()
                    print(f"   GPU: {gpu_info.split('|')[1].strip() if '|' in gpu_info else 'Unknown'}")
                    break
        else:
            print("❌ nvidia-smi not working")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 10:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks_passed += 1
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (need 3.10+)")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        checks_passed += 1
    else:
        print("⚠️  Not in virtual environment (recommended)")
        checks_passed += 1  # Don't fail for this
    
    return checks_passed == total_checks

def check_cuda_stack():
    """Check CUDA library stack compatibility"""
    print_header("CUDA Library Stack")
    
    checks_passed = 0
    total_checks = 3
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   CUDA available: Yes")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.get_device_name(0)}")
            checks_passed += 1
        else:
            print("❌ CUDA not available to PyTorch")
            
    except ImportError:
        print("❌ PyTorch not installed")
    
    # Check CTranslate2
    try:
        import ctranslate2
        print(f"✅ CTranslate2 {ctranslate2.__version__}")
        
        cuda_devices = ctranslate2.get_cuda_device_count()
        if cuda_devices > 0:
            print(f"   CUDA devices: {cuda_devices}")
            checks_passed += 1
        else:
            print("❌ CTranslate2 cannot access CUDA")
            
    except ImportError:
        print("❌ CTranslate2 not installed")
    
    # Check version compatibility
    try:
        torch_cuda = torch.version.cuda
        if torch_cuda and torch_cuda.startswith('11.8'):
            print("✅ PyTorch using CUDA 11.8 (recommended)")
            checks_passed += 1
        elif torch_cuda and torch_cuda.startswith('12.'):
            print("⚠️  PyTorch using CUDA 12.x (may cause issues)")
            checks_passed += 1  # Don't fail, but warn
        else:
            print(f"❌ PyTorch CUDA version {torch_cuda} may be incompatible")
    except:
        print("❌ Cannot determine PyTorch CUDA version")
    
    return checks_passed >= 2  # Allow some flexibility

def check_ml_packages():
    """Check ML package compatibility"""
    print_header("ML Package Compatibility")
    
    checks_passed = 0
    total_checks = 4
    
    # Check Resemblyzer
    try:
        print("Testing Resemblyzer initialization...")
        start_time = time.time()
        
        from resemblyzer import VoiceEncoder
        encoder = VoiceEncoder()
        
        init_time = time.time() - start_time
        print(f"✅ Resemblyzer initialized in {init_time:.2f}s")
        print(f"   Device: {encoder.device}")
        
        if init_time < 30:
            print("   Performance: Good")
        else:
            print("   Performance: Slow (may indicate issues)")
            
        checks_passed += 1
        
    except Exception as e:
        print(f"❌ Resemblyzer failed: {e}")
    
    # Check FasterWhisper
    try:
        from faster_whisper import WhisperModel
        print("✅ FasterWhisper import successful")
        
        # Test tiny model initialization
        try:
            model = WhisperModel("tiny", device="cpu")
            print("   CPU mode: Working")
            checks_passed += 1
        except Exception as e:
            print(f"   CPU mode: Failed - {e}")
        
        try:
            model = WhisperModel("tiny", device="cuda")
            print("   CUDA mode: Working")
            checks_passed += 1
        except Exception as e:
            print(f"   CUDA mode: Failed - {e}")
            
    except ImportError:
        print("❌ FasterWhisper not installed")
    
    # Check NumPy compatibility
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"✅ NumPy {numpy_version}")
        
        if numpy_version.startswith('1.'):
            print("   Version: Compatible")
            checks_passed += 1
        else:
            print("   Version: May cause issues with some packages")
            
    except ImportError:
        print("❌ NumPy not installed")
    
    return checks_passed >= 3

def check_fastrtc_components():
    """Check FastRTC specific components"""
    print_header("FastRTC Components")
    
    checks_passed = 0
    total_checks = 4
    
    # Add backend to path
    backend_path = Path(__file__).parent / "backend"
    if backend_path.exists():
        sys.path.insert(0, str(backend_path))
    else:
        print("❌ Backend directory not found")
        return False
    
    # Check STT engine
    try:
        from src.audio.engines.stt.faster_whisper_stt import FasterWhisperSTT
        stt = FasterWhisperSTT()
        print("✅ STT engine initialized")
        checks_passed += 1
    except Exception as e:
        print(f"❌ STT engine failed: {e}")
    
    # Check TTS engine
    try:
        from src.audio.engines.tts.kokoro_tts import KokoroTTS
        print("✅ TTS engine import successful")
        checks_passed += 1
    except Exception as e:
        print(f"❌ TTS engine failed: {e}")
    
    # Check Voice Assistant
    try:
        from src.core.voice_assistant import VoiceAssistant
        from src.config.settings import load_config
        
        config = load_config()
        va = VoiceAssistant(config=config)
        print("✅ Voice Assistant initialized")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Voice Assistant failed: {e}")
    
    # Check User Identification
    try:
        from src.audio.user_identification import SpokenUserIdentifier
        user_id = SpokenUserIdentifier(enable_voice_auth=True)
        print(f"✅ User Identification (voice auth: {user_id.enable_voice_auth})")
        checks_passed += 1
    except Exception as e:
        print(f"❌ User Identification failed: {e}")
    
    return checks_passed >= 3

def test_pipeline_integration():
    """Test actual pipeline integration"""
    print_header("Pipeline Integration Test")
    
    try:
        # Add backend to path
        backend_path = Path(__file__).parent / "backend"
        sys.path.insert(0, str(backend_path))
        
        from src.integration.unified_callback_handler import UnifiedCallbackHandler
        from src.core.voice_assistant import VoiceAssistant
        from src.config.settings import load_config
        from src.utils.async_utils import AsyncEnvironmentManager
        
        print("Creating voice assistant...")
        config = load_config()
        voice_assistant = VoiceAssistant(config=config)
        
        print("Setting up async environment...")
        async_env_manager = AsyncEnvironmentManager()
        success = async_env_manager.setup_async_environment(voice_assistant)
        
        if not success:
            print("❌ Async environment setup failed")
            return False
        
        print("Creating unified callback handler...")
        handler = UnifiedCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=voice_assistant.tts_engine,
            voice_mapper=voice_assistant.voice_mapper,
            event_loop=async_env_manager.get_event_loop()
        )
        
        stats = handler.get_handler_stats()
        print(f"✅ Handler type: {stats.get('handler_type', 'unknown')}")
        print(f"   Threading enabled: {stats.get('threading_enabled', False)}")
        print(f"   Fallback enabled: {stats.get('fallback_enabled', False)}")
        
        # Clean up
        handler.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def generate_recommendations(results):
    """Generate specific recommendations based on test results"""
    print_header("Recommendations")
    
    system_ok, cuda_ok, ml_ok, fastrtc_ok, pipeline_ok = results
    
    if all(results):
        print("🎉 All tests passed! Your environment is ready for development.")
        print("\n💡 Next steps:")
        print("   1. ./fastrtc.sh dev")
        print("   2. Test voice authentication features")
        return True
    
    print("🔧 Issues detected. Here's how to fix them:")
    
    if not system_ok:
        print("\n📋 System Issues:")
        print("   • Ensure NVIDIA drivers are installed on Windows host")
        print("   • Verify WSL2 GPU support is enabled")
        print("   • Run 'nvidia-smi' to test GPU access")
    
    if not cuda_ok:
        print("\n🔥 CUDA Issues:")
        print("   • Reinstall PyTorch with CUDA 11.8:")
        print("     pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
        print("   • Reinstall CTranslate2:")
        print("     pip install ctranslate2==4.2.0 --force-reinstall")
    
    if not ml_ok:
        print("\n🤖 ML Package Issues:")
        print("   • Resemblyzer slow? Try CPU mode temporarily")
        print("   • FasterWhisper issues? Check model download permissions")
        print("   • NumPy 2.0+ detected? Downgrade to 1.26.4")
    
    if not fastrtc_ok:
        print("\n🚀 FastRTC Issues:")
        print("   • Missing dependencies? Run: pip install -r backend/requirements.txt")
        print("   • Import errors? Check Python path and virtual environment")
    
    if not pipeline_ok:
        print("\n⚙️  Pipeline Issues:")
        print("   • Use async pipeline: ./fastrtc.sh dev (remove --threading)")
        print("   • Check dependency conflicts with: python backend/analyze_dependencies.py")
    
    print("\n🆘 Emergency fix:")
    print("   ./fastrtc.sh dev  # Use async pipeline to bypass threading issues")
    
    return False

def main():
    """Run complete environment validation"""
    print("🔍 FastRTC Environment Validation")
    print("Target: Windows 11 WSL2 + RTX 3090")
    print("=" * 50)
    
    # Run all checks
    results = [
        check_system_prerequisites(),
        check_cuda_stack(),
        check_ml_packages(),
        check_fastrtc_components(),
        test_pipeline_integration(),
    ]
    
    # Generate summary
    print_header("Validation Summary")
    
    test_names = [
        "System Prerequisites",
        "CUDA Library Stack", 
        "ML Package Compatibility",
        "FastRTC Components",
        "Pipeline Integration"
    ]
    
    for name, passed in zip(test_names, results):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")
    
    # Generate recommendations
    success = generate_recommendations(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
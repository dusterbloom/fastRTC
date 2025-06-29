#!/usr/bin/env python3
"""
Pipeline Debug Test - Isolate Voice Authentication Pipeline Issues
================================================================

This test simulates the real app pipeline to identify where the process hangs.
Based on analysis, potential bottlenecks are:
1. Resemblyzer initialization during voice authentication
2. Threading pipeline vs async pipeline differences
3. Audio buffer processing for voice auth
4. STT engine (faster whisper) processing
5. Voice authentication flow blocking the pipeline

The test will run step-by-step to isolate the exact hanging point.
"""

import os
import sys
import time
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import Optional

# Set environment variables before imports
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['USE_THREADING_PIPELINE'] = 'true'  # Test with threading enabled
os.environ['THREADING_FALLBACK_TO_ASYNC'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

print("🧪 Starting Pipeline Debug Test")
print("=" * 50)

def test_step(step_name: str, func, *args, **kwargs):
    """Test a step and measure execution time"""
    print(f"\n🔍 Testing: {step_name}")
    start_time = time.time()
    
    try:
        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✅ {step_name} - SUCCESS ({elapsed:.2f}s)")
        return result, True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {step_name} - FAILED ({elapsed:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_resemblyzer_import():
    """Test resemblyzer import and initialization"""
    print("Testing resemblyzer import...")
    from src.audio.voice_embeddings import VoiceEmbeddingManager, RESEMBLYZER_AVAILABLE
    print(f"Resemblyzer available: {RESEMBLYZER_AVAILABLE}")
    
    if RESEMBLYZER_AVAILABLE:
        print("Initializing VoiceEmbeddingManager...")
        manager = VoiceEmbeddingManager()
        print("VoiceEmbeddingManager initialized successfully")
        return manager
    return None

def test_voice_assistant_creation():
    """Test voice assistant creation"""
    print("Creating VoiceAssistant...")
    from src.core.voice_assistant import VoiceAssistant
    from src.config.settings import load_config
    
    config = load_config()
    voice_assistant = VoiceAssistant(config=config)
    print("VoiceAssistant created successfully")
    return voice_assistant

async def test_stt_engine():
    """Test STT engine initialization"""
    print("Testing STT engine...")
    from src.audio.engines.stt.faster_whisper_stt import FasterWhisperSTT
    
    stt_engine = FasterWhisperSTT()
    print("STT engine initialized successfully")
    
    # Test with dummy audio
    dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    print("Testing STT transcription...")
    result = await stt_engine.transcribe(dummy_audio)  # Remove sample_rate, use async
    print(f"STT result: {result}")
    return stt_engine

def test_threading_pipeline():
    """Test threading pipeline initialization"""
    print("Testing threading pipeline...")
    from src.integration.threading_callback_handler import ThreadingCallbackHandler
    from src.core.voice_assistant import VoiceAssistant
    from src.config.settings import load_config
    
    config = load_config()
    voice_assistant = VoiceAssistant(config=config)
    
    handler = ThreadingCallbackHandler(
        voice_assistant=voice_assistant,
        stt_engine=voice_assistant.stt_engine,
        tts_engine=voice_assistant.tts_engine,
        voice_mapper=voice_assistant.voice_mapper
    )
    
    print("Starting threading handler...")
    handler.start()
    print("Threading handler started successfully")
    
    # Test with dummy audio
    print("Testing audio processing...")
    dummy_audio = np.random.randn(16000).astype(np.float32)
    audio_tuple = (16000, dummy_audio)
    
    # Process audio (this is where it might hang)
    print("Processing audio through threading pipeline...")
    start_time = time.time()
    
    try:
        # Set a timeout for this test
        result_generator = handler.process_audio_stream(audio_tuple)
        
        # Try to get first result with timeout
        results = []
        timeout_seconds = 30
        
        for i, result in enumerate(result_generator):
            results.append(result)
            elapsed = time.time() - start_time
            print(f"Got result {i+1} after {elapsed:.2f}s: {type(result)}")
            
            if elapsed > timeout_seconds:
                print(f"⚠️ Timeout after {timeout_seconds}s, stopping...")
                break
                
            if i >= 2:  # Get a few results then stop
                break
                
        print(f"Audio processing completed with {len(results)} results")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Audio processing failed after {elapsed:.2f}s: {e}")
        raise
    finally:
        print("Stopping threading handler...")
        handler.stop()
        
    return handler

def test_voice_auth_flow():
    """Test voice authentication flow specifically"""
    print("Testing voice authentication flow...")
    from src.audio.user_identification import SpokenUserIdentifier
    
    # Test with voice auth enabled
    user_id_manager = SpokenUserIdentifier(enable_voice_auth=True)
    print(f"Voice auth enabled: {user_id_manager.enable_voice_auth}")
    
    if user_id_manager.enable_voice_auth:
        # Test audio buffer setting
        dummy_audio = np.random.randn(16000).astype(np.float32)
        print("Setting audio buffer for voice auth...")
        
        if hasattr(user_id_manager, 'voice_manager') and user_id_manager.voice_manager:
            user_id_manager.voice_manager.set_audio_buffer(dummy_audio)
            print("Audio buffer set successfully")
        else:
            print("⚠️ Voice manager not available")
    
    return user_id_manager

def test_unified_callback_handler():
    """Test unified callback handler"""
    print("Testing unified callback handler...")
    from src.integration.unified_callback_handler import UnifiedCallbackHandler
    from src.core.voice_assistant import VoiceAssistant
    from src.config.settings import load_config
    from src.utils.async_utils import AsyncEnvironmentManager
    
    config = load_config()
    voice_assistant = VoiceAssistant(config=config)
    
    # Create async environment manager
    async_env_manager = AsyncEnvironmentManager()
    success = async_env_manager.setup_async_environment(voice_assistant)
    if not success:
        raise RuntimeError("Failed to setup async environment")
    
    handler = UnifiedCallbackHandler(
        voice_assistant=voice_assistant,
        stt_engine=voice_assistant.stt_engine,
        tts_engine=voice_assistant.tts_engine,
        voice_mapper=voice_assistant.voice_mapper,
        event_loop=async_env_manager.get_event_loop()
    )
    
    stats = handler.get_handler_stats()
    print(f"Handler stats: {stats}")
    
    # Test audio processing
    dummy_audio = np.random.randn(16000).astype(np.float32)
    audio_tuple = (16000, dummy_audio)
    
    print("Processing audio through unified handler...")
    start_time = time.time()
    
    try:
        result_generator = handler.process_audio_stream(audio_tuple)
        results = []
        timeout_seconds = 30
        
        for i, result in enumerate(result_generator):
            results.append(result)
            elapsed = time.time() - start_time
            print(f"Got result {i+1} after {elapsed:.2f}s")
            
            if elapsed > timeout_seconds:
                print(f"⚠️ Timeout after {timeout_seconds}s, stopping...")
                break
                
            if i >= 2:
                break
                
        print(f"Unified handler processing completed with {len(results)} results")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Unified handler processing failed after {elapsed:.2f}s: {e}")
        raise
    finally:
        handler.stop()
        
    return handler

def main():
    """Run all tests to isolate the hanging issue"""
    print("🚀 Starting comprehensive pipeline debug test")
    print(f"Threading enabled: {os.getenv('USE_THREADING_PIPELINE')}")
    print(f"Fallback enabled: {os.getenv('THREADING_FALLBACK_TO_ASYNC')}")
    
    tests = [
        ("Resemblyzer Import & Init", test_resemblyzer_import),
        ("Voice Assistant Creation", test_voice_assistant_creation),
        ("STT Engine Test", test_stt_engine),
        ("Voice Auth Flow", test_voice_auth_flow),
        ("Threading Pipeline", test_threading_pipeline),
        ("Unified Callback Handler", test_unified_callback_handler),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        result, success = test_step(test_name, test_func)
        results[test_name] = {"success": success, "result": result}
        
        if not success:
            print(f"\n💥 PIPELINE HANGING POINT IDENTIFIED: {test_name}")
            print("This is likely where your pipeline is getting stuck!")
            break
        
        # Small delay between tests
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print("\n🔍 Analysis:")
    if all(r["success"] for r in results.values()):
        print("✅ All tests passed! The issue might be in the integration or timing.")
        print("💡 Try running the real app with more verbose logging.")
    else:
        failed_tests = [name for name, result in results.items() if not result["success"]]
        print(f"❌ Failed at: {failed_tests[0]}")
        print("💡 This is likely where your pipeline hangs in the real app.")
        
        if "Resemblyzer" in failed_tests[0]:
            print("🔧 Suggestion: Resemblyzer/voice auth is causing issues. Try disabling voice auth.")
        elif "Threading" in failed_tests[0]:
            print("🔧 Suggestion: Threading pipeline has issues. Try using async pipeline instead.")
        elif "STT" in failed_tests[0]:
            print("🔧 Suggestion: STT engine (faster whisper) is hanging. Check CUDA/model loading.")

if __name__ == "__main__":
    main()
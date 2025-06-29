#!/usr/bin/env python3
"""
STT Worker Fix Test
==================

Test to confirm that the STT worker is the actual issue, not voice auth.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Set environment variables
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'false'  # Test with async pipeline
os.environ['LOG_LEVEL'] = 'DEBUG'

sys.path.insert(0, str(Path(__file__).parent.resolve()))

print("🔧 STT Worker Fix Test")
print("=" * 30)

def test_async_pipeline():
    """Test the async pipeline without threading"""
    print("Testing async pipeline (no threading)...")
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
    print(f"Handler type: {stats.get('handler_type')}")
    
    # Test with real speech-like audio (not just noise)
    print("Creating speech-like audio...")
    # Create a simple sine wave that might be recognized as speech
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.3
    
    audio_tuple = (sample_rate, audio)
    
    print("Processing audio through async pipeline...")
    start_time = time.time()
    
    try:
        result_generator = handler.process_audio_stream(audio_tuple)
        results = []
        
        for i, result in enumerate(result_generator):
            results.append(result)
            elapsed = time.time() - start_time
            print(f"Got result {i+1} after {elapsed:.2f}s: {type(result)}")
            
            if elapsed > 30:  # 30 second timeout
                print("⚠️ Timeout, stopping...")
                break
                
            if i >= 2:
                break
                
        print(f"✅ Async pipeline completed with {len(results)} results")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Async pipeline failed after {elapsed:.2f}s: {e}")
        return False
    finally:
        handler.stop()

def test_threading_pipeline_without_voice_auth():
    """Test threading pipeline with voice auth disabled"""
    print("Testing threading pipeline with voice auth DISABLED...")
    
    # Temporarily disable voice auth
    os.environ['DISABLE_VOICE_AUTH'] = 'true'
    
    from src.integration.threading_callback_handler import ThreadingCallbackHandler
    from src.core.voice_assistant import VoiceAssistant
    from src.config.settings import load_config
    
    config = load_config()
    voice_assistant = VoiceAssistant(config=config)
    
    # Disable voice auth if possible
    if hasattr(voice_assistant, 'voice_print_manager'):
        if hasattr(voice_assistant.voice_print_manager, 'enable_voice_auth'):
            voice_assistant.voice_print_manager.enable_voice_auth = False
            print("🔧 Voice auth disabled for test")
    
    handler = ThreadingCallbackHandler(
        voice_assistant=voice_assistant,
        stt_engine=voice_assistant.stt_engine,
        tts_engine=voice_assistant.tts_engine,
        voice_mapper=voice_assistant.voice_mapper
    )
    
    print("Starting threading handler...")
    handler.start()
    
    try:
        # Test with speech-like audio
        sample_rate = 16000
        duration = 2.0
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.3
        
        audio_tuple = (sample_rate, audio)
        
        print("Processing audio through threading pipeline (voice auth disabled)...")
        start_time = time.time()
        
        result_generator = handler.process_audio_stream(audio_tuple)
        results = []
        
        for i, result in enumerate(result_generator):
            results.append(result)
            elapsed = time.time() - start_time
            print(f"Got result {i+1} after {elapsed:.2f}s")
            
            if elapsed > 30:
                print("⚠️ Timeout, stopping...")
                break
                
            if i >= 2:
                break
        
        print(f"✅ Threading pipeline (no voice auth) completed with {len(results)} results")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Threading pipeline failed after {elapsed:.2f}s: {e}")
        return False
    finally:
        print("Stopping threading handler...")
        handler.stop()

def main():
    print("🚀 Testing both pipelines to isolate the issue...")
    
    # Test 1: Async pipeline (should work)
    print("\n" + "="*50)
    print("TEST 1: Async Pipeline (no threading)")
    async_works = test_async_pipeline()
    
    time.sleep(3)
    
    # Test 2: Threading pipeline without voice auth
    print("\n" + "="*50)
    print("TEST 2: Threading Pipeline (voice auth disabled)")
    threading_works = test_threading_pipeline_without_voice_auth()
    
    print("\n" + "="*50)
    print("🏁 DIAGNOSIS:")
    
    if async_works and threading_works:
        print("✅ Both pipelines work - issue might be voice auth integration")
    elif async_works and not threading_works:
        print("❌ THREADING PIPELINE IS THE PROBLEM")
        print("💡 Solution: Use async pipeline instead:")
        print("   ./fastrtc.sh dev  # Remove --threading flag")
    elif not async_works and not threading_works:
        print("❌ STT ENGINE IS THE PROBLEM")
        print("💡 Solution: Check STT engine configuration")
    else:
        print("🤔 Unexpected result - need more investigation")

if __name__ == "__main__":
    main()
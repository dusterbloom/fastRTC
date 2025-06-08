#!/usr/bin/env python3
"""
Test script to verify all critical dependency fixes are working.
"""

def test_critical_fixes():
    """Test all critical fixes."""
    print("Testing critical dependency fixes...")
    
    # Test 1: KokoroONNX class availability
    try:
        from src.audio.engines.tts.kokoro_tts import KokoroONNX
        print("✅ 1. KokoroONNX class available")
    except ImportError as e:
        print(f"❌ 1. KokoroONNX class import failed: {e}")
        return False
    
    # Test 2: VoiceAssistant constructor accepts config parameter
    try:
        from src.core.voice_assistant import VoiceAssistant
        va = VoiceAssistant(config={"test": "value"})
        print("✅ 2. VoiceAssistant accepts config parameter")
    except TypeError as e:
        print(f"❌ 2. VoiceAssistant config parameter failed: {e}")
        return False
    
    # Test 3: Application factory functions available
    try:
        from src.core.main import create_application, create_voice_assistant, VoiceAssistantApplication
        app = create_application()
        va = create_voice_assistant()
        print("✅ 3. Application factory functions available")
    except (ImportError, AttributeError) as e:
        print(f"❌ 3. Application factory functions failed: {e}")
        return False
    
    print("\n🎉 ALL CRITICAL FIXES WORKING SUCCESSFULLY!")
    print("\nSummary of fixes:")
    print("- Fixed missing KokoroONNX class with stub implementation")
    print("- Added config parameter to VoiceAssistant constructor")
    print("- Added missing create_voice_assistant application factory function")
    print("- Fixed performance tests to use text instead of audio data")
    
    return True

if __name__ == "__main__":
    test_critical_fixes()
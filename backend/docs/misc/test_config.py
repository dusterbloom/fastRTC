#!/usr/bin/env python3
"""Simple test script to validate the configuration system."""

from src.config.settings import load_config
from src.config.language_config import get_kokoro_language, get_available_voices
from src.core.interfaces import AudioData
import numpy as np

def main():
    print("Testing FastRTC Voice Assistant Configuration...")
    
    # Test configuration loading
    try:
        config = load_config()
        print("✓ Configuration loaded successfully!")
        print(f"  - Audio sample rate: {config.audio.sample_rate}")
        print(f"  - LLM service: {'Ollama' if config.llm.use_ollama else 'LM Studio'}")
        print(f"  - Preferred TTS voice: {config.tts.preferred_voice}")
        print(f"  - Memory LLM model: {config.memory.llm_model}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    # Test language configuration
    try:
        lang_code = get_kokoro_language("en")
        voices = get_available_voices(lang_code)
        print(f"✓ Language configuration working!")
        print(f"  - English maps to: {lang_code}")
        print(f"  - Available voices: {voices[:3]}...")
    except Exception as e:
        print(f"✗ Language configuration failed: {e}")
        return False
    
    # Test AudioData interface
    try:
        audio = AudioData(
            samples=np.random.random(16000).astype(np.float32),
            sample_rate=16000,
            duration=1.0
        )
        print(f"✓ AudioData interface working!")
        print(f"  - Sample shape: {audio.samples.shape}")
        print(f"  - Duration: {audio.duration}s")
    except Exception as e:
        print(f"✗ AudioData interface failed: {e}")
        return False
    
    print("\n🎉 Phase 1 infrastructure validation complete!")
    return True

if __name__ == "__main__":
    main()
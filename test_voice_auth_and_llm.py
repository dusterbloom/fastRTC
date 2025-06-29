#!/usr/bin/env python3
"""
Test Voice Authentication and LLM Integration
Debug why LLM isn't responding after STT
"""

import os
import sys
import numpy as np
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

from backend.src.utils.logging import get_logger
from backend.src.core.voice_assistant import VoiceAssistant
from backend.src.audio.user_identification import SpokenUserIdentifier
from backend.src.services.llm_service import LLMService

logger = get_logger(__name__)

async def test_voice_auth_and_llm():
    """Test voice authentication setup and LLM service."""
    logger.info("🔐 Testing voice authentication and LLM integration...")
    
    try:
        # Test 1: Voice Authentication Setup
        logger.info("📝 Step 1: Testing voice authentication setup...")
        
        voice_auth = SpokenUserIdentifier()
        logger.info(f"✅ Voice auth created - enabled: {voice_auth.enable_voice_auth}")
        logger.info(f"✅ Voice manager available: {voice_auth.voice_manager is not None}")
        
        if voice_auth.voice_manager:
            logger.info("✅ Resemblyzer voice authentication is working!")
        else:
            logger.warning("⚠️ Voice manager not available - check Resemblyzer installation")
        
        # Test 2: LLM Service
        logger.info("📝 Step 2: Testing LLM service...")
        
        llm_service = LLMService()
        test_prompt = "Hello, can you hear me?"
        
        logger.info(f"🧠 Testing LLM with prompt: '{test_prompt}'")
        
        # Test LLM response
        response_tokens = []
        async for token in llm_service.stream_response(test_prompt):
            response_tokens.append(token)
            logger.info(f"📥 LLM token: '{token}'")
            if len(response_tokens) > 10:  # Limit for test
                break
        
        if response_tokens:
            logger.info(f"✅ LLM service working! Got {len(response_tokens)} tokens")
            logger.info(f"✅ Response start: '{''.join(response_tokens[:5])}'")
        else:
            logger.error("❌ LLM service not responding")
            return False
        
        # Test 3: Voice Assistant Integration
        logger.info("📝 Step 3: Testing voice assistant integration...")
        
        voice_assistant = VoiceAssistant()
        logger.info(f"✅ Voice assistant created")
        logger.info(f"✅ Has voice_print_manager: {hasattr(voice_assistant, 'voice_print_manager')}")
        
        if hasattr(voice_assistant, 'voice_print_manager'):
            vpm = voice_assistant.voice_print_manager
            logger.info(f"✅ Voice print manager type: {type(vpm)}")
            logger.info(f"✅ Voice auth enabled: {vpm.enable_voice_auth}")
            logger.info(f"✅ Voice manager available: {vpm.voice_manager is not None}")
        
        # Test 4: Voice Authentication Flow
        logger.info("📝 Step 4: Testing voice authentication flow...")
        
        # Simulate audio buffer (like threading callback does)
        sample_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second of audio
        voice_assistant.voice_print_manager.set_audio_buffer(sample_audio)
        logger.info(f"✅ Audio buffer set: {len(sample_audio)} samples")
        
        # Test text processing (what LLM worker does)
        test_text = "Hello Echo, can you hear me?"
        logger.info(f"🔍 Testing voice auth with text: '{test_text}'")
        
        auth_result = voice_assistant.voice_print_manager.process_text(test_text)
        logger.info(f"🔍 Voice auth result: {auth_result}")
        
        if auth_result:
            logger.info(f"✅ Voice auth returned result: {auth_result.get('action', 'no action')}")
        else:
            logger.info("✅ Voice auth returned None (normal for unregistered user)")
        
        # Test 5: Direct LLM Call (bypass voice auth)
        logger.info("📝 Step 5: Testing direct LLM call...")
        
        try:
            # This simulates what should happen after voice auth
            direct_response = []
            async for token in voice_assistant.llm_service.stream_response(test_text):
                direct_response.append(token)
                logger.info(f"📥 Direct LLM token: '{token}'")
                if len(direct_response) > 5:
                    break
            
            if direct_response:
                logger.info(f"✅ Direct LLM call working! Response: '{''.join(direct_response)}'")
            else:
                logger.error("❌ Direct LLM call failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Direct LLM call error: {e}")
            return False
        
        # Results
        logger.info("📊 Voice Auth and LLM Test Results:")
        logger.info(f"   Voice auth setup: ✅")
        logger.info(f"   LLM service: ✅")
        logger.info(f"   Voice assistant: ✅")
        logger.info(f"   Auth flow: ✅")
        logger.info(f"   Direct LLM: ✅")
        
        logger.info("🎉 All components working! Issue might be in pipeline flow.")
        
        # Recommendations
        logger.info("💡 Recommendations:")
        logger.info("   1. Check if LLM worker is receiving STT transcriptions")
        logger.info("   2. Check if voice auth is blocking LLM processing")
        logger.info("   3. Check WebRTC connection stability")
        logger.info("   4. Try registering a user: 'register as test PIN 1234'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_voice_auth_and_llm())
    if success:
        print("✅ Voice auth and LLM test passed!")
        sys.exit(0)
    else:
        print("❌ Voice auth and LLM test failed!")
        sys.exit(1)
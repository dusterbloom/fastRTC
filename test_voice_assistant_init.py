#!/usr/bin/env python3
"""
Test Voice Assistant Initialization
Check if voice assistant is properly initialized with HTTP session
"""

import os
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

from backend.src.utils.logging import get_logger
from backend.src.core.voice_assistant import VoiceAssistant
from backend.src.utils.async_utils import AsyncEnvironmentManager

logger = get_logger(__name__)

async def test_voice_assistant_initialization():
    """Test voice assistant initialization like the main app does."""
    logger.info("🧠 Testing voice assistant initialization...")
    
    try:
        # Step 1: Create voice assistant (like main.py does)
        logger.info("📝 Step 1: Creating voice assistant instance...")
        voice_assistant = VoiceAssistant()
        logger.info(f"✅ Voice assistant created")
        
        # Step 2: Check initial state
        logger.info("📝 Step 2: Checking initial state...")
        logger.info(f"   Has voice_print_manager: {hasattr(voice_assistant, 'voice_print_manager')}")
        logger.info(f"   Has llm_service: {hasattr(voice_assistant, 'llm_service')}")
        logger.info(f"   Has http_session: {hasattr(voice_assistant, 'http_session')}")
        
        if hasattr(voice_assistant, 'llm_service'):
            llm_service = voice_assistant.llm_service
            logger.info(f"   LLM service type: {type(llm_service)}")
            logger.info(f"   LLM has http_session: {hasattr(llm_service, 'http_session')}")
            if hasattr(llm_service, 'http_session'):
                logger.info(f"   LLM http_session value: {llm_service.http_session}")
        
        # Step 3: Initialize async environment (like main.py does)
        logger.info("📝 Step 3: Setting up async environment...")
        async_env_manager = AsyncEnvironmentManager()
        
        # This should call voice_assistant.initialize_async()
        success = async_env_manager.setup_async_environment(voice_assistant)
        if not success:
            logger.error("❌ Failed to setup async environment")
            return False
        
        logger.info("✅ Async environment setup complete")
        
        # Step 4: Check state after initialization
        logger.info("📝 Step 4: Checking state after initialization...")
        
        # Wait a bit for initialization to complete
        await asyncio.sleep(2.0)
        
        if hasattr(voice_assistant, 'llm_service'):
            llm_service = voice_assistant.llm_service
            logger.info(f"   LLM http_session after init: {llm_service.http_session}")
            logger.info(f"   LLM http_session type: {type(llm_service.http_session)}")
            
            if llm_service.http_session:
                logger.info("✅ LLM service has HTTP session!")
                
                # Step 5: Test LLM streaming
                logger.info("📝 Step 5: Testing LLM streaming...")
                try:
                    response_tokens = []
                    async for token in llm_service.stream_response("Hello, can you hear me?"):
                        response_tokens.append(token)
                        logger.info(f"📥 LLM token: '{token}'")
                        if len(response_tokens) > 5:
                            break
                    
                    if response_tokens:
                        logger.info(f"✅ LLM streaming works! Got {len(response_tokens)} tokens")
                        logger.info(f"✅ Response: '{''.join(response_tokens)}'")
                    else:
                        logger.error("❌ LLM streaming returned no tokens")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ LLM streaming error: {e}")
                    return False
            else:
                logger.error("❌ LLM service missing HTTP session after initialization")
                return False
        
        # Step 6: Test voice authentication
        logger.info("📝 Step 6: Testing voice authentication...")
        if hasattr(voice_assistant, 'voice_print_manager'):
            vpm = voice_assistant.voice_print_manager
            logger.info(f"   Voice auth enabled: {vpm.enable_voice_auth}")
            logger.info(f"   Voice manager available: {vpm.voice_manager is not None}")
            
            if vpm.voice_manager:
                logger.info("✅ Voice authentication ready!")
            else:
                logger.warning("⚠️ Voice authentication not available")
        
        # Cleanup
        async_env_manager.shutdown()
        
        logger.info("🎉 Voice assistant initialization test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_voice_assistant_initialization())
    if success:
        print("✅ Voice assistant initialization test passed!")
        sys.exit(0)
    else:
        print("❌ Voice assistant initialization test failed!")
        sys.exit(1)
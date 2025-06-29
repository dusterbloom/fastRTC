#!/usr/bin/env python3
"""
Debug Voice Authentication in Real App
Check the actual state of voice authentication in the running system
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.voice_assistant import VoiceAssistant

logger = get_logger(__name__)

def debug_voice_auth():
    """Debug voice authentication setup in real app context."""
    logger.info("🔍 Debugging voice authentication setup...")
    
    try:
        # Create voice assistant like the real app does
        logger.info("📝 Creating voice assistant...")
        voice_assistant = VoiceAssistant()
        
        # Check voice_print_manager
        logger.info("📝 Checking voice_print_manager...")
        if hasattr(voice_assistant, 'voice_print_manager'):
            vpm = voice_assistant.voice_print_manager
            logger.info(f"✅ voice_print_manager exists: {type(vpm)}")
            logger.info(f"   Enable voice auth: {vpm.enable_voice_auth}")
            logger.info(f"   Voice manager: {vpm.voice_manager is not None}")
            
            if vpm.voice_manager:
                logger.info(f"   Voice manager type: {type(vpm.voice_manager)}")
                logger.info("✅ Voice authentication should be working!")
                
                # Test voice auth patterns
                logger.info("📝 Testing voice auth patterns...")
                
                test_phrases = [
                    "Hello Echo, can you hear me?",  # Should return None (no auth)
                    "register my voice",             # Should start enrollment
                    "it's me",                      # Should try authentication
                ]
                
                for phrase in test_phrases:
                    logger.info(f"🔍 Testing phrase: '{phrase}'")
                    result = vpm.process_text(phrase)
                    logger.info(f"   Result: {result}")
                
            else:
                logger.error("❌ Voice manager is None - Resemblyzer issue?")
                
        else:
            logger.error("❌ voice_print_manager not found on voice assistant")
        
        # Check LLM service
        logger.info("📝 Checking LLM service...")
        if hasattr(voice_assistant, 'llm_service'):
            llm_service = voice_assistant.llm_service
            logger.info(f"✅ LLM service exists: {type(llm_service)}")
            logger.info(f"   HTTP session: {getattr(llm_service, 'http_session', 'NOT_SET')}")
        else:
            logger.error("❌ LLM service not found")
        
        # Recommendations
        logger.info("💡 Voice Authentication Commands:")
        logger.info("   To enroll: 'register my voice' or 'enroll voice'")
        logger.info("   To authenticate: 'it's me' or 'this is me'")
        logger.info("   Normal chat: Any other phrase (should work without auth)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = debug_voice_auth()
    if success:
        print("✅ Voice auth debug completed!")
        sys.exit(0)
    else:
        print("❌ Voice auth debug failed!")
        sys.exit(1)
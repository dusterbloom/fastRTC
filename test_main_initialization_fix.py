#!/usr/bin/env python3
"""
Test Main Application Initialization Fix
Verify voice assistant is properly initialized before callback handler creation
"""

import os
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.main import VoiceAssistantApplication

logger = get_logger(__name__)

async def test_main_initialization():
    """Test the main application initialization with our fix."""
    logger.info("🚀 Testing main application initialization...")
    
    try:
        # Create application instance
        app = VoiceAssistantApplication()
        
        # Initialize application (this should now wait for proper voice assistant init)
        logger.info("📝 Initializing application...")
        success = await app.initialize()
        
        if not success:
            logger.error("❌ Application initialization failed")
            return False
        
        logger.info("✅ Application initialization completed")
        
        # Check voice assistant state
        if app.voice_assistant:
            logger.info("📝 Checking voice assistant state...")
            
            # Check LLM service
            if hasattr(app.voice_assistant, 'llm_service'):
                llm_service = app.voice_assistant.llm_service
                logger.info(f"   LLM service: {type(llm_service)}")
                
                if hasattr(llm_service, 'http_session'):
                    logger.info(f"   HTTP session: {llm_service.http_session is not None}")
                    if llm_service.http_session:
                        logger.info("✅ LLM service properly initialized!")
                    else:
                        logger.error("❌ LLM service missing HTTP session")
                        return False
                else:
                    logger.error("❌ LLM service missing http_session attribute")
                    return False
            else:
                logger.error("❌ Voice assistant missing llm_service")
                return False
            
            # Check voice authentication
            if hasattr(app.voice_assistant, 'voice_print_manager'):
                vpm = app.voice_assistant.voice_print_manager
                logger.info(f"   Voice auth enabled: {vpm.enable_voice_auth}")
                logger.info(f"   Voice manager: {vpm.voice_manager is not None}")
                logger.info("✅ Voice authentication available!")
            else:
                logger.warning("⚠️ Voice authentication not available")
        
        # Check callback handler
        if app.callback_handler:
            logger.info("📝 Checking callback handler...")
            logger.info(f"   Handler type: {app.callback_handler.handler_type}")
            logger.info(f"   Active handler: {type(app.callback_handler.active_handler)}")
            logger.info("✅ Callback handler created!")
        else:
            logger.error("❌ Callback handler not created")
            return False
        
        # Cleanup
        logger.info("🧹 Cleaning up...")
        app.shutdown()
        
        logger.info("🎉 Main initialization test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_main_initialization())
    if success:
        print("✅ Main initialization test passed!")
        sys.exit(0)
    else:
        print("❌ Main initialization test failed!")
        sys.exit(1)
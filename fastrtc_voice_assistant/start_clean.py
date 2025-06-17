#!/usr/bin/env python3
"""
Simplified FastRTC Voice Assistant Entry Point
============================================

A minimal FastAPI entry point that leverages the refactored architecture.
This replaces the 413-line start.py with a clean, maintainable version.

Launch with:
    python start_clean.py
    or
    fastRTC/python -m uvicorn fastrtc_voice_assistant.start_clean:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core.voice_assistant import VoiceAssistant
from src.integration.fastrtc_bridge import FastRTCBridge
from src.integration.callback_handler import StreamCallbackHandler
from src.utils.async_utils import AsyncEnvironmentManager
from src.config.settings import load_config
from src.utils.logging import get_logger, setup_logging

# Initial setup
setup_logging()
logger = get_logger(__name__)
logger.critical("🚨 TOP LEVEL LOGGER TEST IN START_CLEAN.PY 🚨") # New test log

# Global components
voice_assistant: Optional[VoiceAssistant] = None
fastrtc_bridge: Optional[FastRTCBridge] = None
callback_handler: Optional[StreamCallbackHandler] = None
async_env_manager: Optional[AsyncEnvironmentManager] = None

async def initialize_voice_assistant():
    """Initialize the voice assistant and all its components."""
    global voice_assistant, fastrtc_bridge, callback_handler, async_env_manager
    
    if voice_assistant is not None:
        return  # Already initialized
    
    try:
        logger.info("🚀 Initializing Voice Assistant...")
        
        # DIAGNOSTIC: Check config loading
        logger.critical("🔍 [DIAGNOSTIC] Starting config loading...")
        config = load_config()
        logger.critical("✅ [DIAGNOSTIC] Config loaded successfully")
        
        # DIAGNOSTIC: Check VoiceAssistant creation
        logger.critical("🔍 [DIAGNOSTIC] Starting VoiceAssistant initialization...")
        logger.critical("🔍 [DIAGNOSTIC] This may take 5-15 minutes for model loading...")
        voice_assistant = VoiceAssistant(config=config)
        logger.critical("✅ [DIAGNOSTIC] VoiceAssistant created successfully")
        logger.critical(f"[DEBUG] VoiceAssistant created: {voice_assistant}")
        logger.critical(f"[DEBUG] STT engine: {getattr(voice_assistant, 'stt_engine', None)}")
        logger.critical(f"[DEBUG] TTS engine: {getattr(voice_assistant, 'tts_engine', None)}")
        
        # DIAGNOSTIC: Check async environment setup
        logger.critical("🔍 [DIAGNOSTIC] Starting AsyncEnvironmentManager setup...")
        async_env_manager = AsyncEnvironmentManager() # Corrected class name
        success = async_env_manager.setup_async_environment(voice_assistant)
        if not success:
            raise RuntimeError("Failed to setup async environment")
        logger.critical("✅ [DIAGNOSTIC] AsyncEnvironmentManager setup complete")
        
        # DIAGNOSTIC: Check FastRTC bridge creation
        logger.critical("🔍 [DIAGNOSTIC] Starting FastRTCBridge creation...")
        fastrtc_bridge = FastRTCBridge()
        logger.critical(f"✅ [DIAGNOSTIC] FastRTCBridge created: {fastrtc_bridge}")
        
        # DIAGNOSTIC: Check callback handler creation
        logger.critical("🔍 [DIAGNOSTIC] Starting StreamCallbackHandler creation...")
        callback_handler = StreamCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=voice_assistant.tts_engine,
            language_detector=voice_assistant.language_detector,
            voice_mapper=voice_assistant.voice_mapper,
            event_loop=async_env_manager.get_event_loop()
        )
        logger.critical("✅ [DIAGNOSTIC] StreamCallbackHandler created successfully")
        logger.critical(f"[DEBUG] StreamCallbackHandler created: {callback_handler}")
        logger.critical(f"[DEBUG] CallbackHandler STT: {getattr(callback_handler, 'stt_engine', None)}")
        
        logger.info("✅ Voice Assistant initialization complete")

        
    except Exception as e:
        logger.error(f"❌ Failed to initialize voice assistant: {e}")
        raise

async def create_fastrtc_stream():
    """Create and configure the FastRTC stream."""
    logger.critical("🔍 [DIAGNOSTIC] create_fastrtc_stream() started")
    
    if not all([voice_assistant, fastrtc_bridge, callback_handler]):
        logger.critical("🔍 [DIAGNOSTIC] Components not ready, re-initializing...")
        await initialize_voice_assistant()
    
    logger.critical("🔍 [DIAGNOSTIC] About to import SileroVadOptions...")
    # Create the FastRTC stream with our callback
    from fastrtc import SileroVadOptions
    logger.critical("✅ [DIAGNOSTIC] SileroVadOptions imported successfully")
    
    logger.critical("🔍 [DIAGNOSTIC] About to create FastRTC stream...")
    stream = fastrtc_bridge.create_stream(
        callback_function=callback_handler.process_audio_stream,
        speech_threshold=0.05,  # Sensitive speech detection
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
 
    )
    logger.critical("✅ [DIAGNOSTIC] FastRTC stream creation completed")
    
    logger.info("🎤 FastRTC stream created successfully")
    return stream

# ────────────────────────── FASTAPI APP  ────────────────────────────

# CORS setup for front-end integration
origins = [
    "http://localhost:5173",
    "http://localhost:3005",
    "http://127.0.0.1:3005",
    "http://frontend:3005",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Add debugging for CORS issues
logger.info(f"🌐 CORS origins configured: {origins}")

app = FastAPI(title="FastRTC Voice Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)


# Add error handler for connection issues
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )


@app.on_event("startup")
async def startup_event():

    global fastrtc_bridge
    if fastrtc_bridge:
        try:
            fastrtc_bridge.mount(app, path="/assistant")
            logger.info("🌐 WebRTC routes mounted successfully")
        except Exception as e:
            logger.error(f"❌ Failed to mount WebRTC routes: {e}")
            raise


    """Initialize the voice assistant and mount FastRTC routes."""
    print(">>> FastAPI startup_event triggered")
    logger.critical("🔍 [DIAGNOSTIC] FastAPI startup_event starting...")
    try:
        # DIAGNOSTIC: Voice assistant initialization
        logger.critical("🔍 [DIAGNOSTIC] About to call initialize_voice_assistant()...")
        await initialize_voice_assistant()
        logger.critical("✅ [DIAGNOSTIC] initialize_voice_assistant() completed")
        
        # DIAGNOSTIC: FastRTC stream creation
        logger.critical("🔍 [DIAGNOSTIC] About to create FastRTC stream...")
        stream = await create_fastrtc_stream()
        logger.critical("✅ [DIAGNOSTIC] FastRTC stream created")
        
        logger.critical("🔍 [DIAGNOSTIC] About to mount FastRTC routes...")
        stream.mount(app, path="/assistant")
        logger.critical("✅ [DIAGNOSTIC] FastRTC routes mounted")
        
        logger.info("🌐 FastRTC routes mounted at /assistant")
        logger.info("✅ FastRTC Voice Assistant is ready!")
        logger.critical("🎉 [DIAGNOSTIC] STARTUP COMPLETE - ALL SYSTEMS READY!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start voice assistant: {e}")
        logger.critical(f"💥 [DIAGNOSTIC] STARTUP FAILED: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on FastAPI shutdown."""
    global voice_assistant, fastrtc_bridge, async_env_manager
    
    try:
        logger.info("🛑 Shutting down voice assistant...")
        
        if fastrtc_bridge:
            fastrtc_bridge.stop_stream()
            
        if async_env_manager:
            async_env_manager.shutdown(timeout=10)
            
        logger.info("👋 Voice assistant shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("🏠 Root endpoint accessed")
    return {
        "message": "FastRTC Voice Assistant is running",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck."""
    try:
        # Verify core components are initialized
        if not all([voice_assistant, fastrtc_bridge, callback_handler]):
            return {"status": "unhealthy", "reason": "Components not initialized"}
            
        return {
            "status": "healthy",
            "message": "FastRTC Voice Assistant is running",
            "components": {
                "voice_assistant": bool(voice_assistant),
                "fastrtc_bridge": bool(fastrtc_bridge),
                "callback_handler": bool(callback_handler)
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": f"Health check failed: {str(e)}"}

@app.get("/cors-test")
async def cors_test():
    """Simple CORS test endpoint."""
    logger.info("🌐 CORS test endpoint accessed")
    return {
        "message": "CORS test successful",
        "timestamp": str(datetime.now()),
        "headers_ok": True
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        if not voice_assistant:
            return {"status": "unhealthy", "error": "Voice assistant not initialized"}
            
        return {
            "status": "healthy",
            "components": {
                "voice_assistant": voice_assistant is not None,
                "fastrtc_bridge": fastrtc_bridge is not None,
                "callback_handler": callback_handler is not None,
                "async_env_manager": async_env_manager is not None,
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Optional: serve a built SPA from /frontend/dist
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")

# ────────────────────────── ENTRY POINT  ────────────────────────────
def detect_environment():
    """Detect if running in Docker container or locally."""
    # Check common Docker environment indicators
    if os.path.exists('/.dockerenv'):
        return "docker"
    if os.getenv("ENVIRONMENT") == "production":  # Set in docker-compose.yml
        return "docker"
    if str(Path(__file__).parent.resolve()) == "/workspace":
        return "docker"
    return "local"

if __name__ == "__main__":
    import uvicorn
    
    # Determine module reference based on environment
    environment = detect_environment()
    if environment == "docker":
        app_module = "start_clean:app"
        default_port = 8080
        default_host = "0.0.0.0" # "voice-assistant"
    else:
        app_module = "fastrtc_voice_assistant.start_clean:app"
        default_port = 8000
        default_host = "0.0.0.0"
    
    logger.info(f"🌍 Environment detected: {environment}")
    logger.info(f"📦 Module reference: {app_module}")
    
    uvicorn.run(
        app_module,
        host=os.getenv("HOST", default_host),
        port=int(os.getenv("PORT", default_port)),
        reload=True,
        log_config=None
    )

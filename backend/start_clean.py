#!/usr/bin/env python3
"""
Simplified FastRTC Voice Assistant Entry Point
============================================

A minimal FastAPI entry point that leverages the refactored architecture.
This replaces the 413-line start.py with a clean, maintainable version.

Launch with:
    python start_clean.py
    or
    fastRTC/python -m uvicorn backend.start_clean:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core.voice_assistant import VoiceAssistant
from src.integration.fastrtc_bridge import FastRTCBridge
from src.integration.callback_handler import StreamCallbackHandler
from src.utils.async_utils import AsyncEnvironmentManager
from src.config.settings import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.process_killer import force_kill_after_timeout, setup_signal_handlers, kill_child_processes

# Initial setup
setup_logging("DEBUG")  # Set up logging with DEBUG level
logger = get_logger(__name__)
setup_signal_handlers()  # Set up Ctrl+C and kill signal handlers
logger.critical("🚨 TOP LEVEL LOGGER TEST IN START_CLEAN.PY 🚨") # New test log

# Global components
voice_assistant: Optional[VoiceAssistant] = None
fastrtc_bridge: Optional[FastRTCBridge] = None
callback_handler: Optional[StreamCallbackHandler] = None
async_env_manager: Optional[AsyncEnvironmentManager] = None

async def initialize_voice_assistant():
    """Initialize the voice assistant and all its components."""
    import time
    global voice_assistant, fastrtc_bridge, callback_handler, async_env_manager

    if voice_assistant is not None:
        return  # Already initialized

    try:
        logger.critical(f"[TIMING] [START] initialize_voice_assistant at {time.strftime('%H:%M:%S')}")
        logger.info("🚀 Initializing Voice Assistant...")

        # Step 1: Create voice assistant
        print(f"[DEBUG] About to create VoiceAssistant at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [START] VoiceAssistant(config=load_config()) at {time.strftime('%H:%M:%S')}")
        voice_assistant = VoiceAssistant(config=load_config())
        print(f"[DEBUG] VoiceAssistant created successfully at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [END] VoiceAssistant(config=load_config()) at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[DEBUG] VoiceAssistant created: {voice_assistant}")
        logger.critical(f"[DEBUG] STT engine: {getattr(voice_assistant, 'stt_engine', None)}")
        logger.critical(f"[DEBUG] TTS engine: {getattr(voice_assistant, 'tts_engine', None)}")

        # Step 2: Initialize async environment
        print(f"[DEBUG] About to create AsyncEnvironmentManager at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [START] AsyncEnvironmentManager() at {time.strftime('%H:%M:%S')}")
        async_env_manager = AsyncEnvironmentManager()
        print(f"[DEBUG] AsyncEnvironmentManager created at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [END] AsyncEnvironmentManager() at {time.strftime('%H:%M:%S')}")
        print(f"[DEBUG] About to setup async environment at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [START] async_env_manager.setup_async_environment at {time.strftime('%H:%M:%S')}")
        success = async_env_manager.setup_async_environment(voice_assistant)
        print(f"[DEBUG] Async environment setup complete at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [END] async_env_manager.setup_async_environment at {time.strftime('%H:%M:%S')}")
        if not success:
            raise RuntimeError("Failed to setup async environment")

        # Step 3: Create FastRTC bridge
        print(f"[DEBUG] About to create FastRTCBridge at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [START] FastRTCBridge() at {time.strftime('%H:%M:%S')}")
        fastrtc_bridge = FastRTCBridge()
        print(f"[DEBUG] FastRTCBridge created at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [END] FastRTCBridge() at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[DEBUG] FastRTCBridge created: {fastrtc_bridge}")

        # Step 4: Create callback handler with all dependencies
        print(f"[DEBUG] About to create StreamCallbackHandler at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [START] StreamCallbackHandler(...) at {time.strftime('%H:%M:%S')}")
        callback_handler = StreamCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=voice_assistant.tts_engine,
            voice_mapper=voice_assistant.voice_mapper,
            event_loop=async_env_manager.get_event_loop()
        )
        print(f"[DEBUG] StreamCallbackHandler created at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[TIMING] [END] StreamCallbackHandler(...) at {time.strftime('%H:%M:%S')}")
        logger.critical(f"[DEBUG] StreamCallbackHandler created: {callback_handler}")
        logger.critical(f"[DEBUG] CallbackHandler STT: {getattr(callback_handler, 'stt_engine', None)}")

        print(f"[DEBUG] Voice Assistant initialization complete at {time.strftime('%H:%M:%S')}")
        logger.info("✅ Voice Assistant initialization complete")
        logger.critical(f"[TIMING] [END] initialize_voice_assistant at {time.strftime('%H:%M:%S')}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize voice assistant: {e}")
        raise

async def create_fastrtc_stream():
    """Create and configure the FastRTC stream."""
    import time
    if not all([voice_assistant, fastrtc_bridge, callback_handler]):
        await initialize_voice_assistant()

    # Create the FastRTC stream with our callback
    logger.critical(f"[TIMING] [START] fastrtc_bridge.create_stream at {time.strftime('%H:%M:%S')}")
    from fastrtc import SileroVadOptions
    stream = fastrtc_bridge.create_stream(
        callback_function=callback_handler.process_audio_stream,
        speech_threshold=0.05,  # Sensitive speech detection
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
    )
    logger.critical(f"[TIMING] [END] fastrtc_bridge.create_stream at {time.strftime('%H:%M:%S')}")

    logger.info("🎤 FastRTC stream created successfully")
    return stream

# ────────────────────────── FASTAPI APP  ────────────────────────────
app = FastAPI(title="FastRTC Voice Assistant", version="1.0.0")

# CORS setup for front-end integration
origins = [
    "http://localhost:5173",               # Original Vite dev-server port
    "http://localhost:3001",               # Next.js dev-server port
    "http://127.0.0.1:3001",              # Explicit IP for Next.js dev-server
    os.getenv("FRONTEND_URL", ""),         # prod domain, if set
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the voice assistant and mount FastRTC routes."""
    print(">>> FastAPI startup_event triggered")
    try:
        # Initialize voice assistant components
        print("[DEBUG] startup_event: About to start voice assistant initialization")
        logger.info("🔧 Starting voice assistant initialization...")
        await initialize_voice_assistant()
        print("[DEBUG] startup_event: Voice assistant initialization completed")
        logger.info("✅ Voice assistant initialization completed!")
        
        # Create and mount FastRTC stream
        print("[DEBUG] startup_event: About to create FastRTC stream")
        logger.info("🔧 Creating FastRTC stream...")
        stream = await create_fastrtc_stream()
        
        # Store bridge reference for future use
        print("[DEBUG] startup_event: Setting up FastRTC bridge reference")
        voice_assistant.fastrtc_bridge = fastrtc_bridge
        
        print("[DEBUG] startup_event: FastRTC stream created, about to mount")
        logger.info("🔧 Mounting FastRTC stream...")
        stream.mount(app, path="/assistant")
        print("[DEBUG] startup_event: FastRTC stream mounted successfully")
        
        logger.info("🌐 FastRTC routes mounted at /assistant")
        logger.info("✅ FastRTC Voice Assistant is ready!")
        print("[DEBUG] startup_event: Startup complete!")
        
    except Exception as e:
        print(f"[DEBUG] startup_event: Exception occurred: {e}")
        logger.error(f"❌ Failed to start voice assistant: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on FastAPI shutdown."""
    global voice_assistant, fastrtc_bridge, async_env_manager
    
    try:
        logger.info("🛑 Shutting down voice assistant...")
        
        # Start force-kill timer (KISS principle: just kill everything after timeout)
        force_kill_after_timeout(timeout_seconds=3.0)
        
        if fastrtc_bridge:
            fastrtc_bridge.stop_stream()
            
        if async_env_manager:
            async_env_manager.shutdown(timeout=3.0)  # Reduced timeout
            
        # Kill any child processes
        kill_child_processes()
            
        logger.info("👋 Voice assistant shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
        # Force kill immediately on error
        force_kill_after_timeout(timeout_seconds=1.0)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "FastRTC Voice Assistant is running",
        "status": "healthy",
        "version": "1.0.0"
    }

class LanguageRequest(BaseModel):
    language: str

@app.post("/api/set-language")
async def set_language(request: LanguageRequest):
    """Set the voice assistant language."""
    try:
        if not voice_assistant:
            return {"status": "error", "message": "Voice assistant not initialized"}
        
        # Validate language code
        valid_languages = ['a', 'b', 'e', 'i', 'f', 'p', 'j', 'z', 'h']
        if request.language not in valid_languages:
            return {"status": "error", "message": f"Invalid language code: {request.language}"}
        
        # Set the language
        voice_assistant.current_language = request.language
        
        return {
            "status": "success", 
            "message": f"Language set to {request.language}",
            "language": request.language
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "start_clean:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_config=None  
    )

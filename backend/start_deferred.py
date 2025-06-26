#!/usr/bin/env python3
"""
FastRTC Voice Assistant - Non-Blocking Startup Solution
=====================================================

Core Problem: Voice assistant initialization blocks FastAPI startup
Solution: Deferred initialization with background task execution
"""

from __future__ import annotations

import os
os.environ['STT_BACKEND'] = 'faster'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import asyncio
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core.voice_assistant import VoiceAssistant
from src.integration.fastrtc_bridge import FastRTCBridge
from src.integration.callback_handler import StreamCallbackHandler
from src.utils.async_utils import AsyncEnvironmentManager
from src.config.settings import load_config
from src.utils.logging import get_logger, setup_logging

# Initial setup
setup_logging("DEBUG")
logger = get_logger(__name__)
logger.debug("🚨 TOP LEVEL LOGGER TEST IN START_CLEAN.PY 🚨") # New test log

# Global components with thread-safe initialization
_components_lock = threading.Lock()
_initialization_task: Optional[asyncio.Task] = None

class GlobalComponents:
    """Thread-safe container for voice assistant components"""
    def __init__(self):
        self.voice_assistant: Optional[VoiceAssistant] = None
        self.fastrtc_bridge: Optional[FastRTCBridge] = None
        self.callback_handler: Optional[StreamCallbackHandler] = None
        self.async_env_manager: Optional[AsyncEnvironmentManager] = None
        self.stream = None
        self.is_initialized = False
        self.initialization_error: Optional[str] = None

components = GlobalComponents()

async def initialize_voice_assistant_deferred(app: FastAPI):
    """
    Initialize voice assistant in background without blocking server startup.
    This runs AFTER the server is listening on port 8000.
    """
    global components
    
    try:
        logger.info("🚀 Starting deferred voice assistant initialization...")
        
        # Small delay to ensure server is fully ready
        await asyncio.sleep(1.0)
        
        # Create voice assistant
        logger.info("Creating VoiceAssistant...")
        voice_assistant = VoiceAssistant(config=load_config())
        
        # Create async environment manager
        async_env_manager = AsyncEnvironmentManager()
        success = async_env_manager.setup_async_environment(voice_assistant)
        if not success:
            raise RuntimeError("Failed to setup async environment")
        
        # Create FastRTC bridge
        fastrtc_bridge = FastRTCBridge()
        
        # Create callback handler
        callback_handler = StreamCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=voice_assistant.tts_engine,
            voice_mapper=voice_assistant.voice_mapper,
            event_loop=async_env_manager.get_event_loop()
        )
        
        # Create FastRTC stream with proper network configuration
        logger.info("Creating FastRTC stream...")
        from fastrtc import SileroVadOptions
        
        # Get server configuration from environment
        server_host = os.getenv("HOST", "0.0.0.0")
        server_port = int(os.getenv("PORT", "8000"))
        external_ip = os.getenv("EXTERNAL_IP", "localhost")
        
        logger.info(f"Server config - Host: {server_host}, Port: {server_port}, External IP: {external_ip}")
        
        stream = fastrtc_bridge.create_stream(
            callback_function=callback_handler.process_audio_stream,
            speech_threshold=0.05,
            server_name=server_host,
            server_port=server_port,
            share=False,
        )
        
        # Mount stream to FastAPI app
        logger.info("Mounting FastRTC stream to /assistant...")
        logger.debug(f"Stream to mount: {stream}")
        logger.debug(f"App to mount to: {app}")
        stream.mount(app, path="/assistant")
        logger.debug("FastRTC stream mounted successfully")
        
        # Store components
        with _components_lock:
            components.voice_assistant = voice_assistant
            components.fastrtc_bridge = fastrtc_bridge
            components.callback_handler = callback_handler
            components.async_env_manager = async_env_manager
            components.stream = stream
            components.is_initialized = True
        
        logger.info("✅ Voice assistant initialization complete! WebRTC endpoint available at /assistant")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize voice assistant: {e}")
        import traceback
        traceback.print_exc()
        with _components_lock:
            components.initialization_error = str(e)

# Create FastAPI app WITHOUT lifespan context manager
app = FastAPI(
    title="FastRTC Voice Assistant",
    version="1.0.0",
    description="Real-time voice assistant with WebRTC support"
)

# CORS setup
origins = [
    "http://localhost:5173",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    os.getenv("FRONTEND_URL", ""),
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
    """
    Start voice assistant initialization in background.
    This doesn't block the server from starting.
    """
    global _initialization_task
    # Create background task for initialization
    _initialization_task = asyncio.create_task(initialize_voice_assistant_deferred(app))
    logger.info("🚀 Server started, voice assistant initializing in background...")

@app.get("/")
async def root():
    """Health check endpoint with initialization status."""
    return {
        "message": "FastRTC Voice Assistant",
        "status": "running",
        "version": "1.0.0",
        "voice_assistant": {
            "initialized": components.is_initialized,
            "error": components.initialization_error
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if components.is_initialized else "initializing",
        "components": {
            "voice_assistant": components.voice_assistant is not None,
            "fastrtc_bridge": components.fastrtc_bridge is not None,
            "callback_handler": components.callback_handler is not None,
            "async_env_manager": components.async_env_manager is not None,
            "stream_mounted": components.stream is not None,
        },
        "initialization_error": components.initialization_error
    }

@app.get("/initialization/status")
async def initialization_status():
    """Check voice assistant initialization progress."""
    if components.is_initialized:
        return JSONResponse({
            "status": "complete",
            "message": "Voice assistant ready",
            "webrtc_endpoint": "/assistant"
        })
    elif components.initialization_error:
        return JSONResponse({
            "status": "failed",
            "error": components.initialization_error
        }, status_code=500)
    else:
        return JSONResponse({
            "status": "in_progress",
            "message": "Voice assistant initializing..."
        }, status_code=202)

@app.post("/debug/reinitialize")
async def debug_reinitialize(background_tasks: BackgroundTasks):
    """Force re-initialization of voice assistant (for debugging)."""
    if _initialization_task and not _initialization_task.done():
        return JSONResponse({
            "status": "error",
            "message": "Initialization already in progress"
        }, status_code=409)
    
    # Reset components
    with _components_lock:
        components.is_initialized = False
        components.initialization_error = None
    
    # Start new initialization
    background_tasks.add_task(initialize_voice_assistant_deferred, app)
    return {"status": "started", "message": "Re-initialization started in background"}

# Optional: serve frontend
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")

# Entry point
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FastRTC Voice Assistant Server...")
    print("📡 Server will be available at http://localhost:8000")
    print("🎤 WebRTC endpoint will be mounted at /assistant after initialization")
    print("💡 Check /health for component status")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
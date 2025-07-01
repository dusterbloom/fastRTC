"""
WhisperLive API endpoints for FastRTC integration.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ..integration.threading_callback_handler import ThreadingCallbackHandler
from ..utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/whisperlive", tags=["whisperlive"])

# Global reference to callback handler (will be set by main app)
callback_handler: ThreadingCallbackHandler = None

def set_callback_handler(handler: ThreadingCallbackHandler):
    """Set the global callback handler reference."""
    global callback_handler
    callback_handler = handler

@router.post("/start")
async def start_whisperlive():
    """Start WhisperLive microphone streaming."""
    if callback_handler is None:
        raise HTTPException(status_code=500, detail="Callback handler not initialized")
    
    try:
        result = callback_handler.start_whisperlive_streaming()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error starting WhisperLive: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_whisperlive():
    """Stop WhisperLive microphone streaming."""
    if callback_handler is None:
        raise HTTPException(status_code=500, detail="Callback handler not initialized")
    
    try:
        result = callback_handler.stop_whisperlive_streaming()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error stopping WhisperLive: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_whisperlive_status():
    """Get current WhisperLive streaming status."""
    if callback_handler is None:
        raise HTTPException(status_code=500, detail="Callback handler not initialized")
    
    try:
        return JSONResponse(content={
            "status": "active" if callback_handler.whisper_live_active else "inactive",
            "mode": callback_handler.whisperlive_mode,
            "last_transcription": callback_handler.last_transcription
        })
    except Exception as e:
        logger.error(f"Error getting WhisperLive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
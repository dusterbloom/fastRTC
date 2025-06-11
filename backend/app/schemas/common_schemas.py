from typing import Optional, List, Any, Dict # Added Dict
from pydantic import BaseModel, Field

# --- General ---
class UserSession(BaseModel):
    user_id: Optional[str] = Field(None, description="Unique identifier for the user.")
    session_id: Optional[str] = Field(None, description="Unique identifier for the conversation session.")

class ErrorResponse(BaseModel):
    detail: str

# --- STT ---
class STTRequest(BaseModel):
    # Audio will be handled by FastAPI's UploadFile or WebSocket binary frames
    language: Optional[str] = Field(None, description="Optional BCP-47 language code for STT hint.")

class STTResponse(BaseModel):
    text: str
    detected_language: Optional[str] = Field(None, description="Detected language code (e.g., Whisper/MediaPipe format).")
    kokoro_language: Optional[str] = Field(None, description="Detected language in Kokoro format.")
    # confidence: Optional[float] # If STT provides it

# --- TTS ---
class TTSRequest(BaseModel):
    text: str
    language: str = Field(description="Kokoro language code for TTS.") # e.g., 'a' for American English
    voice_id: Optional[str] = Field(None, description="Specific voice ID for the selected language.")
    # model: Optional[str] # If we want to allow choosing TTS model

# TTS Response will be StreamingResponse (audio/mpeg, audio/wav etc.) or binary WebSocket frames

# --- Converse (LLM Interaction) ---
class ConverseRequest(UserSession):
    text: str

class ConverseResponse(UserSession):
    response_text: str

# --- Language Detection ---
class LanguageDetectionRequest(BaseModel):
    text: str

class LanguageDetectionResponse(BaseModel):
    detected_language_code: Optional[str] = Field(None, description="Detected language (e.g., ISO 639-1).")
    kokoro_language_code: Optional[str] = Field(None, description="Kokoro format language code.")
    confidence: Optional[float] = Field(None, description="Confidence score of the detection.")

# --- Session Management ---
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None

class SessionResponse(UserSession):
    created_at: Optional[str] = None # ISO format datetime
    last_accessed: Optional[str] = None # ISO format datetime
    turn_count: Optional[int] = None
    # Potentially more stats from VoiceAssistant.get_system_stats()

class SessionDeleteResponse(BaseModel):
    message: str
    session_id: str

# --- Memory Management ---
class MemoryQueryRequest(UserSession):
    query_text: str
    # top_k: Optional[int] = 3

class MemoryQueryResultItem(BaseModel):
    text: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None # e.g., timestamp, source

class MemoryQueryResponse(BaseModel):
    results: List[MemoryQueryResultItem]
    summary: Optional[str] = None # Optional summary of results

class MemoryAddRequest(UserSession):
    user_text: str
    assistant_text: str
    # metadata: Optional[Dict[str, Any]] = None

class MemoryAddResponse(BaseModel):
    message: str
    entry_id: Optional[str] = None # ID of the added memory entry

class MemorySummaryResponse(UserSession):
    total_entries: Optional[int] = None
    last_updated: Optional[str] = None # ISO format datetime
    # Other relevant stats
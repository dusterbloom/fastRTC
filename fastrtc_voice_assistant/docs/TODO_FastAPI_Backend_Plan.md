# TODO: Plan for FastAPI Backend Integration with FastRTC Voice Assistant

This document outlines the plan to create a production-ready FastAPI backend that interfaces with the existing `fastrtc_voice_assistant` project and can serve a future frontend.

## 1. Goal

Develop a FastAPI backend providing a comprehensive API for the `fastrtc_voice_assistant` functionalities, supporting both HTTP and WebSocket communication for flexibility and real-time performance.

## 2. Core Functionalities to Expose

The backend will expose the following features:

*   **Speech-to-Text (STT)**: Transcribe audio to text.
*   **Text-to-Speech (TTS)**: Synthesize text into audio.
*   **LLM Interaction**: Get responses from the LLM, incorporating conversation history and agentic memory.
*   **Language Detection**: Detect language from text.
*   **Session Management**: Manage conversation sessions (create, retrieve info, reset/delete).
*   **Agentic Memory Access**: Query or interact with the agentic memory.

## 3. API Design

### 3.1. HTTP Endpoints

Standard RESTful endpoints for discrete operations.

*   **`/stt` (POST)**
    *   Request: Audio data (e.g., `UploadFile`), optional language.
    *   Response: `{"text": "...", "detected_language": "...", "kokoro_language": "..."}`
*   **`/tts` (POST)**
    *   Request: `{"text": "...", "language": "...", "voice_id": "..."}`
    *   Response: Audio stream (`StreamingResponse`) or audio file.
*   **`/converse` (POST)** (Primarily for non-streaming clients)
    *   Request: `{"text": "...", "session_id": "...", "user_id": "..."}`
    *   Response: `{"response_text": "...", "session_id": "...", "user_id": "..."}`
*   **`/detect_language` (POST)**
    *   Request: `{"text": "..."}`
    *   Response: `{"detected_language_code": "...", "kokoro_language_code": "...", "confidence": ...}`
*   **Session Management (`/session`)**:
    *   `POST /session`: Create new session. Req: `{"user_id": "..."}`. Resp: `{"session_id": "...", "user_id": "..."}`.
    *   `GET /session/{session_id}`: Get session info.
    *   `DELETE /session/{session_id}`: Reset/delete session.
*   **Agentic Memory Access (`/memory`)**:
    *   `POST /memory/query`: Query memory. Req: `{"query_text": "...", "user_id": "..."}`.
    *   `POST /memory/add`: Add entry. Req: `{"user_text": "...", "assistant_text": "...", "user_id": "..."}`.
    *   `GET /memory/summary`: Get memory summary. Req: `{"user_id": "..."}`.

### 3.2. WebSocket Endpoint (`/ws/converse`)

For real-time, full-duplex conversational flow.

*   **Connection Initiation**: Client sends `auth` message with API key, user/session info, expected audio format, and sample rate.
*   **Client-to-Server Messages**:
    *   `auth` (JSON): Initialize connection.
    *   *Binary Audio Chunks*: Raw audio data for STT (format/rate specified in `auth`).
    *   `audio_complete` (JSON): Signal end of user speech.
    *   `text_input` (JSON): For typed user messages.
*   **Server-to-Client Messages**:
    *   `session_started` (JSON): Confirm auth, provide session details, and expected TTS audio format/rate.
    *   `stt_partial` (JSON): Intermediate transcription.
    *   `stt_final` (JSON): Final transcription.
    *   `llm_response` (JSON): Assistant's text reply.
    *   *Binary Audio Chunks*: Raw TTS audio data.
    *   `tts_complete` (JSON): Signal end of TTS audio.
    *   `error` (JSON): Error reporting.

## 4. Authentication

*   API key authentication for all HTTP endpoints (e.g., via an `X-API-Key` header).
*   For WebSockets, the API key will be sent in the initial `auth` message.

## 5. Directory Structure

A new `backend/` directory will be created at `fastRTC/backend/`.

```
fastRTC/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app, lifespan events
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── http_routes.py
│   │   │   └── ws_routes.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── assistant_service.py # Manages VoiceAssistant instances
│   │   ├── schemas/               # Pydantic models
│   │   │   └── ...
│   │   ├── core/                  # Backend-specific core (auth, config)
│   │   │   └── ...
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── fastrtc_voice_assistant/       # Existing project
│   ├── src/
│   │   └── core/
│   │       └── voice_assistant.py # To be used by backend
│   └── ...
└── frontend/
```

## 6. Core Logic Integration & State Management

*   The `backend/app/services/assistant_service.py` will manage `VoiceAssistant` instances from `fastrtc_voice_assistant.src.core.voice_assistant`.
*   **WebSocket**: A new `VoiceAssistant` instance will be created per WebSocket connection, initialized with `user_id` and `session_id` from the client's `auth` message. `initialize_async()` and `cleanup_async()` will be tied to the WebSocket lifecycle.
*   **HTTP**: `VoiceAssistant` instances will be created/retrieved based on `session_id` (passed in requests). A caching mechanism with eviction (e.g., LRU with timeout) will manage these instances in `AssistantService`.
*   **`VoiceAssistant` Class Modifications**:
    *   `__init__`: Modify to accept `user_id` and `session_id` as parameters, instead of auto-generating or hardcoding them.
    *   Consider adapting STT engine for chunk-based input if not already supported, for smoother WebSocket integration.
    *   (Potential Optimization) Modify `VoiceAssistant` or `LLMService` to accept a shared `aiohttp.ClientSession` to reduce resource overhead.

## 7. Docker Integration

*   A new `backend` service will be added to `fastrtc_voice_assistant/docker-compose.yml`.
*   This service will build using `backend/Dockerfile`.
*   **Volume Mounts**: `fastrtc_voice_assistant/src` will be mounted into the `backend` container (e.g., at `/app/fastrtc_voice_assistant_src`).
*   **`PYTHONPATH`**: Will be set in `backend/Dockerfile` to include the mounted `fastrtc_voice_assistant_src` directory, allowing the backend to import modules from the existing project.
*   The existing `voice-assistant` (Gradio) service in `docker-compose.yml` will depend on the new `backend` service.

## 8. Configuration Management

*   The `backend` service in `docker-compose.yml` will use `env_file` to load configuration variables from the existing `fastrtc_voice_assistant/.env.development` file (or a new shared `.env` file at the project root).
*   This ensures consistency for settings like database URLs, model paths, etc., used by `VoiceAssistant`.

## 9. Client Adaptation (Gradio/FastRTC)

*   The existing Gradio application (primarily `start.py`, and potentially `gradio3.py` if it's to be maintained as a client) will be modified.
*   Its callback (`voice_assistant_callback_rt`) will no longer directly instantiate or call `VoiceAssistant`. Instead, it will:
    *   Establish a WebSocket connection to the `backend` service's `/ws/converse` endpoint.
    *   Send audio chunks received from FastRTC over the WebSocket.
    *   Receive STT text, LLM responses, and TTS audio chunks from the backend via WebSocket.
    *   Yield received TTS audio chunks back to the FastRTC bridge for playback.

## 10. Implementation Steps (High-Level TODOs)

1.  [ ] **Setup `backend/` Directory**: Create the proposed directory structure.
2.  [ ] **Develop `backend/app/core/config.py`**: Load environment variables.
3.  [ ] **Develop `backend/app/core/auth.py`**: Implement API key authentication scheme.
4.  [ ] **Define `backend/app/schemas/`**: Create Pydantic models for API requests/responses and WebSocket messages.
5.  [ ] **Modify `fastrtc_voice_assistant.src.core.voice_assistant.VoiceAssistant`**:
    *   [ ] Update `__init__` to accept `user_id`, `session_id`.
    *   [ ] (Optional) Adapt to use a shared `aiohttp.ClientSession`.
    *   [ ] Ensure STT can handle streamed/chunked input for WebSockets.
6.  [ ] **Develop `backend/app/services/assistant_service.py`**:
    *   [ ] Implement `VoiceAssistant` instance management for HTTP and WebSockets.
    *   [ ] Implement `startup` and `shutdown` methods for resource management.
    *   [ ] Implement wrapper methods for STT, TTS, Converse functionalities.
7.  [ ] **Develop HTTP Routes (`backend/app/api/http_routes.py`)**:
    *   [ ] Implement all defined HTTP endpoints, using `AssistantService`.
    *   [ ] Integrate API key authentication.
8.  [ ] **Develop WebSocket Route (`backend/app/api/ws_routes.py`)**:
    *   [ ] Implement the `/ws/converse` endpoint.
    *   [ ] Handle WebSocket lifecycle, `auth` message, binary audio, and JSON control messages.
    *   [ ] Use `AssistantService` to interact with the dedicated `VoiceAssistant` instance.
9.  [ ] **Develop `backend/app/main.py`**:
    *   [ ] Instantiate FastAPI app.
    *   [ ] Include API routers.
    *   [ ] Implement lifespan events for `AssistantService` startup/shutdown.
10. [ ] **Create `backend/requirements.txt`**: List FastAPI, Uvicorn, etc.
11. [ ] **Create `backend/Dockerfile`**.
12. [ ] **Update `fastrtc_voice_assistant/docker-compose.yml`**: Add the `backend` service and update the existing `voice-assistant` (Gradio) service.
13. [ ] **Refactor Gradio Application (`start.py` and potentially `gradio3.py`)**:
    *   [ ] Modify to connect to the backend's WebSocket endpoint.
    *   [ ] Remove direct `VoiceAssistant` usage.
14. [ ] **Testing**: Thoroughly test all HTTP endpoints and the WebSocket conversational flow.
    *   [ ] Unit tests for backend components.
    *   [ ] Integration tests for API endpoints.
    *   [ ] End-to-end tests with Gradio client.

---

## 11. Pydantic Schema Outline

This section outlines the main Pydantic models for request/response validation and WebSocket messaging. These will primarily reside in `backend/app/schemas/`.

### File: `backend/app/schemas/common_schemas.py` (or similar)

```python
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
```

### File: `backend/app/schemas/ws_schemas.py`

```python
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field

# --- WebSocket General ---
class WSBaseMessage(BaseModel):
    type: str

# --- Client to Server WebSocket Messages ---
class WSAuthRequest(WSBaseMessage):
    type: Literal["auth"] = "auth"
    api_key: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None # To resume a session
    audio_format: Optional[str] = Field("float32", description="Format of upcoming binary audio frames.")
    sample_rate: Optional[int] = Field(16000, description="Sample rate of upcoming binary audio.")

class WSAudioComplete(WSBaseMessage):
    type: Literal["audio_complete"] = "audio_complete"

class WSTextInput(WSBaseMessage):
    type: Literal["text_input"] = "text_input"
    text: str

# Union type for C2S JSON messages
ClientToServerJSONMessage = Union[WSAuthRequest, WSAudioComplete, WSTextInput]


# --- Server to Client WebSocket Messages ---
class WSSessionStarted(WSBaseMessage):
    type: Literal["session_started"] = "session_started"
    session_id: str
    user_id: str
    expected_tts_audio_format: Optional[str] = Field("float32", description="Format of upcoming binary TTS audio.")
    expected_tts_sample_rate: Optional[int] = Field(16000, description="Sample rate of upcoming binary TTS audio.")

class WSSTTPartial(WSBaseMessage):
    type: Literal["stt_partial"] = "stt_partial"
    text: str

class WSSTTFinal(WSBaseMessage):
    type: Literal["stt_final"] = "stt_final"
    text: str
    detected_language: Optional[str] = None
    kokoro_language: Optional[str] = None

class WSLLMResponse(WSBaseMessage):
    type: Literal["llm_response"] = "llm_response"
    text: str

class WSTTSComplete(WSBaseMessage):
    type: Literal["tts_complete"] = "tts_complete"

class WSError(WSBaseMessage):
    type: Literal["error"] = "error"
    message: str
    details: Optional[str] = None

# Union type for S2C JSON messages
ServerToClientJSONMessage = Union[
    WSSessionStarted, WSSTTPartial, WSSTTFinal, WSLLMResponse, WSTTSComplete, WSError
]
```

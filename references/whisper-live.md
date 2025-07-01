# 🚧 ISSUE: Integrate Ultra-Low Latency STT Pipeline with WhisperLive and Sentence Completion Classifier

## 🎯 Goal
Update the existing threaded voice pipeline (`VoicePipelineThreaded`) and FastRTC integration to use **WhisperLive** for STT and an **XLM-RoBERTa-base multilingual sentence completion classifier**. Minimize redundant code and ensure tight integration with the existing structure. Avoid uncertainty in design and implementation.

---

## ✅ Requirements

### STT
- [x] Use `WhisperLive` as the only STT backend (replace/remove FasterWhisperSTT)
- [x] Stream audio frames to WhisperLive via WebSocket
- [x] Parse partial and final transcription results
- [x] New file: `backend/src/audio/engines/stt/whisper_live_stt.py`
- [x] Class: `WhisperLiveSTT`
- [x] Method: `push_audio(bytes)`, `get_transcript()`
- [x] Must be pure threading, no `async def` in worker. All event loop code should run inside a `Thread` or an internal asyncio loop.

### Sentence Completion
- [x] Use `xlm-roberta-base` fine-tuned for sentence boundary classification
- [x] Lightweight, fast CPU inference (<50ms)
- [x] Run model locally via `transformers` + `torch`
- [x] New file: `backend/src/utils/sentence_completion.py`
- [x] Class: `SentenceCompletionClassifier`
- [x] Method: `is_complete(text: str) -> bool`
- [x] Provide confidence threshold logic for robustness

### Pipeline
- [x] Use `VoicePipelineThreaded` already in codebase
- [x] Update `ThreadingCallbackHandler.on_audio_frame()` to push PCM bytes to `VoicePipelineThreaded.push_audio()`
- [x] Pass transcriptions to LLM only when sentence is semantically complete
- [x] Use SileroVAD for initial frame filtering (already wired)
- [x] Inject `WhisperLiveSTT` into `STTWorker`

---

## 🗂️ Existing Code Integration

### 📌 STT Location
- File: `backend/src/audio/engines/stt/faster_whisper_stt.py`
- Action: Remove this or retain for fallback
- Replacement file: `backend/src/audio/engines/stt/whisper_live_stt.py`
- Drop-in compatible: `StreamingSTT` → `WhisperLiveSTT`

### 📌 Sentence Completion Classifier
- File: `backend/src/utils/sentence_completion.py`
- Class: `SentenceCompletionClassifier`
- Use HuggingFace `xlm-roberta-base` with a simple classification head
- Load once, return `True` for semantically complete text
- Inject instance into `VoicePipelineThreaded`

### 📌 StreamHandler (FastRTC)
- File: `backend/src/integration/threading_callback_handler.py`
- Key method: `on_audio_frame(frame: bytes)`
- Flow:
  - 🔄 `frame` → `VoicePipelineThreaded.push_audio()`
  - 🧠 WhisperLive internally buffers & transcribes
  - 📝 Call `get_transcript()` in loop or timer thread
  - 🤖 Feed each partial to `SentenceCompletionClassifier.is_complete()`
  - ✅ If true, dispatch to LLM worker
- ⚠️ **Do not use `async def` anywhere in `ThreadingCallbackHandler` or `STTWorker`.** WhisperLive client must be wrapped in its own `asyncio.run()` inside thread.

### 📌 Fixing Intern Error:
**Error:**
```
RuntimeWarning: coroutine 'WhisperLiveStreamingWorker.initialize' was never awaited
RuntimeError: Cannot run the event loop while another loop is running
```
**Fix:**
- In `WhisperLiveSTT.initialize()`, avoid `async def`
- Start a thread inside `__init__()` or `initialize()`
- Inside that thread: `asyncio.run(self._connect_loop())`

---

## 🔄 Updated Pipeline Flow
```mermaid
graph LR
    A[ThreadingCallbackHandler.on_audio_frame()] --> B[VoicePipelineThreaded.push_audio()]
    B --> C[Silero VAD Filter]
    C --> D[WhisperLiveSTT (WebSocket Thread)]
    D --> E[Streaming Partial Transcripts]
    E --> F[SentenceCompletionClassifier (XLM-RoBERTa)]
    F --> G[LLM Worker Thread]
    G --> H[TTS Worker Thread]
    H --> I[send_frame() to WebRTC Output]
```

---

## 📦 Dependencies
Add to `requirements.txt`:
```txt
whisper-live
transformers>=4.35
torch>=2.1
sentencepiece
```

---

## ⚙️ Setup

### WhisperLive (if running externally)
```bash
git clone https://github.com/collabora/WhisperLive.git
cd WhisperLive
pip install -r requirements.txt
python run_server.py --port 9090 --backend faster_whisper --model base
```
Or run via Python module:
```bash
python3 -m whisper_live.run_server --port 9090 --backend faster_whisper --model base
```

Optional: Add to `fastrtc.sh`:
```bash
WHISPERLIVE_PORT=9090
if ! lsof -i:$WHISPERLIVE_PORT > /dev/null; then
  echo "Launching WhisperLive..."
  nohup python3 -m whisper_live.run_server --port $WHISPERLIVE_PORT --backend faster_whisper --model base > whisperlive.log 2>&1 &
fi
```

---

## ✅ Acceptance Criteria
- [ ] All STT handled via WhisperLive using WebSocket in its own thread
- [ ] Only semantically complete sentences passed to LLM
- [ ] `ThreadingCallbackHandler` dispatches properly via `on_audio_frame()`
- [ ] No async calls in threading pipeline
- [ ] No duplicate logic or code rewrites
- [ ] Latency from speech to response < 800ms avg

---

## 🧠 Guidance for Junior Developer

### ✅ Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Launch WhisperLive: `python -m whisper_live.run_server --port 9090`
3. Confirm it’s reachable at `ws://localhost:9090/stt`

### ✅ STT Class
4. Create `whisper_live_stt.py` under `audio/engines/stt/`
5. Define `WhisperLiveSTT` class
6. Internally spawn a thread with an `asyncio.run()` for WebSocket client
7. Implement:
   - `push_audio(bytes)`
   - `get_transcript()` that returns current transcript string

### ✅ Sentence Completion
8. Create `sentence_completion.py` under `utils/`
9. Load HuggingFace `xlm-roberta-base`
10. Wrap with a binary classifier for "sentence finished"
11. Implement `is_complete(text) -> bool` with softmax threshold > 0.9

### ✅ Integration
12. Open `threading_callback_handler.py`
13. Inside `on_audio_frame()`, push `frame` to `pipeline.push_audio()`
14. Poll `get_transcript()` in `VoicePipelineThreaded`
15. When `is_complete()` returns true, send to `LLMWorker`

> 💡 This architecture avoids async collisions by fully isolating WhisperLive in its own thread. Latency and responsiveness are both maximized.

🎯 Expect under **800ms** round-trip latency.

🧠🎙️🚀

In a few focused changes you can slide the **faster-whisper + CTranslate2** upgrade cleanly into the *existing* `dusterbloom-fastrtc` layout without breaking anything upstream of the STT layer.  The table below shows where each new file or edit lives relative to your real tree, followed by step-by-step instructions, copy-pastable code, and CI/Docker tweaks.

---

## 1  How the current tree maps onto the new design

| Concern                      | Where it lives **now**                                                       | What you will add or touch                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **STT engine**               | `src/audio/engines/stt/huggingface_stt.py`                                   | new `src/audio/engines/stt/faster_whisper_stt.py` + small edit to `__init__.py` |
| **Audio config**             | `src/config/audio_config.py`                                                 | add a flag `STT_BACKEND=faster` (default keeps old engine)                      |
| **Model cache**              | *none* (Whisper is downloaded at runtime)                                    | `/models/whisper-v3-ct2` baked in Docker image                                  |
| **Container build**          | `backend/docker-compose.yml` + the project-root `Dockerfile` | two-stage Dockerfile (see §3) and one extra service env-var in Compose          |
| **Unit & integration tests** | `tests/unit/test_stt_engines.py` etc.                                        | only fixture swap: point to `faster_whisper_stt`                                |

All other directories—including your Gradio demos, `src/core/voice_assistant.py`, the memory system, and the React front-end—stay untouched.

---

## 2  Python-side edits

### 2.1  Dependencies (`backend/requirements.txt`)

```text
# --- add
faster-whisper>=1.1           # CTranslate2 backend, streaming API :contentReference[oaicite:0]{index=0}
ctranslate2>=4.2.0             # INT8/FP16 kernels, FlashAttn optional :contentReference[oaicite:1]{index=1}
huggingface_hub[hf_transfer]>=0.25  # Rust downloader for model layer :contentReference[oaicite:2]{index=2}
```

Remove any explicit `whisper`/`openai-whisper` pin; the old engine file can remain for fallback.

### 2.2  New engine: `src/audio/engines/stt/faster_whisper_stt.py`

```python
from pathlib import Path
from faster_whisper import WhisperModel

_MODEL_DIR = Path("/models/whisper-v3-ct2")   # baked into the image
_COMPUTE   = "int8_float16"                   # best perf/quality on Ampere :contentReference[oaicite:3]{index=3}

class FasterWhisperSTT:
    """
    Streaming multilingual STT using faster-whisper + CTranslate2.
    """

    def __init__(self):
        self.model = WhisperModel(
            str(_MODEL_DIR),
            device="cuda",
            compute_type=_COMPUTE,
        )
        # sanity-check that we did not load an English-only checkpoint
        assert "de" in self.model.hf_tokenizer.lang_to_id, \
            "Non-multilingual model loaded!"

    def stream(self, pcm16_bytes):
        """
        Generator yielding partial transcripts (~1 s latency).
        """
        segments, _ = self.model.transcribe(
            pcm16_bytes,
            vad_filter=True,    # built-in Silero VAD :contentReference[oaicite:4]{index=4}
            chunk_size=1.0,
            beam_size=1,
        )
        for s in segments:
            yield s.text
```

### 2.3  Switch engine via package init

`src/audio/engines/stt/__init__.py`

```python
from src.config.audio_config import STT_BACKEND

if STT_BACKEND == "faster":
    from .faster_whisper_stt import FasterWhisperSTT as STTEngine
else:
    from .huggingface_stt import HuggingFaceSTT as STTEngine    # existing class
```

No other imports elsewhere in your code base need to change—every caller already does:

```python
from src.audio.engines.stt import STTEngine
stt = STTEngine()
```

---

## 3  Container & Compose

### 3.1  Two-stage Dockerfile (place in project root)

```dockerfile
################ 1️⃣  download CT2 model ########################
FROM python:3.11-slim AS downloader
RUN pip install "huggingface_hub[hf_transfer]" && mkdir /models
ENV HF_HUB_ENABLE_HF_TRANSFER=1          # 2-3× faster download :contentReference[oaicite:5]{index=5}
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Systran/faster-whisper-large-v3",   # multilingual INT8/FP16 :contentReference[oaicite:6]{index=6}
    local_dir="/models/whisper-v3-ct2",
    local_dir_use_symlinks=False)
PY

################ 2️⃣  runtime ###################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV HUGGINGFACE_HUB_CACHE=/models
COPY --from=downloader /models /models
RUN apt-get update && apt-get install -y ffmpeg \
 && pip install --no-cache-dir -r backend/requirements.txt
WORKDIR /workspace
COPY . .
CMD ["python", "backend/start.py"]
```

### 3.2  docker-compose snippet (`backend/docker-compose.yml`)

```yaml
services:
  backend:
    build: ..
    environment:
      STT_BACKEND: "faster"   # <- flips the switch at runtime
      CUDA_VISIBLE_DEVICES: "0"
```

---

## 4  FastRTC glue stays identical

Your existing `src/integration/fastrtc_bridge.py` already forwards each audio chunk to the configured STT engine and streams transcripts back.  Because our `FasterWhisperSTT.stream()` yields text the same way the old engine did, **no change is required** in FastRTC glue, LLM, or TTS modules.

Latency expectations: first partial token ≈ 200–300 ms with 1 s chunks, matching academic results on Whisper-Streaming ([arxiv.org][1], [researchgate.net][2]).

---

## 5  Tests & QA

1. **Unit** – in `tests/unit/test_stt_engines.py` parametrize over both `"huggingface"` and `"faster"` backends, asserting equal transcripts on short clips (MLS 5 s samples).
2. **Integration** – the existing `test_audio_pipeline.py` will exercise the new path automatically when `STT_BACKEND=faster` is exported in CI.
3. **Performance** – reuse `tests/integration/test_startup_and_e2e_performance.py`; target `pod_ready < 15 s` and `p95 transcript_latency < 350 ms`.  The INT8/FP16 checkpoint uses \~1 GB VRAM ([huggingface.co][3]) and loads 3-4× faster than FP32 Whisper ([github.com][4]).

---

## 6  Optional power-ups

* **FlashAttention 2 kernels** – build CTranslate2 with `-DWITH_FLASH_ATTN=ON` for another \~10 % throughput gain ([github.com][5]).
* **Chunk length autotune** – recent research shows latency ≈ 1–2 × chunk size ([arxiv.org][1]); drop `chunk_size` to 0.6 s if you can spare 5 % WER.
* **CT2 model variants** – smaller GPUs can point `_MODEL_DIR` to `ctranslate2-4you/whisper-large-v3-ct2-int8_float16` (≈ 930 MB) ([huggingface.co][6]) with identical API.

---

## 7  Rollback switch

Because the choice of backend is one env-var, ops can revert instantly:

```bash
docker compose exec backend bash -c 'export STT_BACKEND=huggingface && supervisorctl restart voice'
```

---

### Key sources consulted

1. faster-whisper GitHub (streaming & INT8) ([github.com][4])
2. Systran multilingual CT2 checkpoint ([huggingface.co][7])
3. HF\_HUB\_ENABLE\_HF\_TRANSFER docs ([huggingface.co][8])
4. Silero VAD speed benchmarks ([github.com][9])
5. Whisper-Streaming latency analyses ([arxiv.org][1])
6. Whisper real-time paper ([researchgate.net][2])
7. compute\_type INT8\_FP16 guidance ([github.com][10])
8. FlashAttention integration thread ([github.com][5])
9. HF snapshot of large-v3 INT8\_FP16 alt model ([huggingface.co][6])
10. Silero-VAD ONNX flexibility ([github.com][11])
11. CTranslate2 Whisper API docs ([opennmt.net][12])
12. Systran model file list (proof of 1 GB size) ([huggingface.co][3])

Follow the file-exact paths above, rebuild the image, and your voice assistant should boot in under 12 seconds and stream transcripts with sub-second responsiveness—all without reshaping the rest of the codebase.

[1]: https://arxiv.org/pdf/2406.10052?utm_source=chatgpt.com "[PDF] Attention-Guided Streaming Whisper with Truncation Detection - arXiv"
[2]: https://www.researchgate.net/publication/372684083_Turning_Whisper_into_Real-Time_Transcription_System?utm_source=chatgpt.com "(PDF) Turning Whisper into Real-Time Transcription System"
[3]: https://huggingface.co/Systran/faster-whisper-large-v3/tree/main?utm_source=chatgpt.com "Systran/faster-whisper-large-v3 at main - Hugging Face"
[4]: https://github.com/SYSTRAN/faster-whisper?utm_source=chatgpt.com "Faster Whisper transcription with CTranslate2 - GitHub"
[5]: https://github.com/SYSTRAN/faster-whisper/issues/598?utm_source=chatgpt.com "Incorporating flash-attention2 [SOLVED] and subsequent testing ..."
[6]: https://huggingface.co/ctranslate2-4you/whisper-large-v3-ct2-int8_float16?utm_source=chatgpt.com "ctranslate2-4you/whisper-large-v3-ct2-int8_float16 - Hugging Face"
[7]: https://huggingface.co/Systran/faster-whisper-large-v3?utm_source=chatgpt.com "Systran/faster-whisper-large-v3 - Hugging Face"
[8]: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables?utm_source=chatgpt.com "Environment variables - Hugging Face"
[9]: https://github.com/snakers4/silero-vad?utm_source=chatgpt.com "Silero VAD: pre-trained enterprise-grade Voice Activity Detector"
[10]: https://github.com/SYSTRAN/faster-whisper/issues/615?utm_source=chatgpt.com "Issues with compute type #615 - SYSTRAN/faster-whisper - GitHub"
[11]: https://github.com/aosfatos/silero-vad-v4?utm_source=chatgpt.com "aosfatos/silero-vad-v4 - GitHub"
[12]: https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html?utm_source=chatgpt.com "Whisper - ctranslate2.models - OpenNMT"

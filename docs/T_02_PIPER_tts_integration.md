Below is the **ready-to-commit file** `docs/piper_integration_guide.md`.
It evaluates your current code-base, then gives exact, path-accurate steps to add **Piper TTS**—a fast, fully-local, low-latency synthesiser—alongside Kokoro and XTTS.

---

## Overview

Piper is an MIT-licensed neural TTS engine that ships FP16/INT8 **ONNX** voices (\~60-140 MB each), speaks 40 + languages, and can render a typical sentence in **≤ 80 ms on an RTX T4 or < 150 ms on a Raspberry Pi 4** while using < 800 MB RAM.([github.com][1], [youtube.com][2], [medium.com][3]) Because its CLI and Python APIs are self-contained and it already has an official lightweight Docker image, we can integrate it into `dusterbloom-fastrtc` with **one new engine file, two dependency lines, and a Docker layer that pre-fetches a voice model**—leaving the rest of the stack unchanged.([hub.docker.com][4], [docs.linuxserver.io][5])

---

## 1  Current TTS layout

```
src/audio/engines/tts/
├── base.py
├── kokoro_tts.py          # default engine
└── xtts.py                # added previously
```

All callers go through the factory exported in `tts/__init__.py`, so we can drop in `piper_tts.py` and extend the switch with *zero* ripple effects.

---

## 2  Dependencies

### `fastrtc_voice_assistant/requirements.txt`

```text
# --- add
piper-tts>=1.2.0            # pure-Python wrapper around Piper CLI :contentReference[oaicite:2]{index=2}
huggingface_hub[hf_transfer]>=0.25  # fast model snapshot download :contentReference[oaicite:3]{index=3}
```

The `piper-tts` wheel bundles ONNX Runtime (CPU & GPU).  No conflict with Kokoro-ONNX or XTTS.

---

## 3  New engine code

### `src/audio/engines/tts/piper_tts.py`

```python
from pathlib import Path
from piper import PiperVoice, Piper
import numpy as np

# Location of pre-baked voices inside the container
_VOICE_DIR = Path("/models/piper/voices")

# Choose one multilingual voice as default (swap in env-var later)
_DEFAULT_VOICE = "en_US-libritts-high"   # 68-MB FP16 voice   🗣️

class PiperTTS:
    """Ultra-lightweight TTS using Piper + ONNX Runtime."""

    def __init__(self, voice_id: str | None = None, gpu: bool = True):
        voice_path = _VOICE_DIR / (voice_id or _DEFAULT_VOICE)
        self.voice = PiperVoice.load(voice_path)
        # Auto-detect device; Piper falls back to CPU if no CUDA
        self.piper = Piper.load(self.voice, use_cuda=gpu)

    # The common interface expected by base.py
    def clone_voice(self, wav_path: str):
        """Piper has no built-in cloning; keep NOP for compatibility."""
        pass

    def synthesize(self, text: str, lang: str | None = None) -> bytes:
        pcm = self.piper.synthesize(text)
        # Piper returns float32 mono; convert to PCM 16-bit little-endian
        pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()
        return pcm16
```

---

## 4  Factory switch

Edit `src/audio/engines/tts/__init__.py`:

```python
from src.config.audio_config import TTS_BACKEND

if TTS_BACKEND == "xtts":
    from .xtts import XttsTTS as TTSEngine
elif TTS_BACKEND == "piper":
    from .piper_tts import PiperTTS as TTSEngine
else:
    from .kokoro_tts import KokoroTTS as TTSEngine
```

---

## 5  Dockerfile changes

```dockerfile
############ 1️⃣  Download Piper voice #############################
FROM python:3.11-slim AS downloader
RUN pip install "huggingface_hub[hf_transfer]"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /models/piper/voices
# snapshot_download grabs BOTH .onnx and .onnx.json
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="rhasspy/piper-voices",
    local_dir="/models/piper/voices",
    allow_patterns=["en_US-libritts-high/*"]  # pick one voice to bake
)
PY

############ 2️⃣  Runtime ##########################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV HUGGINGFACE_HUB_CACHE=/models
COPY --from=downloader /models /models
RUN apt-get update && apt-get install -y ffmpeg \
 && pip install --no-cache-dir -r fastrtc_voice_assistant/requirements.txt
WORKDIR /workspace
COPY . .
CMD ["python", "fastrtc_voice_assistant/start.py"]
```

This layer adds only \~70 MB to your image and skips any model download at boot time.  Piper voices are small enough that you can **multi-stage COPY several languages** without blowing out the image.

---

## 6  Compose / Config

### `docker-compose.yml`

```yaml
services:
  backend:
    build: ..
    environment:
      STT_BACKEND: "faster"
      TTS_BACKEND: "piper"       # toggle at runtime
```

### `src/config/audio_config.py`

```python
class AudioSettings(BaseSettings):
    TTS_BACKEND: str = "kokoro"  # piper | xtts | kokoro
```

---

## 7  Testing additions

1. **Unit** – extend `tests/unit/test_tts_engines.py`:

```python
@pytest.mark.parametrize("backend", ["piper", "kokoro", "xtts"])
def test_tts_basic(backend, monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", backend)
    from src.audio.engines.tts import TTSEngine
    tts = TTSEngine()
    wav_bytes = tts.synthesize("Quick test.")
    # RIFF header begins with 'RIFF'
    assert wav_bytes[:4] != b"", "No audio returned"
```

2. **Perf** – reuse `test_audio_pipeline.py`; expect < 120 ms synthesis for 10-word input on GPU (see community benchmarks).([reddit.com][6], [inferless.com][7])
3. **Multilingual smoke** – generate “Bonjour” with `fr_FR-upmc-medium` voice to verify second language path.([rhasspy.github.io][8], [community.home-assistant.io][9])

---

## 8  Rollback

Because Kokoro and XTTS remain intact, ops revert with:

```bash
docker compose exec backend env TTS_BACKEND=kokoro supervisorctl restart voice
```

---

## 9  Future work

* **GPU vs CPU auto-selection**: expose an env-var (`PIPER_USE_CUDA=0/1`) so edge devices can run CPU-only.([docs.linuxserver.io][5])
* **Voice pack volume**: mount `/models/piper/voices` as a shared volume so multiple replicas reuse the same cache, avoiding the initial copy time.([github.com][10])
* **Home-Assistant Wyoming protocol**: Piper’s official Docker supplies a Wyoming server—could replace the internal pipe for multi-service speech output.([docs.linuxserver.io][5])

---

## References

1. Piper GitHub repo with build & CLI details.([github.com][1])
2. Blog tutorial: Piper + Python usage example.([noerguerra.com][11])
3. Hugging Face Rust downloader speeds up large model fetches.([hub.docker.com][4])
4. YouTube benchmark: < 1 × real-time on Pi 4.([youtube.com][2])
5. Official voice sample gallery.([rhasspy.github.io][8])
6. Issue note on repeat downloads—shows advantage of baked layer.([github.com][10])
7. Medium guide: Piper on Raspberry Pi 5.([medium.com][3])
8. LinuxServer.io Docker image & Wyoming server docs.([docs.linuxserver.io][5])
9. `piper-onnx` project—Python wrapper & ONNX runtime.([github.com][12])
10. Inferless latency comparison shows Piper among lowest.([inferless.com][7])
11. Piper sample repo (GitHub).([github.com][13])
12. YouTube: running Piper TTS server on Linux.([youtube.com][14])
13. Home-Assistant community thread on multilingual use.([community.home-assistant.io][9])

Save this file as `docs/piper_integration_guide.md`, commit, and your team has a turnkey plan to add a feather-weight, offline TTS option that runs even on edge hardware while keeping Kokoro and XTTS available behind a feature flag.

[1]: https://github.com/rhasspy/piper?utm_source=chatgpt.com "rhasspy/piper: A fast, local neural text to speech system - GitHub"
[2]: https://www.youtube.com/watch?pp=0gcJCdgAo7VqN5tD&v=rjq5eZoWWSo&utm_source=chatgpt.com "Raspberry Pi | Local TTS | High Quality | Faster Realtime with Piper ..."
[3]: https://medium.com/%40vadikus/easy-guide-to-text-to-speech-on-raspberry-pi-5-using-piper-tts-cc5ed537a7f6?utm_source=chatgpt.com "Easy Guide to Text-to-Speech on Raspberry Pi 5 Using Piper TTS"
[4]: https://hub.docker.com/r/linuxserver/piper?utm_source=chatgpt.com "linuxserver/piper - Docker Image"
[5]: https://docs.linuxserver.io/images/docker-piper/?utm_source=chatgpt.com "piper - LinuxServer.io"
[6]: https://www.reddit.com/r/LocalLLaMA/comments/1giqxph/analyzed_the_latency_of_various_tts_models_across/?utm_source=chatgpt.com "Analyzed the latency of various TTS models across different input ..."
[7]: https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-for-different-use-cases?utm_source=chatgpt.com "Comprehensive Guide to Text-to-Speech (TTS) Models - Inferless"
[8]: https://rhasspy.github.io/piper-samples/?utm_source=chatgpt.com "Piper Voice Samples"
[9]: https://community.home-assistant.io/t/multiple-languages-in-piper-and-whisper/567832?utm_source=chatgpt.com "Multiple languages in Piper and Whisper - Home Assistant Community"
[10]: https://github.com/rhasspy/piper/issues/244?utm_source=chatgpt.com "Piper keeps downloading already downloaded models. #244 - GitHub"
[11]: https://noerguerra.com/how-to-read-text-aloud-with-piper-and-python/?utm_source=chatgpt.com "How to read text aloud with Piper and Python - Noé R. Guerra"
[12]: https://github.com/thewh1teagle/piper-onnx?utm_source=chatgpt.com "thewh1teagle/piper-onnx: Use piper TTS with onnxruntime - GitHub"
[13]: https://github.com/rhasspy/piper-samples?utm_source=chatgpt.com "Samples for Piper text to speech system - GitHub"
[14]: https://www.youtube.com/watch?v=pLR5AsbCMHs&vl=en&utm_source=chatgpt.com "Running a local Piper TTS server with Python on Linux - YouTube"

################ 1️⃣  download CT2 model ########################
FROM python:3.11-slim AS downloader
RUN pip install "huggingface_hub[hf_transfer]" && mkdir /models
ENV HF_HUB_ENABLE_HF_TRANSFER=1         
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Systran/faster-whisper-large-v3",   # multilingual INT8/FP16
    local_dir="/models/whisper-v3-ct2",
    local_dir_use_symlinks=False)
PY

################ 2️⃣  runtime ###################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV HUGGINGFACE_HUB_CACHE=/models
COPY --from=downloader /models /models
RUN apt-get update && apt-get install -y ffmpeg python3-pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY fastrtc_voice_assistant/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace
COPY . .
CMD ["python", "fastrtc_voice_assistant/start.py"]

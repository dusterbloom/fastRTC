# NOTE: During build, outbound HTTPS access to huggingface.co is required to download models.
# No other external network access is required by default.
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
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace
COPY . .
CMD ["python3", "fastrtc_voice_assistant/start.py"]

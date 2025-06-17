# Optimized Dockerfile - Models downloaded at runtime to persistent volume
# Build time: ~2-3 minutes instead of 10-15 minutes
################ 🚀 Fast Runtime Image ########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV HUGGINGFACE_HUB_CACHE=/models
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy application code
COPY . .
CMD ["python3", "start_clean.py"]

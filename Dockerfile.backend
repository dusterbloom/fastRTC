# Dockerfile.backend (project root) - Backend container with WebRTC support
FROM python:3.11-slim

# Install system dependencies for WebRTC
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY run.py .

# Create configuration directory
RUN mkdir -p /app/config/webrtc

# Expose port (when not using host networking)
EXPOSE 8000

CMD ["python", "run.py"]

# Use an official Python runtime as a parent image
# Try ubuntu-based Python if slim fails
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:/app/shared"

# Install system dependencies required for audio processing and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    # Audio processing dependencies
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    # For faster numpy/scipy builds
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend application code
COPY ./app ./app

# Create directories for shared mounts
RUN mkdir -p /app/shared/src /app/shared/models /app/shared/config

# Create directories for logs and data with proper permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Expose the port
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
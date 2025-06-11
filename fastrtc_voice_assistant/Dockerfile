# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies if any are needed (e.g., for audio processing)
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libportaudio2

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./src ./src
COPY ./start.py .
COPY ./gradio3.py . 
COPY ./gradio4.py . 
# Add other necessary files/directories like models, config if they are directly used by start.py
COPY ./models ./models
# COPY ./config ./config # Config is volume-mounted in docker-compose, so maybe not needed here

# Expose the port the app runs on (as defined in docker-compose.yml)
EXPOSE 8080

# Define the command to run the application
CMD ["python", "start.py"]
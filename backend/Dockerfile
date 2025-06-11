# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app/backend

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (if any are known to be needed by VoiceAssistant or its deps)
# For now, we'll assume standard Python deps are enough. Add if needed.
# RUN apt-get update && apt-get install -y --no-install-recommends some-package

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code into the container
COPY ./app ./app

# This ARG will be set by docker-compose to the relative path of fastrtc_voice_assistant
ARG FASTRTC_VOICE_ASSISTANT_SRC_PATH=../fastrtc_voice_assistant/src

# Set PYTHONPATH to include the backend's app directory and the mounted fastrtc_voice_assistant src
# The actual fastrtc_voice_assistant_src will be mounted by docker-compose
ENV PYTHONPATH="/usr/src/app/backend:/usr/src/app/fastrtc_voice_assistant_src:${PYTHONPATH}"

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application using Uvicorn
# The --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
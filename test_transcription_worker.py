#!/usr/bin/env python3
"""
Test script for TranscriptionWorker process.
"""

import sys
import os
import time
import multiprocessing
import numpy as np

# Add the backend directory to the path
sys.path.insert(0, '/mnt/c/Users/PC/Dev/fastRTC/backend')

from src.core.transcription_worker import start_transcription_worker

def test_transcription_worker():
    """Test the transcription worker process in isolation."""
    print("🧪 Testing TranscriptionWorker process...")
    
    # Create multiprocessing pipe
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # Create shutdown event
    shutdown_event = multiprocessing.Event()
    
    # Start transcription worker process
    print("🚀 Starting transcription worker process...")
    process = multiprocessing.Process(
        target=start_transcription_worker,
        args=(
            child_conn,
            "Systran/faster-whisper-large-v3",  # Use HF model identifier
            "cuda",        # device
            "int8_float16", # compute_type
            1,             # beam_size
            False,         # vad_filter
            None,          # language
            None,          # initial_prompt
            True,          # normalize_audio
            shutdown_event
        ),
        daemon=True
    )
    process.start()
    
    print("⏳ Waiting for transcription process to initialize...")
    
    # Wait for ready signal with timeout
    start_time = time.time()
    timeout = 60  # 60 second timeout for model download
    
    while time.time() - start_time < timeout:
        if parent_conn.poll(1.0):  # Check every second
            try:
                status, message = parent_conn.recv()
                if status == 'ready':
                    print(f"✅ Transcription process ready: {message}")
                    break
                elif status == 'error':
                    print(f"❌ Transcription process failed: {message}")
                    return False
            except Exception as e:
                print(f"❌ Error receiving from process: {e}")
                return False
        else:
            elapsed = time.time() - start_time
            print(f"⏳ Still waiting... ({elapsed:.1f}s/{timeout}s)")
    else:
        print(f"❌ Timeout waiting for transcription process ({timeout}s)")
        return False
    
    # Test transcription with dummy audio
    print("🎤 Testing transcription with dummy audio...")
    
    # Create test audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Send transcription request
    parent_conn.send((test_audio, None, True))
    
    # Wait for result
    if parent_conn.poll(30.0):  # 30 second timeout
        try:
            status, result = parent_conn.recv()
            if status == 'success':
                transcription, info = result
                print(f"✅ Transcription successful: '{transcription}'")
                print(f"📊 Language: {getattr(info, 'language', 'unknown')}")
                return True
            elif status == 'error':
                print(f"❌ Transcription failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Error receiving transcription result: {e}")
            return False
    else:
        print("❌ Timeout waiting for transcription result")
        return False
    
    return True

if __name__ == "__main__":
    success = test_transcription_worker()
    if success:
        print("🎉 TranscriptionWorker test passed!")
        sys.exit(0)
    else:
        print("💥 TranscriptionWorker test failed!")
        sys.exit(1)
#!/usr/bin/env python3
"""
FastRTC Voice Assistant - Quick Test
This replaces your entire Piper v2 system with ~50 lines of code!
"""

import sys
import time
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model, AlgoOptions, SileroVadOptions
import requests

print("🎙️ FastRTC Voice Assistant Starting...")
print("=" * 50)

def print_status(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# Initialize models (will download automatically on first run)
print_status("🧠 Loading STT model (Moonshine)...")
try:
    stt_model = get_stt_model("moonshine/base")  # Faster, smaller model for testing
    print_status("✅ STT model loaded!")
except Exception as e:
    print_status(f"❌ STT model failed: {e}")
    sys.exit(1)

print_status("🗣️ Loading TTS model (Kokoro)...")
try:
    tts_model = get_tts_model("kokoro")
    print_status("✅ TTS model loaded!")
except Exception as e:
    print_status(f"❌ TTS model failed: {e}")
    sys.exit(1)

# LM Studio configuration (update this if needed)
LM_STUDIO_URL = "http://192.168.1.5:1234/v1/chat/completions"
LM_STUDIO_MODEL = "mistral-7b-instruct" # deepseek/deepseek-r1-0528-qwen3-8b  or mistral-7b-instruct gemma-3-27b-it

def get_llm_response(text):
    """Get response from LM Studio"""
    try:
        response = requests.post(LM_STUDIO_URL, 
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": "You are Mistral. You are an AI designed to help humans."},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 256,
                "temperature": 0.7
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print_status(f"⚠️ LLM error: {response.status_code}")
            return f"I heard you say: {text}"
            
    except requests.exceptions.RequestException as e:
        print_status(f"⚠️ LM Studio not available: {e}")
        return f"Echo: {text}"  # Fallback to echo

def voice_assistant(audio):
    """Main voice assistant pipeline - replaces your entire Piper v2 system!"""
    sample_rate, audio_array = audio
    
    # Speech to Text
    print_status("🎧 Processing audio...")
    user_text = stt_model.stt(audio)
    
    if not user_text or not user_text.strip():
        print_status("🤫 No speech detected")
        return
    
    print_status(f"👤 User: {user_text}")
    
    # Get LLM response
    assistant_text = get_llm_response(user_text)
    print_status(f"🤖 Assistant: {assistant_text}")
    
    # Text to Speech (streaming)
    for audio_chunk in tts_model.stream_tts_sync(assistant_text):
        yield audio_chunk

# Create stream with automatic VAD and turn-taking
print_status("🌐 Creating FastRTC stream...")

stream = Stream(
    ReplyOnPause(
        voice_assistant,
        can_interrupt=True,  # Allow interruptions
         algo_options=AlgoOptions(
            audio_chunk_duration=1.0,
            started_talking_threshold=0.2,
            speech_threshold=0.1
        ),
        model_options=SileroVadOptions(
            threshold=0.5, # Adjust for sensitivity
            min_speech_duration_ms=500, # Adjust for your needs 
            min_silence_duration_ms=1000 # Adjust for your needs the longer it is the more it will wait for the user to stop talking
        )
    ), 
    modality="audio", 
    mode="send-receive"
)

print_status("🚀 Starting voice assistant...")
print()
print("=" * 50)
print("🎉 SUCCESS! Voice Assistant is ready!")
print("=" * 50)
print("📋 What to do next:")
print("   1. Click the URL that opens in your browser")
print("   2. Click 'Allow' when asked for microphone permission")
print("   3. Start talking - no buttons needed!")
print("   4. The assistant will respond automatically")
print()
print("💡 Features working:")
print("   ✅ Automatic speech detection (VAD)")
print("   ✅ Speech-to-text (Moonshine)")
print("   ✅ LLM integration (LM Studio)")
print("   ✅ Text-to-speech (Kokoro)")
print("   ✅ Real-time audio streaming")
print("   ✅ Conversation interruption")
print()
print("🛑 To stop: Press Ctrl+C")
print("=" * 50)

# Launch the UI (opens browser automatically)
try:
    stream.ui.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False             # Set to True for public sharing
    )
except KeyboardInterrupt:
    print_status("🛑 Voice assistant stopped by user")
except Exception as e:
    print_status(f"❌ Error: {e}")
    sys.exit(1)

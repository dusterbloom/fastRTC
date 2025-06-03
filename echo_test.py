#!/usr/bin/env python3
"""
Simple Echo Test - No LLM required
Use this to test if FastRTC basics work
"""

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model

print("🔄 Simple Echo Test - Starting...")

# Load models
stt_model = get_stt_model("moonshine/tiny")
tts_model = get_tts_model("kokoro")

def echo_assistant(audio):
    """Simple echo - just repeat what you say"""
    sample_rate, audio_array = audio
    
    text = stt_model.stt(audio)
    if text and text.strip():
        print(f"You said: {text}")
        response = f"You said: {text}"
        
        for audio_chunk in tts_model.stream_tts_sync(response):
            yield audio_chunk

# Create stream
stream = Stream(ReplyOnPause(echo_assistant), modality="audio", mode="send-receive")

print("🎉 Echo test ready! Browser should open automatically.")
print("💬 Say something and it will repeat it back!")

stream.ui.launch(server_port=7861)

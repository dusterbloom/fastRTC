#!/usr/bin/env python3
"""
Simple WhisperLive test script using websocket-client (like official client)
Load audio file, send to WhisperLive, get transcription
"""

import websocket
import json
import numpy as np
import wave
import threading
import time

def load_wav_file(file_path):
    """Load WAV file and resample to 16kHz for WhisperLive"""
    with wave.open(file_path, 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        # Convert to float32 in range [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0
        original_rate = wav.getframerate()
        
        # Resample to 16kHz if needed
        if original_rate != 16000:
            # Simple downsample by taking every Nth sample
            ratio = original_rate / 16000
            indices = np.arange(0, len(audio_float), ratio).astype(int)
            audio_float = audio_float[indices]
            sample_rate = 16000
            print(f"Resampled from {original_rate}Hz to {sample_rate}Hz")
        else:
            sample_rate = original_rate
            
        return audio_float, sample_rate

class WhisperLiveTest:
    def __init__(self):
        self.transcription_received = False
        self.transcription_text = ""
        
    def on_message(self, ws, message):
        print(f"Received: {message}")
        try:
            data = json.loads(message)
            if data.get("message") == "SERVER_READY":
                print("Server ready!")
                return
            
            # Look for transcription
            text = data.get("text", "")
            segments = data.get("segments", [])
            if segments:
                text = " ".join([seg.get("text", "") for seg in segments])
            
            if text.strip():
                self.transcription_text = text.strip()
                self.transcription_received = True
                print(f"SUCCESS! Got transcription: '{self.transcription_text}'")
        except:
            pass
            
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed: {close_status_code}")
        
    def on_open(self, ws):
        print("Connected!")
        
        # Send config
        config = {
            "uid": "test_client", 
            "language": "en",
            "task": "transcribe",
            "model": "base",
            "use_vad": False,
            "save_output_recording": True,
            "enable_llm": False
        }
        ws.send(json.dumps(config))
        print("Sent config")
        
        # Load and send audio
        audio_file = '/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav'
        audio_data, sample_rate = load_wav_file(audio_file)
        
        print(f"Loaded audio: {audio_data.shape}, {sample_rate}Hz")
        print(f"Duration: {len(audio_data)/sample_rate:.1f}s")
        
        # Send audio in chunks using BINARY opcode (key difference!)
        chunk_size = 4096
        total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size != 0 else 0)
        print(f"Sending {total_chunks} chunks with BINARY opcode...")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            ws.send(chunk.tobytes(), websocket.ABNF.OPCODE_BINARY)
        
        # Send END_OF_AUDIO with BINARY opcode
        ws.send(b"END_OF_AUDIO", websocket.ABNF.OPCODE_BINARY)
        print("Sent END_OF_AUDIO with BINARY opcode")

def test_whisperlive():
    test = WhisperLiveTest()
    
    ws = websocket.WebSocketApp("ws://localhost:9090",
                              on_open=test.on_open,
                              on_message=test.on_message,
                              on_error=test.on_error,
                              on_close=test.on_close)
    
    # Run in thread
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    # Wait for transcription
    start_time = time.time()
    while time.time() - start_time < 20:
        if test.transcription_received:
            print(f"Final result: {test.transcription_text}")
            break
        time.sleep(0.1)
    
    if not test.transcription_received:
        print("TIMEOUT: No transcription received")
    
    ws.close()

if __name__ == "__main__":
    test_whisperlive()
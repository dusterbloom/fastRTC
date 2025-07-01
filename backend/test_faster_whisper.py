#!/usr/bin/env python3
import faster_whisper
import wave
import numpy as np

print('Testing faster-whisper with real audio file...')
model = faster_whisper.WhisperModel('tiny', device='cpu', compute_type='int8')

# Load the same audio file used in tests
with wave.open('tests/samples/audio_en.wav', 'rb') as wav:
    frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    audio_float = audio.astype(np.float32) / 32768.0
    sample_rate = wav.getframerate()

print(f'Loaded audio: {len(audio_float)} samples at {sample_rate}Hz')
print(f'Duration: {len(audio_float)/sample_rate:.1f}s')

# Resample to 16kHz if needed
if sample_rate != 16000:
    ratio = sample_rate / 16000
    indices = np.arange(0, len(audio_float), ratio).astype(int)
    audio_float = audio_float[indices]
    sample_rate = 16000
    print(f'Resampled to 16kHz: {len(audio_float)} samples')

segments, info = model.transcribe(audio_float, beam_size=1)
segments = list(segments)
print(f'Transcription result: {len(segments)} segments')
for segment in segments:
    print(f'Text: "{segment.text}"')
print(f'Language: {info.language}, confidence: {info.language_probability:.2f}')
# WhisperLive Working Setup Documentation

## Overview
This document captures the working configuration for WhisperLive integration with FastRTC after successful troubleshooting.

## Working Configuration

### Library Versions (Critical - Must Match Exactly)
```
ctranslate2==4.4.0
faster-whisper==1.1.0  
whisper-live==0.7.1
openai-whisper==20240930
onnxruntime-gpu==1.17.0  # For CUDA support
tokenizers==0.20.3
numpy==1.26.4
```

### Key Dependencies
```
kaldialign
torchaudio
openvino
openvino-genai
openvino-tokenizers
optimum
optimum-intel
```

## Working Server Startup

### WhisperLive Server Command
```bash
python run_whisper_server.py --port 9090 --backend faster_whisper --omp_num_threads 1
```

### Environment Variables
```bash
USE_WHISPER_LIVE=true
STT_BACKEND=whisper-live
FASTER_WHISPER_MODEL_PATH=/home/peppi/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
```

## Working Client Code

### Correct Usage Pattern
```python
from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    "localhost", 
    9090, 
    lang="en", 
    translate=False, 
    model="small",
    use_vad=False,  # FastRTC handles VAD
    save_output_recording=True,
    output_recording_filename="./output.wav"
)

# For audio file transcription
result = client(audio_file_path)
```

### What DOESN'T Work
- Raw WebSocket connections with manual chunking
- Bulk audio uploads (36+ seconds at once)
- Using newer ctranslate2 versions (breaks model loading)
- Mismatched library versions (causes VadOptions errors)

## Successful Test Results

### Real-Time Transcription Output
```
The stale smell of old beer lingers.
It takes heat to bring out the odor a cold dip restores health and zest a salt pickle tastes fine with ham tacos
I'll pass door are my favorite.
A zestful food is the hot cross bun.
```

### Server Logs (Working)
```
INFO:root:Using Device=cuda with precision float16
INFO:root:Running faster_whisper backend.
INFO:faster_whisper:Processing audio with duration 00:01.024
INFO:faster_whisper:Processing audio with duration 00:02.048
[...progressive duration processing...]
```

## Troubleshooting History

### Issues Resolved
1. **Model Loading Hang**: Fixed by downgrading ctranslate2 to 4.4.0
2. **VadOptions 'onset' Error**: Fixed by matching exact library versions
3. **No Transcription Output**: Fixed by using official TranscriptionClient instead of raw WebSocket
4. **CUDA Not Available**: Fixed by installing onnxruntime-gpu==1.17.0

### Critical Dependencies
- **ctranslate2 4.4.0**: Required for CUDA 12 compatibility
- **faster-whisper 1.1.0**: Must match whisper-live expectations
- **onnxruntime-gpu**: Required for both CUDA and VAD functionality

## FastRTC Integration

### Environment Setup
```bash
./fastrtc.sh dev --threading --whisper-live
```

### Threading Configuration
```python
USE_THREADING_PIPELINE=true
THREADING_FALLBACK_TO_ASYNC=true
```

### STT Backend Selection
```python
STT_BACKEND=whisper-live  # Not 'whisper_live'
```

## Performance Characteristics

### Audio Processing
- Real-time streaming with progressive transcription
- CUDA GPU acceleration working
- Voice Activity Detection handled by FastRTC (not WhisperLive)
- Audio format: 16kHz, float32, mono

### Output Format
- Real-time progressive text updates
- SRT file generation with timestamps
- WebSocket streaming compatible

## Next Steps
- Test full FastRTC integration with working onnxruntime
- Verify end-to-end: audio → WhisperLive → transcription → Ollama
- Performance testing and optimization

## Notes
- WhisperLive works independently with CUDA
- FastRTC VAD requires working onnxruntime installation
- Version compatibility is critical - any mismatch breaks functionality
- Official TranscriptionClient is the only reliable approach
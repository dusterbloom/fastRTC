# WhisperLive Integration for FastRTC

This document describes the WhisperLive integration for FastRTC's threading pipeline, providing real-time streaming speech-to-text capabilities with faster-whisper backend.

## Overview

WhisperLive provides a client-server architecture for real-time speech transcription using OpenAI's Whisper models. This integration adds WhisperLive as an alternative STT backend alongside the existing faster-whisper implementation, specifically optimized for the threading pipeline.

## Features

- **Real-time streaming transcription** with minimal latency
- **Client-server architecture** for scalable deployment
- **Multiple backend support** (faster-whisper, TensorRT, OpenVINO)
- **Automatic server management** with health checks
- **Threading pipeline integration** for optimal performance
- **Voice Activity Detection (VAD)** support
- **Multi-language support** with translation capabilities

## Installation

WhisperLive is already included in the requirements:

```bash
pip install whisper-live
```

## Configuration

### Environment Variables

Add these to your `.env.development` or set via command line:

```bash
# Enable WhisperLive STT backend
USE_WHISPER_LIVE=true

# WhisperLive server configuration
WHISPER_LIVE_HOST=localhost
WHISPER_LIVE_PORT=9090
WHISPER_LIVE_MODEL=small
WHISPER_LIVE_LANGUAGE=en
WHISPER_LIVE_BACKEND=faster_whisper
WHISPER_LIVE_AUTO_START=true

# Threading pipeline (required for WhisperLive)
USE_THREADING_PIPELINE=true
```

### Command Line Usage

```bash
# Start with WhisperLive STT backend
./fastrtc.sh dev --threading --whisper-live

# With custom configuration
WHISPER_LIVE_MODEL=large ./fastrtc.sh dev --threading --whisper-live

# Without automatic server startup
WHISPER_LIVE_AUTO_START=false ./fastrtc.sh dev --threading --whisper-live
```

## Architecture

### Components

1. **WhisperLiveSTTEngine** (`whisper_live_stt.py`)
   - Implements the base STT engine interface
   - Manages WhisperLive client connections
   - Handles server health checks and auto-startup
   - Provides async transcription methods

2. **WhisperLiveStreamingWorker** (`whisper_live_worker.py`)
   - Specialized worker for threading pipeline
   - Optimized for real-time audio streaming
   - Manages audio buffering and sentence assembly
   - Handles partial and final transcription results

3. **Threading Integration** (`threading_callback_handler.py`)
   - Automatic backend selection based on configuration
   - Proper initialization and cleanup of WhisperLive resources
   - Seamless integration with existing pipeline

### Data Flow

```
Audio Input → WhisperLive Worker → WhisperLive Server → Transcription Results → Pipeline
```

1. Audio chunks are buffered in the WhisperLive worker
2. When sufficient audio is accumulated, streaming transcription begins
3. The WhisperLive server processes audio with faster-whisper backend
4. Partial and final results are sent back to the pipeline
5. Results are formatted and passed to the LLM worker

## Server Management

### Automatic Server Startup

When `WHISPER_LIVE_AUTO_START=true`, the system will:

1. Check if WhisperLive server is running on the configured port
2. If not running, start a new server process automatically
3. Wait for server to become healthy before proceeding
4. Clean up server process on shutdown

### Manual Server Management

You can also run the WhisperLive server manually:

```bash
# Start server with faster-whisper backend
python -m whisper_live.server --port 9090 --backend faster_whisper

# With custom model
python -m whisper_live.server --port 9090 --backend faster_whisper -fw "Systran/faster-whisper-large-v3"

# With TensorRT backend (requires setup)
python -m whisper_live.server --port 9090 --backend tensorrt -trt /path/to/tensorrt/model
```

## Performance Optimization

### Threading Pipeline Benefits

- **Parallel processing**: STT, LLM, and TTS run in separate threads
- **Reduced latency**: Streaming transcription with minimal buffering
- **Better resource utilization**: Optimal CPU/GPU usage
- **Improved responsiveness**: Non-blocking audio processing

### Configuration Tuning

```python
# In WhisperLiveStreamingWorker
min_audio_length = 1.0      # Minimum audio before processing (seconds)
chunk_duration = 0.5        # Audio chunk size (seconds)
processing_timeout = 0.05   # Worker processing timeout
buffer_timeout = 2.0        # Sentence buffer timeout
```

## Monitoring and Debugging

### Health Checks

The system provides health checks for:
- WhisperLive server connectivity
- Client connection status
- Worker thread health
- Audio processing pipeline

### Logging

Enable debug logging to monitor WhisperLive operations:

```bash
LOG_LEVEL=DEBUG ./fastrtc.sh dev --threading --whisper-live
```

### Statistics

Access runtime statistics via the worker:

```python
stats = whisper_live_worker.get_stats()
print(f"Active streams: {stats['active_streams']}")
print(f"Average latency: {stats['avg_latency']}")
```

## Testing

Run the integration test to verify everything works:

```bash
python test_whisper_live_integration.py
```

This test will:
1. Initialize the WhisperLive STT engine
2. Start a WhisperLive server if needed
3. Test transcription with synthetic audio
4. Verify worker integration with the pipeline
5. Clean up all resources

## Troubleshooting

### Common Issues

1. **Server startup fails**
   - Check if port 9090 is available
   - Verify whisper-live is installed correctly
   - Check firewall settings

2. **Connection timeouts**
   - Increase connection timeout in configuration
   - Verify server is running and healthy
   - Check network connectivity

3. **Poor transcription quality**
   - Try a larger model (medium, large)
   - Adjust VAD settings
   - Check audio quality and sample rate

4. **High latency**
   - Reduce `min_audio_length` for faster processing
   - Use smaller model for speed
   - Enable GPU acceleration

### Debug Commands

```bash
# Check if server is running
curl http://localhost:9090/health

# Test WhisperLive client directly
python -c "from whisper_live.client import TranscriptionClient; client = TranscriptionClient('localhost', 9090); print('Connected!')"

# Monitor worker threads
ps aux | grep whisper
```

## Migration from faster-whisper

To migrate from the existing faster-whisper STT:

1. **Backup current configuration**
2. **Add WhisperLive environment variables**
3. **Test with `--whisper-live` flag**
4. **Update production configuration**

The migration is seamless - both backends can coexist and be switched via configuration.

## Future Enhancements

- **Multi-client support** for concurrent transcriptions
- **Custom model fine-tuning** integration
- **Advanced VAD configuration** options
- **Real-time translation** capabilities
- **WebSocket streaming** for browser clients

## References

- [WhisperLive GitHub](https://github.com/collabora/WhisperLive)
- [Faster-Whisper Documentation](https://github.com/SYSTRAN/faster-whisper)
- [FastRTC Threading Pipeline](./threading-pipeline.md)
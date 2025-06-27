# Threading Pipeline Architecture

## Overview

This document describes the new pure threading-based pipeline architecture that replaces the sync/async complexity with event-driven worker threads and queues.

## Architecture Changes

### Before (Sync/Async Bridging)
```
FastRTC Audio → Sync Callback → Async Bridge → Services → Back to Sync
```
- 6+ thread context switches per request
- Complex event loop management 
- Sync/async boundary issues
- Timeout coordination complexity

### After (Pure Threading)
```
FastRTC Audio → Threading Pipeline → Worker Queues → Direct Output
```
- Pure threading with queues
- Event-driven coordination
- Predictable performance
- Eliminated sync/async complexity

## Key Components

### 1. AudioPipelineManager
Central coordinator managing:
- Worker thread lifecycle
- Queue-based communication
- Generation state tracking
- Event coordination

### 2. Worker Threads
- **STTStreamingWorker**: Speech-to-text processing
- **LLMStreamingWorker**: Token streaming from LLM
- **TTSStreamingWorker**: Text-to-speech synthesis
- **TTSOutputWorker**: Audio output delivery

### 3. Event-Driven State Management
- **GenerationState**: Tracks each request lifecycle
- **Threading Events**: Coordinate between workers
- **InterruptionManager**: Handles user interruptions

## Usage

### Enable Threading Pipeline
```bash
# Set environment variable
export USE_THREADING_PIPELINE=true

# Or copy configuration template
cp .env.threading .env
```

### Feature Flags
- `USE_THREADING_PIPELINE`: Enable new threading system
- `THREADING_FALLBACK_TO_ASYNC`: Fallback to old system on errors
- `DEBUG_THREADING`: Enable threading debug logs

### Testing
```bash
# Test threading components
python test_threading_pipeline.py

# Run with threading enabled
USE_THREADING_PIPELINE=true python src/core/main.py
```

## Performance Benefits

### Expected Improvements
- **60-80% latency reduction** (eliminate thread handoffs)
- **Predictable performance** (no event loop coordination)
- **Better interruption** (event-based, not polling)
- **Easier debugging** (linear data flow)

### Preserved Features
✅ **Streaming STT→LLM→TTS pipeline**  
✅ **TTS interruption system**  
✅ **A-MEM memory isolation**  
✅ **Performance monitoring**  
✅ **User identification**  
✅ **Clean module interfaces**

## Migration Strategy

### Phase 1: Parallel Deployment
- Both systems run in parallel
- Feature flag controls which is active
- Automatic fallback on threading failures

### Phase 2: Gradual Transition
- Monitor performance metrics
- Increase threading usage gradually
- Collect feedback and optimize

### Phase 3: Full Migration
- Remove async bridging complexity
- Clean up legacy code
- Finalize threading optimizations

## Configuration

### Environment Variables
```bash
# Core settings
USE_THREADING_PIPELINE=true
THREADING_FALLBACK_TO_ASYNC=true
THREADING_MAX_QUEUE_SIZE=100

# Performance tuning
THREADING_PROCESSING_TIMEOUT=0.1
THREADING_GENERATION_TIMEOUT=30.0

# Worker configuration
THREADING_STT_CONFIDENCE=0.6
THREADING_TTS_CHUNK_SIZE=1024
```

### Monitoring
```python
# Get pipeline statistics
stats = callback_handler.get_handler_stats()
print(f"Handler type: {stats['handler_type']}")
print(f"Pipeline stats: {stats['pipeline_stats']}")
```

## Troubleshooting

### Common Issues

1. **Threading not enabled**: Check `USE_THREADING_PIPELINE` environment variable
2. **Performance regression**: Check queue sizes in stats
3. **Worker failures**: Check individual worker statistics
4. **Memory leaks**: Monitor generation cleanup

### Debug Commands
```bash
# Enable debug logging
export DEBUG_THREADING=true
export DEBUG_STREAMING=true

# Test components individually
python test_threading_pipeline.py

# Monitor queue health
# Use callback_handler.get_handler_stats() to check queue sizes
```

## File Structure

```
src/core/
├── pipeline_manager.py          # Central pipeline coordinator
├── pipeline_workers.py          # Base worker classes
├── stt_worker.py               # STT processing worker
├── llm_worker.py               # LLM streaming worker
└── tts_worker.py               # TTS synthesis worker

src/integration/
├── threading_callback_handler.py  # Threading-based callback
└── unified_callback_handler.py    # Unified interface

src/config/
└── threading_config.py         # Threading configuration
```

## Integration Points

### Preserved Interfaces
- **Audio Engines**: No changes to STT/TTS engines
- **Memory System**: A-MEM integration unchanged  
- **LLM Service**: Voice assistant interface preserved
- **Configuration**: Existing config system extended

### New Components
- **Pipeline Manager**: Central coordination
- **Worker Threads**: Specialized processing
- **Event System**: Inter-worker communication
- **Unified Handler**: Seamless switching

This architecture provides a solid foundation for predictable, high-performance real-time audio processing while preserving all existing functionality and interfaces.
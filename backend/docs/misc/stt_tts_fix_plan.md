# TTS Audio Quality Fix Implementation Plan

## 🔍 Root Cause Analysis

After comparing the working V4 implementation with the refactored `gradio2.py`, I've identified the exact differences causing the "music-like" audio output:

### Key Differences Found:

1. **Chunk Size**: V4 uses `min(1024, current_chunk_array.size)` vs refactored uses `min(2048, current_chunk_array.size)`
2. **Audio Processing**: V4 has no normalization vs refactored has aggressive normalization
3. **Sample Rate Handling**: V4 uses `sample_rate` for raw arrays vs refactored uses hardcoded `kokoro_sample_rate = 24000`
4. **TTS Model Access**: Both use direct access but with different error handling

## 🎯 Critical Fixes Needed

### Fix 1: Restore V4 Chunk Size
**Current (Broken)**:
```python
chunk_size = min(2048, current_chunk_array.size)  # Too large
```

**V4 Working**:
```python
chunk_size = min(1024, current_chunk_array.size)  # Correct size
```

### Fix 2: Remove Audio Normalization
**Current (Broken)**:
```python
# CRITICAL FIX: Less aggressive normalization to preserve audio quality
max_val = np.max(np.abs(current_chunk_array))
if max_val > 0.95:  # Only normalize if really clipping
    current_chunk_array = current_chunk_array * (0.9 / max_val)
```

**V4 Working**:
```python
# NO NORMALIZATION - Raw audio from TTS model
```

### Fix 3: Fix Sample Rate for Raw Arrays
**Current (Broken)**:
```python
# CRITICAL FIX: Use Kokoro's default sample rate (24000) instead of 16000
kokoro_sample_rate = 24000
yield (kokoro_sample_rate, mini_chunk), additional_outputs
```

**V4 Working**:
```python
yield (sample_rate, mini_chunk), additional_outputs  # Use input sample_rate
```

### Fix 4: Remove Excessive Logging
**Current (Broken)**:
```python
logger.info(f"🔧 TTS Debug: Yielding chunk {i//chunk_size+1}, size={mini_chunk.size}, range=[{np.min(mini_chunk):.3f}, {np.max(mini_chunk):.3f}]")
```

**V4 Working**:
```python
# No per-chunk logging - only final summary
```

## 📝 Complete Implementation

### Replace Lines 380-428 in gradio2.py

**REMOVE THIS SECTION**:
```python
                # CRITICAL FIX: Stream directly like V4 with proper audio format
                chunk_count = 0
                total_samples = 0
                for tts_output_item in voice_assistant.tts_engine.tts_model.stream_tts_sync(assistant_response_text, tts_options):
                    if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2 and isinstance(tts_output_item[1], np.ndarray):
                        current_sr, current_chunk_array = tts_output_item
                        if current_chunk_array.size > 0:
                            chunk_count += 1
                            total_samples += current_chunk_array.size
                            
                            # CRITICAL FIX: Ensure proper audio format for output
                            # Convert to float32 and ensure proper range
                            if current_chunk_array.dtype != np.float32:
                                current_chunk_array = current_chunk_array.astype(np.float32)
                            
                            # CRITICAL FIX: Less aggressive normalization to preserve audio quality
                            max_val = np.max(np.abs(current_chunk_array))
                            if max_val > 0.95:  # Only normalize if really clipping
                                current_chunk_array = current_chunk_array * (0.9 / max_val)
                            
                            # CRITICAL FIX: Use correct sample rate from TTS model
                            # Yield smaller chunks to prevent timeouts (exactly like V4)
                            chunk_size = min(2048, current_chunk_array.size)  # Slightly larger chunks for better audio
                            for i in range(0, current_chunk_array.size, chunk_size):
                                mini_chunk = current_chunk_array[i:i+chunk_size]
                                if mini_chunk.size > 0:
                                    logger.info(f"🔧 TTS Debug: Yielding chunk {i//chunk_size+1}, size={mini_chunk.size}, range=[{np.min(mini_chunk):.3f}, {np.max(mini_chunk):.3f}]")
                                    yield (current_sr, mini_chunk), additional_outputs
                    elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                        chunk_count += 1
                        total_samples += tts_output_item.size
                        
                        # CRITICAL FIX: Ensure proper audio format
                        if tts_output_item.dtype != np.float32:
                            tts_output_item = tts_output_item.astype(np.float32)
                        
                        # CRITICAL FIX: Less aggressive normalization
                        max_val = np.max(np.abs(tts_output_item))
                        if max_val > 0.95:  # Only normalize if really clipping
                            tts_output_item = tts_output_item * (0.9 / max_val)
                        
                        # CRITICAL FIX: Use Kokoro's default sample rate (24000) instead of 16000
                        kokoro_sample_rate = 24000
                        chunk_size = min(2048, tts_output_item.size)
                        for i in range(0, tts_output_item.size, chunk_size):
                            mini_chunk = tts_output_item[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                logger.info(f"🔧 TTS Debug: Yielding raw chunk {i//chunk_size+1}, size={mini_chunk.size}, range=[{np.min(mini_chunk):.3f}, {np.max(mini_chunk):.3f}]")
                                yield (kokoro_sample_rate, mini_chunk), additional_outputs
```

**REPLACE WITH V4-EXACT CODE**:
```python
                # Stream directly like V4 - NO MODIFICATIONS
                chunk_count = 0
                total_samples = 0
                for tts_output_item in voice_assistant.tts_engine.tts_model.stream_tts_sync(assistant_response_text, tts_options):
                    if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2 and isinstance(tts_output_item[1], np.ndarray):
                        current_sr, current_chunk_array = tts_output_item
                        if current_chunk_array.size > 0:
                            chunk_count += 1
                            total_samples += current_chunk_array.size
                            # Yield smaller chunks to prevent timeouts
                            chunk_size = min(1024, current_chunk_array.size)
                            for i in range(0, current_chunk_array.size, chunk_size):
                                mini_chunk = current_chunk_array[i:i+chunk_size]
                                if mini_chunk.size > 0:
                                    yield (current_sr, mini_chunk), additional_outputs
                    elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                        chunk_count += 1
                        total_samples += tts_output_item.size
                        chunk_size = min(1024, tts_output_item.size)
                        for i in range(0, tts_output_item.size, chunk_size):
                            mini_chunk = tts_output_item[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (sample_rate, mini_chunk), additional_outputs
```

## 🔧 Additional Fixes

### Fix TTS Model Access
Ensure the TTS model is accessible. If `voice_assistant.tts_engine.tts_model` doesn't work, use:

```python
# Get direct access to TTS model like V4
tts_model = voice_assistant.tts_engine.tts_model
# OR if that fails:
# tts_model = get_tts_model("kokoro")  # Direct import like V4
```

### Fix Sample Rate Variable
Make sure `sample_rate` is available in scope:

```python
# At the top of the callback function, ensure sample_rate is defined
sample_rate = 16000  # Or get from audio_data_tuple
```

## 🧪 Testing Steps

1. **Apply the fixes** to `gradio2.py`
2. **Test with Italian input** - should hear clear speech, not music
3. **Compare audio quality** with V4 implementation
4. **Test different voices** to ensure consistency
5. **Monitor logs** for any errors

## 📊 Expected Results

After applying these fixes:
- ✅ Audio output should be clear speech instead of "music"
- ✅ TTS quality should match V4 implementation exactly
- ✅ Chunk processing should be efficient (1024 vs 2048)
- ✅ No audio distortion from normalization
- ✅ Proper sample rate handling

## 🚨 Critical Points

1. **DO NOT** add any audio normalization - V4 works without it
2. **USE** exact chunk size of 1024 like V4
3. **PRESERVE** original sample rates from TTS model
4. **REMOVE** excessive debug logging that may affect performance
5. **ENSURE** direct TTS model access works properly

The key insight is that the refactored code tried to "improve" the audio processing, but these "improvements" actually broke the audio quality. The V4 implementation works perfectly as-is and should be replicated exactly.
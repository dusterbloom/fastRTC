# Complete TTS Fix Implementation Guide

## 🎯 Overview

This guide provides step-by-step instructions to fix both TTS issues in the refactored FastRTC Voice Assistant:

1. **Audio Quality Issue**: "Music-like" sound instead of clear speech
2. **Voice Mapping Issue**: Italian language not using `if_sara` voice

## 🔧 Implementation Steps

### Step 1: Fix TTS Audio Quality (Priority 1)

**File to Edit**: `backend/gradio2.py`

**Location**: Lines 380-428 (the TTS streaming section)

**Action**: Replace the entire TTS streaming section with V4-exact code

#### 1.1 Find This Section (REMOVE):
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

#### 1.2 Replace With V4-Exact Code:
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

### Step 2: Fix Voice Mapping (Priority 2)

**File to Edit**: `backend/src/core/voice_assistant.py`

#### 2.1 Add Missing Method

Add this method to the `VoiceAssistant` class:

```python
def get_voices_for_language(self, language: str) -> List[str]:
    """Get appropriate voices for the detected language.
    
    Args:
        language: Language code (e.g., 'i', 'e', 'f', 'p', 'a')
        
    Returns:
        List[str]: List of voice identifiers for the language
    """
    # Import here to avoid circular imports
    from ..audio.language.voice_mapper import VoiceMapper
    
    # Create voice mapper if not exists
    if not hasattr(self, '_voice_mapper'):
        self._voice_mapper = VoiceMapper()
    
    return self._voice_mapper.get_voices_for_language(language)
```

#### 2.2 Update Imports

Add to the imports at the top of `voice_assistant.py`:

```python
from typing import List, Dict, Any, Optional, Tuple, Generator
```

#### 2.3 Initialize Voice Mapper (Optional)

Add to the `__init__` method of `VoiceAssistant`:

```python
def __init__(self, ...):
    # ... existing initialization ...
    
    # Initialize voice mapper
    from ..audio.language.voice_mapper import VoiceMapper
    self._voice_mapper = VoiceMapper()
    
    logger.info("🎤 Voice mapper initialized in VoiceAssistant")
```

### Step 3: Verify Sample Rate Variable

**File**: `backend/gradio2.py`

**Location**: Around line 193 in the callback function

**Ensure this variable is available**:
```python
def smart_voice_assistant_callback_rt(audio_data_tuple: tuple):
    # ... existing code ...
    
    # Make sure sample_rate is defined early
    sample_rate, audio_array = assistant_instance.process_audio_array(audio_data_tuple)
    
    # ... rest of the function ...
```

## 🧪 Testing Protocol

### Test 1: Audio Quality Fix
1. **Run the voice assistant**
2. **Speak any phrase** in any language
3. **Listen to the response** - should be clear speech, not "music"
4. **Check logs** for successful TTS completion

### Test 2: Voice Mapping Fix
1. **Speak in Italian**: "Ciao, come stai?"
2. **Check logs** for:
   ```
   INFO:__main__:🎤 TTS using language 'i' with voices: ['if_sara', 'im_nicola']
   ```
3. **Verify no fallback warning**
4. **Listen for Italian voice** (if_sara)

### Test 3: Other Languages
1. **Test Spanish**: "Hola, ¿cómo estás?"
   - Expected voices: `['ef_dora', 'em_alex', 'em_santa']`
2. **Test French**: "Bonjour, comment allez-vous?"
   - Expected voices: `['ff_siwis']`

## 🔍 Troubleshooting

### Issue: TTS Model Access Error
**Error**: `AttributeError: 'TTSEngine' object has no attribute 'tts_model'`

**Solution**: Use direct TTS model access:
```python
# Replace this line in gradio2.py around line 383:
for tts_output_item in voice_assistant.tts_engine.tts_model.stream_tts_sync(assistant_response_text, tts_options):

# With this:
tts_model = get_tts_model("kokoro")  # Direct import like V4
for tts_output_item in tts_model.stream_tts_sync(assistant_response_text, tts_options):
```

### Issue: Voice Mapper Not Found
**Error**: `ModuleNotFoundError: No module named 'src.audio.language.voice_mapper'`

**Solution**: Use direct config access:
```python
def get_voices_for_language(self, language: str) -> List[str]:
    """Get appropriate voices for the detected language."""
    from ..config.language_config import get_available_voices
    return get_available_voices(language)
```

### Issue: Sample Rate Not Defined
**Error**: `NameError: name 'sample_rate' is not defined`

**Solution**: Define sample_rate at the top of the callback:
```python
def smart_voice_assistant_callback_rt(audio_data_tuple: tuple):
    # ... existing code ...
    sample_rate = 16000  # Default fallback
    if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
        sample_rate, audio_array = audio_data_tuple
    # ... rest of function ...
```

## 📊 Expected Results

### Before Fixes:
```
WARNING:src.audio.language.voice_mapper:No voices found for language 'it', falling back to 'a'        
INFO:__main__:🎤 TTS using language 'it' with voices: ['af_heart', 'af_bella', 'af_sarah']
INFO:__main__:🔧 TTS Debug: Yielding chunk 1, size=2048, range=[-0.283, 0.272]
[Audio sounds like music/noise]
```

### After Fixes:
```
INFO:__main__:🎤 TTS using language 'it' with voices: ['if_sara', 'im_nicola']
INFO:__main__:✅ TTS SUCCESS using Italian voice: if_sara
[Audio sounds like clear Italian speech]
```

## 🚨 Critical Success Factors

1. **Exact V4 Replication**: Don't modify the TTS streaming code - use V4 exactly
2. **No Audio Processing**: Remove all normalization and dtype conversions
3. **Correct Chunk Size**: Use 1024, not 2048
4. **Method Implementation**: Ensure `get_voices_for_language()` method exists
5. **Sample Rate Handling**: Use original sample rates from TTS model

## 📝 Implementation Checklist

- [ ] Replace TTS streaming section in gradio2.py with V4-exact code
- [ ] Add `get_voices_for_language()` method to VoiceAssistant
- [ ] Update imports in voice_assistant.py
- [ ] Test audio quality with any language
- [ ] Test Italian voice mapping specifically
- [ ] Verify no fallback warnings in logs
- [ ] Test other languages for regression

Following this guide should completely resolve both TTS issues and restore the working functionality from the V4 implementation.
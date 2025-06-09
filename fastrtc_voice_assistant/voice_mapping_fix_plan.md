# Voice Mapping Fix Implementation Plan

## 🔍 Problem Analysis

The logs show:
```
WARNING:src.audio.language.voice_mapper:No voices found for language 'it', falling back to 'a'        
INFO:__main__:🎤 TTS using language 'it' with voices: ['af_heart', 'af_bella', 'af_sarah']
```

This indicates:
1. Language detection correctly identifies Italian ('it')
2. Voice mapper fails to find Italian voices and falls back to 'a' (American English)
3. But then it uses Italian language with American English voices
4. Should use Italian voices: `['if_sara', 'im_nicola']`

## 🎯 Root Cause

The issue is in `gradio2.py` line 355:
```python
tts_voices_to_try = voice_assistant.get_voices_for_language(voice_assistant.current_language)
```

**Problem**: The `VoiceAssistant` class doesn't have a `get_voices_for_language()` method!

## 🔧 Solution Implementation

### Step 1: Add Missing Method to VoiceAssistant

**File**: `fastrtc_voice_assistant/src/core/voice_assistant.py`

**Add this method** to the `VoiceAssistant` class:

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

### Step 2: Initialize Voice Mapper in VoiceAssistant

**Add to the `__init__` method** of `VoiceAssistant`:

```python
def __init__(self, ...):
    # ... existing initialization ...
    
    # Initialize voice mapper
    from ..audio.language.voice_mapper import VoiceMapper
    self._voice_mapper = VoiceMapper()
    
    logger.info("🎤 Voice mapper initialized in VoiceAssistant")
```

### Step 3: Add Required Import

**Add to imports** at the top of `voice_assistant.py`:

```python
from typing import List, Dict, Any, Optional, Tuple, Generator
```

### Step 4: Verify Language Configuration

**Check** that `src/config/language_config.py` has correct Italian mapping:

```python
KOKORO_VOICE_MAP: Dict[str, List[str]] = {
    'a': ['af_heart', 'af_bella', 'af_sarah'],                    # American English
    'b': ['bf_emma', 'bf_isabella', 'bm_george'],                 # British English  
    'i': ['if_sara', 'im_nicola'],                                # Italian ✅
    'e': ['ef_dora', 'em_alex', 'em_santa'],                      # Spanish
    'f': ['ff_siwis'],                                             # French
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],                      # Portuguese
    'j': ['jf_alpha', 'jf_gongitsune', 'jm_kumo'],               # Japanese
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zm_yunjian', 'zm_yunxi'],  # Chinese
    'h': ['hf_alpha', 'hf_beta', 'hm_omega'],                     # Hindi
}
```

## 🧪 Testing the Fix

### Test 1: Voice Mapping Function
```python
# Test in Python console
from src.core.voice_assistant import VoiceAssistant
va = VoiceAssistant()
italian_voices = va.get_voices_for_language('i')
print(f"Italian voices: {italian_voices}")
# Expected: ['if_sara', 'im_nicola']
```

### Test 2: Language Detection Chain
```python
# Test complete chain
va.current_language = 'i'  # Set to Italian
voices = va.get_voices_for_language(va.current_language)
print(f"Voices for {va.current_language}: {voices}")
# Expected: ['if_sara', 'im_nicola']
```

### Test 3: Full Integration
1. Run the voice assistant
2. Speak in Italian: "Ciao, come stai?"
3. Check logs for:
   ```
   INFO:__main__:🎤 TTS using language 'i' with voices: ['if_sara', 'im_nicola']
   ```
4. Verify no fallback warning

## 🔄 Alternative Implementation

If the above doesn't work, use direct config access:

```python
def get_voices_for_language(self, language: str) -> List[str]:
    """Get appropriate voices for the detected language."""
    from ..config.language_config import get_available_voices
    return get_available_voices(language)
```

## 📊 Expected Results

After implementing the fix:

**Before (Broken)**:
```
WARNING:src.audio.language.voice_mapper:No voices found for language 'it', falling back to 'a'        
INFO:__main__:🎤 TTS using language 'it' with voices: ['af_heart', 'af_bella', 'af_sarah']
```

**After (Fixed)**:
```
INFO:__main__:🎤 TTS using language 'it' with voices: ['if_sara', 'im_nicola']
INFO:__main__:✅ TTS SUCCESS using Italian voice: if_sara
```

## 🚨 Critical Points

1. **Method Missing**: The core issue is `get_voices_for_language()` method doesn't exist
2. **Voice Mapper**: Need to properly initialize and use the VoiceMapper class
3. **Language Consistency**: Ensure language codes are consistent throughout the pipeline
4. **Import Management**: Avoid circular imports by importing within methods if needed

## 🔍 Debugging Steps

If the fix doesn't work immediately:

1. **Check method exists**:
   ```python
   hasattr(voice_assistant, 'get_voices_for_language')
   ```

2. **Check voice mapper**:
   ```python
   voice_assistant._voice_mapper.get_voices_for_language('i')
   ```

3. **Check config**:
   ```python
   from src.config.language_config import KOKORO_VOICE_MAP
   print(KOKORO_VOICE_MAP['i'])
   ```

4. **Add debug logging**:
   ```python
   logger.info(f"🔧 Debug: Language={language}, Voices={voices}")
   ```

## 📝 Implementation Order

1. **First**: Add the missing method to `VoiceAssistant`
2. **Second**: Initialize voice mapper in `__init__`
3. **Third**: Test with Italian input
4. **Fourth**: Verify all languages work correctly

This fix will ensure that Italian language detection properly leads to `if_sara` voice selection instead of falling back to American English voices.
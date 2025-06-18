"""Language configuration mappings for multilingual support.

This module contains the language mappings extracted from the original
monolithic implementation, providing support for Whisper STT to Kokoro TTS
language code conversion and voice selection.
"""

from typing import Dict, List

# Language code mappings for Kokoro TTS (extracted from original)
WHISPER_TO_KOKORO_LANG: Dict[str, str] = {
    'en': 'a',    # American English (default)
    'it': 'i',    # Italian  
    'es': 'e',    # Spanish
    'fr': 'f',    # French
    'de': 'a',    # German -> fallback to English (not natively supported)
    'pt': 'p',    # Portuguese -> Brazilian Portuguese
    'ja': 'j',    # Japanese
    'ko': 'a',    # Korean -> fallback to English (not natively supported)
    'zh': 'z',    # Chinese -> Mandarin Chinese
    'hi': 'h',    # Hindi
}

# Language to voice mapping with official Kokoro voice names (extracted from original)
KOKORO_VOICE_MAP: Dict[str, List[str]] = {
    'a': ['af_heart', 'af_bella', 'af_sarah'],                    # American English (best quality voices)
    'b': ['bf_emma', 'bf_isabella', 'bm_george'],                 # British English  
    'i': ['if_sara', 'im_nicola'],                                # Italian
    'e': ['ef_dora', 'em_alex', 'em_santa'],                      # Spanish
    'f': ['ff_siwis'],                                             # French (only one voice)
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],                      # Brazilian Portuguese
    'j': ['jf_alpha', 'jf_gongitsune', 'jm_kumo'],               # Japanese
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zm_yunjian', 'zm_yunxi'],  # Mandarin Chinese
    'h': ['hf_alpha', 'hf_beta', 'hm_omega'],                     # Hindi
}

# TTS language mapping for Kokoro (extracted from original)
KOKORO_TTS_LANG_MAP: Dict[str, str] = {
    'a': 'en-us',  # American English
    'b': 'en-gb',  # British English
    'i': 'it',     # Italian
    'e': 'es',     # Spanish
    'f': 'fr-fr',  # French
    'p': 'pt-br',  # Brazilian Portuguese
    'j': 'ja-jp',  # Japanese
    'z': 'zh-cn',  # Mandarin Chinese
    'h': 'hi-in',  # Hindi
}

# Default language setting (extracted from original)
DEFAULT_LANGUAGE = 'a'  # American English

# Language names mapping (for display purposes)
LANGUAGE_NAMES: Dict[str, str] = {
    'a': 'American English',
    'b': 'British English',
    'i': 'Italian',
    'e': 'Spanish',
    'f': 'French',
    'p': 'Portuguese',
    'j': 'Japanese',
    'z': 'Chinese',
    'h': 'Hindi'
}

# Language abbreviations mapping (for compact display)
LANGUAGE_ABBREVIATIONS: Dict[str, str] = {
    'a': 'EN-US',
    'b': 'EN-GB',
    'i': 'IT',
    'e': 'ES',
    'f': 'FR',
    'p': 'PT-BR',
    'j': 'JA',
    'z': 'ZH',
    'h': 'HI'
}


def get_kokoro_language(whisper_lang: str) -> str:
    """Convert Whisper language code to Kokoro language code.
    
    Args:
        whisper_lang: Language code from Whisper STT
        
    Returns:
        str: Corresponding Kokoro language code, defaults to 'a' (American English)
    """
    return WHISPER_TO_KOKORO_LANG.get(whisper_lang, DEFAULT_LANGUAGE)


def get_available_voices(kokoro_lang: str) -> List[str]:
    """Get available voices for a Kokoro language code.
    
    Args:
        kokoro_lang: Kokoro language code
        
    Returns:
        List[str]: List of available voice names, defaults to American English voices
    """
    return KOKORO_VOICE_MAP.get(kokoro_lang, KOKORO_VOICE_MAP[DEFAULT_LANGUAGE])


def get_tts_language(kokoro_lang: str) -> str:
    """Get TTS language string for Kokoro language code.
    
    Args:
        kokoro_lang: Kokoro language code
        
    Returns:
        str: TTS language string, defaults to 'en-us'
    """
    return KOKORO_TTS_LANG_MAP.get(kokoro_lang, KOKORO_TTS_LANG_MAP[DEFAULT_LANGUAGE])


def is_language_supported(whisper_lang: str) -> bool:
    """Check if a Whisper language is supported.
    
    Args:
        whisper_lang: Language code from Whisper STT
        
    Returns:
        bool: True if language is supported, False otherwise
    """
    return whisper_lang in WHISPER_TO_KOKORO_LANG


def get_supported_languages() -> List[str]:
    """Get list of all supported Whisper language codes.
    
    Returns:
        List[str]: List of supported language codes
    """
    return list(WHISPER_TO_KOKORO_LANG.keys())
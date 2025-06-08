"""Voice mapping logic for FastRTC Voice Assistant."""

from typing import List, Dict, Optional
from ...config.language_config import KOKORO_VOICE_MAP, DEFAULT_LANGUAGE
from ...utils.logging import get_logger

logger = get_logger(__name__)


class VoiceMapper:
    """Maps languages to appropriate voices for TTS synthesis."""
    
    def __init__(self, voice_map: Optional[Dict[str, List[str]]] = None):
        """Initialize voice mapper.
        
        Args:
            voice_map: Custom voice mapping (uses default if None)
        """
        self.voice_map = voice_map or KOKORO_VOICE_MAP.copy()
        self.fallback_language = DEFAULT_LANGUAGE
        
        logger.info(f"🎤 Voice mapper initialized with {len(self.voice_map)} language mappings")
    
    def get_voices_for_language(self, language: str) -> List[str]:
        """Get appropriate voices for the detected language.
        
        Args:
            language: Language code (e.g., 'i', 'e', 'f', 'p', 'a')
            
        Returns:
            List[str]: List of voice identifiers for the language
        """
        # Get voices for the requested language
        voices = self.voice_map.get(language, [])
        
        # If no voices found, fall back to default language
        if not voices:
            logger.warning(f"No voices found for language '{language}', falling back to '{self.fallback_language}'")
            voices = self.voice_map.get(self.fallback_language, [])
        
        # Return a copy to prevent external modification
        return voices.copy() if voices else []
    
    def get_primary_voice(self, language: str) -> Optional[str]:
        """Get the primary (first) voice for a language.
        
        Args:
            language: Language code
            
        Returns:
            Optional[str]: Primary voice identifier or None if no voices available
        """
        voices = self.get_voices_for_language(language)
        return voices[0] if voices else None
    
    def get_voice_with_fallback(self, language: str, preferred_voice: Optional[str] = None) -> List[str]:
        """Get voices with fallback strategy for TTS synthesis.
        
        Args:
            language: Language code
            preferred_voice: Preferred voice identifier (optional)
            
        Returns:
            List[str]: Ordered list of voices to try (includes None as final fallback)
        """
        voices = self.get_voices_for_language(language)
        
        # Build ordered list of voices to try
        voices_to_try = []
        
        # 1. Add preferred voice first if specified and available
        if preferred_voice and preferred_voice in voices:
            voices_to_try.append(preferred_voice)
            # Remove from main list to avoid duplication
            voices = [v for v in voices if v != preferred_voice]
        
        # 2. Add remaining voices for the language
        voices_to_try.extend(voices)
        
        # 3. Add None as final fallback (uses TTS engine default)
        voices_to_try.append(None)
        
        logger.debug(f"Voice fallback order for '{language}': {voices_to_try[:3]}...")
        return voices_to_try
    
    def is_voice_available(self, language: str, voice: str) -> bool:
        """Check if a voice is available for a language.
        
        Args:
            language: Language code
            voice: Voice identifier
            
        Returns:
            bool: True if voice is available for the language
        """
        voices = self.get_voices_for_language(language)
        return voice in voices
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes.
        
        Returns:
            List[str]: List of supported language codes
        """
        return list(self.voice_map.keys())
    
    def get_language_info(self, language: str) -> Dict[str, any]:
        """Get detailed information about a language.
        
        Args:
            language: Language code
            
        Returns:
            Dict[str, any]: Language information including voices and metadata
        """
        voices = self.get_voices_for_language(language)
        
        # Language name mapping
        language_names = {
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
        
        return {
            'code': language,
            'name': language_names.get(language, f'Language {language}'),
            'voices': voices,
            'voice_count': len(voices),
            'primary_voice': voices[0] if voices else None,
            'is_supported': len(voices) > 0
        }
    
    def get_all_language_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about all supported languages.
        
        Returns:
            Dict[str, Dict[str, any]]: Mapping of language codes to language info
        """
        return {
            lang: self.get_language_info(lang) 
            for lang in self.get_supported_languages()
        }
    
    def add_voice_mapping(self, language: str, voices: List[str]) -> None:
        """Add or update voice mapping for a language.
        
        Args:
            language: Language code
            voices: List of voice identifiers
        """
        self.voice_map[language] = voices.copy()
        logger.info(f"Updated voice mapping for '{language}': {len(voices)} voices")
    
    def remove_voice_mapping(self, language: str) -> bool:
        """Remove voice mapping for a language.
        
        Args:
            language: Language code
            
        Returns:
            bool: True if mapping was removed, False if it didn't exist
        """
        if language in self.voice_map:
            del self.voice_map[language]
            logger.info(f"Removed voice mapping for '{language}'")
            return True
        return False
    
    def validate_voice_map(self) -> Dict[str, List[str]]:
        """Validate the current voice map and return any issues.
        
        Returns:
            Dict[str, List[str]]: Mapping of language codes to validation issues
        """
        issues = {}
        
        for language, voices in self.voice_map.items():
            lang_issues = []
            
            if not voices:
                lang_issues.append("No voices defined")
            elif not isinstance(voices, list):
                lang_issues.append("Voices must be a list")
            else:
                for voice in voices:
                    if not isinstance(voice, str):
                        lang_issues.append(f"Invalid voice type: {type(voice)}")
                    elif not voice.strip():
                        lang_issues.append("Empty voice identifier")
            
            if lang_issues:
                issues[language] = lang_issues
        
        # Check for fallback language
        if self.fallback_language not in self.voice_map:
            issues['_fallback'] = [f"Fallback language '{self.fallback_language}' not in voice map"]
        
        return issues
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the voice mapping.
        
        Returns:
            Dict[str, any]: Voice mapping statistics
        """
        total_voices = sum(len(voices) for voices in self.voice_map.values())
        languages_with_voices = sum(1 for voices in self.voice_map.values() if voices)
        
        return {
            'total_languages': len(self.voice_map),
            'languages_with_voices': languages_with_voices,
            'total_voices': total_voices,
            'avg_voices_per_language': total_voices / len(self.voice_map) if self.voice_map else 0,
            'fallback_language': self.fallback_language,
            'validation_issues': len(self.validate_voice_map())
        }
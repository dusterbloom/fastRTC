"""Unit tests for language detection components."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.audio.language.detector import (
    MediaPipeLanguageDetector,
    KeywordLanguageDetector,
    HybridLanguageDetector
)
from src.audio.language.voice_mapper import VoiceMapper
from src.config.language_config import DEFAULT_LANGUAGE


class TestKeywordLanguageDetector:
    """Test cases for KeywordLanguageDetector."""
    
    def test_keyword_detector_initialization(self):
        """Test keyword detector initialization."""
        detector = KeywordLanguageDetector()
        
        assert detector.is_available() is True
        assert 'i' in detector.language_keywords  # Italian
        assert 'e' in detector.language_keywords  # Spanish
        assert 'f' in detector.language_keywords  # French
        assert 'p' in detector.language_keywords  # Portuguese
    
    @pytest.mark.parametrize("text,expected_lang,min_confidence", [
        ("ciao come stai molto bene", "i", 0.2),  # Italian
        ("hola como estas muy bien", "e", 0.2),   # Spanish
        ("bonjour comment allez vous", "f", 0.2), # French
        ("ola como vai muito bem", "p", 0.2),     # Portuguese
        ("konnichiwa arigatou gozaimasu", "j", 0.2), # Japanese
        ("nihao xiexie zaijian", "z", 0.2),       # Chinese
        ("namaste dhanyawad kaise", "h", 0.2),    # Hindi
    ])
    def test_keyword_detector_language_detection(self, text, expected_lang, min_confidence):
        """Test keyword-based language detection."""
        detector = KeywordLanguageDetector()
        
        lang, confidence = detector.detect_language(text)
        
        assert lang == expected_lang
        assert confidence >= min_confidence
    
    def test_keyword_detector_insufficient_matches(self):
        """Test behavior with insufficient keyword matches."""
        detector = KeywordLanguageDetector()
        
        # Single keyword - should not be confident enough for multi-word detection
        # but "ciao" appears twice in Italian keywords, so it will match
        lang, confidence = detector.detect_language("hello world")  # No matches
        assert lang == DEFAULT_LANGUAGE
        assert confidence == 0.0
    
    def test_keyword_detector_single_word_detection(self):
        """Test single word language detection."""
        detector = KeywordLanguageDetector()
        
        test_cases = [
            ("I want to speak italiano", "i"),
            ("Quiero hablar español", "e"), 
            ("Je veux parler français", "f"),
            ("Quero falar português", "p")
        ]
        
        for text, expected_lang in test_cases:
            lang, confidence = detector.detect_language(text)
            assert lang == expected_lang
            # Confidence should be based on keyword matches, not fixed at 0.8
            assert confidence > 0.0
    
    def test_keyword_detector_empty_text(self):
        """Test behavior with empty text."""
        detector = KeywordLanguageDetector()
        
        for empty_text in ["", "   ", None]:
            lang, confidence = detector.detect_language(empty_text or "")
            assert lang == DEFAULT_LANGUAGE
            assert confidence == 0.0
    
    def test_keyword_detector_case_insensitive(self):
        """Test case insensitive detection."""
        detector = KeywordLanguageDetector()
        
        # Test with different cases
        test_texts = [
            "CIAO COME STAI MOLTO BENE",
            "Ciao Come Stai Molto Bene", 
            "ciao come stai molto bene"
        ]
        
        for text in test_texts:
            lang, confidence = detector.detect_language(text)
            assert lang == "i"  # Italian
            assert confidence > 0.0
    
    def test_keyword_detector_mixed_languages(self):
        """Test detection with mixed languages."""
        detector = KeywordLanguageDetector()
        
        # Text with both Italian and Spanish words
        mixed_text = "ciao hola come estas molto bien"
        lang, confidence = detector.detect_language(mixed_text)
        
        # Should detect the language with more matches
        assert lang in ["i", "e"]
        assert confidence > 0.0


class TestMediaPipeLanguageDetector:
    """Test cases for MediaPipeLanguageDetector."""
    
    def test_mediapipe_detector_initialization_without_mediapipe(self):
        """Test initialization when MediaPipe is not available."""
        with patch('src.audio.language.detector.mp', side_effect=ImportError):
            detector = MediaPipeLanguageDetector()
            assert detector.is_available() is False
    
    @patch('src.audio.language.detector.mp')
    def test_mediapipe_detector_initialization_success(self, mock_mp):
        """Test successful initialization with MediaPipe."""
        # Mock MediaPipe components
        mock_detector = Mock()
        mock_text = Mock()
        mock_text.LanguageDetector.create_from_options.return_value = mock_detector
        mock_mp.tasks.python.text = mock_text
        
        detector = MediaPipeLanguageDetector()
        assert detector._detector == mock_detector
        assert detector.is_available() is True
    
    @patch('src.audio.language.detector.mp')
    def test_mediapipe_detector_language_detection(self, mock_mp):
        """Test MediaPipe language detection."""
        # Mock detection result
        mock_detection = Mock()
        mock_detection.language_code = 'it'
        mock_detection.probability = 0.95
        
        mock_result = Mock()
        mock_result.detections = [mock_detection]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_result
        
        # Mock MediaPipe setup
        mock_text = Mock()
        mock_text.LanguageDetector.create_from_options.return_value = mock_detector
        mock_mp.tasks.python.text = mock_text
        
        detector = MediaPipeLanguageDetector()
        lang, confidence = detector.detect_language("Ciao come stai")
        
        assert lang == 'i'  # Italian mapped to Kokoro code
        assert confidence == 0.95
    
    @patch('src.audio.language.detector.mp')
    def test_mediapipe_detector_no_detections(self, mock_mp):
        """Test behavior when no detections are found."""
        mock_result = Mock()
        mock_result.detections = []
        
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_result
        
        mock_text = Mock()
        mock_text.LanguageDetector.create_from_options.return_value = mock_detector
        mock_mp.tasks.python.text = mock_text
        
        detector = MediaPipeLanguageDetector()
        lang, confidence = detector.detect_language("Some text")
        
        assert lang == DEFAULT_LANGUAGE
        assert confidence == 0.0
    
    @patch('src.audio.language.detector.mp')
    def test_mediapipe_detector_language_mapping(self, mock_mp):
        """Test language code mapping from MediaPipe to Kokoro."""
        test_mappings = [
            ('en', 'a'),  # English -> American English
            ('it', 'i'),  # Italian
            ('es', 'e'),  # Spanish
            ('fr', 'f'),  # French
            ('pt', 'p'),  # Portuguese
            ('ja', 'j'),  # Japanese
            ('zh', 'z'),  # Chinese
            ('hi', 'h'),  # Hindi
            ('de', 'a'),  # German -> fallback to English
            ('unknown', 'a')  # Unknown -> fallback to English
        ]
        
        for mediapipe_lang, expected_kokoro in test_mappings:
            mock_detection = Mock()
            mock_detection.language_code = mediapipe_lang
            mock_detection.probability = 0.8
            
            mock_result = Mock()
            mock_result.detections = [mock_detection]
            
            mock_detector = Mock()
            mock_detector.detect.return_value = mock_result
            
            mock_text = Mock()
            mock_text.LanguageDetector.create_from_options.return_value = mock_detector
            mock_mp.tasks.python.text = mock_text
            
            detector = MediaPipeLanguageDetector()
            lang, confidence = detector.detect_language("Test text")
            
            assert lang == expected_kokoro
    
    @patch('src.audio.language.detector.mp')
    def test_mediapipe_detector_error_handling(self, mock_mp):
        """Test error handling in MediaPipe detection."""
        mock_detector = Mock()
        mock_detector.detect.side_effect = Exception("Detection failed")
        
        mock_text = Mock()
        mock_text.LanguageDetector.create_from_options.return_value = mock_detector
        mock_mp.tasks.python.text = mock_text
        
        detector = MediaPipeLanguageDetector()
        lang, confidence = detector.detect_language("Test text")
        
        assert lang == DEFAULT_LANGUAGE
        assert confidence == 0.0


class TestHybridLanguageDetector:
    """Test cases for HybridLanguageDetector."""
    
    def test_hybrid_detector_initialization(self):
        """Test hybrid detector initialization."""
        detector = HybridLanguageDetector()
        
        assert detector.confidence_threshold == 0.7
        assert detector.mediapipe_detector is not None
        assert detector.keyword_detector is not None
        assert detector.is_available() is True  # Keyword detector is always available
    
    def test_hybrid_detector_high_confidence_mediapipe(self):
        """Test using MediaPipe when confidence is high."""
        detector = HybridLanguageDetector(confidence_threshold=0.7)
        
        # Mock MediaPipe to return high confidence
        with patch.object(detector.mediapipe_detector, 'is_available', return_value=True), \
             patch.object(detector.mediapipe_detector, 'detect_language', return_value=('i', 0.9)):
            
            lang, confidence = detector.detect_language("Ciao come stai")
            
            assert lang == 'i'
            assert confidence == 0.9
    
    def test_hybrid_detector_low_confidence_fallback(self):
        """Test fallback to keyword detection when MediaPipe confidence is low."""
        detector = HybridLanguageDetector(confidence_threshold=0.7)
        
        # Mock MediaPipe to return low confidence
        with patch.object(detector.mediapipe_detector, 'is_available', return_value=True), \
             patch.object(detector.mediapipe_detector, 'detect_language', return_value=('a', 0.3)), \
             patch.object(detector.keyword_detector, 'detect_language', return_value=('i', 0.6)):
            
            lang, confidence = detector.detect_language("ciao come stai molto bene")
            
            assert lang == 'i'
            assert confidence == 0.6
    
    def test_hybrid_detector_mediapipe_unavailable(self):
        """Test fallback when MediaPipe is unavailable."""
        detector = HybridLanguageDetector()
        
        # Mock MediaPipe as unavailable
        with patch.object(detector.mediapipe_detector, 'is_available', return_value=False), \
             patch.object(detector.keyword_detector, 'detect_language', return_value=('e', 0.5)):
            
            lang, confidence = detector.detect_language("hola como estas")
            
            assert lang == 'e'
            assert confidence == 0.5
    
    def test_hybrid_detector_status(self):
        """Test detector status reporting."""
        detector = HybridLanguageDetector()
        
        with patch.object(detector.mediapipe_detector, 'is_available', return_value=True), \
             patch.object(detector.keyword_detector, 'is_available', return_value=True):
            
            status = detector.get_detector_status()
            
            assert status['mediapipe'] is True
            assert status['keyword'] is True
            assert status['hybrid'] is True


class TestVoiceMapper:
    """Test cases for VoiceMapper."""
    
    def test_voice_mapper_initialization(self):
        """Test voice mapper initialization."""
        mapper = VoiceMapper()
        
        assert len(mapper.voice_map) > 0
        assert mapper.fallback_language == DEFAULT_LANGUAGE
        assert mapper.get_supported_languages()
    
    def test_voice_mapper_custom_mapping(self):
        """Test voice mapper with custom mapping."""
        custom_map = {
            'test_lang': ['voice1', 'voice2'],
            'another_lang': ['voice3']
        }
        
        mapper = VoiceMapper(voice_map=custom_map)
        
        assert mapper.voice_map == custom_map
        assert mapper.get_supported_languages() == ['test_lang', 'another_lang']
    
    def test_voice_mapper_get_voices_for_language(self):
        """Test getting voices for a language."""
        test_map = {
            'i': ['italian_voice1', 'italian_voice2'],
            'e': ['spanish_voice1']
        }
        
        mapper = VoiceMapper(voice_map=test_map)
        
        # Test existing language
        voices = mapper.get_voices_for_language('i')
        assert voices == ['italian_voice1', 'italian_voice2']
        
        # Test non-existing language (should fallback)
        voices = mapper.get_voices_for_language('unknown')
        assert voices == []  # No fallback in test map
    
    def test_voice_mapper_primary_voice(self):
        """Test getting primary voice."""
        test_map = {
            'i': ['voice1', 'voice2', 'voice3']
        }
        
        mapper = VoiceMapper(voice_map=test_map)
        
        primary = mapper.get_primary_voice('i')
        assert primary == 'voice1'
        
        primary = mapper.get_primary_voice('unknown')
        assert primary is None
    
    def test_voice_mapper_voice_with_fallback(self):
        """Test voice selection with fallback strategy."""
        test_map = {
            'i': ['voice1', 'voice2', 'voice3']
        }
        
        mapper = VoiceMapper(voice_map=test_map)
        
        # Test with preferred voice
        voices = mapper.get_voice_with_fallback('i', 'voice2')
        assert voices[0] == 'voice2'  # Preferred first
        assert 'voice1' in voices
        assert 'voice3' in voices
        assert voices[-1] is None  # None as final fallback
        
        # Test without preferred voice
        voices = mapper.get_voice_with_fallback('i')
        assert voices[0] == 'voice1'
        assert voices[-1] is None
    
    def test_voice_mapper_voice_availability(self):
        """Test voice availability checking."""
        test_map = {
            'i': ['voice1', 'voice2']
        }
        
        mapper = VoiceMapper(voice_map=test_map)
        
        assert mapper.is_voice_available('i', 'voice1') is True
        assert mapper.is_voice_available('i', 'voice3') is False
        assert mapper.is_voice_available('unknown', 'voice1') is False
    
    def test_voice_mapper_language_info(self):
        """Test language information retrieval."""
        test_map = {
            'i': ['voice1', 'voice2']
        }
        
        mapper = VoiceMapper(voice_map=test_map)
        
        info = mapper.get_language_info('i')
        
        assert info['code'] == 'i'
        assert info['name'] == 'Italian'
        assert info['voices'] == ['voice1', 'voice2']
        assert info['voice_count'] == 2
        assert info['primary_voice'] == 'voice1'
        assert info['is_supported'] is True
    
    def test_voice_mapper_add_remove_mapping(self):
        """Test adding and removing voice mappings."""
        # Start with a minimal voice map that has fallback language
        mapper = VoiceMapper(voice_map={'a': ['fallback_voice']})
        
        # Add mapping
        mapper.add_voice_mapping('test', ['voice1', 'voice2'])
        assert mapper.get_voices_for_language('test') == ['voice1', 'voice2']
        
        # Remove mapping
        removed = mapper.remove_voice_mapping('test')
        assert removed is True
        # After removal, should fallback to default language voices
        assert mapper.get_voices_for_language('test') == ['fallback_voice']
        
        # Try to remove non-existing
        removed = mapper.remove_voice_mapping('nonexistent')
        assert removed is False
    
    def test_voice_mapper_validation(self):
        """Test voice map validation."""
        # Valid map
        valid_map = {
            'i': ['voice1', 'voice2'],
            'e': ['voice3']
        }
        mapper = VoiceMapper(voice_map=valid_map)
        issues = mapper.validate_voice_map()
        assert len(issues) == 1  # Only fallback language issue
        
        # Invalid map
        invalid_map = {
            'i': [],  # No voices
            'e': 'not_a_list',  # Wrong type
            'f': ['voice1', '']  # Empty voice
        }
        mapper = VoiceMapper(voice_map=invalid_map)
        issues = mapper.validate_voice_map()
        assert len(issues) > 1
    
    def test_voice_mapper_stats(self):
        """Test voice mapping statistics."""
        test_map = {
            'i': ['voice1', 'voice2'],
            'e': ['voice3'],
            'f': []  # No voices
        }
        
        mapper = VoiceMapper(voice_map=test_map)
        stats = mapper.get_stats()
        
        assert stats['total_languages'] == 3
        assert stats['languages_with_voices'] == 2
        assert stats['total_voices'] == 3
        assert stats['avg_voices_per_language'] == 1.0
        assert stats['validation_issues'] > 0  # Empty voices + fallback
"""Language detection implementations for FastRTC Voice Assistant."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from ...core.interfaces import LanguageDetector
from ...config.language_config import DEFAULT_LANGUAGE
from ...utils.logging import get_logger

logger = get_logger(__name__)


class MediaPipeLanguageDetector(LanguageDetector):
    """MediaPipe-based language detection with high accuracy."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize MediaPipe language detector.
        
        Args:
            model_path: Path to MediaPipe language detection model
        """
        self.model_path = model_path or self._ensure_model_downloaded()
        self._detector = None
        self._initialize_detector()
        
        # Language code mapping from MediaPipe to Kokoro TTS
        self.mediapipe_to_kokoro = {
            'en': 'a',    # English -> American English
            'it': 'i',    # Italian
            'es': 'e',    # Spanish
            'fr': 'f',    # French
            'de': 'a',    # German -> fallback to English (not natively supported)
            'pt': 'p',    # Portuguese -> Brazilian Portuguese
            'ja': 'j',    # Japanese
            'ko': 'a',    # Korean -> fallback to English
            'zh': 'z',    # Chinese -> Mandarin Chinese
            'hi': 'h',    # Hindi
            'ru': 'a',    # Russian -> fallback to English
            'ar': 'a',    # Arabic -> fallback to English
        }
    
    def _ensure_model_downloaded(self) -> str:
        """Download MediaPipe language detection model if not present.
        
        Returns:
            str: Path to the model file
        """
        model_dir = Path("models/mediapipe")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "language_detector.tflite"
        
        if not model_path.exists():
            try:
                import requests
                logger.info("Downloading MediaPipe language detection model...")
                url = "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite"
                
                response = requests.get(url)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download MediaPipe model: {e}")
                # Return a placeholder path - detector will handle the missing file
                pass
        
        return str(model_path)
    
    def _initialize_detector(self) -> None:
        """Initialize the MediaPipe language detector."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import text
            
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = text.LanguageDetectorOptions(base_options=base_options)
            self._detector = text.LanguageDetector.create_from_options(options)
            logger.info("✅ MediaPipe language detector initialized")
            
        except ImportError as e:
            logger.warning(f"MediaPipe not available: {e}")
            logger.warning("Install with: pip install mediapipe")
            self._detector = None
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe language detector: {e}")
            self._detector = None
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language from text using MediaPipe.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple[str, float]: (kokoro_language_code, confidence_score)
        """
        if not text or not text.strip():
            return DEFAULT_LANGUAGE, 0.0
        
        if not self._detector:
            logger.warning("MediaPipe detector not available, falling back to default")
            return DEFAULT_LANGUAGE, 0.0
        
        try:
            # Get detection results from MediaPipe
            detection_result = self._detector.detect(text.strip())
            
            if not detection_result.detections:
                return DEFAULT_LANGUAGE, 0.0
            
            # Get the most confident detection
            best_detection = max(detection_result.detections, key=lambda d: d.probability)
            
            # Map MediaPipe language code to Kokoro language code
            mediapipe_lang = best_detection.language_code
            kokoro_lang = self.mediapipe_to_kokoro.get(mediapipe_lang, DEFAULT_LANGUAGE)
            confidence = best_detection.probability
            
            logger.info(f"🔍 MediaPipe detected: {mediapipe_lang} -> {kokoro_lang} (confidence: {confidence:.3f})")
            
            return kokoro_lang, confidence
            
        except Exception as e:
            logger.error(f"MediaPipe language detection error: {e}")
            return DEFAULT_LANGUAGE, 0.0
    
    def is_available(self) -> bool:
        """Check if the language detector is available and ready.
        
        Returns:
            bool: True if detector is ready, False otherwise
        """
        return self._detector is not None


class KeywordLanguageDetector(LanguageDetector):
    """Fallback keyword-based language detection (legacy approach)."""
    
    def __init__(self):
        """Initialize keyword-based language detector."""
        # Expanded language keyword sets from original implementation
        self.language_keywords = {
            'i': [  # Italian
                'ciao', 'grazie', 'prego', 'bene', 'come stai', 'buongiorno', 'buonasera',
                'molto', 'sono', 'dove', 'voglio', 'che', 'parli', 'italiano', 'parlare',
                'posso', 'puoi', 'sì', 'no', 'questo', 'quello', 'quando', 'perché',
                'come', 'cosa', 'chi', 'quale', 'anche', 'ancora', 'dopo', 'prima',
                'sempre', 'mai', 'già', 'oggi', 'ieri', 'domani', 'casa', 'lavoro',
                'famiglia', 'amico', 'tempo', 'bello', 'buono', 'grande', 'piccolo',
                'nuovo', 'vecchio', 'fare', 'dire', 'andare', 'venire', 'vedere',
                'sapere', 'dare', 'volere', 'dovere', 'potere', 'stare', 'avere',
                'essere', 'mi', 'ti', 'ci', 'vi', 'lo', 'la', 'li', 'le', 'gli',
                'ne', 'con', 'per', 'da', 'in', 'su', 'di', 'del', 'della', 'dello'
            ],
            'e': [  # Spanish
                'hola', 'gracias', 'por favor', 'bueno', 'como estas', 'buenos dias',
                'muy', 'soy', 'donde', 'quiero', 'que', 'hablar', 'español', 'puedo',
                'puedes', 'sí', 'no', 'este', 'ese', 'cuando', 'porque', 'como',
                'que', 'quien', 'cual', 'también', 'todavia', 'después', 'antes',
                'siempre', 'nunca', 'ya', 'hoy', 'ayer', 'mañana', 'casa', 'trabajo',
                'familia', 'amigo', 'tiempo', 'hacer', 'decir', 'ir', 'venir', 'ver',
                'saber', 'dar', 'querer', 'deber', 'poder', 'estar', 'tener', 'ser'
            ],
            'f': [  # French
                'bonjour', 'merci', 'comment allez', 'tres bien', 'bonsoir', 'je suis',
                'tres', 'ou', 'veux', 'que', 'parler', 'français', 'peux', 'pouvez',
                'oui', 'non', 'ce', 'cette', 'quand', 'pourquoi', 'comment', 'quoi',
                'qui', 'quel', 'aussi', 'encore', 'après', 'avant', 'toujours',
                'jamais', 'déjà', 'aujourd', 'hier', 'demain', 'maison', 'travail',
                'famille', 'ami', 'temps', 'faire', 'dire', 'aller', 'venir', 'voir',
                'savoir', 'donner', 'vouloir', 'devoir', 'pouvoir', 'être', 'avoir'
            ],
            'p': [  # Portuguese
                'ola', 'obrigado', 'obrigada', 'por favor', 'bom dia', 'como vai',
                'muito', 'sou', 'onde', 'quero', 'que', 'falar', 'português', 'posso',
                'pode', 'sim', 'não', 'este', 'esse', 'quando', 'porque', 'como',
                'que', 'quem', 'qual', 'também', 'ainda', 'depois', 'antes', 'sempre',
                'nunca', 'já', 'hoje', 'ontem', 'amanhã', 'casa', 'trabalho',
                'família', 'amigo', 'tempo', 'fazer', 'dizer', 'ir', 'vir', 'ver',
                'saber', 'dar', 'querer', 'dever', 'poder', 'estar', 'ter', 'ser'
            ],
            'j': [  # Japanese
                'konnichiwa', 'arigatou', 'sumimasen', 'hajimemashite', 'sayonara',
                'watashi', 'desu', 'anata', 'kore', 'sore', 'doko', 'nani', 'dare',
                'itsu', 'doushite', 'hai', 'iie'
            ],
            'z': [  # Chinese
                'nihao', 'xiexie', 'zaijian', 'duibuqi', 'wo', 'shi', 'nali', 'shenme',
                'shei', 'nali', 'weishenme', 'shi', 'bu'
            ],
            'h': [  # Hindi
                'namaste', 'dhanyawad', 'kaise', 'aap', 'main', 'hoon', 'kahan',
                'kya', 'kaun', 'kab', 'kyun', 'haan', 'nahin'
            ]
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language from text using keyword matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple[str, float]: (language_code, confidence_score)
        """
        if not text or not text.strip():
            return DEFAULT_LANGUAGE, 0.0
        
        text_lower = text.lower()
        
        # Count matches for each language
        language_matches = {}
        for lang_code, keywords in self.language_keywords.items():
            matches = sum(1 for word in keywords if word in text_lower)
            if matches > 0:
                language_matches[lang_code] = matches
        
        # Debug logging
        if language_matches:
            match_str = ", ".join([f"{lang}:{count}" for lang, count in language_matches.items()])
            logger.info(f"🔍 Keyword detection - {match_str}")
        
        # Require at least 2 matches to be confident, return the highest
        if language_matches:
            max_matches = max(language_matches.values())
            if max_matches >= 2:
                # Find language with highest matches, break ties by preferring non-Italian
                candidates = [lang for lang, count in language_matches.items() if count == max_matches]
                if len(candidates) == 1:
                    best_language = candidates[0]
                else:
                    # If there's a tie, prefer non-Italian languages to avoid false positives
                    non_italian = [lang for lang in candidates if lang != 'i']
                    best_language = non_italian[0] if non_italian else candidates[0]
                
                # Calculate confidence based on number of matches
                confidence = min(0.9, max_matches * 0.1)  # Cap at 0.9, scale by matches
                return best_language, confidence
        
        # Single word detection for key phrases
        single_word_detections = {
            'i': ['italiano', 'italiana'],
            'e': ['español', 'castellano'],
            'f': ['français', 'francais'],
            'p': ['português', 'portugues'],
            'j': ['nihongo', 'japanese'],
            'z': ['zhongwen', 'chinese'],
            'h': ['hindi', 'hindustani']
        }
        
        for lang_code, phrases in single_word_detections.items():
            if any(phrase in text_lower for phrase in phrases):
                logger.info(f"🔍 Single word detection: {lang_code}")
                return lang_code, 0.8
        
        # Default to American English
        return DEFAULT_LANGUAGE, 0.0
    
    def is_available(self) -> bool:
        """Check if the language detector is available and ready.
        
        Returns:
            bool: True if detector is ready, False otherwise
        """
        return True  # Keyword detector is always available


class HybridLanguageDetector(LanguageDetector):
    """Hybrid language detector combining MediaPipe with keyword fallback."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize hybrid language detector.
        
        Args:
            confidence_threshold: Minimum confidence for MediaPipe detection
        """
        self.confidence_threshold = confidence_threshold
        self.mediapipe_detector = MediaPipeLanguageDetector()
        self.keyword_detector = KeywordLanguageDetector()
        
        logger.info(f"🔍 Hybrid language detector initialized (threshold: {confidence_threshold})")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language using MediaPipe with keyword fallback.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple[str, float]: (language_code, confidence_score)
        """
        if not text or not text.strip():
            return DEFAULT_LANGUAGE, 0.0
        
        # Try MediaPipe first if available
        if self.mediapipe_detector.is_available():
            lang_code, confidence = self.mediapipe_detector.detect_language(text)
            
            if confidence >= self.confidence_threshold:
                logger.info(f"🎯 MediaPipe detection: {lang_code} (confidence: {confidence:.3f})")
                return lang_code, confidence
            else:
                logger.info(f"🔄 MediaPipe confidence too low ({confidence:.3f}), trying keyword fallback")
        else:
            logger.info("🔄 MediaPipe not available, using keyword detection")
        
        # Fall back to keyword detection
        lang_code, confidence = self.keyword_detector.detect_language(text)
        logger.info(f"🔤 Keyword detection: {lang_code} (confidence: {confidence:.3f})")
        
        return lang_code, confidence
    
    def is_available(self) -> bool:
        """Check if the language detector is available and ready.
        
        Returns:
            bool: True if detector is ready, False otherwise
        """
        # Hybrid detector is available if at least one method is available
        return self.mediapipe_detector.is_available() or self.keyword_detector.is_available()
    
    def get_detector_status(self) -> Dict[str, bool]:
        """Get status of individual detectors.
        
        Returns:
            Dict[str, bool]: Status of each detector
        """
        return {
            'mediapipe': self.mediapipe_detector.is_available(),
            'keyword': self.keyword_detector.is_available(),
            'hybrid': self.is_available()
        }
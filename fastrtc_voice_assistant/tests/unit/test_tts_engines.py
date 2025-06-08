"""Unit tests for TTS engines."""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.core.interfaces import AudioData
from src.core.exceptions import TTSError
from src.audio.engines.tts.base import BaseTTSEngine
from src.audio.engines.tts.kokoro_tts import KokoroTTSEngine, KokoroTTSOptions


class TestBaseTTSEngine:
    """Test cases for BaseTTSEngine."""
    
    def test_base_tts_initialization(self):
        """Test base TTS engine initialization."""
        # Create concrete implementation for testing
        class TestTTSEngine(BaseTTSEngine):
            async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
                samples = np.random.random(1000).astype(np.float32)
                return AudioData(samples=samples, sample_rate=24000, duration=1000/24000)
            
            def get_available_voices(self, language: str) -> list:
                return ["voice1", "voice2"]
        
        engine = TestTTSEngine()
        
        assert engine.stats['syntheses'] == 0
        assert engine.stats['total_text_length'] == 0
        assert engine.stats['total_audio_duration'] == 0.0
        assert engine.stats['total_processing_time'] == 0.0
        assert engine.stats['errors'] == 0
        assert engine.stats['last_synthesis'] is None
        assert engine.is_available() is False  # Not set as available yet
    
    @pytest.mark.asyncio
    async def test_base_tts_synthesis_success(self):
        """Test successful synthesis with stats tracking."""
        class TestTTSEngine(BaseTTSEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
                await asyncio.sleep(0.001)  # Small delay for timing
                samples = np.random.random(1000).astype(np.float32)
                return AudioData(samples=samples, sample_rate=24000, duration=1000/24000)
            
            def get_available_voices(self, language: str) -> list:
                return ["voice1"]
        
        engine = TestTTSEngine()
        
        # Synthesize
        result = await engine.synthesize("Hello world", "voice1", "en")
        
        # Check result
        assert isinstance(result, AudioData)
        assert len(result.samples) == 1000
        assert result.sample_rate == 24000
        
        # Check stats
        stats = engine.get_stats()
        assert stats['syntheses'] == 1
        assert stats['total_text_length'] == len("Hello world")
        assert stats['total_audio_duration'] == result.duration
        assert stats['total_processing_time'] > 0
        assert stats['avg_processing_time'] > 0
        assert stats['real_time_factor'] > 0
        assert stats['chars_per_second'] > 0
        assert stats['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_base_tts_synthesis_failure(self):
        """Test synthesis failure handling."""
        class FailingTTSEngine(BaseTTSEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
                raise ValueError("Synthesis failed")
            
            def get_available_voices(self, language: str) -> list:
                return ["voice1"]
        
        engine = FailingTTSEngine()
        
        with pytest.raises(TTSError):
            await engine.synthesize("Hello", "voice1", "en")
        
        # Check error stats
        stats = engine.get_stats()
        assert stats['errors'] == 1
        assert stats['syntheses'] == 0
        assert stats['success_rate'] == 0.0
    
    @pytest.mark.asyncio
    async def test_base_tts_unavailable_engine(self):
        """Test synthesis with unavailable engine."""
        class TestTTSEngine(BaseTTSEngine):
            async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
                samples = np.random.random(100).astype(np.float32)
                return AudioData(samples=samples, sample_rate=24000, duration=100/24000)
            
            def get_available_voices(self, language: str) -> list:
                return ["voice1"]
        
        engine = TestTTSEngine()  # Not set as available
        
        with pytest.raises(TTSError, match="not available"):
            await engine.synthesize("Hello", "voice1", "en")
    
    @pytest.mark.asyncio
    async def test_base_tts_empty_text(self):
        """Test synthesis with empty text."""
        class TestTTSEngine(BaseTTSEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
                samples = np.random.random(100).astype(np.float32)
                return AudioData(samples=samples, sample_rate=24000, duration=100/24000)
            
            def get_available_voices(self, language: str) -> list:
                return ["voice1"]
        
        engine = TestTTSEngine()
        
        # Test empty and whitespace-only text
        for empty_text in ["", "   ", None]:
            with pytest.raises(TTSError, match="cannot be empty"):
                await engine.synthesize(empty_text or "", "voice1", "en")
    
    def test_base_tts_stats_reset(self):
        """Test statistics reset functionality."""
        class TestTTSEngine(BaseTTSEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
                samples = np.random.random(100).astype(np.float32)
                return AudioData(samples=samples, sample_rate=24000, duration=100/24000)
            
            def get_available_voices(self, language: str) -> list:
                return ["voice1"]
        
        engine = TestTTSEngine()
        
        # Manually update stats
        engine.stats['syntheses'] = 5
        engine.stats['errors'] = 2
        
        # Reset stats
        engine.reset_stats()
        
        # Verify reset
        assert engine.stats['syntheses'] == 0
        assert engine.stats['errors'] == 0
        assert engine.stats['total_text_length'] == 0


class TestKokoroTTSOptions:
    """Test cases for KokoroTTSOptions."""
    
    def test_kokoro_tts_options_defaults(self):
        """Test default options."""
        options = KokoroTTSOptions()
        
        assert options.speed == 1.0
        assert options.lang == "en-us"
        assert options.voice is None
    
    def test_kokoro_tts_options_custom(self):
        """Test custom options."""
        options = KokoroTTSOptions(
            speed=1.2,
            lang="it-it",
            voice="italian_voice"
        )
        
        assert options.speed == 1.2
        assert options.lang == "it-it"
        assert options.voice == "italian_voice"


class TestKokoroTTSEngine:
    """Test cases for KokoroTTSEngine."""
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_initialization_success(self, mock_kokoro_class):
        """Test successful Kokoro TTS initialization."""
        # Mock KokoroONNX
        mock_kokoro_instance = Mock()
        mock_kokoro_instance.model.voices = ["voice1", "voice2", "voice3"]
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        assert engine.tts_model == mock_kokoro_instance
        assert engine.is_available() is True
    
    def test_kokoro_tts_initialization_import_error(self):
        """Test initialization when Kokoro is not available."""
        with patch('builtins.__import__', side_effect=ImportError("kokoro_onnx not found")):
            engine = KokoroTTSEngine()
            assert engine.is_available() is False
            assert engine.tts_model is None
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_initialization_general_error(self, mock_kokoro_class):
        """Test initialization with general error."""
        mock_kokoro_class.side_effect = Exception("Initialization failed")
        
        engine = KokoroTTSEngine()
        assert engine.is_available() is False
        assert engine.tts_model is None
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    @pytest.mark.asyncio
    async def test_kokoro_tts_synthesis_success(self, mock_kokoro_class):
        """Test successful synthesis."""
        # Mock TTS model
        mock_kokoro_instance = Mock()
        
        # Mock streaming output
        def mock_stream_tts_sync(text, options):
            # Yield some audio chunks
            yield (24000, np.random.random(1000).astype(np.float32))
            yield (24000, np.random.random(500).astype(np.float32))
        
        mock_kokoro_instance.stream_tts_sync = mock_stream_tts_sync
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        # Synthesize
        result = await engine.synthesize("Hello world", "voice1", "i")
        
        # Check result
        assert isinstance(result, AudioData)
        assert result.sample_rate == 24000
        assert len(result.samples) == 1500  # 1000 + 500
        assert result.duration > 0
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    @pytest.mark.asyncio
    async def test_kokoro_tts_synthesis_no_audio(self, mock_kokoro_class):
        """Test synthesis when no audio is generated."""
        mock_kokoro_instance = Mock()
        mock_kokoro_instance.stream_tts_sync = lambda text, options: iter([])  # Empty generator
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        with pytest.raises(TTSError, match="No audio generated"):
            await engine.synthesize("Hello", "voice1", "en")
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    @pytest.mark.asyncio
    async def test_kokoro_tts_synthesis_failure(self, mock_kokoro_class):
        """Test synthesis failure handling."""
        mock_kokoro_instance = Mock()
        mock_kokoro_instance.stream_tts_sync.side_effect = Exception("Synthesis failed")
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        with pytest.raises(TTSError):
            await engine.synthesize("Hello", "voice1", "en")
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_get_available_voices(self, mock_kokoro_class):
        """Test getting available voices."""
        mock_kokoro_class.return_value = Mock()
        
        engine = KokoroTTSEngine()
        
        # Test with known language
        voices = engine.get_available_voices("i")  # Italian
        assert isinstance(voices, list)
        
        # Test with unknown language
        voices = engine.get_available_voices("unknown")
        assert isinstance(voices, list)
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_stream_synthesis(self, mock_kokoro_class):
        """Test streaming synthesis."""
        mock_kokoro_instance = Mock()
        
        def mock_stream_tts_sync(text, options):
            yield (24000, np.random.random(1000).astype(np.float32))
            yield (24000, np.random.random(500).astype(np.float32))
        
        mock_kokoro_instance.stream_tts_sync = mock_stream_tts_sync
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        # Test streaming
        chunks = list(engine.stream_synthesis("Hello world", "voice1", "i"))
        
        assert len(chunks) > 0
        for sample_rate, audio_chunk in chunks:
            assert sample_rate == 24000
            assert isinstance(audio_chunk, np.ndarray)
            assert audio_chunk.dtype == np.float32
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_stream_synthesis_failure(self, mock_kokoro_class):
        """Test streaming synthesis failure."""
        mock_kokoro_instance = Mock()
        mock_kokoro_instance.stream_tts_sync.side_effect = Exception("Streaming failed")
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        with pytest.raises(TTSError):
            list(engine.stream_synthesis("Hello", "voice1", "en"))
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_combine_audio_chunks(self, mock_kokoro_class):
        """Test combining audio chunks."""
        mock_kokoro_class.return_value = Mock()
        
        engine = KokoroTTSEngine()
        
        # Create test chunks
        chunk1 = AudioData(
            samples=np.ones(100, dtype=np.float32),
            sample_rate=24000,
            duration=100/24000
        )
        chunk2 = AudioData(
            samples=np.ones(200, dtype=np.float32) * 0.5,
            sample_rate=24000,
            duration=200/24000
        )
        
        # Test combining
        combined = engine._combine_audio_chunks([chunk1, chunk2])
        
        assert len(combined.samples) == 300
        assert combined.sample_rate == 24000
        assert combined.duration == (100 + 200) / 24000
        
        # Test single chunk
        single = engine._combine_audio_chunks([chunk1])
        assert single == chunk1
        
        # Test empty chunks
        with pytest.raises(TTSError, match="No audio chunks"):
            engine._combine_audio_chunks([])
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_model_info(self, mock_kokoro_class):
        """Test model information retrieval."""
        mock_kokoro_instance = Mock()
        mock_kokoro_instance.model.voices = ["voice1", "voice2"]
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        info = engine.get_model_info()
        
        assert info['model_type'] == 'Kokoro ONNX'
        assert info['is_available'] is True
        assert info['model_loaded'] is True
        assert 'supported_languages' in info
        assert 'voice_map' in info
        assert 'available_voices' in info
        assert info['available_voices'] == ["voice1", "voice2"]
    
    @pytest.mark.asyncio
    async def test_kokoro_tts_uninitialized_model(self):
        """Test synthesis with uninitialized model."""
        engine = KokoroTTSEngine()
        engine.tts_model = None  # Simulate failed initialization
        
        with pytest.raises(TTSError, match=r"(not initialized|not available|model.*not.*loaded)"):
            await engine.synthesize("Hello", "voice1", "en")
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    @pytest.mark.asyncio
    async def test_kokoro_tts_different_output_formats(self, mock_kokoro_class):
        """Test handling different output formats from Kokoro."""
        mock_kokoro_instance = Mock()
        
        def mock_stream_tts_sync(text, options):
            # Test different output formats
            yield (24000, np.random.random(100).astype(np.float32))  # Tuple format
            yield np.random.random(200).astype(np.float32)  # Array format only
        
        mock_kokoro_instance.stream_tts_sync = mock_stream_tts_sync
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        result = await engine.synthesize("Hello", "voice1", "en")
        
        # Should handle both formats and combine them
        assert len(result.samples) == 300  # 100 + 200
        assert result.sample_rate == 24000
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    @pytest.mark.parametrize("language,expected_lang", [
        ("i", "it-it"),  # Italian
        ("e", "es-es"),  # Spanish  
        ("f", "fr-fr"),  # French
        ("a", "en-us"),  # American English
        ("unknown", "en-us")  # Fallback
    ])
    @pytest.mark.asyncio
    async def test_kokoro_tts_language_mapping(self, mock_kokoro_class, language, expected_lang):
        """Test language code mapping."""
        mock_kokoro_instance = Mock()
        
        def mock_stream_tts_sync(text, options):
            # Verify the language mapping
            assert options.lang == expected_lang
            yield (24000, np.random.random(100).astype(np.float32))
        
        mock_kokoro_instance.stream_tts_sync = mock_stream_tts_sync
        mock_kokoro_class.return_value = mock_kokoro_instance
        
        engine = KokoroTTSEngine()
        
        await engine.synthesize("Hello", "voice1", language)
    
    @patch('src.audio.engines.tts.kokoro_tts.KokoroONNX')
    def test_kokoro_tts_sample_rate_mismatch_warning(self, mock_kokoro_class):
        """Test handling of sample rate mismatches in chunks."""
        mock_kokoro_class.return_value = Mock()
        
        engine = KokoroTTSEngine()
        
        # Create chunks with different sample rates
        chunk1 = AudioData(
            samples=np.ones(100, dtype=np.float32),
            sample_rate=24000,
            duration=100/24000
        )
        chunk2 = AudioData(
            samples=np.ones(100, dtype=np.float32),
            sample_rate=16000,  # Different sample rate
            duration=100/16000
        )
        
        # Should still combine but log warning
        with patch('src.audio.engines.tts.kokoro_tts.logger') as mock_logger:
            combined = engine._combine_audio_chunks([chunk1, chunk2])
            
            # Should use first chunk's sample rate
            assert combined.sample_rate == 24000
            assert len(combined.samples) == 200
            
            # Should log warning about mismatch
            mock_logger.warning.assert_called_once()
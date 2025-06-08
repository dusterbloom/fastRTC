"""Unit tests for STT engines."""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.core.interfaces import AudioData, TranscriptionResult
from src.core.exceptions import STTError
from src.audio.engines.stt.base import BaseSTTEngine
from src.audio.engines.stt.huggingface_stt import HuggingFaceSTTEngine


class TestBaseSTTEngine:
    """Test cases for BaseSTTEngine."""
    
    def test_base_stt_initialization(self):
        """Test base STT engine initialization."""
        # Create concrete implementation for testing
        class TestSTTEngine(BaseSTTEngine):
            async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
                return TranscriptionResult(text="test transcription")
        
        engine = TestSTTEngine()
        
        assert engine.stats['transcriptions'] == 0
        assert engine.stats['total_audio_duration'] == 0.0
        assert engine.stats['total_processing_time'] == 0.0
        assert engine.stats['errors'] == 0
        assert engine.stats['last_transcription'] is None
        assert engine.is_available() is False  # Not set as available yet
    
    @pytest.mark.asyncio
    async def test_base_stt_transcription_success(self):
        """Test successful transcription with stats tracking."""
        class TestSTTEngine(BaseSTTEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
                await asyncio.sleep(0.001)  # Small delay for timing
                return TranscriptionResult(text="test transcription", confidence=0.95)
        
        engine = TestSTTEngine()
        
        # Create test audio
        samples = np.random.random(1000).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration=1000/16000)
        
        # Transcribe
        result = await engine.transcribe(audio)
        
        # Check result
        assert result.text == "test transcription"
        assert result.confidence == 0.95
        
        # Check stats
        stats = engine.get_stats()
        assert stats['transcriptions'] == 1
        assert stats['total_audio_duration'] == audio.duration
        assert stats['total_processing_time'] > 0
        assert stats['avg_processing_time'] > 0
        assert stats['real_time_factor'] > 0
        assert stats['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_base_stt_transcription_failure(self):
        """Test transcription failure handling."""
        class FailingSTTEngine(BaseSTTEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
                raise ValueError("Transcription failed")
        
        engine = FailingSTTEngine()
        
        samples = np.random.random(100).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration=100/16000)
        
        with pytest.raises(STTError):
            await engine.transcribe(audio)
        
        # Check error stats
        stats = engine.get_stats()
        assert stats['errors'] == 1
        assert stats['transcriptions'] == 0
        assert stats['success_rate'] == 0.0
    
    @pytest.mark.asyncio
    async def test_base_stt_unavailable_engine(self):
        """Test transcription with unavailable engine."""
        class TestSTTEngine(BaseSTTEngine):
            async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
                return TranscriptionResult(text="test")
        
        engine = TestSTTEngine()  # Not set as available
        
        samples = np.random.random(100).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration=100/16000)
        
        with pytest.raises(STTError, match="not available"):
            await engine.transcribe(audio)
    
    def test_base_stt_stats_reset(self):
        """Test statistics reset functionality."""
        class TestSTTEngine(BaseSTTEngine):
            def __init__(self):
                super().__init__()
                self._set_available(True)
            
            async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
                return TranscriptionResult(text="test")
        
        engine = TestSTTEngine()
        
        # Manually update stats
        engine.stats['transcriptions'] = 5
        engine.stats['errors'] = 2
        
        # Reset stats
        engine.reset_stats()
        
        # Verify reset
        assert engine.stats['transcriptions'] == 0
        assert engine.stats['errors'] == 0
        assert engine.stats['total_audio_duration'] == 0.0


class TestHuggingFaceSTTEngine:
    """Test cases for HuggingFaceSTTEngine."""
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.torch')
    def test_huggingface_stt_initialization_success(self, mock_torch, mock_pipeline, mock_processor, mock_model):
        """Test successful HuggingFace STT initialization."""
        # Mock torch and device detection
        mock_torch.cuda.is_available.return_value = False
        
        # Mock model components
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Initialize engine
        engine = HuggingFaceSTTEngine()
        
        # Verify initialization
        assert engine.device == "cpu"
        assert engine.model == mock_model_instance
        assert engine.processor == mock_processor_instance
        assert engine.pipeline == mock_pipeline_instance
        assert engine.is_available() is True
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    def test_huggingface_stt_initialization_failure(self, mock_model):
        """Test HuggingFace STT initialization failure."""
        # Mock model loading failure
        mock_model.from_pretrained.side_effect = Exception("Model loading failed")
        
        with pytest.raises(STTError):
            HuggingFaceSTTEngine()
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.torch')
    @pytest.mark.asyncio
    async def test_huggingface_stt_transcription(self, mock_torch, mock_pipeline, mock_processor, mock_model):
        """Test HuggingFace STT transcription."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False
        
        # Mock model components
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        
        # Mock pipeline with transcription result
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = {
            "text": "Hello world",
            "language": "en"
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Initialize engine
        engine = HuggingFaceSTTEngine()
        
        # Create test audio
        samples = np.random.random(1000).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration=1000/16000)
        
        # Transcribe
        result = await engine.transcribe(audio)
        
        # Verify result
        assert result.text == "Hello world"
        assert result.language == "en"
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.torch')
    @pytest.mark.asyncio
    async def test_huggingface_stt_transcription_with_chunks(self, mock_torch, mock_pipeline, mock_processor, mock_model):
        """Test transcription with chunk information."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False
        
        # Mock model components
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        
        # Mock pipeline with chunks
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = {
            "text": "Hello world",
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5], "language": "en"},
                {"text": "world", "timestamp": [0.5, 1.0], "language": "en"}
            ]
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        engine = HuggingFaceSTTEngine()
        
        samples = np.random.random(1000).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration=1000/16000)
        
        result = await engine.transcribe(audio)
        
        assert result.text == "Hello world"
        assert result.chunks is not None
        assert len(result.chunks) == 2
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.torch')
    @pytest.mark.asyncio
    async def test_huggingface_stt_transcription_failure(self, mock_torch, mock_pipeline, mock_processor, mock_model):
        """Test transcription failure handling."""
        # Mock torch
        mock_torch.cuda.is_available.return_value = False
        
        # Mock model components
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        
        # Mock pipeline failure
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.side_effect = Exception("Pipeline failed")
        mock_pipeline.return_value = mock_pipeline_instance
        
        engine = HuggingFaceSTTEngine()
        
        samples = np.random.random(100).astype(np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, duration=100/16000)
        
        with pytest.raises(STTError):
            await engine.transcribe(audio)
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.get_device')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    def test_huggingface_stt_device_detection(self, mock_pipeline, mock_processor, mock_model, mock_get_device):
        """Test device detection logic."""
        # Test different device scenarios
        test_cases = [
            ("cuda", True),
            ("cpu", False),
            ("mps", False)
        ]
        
        for expected_device, force_cpu in test_cases:
            mock_get_device.return_value = expected_device
            mock_model.from_pretrained.return_value = Mock()
            mock_processor.from_pretrained.return_value = Mock()
            mock_pipeline.return_value = Mock()
            
            engine = HuggingFaceSTTEngine(force_cpu=force_cpu)
            
            assert engine.device == expected_device
    
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.torch')
    def test_huggingface_stt_model_info(self, mock_torch, mock_pipeline, mock_processor, mock_model):
        """Test model information retrieval."""
        mock_torch.cuda.is_available.return_value = True
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        engine = HuggingFaceSTTEngine(model_id="test-model")
        info = engine.get_model_info()
        
        assert info['model_id'] == "test-model"
        assert info['device'] == "cuda"
        assert info['is_available'] is True
        assert info['model_loaded'] is True
        assert info['pipeline_loaded'] is True
    
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoModelForSpeechSeq2Seq')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.AutoProcessor')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.pipeline')
    @patch('fastrtc_voice_assistant.src.audio.engines.stt.huggingface_stt.torch')
    @pytest.mark.asyncio
    async def test_huggingface_stt_dtype_handling(self, mock_torch, mock_pipeline, mock_processor, mock_model, dtype):
        """Test handling of different audio data types."""
        mock_torch.cuda.is_available.return_value = False
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = {"text": "test"}
        mock_pipeline.return_value = mock_pipeline_instance
        
        engine = HuggingFaceSTTEngine()
        
        # Create audio with specific dtype
        if dtype in [np.int16]:
            samples = (np.random.random(100) * 1000).astype(dtype)
        else:
            samples = np.random.random(100).astype(dtype)
        
        audio = AudioData(samples=samples.astype(np.float32), sample_rate=16000, duration=100/16000)
        
        result = await engine.transcribe(audio)
        assert result.text == "test"


class TestSTTEngineUtilities:
    """Test utility functions for STT engines."""
    
    def test_get_device_function(self):
        """Test device detection function."""
        from src.audio.engines.stt.huggingface_stt import get_device
        
        # Test force CPU
        device = get_device(force_cpu=True)
        assert device == "cpu"
        
        # Test automatic detection (mocked)
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device(force_cpu=False)
            assert device == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            device = get_device(force_cpu=False)
            assert device in ["cpu", "mps"]  # Depends on system
    
    def test_get_torch_and_np_dtypes(self):
        """Test dtype selection function."""
        from src.audio.engines.stt.huggingface_stt import get_torch_and_np_dtypes
        import torch
        
        # Test CUDA device
        torch_dtype, np_dtype = get_torch_and_np_dtypes("cuda", use_bfloat16=False)
        assert torch_dtype == torch.float16
        assert np_dtype == np.float32
        
        # Test CPU device
        torch_dtype, np_dtype = get_torch_and_np_dtypes("cpu", use_bfloat16=False)
        assert torch_dtype == torch.float32
        assert np_dtype == np.float32
        
        # Test bfloat16
        torch_dtype, np_dtype = get_torch_and_np_dtypes("cuda", use_bfloat16=True)
        assert torch_dtype == torch.bfloat16
        assert np_dtype == np.float32
    
    def test_is_flash_attn_2_available(self):
        """Test Flash Attention 2 availability check."""
        from src.audio.engines.stt.huggingface_stt import is_flash_attn_2_available
        
        # Test when flash_attn is available
        with patch('builtins.__import__', return_value=Mock()):
            assert is_flash_attn_2_available() is True
        
        # Test when flash_attn is not available
        with patch('builtins.__import__', side_effect=ImportError):
            assert is_flash_attn_2_available() is False
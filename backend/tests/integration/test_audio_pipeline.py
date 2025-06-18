"""
Integration tests for the complete audio processing pipeline.

Tests the end-to-end audio processing flow from input to output.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from src.core.voice_assistant import VoiceAssistant
from src.integration import StreamCallbackHandler
from src.audio import BluetoothAudioProcessor, HuggingFaceSTTEngine, KokoroTTSEngine
from src.audio import HybridLanguageDetector, VoiceMapper


class TestAudioPipelineIntegration:
    """Test the complete audio processing pipeline."""
    
    @pytest.fixture
    def mock_voice_assistant(self):
        """Create a mock voice assistant for testing."""
        assistant = Mock(spec=VoiceAssistant)
        assistant.current_language = 'a'
        assistant.voice_detection_successes = 0
        assistant.turn_count = 0
        assistant.total_response_time = []
        assistant.conversation_buffer = Mock()
        assistant.conversation_buffer.append = Mock()
        assistant.amem_memory = Mock()
        assistant.amem_memory.get_stats = Mock(return_value={
            'mem_ops': 5,
            'user_name_cache': 'test_user'
        })
        assistant.audio_processor = Mock()
        assistant.audio_processor.get_detection_stats = Mock(return_value={
            'avg_rms': 0.123,
            'calibrated': True
        })
        return assistant
    
    @pytest.fixture
    def mock_stt_engine(self):
        """Create a mock STT engine."""
        engine = Mock(spec=HuggingFaceSTTEngine)
        engine.transcribe = Mock(return_value={
            "text": "Hello, how are you?",
            "language": "en"
        })
        return engine
    
    @pytest.fixture
    def mock_tts_engine(self):
        """Create a mock TTS engine."""
        engine = Mock(spec=KokoroTTSEngine)
        # Mock TTS streaming output
        sample_rate = 16000
        audio_chunk = np.random.random(1024).astype(np.float32)
        engine.stream_tts_sync = Mock(return_value=[(sample_rate, audio_chunk)])
        return engine
    
    @pytest.fixture
    def mock_language_detector(self):
        """Create a mock language detector."""
        detector = Mock(spec=HybridLanguageDetector)
        detector.detect_language = Mock(return_value=('a', 0.95))
        return detector
    
    @pytest.fixture
    def mock_voice_mapper(self):
        """Create a mock voice mapper."""
        mapper = Mock(spec=VoiceMapper)
        mapper.get_voices_for_language = Mock(return_value=['voice1', 'voice2'])
        return mapper
    
    @pytest.fixture
    def callback_handler(self, mock_voice_assistant, mock_stt_engine, 
                        mock_tts_engine, mock_language_detector, mock_voice_mapper):
        """Create a callback handler with mocked dependencies."""
        return StreamCallbackHandler(
            voice_assistant=mock_voice_assistant,
            stt_engine=mock_stt_engine,
            tts_engine=mock_tts_engine,
            language_detector=mock_language_detector,
            voice_mapper=mock_voice_mapper,
            event_loop=None
        )
    
    def test_audio_pipeline_basic_flow(self, callback_handler, mock_voice_assistant):
        """Test the basic audio processing flow with real audio data."""
        # Use real audio from audio_samples
        audio_data = create_test_audio(duration=1.0, frequency=440, sample_rate=16000)
        sample_rate = audio_data.sample_rate
        audio_tuple = (sample_rate, audio_data.samples)
        
        # Set up mock to handle real audio (simplified for now)
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data.samples))
        mock_voice_assistant.get_llm_response_smart = AsyncMock(return_value="I'm doing well, thank you!")
        
        # Process the audio stream
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify the pipeline was executed
        mock_voice_assistant.process_audio_array.assert_called_once()
        assert len(results) > 0
        
        # Verify audio output format
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            audio_output, additional_outputs = result
            assert isinstance(audio_output, tuple)
            assert len(audio_output) == 2
            sr, audio_array = audio_output
            assert sr == sample_rate
            assert isinstance(audio_array, np.ndarray)
            assert audio_array.size > 0
    
    def test_audio_pipeline_empty_audio(self, callback_handler, mock_voice_assistant):
        """Test handling of empty audio input."""
        # Mock empty audio data
        sample_rate = 16000
        empty_audio = np.array([], dtype=np.float32)
        audio_tuple = (sample_rate, empty_audio)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, empty_audio))
        
        # Process empty audio
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should return silent frame
        assert len(results) == 1
        audio_output, additional_outputs = results[0]
        sr, audio_array = audio_output
        assert sr == 16000
        assert len(audio_array) == 0
    
    def test_audio_pipeline_language_detection(self, callback_handler, mock_voice_assistant,
                                             mock_language_detector):
        """Test language detection in the pipeline."""
        # Mock audio data
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        # Mock Italian language detection
        mock_language_detector.detect_language = Mock(return_value=('i', 0.92))
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        mock_voice_assistant.get_llm_response_smart = AsyncMock(return_value="Ciao!")
        
        # Process the audio
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify language detection was called
        mock_language_detector.detect_language.assert_called_once()
        
        # Verify language was updated
        assert mock_voice_assistant.current_language == 'i'
    
    def test_audio_pipeline_error_handling(self, callback_handler, mock_voice_assistant):
        """Test error handling in the audio pipeline."""
        # Mock audio data
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        # Mock an error in audio processing
        mock_voice_assistant.process_audio_array = Mock(side_effect=Exception("Audio processing error"))
        
        # Process the audio - should handle error gracefully
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should return error recovery audio
        assert len(results) > 0
    
    def test_audio_pipeline_tts_fallback(self, callback_handler, mock_voice_assistant,
                                       mock_tts_engine, mock_voice_mapper):
        """Test TTS voice fallback mechanism."""
        # Mock audio data
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        # Mock voice mapper to return multiple voices
        mock_voice_mapper.get_voices_for_language = Mock(return_value=['voice1', 'voice2', 'voice3'])
        
        # Mock TTS engine to fail on first voice, succeed on second
        def mock_stream_tts_side_effect(text, options):
            if hasattr(options, 'voice') and options.voice == 'voice1':
                raise Exception("Voice 1 failed")
            else:
                return [(16000, np.random.random(512).astype(np.float32))]
        
        mock_tts_engine.stream_tts_sync = Mock(side_effect=mock_stream_tts_side_effect)
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        mock_voice_assistant.get_llm_response_smart = AsyncMock(return_value="Test response")
        
        # Process the audio
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should have attempted multiple voices
        assert mock_tts_engine.stream_tts_sync.call_count >= 2
        assert len(results) > 0
    
    @patch('src.integration.callback_handler.run_coro_from_sync_thread_with_timeout')
    def test_audio_pipeline_llm_timeout(self, mock_run_coro, callback_handler, 
                                      mock_voice_assistant):
        """Test LLM timeout handling in the pipeline."""
        # Mock audio data
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        # Mock timeout error
        mock_run_coro.side_effect = TimeoutError("LLM timeout")
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        
        # Process the audio
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should handle timeout gracefully and return audio
        assert len(results) > 0
    
    def test_audio_pipeline_statistics_update(self, callback_handler, mock_voice_assistant):
        """Test that statistics are properly updated during processing."""
        # Mock audio data
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        mock_voice_assistant.get_llm_response_smart = AsyncMock(return_value="Response")
        
        # Process the audio
        list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify statistics were updated
        assert mock_voice_assistant.voice_detection_successes == 1
        assert mock_voice_assistant.turn_count == 1
        mock_voice_assistant.conversation_buffer.append.assert_called_once()
    
    def test_audio_pipeline_chunk_streaming(self, callback_handler, mock_voice_assistant,
                                          mock_tts_engine):
        """Test that audio is properly chunked for streaming."""
        # Mock audio data
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        # Mock large TTS output that should be chunked
        large_audio = np.random.random(5000).astype(np.float32)
        mock_tts_engine.stream_tts_sync = Mock(return_value=[(16000, large_audio)])
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        mock_voice_assistant.get_llm_response_smart = AsyncMock(return_value="Long response")
        
        # Process the audio
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should have multiple chunks
        assert len(results) > 1
        
        # Verify chunk sizes are reasonable
        for result in results:
            audio_output, _ = result
            _, chunk_audio = audio_output
            assert len(chunk_audio) <= 1024  # Max chunk size
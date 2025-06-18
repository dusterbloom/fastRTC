"""
Integration tests for FastRTC bridge and stream management.

Tests the FastRTC integration components and stream lifecycle.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.integration import FastRTCBridge, StreamCallbackHandler
from src.core.voice_assistant import VoiceAssistant


class TestFastRTCIntegration:
    """Test FastRTC bridge integration."""
    
    @pytest.fixture
    def fastrtc_bridge(self):
        """Create a FastRTC bridge instance."""
        return FastRTCBridge()
    
    @pytest.fixture
    def mock_callback_function(self):
        """Create a mock callback function."""
        def callback(audio_data):
            yield (16000, []), {}
        return callback
    
    def test_fastrtc_bridge_initialization(self, fastrtc_bridge):
        """Test FastRTC bridge initialization."""
        assert fastrtc_bridge.stream is None
        assert fastrtc_bridge.is_running is False
    
    @patch('src.integration.fastrtc_bridge.Stream')
    @patch('src.integration.fastrtc_bridge.ReplyOnPause')
    def test_create_stream_success(self, mock_reply_on_pause, mock_stream, 
                                 fastrtc_bridge, mock_callback_function):
        """Test successful stream creation."""
        # Mock the Stream and ReplyOnPause classes
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        mock_reply_on_pause_instance = Mock()
        mock_reply_on_pause.return_value = mock_reply_on_pause_instance
        
        # Create stream
        stream = fastrtc_bridge.create_stream(
            callback_function=mock_callback_function,
            speech_threshold=0.2
        )
        
        # Verify stream creation
        assert stream == mock_stream_instance
        assert fastrtc_bridge.stream == mock_stream_instance
        mock_stream.assert_called_once()
        mock_reply_on_pause.assert_called_once()
    
    def test_get_audio_constraints(self, fastrtc_bridge):
        """Test audio constraints configuration."""
        constraints = fastrtc_bridge._get_audio_constraints()
        
        # Verify required constraints
        assert constraints["echoCancellation"] is True
        assert constraints["noiseSuppression"] is True
        assert constraints["autoGainControl"] is True
        assert constraints["sampleRate"]["ideal"] == 16000
        assert constraints["sampleSize"]["ideal"] == 16
        assert constraints["channelCount"]["exact"] == 1
        assert "latency" in constraints
    
    def test_launch_stream_without_creation(self, fastrtc_bridge):
        """Test launching stream without creating it first."""
        with pytest.raises(RuntimeError, match="Stream not created"):
            fastrtc_bridge.launch_stream()
    
    @patch('src.integration.fastrtc_bridge.Stream')
    def test_launch_stream_success(self, mock_stream, fastrtc_bridge, mock_callback_function):
        """Test successful stream launch."""
        # Create mock stream with UI
        mock_stream_instance = Mock()
        mock_ui = Mock()
        mock_stream_instance.ui = mock_ui
        mock_stream.return_value = mock_stream_instance
        
        # Create and launch stream
        fastrtc_bridge.create_stream(mock_callback_function)
        fastrtc_bridge.launch_stream(quiet=True)  # Quiet to avoid print statements
        
        # Verify launch
        assert fastrtc_bridge.is_running is True
        mock_ui.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=7860,
            quiet=True,
            share=False
        )
    
    @patch('src.integration.fastrtc_bridge.Stream')
    def test_launch_stream_with_custom_params(self, mock_stream, fastrtc_bridge, 
                                            mock_callback_function):
        """Test stream launch with custom parameters."""
        # Create mock stream
        mock_stream_instance = Mock()
        mock_ui = Mock()
        mock_stream_instance.ui = mock_ui
        mock_stream.return_value = mock_stream_instance
        
        # Create and launch with custom params
        fastrtc_bridge.create_stream(mock_callback_function)
        fastrtc_bridge.launch_stream(
            server_name="localhost",
            server_port=8080,
            share=True,
            quiet=True
        )
        
        # Verify custom parameters
        mock_ui.launch.assert_called_once_with(
            server_name="localhost",
            server_port=8080,
            quiet=True,
            share=True
        )
    
    @patch('src.integration.fastrtc_bridge.Stream')
    def test_launch_stream_error_handling(self, mock_stream, fastrtc_bridge, 
                                        mock_callback_function):
        """Test error handling during stream launch."""
        # Create mock stream that raises error on launch
        mock_stream_instance = Mock()
        mock_ui = Mock()
        mock_ui.launch.side_effect = Exception("Launch failed")
        mock_stream_instance.ui = mock_ui
        mock_stream.return_value = mock_stream_instance
        
        # Create stream and attempt launch
        fastrtc_bridge.create_stream(mock_callback_function)
        
        with pytest.raises(Exception, match="Launch failed"):
            fastrtc_bridge.launch_stream(quiet=True)
        
        # Verify is_running is set to False on error
        assert fastrtc_bridge.is_running is False
    
    def test_stop_stream(self, fastrtc_bridge):
        """Test stream stopping."""
        # Set up running stream
        fastrtc_bridge.stream = Mock()
        fastrtc_bridge.is_running = True
        
        # Stop stream
        fastrtc_bridge.stop_stream()
        
        # Verify state
        assert fastrtc_bridge.is_running is False
    
    def test_get_stream_status(self, fastrtc_bridge):
        """Test stream status reporting."""
        # Test initial status
        status = fastrtc_bridge.get_stream_status()
        assert status["is_running"] is False
        assert status["stream_created"] is False
        assert status["stream_type"] is None
        
        # Test with created stream
        fastrtc_bridge.stream = Mock()
        fastrtc_bridge.is_running = True
        
        status = fastrtc_bridge.get_stream_status()
        assert status["is_running"] is True
        assert status["stream_created"] is True
        assert status["stream_type"] == "audio"
    
    @patch('src.integration.fastrtc_bridge.Stream')
    def test_create_stream_with_custom_threshold(self, mock_stream, fastrtc_bridge, 
                                               mock_callback_function):
        """Test stream creation with custom speech threshold."""
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        
        # Create stream with custom threshold
        custom_threshold = 0.25
        stream = fastrtc_bridge.create_stream(
            callback_function=mock_callback_function,
            speech_threshold=custom_threshold
        )
        
        # Verify stream was created
        assert stream == mock_stream_instance
        mock_stream.assert_called_once()
    
    @patch('src.integration.fastrtc_bridge.Stream')
    def test_create_stream_error_handling(self, mock_stream, fastrtc_bridge, 
                                        mock_callback_function):
        """Test error handling during stream creation."""
        # Mock Stream to raise an error
        mock_stream.side_effect = Exception("Stream creation failed")
        
        # Attempt to create stream
        with pytest.raises(Exception, match="Stream creation failed"):
            fastrtc_bridge.create_stream(mock_callback_function)
        
        # Verify stream is not set
        assert fastrtc_bridge.stream is None


class TestStreamCallbackIntegration:
    """Test stream callback handler integration."""
    
    @pytest.fixture
    def mock_voice_assistant(self):
        """Create a mock voice assistant."""
        assistant = Mock(spec=VoiceAssistant)
        assistant.current_language = 'a'
        assistant.voice_detection_successes = 0
        assistant.turn_count = 0
        assistant.total_response_time = []
        assistant.conversation_buffer = Mock()
        assistant.amem_memory = Mock()
        assistant.audio_processor = Mock()
        return assistant
    
    @pytest.fixture
    def callback_handler(self, mock_voice_assistant):
        """Create a callback handler with mocked dependencies."""
        return StreamCallbackHandler(
            voice_assistant=mock_voice_assistant,
            stt_engine=Mock(),
            tts_engine=Mock(),
            language_detector=Mock(),
            voice_mapper=Mock(),
            event_loop=None
        )
    
    def test_callback_handler_initialization(self, callback_handler, mock_voice_assistant):
        """Test callback handler initialization."""
        assert callback_handler.voice_assistant == mock_voice_assistant
        assert callback_handler.stt_engine is not None
        assert callback_handler.tts_engine is not None
        assert callback_handler.language_detector is not None
        assert callback_handler.voice_mapper is not None
    
    def test_callback_handler_with_none_assistant(self):
        """Test callback handler with None voice assistant."""
        handler = StreamCallbackHandler(
            voice_assistant=None,
            stt_engine=Mock(),
            tts_engine=Mock(),
            language_detector=Mock(),
            voice_mapper=Mock(),
            event_loop=None
        )
        
        # Process audio with None assistant
        audio_tuple = (16000, [])
        results = list(handler.process_audio_stream(audio_tuple))
        
        # Should return empty output
        assert len(results) == 1
        audio_output, additional_outputs = results[0]
        sr, audio_array = audio_output
        assert sr == 16000
        assert len(audio_array) == 0
    
    def test_audio_to_bytes_conversion(self, callback_handler):
        """Test audio to bytes conversion."""
        import numpy as np
        
        # Test normal audio conversion
        sample_rate = 16000
        audio_array = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_array)
        
        audio_bytes = callback_handler._audio_to_bytes(audio_tuple)
        
        # Verify conversion
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) == audio_array.nbytes
    
    def test_audio_to_bytes_normalization(self, callback_handler):
        """Test audio normalization during conversion."""
        import numpy as np
        
        # Test audio that needs normalization (values > 1.0)
        sample_rate = 16000
        audio_array = np.array([2.0, -3.0, 1.5, -0.5], dtype=np.float32)
        audio_tuple = (sample_rate, audio_array)
        
        audio_bytes = callback_handler._audio_to_bytes(audio_tuple)
        
        # Verify conversion completed without error
        assert isinstance(audio_bytes, bytes)
    
    def test_get_kokoro_language_mapping(self, callback_handler):
        """Test Whisper to Kokoro language mapping."""
        # Test known mappings
        assert callback_handler._get_kokoro_language('en') == 'a'
        assert callback_handler._get_kokoro_language('it') == 'i'
        assert callback_handler._get_kokoro_language('es') == 'e'
        assert callback_handler._get_kokoro_language('fr') == 'f'
        
        # Test unknown language (should default to English)
        assert callback_handler._get_kokoro_language('unknown') == 'a'
    
    def test_yield_audio_chunks(self, callback_handler):
        """Test audio chunking for streaming."""
        import numpy as np
        from fastrtc import AdditionalOutputs
        
        # Create large audio array
        sample_rate = 16000
        large_audio = np.random.random(5000).astype(np.float32)
        additional_outputs = AdditionalOutputs()
        
        # Get chunks
        chunks = list(callback_handler._yield_audio_chunks(
            sample_rate, large_audio, additional_outputs, chunk_size=1024
        ))
        
        # Verify chunking
        assert len(chunks) > 1
        
        # Verify chunk format
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            audio_output, outputs = chunk
            sr, audio_array = audio_output
            assert sr == sample_rate
            assert len(audio_array) <= 1024
    
    @patch('src.integration.callback_handler.logger')
    def test_error_logging(self, mock_logger, callback_handler, mock_voice_assistant):
        """Test error logging in callback handler."""
        import numpy as np
        
        # Mock an error in processing
        mock_voice_assistant.process_audio_array.side_effect = Exception("Test error")
        
        # Process audio
        audio_tuple = (16000, np.random.random(100).astype(np.float32))
        list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify error was logged
        mock_logger.error.assert_called()
"""
Integration tests for complete conversation flow.

Tests the end-to-end conversation processing from audio input to audio output.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.core.voice_assistant import VoiceAssistant
from src.core.main import VoiceAssistantApplication
from src.integration import StreamCallbackHandler, FastRTCBridge
from src.utils.async_utils import AsyncEnvironmentManager


class TestFullConversationFlow:
    """Test complete conversation flow integration."""
    
    @pytest.fixture
    def mock_voice_assistant(self):
        """Create a comprehensive mock voice assistant."""
        assistant = Mock(spec=VoiceAssistant)
        assistant.current_language = 'a'
        assistant.voice_detection_successes = 0
        assistant.turn_count = 0
        assistant.total_response_time = []
        assistant.conversation_buffer = Mock()
        assistant.conversation_buffer.append = Mock()
        assistant.conversation_buffer.get_recent_messages = Mock(return_value=[])
        
        # Mock memory and stats
        assistant.amem_memory = Mock()
        assistant.amem_memory.get_stats = Mock(return_value={
            'mem_ops': 5,
            'user_name_cache': 'test_user'
        })
        assistant.amem_memory.add_to_memory_smart = AsyncMock()
        
        # Mock audio processor
        assistant.audio_processor = Mock()
        assistant.audio_processor.get_detection_stats = Mock(return_value={
            'avg_rms': 0.123,
            'calibrated': True
        })
        
        # Mock LLM response
        assistant.get_llm_response_smart = AsyncMock(return_value="Hello! How can I help you today?")
        
        return assistant
    
    @pytest.fixture
    def conversation_components(self, mock_voice_assistant):
        """Create all components needed for conversation flow."""
        # Mock STT engine
        stt_engine = Mock()
        stt_engine.transcribe = Mock(return_value={
            "text": "Hello, how are you?",
            "language": "en"
        })
        
        # Mock TTS engine with streaming output
        tts_engine = Mock()
        sample_rate = 16000
        audio_chunk = np.random.random(1024).astype(np.float32)
        tts_engine.stream_tts_sync = Mock(return_value=[(sample_rate, audio_chunk)])
        
        # Mock language detector
        language_detector = Mock()
        language_detector.detect_language = Mock(return_value=('a', 0.95))
        
        # Mock voice mapper
        voice_mapper = Mock()
        voice_mapper.get_voices_for_language = Mock(return_value=['voice1', 'voice2'])
        
        # Create callback handler
        callback_handler = StreamCallbackHandler(
            voice_assistant=mock_voice_assistant,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            language_detector=language_detector,
            voice_mapper=voice_mapper,
            event_loop=None
        )
        
        return {
            'callback_handler': callback_handler,
            'stt_engine': stt_engine,
            'tts_engine': tts_engine,
            'language_detector': language_detector,
            'voice_mapper': voice_mapper
        }
    
    def test_complete_conversation_turn(self, mock_voice_assistant, conversation_components):
        """Test a complete conversation turn from audio input to audio output."""
        callback_handler = conversation_components['callback_handler']
        
        # Mock audio input
        sample_rate = 16000
        audio_data = np.random.random(2048).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        # Mock voice assistant audio processing
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        
        # Process the conversation turn
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify the complete flow
        # 1. Audio was processed
        mock_voice_assistant.process_audio_array.assert_called_once_with(audio_tuple)
        
        # 2. STT was performed
        conversation_components['stt_engine'].transcribe.assert_called_once()
        
        # 3. Language detection was performed
        conversation_components['language_detector'].detect_language.assert_called_once()
        
        # 4. LLM response was generated
        mock_voice_assistant.get_llm_response_smart.assert_called_once()
        
        # 5. TTS was performed
        conversation_components['tts_engine'].stream_tts_sync.assert_called_once()
        
        # 6. Audio output was generated
        assert len(results) > 0
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            audio_output, additional_outputs = result
            assert isinstance(audio_output, tuple)
            assert len(audio_output) == 2
        
        # 7. Statistics were updated
        assert mock_voice_assistant.voice_detection_successes == 1
        assert mock_voice_assistant.turn_count == 1
        mock_voice_assistant.conversation_buffer.append.assert_called_once()
    
    def test_multilingual_conversation_flow(self, mock_voice_assistant, conversation_components):
        """Test conversation flow with language switching."""
        callback_handler = conversation_components['callback_handler']
        
        # First turn in English
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        conversation_components['stt_engine'].transcribe.return_value = {
            "text": "Hello, how are you?",
            "language": "en"
        }
        conversation_components['language_detector'].detect_language.return_value = ('a', 0.95)
        
        # Process first turn
        list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify English language
        assert mock_voice_assistant.current_language == 'a'
        
        # Second turn in Italian
        conversation_components['stt_engine'].transcribe.return_value = {
            "text": "Ciao, come stai?",
            "language": "it"
        }
        conversation_components['language_detector'].detect_language.return_value = ('i', 0.92)
        
        # Process second turn
        list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify language switched to Italian
        assert mock_voice_assistant.current_language == 'i'
        
        # Verify voice mapper was called for Italian
        conversation_components['voice_mapper'].get_voices_for_language.assert_called_with('i')
    
    def test_conversation_with_memory_integration(self, mock_voice_assistant, conversation_components):
        """Test conversation flow with memory integration."""
        callback_handler = conversation_components['callback_handler']
        
        # Mock conversation history
        mock_voice_assistant.conversation_buffer.get_recent_messages.return_value = [
            {'user': 'What is my name?', 'assistant': 'I don\'t know your name yet.'},
            {'user': 'My name is Alice', 'assistant': 'Nice to meet you, Alice!'}
        ]
        
        # Process audio input
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        conversation_components['stt_engine'].transcribe.return_value = {
            "text": "What did I tell you my name was?",
            "language": "en"
        }
        
        # Process conversation turn
        list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify memory integration
        mock_voice_assistant.get_llm_response_smart.assert_called_once_with("What did I tell you my name was?")
        
        # Verify conversation was added to buffer
        mock_voice_assistant.conversation_buffer.append.assert_called_once()
        call_args = mock_voice_assistant.conversation_buffer.append.call_args[0][0]
        assert 'user' in call_args
        assert 'assistant' in call_args
        assert 'timestamp' in call_args
        assert call_args['user'] == "What did I tell you my name was?"
    
    def test_conversation_error_recovery(self, mock_voice_assistant, conversation_components):
        """Test conversation flow with error recovery."""
        callback_handler = conversation_components['callback_handler']
        
        # Mock STT error
        conversation_components['stt_engine'].transcribe.side_effect = Exception("STT failed")
        
        # Process audio input
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        
        # Process conversation turn - should handle error gracefully
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should still return some audio output (error recovery)
        assert len(results) > 0
    
    def test_conversation_performance_tracking(self, mock_voice_assistant, conversation_components):
        """Test conversation flow with performance tracking."""
        callback_handler = conversation_components['callback_handler']
        
        # Process multiple conversation turns
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        
        # Process 5 turns to trigger statistics display
        for i in range(5):
            list(callback_handler.process_audio_stream(audio_tuple))
        
        # Verify statistics were collected
        assert mock_voice_assistant.turn_count == 5
        assert mock_voice_assistant.voice_detection_successes == 5
        
        # Verify memory stats were called (for statistics display)
        mock_voice_assistant.amem_memory.get_stats.assert_called()
        mock_voice_assistant.audio_processor.get_detection_stats.assert_called()
    
    def test_conversation_with_tts_fallback(self, mock_voice_assistant, conversation_components):
        """Test conversation flow with TTS voice fallback."""
        callback_handler = conversation_components['callback_handler']
        
        # Mock multiple voices with first one failing
        conversation_components['voice_mapper'].get_voices_for_language.return_value = [
            'voice1', 'voice2', 'voice3'
        ]
        
        # Mock TTS to fail on first voice, succeed on second
        def mock_tts_side_effect(text, options):
            if hasattr(options, 'voice') and options.voice == 'voice1':
                raise Exception("Voice 1 failed")
            else:
                return [(16000, np.random.random(512).astype(np.float32))]
        
        conversation_components['tts_engine'].stream_tts_sync.side_effect = mock_tts_side_effect
        
        # Process conversation turn
        sample_rate = 16000
        audio_data = np.random.random(1024).astype(np.float32)
        audio_tuple = (sample_rate, audio_data)
        
        mock_voice_assistant.process_audio_array = Mock(return_value=(sample_rate, audio_data))
        
        results = list(callback_handler.process_audio_stream(audio_tuple))
        
        # Should have attempted multiple voices and succeeded
        assert conversation_components['tts_engine'].stream_tts_sync.call_count >= 2
        assert len(results) > 0


class TestVoiceAssistantApplicationIntegration:
    """Test the complete voice assistant application integration."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for application testing."""
        with patch('src.core.main.VoiceAssistant') as mock_va, \
             patch('src.core.main.FastRTCBridge') as mock_bridge, \
             patch('src.core.main.StreamCallbackHandler') as mock_handler, \
             patch('src.core.main.AsyncEnvironmentManager') as mock_async:
            
            # Mock VoiceAssistant
            mock_va_instance = Mock()
            mock_va.return_value = mock_va_instance
            
            # Mock FastRTCBridge
            mock_bridge_instance = Mock()
            mock_bridge.return_value = mock_bridge_instance
            mock_bridge_instance.create_stream.return_value = Mock()
            
            # Mock StreamCallbackHandler
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            
            # Mock AsyncEnvironmentManager
            mock_async_instance = Mock()
            mock_async.return_value = mock_async_instance
            mock_async_instance.setup_async_environment.return_value = True
            mock_async_instance.get_event_loop.return_value = Mock()
            
            return {
                'voice_assistant': mock_va_instance,
                'fastrtc_bridge': mock_bridge_instance,
                'callback_handler': mock_handler_instance,
                'async_manager': mock_async_instance
            }
    
    @pytest.mark.asyncio
    async def test_application_initialization(self, mock_dependencies):
        """Test complete application initialization."""
        from src.core.main import VoiceAssistantApplication
        
        app = VoiceAssistantApplication()
        
        # Initialize the application
        success = await app.initialize()
        
        # Verify initialization was successful
        assert success is True
        assert app.voice_assistant is not None
        assert app.fastrtc_bridge is not None
        assert app.callback_handler is not None
        assert app.async_env_manager is not None
    
    @pytest.mark.asyncio
    async def test_application_initialization_failure(self):
        """Test application initialization failure handling."""
        from src.core.main import VoiceAssistantApplication
        
        with patch('src.core.main.VoiceAssistant') as mock_va:
            mock_va.side_effect = Exception("Initialization failed")
            
            app = VoiceAssistantApplication()
            
            # Initialize the application - should fail gracefully
            success = await app.initialize()
            
            assert success is False
    
    def test_application_status_reporting(self, mock_dependencies):
        """Test application status reporting."""
        from src.core.main import VoiceAssistantApplication
        
        app = VoiceAssistantApplication()
        app.voice_assistant = mock_dependencies['voice_assistant']
        app.fastrtc_bridge = mock_dependencies['fastrtc_bridge']
        app.callback_handler = mock_dependencies['callback_handler']
        app.async_env_manager = mock_dependencies['async_manager']
        app.is_running = True
        
        # Mock component status methods
        mock_dependencies['voice_assistant'].get_system_stats.return_value = {
            'session_info': {'turn_count': 5}
        }
        mock_dependencies['fastrtc_bridge'].get_stream_status.return_value = {
            'is_running': True
        }
        mock_dependencies['async_manager'].is_ready.return_value = True
        mock_dependencies['async_manager'].event_loop = Mock()
        mock_dependencies['async_manager'].event_loop.is_running.return_value = True
        
        # Get application status
        status = app.get_status()
        
        # Verify status structure
        assert status['is_running'] is True
        assert 'components' in status
        assert 'voice_assistant' in status
        assert 'fastrtc_bridge' in status
        assert 'async_environment' in status
        
        # Verify component status
        assert status['components']['voice_assistant'] is True
        assert status['components']['fastrtc_bridge'] is True
        assert status['components']['callback_handler'] is True
        assert status['components']['async_env_manager'] is True
    
    def test_application_shutdown(self, mock_dependencies):
        """Test application shutdown process."""
        from src.core.main import VoiceAssistantApplication
        
        app = VoiceAssistantApplication()
        app.fastrtc_bridge = mock_dependencies['fastrtc_bridge']
        app.async_env_manager = mock_dependencies['async_manager']
        app.is_running = True
        
        # Mock shutdown methods
        mock_dependencies['fastrtc_bridge'].stop_stream = Mock()
        mock_dependencies['async_manager'].shutdown = Mock()
        
        # Shutdown the application
        with patch('sys.exit') as mock_exit:
            app.shutdown()
            
            # Verify shutdown process
            mock_dependencies['fastrtc_bridge'].stop_stream.assert_called_once()
            mock_dependencies['async_manager'].shutdown.assert_called_once_with(timeout=15)
            assert app.is_running is False
            mock_exit.assert_called_once_with(0)
    
    @patch('src.core.main.setup_logging')
    @pytest.mark.asyncio
    async def test_create_application_factory(self, mock_setup_logging):
        """Test application factory function."""
        from src.core.main import create_application
        
        app = await create_application()
        
        # Verify logging was set up
        mock_setup_logging.assert_called_once()
        
        # Verify application was created
        assert app is not None
        assert hasattr(app, 'initialize')
        assert hasattr(app, 'run')
        assert hasattr(app, 'shutdown')
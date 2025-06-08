"""
Integration tests for the main VoiceAssistant orchestrator.

Tests the complete voice assistant system integration and dependency injection.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from collections import deque

from src.core import VoiceAssistant
from src.audio import BluetoothAudioProcessor, HuggingFaceSTTEngine, KokoroTTSEngine
from src.audio import HybridLanguageDetector, VoiceMapper
from src.memory import AMemMemoryManager, ResponseCache, ConversationBuffer
from src.services import LLMService, AsyncManager


class TestVoiceAssistantIntegration:
    """Test the main VoiceAssistant orchestrator."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for VoiceAssistant."""
        return {
            'audio_processor': Mock(spec=BluetoothAudioProcessor),
            'stt_engine': Mock(spec=HuggingFaceSTTEngine),
            'tts_engine': Mock(spec=KokoroTTSEngine),
            'language_detector': Mock(spec=HybridLanguageDetector),
            'voice_mapper': Mock(spec=VoiceMapper),
            'memory_manager': Mock(spec=AMemMemoryManager),
            'response_cache': Mock(spec=ResponseCache),
            'conversation_buffer': Mock(spec=ConversationBuffer),
            'llm_service': Mock(spec=LLMService),
            'async_manager': Mock(spec=AsyncManager)
        }
    
    @pytest.fixture
    def voice_assistant(self, mock_dependencies):
        """Create a VoiceAssistant with mocked dependencies."""
        with patch('src.core.voice_assistant.VoiceAssistant._setup_memory_manager') as mock_setup:
            mock_setup.return_value = mock_dependencies['memory_manager']
            
            assistant = VoiceAssistant(
                audio_processor=mock_dependencies['audio_processor'],
                stt_engine=mock_dependencies['stt_engine'],
                tts_engine=mock_dependencies['tts_engine'],
                language_detector=mock_dependencies['language_detector'],
                voice_mapper=mock_dependencies['voice_mapper'],
                response_cache=mock_dependencies['response_cache'],
                conversation_buffer=mock_dependencies['conversation_buffer'],
                llm_service=mock_dependencies['llm_service'],
                async_manager=mock_dependencies['async_manager']
            )
            return assistant
    
    def test_voice_assistant_initialization(self, voice_assistant):
        """Test VoiceAssistant initialization with dependency injection."""
        assert voice_assistant.audio_processor is not None
        assert voice_assistant.stt_engine is not None
        assert voice_assistant.tts_engine is not None
        assert voice_assistant.language_detector is not None
        assert voice_assistant.voice_mapper is not None
        assert voice_assistant.memory_manager is not None
        assert voice_assistant.response_cache is not None
        assert voice_assistant.conversation_buffer is not None
        assert voice_assistant.llm_service is not None
        assert voice_assistant.async_manager is not None
        
        # Test initial state
        assert voice_assistant.current_language == 'a'  # DEFAULT_LANGUAGE
        assert voice_assistant.turn_count == 0
        assert voice_assistant.voice_detection_successes == 0
        assert isinstance(voice_assistant.total_response_time, deque)
        assert voice_assistant.http_session is None
    
    def test_voice_assistant_default_initialization(self):
        """Test VoiceAssistant initialization with default components."""
        with patch('src.core.voice_assistant.VoiceAssistant._setup_memory_manager') as mock_setup:
            mock_setup.return_value = Mock(spec=AMemMemoryManager)
            
            # Create assistant without providing dependencies
            assistant = VoiceAssistant()
            
            # Verify default components are created
            assert isinstance(assistant.audio_processor, BluetoothAudioProcessor)
            assert isinstance(assistant.stt_engine, HuggingFaceSTTEngine)
            assert isinstance(assistant.tts_engine, KokoroTTSEngine)
            assert isinstance(assistant.language_detector, HybridLanguageDetector)
            assert isinstance(assistant.voice_mapper, VoiceMapper)
            assert isinstance(assistant.response_cache, ResponseCache)
            assert isinstance(assistant.conversation_buffer, ConversationBuffer)
            assert isinstance(assistant.llm_service, LLMService)
            assert isinstance(assistant.async_manager, AsyncManager)
    
    @patch('src.core.voice_assistant.QdrantClient')
    def test_setup_memory_manager(self, mock_qdrant_client):
        """Test memory manager setup with Qdrant."""
        # Mock Qdrant client and collections
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name='existing_collection')]
        mock_client.get_collections.return_value = mock_collections_response
        
        with patch('src.core.voice_assistant.AMemMemoryManager') as mock_amem:
            mock_memory_manager = Mock()
            mock_amem.return_value = mock_memory_manager
            
            assistant = VoiceAssistant()
            
            # Verify Qdrant setup
            mock_qdrant_client.assert_called_once_with(host="localhost", port=6333)
            mock_client.get_collections.assert_called_once()
            mock_client.create_collection.assert_called_once()
            mock_amem.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_async(self, voice_assistant):
        """Test async initialization."""
        # Mock async manager initialization
        voice_assistant.async_manager.initialize = AsyncMock()
        voice_assistant.memory_manager.start_background_processor = AsyncMock()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value = mock_session_instance
            
            await voice_assistant.initialize_async()
            
            # Verify async components were initialized
            assert voice_assistant.http_session == mock_session_instance
            voice_assistant.memory_manager.start_background_processor.assert_called_once()
            voice_assistant.async_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_async(self, voice_assistant):
        """Test async cleanup."""
        # Set up mock session and components
        mock_session = AsyncMock()
        voice_assistant.http_session = mock_session
        voice_assistant.memory_manager.shutdown = AsyncMock()
        voice_assistant.async_manager.cleanup = AsyncMock()
        
        await voice_assistant.cleanup_async()
        
        # Verify cleanup was called
        mock_session.close.assert_called_once()
        voice_assistant.memory_manager.shutdown.assert_called_once()
        voice_assistant.async_manager.cleanup.assert_called_once()
    
    def test_process_audio_array(self, voice_assistant):
        """Test audio array processing."""
        mock_audio_data = (16000, [1, 2, 3, 4])
        expected_result = (16000, [1, 2, 3, 4])
        
        voice_assistant.audio_processor.preprocess_bluetooth_audio.return_value = expected_result
        
        result = voice_assistant.process_audio_array(mock_audio_data)
        
        assert result == expected_result
        voice_assistant.audio_processor.preprocess_bluetooth_audio.assert_called_once_with(mock_audio_data)
    
    @pytest.mark.asyncio
    async def test_process_audio_turn_with_cache(self, voice_assistant):
        """Test audio turn processing with cached response."""
        user_text = "Hello, how are you?"
        cached_response = "I'm doing well, thank you!"
        
        # Mock cached response
        voice_assistant.response_cache.get.return_value = cached_response
        
        result = await voice_assistant.process_audio_turn(user_text)
        
        assert result == cached_response
        voice_assistant.response_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_audio_turn_without_cache(self, voice_assistant):
        """Test audio turn processing without cached response."""
        user_text = "What's the weather like?"
        llm_response = "I don't have access to current weather data."
        
        # Mock no cached response
        voice_assistant.response_cache.get.return_value = None
        voice_assistant.llm_service.generate_response = AsyncMock(return_value=llm_response)
        voice_assistant.memory_manager.add_to_memory_smart = AsyncMock()
        voice_assistant.response_cache.set = Mock()
        voice_assistant.conversation_buffer.get_recent_messages = Mock(return_value=[])
        
        result = await voice_assistant.process_audio_turn(user_text)
        
        assert result == llm_response
        voice_assistant.llm_service.generate_response.assert_called_once()
        voice_assistant.memory_manager.add_to_memory_smart.assert_called_once_with(user_text, llm_response)
        voice_assistant.response_cache.set.assert_called_once_with(user_text, llm_response)
    
    def test_detect_language_from_text(self, voice_assistant):
        """Test language detection from text."""
        test_text = "Ciao, come stai?"
        expected_language = 'i'
        expected_confidence = 0.95
        
        voice_assistant.language_detector.detect_language.return_value = (expected_language, expected_confidence)
        
        result = voice_assistant.detect_language_from_text(test_text)
        
        assert result == expected_language
        voice_assistant.language_detector.detect_language.assert_called_once_with(test_text)
    
    def test_get_voices_for_language(self, voice_assistant):
        """Test getting voices for a specific language."""
        language_code = 'i'
        expected_voices = ['italian_voice_1', 'italian_voice_2']
        
        voice_assistant.voice_mapper.get_voices_for_language.return_value = expected_voices
        
        result = voice_assistant.get_voices_for_language(language_code)
        
        assert result == expected_voices
        voice_assistant.voice_mapper.get_voices_for_language.assert_called_once_with(language_code)
    
    def test_cache_operations(self, voice_assistant):
        """Test response caching operations."""
        text = "Test input"
        response = "Test response"
        
        # Test caching
        voice_assistant.cache_response(text, response)
        voice_assistant.response_cache.set.assert_called_once_with(text, response)
        
        # Test retrieval
        voice_assistant.response_cache.get.return_value = response
        result = voice_assistant.get_cached_response(text)
        
        assert result == response
        voice_assistant.response_cache.get.assert_called_with(text, ttl_seconds=180)
    
    def test_get_system_stats(self, voice_assistant):
        """Test system statistics collection."""
        # Mock component stats
        voice_assistant.memory_manager.get_stats.return_value = {
            'mem_ops': 10,
            'user_name_cache': 'test_user'
        }
        voice_assistant.audio_processor.get_detection_stats.return_value = {
            'avg_rms': 0.123,
            'calibrated': True
        }
        voice_assistant.get_voices_for_language.return_value = ['voice1', 'voice2']
        voice_assistant.response_cache._cache = {'key1': 'value1', 'key2': 'value2'}
        
        # Add some response times
        voice_assistant.total_response_time.extend([1.0, 2.0, 1.5])
        voice_assistant.turn_count = 5
        voice_assistant.voice_detection_successes = 3
        
        stats = voice_assistant.get_system_stats()
        
        # Verify stats structure
        assert 'session_info' in stats
        assert 'performance' in stats
        assert 'language' in stats
        assert 'memory' in stats
        assert 'audio' in stats
        
        # Verify session info
        assert stats['session_info']['turn_count'] == 5
        assert stats['session_info']['voice_detections'] == 3
        
        # Verify performance stats
        assert stats['performance']['avg_response_time'] == 1.5
        assert stats['performance']['cache_size'] == 2
    
    def test_get_system_stats_error_handling(self, voice_assistant):
        """Test system statistics error handling."""
        # Mock an error in stats collection
        voice_assistant.memory_manager.get_stats.side_effect = Exception("Stats error")
        
        stats = voice_assistant.get_system_stats()
        
        # Should return error info
        assert 'error' in stats
        assert stats['error'] == "Stats error"
    
    def test_reset_session(self, voice_assistant):
        """Test session reset functionality."""
        # Set up some state
        voice_assistant.turn_count = 10
        voice_assistant.voice_detection_successes = 5
        voice_assistant.total_response_time.extend([1.0, 2.0, 1.5])
        voice_assistant.current_language = 'i'
        voice_assistant.conversation_buffer.clear = Mock()
        
        old_session_id = voice_assistant.session_id
        
        voice_assistant.reset_session()
        
        # Verify reset
        assert voice_assistant.turn_count == 0
        assert voice_assistant.voice_detection_successes == 0
        assert len(voice_assistant.total_response_time) == 0
        assert voice_assistant.current_language == 'a'  # DEFAULT_LANGUAGE
        assert voice_assistant.session_id != old_session_id
        voice_assistant.conversation_buffer.clear.assert_called_once()
    
    def test_voice_assistant_repr(self, voice_assistant):
        """Test string representation of VoiceAssistant."""
        voice_assistant.turn_count = 5
        
        repr_str = repr(voice_assistant)
        
        assert 'VoiceAssistant' in repr_str
        assert voice_assistant.user_id in repr_str
        assert voice_assistant.session_id in repr_str
        assert voice_assistant.current_language in repr_str
        assert str(voice_assistant.turn_count) in repr_str
    
    @pytest.mark.asyncio
    async def test_get_llm_response_smart(self, voice_assistant):
        """Test smart LLM response generation."""
        user_text = "Tell me a joke"
        expected_response = "Why did the chicken cross the road?"
        
        # Mock the process_audio_turn method
        voice_assistant.process_audio_turn = AsyncMock(return_value=expected_response)
        
        result = await voice_assistant.get_llm_response_smart(user_text)
        
        assert result == expected_response
        voice_assistant.process_audio_turn.assert_called_once_with(user_text)


class TestVoiceAssistantDependencyInjection:
    """Test dependency injection patterns in VoiceAssistant."""
    
    def test_partial_dependency_injection(self):
        """Test VoiceAssistant with partial dependency injection."""
        # Provide only some dependencies
        custom_audio_processor = Mock(spec=BluetoothAudioProcessor)
        custom_language_detector = Mock(spec=HybridLanguageDetector)
        
        with patch('src.core.voice_assistant.VoiceAssistant._setup_memory_manager') as mock_setup:
            mock_setup.return_value = Mock(spec=AMemMemoryManager)
            
            assistant = VoiceAssistant(
                audio_processor=custom_audio_processor,
                language_detector=custom_language_detector
            )
            
            # Verify custom components are used
            assert assistant.audio_processor == custom_audio_processor
            assert assistant.language_detector == custom_language_detector
            
            # Verify default components are created for others
            assert isinstance(assistant.stt_engine, HuggingFaceSTTEngine)
            assert isinstance(assistant.tts_engine, KokoroTTSEngine)
            assert isinstance(assistant.voice_mapper, VoiceMapper)
    
    def test_full_dependency_injection(self):
        """Test VoiceAssistant with full dependency injection."""
        # Create all custom dependencies
        custom_deps = {
            'audio_processor': Mock(spec=BluetoothAudioProcessor),
            'stt_engine': Mock(spec=HuggingFaceSTTEngine),
            'tts_engine': Mock(spec=KokoroTTSEngine),
            'language_detector': Mock(spec=HybridLanguageDetector),
            'voice_mapper': Mock(spec=VoiceMapper),
            'memory_manager': Mock(spec=AMemMemoryManager),
            'response_cache': Mock(spec=ResponseCache),
            'conversation_buffer': Mock(spec=ConversationBuffer),
            'llm_service': Mock(spec=LLMService),
            'async_manager': Mock(spec=AsyncManager)
        }
        
        with patch('src.core.voice_assistant.VoiceAssistant._setup_memory_manager') as mock_setup:
            mock_setup.return_value = custom_deps['memory_manager']
            
            assistant = VoiceAssistant(**custom_deps)
            
            # Verify all custom components are used
            for attr_name, custom_component in custom_deps.items():
                if attr_name != 'memory_manager':  # memory_manager is handled specially
                    assert getattr(assistant, attr_name) == custom_component
            
            assert assistant.memory_manager == custom_deps['memory_manager']
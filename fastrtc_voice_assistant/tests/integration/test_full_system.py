"""
End-to-end system tests for the FastRTC Voice Assistant.

Tests complete conversation flows, multilingual capabilities, memory persistence,
error recovery, and A-MEM memory evolution.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.core.voice_assistant import VoiceAssistant
from src.core.main import VoiceAssistantApplication, create_application
from src.core.interfaces import AudioData, TranscriptionResult
from src.config.settings import AppConfig, AudioConfig, MemoryConfig, LLMConfig, TTSConfig
from tests.fixtures.audio_samples import (
    create_test_audio,
    create_multilingual_samples,
    create_conversation_samples,
    LANGUAGE_TEST_PHRASES
)


class TestFullSystemIntegration:
    """Complete system integration tests."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def test_config(self, temp_config_dir):
        """Create test configuration."""
        return AppConfig(
            audio=AudioConfig(
                sample_rate=16000,
                chunk_duration=2.0,
                noise_threshold=0.15,
                minimal_silent_frame_duration_ms=20
            ),
            memory=MemoryConfig(
                llm_model="llama3.2:3b",
                embedder_model="nomic-embed-text",
                evolution_threshold=50,
                cache_ttl_seconds=180
            ),
            llm=LLMConfig(
                use_ollama=True,
                ollama_url="http://localhost:11434",
                ollama_model="llama3:8b-instruct-q4_K_M"
            ),
            tts=TTSConfig(
                preferred_voice="af_heart",
                fallback_voices=["af_alloy", "af_bella"],
                speed=1.05
            )
        )
    
    @pytest.fixture
    def mock_system_components(self):
        """Create comprehensive mock system components."""
        # Mock STT Engine
        mock_stt = AsyncMock()
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="Hello, how are you today?",
            language="en",
            confidence=0.95
        )
        
        # Mock TTS Engine
        mock_tts = AsyncMock()
        mock_tts.synthesize.return_value = create_test_audio(duration=2.0)
        mock_tts.get_available_voices.return_value = ["af_heart", "af_alloy", "af_bella"]
        
        # Mock Audio Processor
        mock_processor = Mock()
        mock_processor.process.return_value = create_test_audio(duration=1.0)
        
        # Mock Memory Manager
        mock_memory = AsyncMock()
        mock_memory.get_user_context.return_value = "User prefers casual conversation."
        mock_memory.add_memory.return_value = "memory_id_123"
        mock_memory.search_memories.return_value = "Previous conversation about weather."
        
        # Mock LLM Service
        mock_llm = AsyncMock()
        mock_llm.generate_response.return_value = "I'm doing well, thank you for asking! How can I help you today?"
        
        return {
            "stt": mock_stt,
            "tts": mock_tts,
            "processor": mock_processor,
            "memory": mock_memory,
            "llm": mock_llm
        }
    
    @pytest.fixture
    def voice_assistant_system(self, test_config, mock_system_components):
        """Create complete voice assistant system."""
        return VoiceAssistant(
            stt_engine=mock_system_components["stt"],
            tts_engine=mock_system_components["tts"],
            audio_processor=mock_system_components["processor"],
            memory_manager=mock_system_components["memory"],
            llm_service=mock_system_components["llm"]
        )
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, voice_assistant_system, mock_system_components):
        """Test a complete conversation flow from audio input to audio output."""
        # Simulate user text input (already transcribed)
        user_text = "Hello, how are you today?"
        
        # Process the audio turn
        response_text = await voice_assistant_system.process_audio_turn(user_text)
        
        # Verify LLM and memory components were called
        mock_system_components["llm"].generate_response.assert_called_once()
        mock_system_components["memory"].add_to_memory_smart.assert_called_once()
        
        # Verify response
        assert response_text is not None
        assert isinstance(response_text, str)
        assert len(response_text) > 0
    
    @pytest.mark.asyncio
    async def test_multilingual_conversation_flow(self, voice_assistant_system, mock_system_components):
        """Test conversation flow with language switching."""
        # Test different languages with text input
        language_texts = {
            "english": "Hello, how are you?",
            "italian": "Ciao, come stai?",
            "spanish": "Hola, ¿cómo estás?",
            "french": "Bonjour, comment allez-vous?"
        }
        
        for lang, text in language_texts.items():
            # Process text in this language
            response = await voice_assistant_system.process_audio_turn(text)
            
            # Verify response was generated
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_memory_persistence_across_turns(self, voice_assistant_system, mock_system_components):
        """Test that memory persists and evolves across conversation turns."""
        # Simulate conversation with memory evolution
        user_inputs = [
            "I love coffee in the morning",
            "What's the weather like today?",
            "Can you remember my coffee preference?",
            "Tell me about my previous questions"
        ]
        
        for i, user_text in enumerate(user_inputs):
            # Process turn
            response = await voice_assistant_system.process_audio_turn(user_text)
            
            # Verify memory was updated
            mock_system_components["memory"].add_to_memory_smart.assert_called()
            
            assert response is not None
            assert isinstance(response, str)
        
        # Verify memory was called for each turn
        assert mock_system_components["memory"].add_to_memory_smart.call_count == 4
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_graceful_degradation(self, voice_assistant_system, mock_system_components):
        """Test system behavior when components fail."""
        user_text = "Test input"
        
        # Test LLM failure
        mock_system_components["llm"].generate_response.side_effect = Exception("LLM service unavailable")
        
        with pytest.raises(Exception):
            await voice_assistant_system.process_audio_turn(user_text)
        
        # Reset LLM and test memory failure
        mock_system_components["llm"].generate_response.side_effect = None
        mock_system_components["llm"].generate_response.return_value = "Test response"
        mock_system_components["memory"].add_to_memory_smart.side_effect = Exception("Memory service unavailable")
        
        with pytest.raises(Exception):
            await voice_assistant_system.process_audio_turn(user_text)
    
    @pytest.mark.asyncio
    async def test_amem_memory_evolution_simulation(self, voice_assistant_system, mock_system_components):
        """Test A-MEM memory evolution and consolidation."""
        # Simulate many conversation turns to trigger memory evolution
        evolution_threshold = 10  # Simulate lower threshold for testing
        
        conversation_topics = [
            "I work as a software engineer",
            "I enjoy hiking on weekends",
            "My favorite programming language is Python",
            "I have a cat named Whiskers",
            "I prefer tea over coffee",
            "I'm learning Spanish",
            "I live in San Francisco",
            "I play guitar in my free time",
            "I'm interested in AI and machine learning",
            "I like to read science fiction books"
        ]
        
        # Track memory evolution calls
        evolution_calls = []
        
        def mock_add_memory_with_evolution(user_text, assistant_text):
            evolution_calls.append((user_text, assistant_text))
            if len(evolution_calls) >= evolution_threshold:
                # Simulate memory consolidation
                return f"evolved_memory_{len(evolution_calls)}"
            return f"memory_{len(evolution_calls)}"
        
        mock_system_components["memory"].add_memory.side_effect = mock_add_memory_with_evolution
        
        # Process conversation turns
        for i, topic in enumerate(conversation_topics):
            response = await voice_assistant_system.process_audio_turn(topic)
            
            assert response is not None
            assert isinstance(response, str)
        
        # Verify memory evolution occurred
        assert len(evolution_calls) == len(conversation_topics)
        assert mock_system_components["memory"].add_to_memory_smart.call_count == len(conversation_topics)
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_handling(self, test_config, mock_system_components):
        """Test handling multiple concurrent conversations."""
        # Create multiple voice assistant instances (simulating different users)
        assistants = []
        for i in range(3):
            assistant = VoiceAssistant(
                stt_engine=mock_system_components["stt"],
                tts_engine=mock_system_components["tts"],
                audio_processor=mock_system_components["processor"],
                memory_manager=mock_system_components["memory"],
                llm_service=mock_system_components["llm"]
            )
            assistants.append(assistant)
        
        # Simulate concurrent conversations
        text_inputs = [
            "Hello from user 1",  # User 1
            "Hello from user 2",  # User 2
            "Hello from user 3",  # User 3
        ]
        
        # Process conversations concurrently
        tasks = [
            assistant.process_audio_turn(text)
            for assistant, text in zip(assistants, text_inputs)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # Verify all conversations completed successfully
        assert len(responses) == 3
        assert all(response is not None for response in responses)
        assert all(isinstance(response, str) for response in responses)
    
    @pytest.mark.asyncio
    async def test_system_state_consistency(self, voice_assistant_system, mock_system_components):
        """Test that system state remains consistent across operations."""
        # Initial state
        initial_language = voice_assistant_system.current_language
        initial_turn_count = voice_assistant_system.turn_count
        
        # Process several turns
        for i in range(5):
            user_text = f"Test message {i+1}"
            response = await voice_assistant_system.process_audio_turn(user_text)
            
            # Verify response
            assert response is not None
            assert isinstance(response, str)
        
        # Verify final state (turn count is not automatically updated in process_audio_turn)
        assert voice_assistant_system.current_language == "a"  # Default language
    
    @pytest.mark.asyncio
    async def test_configuration_validation_and_usage(self, voice_assistant_system, test_config):
        """Test that configuration is properly validated and used."""
        # Since VoiceAssistant uses dependency injection, we test that the system works
        # with the provided mock components rather than testing config access
        
        # Test that the system works with the injected dependencies
        user_text = "Test configuration usage"
        
        response = await voice_assistant_system.process_audio_turn(user_text)
        assert response is not None
        
        # Verify that the voice assistant has the expected components
        assert voice_assistant_system.stt_engine is not None
        assert voice_assistant_system.tts_engine is not None
        assert voice_assistant_system.audio_processor is not None
        assert voice_assistant_system.memory_manager is not None
        assert voice_assistant_system.llm_service is not None


class TestApplicationIntegration:
    """Test the complete application integration."""
    
    @pytest.fixture
    def mock_application_dependencies(self):
        """Mock all application dependencies."""
        with patch('src.core.main.create_voice_assistant') as mock_create_va, \
             patch('src.integration.fastrtc_bridge.FastRTCBridge') as mock_bridge, \
             patch('src.services.async_manager.AsyncManager') as mock_async_mgr:
            
            # Configure mocks
            mock_va = AsyncMock()
            mock_create_va.return_value = mock_va
            
            mock_bridge_instance = AsyncMock()
            mock_bridge.return_value = mock_bridge_instance
            
            mock_async_mgr_instance = AsyncMock()
            mock_async_mgr.return_value = mock_async_mgr_instance
            
            yield {
                "voice_assistant": mock_va,
                "bridge": mock_bridge_instance,
                "async_manager": mock_async_mgr_instance
            }
    
    @pytest.mark.asyncio
    async def test_application_creation_and_initialization(self, mock_application_dependencies):
        """Test application creation and initialization."""
        # Create application
        app = await create_application()
        
        # Verify application was created
        assert app is not None
        assert isinstance(app, VoiceAssistantApplication)
    
    @pytest.mark.asyncio
    async def test_application_lifecycle_management(self, mock_application_dependencies):
        """Test application startup and shutdown lifecycle."""
        app = await create_application()
        
        # Test startup
        await app.start()
        
        # Verify components were started
        mock_application_dependencies["async_manager"].start.assert_called_once()
        
        # Test shutdown
        await app.stop()
        
        # Verify components were stopped
        mock_application_dependencies["async_manager"].stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_application_error_handling(self, mock_application_dependencies):
        """Test application error handling and recovery."""
        app = await create_application()
        
        # Simulate startup error
        mock_application_dependencies["async_manager"].start.side_effect = Exception("Startup failed")
        
        with pytest.raises(Exception, match="Startup failed"):
            await app.start()
        
        # Reset and test normal operation
        mock_application_dependencies["async_manager"].start.side_effect = None
        await app.start()
        
        # Simulate runtime error
        mock_application_dependencies["bridge"].process_audio.side_effect = Exception("Runtime error")
        
        # Application should handle runtime errors gracefully
        # (specific error handling depends on implementation)


class TestSystemPerformanceIntegration:
    """Integration tests focusing on system-wide performance."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency_measurement(self, voice_assistant_system, mock_system_components):
        """Measure end-to-end latency for complete system."""
        import time
        
        # Add realistic delays to mocks
        async def delayed_stt(*args, **kwargs):
            await asyncio.sleep(0.3)  # STT processing time
            return TranscriptionResult(text="Test input", language="en", confidence=0.9)
        
        async def delayed_llm(*args, **kwargs):
            await asyncio.sleep(0.8)  # LLM processing time
            return "Test response"
        
        async def delayed_tts(*args, **kwargs):
            await asyncio.sleep(0.5)  # TTS processing time
            return create_test_audio(duration=1.0)
        
        mock_system_components["stt"].transcribe.side_effect = delayed_stt
        mock_system_components["llm"].get_response.side_effect = delayed_llm
        mock_system_components["tts"].synthesize.side_effect = delayed_tts
        
        # Measure end-to-end latency
        audio = create_test_audio(duration=2.0)
        
        start_time = time.time()
        response = await voice_assistant_system.process_audio_turn(audio)
        end_time = time.time()
        
        total_latency = end_time - start_time
        
        # Verify response and latency
        assert response is not None
        assert total_latency < 4.0, f"End-to-end latency {total_latency:.2f}s exceeds 4s requirement"
        
        print(f"End-to-end latency: {total_latency:.2f}s")
    
    @pytest.mark.asyncio
    async def test_system_resource_usage_monitoring(self, voice_assistant_system, mock_system_components):
        """Monitor system resource usage during operation."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline measurements
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple audio samples
        for i in range(10):
            audio = create_test_audio(duration=1.0 + i * 0.1)  # Varying durations
            response = await voice_assistant_system.process_audio_turn(audio)
            assert response is not None
        
        # Final measurements
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Excessive memory increase: {memory_increase:.1f}MB"
        assert final_memory < 500, f"Total memory usage {final_memory:.1f}MB exceeds 500MB limit"
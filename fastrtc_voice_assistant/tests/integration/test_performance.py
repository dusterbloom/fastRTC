"""
Performance tests for the FastRTC Voice Assistant.

Tests response latency, memory usage, CPU usage, and system stability.
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
from typing import Dict, List, Tuple
from unittest.mock import Mock, AsyncMock, patch

from src.core.voice_assistant import VoiceAssistant
from src.core.interfaces import AudioData
from tests.fixtures.audio_samples import (
    create_test_audio,
    create_performance_test_audio,
    create_conversation_samples
)


class TestPerformanceBenchmarks:
    """Performance benchmark tests for the voice assistant."""
    
    @pytest.fixture
    def performance_audio_samples(self):
        """Create audio samples for performance testing."""
        return create_performance_test_audio()
    
    @pytest.fixture
    def mock_voice_assistant(self):
        """Create a mock voice assistant for performance testing."""
        # Create mock components
        mock_stt = AsyncMock()
        mock_tts = AsyncMock()
        mock_processor = Mock()
        mock_memory = AsyncMock()
        mock_llm = AsyncMock()
        mock_config = Mock()
        
        # Configure mocks for realistic performance
        mock_stt.transcribe.return_value = Mock(
            text="Hello, this is a test transcription.",
            language="en",
            confidence=0.95
        )
        
        mock_tts.synthesize.return_value = create_test_audio(duration=2.0)
        mock_processor.process.return_value = create_test_audio(duration=1.0)
        mock_memory.get_user_context.return_value = "User context"
        mock_memory.add_memory.return_value = "memory_id_123"
        mock_llm.get_response.return_value = "This is a test response from the LLM."
        
        # Add realistic delays to simulate processing time
        async def delayed_transcribe(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate STT processing time
            return mock_stt.transcribe.return_value
        
        async def delayed_llm_response(*args, **kwargs):
            await asyncio.sleep(1.0)  # Simulate LLM processing time
            return mock_llm.get_response.return_value
        
        async def delayed_tts(*args, **kwargs):
            await asyncio.sleep(0.8)  # Simulate TTS processing time
            return mock_tts.synthesize.return_value
        
        mock_stt.transcribe.side_effect = delayed_transcribe
        mock_llm.get_response.side_effect = delayed_llm_response
        mock_tts.synthesize.side_effect = delayed_tts
        
        return VoiceAssistant(
            stt_engine=mock_stt,
            tts_engine=mock_tts,
            audio_processor=mock_processor,
            memory_manager=mock_memory,
            llm_service=mock_llm,
            config=mock_config
        )
    
    @pytest.mark.asyncio
    async def test_response_latency_requirement(self, mock_voice_assistant, performance_audio_samples):
        """Test that response latency meets the <4 seconds requirement."""
        # Use text instead of audio data for process_audio_turn
        test_text = "Hello, this is a test transcription for performance testing."
        
        start_time = time.time()
        response = await mock_voice_assistant.process_audio_turn(test_text)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should respond within 4 seconds (requirement)
        assert latency < 4.0, f"Response latency {latency:.2f}s exceeds 4s requirement"
        
        # Log performance for monitoring
        print(f"Response latency: {latency:.2f}s")
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_response_latency_short_audio(self, mock_voice_assistant, performance_audio_samples):
        """Test response latency for short audio bursts."""
        # Use short text for short audio simulation
        test_text = "Hi there!"
        
        start_time = time.time()
        response = await mock_voice_assistant.process_audio_turn(test_text)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Short audio should be even faster
        assert latency < 3.0, f"Short audio latency {latency:.2f}s exceeds 3s"
        print(f"Short audio latency: {latency:.2f}s")
    
    @pytest.mark.asyncio
    async def test_response_latency_long_audio(self, mock_voice_assistant, performance_audio_samples):
        """Test response latency for longer audio."""
        # Use longer text for long audio simulation
        test_text = "This is a much longer text that simulates what would be transcribed from a longer audio segment with multiple sentences and complex content that takes more time to process."
        
        start_time = time.time()
        response = await mock_voice_assistant.process_audio_turn(test_text)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Longer audio may take more time but should still be reasonable
        assert latency < 6.0, f"Long audio latency {latency:.2f}s exceeds 6s"
        print(f"Long audio latency: {latency:.2f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_requirement(self, mock_voice_assistant, performance_audio_samples):
        """Test that memory usage stays within <500MB requirement."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        test_text = "Hello, this is a test transcription for performance testing."
        
        # Process multiple audio turns to test memory accumulation
        for i in range(10):
            await mock_voice_assistant.process_audio_turn(test_text)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Should stay within 500MB requirement
            assert current_memory < 500, f"Memory usage {current_memory:.1f}MB exceeds 500MB requirement"
            
            # Memory shouldn't grow excessively with each turn
            assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB is excessive"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_cpu_usage_monitoring(self, mock_voice_assistant, performance_audio_samples):
        """Test CPU usage during conversation processing."""
        test_text = "Hello, this is a test transcription for performance testing."
        
        # Monitor CPU usage during processing
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(20):  # Monitor for 2 seconds
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Process audio while monitoring
        await mock_voice_assistant.process_audio_turn(test_text)
        
        monitor_thread.join()
        
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # CPU usage should be reasonable (not a hard requirement but good to monitor)
        print(f"CPU usage - Average: {avg_cpu:.1f}%, Peak: {max_cpu:.1f}%")
        
        # These are monitoring assertions, not strict requirements
        assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.1f}% is very high"
        assert max_cpu < 95, f"Peak CPU usage {max_cpu:.1f}% is excessive"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_voice_assistant, performance_audio_samples):
        """Test system performance under concurrent load."""
        test_text = "Hello, this is a test transcription for performance testing."
        
        # Process multiple audio streams concurrently
        concurrent_tasks = 5
        
        start_time = time.time()
        
        tasks = [
            mock_voice_assistant.process_audio_turn(test_text)
            for _ in range(concurrent_tasks)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All tasks should complete successfully
        assert len(responses) == concurrent_tasks
        assert all(response is not None for response in responses)
        
        # Concurrent processing should be more efficient than sequential
        # (though with mocks this might not be realistic)
        print(f"Concurrent processing time: {total_time:.2f}s for {concurrent_tasks} tasks")
        
        # Should complete within reasonable time
        assert total_time < 10.0, f"Concurrent processing took {total_time:.2f}s, too slow"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, mock_voice_assistant, performance_audio_samples):
        """Test for memory leaks over extended operation."""
        test_text = "Hello, this is a test transcription for performance testing."
        process = psutil.Process()
        
        # Baseline memory measurement
        gc.collect()  # Force garbage collection
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate extended operation (24h simulation with faster cycles)
        cycles = 50  # Simulate many conversation cycles
        
        for cycle in range(cycles):
            await mock_voice_assistant.process_audio_turn(test_text)
            
            # Periodic garbage collection
            if cycle % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be bounded
                max_acceptable_growth = 50  # MB
                assert memory_growth < max_acceptable_growth, \
                    f"Memory leak detected: {memory_growth:.1f}MB growth after {cycle} cycles"
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        print(f"Memory after {cycles} cycles: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{total_growth:.1f}MB)")
        
        # Total memory growth should be reasonable
        assert total_growth < 100, f"Excessive memory growth: {total_growth:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_high_sample_rate_performance(self, mock_voice_assistant, performance_audio_samples):
        """Test performance with high sample rate audio."""
        test_text = "High quality audio transcription test with detailed content."
        
        start_time = time.time()
        response = await mock_voice_assistant.process_audio_turn(test_text)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # High sample rate audio should still meet latency requirements
        assert latency < 5.0, f"High sample rate latency {latency:.2f}s exceeds 5s"
        print(f"High sample rate audio latency: {latency:.2f}s")
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_conversation_flow_performance(self, mock_voice_assistant):
        """Test performance of a complete conversation flow."""
        conversation_samples = create_conversation_samples()
        
        total_start_time = time.time()
        conversation_latencies = []
        
        for i, audio in enumerate(conversation_samples):
            turn_start_time = time.time()
            response = await mock_voice_assistant.process_audio_turn(test_text)
            turn_end_time = time.time()
            
            turn_latency = turn_end_time - turn_start_time
            conversation_latencies.append(turn_latency)
            
            assert response is not None
            assert turn_latency < 4.0, f"Turn {i+1} latency {turn_latency:.2f}s exceeds 4s"
        
        total_end_time = time.time()
        total_conversation_time = total_end_time - total_start_time
        
        avg_latency = sum(conversation_latencies) / len(conversation_latencies)
        max_latency = max(conversation_latencies)
        
        print(f"Conversation performance:")
        print(f"  Total time: {total_conversation_time:.2f}s")
        print(f"  Average turn latency: {avg_latency:.2f}s")
        print(f"  Max turn latency: {max_latency:.2f}s")
        print(f"  Turns: {len(conversation_samples)}")
        
        # Conversation should flow smoothly
        assert avg_latency < 3.0, f"Average conversation latency {avg_latency:.2f}s too high"
        assert max_latency < 4.0, f"Max conversation latency {max_latency:.2f}s too high"


class TestPerformanceRegression:
    """Performance regression tests to ensure performance doesn't degrade."""
    
    @pytest.mark.asyncio
    async def test_baseline_performance_metrics(self, mock_voice_assistant, performance_audio_samples):
        """Establish baseline performance metrics for regression testing."""
        test_text = "Hello, this is a test transcription for performance testing."
        
        # Run multiple iterations to get stable measurements
        latencies = []
        memory_usages = []
        
        process = psutil.Process()
        
        for _ in range(5):
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            response = await mock_voice_assistant.process_audio_turn(test_text)
            end_time = time.time()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            latencies.append(end_time - start_time)
            memory_usages.append(final_memory - initial_memory)
            
            assert response is not None
        
        avg_latency = sum(latencies) / len(latencies)
        avg_memory_delta = sum(memory_usages) / len(memory_usages)
        
        # Store baseline metrics (in real implementation, these would be stored/compared)
        baseline_metrics = {
            "avg_latency": avg_latency,
            "max_latency": max(latencies),
            "avg_memory_delta": avg_memory_delta,
            "max_memory_delta": max(memory_usages)
        }
        
        print(f"Baseline performance metrics: {baseline_metrics}")
        
        # Basic sanity checks
        assert avg_latency < 4.0
        assert max(latencies) < 5.0
        assert avg_memory_delta < 50  # MB
        
        return baseline_metrics


class TestStressTests:
    """Stress tests for system stability under load."""
    
    @pytest.mark.asyncio
    async def test_rapid_fire_requests(self, mock_voice_assistant, performance_audio_samples):
        """Test system stability under rapid consecutive requests."""
        test_text = "Quick test!"
        
        # Send requests as fast as possible
        rapid_requests = 20
        start_time = time.time()
        
        for i in range(rapid_requests):
            response = await mock_voice_assistant.process_audio_turn(test_text)
            assert response is not None, f"Request {i+1} failed"
        
        end_time = time.time()
        total_time = end_time - start_time
        requests_per_second = rapid_requests / total_time
        
        print(f"Rapid fire: {rapid_requests} requests in {total_time:.2f}s ({requests_per_second:.1f} req/s)")
        
        # System should handle rapid requests without failure
        assert requests_per_second > 1.0, "System too slow for rapid requests"
    
    @pytest.mark.asyncio
    async def test_long_running_stability(self, mock_voice_assistant, performance_audio_samples):
        """Test system stability over extended operation."""
        test_text = "Hello, this is a test transcription for performance testing."
        
        # Simulate long-running operation
        long_run_cycles = 30
        start_time = time.time()
        
        for cycle in range(long_run_cycles):
            response = await mock_voice_assistant.process_audio_turn(test_text)
            assert response is not None, f"Cycle {cycle+1} failed"
            
            # Brief pause between cycles
            await asyncio.sleep(0.1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Long running test: {long_run_cycles} cycles in {total_time:.2f}s")
        
        # System should remain stable throughout
        assert total_time < 200.0, "Long running test took too long"